"""LLM orchestration for the web-search chatbot and its dedicated memory-evaluator model."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .config import LOCAL_MODEL_PREFIX, Settings
from .prompts import HUMAN_TEMPLATE, MEMORY_EVALUATOR_PROMPT, SYSTEM_PROMPT
from .utils import (
    SearchResult,
    build_prompt_variables,
    extract_user_facts,
    merge_memory_facts,
    parse_memory_evaluation,
)


class _OllamaHTTPChatModel(BaseChatModel):
    """Lightweight LangChain chat model talking to a local Ollama server over HTTP."""

    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "ollama-http"

    @staticmethod
    def _format_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
        formatted: list[dict[str, str]] = []
        for msg in messages:
            role = (
                "system"
                if isinstance(msg, SystemMessage)
                else ("assistant" if isinstance(msg, AIMessage) else "user")
            )
            formatted.append({"role": role, "content": str(msg.content)})
        return formatted

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.base_url.rstrip('/')}/api/chat", json=payload)
            resp.raise_for_status()
            content = str(resp.json().get("message", {}).get("content") or "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        with (
            httpx.Client(timeout=120.0) as client,
            client.stream("POST", f"{self.base_url.rstrip('/')}/api/chat", json=payload) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                piece = str(data.get("message", {}).get("content") or "")
                if piece:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def create_prompt() -> ChatPromptTemplate:
    """Create the chat prompt combining the system rules and the template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_TEMPLATE),
        ]
    ).partial(memory_facts="Nenhuma preferência ou fato registrado ainda.")


def create_memory_prompt() -> ChatPromptTemplate:
    """Create the prompt for the separate memory-evaluator model."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", MEMORY_EVALUATOR_PROMPT),
            ("human", "Mensagem do usuário:\n{user_message}"),
        ]
    )


def create_llm(settings: Settings, temperature: float) -> BaseChatModel:
    """Create the chat model from ``settings``."""
    target = settings.active_model
    if settings.uses_local_model and not settings.api_key:
        return _OllamaHTTPChatModel(
            model=target.removeprefix(LOCAL_MODEL_PREFIX),
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
    if settings.uses_local_model and not target.startswith("claude-"):
        return _OllamaHTTPChatModel(
            model=target.removeprefix(LOCAL_MODEL_PREFIX),
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    if not settings.api_key:
        raise ValueError("A chave ANTHROPIC_API_KEY não foi encontrada no ambiente.")

    return ChatAnthropic(
        model=target,
        temperature=temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.api_key,
    )


def create_memory_llm(settings: Settings) -> BaseChatModel:
    """Create the dedicated, separate model that evaluates and writes to session memory."""
    return _OllamaHTTPChatModel(
        model=settings.memory_model.removeprefix(LOCAL_MODEL_PREFIX),
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )


def evaluate_and_write_memory(
    settings: Settings,
    user_query: str,
    existing_facts: Sequence[str],
    memory_llm: Runnable | None = None,
) -> list[str]:
    """Use a separate model to evaluate whether ``user_query`` should be written to memory."""
    candidates = extract_user_facts(user_query)
    if not candidates:
        return list(existing_facts)

    try:
        evaluator = memory_llm if memory_llm is not None else create_memory_llm(settings)
        chain = create_memory_prompt() | evaluator | StrOutputParser()
        raw_text = str(chain.invoke({"user_message": user_query}))
        extracted = parse_memory_evaluation(raw_text, user_prompt=user_query)
    except Exception:  # noqa: BLE001 - fallback to deterministic extraction if offline
        extracted = candidates

    return merge_memory_facts(existing_facts, extracted)


def create_chain(settings: Settings, temperature: float) -> Runnable:
    """Compose prompt → model → text parser into a single runnable."""
    return create_prompt() | create_llm(settings, temperature) | StrOutputParser()


def answer_question(
    chain: Runnable,
    question: str,
    results: Sequence[SearchResult],
    memory_facts: Sequence[str] = (),
) -> str:
    """Run the chain and return the model's answer as plain text."""
    return str(chain.invoke(build_prompt_variables(question, results, memory_facts)))


def stream_answer(
    chain: Runnable,
    question: str,
    results: Sequence[SearchResult],
    memory_facts: Sequence[str] = (),
) -> Iterator[str]:
    """Yield the model's answer incrementally, as text chunks."""
    for chunk in chain.stream(build_prompt_variables(question, results, memory_facts)):
        text = str(chunk)
        if text:
            yield text
