"""LLM orchestration for the web-search chatbot."""

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
from .prompts import HUMAN_TEMPLATE, SYSTEM_PROMPT
from .utils import SearchResult, build_prompt_variables


class _OllamaHTTPChatModel(BaseChatModel):
    """Lightweight LangChain chat model talking to a local Ollama server over HTTP.

    Used when ``settings.model`` starts with ``ollama/`` so the web-search
    chatbot can run end-to-end without an Anthropic API key.
    """

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


def create_llm(settings: Settings, temperature: float) -> BaseChatModel:
    """Create the chat model from ``settings`` (Anthropic Claude or local Ollama)."""
    if settings.uses_local_model:
        local_name = settings.model.removeprefix(LOCAL_MODEL_PREFIX)
        return _OllamaHTTPChatModel(
            model=local_name,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    if not settings.api_key:
        raise ValueError("A chave ANTHROPIC_API_KEY não foi encontrada no ambiente.")

    return ChatAnthropic(
        model=settings.model,
        temperature=temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.api_key,
    )


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
    """Yield the model's answer incrementally, as text chunks.

    The chain ends with ``StrOutputParser``, so every streamed chunk is already
    a plain string. Empty chunks are skipped to avoid needless UI updates.
    """
    for chunk in chain.stream(build_prompt_variables(question, results, memory_facts)):
        text = str(chunk)
        if text:
            yield text
