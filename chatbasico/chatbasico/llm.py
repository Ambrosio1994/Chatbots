"""LLM orchestration for the basic chatbot and its dedicated memory-evaluator model.

This module is intentionally free of Streamlit imports so it can be reused and
tested on its own.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic

from .config import LOCAL_MODEL_PREFIX, Settings
from .prompts import MEMORY_EVALUATOR_PROMPT
from .utils import extract_user_facts, merge_memory_facts, parse_memory_evaluation

Messages = Sequence[Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class _OllamaDirectClient:
    """Minimal LlamaIndex-like client talking to a local Ollama server over HTTP."""

    model: str
    base_url: str
    temperature: float

    @staticmethod
    def _serialize(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": str(msg.content or ""),
            }
            for msg in messages
        ]

    def chat(self, messages: list[ChatMessage]) -> SimpleNamespace:
        payload = {
            "model": self.model,
            "messages": self._serialize(messages),
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.base_url.rstrip('/')}/api/chat", json=payload)
            resp.raise_for_status()
            content = str(resp.json().get("message", {}).get("content") or "")
        return SimpleNamespace(message=SimpleNamespace(content=content))

    def stream_chat(self, messages: list[ChatMessage]) -> Iterator[SimpleNamespace]:
        payload = {
            "model": self.model,
            "messages": self._serialize(messages),
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
                delta = str(data.get("message", {}).get("content") or "")
                if delta:
                    yield SimpleNamespace(delta=delta)


def create_llm(settings: Settings, temperature: float) -> Anthropic | _OllamaDirectClient:
    """Create an LLM client from ``settings``."""
    target = settings.active_model
    if settings.uses_local_model and not settings.api_key:
        return _OllamaDirectClient(
            model=target.removeprefix(LOCAL_MODEL_PREFIX),
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
    if settings.uses_local_model and not target.startswith("claude-"):
        return _OllamaDirectClient(
            model=target.removeprefix(LOCAL_MODEL_PREFIX),
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    if not settings.api_key:
        raise ValueError(
            "A chave da API não foi encontrada. "
            "Informe a chave na barra lateral ou escolha um modelo do seletor."
        )

    return Anthropic(
        model=target,
        max_tokens=settings.max_tokens,
        temperature=temperature,
        api_key=settings.api_key,
    )


def create_memory_llm(settings: Settings) -> _OllamaDirectClient | Anthropic:
    """Create the separate model dedicated to evaluating what should be written to memory."""
    return _OllamaDirectClient(
        model=settings.memory_model.removeprefix(LOCAL_MODEL_PREFIX),
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )


def evaluate_and_write_memory(
    settings: Settings,
    user_query: str,
    existing_facts: Sequence[str],
    memory_llm: Anthropic | _OllamaDirectClient | None = None,
) -> list[str]:
    """Use a separate model to evaluate whether ``user_query`` should be written to memory."""
    candidates = extract_user_facts(user_query)
    if not candidates:
        return list(existing_facts)

    try:
        evaluator = memory_llm if memory_llm is not None else create_memory_llm(settings)
        eval_messages = [
            {"role": "system", "content": MEMORY_EVALUATOR_PROMPT},
            {"role": "user", "content": f"Mensagem do usuário:\n{user_query}"},
        ]
        raw_text = generate_response(evaluator, eval_messages)
        extracted = parse_memory_evaluation(raw_text, user_prompt=user_query)
    except Exception:  # noqa: BLE001 - fallback to deterministic extraction if offline
        extracted = candidates

    return merge_memory_facts(existing_facts, extracted)


def to_chat_messages(messages: Messages) -> list[ChatMessage]:
    """Convert plain dictionaries into LlamaIndex ``ChatMessage`` objects."""
    return [ChatMessage(role=message["role"], content=message["content"]) for message in messages]


def generate_response(llm: Anthropic | _OllamaDirectClient, messages: Messages) -> str:
    """Return the full model answer for ``messages``."""
    return str(llm.chat(to_chat_messages(messages)).message.content)


def stream_response(llm: Anthropic | _OllamaDirectClient, messages: Messages) -> Iterator[str]:
    """Yield the model answer token by token."""
    for chunk in llm.stream_chat(to_chat_messages(messages)):
        yield chunk.delta
