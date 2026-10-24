"""LLM orchestration for the basic chatbot.

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

Messages = Sequence[Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class _OllamaDirectClient:
    """Minimal LlamaIndex-like client talking to a local Ollama server over HTTP.

    Exposes ``chat`` and ``stream_chat`` with the same response shape used by
    :func:`generate_response` and :func:`stream_response` so local testing works
    without an Anthropic API key.
    """

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
    """Create an LLM client from ``settings``.

    The temperature is passed explicitly because LlamaIndex binds it at
    construction time, and the UI lets the user change it between turns.
    """
    if settings.uses_local_model:
        return _OllamaDirectClient(
            model=settings.model.removeprefix(LOCAL_MODEL_PREFIX),
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    if not settings.api_key:
        raise ValueError(
            "A chave ANTHROPIC_API_KEY não foi encontrada. "
            "Defina-a no arquivo .env antes de conversar."
        )

    return Anthropic(
        model=settings.model,
        max_tokens=settings.max_tokens,
        temperature=temperature,
        api_key=settings.api_key,
    )


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
