"""LLM orchestration for the basic chatbot.

This module is intentionally free of Streamlit imports so it can be reused and
tested on its own.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic

from .config import Settings

Messages = Sequence[Mapping[str, str]]


def create_llm(settings: Settings, temperature: float) -> Anthropic:
    """Create an Anthropic client from ``settings``.

    The temperature is passed explicitly because LlamaIndex binds it at
    construction time, and the UI lets the user change it between turns.
    """
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


def generate_response(llm: Anthropic, messages: Messages) -> str:
    """Return the full model answer for ``messages``."""
    return str(llm.chat(to_chat_messages(messages)).message.content)


def stream_response(llm: Anthropic, messages: Messages) -> Iterator[str]:
    """Yield the model answer token by token."""
    for chunk in llm.stream_chat(to_chat_messages(messages)):
        yield chunk.delta
