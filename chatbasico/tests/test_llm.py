"""Tests for the LLM layer, using a fake client (no network calls)."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")
pytest.importorskip("llama_index.llms.anthropic")

from llama_index.core.llms import ChatMessage

from chatbasico.llm import generate_response, to_chat_messages


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class FakeLLM:
    """Minimal stand-in that mimics the part of the Anthropic API we use."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: list[ChatMessage] | None = None

    def chat(self, messages: list[ChatMessage]) -> _FakeResponse:
        self.seen = messages
        return _FakeResponse(self.reply)


def test_to_chat_messages_maps_roles() -> None:
    messages = to_chat_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )

    assert all(isinstance(message, ChatMessage) for message in messages)
    assert [message.role.value for message in messages] == ["system", "user"]


def test_generate_response_returns_message_content() -> None:
    llm = FakeLLM("olá!")

    assert generate_response(llm, [{"role": "user", "content": "oi"}]) == "olá!"
    assert llm.seen is not None
    assert llm.seen[-1].content == "oi"
