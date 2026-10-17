"""Tests for the LLM layer, using fakes (no network calls)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from ollama_reasoning.llm import create_prompt, generate_answer

# Reasoning delimiters are assembled from pieces to stay robust in editors.
OPEN = "<" + "think" + ">"
CLOSE = "<" + "/think" + ">"


class FakeMessage:
    def __init__(self, content: str, reasoning_content: str | None = None) -> None:
        self.content = content
        self.additional_kwargs: dict[str, str] = {}
        if reasoning_content is not None:
            self.additional_kwargs["reasoning_content"] = reasoning_content


class FakeChain:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message
        self.received: dict[str, str] | None = None

    def invoke(self, variables: dict[str, str]) -> FakeMessage:
        self.received = variables
        return self.message


def test_create_prompt_formats_input_variable() -> None:
    messages = create_prompt().format_messages(input="qual a capital?")

    assert [message.type for message in messages] == ["system", "human"]
    assert messages[1].content == "qual a capital?"


def test_generate_answer_reads_reasoning_from_content() -> None:
    chain = FakeChain(FakeMessage(f"{OPEN}passo a passo{CLOSE}42"))

    reasoning, answer = generate_answer(chain, "pergunta")

    assert chain.received == {"input": "pergunta"}
    assert reasoning == "passo a passo"
    assert answer == "42"


def test_generate_answer_reads_dedicated_reasoning_field() -> None:
    chain = FakeChain(FakeMessage("42", reasoning_content="cálculo interno"))

    reasoning, answer = generate_answer(chain, "pergunta")

    assert reasoning == "cálculo interno"
    assert answer == "42"
