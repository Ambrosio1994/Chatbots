"""Tests for the LLM layer, using fakes (no network calls)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_anthropic")

from chat_com_url.llm import answer_question, create_prompt, stream_answer
from chat_com_url.utils import SearchResult


class FakeChain:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received: dict[str, str] | None = None

    def invoke(self, variables: dict[str, str]) -> str:
        self.received = variables
        return self.reply


def test_create_prompt_formats_all_variables() -> None:
    messages = create_prompt().format_messages(
        question="q?", search_results="resultado", urls="- https://a.com"
    )

    assert [message.type for message in messages] == ["system", "human"]
    assert "resultado" in messages[1].content


def test_answer_question_passes_prompt_variables() -> None:
    chain = FakeChain("resposta final")
    results = [SearchResult(title="A", url="https://a.com", content="conteudo")]

    answer = answer_question(chain, "pergunta", results)

    assert answer == "resposta final"
    assert chain.received is not None
    assert chain.received["question"] == "pergunta"
    assert "conteudo" in chain.received["search_results"]


class FakeStreamingChain:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.received: dict[str, str] | None = None

    def stream(self, variables: dict[str, str]):
        self.received = variables
        yield from self.chunks


def test_stream_answer_yields_chunks_and_skips_empty() -> None:
    chain = FakeStreamingChain(["Resp", "", "osta ", "final"])
    results = [SearchResult(title="A", url="https://a.com", content="conteudo")]

    chunks = list(stream_answer(chain, "pergunta", results))

    assert chunks == ["Resp", "osta ", "final"]
    assert chain.received is not None
    assert chain.received["question"] == "pergunta"
    assert "https://a.com" in chain.received["urls"]


def test_stream_answer_is_lazy() -> None:
    chain = FakeStreamingChain(["x"])

    generator = stream_answer(chain, "q", [])

    assert chain.received is None  # nothing runs until the UI starts consuming
    assert next(generator) == "x"
