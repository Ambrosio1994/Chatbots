"""Tests for streaming: the incremental splitter and ``stream_answer``."""

from __future__ import annotations

import pytest

from ollama_reasoning.utils import ANSWER, REASONING, ReasoningStreamSplitter

# Reasoning delimiters are assembled from pieces to stay robust in editors.
OPEN = "<" + "think" + ">"
CLOSE = "<" + "/think" + ">"


def _feed_all(chunks: list[str]) -> tuple[ReasoningStreamSplitter, list[tuple[str, str]]]:
    splitter = ReasoningStreamSplitter()
    events: list[tuple[str, str]] = []
    for chunk in chunks:
        events.extend(splitter.feed(chunk))
    events.extend(splitter.flush())
    return splitter, events


def test_splitter_plain_answer_streams_immediately() -> None:
    splitter = ReasoningStreamSplitter()

    assert splitter.feed("Olá") == [(ANSWER, "Olá")]
    assert splitter.feed(", mundo") == [(ANSWER, ", mundo")]
    assert splitter.answer == "Olá, mundo"
    assert splitter.reasoning == ""


def test_splitter_inline_tags_in_single_chunk() -> None:
    splitter, _ = _feed_all([f"{OPEN}pensando{CLOSE}resposta"])

    assert splitter.reasoning == "pensando"
    assert splitter.answer == "resposta"


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_splitter_handles_tags_split_across_chunks(size: int) -> None:
    text = f"{OPEN}passo 1, passo 2{CLOSE}\n\nA resposta é 42."
    chunks = [text[i : i + size] for i in range(0, len(text), size)]

    splitter, events = _feed_all(chunks)

    assert splitter.reasoning == "passo 1, passo 2"
    assert splitter.answer == "A resposta é 42."
    # No tag fragment may ever leak into the emitted text.
    assert all("<" not in text and ">" not in text for _, text in events)


def test_splitter_order_of_events() -> None:
    _, events = _feed_all([OPEN, "a", CLOSE, "b"])

    assert events == [(REASONING, "a"), (ANSWER, "b")]


def test_splitter_supports_thinking_tag_case_insensitive() -> None:
    splitter, _ = _feed_all(["<THINKING>", "x", "</Thinking>", "y"])

    assert (splitter.reasoning, splitter.answer) == ("x", "y")


def test_splitter_dedicated_reasoning_field() -> None:
    splitter = ReasoningStreamSplitter()

    assert splitter.feed("", reasoning_content="raciocínio") == [(REASONING, "raciocínio")]
    assert splitter.feed("resposta") == [(ANSWER, "resposta")]
    assert (splitter.reasoning, splitter.answer) == ("raciocínio", "resposta")


def test_splitter_keeps_regular_angle_brackets() -> None:
    splitter, _ = _feed_all(["se a ", "<", " b então ", "<b>negrito</b>"])

    assert splitter.answer == "se a < b então <b>negrito</b>"


def test_splitter_flush_releases_held_partial_tag() -> None:
    splitter = ReasoningStreamSplitter()

    assert splitter.feed("fim <th") == [(ANSWER, "fim ")]
    assert splitter.flush() == [(ANSWER, "<th")]
    assert splitter.answer == "fim <th"


def test_splitter_unclosed_reasoning_stays_reasoning() -> None:
    splitter, _ = _feed_all([OPEN, "ainda pensando"])

    assert (splitter.reasoning, splitter.answer) == ("ainda pensando", "")


# --- stream_answer ---------------------------------------------------------

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessageChunk

from ollama_reasoning import llm as llm_module
from ollama_reasoning.config import Settings
from ollama_reasoning.llm import stream_answer, supports_thinking


class FakeStreamingChain:
    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self.chunks = chunks
        self.received: dict[str, str] | None = None

    def stream(self, variables: dict[str, str]):
        self.received = variables
        yield from self.chunks


def test_stream_answer_with_dedicated_reasoning_field() -> None:
    chain = FakeStreamingChain(
        [
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "pen"}),
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "sando"}),
            AIMessageChunk(content="4"),
            AIMessageChunk(content="2"),
        ]
    )

    events = list(stream_answer(chain, "pergunta"))

    assert chain.received == {"input": "pergunta"}
    assert events == [(REASONING, "pen"), (REASONING, "sando"), (ANSWER, "4"), (ANSWER, "2")]


def test_stream_answer_with_inline_tags() -> None:
    chain = FakeStreamingChain(
        [AIMessageChunk(content=part) for part in ["<th", "ink>raciocínio</th", "ink>resposta"]]
    )

    events = list(stream_answer(chain, "q"))

    assert "".join(t for k, t in events if k == REASONING) == "raciocínio"
    assert "".join(t for k, t in events if k == ANSWER) == "resposta"


class _FakeShow:
    def __init__(self, capabilities: list[str] | None) -> None:
        self.capabilities = capabilities


def _fake_client(capabilities: list[str] | None = None, error: Exception | None = None):
    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def show(self, model: str) -> _FakeShow:
            if error:
                raise error
            return _FakeShow(capabilities)

    return FakeClient


def test_supports_thinking_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "Client", _fake_client(["completion", "thinking"]))

    assert supports_thinking(Settings()) is True


def test_supports_thinking_false_without_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "Client", _fake_client(["completion"]))

    assert supports_thinking(Settings()) is False


def test_supports_thinking_false_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "Client", _fake_client(error=ConnectionError("offline")))

    assert supports_thinking(Settings()) is False
