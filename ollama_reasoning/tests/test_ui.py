"""UI tests: run the real Streamlit app headlessly with a fake Ollama model."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("langchain_ollama")

from langchain_core.messages import AIMessageChunk
from streamlit.testing.v1 import AppTest

from ollama_reasoning import ui
from ollama_reasoning.prompts import INITIAL_MESSAGE

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeStreamingChain:
    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self.chunks = chunks
        self.received: dict[str, object] | None = None

    def stream(self, variables: dict[str, object]):
        self.received = variables
        yield from self.chunks


def _install_fake(
    monkeypatch: pytest.MonkeyPatch, chunks: list[AIMessageChunk], thinking: bool = True
) -> dict[str, object]:
    state: dict[str, object] = {"chain": FakeStreamingChain(chunks)}

    def fake_create_llm(settings, temperature, reasoning=None):
        state.update(model=settings.model, temperature=temperature, reasoning=reasoning)
        return object()

    monkeypatch.setattr(ui, "supports_thinking", lambda settings: thinking)
    monkeypatch.setattr(ui, "create_llm", fake_create_llm)
    monkeypatch.setattr(ui, "build_chain", lambda llm: state["chain"])
    return state


def _reasoning_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content="", additional_kwargs={"reasoning_content": text})


def test_initial_render() -> None:
    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert at.title[0].value.startswith("🧠")
    assert len(at.chat_input) == 1
    assert at.sidebar.text_input[0].value == "deepseek-r1:1.5b"


def test_streams_reasoning_and_strips_reasoning_from_subsequent_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake(
        monkeypatch,
        [
            _reasoning_chunk("Somar "),
            _reasoning_chunk("2 e 3."),
            AIMessageChunk(content="A resposta "),
            AIMessageChunk(content="é 5."),
        ],
    )

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("2+3?").run()

    assert not at.exception
    last = at.session_state.history[-1]
    assert (last["reasoning"], last["content"]) == ("Somar 2 e 3.", "A resposta é 5.")
    assert state["reasoning"] is True
    assert state["chain"].received == {"input": "2+3?", "history": []}

    # Second turn: history passed to chain has only question + clean answer (0% reasoning)
    at.chat_input[0].set_value("E vezes 2?").run()
    sent_history = state["chain"].received["history"]
    assert len(sent_history) == 2
    assert sent_history[0].content == "2+3?"
    assert sent_history[1].content == "A resposta é 5."
    assert "Somar" not in sent_history[1].content


def test_inline_think_tags_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(
        monkeypatch,
        [AIMessageChunk(content=c) for c in ["<thi", "nk>pensei</think>", "\n\nResposta."]],
        thinking=False,
    )

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("q").run()

    last = at.session_state.history[-1]
    assert (last["reasoning"], last["content"]) == ("pensei", "Resposta.")


def test_model_without_reasoning_shows_no_status(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake(monkeypatch, [AIMessageChunk(content="Oi!")], thinking=False)

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("olá").run()

    assert state["reasoning"] is None
    assert len(at.status) == 0
    assert at.session_state.history[-1] == {"role": "assistant", "content": "Oi!", "reasoning": ""}


def test_history_rerender_keeps_reasoning_expander(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, [_reasoning_chunk("r"), AIMessageChunk(content="a")])

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("q").run()
    at.run()  # plain rerun: messages now come from the history

    assert at.expander[0].label == ui.REASONING_LABEL


def test_sidebar_model_and_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake(monkeypatch, [AIMessageChunk(content="ok")])

    at = AppTest.from_file(APP).run()
    at.sidebar.text_input[0].input("deepseek-r1:7b")
    at.sidebar.slider[0].set_value(0.2)
    at.run()
    at.chat_input[0].set_value("q").run()

    assert (state["model"], state["temperature"]) == ("deepseek-r1:7b", 0.2)


def test_connection_error_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, [])

    def boom(llm):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(ui, "build_chain", boom)

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("q").run()

    assert not at.exception
    assert "O Ollama está rodando" in at.error[0].value
    assert [m["content"] for m in at.session_state.history] == [INITIAL_MESSAGE]
