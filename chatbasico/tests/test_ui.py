"""UI tests: run the real Streamlit app headlessly with fake LLMs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("llama_index.llms.anthropic")

from streamlit.testing.v1 import AppTest

from chatbasico import ui
from chatbasico.config import AVAILABLE_MODELS
from chatbasico.prompts import INITIAL_MESSAGE

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeStreamingLLM:
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.seen: list | None = None

    def stream_chat(self, messages):
        self.seen = messages
        for delta in self.deltas:
            yield SimpleNamespace(delta=delta)


def _contents(at: AppTest) -> list[str]:
    return [message["content"] for message in at.session_state.history]


def test_sidebar_has_llamaindex_selector_and_custom_model_inputs_without_memory_controls() -> None:
    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert tuple(at.sidebar.selectbox[0].options) == AVAILABLE_MODELS
    assert [inp.label for inp in at.sidebar.text_input] == ["Outro modelo", "Chave da API"]
    assert len(at.sidebar.checkbox) == 0
    assert [s.label for s in at.sidebar.slider] == ["Temperatura"]
    assert len(at.chat_input) == 1


def test_custom_model_without_api_key_blocks_chat_until_key_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    at = AppTest.from_file(APP).run()
    at.sidebar.text_input[0].input("claude-haiku-5-5").run()

    assert len(at.chat_input) == 0
    assert "Chave da API" in at.info[0].value

    at.sidebar.text_input[1].input("sk-ant-123").run()
    assert len(at.chat_input) == 1


def test_streams_answer_and_separate_evaluator_writes_to_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeStreamingLLM(["Olá", ", ", "Ana!"])
    monkeypatch.setattr(ui, "create_llm", lambda settings, temperature: fake)
    monkeypatch.setattr(
        ui,
        "evaluate_and_write_memory",
        lambda settings, query, existing: (
            [*existing, "Usuária se chama Ana"] if "Ana" in query else list(existing)
        ),
    )

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("Meu nome é Ana.").run()

    assert not at.exception
    assert _contents(at) == [INITIAL_MESSAGE, "Meu nome é Ana.", "Olá, Ana!"]
    assert at.session_state.user_facts == ["Usuária se chama Ana"]
    assert all(m.content != INITIAL_MESSAGE for m in fake.seen)

    at.chat_input[0].set_value("Qual é o meu nome?").run()
    assert "Usuária se chama Ana" in fake.seen[0].content


def test_provider_error_is_shown_and_not_stored_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(settings, temperature):
        raise RuntimeError("rate limit")

    monkeypatch.setattr(ui, "create_llm", boom)

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("oi").run()

    assert not at.exception
    assert "rate limit" in at.error[0].value
    assert _contents(at) == [INITIAL_MESSAGE]
    assert at.session_state.user_facts == []


def test_clear_button_resets_history_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "create_llm", lambda s, t: FakeStreamingLLM(["resposta"]))
    monkeypatch.setattr(ui, "evaluate_and_write_memory", lambda s, q, existing: [*existing, "João"])

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("Meu nome é João.").run()
    assert len(at.session_state.history) == 3
    assert len(at.session_state.user_facts) == 1

    at.sidebar.button[0].click().run()

    assert _contents(at) == [INITIAL_MESSAGE]
    assert at.session_state.user_facts == []
