"""UI tests: run the real Streamlit app headlessly with a fake LLM."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("llama_index.llms.anthropic")

from streamlit.testing.v1 import AppTest

from chatbasico import ui
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


def test_without_api_key_shows_hint_and_no_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert at.title[0].value.startswith("🦙")
    assert "ANTHROPIC_API_KEY" in at.info[0].value
    assert len(at.chat_input) == 0
    assert _contents(at) == [INITIAL_MESSAGE]


def test_streams_answer_excludes_initial_message_and_stores_user_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    fake = FakeStreamingLLM(["Olá", ", ", "Ana!"])

    monkeypatch.setattr(ui, "create_llm", lambda settings, temperature: fake)

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("Meu nome é Ana e prefiro respostas curtas.").run()

    assert not at.exception
    assert _contents(at) == [
        INITIAL_MESSAGE,
        "Meu nome é Ana e prefiro respostas curtas.",
        "Olá, Ana!",
    ]
    assert at.session_state.user_facts == ["Meu nome é Ana e prefiro respostas curtas."]
    # First turn: INITIAL_MESSAGE must NEVER be sent to Claude
    assert all(m.content != INITIAL_MESSAGE for m in fake.seen)

    # Second turn: RAG includes relevant turns + user facts in system prompt
    at.chat_input[0].set_value("Qual é o meu nome?").run()
    assert "Meu nome é Ana" in fake.seen[0].content
    assert [m.content for m in fake.seen[1:]] == [
        "Meu nome é Ana e prefiro respostas curtas.",
        "Olá, Ana!",
        "Qual é o meu nome?",
    ]


def test_sidebar_controls_are_passed_to_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    captured: dict[str, object] = {}

    def fake_create_llm(settings, temperature):
        captured["model"], captured["temperature"] = settings.model, temperature
        return FakeStreamingLLM(["ok"])

    monkeypatch.setattr(ui, "create_llm", fake_create_llm)

    at = AppTest.from_file(APP).run()
    at.sidebar.selectbox[0].select("claude-sonnet-5-5")
    at.sidebar.slider[0].set_value(0.7)
    at.run()
    at.chat_input[0].set_value("oi").run()

    assert captured == {"model": "claude-sonnet-5-5", "temperature": 0.7}


def test_provider_error_is_shown_and_not_stored_in_memory_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr(ui, "create_llm", lambda s, t: FakeStreamingLLM(["resposta"]))

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("Meu nome é João.").run()
    assert len(at.session_state.history) == 3
    assert len(at.session_state.user_facts) == 1

    at.sidebar.button[0].click().run()

    assert _contents(at) == [INITIAL_MESSAGE]
    assert at.session_state.user_facts == []
