"""UI tests: run the real Streamlit app headlessly with fake search + LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("langchain_anthropic")

from streamlit.testing.v1 import AppTest

from chat_com_url import ui
from chat_com_url.prompts import INITIAL_MESSAGE
from chat_com_url.utils import SearchResult

APP = str(Path(__file__).resolve().parent.parent / "app.py")

RESULTS = [
    SearchResult(
        title="Fonte A",
        url="https://a.com",
        content="O Python foi criado por Guido van Rossum em 1991.",
    ),
    SearchResult(
        title="Fonte B",
        url="https://b.com",
        content="A receita de feijoada leva feijão preto e carnes.",
    ),
]


class FakeStreamingChain:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.received: dict[str, str] | None = None

    def stream(self, variables: dict[str, str]):
        self.received = variables
        yield from self.chunks


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Provide both API keys and fake out every network call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily")

    state: dict[str, object] = {"chain": FakeStreamingChain(["Segundo ", "as fontes, ", "42."])}

    def fake_create_search_tool(max_results, api_key):
        state["max_results"], state["tavily_key"] = max_results, api_key
        return object()

    def fake_create_chain(settings, temperature):
        state["model"], state["temperature"] = settings.model, temperature
        return state["chain"]

    monkeypatch.setattr(ui, "create_search_tool", fake_create_search_tool)
    monkeypatch.setattr(ui, "run_search", lambda tool, query: list(RESULTS))
    monkeypatch.setattr(ui, "create_chain", fake_create_chain)
    return state


def test_without_keys_asks_for_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    at = AppTest.from_file(APP).run()

    assert not at.exception
    labels = [field.label for field in at.sidebar.text_input]
    assert labels == ["Chave ANTHROPIC", "Chave TAVILY"]
    assert "TAVILY_API_KEY" in at.info[0].value
    assert len(at.chat_input) == 0


def test_keys_typed_in_sidebar_unlock_chat(
    monkeypatch: pytest.MonkeyPatch, configured: dict[str, object]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    at = AppTest.from_file(APP).run()
    at.sidebar.text_input[0].input("sk-ant-digitada")
    at.sidebar.text_input[1].input("tvly-digitada").run()

    assert len(at.chat_input) == 1
    at.chat_input[0].set_value("pergunta").run()
    assert configured["tavily_key"] == "tvly-digitada"


def test_stores_pages_in_session_memory_runs_rag_and_extracts_facts(
    configured: dict[str, object],
) -> None:
    at = AppTest.from_file(APP).run()
    # Set RAG top_k to 1 so only the most relevant page chunk is sent to the LLM
    at.sidebar.slider[3].set_value(1)
    at.run()

    at.chat_input[0].set_value("Prefiro respostas diretas. Quem criou o Python?").run()

    assert not at.exception
    # Both pages were written to active session memory:
    assert len(at.session_state.page_memory) == 2
    # Only the user preference was saved to user_facts:
    assert at.session_state.user_facts == ["Prefiro respostas diretas."]

    received = configured["chain"].received
    assert "Prefiro respostas diretas." in received["memory_facts"]
    # RAG with top_k=1 delivered only the Python chunk, not the feijoada chunk:
    assert "Guido van Rossum" in received["search_results"]
    assert "feijoada" not in received["search_results"]


def test_sidebar_controls_are_applied(configured: dict[str, object]) -> None:
    at = AppTest.from_file(APP).run()
    at.sidebar.selectbox[0].select("claude-opus-5-5")
    at.sidebar.slider[0].set_value(0.5)  # temperatura
    at.sidebar.slider[2].set_value(5)  # máx. resultados
    at.run()
    at.chat_input[0].set_value("q").run()

    assert configured["model"] == "claude-opus-5-5"
    assert configured["temperature"] == 0.5
    assert configured["max_results"] == 5


def test_search_error_is_shown_and_excluded_from_memory_by_default(
    monkeypatch: pytest.MonkeyPatch, configured: dict[str, object]
) -> None:
    def boom(tool, query):
        raise RuntimeError("tavily fora do ar")

    monkeypatch.setattr(ui, "run_search", boom)

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("q").run()

    assert not at.exception
    assert "tavily fora do ar" in at.error[0].value
    assert [m["content"] for m in at.session_state.history] == [INITIAL_MESSAGE]
    assert at.session_state.user_facts == []
