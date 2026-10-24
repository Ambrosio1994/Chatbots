"""UI tests: run the real Streamlit app headlessly with fake search + LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("langchain_anthropic")

from streamlit.testing.v1 import AppTest

from chat_com_url import ui
from chat_com_url.config import AVAILABLE_MODELS
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
    """Fake out network calls while testing the real Streamlit UI."""
    state: dict[str, object] = {"chain": FakeStreamingChain(["Segundo ", "as fontes, ", "42."])}

    def fake_create_search_tool(max_results, base_url):
        state["max_results"], state["searxng_url"] = max_results, base_url
        return object()

    def fake_create_chain(settings, temperature):
        state["model"], state["temperature"] = settings.active_model, temperature
        return state["chain"]

    monkeypatch.setattr(ui, "create_search_tool", fake_create_search_tool)
    monkeypatch.setattr(ui, "run_search", lambda tool, query: list(RESULTS))
    monkeypatch.setattr(ui, "create_chain", fake_create_chain)
    monkeypatch.setattr(
        ui,
        "evaluate_and_write_memory",
        lambda settings, query, existing: (
            [*existing, "Prefiro respostas diretas."] if "Prefiro" in query else list(existing)
        ),
    )
    return state


def test_sidebar_has_llamaindex_selector_and_custom_inputs_without_memory_controls() -> None:
    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert tuple(at.sidebar.selectbox[0].options) == AVAILABLE_MODELS
    assert [inp.label for inp in at.sidebar.text_input] == ["Outro modelo", "Chave da API"]
    assert len(at.sidebar.checkbox) == 0
    assert [s.label for s in at.sidebar.slider] == [
        "Temperatura",
        "Máx. tokens",
        "Máx. resultados",
    ]
    assert len(at.chat_input) == 1


def test_custom_model_requires_api_key_until_typed(
    monkeypatch: pytest.MonkeyPatch, configured: dict[str, object]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    at = AppTest.from_file(APP).run()
    at.sidebar.text_input[0].input("claude-opus-5-5").run()
    assert len(at.chat_input) == 0

    at.sidebar.text_input[1].input("sk-ant-digitada").run()
    assert len(at.chat_input) == 1
    at.chat_input[0].set_value("pergunta").run()
    assert configured["model"] == "claude-opus-5-5"


def test_stores_pages_in_session_memory_runs_rag_and_separate_model_writes_facts(
    configured: dict[str, object],
) -> None:
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("Prefiro respostas diretas. Quem criou o Python?").run()

    assert not at.exception
    assert len(at.session_state.page_memory) == 2
    assert at.session_state.user_facts == ["Prefiro respostas diretas."]

    received = configured["chain"].received
    assert "Prefiro respostas diretas." in received["memory_facts"]
    assert "Guido van Rossum" in received["search_results"]


def test_sidebar_controls_are_applied(configured: dict[str, object]) -> None:
    at = AppTest.from_file(APP).run()
    at.sidebar.selectbox[0].select("qwen2.5")
    at.sidebar.slider[0].set_value(0.5)
    at.sidebar.slider[2].set_value(5)
    at.run()
    at.chat_input[0].set_value("q").run()

    assert configured["model"] == "qwen2.5"
    assert configured["temperature"] == 0.5
    assert configured["max_results"] == 5


def test_search_error_is_shown_and_excluded_from_memory(
    monkeypatch: pytest.MonkeyPatch, configured: dict[str, object]
) -> None:
    def boom(tool, query):
        raise RuntimeError("searxng fora do ar")

    monkeypatch.setattr(ui, "run_search", boom)

    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("q").run()

    assert not at.exception
    assert "searxng fora do ar" in at.error[0].value
    assert [m["content"] for m in at.session_state.history] == [INITIAL_MESSAGE]
