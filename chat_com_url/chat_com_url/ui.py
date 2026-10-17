"""Streamlit interface for the web-search chatbot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import streamlit as st

from .config import AVAILABLE_MODELS, Settings
from .llm import answer_question, create_chain
from .prompts import INITIAL_MESSAGE
from .search import create_search_tool, run_search
from .utils import SearchResult, format_sources

PAGE_TITLE = "Chatbot com URLs"
ASSISTANT_AVATAR = "🌐"
USER_AVATAR = "🧑"


def _initial_history() -> list[dict[str, object]]:
    return [{"role": "assistant", "content": INITIAL_MESSAGE, "sources": []}]


def _init_state() -> None:
    st.session_state.setdefault("history", _initial_history())


def _clear_history() -> None:
    st.session_state.history = _initial_history()


def _render_sources(sources: Sequence[SearchResult]) -> None:
    if not sources:
        return
    with st.expander(f"Fontes ({len(sources)})"):
        st.markdown(format_sources(sources))


def _render_history() -> None:
    for message in st.session_state.history:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            _render_sources(message.get("sources") or [])


def _api_keys(settings: Settings) -> tuple[str, str]:
    """Return the API keys, asking for them in the sidebar when missing."""
    api_key = settings.api_key
    tavily_api_key = settings.tavily_api_key

    if not api_key:
        api_key = st.text_input("Chave ANTHROPIC", type="password")
    if not tavily_api_key:
        tavily_api_key = st.text_input("Chave TAVILY", type="password")

    return api_key or "", tavily_api_key or ""


def _sidebar(settings: Settings) -> tuple[Settings, float]:
    with st.sidebar:
        st.header("Permissões")
        api_key, tavily_api_key = _api_keys(settings)

        st.header("Configurações")

        default_model = (
            settings.model if settings.model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        )
        model = st.selectbox(
            "Modelo", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(default_model)
        )

        temperature = st.slider(
            "Temperatura", 0.1, 1.0, settings.temperature, 0.1, help="Criatividade da resposta."
        )
        max_tokens = st.slider(
            "Máx. tokens", 100, 2000, settings.max_tokens, 50, help="Tamanho máximo da resposta."
        )
        max_results = st.slider(
            "Máx. resultados", 1, 5, settings.max_results, 1, help="Quantidade de fontes buscadas."
        )

        st.button("Limpar conversa", on_click=_clear_history, use_container_width=True)

    updated = replace(
        settings,
        api_key=api_key or None,
        tavily_api_key=tavily_api_key or None,
        model=model,
        max_tokens=max_tokens,
        max_results=max_results,
    )
    return updated, temperature


def _answer(
    settings: Settings, temperature: float, question: str
) -> tuple[str, list[SearchResult]]:
    """Run the search + answer pipeline for a single question."""
    search_tool = create_search_tool(settings.max_results, settings.tavily_api_key)
    with st.spinner("Pesquisando na web..."):
        results = run_search(search_tool, question)

    chain = create_chain(settings, temperature)
    with st.spinner("Escrevendo a resposta..."):
        answer = answer_question(chain, question, results)

    return answer, results


def run() -> None:
    """Render and drive the chatbot interface."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🌐", layout="centered")
    st.title("🌐 Chatbot com URLs")
    st.caption(
        "Pesquisa na web com Tavily e responde citando as fontes, usando LangChain + Claude."
    )

    settings, temperature = _sidebar(Settings.from_env())

    _init_state()
    _render_history()

    if not settings.is_configured:
        st.info("Informe as chaves `ANTHROPIC_API_KEY` e `TAVILY_API_KEY` para continuar.")
        return

    user_query = st.chat_input(INITIAL_MESSAGE)
    if not user_query:
        return

    st.session_state.history.append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            answer, results = _answer(settings, temperature, user_query)
        except Exception as exc:  # noqa: BLE001 - surface any provider error to the user
            st.session_state.history.pop()
            st.error(f"Não foi possível responder agora: {exc}")
            return

        st.markdown(answer)
        _render_sources(results)

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": results})
