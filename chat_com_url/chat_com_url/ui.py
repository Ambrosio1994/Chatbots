"""Streamlit interface for the web-search chatbot (SearXNG + LlamaIndex/Custom Models)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import streamlit as st

from .config import AVAILABLE_MODELS, Settings
from .llm import create_chain, evaluate_and_write_memory, stream_answer
from .prompts import INITIAL_MESSAGE
from .search import create_search_tool, run_search
from .utils import (
    SearchResult,
    format_sources,
    ingest_pages_into_memory,
    retrieve_relevant_chunks,
    strip_reasoning,
)

PAGE_TITLE = "Chatbot com URLs"
ASSISTANT_AVATAR = "🌐"
USER_AVATAR = "🧑"


def _initial_history() -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": INITIAL_MESSAGE,
            "sources": [],
            "include_in_memory": False,
        }
    ]


def _init_state() -> None:
    st.session_state.setdefault("history", _initial_history())
    st.session_state.setdefault("page_memory", [])
    st.session_state.setdefault("user_facts", [])


def _clear_history() -> None:
    st.session_state.history = _initial_history()
    st.session_state.page_memory = []
    st.session_state.user_facts = []


def _render_sources(sources: Sequence[SearchResult]) -> None:
    if not sources:
        return
    with st.expander(f"Fontes ({len(sources)})"):
        st.markdown(format_sources(sources))


def _render_history() -> None:
    for message in st.session_state.history:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(str(message["role"]), avatar=avatar):
            st.markdown(str(message["content"]))
            _render_sources(message.get("sources") or [])


def _sidebar(settings: Settings) -> tuple[Settings, float]:
    with st.sidebar:
        st.header("Configurações")

        default_model = (
            settings.model if settings.model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        )
        selected_model = st.selectbox(
            "Modelo", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(default_model)
        )

        custom_model = st.text_input(
            "Outro modelo",
            value="",
            placeholder="Ex.: claude-haiku-5-5",
            help="Preencha apenas se quiser usar um modelo externo fora do seletor.",
        )
        custom_api_key = st.text_input(
            "Chave da API",
            value="",
            type="password",
            help="Chave da API para o outro modelo informado acima.",
        )

        temperature = st.slider(
            "Temperatura",
            0.1,
            1.0,
            max(0.1, min(1.0, float(settings.temperature))),
            0.1,
            help="Criatividade da resposta.",
        )
        max_tokens = st.slider(
            "Máx. tokens",
            100,
            2000,
            max(100, min(2000, int(settings.max_tokens))),
            50,
            help="Tamanho máximo da resposta.",
        )
        max_results = st.slider(
            "Máx. resultados",
            1,
            5,
            max(1, min(5, int(settings.max_results))),
            1,
            help="Quantidade de fontes buscadas no SearXNG a cada pergunta.",
        )

        st.button("Limpar conversa", on_click=_clear_history, use_container_width=True)

    effective_key = custom_api_key.strip() or settings.api_key
    updated = replace(
        settings,
        api_key=effective_key or None,
        model=selected_model,
        custom_model=custom_model.strip(),
        max_tokens=max_tokens,
        max_results=max_results,
    )
    return updated, temperature


def _answer(
    settings: Settings,
    temperature: float,
    question: str,
    memory_facts: Sequence[str],
) -> tuple[str, list[SearchResult]]:
    """Search SearXNG, store page content in session memory, retrieve RAG chunks, and stream."""
    search_tool = create_search_tool(settings.max_results, settings.searxng_base_url)
    with st.spinner("Pesquisando na web..."):
        fresh_results = run_search(search_tool, question)

    # 1. Write the page content into the active session memory.
    st.session_state.page_memory = ingest_pages_into_memory(
        st.session_state.get("page_memory", []),
        fresh_results,
    )

    # 2. Run RAG over the session memory so only the most relevant chunks go to the LLM.
    relevant_chunks = retrieve_relevant_chunks(
        question,
        st.session_state.page_memory,
        top_k=settings.rag_top_k,
    )

    chain = create_chain(settings, temperature)
    raw_answer = str(st.write_stream(stream_answer(chain, question, relevant_chunks, memory_facts)))
    clean_answer = strip_reasoning(raw_answer, keep_reasoning=False)

    return clean_answer or raw_answer, relevant_chunks


def run() -> None:
    """Render and drive the chatbot interface."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🌐", layout="centered")
    st.title("🌐 Chatbot com URLs")
    st.caption(
        "Pesquisa na web com SearXNG, grava páginas na memória da sessão com RAG "
        "e responde citando as fontes."
    )

    _init_state()
    settings, temperature = _sidebar(Settings.from_env())
    _render_history()

    if not settings.is_configured:
        st.info("Informe a `Chave da API` na barra lateral para usar o modelo personalizado.")
        return

    user_query = st.chat_input(INITIAL_MESSAGE)
    if not user_query:
        return

    st.session_state.history.append({"role": "user", "content": user_query, "sources": []})
    st.session_state.user_facts = evaluate_and_write_memory(
        settings,
        user_query,
        st.session_state.user_facts,
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            answer, results = _answer(
                settings,
                temperature,
                user_query,
                st.session_state.user_facts,
            )
        except Exception as exc:  # noqa: BLE001 - surface any provider error to the user
            st.session_state.history.pop()
            st.error(f"Não foi possível responder agora: {exc}")
            return

        _render_sources(results)

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": results})
