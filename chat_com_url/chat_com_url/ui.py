"""Streamlit interface for the web-search chatbot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import streamlit as st

from .config import AVAILABLE_MODELS, Settings
from .llm import create_chain, stream_answer
from .prompts import INITIAL_MESSAGE
from .search import create_search_tool, run_search
from .utils import (
    PageChunk,
    SearchResult,
    format_sources,
    ingest_pages_into_memory,
    retrieve_relevant_chunks,
    strip_reasoning,
    update_memory_facts,
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
            help="Quantidade de fontes buscadas na web a cada pergunta.",
        )
        rag_top_k = st.slider(
            "Trechos RAG por pergunta",
            1,
            8,
            max(1, min(8, int(settings.rag_top_k))),
            1,
            help="Quantidade de trechos relevantes da memória de páginas enviados ao LLM.",
        )

        include_reasoning = st.checkbox(
            "Incluir raciocínio na memória",
            value=settings.include_reasoning_in_memory,
            help="Por padrão blocos de raciocínio são ignorados na memória.",
        )
        include_errors = st.checkbox(
            "Incluir erros na memória",
            value=settings.include_errors_in_memory,
            help="Por padrão erros de execução nunca vão para a memória.",
        )

        st.button("Limpar conversa", on_click=_clear_history, use_container_width=True)

        page_chunks: list[PageChunk] = st.session_state.get("page_memory", [])
        user_facts: list[str] = st.session_state.get("user_facts", [])
        with st.expander(
            f"Memória da sessão ({len(page_chunks)} trechos | {len(user_facts)} fatos)"
        ):
            st.caption("**Fatos, preferências e restrições do usuário:**")
            if user_facts:
                for fact in user_facts:
                    st.markdown(f"- {fact}")
            else:
                st.caption("_Nenhum fato/preferência registrado ainda._")
            st.caption(f"**Trechos de páginas indexados para RAG:** {len(page_chunks)}")

    updated = replace(
        settings,
        api_key=api_key or None,
        tavily_api_key=tavily_api_key or None,
        model=model,
        max_tokens=max_tokens,
        max_results=max_results,
        rag_top_k=rag_top_k,
        include_reasoning_in_memory=include_reasoning,
        include_errors_in_memory=include_errors,
    )
    return updated, temperature


def _answer(
    settings: Settings,
    temperature: float,
    question: str,
    memory_facts: Sequence[str],
) -> tuple[str, list[SearchResult]]:
    """Search, store page content in session memory, retrieve top RAG chunks, and stream answer."""
    search_tool = create_search_tool(settings.max_results, settings.tavily_api_key)
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
    clean_answer = strip_reasoning(raw_answer, keep_reasoning=settings.include_reasoning_in_memory)

    return clean_answer or raw_answer, relevant_chunks


def run() -> None:
    """Render and drive the chatbot interface."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🌐", layout="centered")
    st.title("🌐 Chatbot com URLs")
    st.caption(
        "Pesquisa na web com Tavily, grava páginas na memória da sessão com RAG "
        "e responde citando as fontes."
    )

    _init_state()
    settings, temperature = _sidebar(Settings.from_env())
    _render_history()

    if not settings.is_configured:
        st.info("Informe as chaves `ANTHROPIC_API_KEY` e `TAVILY_API_KEY` para continuar.")
        return

    user_query = st.chat_input(INITIAL_MESSAGE)
    if not user_query:
        return

    st.session_state.history.append({"role": "user", "content": user_query, "sources": []})
    st.session_state.user_facts = update_memory_facts(
        st.session_state.user_facts,
        [st.session_state.history[-1]],
        include_reasoning=settings.include_reasoning_in_memory,
        include_errors=settings.include_errors_in_memory,
        initial_message=INITIAL_MESSAGE,
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
            failed_msg = st.session_state.history.pop()
            if settings.include_errors_in_memory:
                failed_msg["is_error"] = True
                failed_msg["content"] = str(exc)
                st.session_state.user_facts = update_memory_facts(
                    st.session_state.user_facts,
                    [failed_msg],
                    include_reasoning=settings.include_reasoning_in_memory,
                    include_errors=True,
                    initial_message=INITIAL_MESSAGE,
                )
            st.error(f"Não foi possível responder agora: {exc}")
            return

        _render_sources(results)

    assistant_entry = {"role": "assistant", "content": answer, "sources": results}
    st.session_state.history.append(assistant_entry)
    st.session_state.user_facts = update_memory_facts(
        st.session_state.user_facts,
        [assistant_entry],
        include_reasoning=settings.include_reasoning_in_memory,
        include_errors=settings.include_errors_in_memory,
        initial_message=INITIAL_MESSAGE,
    )
