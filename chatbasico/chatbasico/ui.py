"""Streamlit interface for the basic chatbot."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from .config import AVAILABLE_MODELS, Settings
from .llm import create_llm, stream_response
from .prompts import INITIAL_MESSAGE, SYSTEM_PROMPT
from .utils import (
    build_chat_messages,
    clamp_temperature,
    index_history_turns,
    strip_reasoning,
    update_memory_facts,
)

PAGE_TITLE = "Chatbot Básico"
ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "🧑"


def _initial_history() -> list[dict[str, object]]:
    return [{"role": "assistant", "content": INITIAL_MESSAGE, "include_in_memory": False}]


def _init_state() -> None:
    st.session_state.setdefault("history", _initial_history())
    st.session_state.setdefault("user_facts", [])


def _clear_history() -> None:
    st.session_state.history = _initial_history()
    st.session_state.user_facts = []


def _render_history() -> None:
    for message in st.session_state.history:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(str(message["role"]), avatar=avatar):
            st.markdown(str(message["content"]))


def _sidebar(settings: Settings) -> tuple[Settings, float]:
    with st.sidebar:
        st.header("Configurações")

        default_model = (
            settings.model if settings.model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        )
        model = st.selectbox(
            "Modelo", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(default_model)
        )

        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=clamp_temperature(settings.temperature),
            step=0.1,
            help="Valores baixos deixam as respostas mais focadas; altos, mais criativas.",
        )

        rag_top_k = st.slider(
            "Turnos RAG por pergunta",
            min_value=0,
            max_value=8,
            value=max(0, min(8, int(settings.rag_top_k))),
            step=1,
            help="Quantidade máxima de turnos relevantes recuperados da memória via RAG.",
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

        turns = index_history_turns(
            st.session_state.get("history", []),
            include_reasoning=include_reasoning,
            include_errors=include_errors,
            initial_message=INITIAL_MESSAGE,
        )
        user_facts: list[str] = st.session_state.get("user_facts", [])
        with st.expander(f"Memória da sessão ({len(turns)} turnos | {len(user_facts)} fatos)"):
            st.caption("**Fatos, preferências e restrições do usuário:**")
            if user_facts:
                for fact in user_facts:
                    st.markdown(f"- {fact}")
            else:
                st.caption("_Nenhum fato/preferência registrado ainda._")
            st.caption(f"**Turnos indexados para RAG:** {len(turns)}")

    updated = replace(
        settings,
        model=model,
        rag_top_k=rag_top_k,
        include_reasoning_in_memory=include_reasoning,
        include_errors_in_memory=include_errors,
    )
    return updated, clamp_temperature(temperature)


def run() -> None:
    """Render and drive the chatbot interface."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🦙", layout="centered")
    st.title("🦙 Chatbot Básico com LlamaIndex + Anthropic")
    st.caption(
        "Converse com o Claude usando memória seletiva de fatos/preferências e RAG da sessão."
    )

    _init_state()
    settings, temperature = _sidebar(Settings.from_env())
    _render_history()

    if not settings.api_key:
        st.info("Defina a variável `ANTHROPIC_API_KEY` no arquivo `.env` para começar a conversar.")
        return

    user_query = st.chat_input("Digite sua dúvida...")
    if not user_query:
        return

    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.user_facts = update_memory_facts(
        st.session_state.user_facts,
        [st.session_state.history[-1]],
        include_reasoning=settings.include_reasoning_in_memory,
        include_errors=settings.include_errors_in_memory,
        initial_message=INITIAL_MESSAGE,
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    messages = build_chat_messages(
        SYSTEM_PROMPT,
        st.session_state.history[:-1],
        user_query,
        memory_facts=st.session_state.user_facts,
        rag_top_k=settings.rag_top_k,
        include_reasoning=settings.include_reasoning_in_memory,
        include_errors=settings.include_errors_in_memory,
        initial_message=INITIAL_MESSAGE,
    )

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            llm = create_llm(settings, temperature)
            raw_answer = str(st.write_stream(stream_response(llm, messages)))
            answer = (
                strip_reasoning(
                    raw_answer,
                    keep_reasoning=settings.include_reasoning_in_memory,
                )
                or raw_answer
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
            st.error(f"Não foi possível gerar a resposta: {exc}")
            return

    assistant_entry = {"role": "assistant", "content": answer}
    st.session_state.history.append(assistant_entry)
    st.session_state.user_facts = update_memory_facts(
        st.session_state.user_facts,
        [assistant_entry],
        include_reasoning=settings.include_reasoning_in_memory,
        include_errors=settings.include_errors_in_memory,
        initial_message=INITIAL_MESSAGE,
    )
