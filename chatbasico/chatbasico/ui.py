"""Streamlit interface for the basic chatbot."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from .config import AVAILABLE_MODELS, Settings
from .llm import create_llm, evaluate_and_write_memory, stream_response
from .prompts import INITIAL_MESSAGE, SYSTEM_PROMPT
from .utils import (
    build_chat_messages,
    clamp_temperature,
    strip_reasoning,
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
            min_value=0.0,
            max_value=1.0,
            value=clamp_temperature(settings.temperature),
            step=0.1,
            help="Valores baixos deixam as respostas mais focadas; altos, mais criativas.",
        )

        st.button("Limpar conversa", on_click=_clear_history, use_container_width=True)

    effective_key = custom_api_key.strip() or settings.api_key
    updated = replace(
        settings,
        model=selected_model,
        custom_model=custom_model.strip(),
        api_key=effective_key or None,
    )
    return updated, clamp_temperature(temperature)


def run() -> None:
    """Render and drive the chatbot interface."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🦙", layout="centered")
    st.title("🦙 Chatbot Básico com LlamaIndex")
    st.caption("Converse usando modelos do LlamaIndex com memória inteligente e RAG da sessão.")

    _init_state()
    settings, temperature = _sidebar(Settings.from_env())
    _render_history()

    if not settings.is_configured:
        st.info("Informe a `Chave da API` na barra lateral para usar o modelo personalizado.")
        return

    user_query = st.chat_input("Digite sua dúvida...")
    if not user_query:
        return

    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.user_facts = evaluate_and_write_memory(
        settings,
        user_query,
        st.session_state.user_facts,
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    messages = build_chat_messages(
        SYSTEM_PROMPT,
        st.session_state.history[:-1],
        user_query,
        memory_facts=st.session_state.user_facts,
        rag_top_k=settings.rag_top_k,
        include_reasoning=False,
        include_errors=False,
        initial_message=INITIAL_MESSAGE,
    )

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            llm = create_llm(settings, temperature)
            raw_answer = str(st.write_stream(stream_response(llm, messages)))
            answer = strip_reasoning(raw_answer, keep_reasoning=False) or raw_answer
        except Exception as exc:  # noqa: BLE001 - surface any provider error to the user
            st.session_state.history.pop()
            st.error(f"Não foi possível gerar a resposta: {exc}")
            return

    st.session_state.history.append({"role": "assistant", "content": answer})
