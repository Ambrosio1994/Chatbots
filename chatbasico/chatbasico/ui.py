"""Streamlit interface for the basic chatbot."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from .config import AVAILABLE_MODELS, Settings
from .llm import create_llm, stream_response
from .prompts import INITIAL_MESSAGE, SYSTEM_PROMPT
from .utils import build_chat_messages, clamp_temperature

PAGE_TITLE = "Chatbot Básico"
ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "🧑"


def _initial_history() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": INITIAL_MESSAGE}]


def _init_state() -> None:
    st.session_state.setdefault("history", _initial_history())


def _clear_history() -> None:
    st.session_state.history = _initial_history()


def _render_history() -> None:
    for message in st.session_state.history:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


def _sidebar(settings: Settings) -> tuple[str, float]:
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

        st.button("Limpar conversa", on_click=_clear_history, use_container_width=True)

    return model, clamp_temperature(temperature)


def run() -> None:
    """Render and drive the chatbot interface."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🦙", layout="centered")
    st.title("🦙 Chatbot Básico com LlamaIndex + Anthropic")
    st.caption("Converse com o Claude usando uma arquitetura em camadas, testável e configurável.")

    settings = Settings.from_env()
    model, temperature = _sidebar(settings)
    settings = replace(settings, model=model)

    _init_state()
    _render_history()

    if not settings.api_key:
        st.info("Defina a variável `ANTHROPIC_API_KEY` no arquivo `.env` para começar a conversar.")
        return

    user_query = st.chat_input("Digite sua dúvida...")
    if not user_query:
        return

    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    messages = build_chat_messages(SYSTEM_PROMPT, st.session_state.history[:-1], user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            llm = create_llm(settings, temperature)
            answer = st.write_stream(stream_response(llm, messages))
        except Exception as exc:  # noqa: BLE001 - surface any provider error to the user
            st.session_state.history.pop()
            st.error(f"Não foi possível gerar a resposta: {exc}")
            return

    st.session_state.history.append({"role": "assistant", "content": answer})
