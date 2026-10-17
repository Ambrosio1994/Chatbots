"""Streamlit interface for the local reasoning chatbot."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from .config import SUGGESTED_MODELS, Settings
from .llm import build_chain, create_llm, generate_answer
from .prompts import INITIAL_MESSAGE

PAGE_TITLE = "Chatbot com Raciocínio"
ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "🧑"
REASONING_LABEL = "🧠 Raciocínio do modelo"


def _initial_history() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": INITIAL_MESSAGE, "reasoning": ""}]


def _init_state() -> None:
    st.session_state.setdefault("history", _initial_history())


def _clear_history() -> None:
    st.session_state.history = _initial_history()


def _render_reasoning(reasoning: str) -> None:
    if reasoning:
        with st.expander(REASONING_LABEL):
            st.markdown(reasoning)


def _render_history() -> None:
    for message in st.session_state.history:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            _render_reasoning(message.get("reasoning", ""))
            st.markdown(message["content"])


def _sidebar(settings: Settings) -> tuple[Settings, float]:
    with st.sidebar:
        st.header("Configurações")

        model = st.text_input(
            "Modelo (Ollama)",
            value=settings.model,
            help="Sugestões: " + ", ".join(SUGGESTED_MODELS),
        )
        st.caption(f"Servidor: {settings.base_url}")

        temperature = st.slider(
            "Temperatura", 0.0, 1.0, settings.temperature, 0.1, help="Criatividade da resposta."
        )

        st.button("Limpar conversa", on_click=_clear_history, use_container_width=True)

    return replace(settings, model=model), temperature


def run() -> None:
    """Render and drive the chatbot interface."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🧠", layout="centered")
    st.title("🧠 Chatbot local com raciocínio (Ollama + DeepSeek-R1)")
    st.caption("Roda 100% na sua máquina. O raciocínio do modelo é exibido à parte da resposta.")

    settings, temperature = _sidebar(Settings.from_env())
    _init_state()
    _render_history()

    user_query = st.chat_input("Digite sua dúvida...")
    if not user_query:
        return

    st.session_state.history.append({"role": "user", "content": user_query, "reasoning": ""})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            with st.spinner("Pensando..."):
                chain = build_chain(create_llm(settings, temperature))
                reasoning, answer = generate_answer(chain, user_query)
        except Exception as exc:  # noqa: BLE001 - surface connectivity errors to the user
            st.session_state.history.pop()
            st.error(
                "Não foi possível gerar a resposta. "
                f"O Ollama está rodando em {settings.base_url}? Detalhe: {exc}"
            )
            return

        _render_reasoning(reasoning)
        st.markdown(answer)

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "reasoning": reasoning}
    )
