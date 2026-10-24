"""Streamlit interface for the local reasoning chatbot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import streamlit as st

from .config import Settings
from .llm import (
    build_chain,
    create_llm,
    evaluate_and_write_memory,
    stream_answer,
    supports_thinking,
)
from .prompts import INITIAL_MESSAGE
from .utils import REASONING, build_history_messages

PAGE_TITLE = "Chatbot com Raciocínio"
ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "🧑"
REASONING_LABEL = "🧠 Raciocínio do modelo"
THINKING_LABEL = "🧠 Pensando..."
WAITING_LABEL = "Aguardando o modelo..."
CURSOR = " ▌"


def _initial_history() -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": INITIAL_MESSAGE,
            "reasoning": "",
            "include_in_memory": False,
        }
    ]


def _init_state() -> None:
    st.session_state.setdefault("history", _initial_history())
    st.session_state.setdefault("user_facts", [])


def _clear_history() -> None:
    st.session_state.history = _initial_history()
    st.session_state.user_facts = []


def _render_reasoning(reasoning: str) -> None:
    if reasoning:
        with st.expander(REASONING_LABEL):
            st.markdown(reasoning)


def _render_history() -> None:
    for message in st.session_state.history:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(str(message["role"]), avatar=avatar):
            _render_reasoning(str(message.get("reasoning") or ""))
            st.markdown(str(message["content"]))


def _sidebar(settings: Settings) -> tuple[Settings, float]:
    with st.sidebar:
        st.header("Configurações")

        temperature = st.slider(
            "Temperatura",
            0.0,
            1.0,
            max(0.0, min(1.0, float(settings.temperature))),
            0.1,
            help="Criatividade da resposta.",
        )

        st.button("Limpar conversa", on_click=_clear_history, use_container_width=True)

    return settings, temperature


def _stream_to_screen(
    settings: Settings,
    temperature: float,
    question: str,
    history: Sequence[Mapping[str, str]],
    memory_facts: Sequence[str],
) -> tuple[str, str]:
    """Stream the model output, showing reasoning and answer as they arrive."""
    reasoning_slot = st.empty()
    answer_slot = st.empty()
    answer_slot.markdown(f"_{WAITING_LABEL}_")

    llm = create_llm(settings, temperature, reasoning=supports_thinking(settings) or None)
    chain = build_chain(llm)

    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    status = reasoning_box = None

    for kind, text in stream_answer(chain, question, history=history, memory_facts=memory_facts):
        if kind == REASONING:
            if status is None:
                status = reasoning_slot.status(THINKING_LABEL, expanded=True)
                reasoning_box = status.empty()
            reasoning_parts.append(text)
            reasoning_box.markdown("".join(reasoning_parts).strip() + CURSOR)
        else:
            if status is not None and not answer_parts:
                reasoning_box.markdown("".join(reasoning_parts).strip())
                status.update(label=REASONING_LABEL, state="complete", expanded=False)
            answer_parts.append(text)
            answer_slot.markdown("".join(answer_parts).strip() + CURSOR)

    reasoning = "".join(reasoning_parts).strip()
    answer = "".join(answer_parts).strip()

    if status is not None and not answer_parts:
        reasoning_box.markdown(reasoning)
        status.update(label=REASONING_LABEL, state="complete", expanded=False)
    answer_slot.markdown(answer or "_O modelo não retornou uma resposta._")

    return reasoning, answer


def run() -> None:
    """Render and drive the chatbot interface."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🧠", layout="centered")
    st.title("🧠 Chatbot local com raciocínio")
    st.caption("Roda 100% na sua máquina. O raciocínio do modelo é exibido à parte da resposta.")

    settings, temperature = _sidebar(Settings.from_env())
    _init_state()
    _render_history()

    user_query = st.chat_input("Digite sua dúvida...")
    if not user_query:
        return

    prior_memory = build_history_messages(
        st.session_state.history,
        max_turns=settings.max_history_turns,
        include_reasoning=False,
        initial_message=INITIAL_MESSAGE,
    )

    st.session_state.history.append({"role": "user", "content": user_query, "reasoning": ""})
    st.session_state.user_facts = evaluate_and_write_memory(
        settings,
        user_query,
        st.session_state.user_facts,
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            reasoning, answer = _stream_to_screen(
                settings,
                temperature,
                user_query,
                prior_memory,
                st.session_state.user_facts,
            )
        except Exception as exc:  # noqa: BLE001 - surface connectivity errors to the user
            st.session_state.history.pop()
            st.error(
                "Não foi possível gerar a resposta. "
                f"O Ollama está rodando em {settings.base_url}? Detalhe: {exc}"
            )
            return

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "reasoning": reasoning}
    )
