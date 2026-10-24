"""Streamlit interface for the local reasoning chatbot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import streamlit as st

from .config import SUGGESTED_MODELS, Settings
from .llm import build_chain, create_llm, stream_answer, supports_thinking
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


def _clear_history() -> None:
    st.session_state.history = _initial_history()


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

        model = st.text_input(
            "Modelo (Ollama)",
            value=settings.model,
            help="Sugestões: " + ", ".join(SUGGESTED_MODELS),
        )
        st.caption(f"Servidor: {settings.base_url}")

        temperature = st.slider(
            "Temperatura",
            0.0,
            1.0,
            max(0.0, min(1.0, float(settings.temperature))),
            0.1,
            help="Criatividade da resposta.",
        )

        max_turns = st.slider(
            "Memória (últimos turnos)",
            min_value=0,
            max_value=10,
            value=max(0, min(10, int(settings.max_history_turns))),
            step=1,
            help="Quantidade de pares (pergunta + resposta limpa) enviados ao modelo.",
        )

        include_reasoning = st.checkbox(
            "Incluir raciocínio na memória",
            value=settings.include_reasoning_in_memory,
            help=(
                "Por padrão o raciocínio (<think>) é descartado do histórico para "
                "não degradar a velocidade na CPU. Marque apenas se quiser que o "
                "modelo releia o próprio raciocínio anterior."
            ),
        )

        st.button("Limpar conversa", on_click=_clear_history, use_container_width=True)

    return (
        replace(
            settings,
            model=model,
            max_history_turns=max_turns,
            include_reasoning_in_memory=include_reasoning,
        ),
        temperature,
    )


def _stream_to_screen(
    settings: Settings,
    temperature: float,
    question: str,
    history: Sequence[Mapping[str, str]],
) -> tuple[str, str]:
    """Stream the model output, showing reasoning and answer as they arrive.

    The reasoning goes into a live ``st.status`` box that is only created once
    the first reasoning token arrives (models without reasoning never show it)
    and collapses as soon as the final answer starts. Returns the stripped
    ``(reasoning, answer)`` pair to be stored in the history.
    """
    reasoning_slot = st.empty()
    answer_slot = st.empty()
    answer_slot.markdown(f"_{WAITING_LABEL}_")

    llm = create_llm(settings, temperature, reasoning=supports_thinking(settings) or None)
    chain = build_chain(llm)

    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    status = reasoning_box = None

    for kind, text in stream_answer(chain, question, history=history):
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
    st.title("🧠 Chatbot local com raciocínio (Ollama + DeepSeek-R1)")
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
        include_reasoning=settings.include_reasoning_in_memory,
        initial_message=INITIAL_MESSAGE,
    )

    st.session_state.history.append({"role": "user", "content": user_query, "reasoning": ""})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            reasoning, answer = _stream_to_screen(settings, temperature, user_query, prior_memory)
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
