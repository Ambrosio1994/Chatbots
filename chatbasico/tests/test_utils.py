"""Tests for the pure helpers in ``chatbasico.utils``."""

from __future__ import annotations

import pytest

from chatbasico.prompts import INITIAL_MESSAGE
from chatbasico.utils import (
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    build_chat_messages,
    clamp_temperature,
    extract_user_facts,
    index_history_turns,
    retrieve_relevant_turns,
    update_memory_facts,
)

OPEN = "<" + "think" + ">"
CLOSE = "<" + "/think" + ">"


def test_build_chat_messages_starts_with_system_and_ends_with_query() -> None:
    messages = build_chat_messages("be nice", [], "oi")

    assert messages[0] == {"role": "system", "content": "be nice"}
    assert messages[-1] == {"role": "user", "content": "oi"}


def test_build_chat_messages_replays_history_in_order() -> None:
    history = [
        {"role": "user", "content": "primeira"},
        {"role": "assistant", "content": "resposta"},
    ]

    messages = build_chat_messages("system", history, "segunda")

    assert messages[1:] == [
        {"role": "user", "content": "primeira"},
        {"role": "assistant", "content": "resposta"},
        {"role": "user", "content": "segunda"},
    ]


def test_build_chat_messages_excludes_initial_welcome_message() -> None:
    history = [
        {"role": "assistant", "content": INITIAL_MESSAGE, "include_in_memory": False},
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "olá!"},
    ]

    messages = build_chat_messages("system", history, "tudo bem?", initial_message=INITIAL_MESSAGE)

    assert all(m["content"] != INITIAL_MESSAGE for m in messages)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


def test_build_chat_messages_excludes_errors_and_reasoning_by_default() -> None:
    history = [
        {"role": "user", "content": "pergunta que falhou", "is_error": True},
        {"role": "user", "content": "pergunta boa"},
        {"role": "assistant", "content": f"{OPEN}segredo interno{CLOSE}resposta limpa"},
    ]

    messages = build_chat_messages("system", history, "nova pergunta")

    serialized = " ".join(m["content"] for m in messages)
    assert "pergunta que falhou" not in serialized
    assert "segredo interno" not in serialized
    assert "resposta limpa" in serialized


def test_build_chat_messages_includes_reasoning_when_requested() -> None:
    history = [
        {"role": "user", "content": "pergunta"},
        {"role": "assistant", "content": f"{OPEN}passo 1{CLOSE}resposta"},
    ]

    messages = build_chat_messages("system", history, "nova", include_reasoning=True)

    assert "passo 1" in messages[2]["content"]


def test_extract_user_facts_keeps_only_facts_preferences_and_constraints() -> None:
    assert extract_user_facts("Qual a capital da França?") == []

    facts = extract_user_facts(
        "Meu nome é Ana e trabalho com ciência de dados. "
        "Prefiro respostas em tópicos e sem jargão. Qual o melhor banco?"
    )

    assert len(facts) == 2
    assert any("Meu nome é Ana" in f for f in facts)
    assert any("Prefiro respostas em tópicos" in f for f in facts)


def test_update_memory_facts_ignores_greeting_and_errors_unless_requested() -> None:
    messages = [
        {"role": "assistant", "content": INITIAL_MESSAGE, "include_in_memory": False},
        {"role": "user", "content": "Sou desenvolvedor Python e prefiro exemplos curtos."},
        {"role": "user", "content": "timeout error", "is_error": True},
    ]

    facts = update_memory_facts([], messages, initial_message=INITIAL_MESSAGE)
    assert len(facts) == 1
    assert "Python" in facts[0]

    facts_with_err = update_memory_facts(
        [], messages, include_errors=True, initial_message=INITIAL_MESSAGE
    )
    assert any("[Erro registrado]" in f for f in facts_with_err)


def test_rag_retrieves_only_relevant_turns_instead_of_full_history() -> None:
    history = [
        {"role": "user", "content": "Receita de bolo de cenoura com chocolate"},
        {"role": "assistant", "content": "Use cenoura, óleo, ovos, açúcar e farinha."},
        {"role": "user", "content": "Como configurar Docker Compose com Postgres"},
        {"role": "assistant", "content": "Defina o serviço postgres:15 e mapeie a porta 5432."},
        {"role": "user", "content": "Roteiro de viagem para Lisboa em Portugal"},
        {"role": "assistant", "content": "Visite Belém, Alfama e Sintra."},
        {"role": "user", "content": "Qual o clima hoje?"},
        {"role": "assistant", "content": "Ensolarado."},
    ]

    turns = index_history_turns(history)
    selected = retrieve_relevant_turns("Qual porta usar no Postgres com Docker?", turns, top_k=2)

    assert len(selected) == 2
    assert any("Docker" in t.user for t in selected)
    assert all("bolo de cenoura" not in t.user for t in selected)


def test_build_chat_messages_does_not_mutate_history() -> None:
    history = [{"role": "user", "content": "oi"}]

    build_chat_messages("system", history, "tchau")

    assert history == [{"role": "user", "content": "oi"}]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, MIN_TEMPERATURE), (0.5, 0.5), (2.0, MAX_TEMPERATURE)],
)
def test_clamp_temperature_keeps_value_in_range(value: float, expected: float) -> None:
    assert clamp_temperature(value) == expected
