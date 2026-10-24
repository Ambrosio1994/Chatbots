"""Tests for the pure helpers in ``ollama_reasoning.utils``."""

from __future__ import annotations

import pytest

from ollama_reasoning.prompts import INITIAL_MESSAGE
from ollama_reasoning.utils import build_history_messages, extract_reasoning, split_reasoning

# The reasoning delimiters are built from pieces so the source stays readable
# even in tooling that mangles angle-bracket tags.
OPEN = "<" + "think" + ">"
CLOSE = "<" + "/think" + ">"


def test_split_reasoning_with_closed_blocks() -> None:
    text = f"{OPEN}Pensando passo a passo.{CLOSE}\nA resposta é 42."

    assert split_reasoning(text) == ("Pensando passo a passo.", "A resposta é 42.")


def test_split_reasoning_is_case_insensitive() -> None:
    text = f"{OPEN.upper()}raciocínio{CLOSE.upper()}resposta"

    assert split_reasoning(text) == ("raciocínio", "resposta")


def test_split_reasoning_joins_multiple_blocks() -> None:
    text = f"{OPEN}primeiro{CLOSE} meio {OPEN}segundo{CLOSE}fim"

    reasoning, answer = split_reasoning(text)

    assert reasoning == "primeiro\nsegundo"
    assert answer == "meio fim"


def test_split_reasoning_without_tags_returns_answer_only() -> None:
    assert split_reasoning("Somente a resposta.") == ("", "Somente a resposta.")


def test_split_reasoning_with_unclosed_tag() -> None:
    text = f"{OPEN}início do raciocínio sem fechar"

    assert split_reasoning(text) == ("início do raciocínio sem fechar", "")


@pytest.mark.parametrize("value", ["", "   ", None])
def test_split_reasoning_handles_empty_input(value: str | None) -> None:
    assert split_reasoning(value or "") == ("", "")


def test_extract_reasoning_prefers_dedicated_field() -> None:
    assert extract_reasoning("resposta", "raciocínio dedicado") == (
        "raciocínio dedicado",
        "resposta",
    )


def test_extract_reasoning_falls_back_to_content_tags() -> None:
    text = f"{OPEN}pensado{CLOSE}resposta"

    assert extract_reasoning(text, None) == ("pensado", "resposta")


def test_build_history_messages_strips_100_percent_of_reasoning_and_welcome() -> None:
    history = [
        {"role": "assistant", "content": INITIAL_MESSAGE, "reasoning": ""},
        {"role": "user", "content": "Quanto é 2+2?", "reasoning": ""},
        {
            "role": "assistant",
            "content": f"{OPEN}in inline{CLOSE}É 4.",
            "reasoning": "raciocínio longo que não pode ir para a memória",
        },
        {"role": "user", "content": "erro", "is_error": True},
    ]

    cleaned = build_history_messages(history, max_turns=5, initial_message=INITIAL_MESSAGE)

    assert cleaned == [
        {"role": "user", "content": "Quanto é 2+2?"},
        {"role": "assistant", "content": "É 4."},
    ]


def test_build_history_messages_keeps_short_window_and_optional_reasoning() -> None:
    history = []
    for i in range(8):
        history.append({"role": "user", "content": f"p{i}", "reasoning": ""})
        history.append({"role": "assistant", "content": f"r{i}", "reasoning": f"think{i}"})

    short = build_history_messages(history, max_turns=4)
    assert len(short) == 8  # 4 pairs
    assert short[0]["content"] == "p4"
    assert "think" not in short[1]["content"]

    with_reasoning = build_history_messages(history, max_turns=1, include_reasoning=True)
    assert len(with_reasoning) == 2
    assert "think7" in with_reasoning[1]["content"]
