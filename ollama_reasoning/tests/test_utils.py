"""Tests for the pure helpers in ``ollama_reasoning.utils``."""

from __future__ import annotations

import pytest

from ollama_reasoning.utils import extract_reasoning, split_reasoning

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
