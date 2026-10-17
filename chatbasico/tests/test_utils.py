"""Tests for the pure helpers in ``chatbasico.utils``."""

from __future__ import annotations

import pytest

from chatbasico.utils import (
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    build_chat_messages,
    clamp_temperature,
)


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
