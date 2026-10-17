"""Pure helpers for the basic chatbot.

Nothing here depends on Streamlit or the LLM provider, which keeps the logic
easy to unit test in isolation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0


def clamp_temperature(value: float) -> float:
    """Clamp ``value`` to the temperature range accepted by the model."""
    return max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, float(value)))


def build_chat_messages(
    system_prompt: str,
    history: Sequence[Mapping[str, str]],
    user_query: str,
) -> list[dict[str, str]]:
    """Assemble the message list sent to the model.

    The result always starts with ``system_prompt``, replays ``history`` and
    ends with the current ``user_query``. The input ``history`` is never
    mutated.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": message["role"], "content": message["content"]} for message in history)
    messages.append({"role": "user", "content": user_query})
    return messages
