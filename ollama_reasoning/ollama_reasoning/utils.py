"""Pure helpers for the local reasoning chatbot.

Reasoning models such as DeepSeek-R1 may expose their chain of thought either in
a dedicated field or inline, wrapped in reasoning tags. These helpers split
"reasoning" from the final "answer" so the UI can present them separately.
"""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)


def split_reasoning(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(reasoning, answer)``.

    Handles reasoning blocks delimited by ``think`` tags (case-insensitive) and
    gracefully degrades when there is no reasoning section.
    """
    if not text or not text.strip():
        return "", ""

    blocks = _THINK_BLOCK.findall(text)
    if blocks:
        reasoning = "\n".join(block.strip() for block in blocks).strip()
        answer = _THINK_BLOCK.sub("", text).strip()
        return reasoning, answer

    open_tag = _THINK_OPEN.search(text)
    if open_tag:
        # Unclosed tag: everything after it is treated as reasoning.
        return text[open_tag.end() :].strip(), ""

    return "", text.strip()


def extract_reasoning(content: str, reasoning_content: str | None = None) -> tuple[str, str]:
    """Return ``(reasoning, answer)``, preferring a dedicated reasoning field.

    Some providers (and newer Ollama versions) return the reasoning separately
    in ``reasoning_content``; when present it takes precedence over the inline
    reasoning tags.
    """
    if reasoning_content and reasoning_content.strip():
        return reasoning_content.strip(), (content or "").strip()
    return split_reasoning(content or "")
