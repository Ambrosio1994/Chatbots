"""Pure helpers for the local reasoning chatbot.

Reasoning models such as DeepSeek-R1 may expose their chain of thought either in
a dedicated field or inline, wrapped in reasoning tags. These helpers split
"reasoning" from the final "answer", filter conversation history, and parse the
output of the dedicated memory-evaluator model.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

_THINK_BLOCK = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)

_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:meu nome e|me chamo|sou o|sou a|eu sou|trabalho com|trabalho na|trabalho no|"
        r"estudo|moro em|tenho \d+\s*anos|minha profissao|minha area|uso o|uso a|utilizo)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:prefiro|gosto de|nao gosto|odeio|quero que|sempre responda|nunca responda|"
        r"responda apenas|responda sempre|responda em|nao use|evite|seja breve|seja detalhado|"
        r"em topicos|em portugues|em ingles|sem jargao|restricao|foco em|"
        r"lembre(?:-se)? que|considere que|meu objetivo|minha meta|orcamento)\b",
        re.IGNORECASE,
    ),
)

DEFAULT_MAX_HISTORY_TURNS = 5


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


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


REASONING = "reasoning"
ANSWER = "answer"

_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")
_CLOSE_TAG = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


def _partial_tag_start(buffer: str, tags: tuple[str, ...]) -> int:
    """Return where a possibly incomplete trailing tag starts, or ``-1``."""
    start = buffer.rfind("<")
    if start == -1:
        return -1
    tail = buffer[start:].lower()
    return start if any(tag.startswith(tail) and tag != tail for tag in tags) else -1


class ReasoningStreamSplitter:
    """Incrementally split a token stream into reasoning and answer pieces."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_reasoning = False
        self._reasoning: list[str] = []
        self._answer: list[str] = []

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning).strip()

    @property
    def answer(self) -> str:
        return "".join(self._answer).strip()

    def _emit(self, kind: str, text: str, events: list[tuple[str, str]]) -> None:
        if not text:
            return
        (self._reasoning if kind == REASONING else self._answer).append(text)
        events.append((kind, text))

    def feed(
        self, content: str = "", reasoning_content: str | None = None
    ) -> list[tuple[str, str]]:
        """Consume one chunk and return the events it produced."""
        events: list[tuple[str, str]] = []
        if reasoning_content:
            self._emit(REASONING, reasoning_content, events)

        self._buffer += content or ""
        while self._buffer:
            pattern = _CLOSE_TAG if self._in_reasoning else _THINK_OPEN
            tags = _CLOSE_TAGS if self._in_reasoning else _OPEN_TAGS
            kind = REASONING if self._in_reasoning else ANSWER

            match = pattern.search(self._buffer)
            if match:
                self._emit(kind, self._buffer[: match.start()], events)
                self._buffer = self._buffer[match.end() :]
                self._in_reasoning = not self._in_reasoning
                continue

            held = _partial_tag_start(self._buffer, tags)
            if held == -1:
                self._emit(kind, self._buffer, events)
                self._buffer = ""
            else:
                self._emit(kind, self._buffer[:held], events)
                self._buffer = self._buffer[held:]
            break

        return events

    def flush(self) -> list[tuple[str, str]]:
        """Release any text still held back (e.g. a lone ``<`` at the end)."""
        events: list[tuple[str, str]] = []
        self._emit(REASONING if self._in_reasoning else ANSWER, self._buffer, events)
        self._buffer = ""
        return events


def extract_user_facts(text: str) -> list[str]:
    """Deterministic extractor for user facts, preferences and constraints."""
    if not text or not text.strip():
        return []
    _, clean_text = split_reasoning(text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;\n])\s+", clean_text) if s.strip()]
    facts: list[str] = []
    for sentence in sentences:
        if (
            any(pattern.search(_strip_accents(sentence)) for pattern in _FACT_PATTERNS)
            and sentence not in facts
        ):
            facts.append(sentence)
    return facts


def parse_memory_evaluation(raw_output: str, user_prompt: str = "") -> list[str]:
    """Parse the output of the dedicated memory-evaluator model into clean facts.

    Strips any ``<think>`` blocks, ignores ``NENHUM`` / empty markers, and
    cross-checks against ``user_prompt`` so hallucinated or irrelevant lines
    never pollute session memory.
    """
    _, cleaned = split_reasoning(raw_output or "")
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = re.sub(r"^[\s\-*•\d.)]+", "", raw_line).strip()
        if not line:
            continue
        upper = _strip_accents(line.upper())
        if upper in {"NENHUM", "NADA", "NONE", "NULL", "NAO", "N/A"} or upper.startswith("NENHUM"):
            continue
        lines.append(line)

    # If the user prompt itself has no durable fact/preference signal, ignore noisy LLM output
    heuristic_facts = extract_user_facts(user_prompt) if user_prompt else []
    if user_prompt and not heuristic_facts:
        return []
    return lines or heuristic_facts


def merge_memory_facts(existing_facts: Sequence[str], new_facts: Sequence[str]) -> list[str]:
    """Append ``new_facts`` into ``existing_facts`` without duplicates."""
    merged = list(existing_facts)
    seen = {_strip_accents(f.lower()) for f in merged}
    for item in new_facts:
        cleaned = item.strip()
        norm = _strip_accents(cleaned.lower())
        if cleaned and norm not in seen:
            seen.add(norm)
            merged.append(cleaned)
    return merged


def build_history_messages(
    history: Sequence[Mapping[str, object]],
    *,
    max_turns: int = DEFAULT_MAX_HISTORY_TURNS,
    include_reasoning: bool = False,
    initial_message: str | None = None,
) -> list[dict[str, str]]:
    """Filter and trim session ``history`` into clean messages for the LLM.

    Rules enforced:
    - Never include the UI greeting (``initial_message``) or messages flagged
      with ``include_in_memory=False`` or ``is_error=True``.
    - Strip 100% of ``reasoning`` / ``<think>`` blocks from previous turns by
      default, keeping only the question and the clean final answer.
    - Keep only complete, valid ``user`` -> ``assistant`` turns within the last
      ``max_turns`` window so CPU inference stays fast.
    """
    if max_turns <= 0 or not history:
        return []

    cleaned: list[dict[str, str]] = []
    for idx, entry in enumerate(history):
        if not entry.get("include_in_memory", True) or entry.get("is_error", False):
            continue

        role = str(entry.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue

        raw_content = str(entry.get("content") or "").strip()
        reasoning_field = str(entry.get("reasoning") or "").strip()

        if (
            idx == 0
            and role == "assistant"
            and (initial_message is None or raw_content == initial_message.strip())
        ):
            continue

        inline_reasoning, clean_answer = split_reasoning(raw_content)
        reasoning = reasoning_field or inline_reasoning
        content = clean_answer if role == "assistant" else raw_content

        if role == "assistant" and include_reasoning and reasoning:
            content = f"<think>\n{reasoning}\n</think>\n\n{content}".strip()

        if not content:
            continue

        if not cleaned and role != "user":
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] = content
            continue

        cleaned.append({"role": role, "content": content})

    if cleaned and cleaned[-1]["role"] == "user":
        cleaned.pop()

    max_messages = max_turns * 2
    if len(cleaned) > max_messages:
        cleaned = cleaned[-max_messages:]

    return cleaned
