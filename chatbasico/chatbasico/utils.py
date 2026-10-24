"""Pure helpers for the basic chatbot: temperature clamping, selective fact memory and RAG.

Nothing here depends on Streamlit or the LLM provider, which keeps the logic
easy to unit test in isolation.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0
DEFAULT_RAG_TOP_K = 3

_THINK_BLOCK = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]{2,}")

_STOPWORDS = frozenset(
    {
        "de",
        "da",
        "do",
        "das",
        "dos",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "uns",
        "umas",
        "para",
        "por",
        "com",
        "sem",
        "sob",
        "sobre",
        "que",
        "qual",
        "quais",
        "como",
        "onde",
        "quando",
        "porque",
        "isso",
        "isto",
        "aquilo",
        "ele",
        "ela",
        "eles",
        "elas",
        "voce",
        "voces",
        "meu",
        "minha",
        "meus",
        "minhas",
        "seu",
        "sua",
        "seus",
        "suas",
        "foi",
        "sao",
        "ser",
        "esta",
        "estao",
        "tem",
        "ter",
        "mais",
        "menos",
        "muito",
        "muita",
        "ola",
        "oi",
        "bom",
        "boa",
    }
)

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


@dataclass(frozen=True, slots=True)
class TurnChunk:
    """A clean (user, assistant) exchange indexed in the session memory for RAG."""

    user: str
    assistant: str
    tokens: tuple[str, ...] = field(default_factory=tuple)


def clamp_temperature(value: float) -> float:
    """Clamp ``value`` to the temperature range accepted by the model."""
    return max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, float(value)))


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize ``text`` for BM25 RAG retrieval."""
    lowered = _strip_accents(text.lower())
    return [tok for tok in _WORD_RE.findall(lowered) if tok not in _STOPWORDS]


def strip_reasoning(text: str, *, keep_reasoning: bool = False) -> str:
    """Remove ``<think>...</think>`` reasoning blocks unless ``keep_reasoning=True``."""
    if not text:
        return ""
    if keep_reasoning:
        return text.strip()
    cleaned = _THINK_BLOCK.sub("", text)
    open_match = _THINK_OPEN.search(cleaned)
    if open_match:
        cleaned = cleaned[: open_match.start()]
    return cleaned.strip()


def extract_user_facts(text: str) -> list[str]:
    """Extract only relevant user facts, preferences and constraints from ``text``."""
    if not text or not text.strip():
        return []

    clean_text = strip_reasoning(text, keep_reasoning=False)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;\n])\s+", clean_text) if s.strip()]
    facts: list[str] = []

    for sentence in sentences:
        normalized = _strip_accents(sentence)
        if any(pattern.search(normalized) for pattern in _FACT_PATTERNS):
            fact = sentence.strip()
            if fact and fact not in facts:
                facts.append(fact)

    return facts


def update_memory_facts(
    existing_facts: Sequence[str],
    messages: Sequence[Mapping[str, object]],
    *,
    include_reasoning: bool = False,
    include_errors: bool = False,
    initial_message: str | None = None,
) -> list[str]:
    """Update the curated user facts/preferences/constraints from conversation messages."""
    facts = list(existing_facts)
    seen = {_strip_accents(f.lower()) for f in facts}

    for idx, message in enumerate(messages):
        if not message.get("include_in_memory", True):
            continue
        if message.get("is_error", False) and not include_errors:
            continue

        role = str(message.get("role") or "").strip()
        raw_content = str(message.get("content") or "").strip()

        if (
            idx == 0
            and role == "assistant"
            and (initial_message is None or raw_content == initial_message.strip())
        ):
            continue

        content = strip_reasoning(raw_content, keep_reasoning=include_reasoning)
        if not content:
            continue

        if message.get("is_error", False) and include_errors:
            candidates = [f"[Erro registrado]: {content}"]
        elif role == "user":
            candidates = extract_user_facts(content)
            if include_reasoning and message.get("reasoning"):
                candidates.append(f"[Raciocínio]: {message['reasoning']}")
        elif role == "assistant" and include_reasoning:
            reasoning_text = str(message.get("reasoning") or "").strip()
            if not reasoning_text:
                m = _THINK_BLOCK.search(raw_content)
                reasoning_text = m.group(1).strip() if m else ""
            candidates = [f"[Raciocínio]: {reasoning_text}"] if reasoning_text else []
        else:
            candidates = []

        for item in candidates:
            norm = _strip_accents(item.lower())
            if norm and norm not in seen:
                seen.add(norm)
                facts.append(item)

    return facts


def index_history_turns(
    history: Sequence[Mapping[str, object]],
    *,
    include_reasoning: bool = False,
    include_errors: bool = False,
    initial_message: str | None = None,
) -> list[TurnChunk]:
    """Pair valid ``user`` -> ``assistant`` turns from ``history`` for RAG indexing.

    Skips the initial greeting (``initial_message``), messages marked with
    ``include_in_memory=False``, and errors/reasoning unless requested.
    """
    filtered: list[tuple[str, str]] = []
    pending_user: str | None = None

    for idx, entry in enumerate(history):
        if not entry.get("include_in_memory", True):
            continue
        if entry.get("is_error", False) and not include_errors:
            continue

        role = str(entry.get("role") or "").strip()
        raw_content = str(entry.get("content") or "").strip()

        if (
            idx == 0
            and role == "assistant"
            and (initial_message is None or raw_content == initial_message.strip())
        ):
            continue

        content = strip_reasoning(raw_content, keep_reasoning=include_reasoning)
        if role == "assistant" and include_reasoning and entry.get("reasoning"):
            content = f"<think>\n{entry['reasoning']}\n</think>\n\n{content}".strip()

        if not content:
            continue

        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            filtered.append((pending_user, content))
            pending_user = None

    return [
        TurnChunk(
            user=u,
            assistant=a,
            tokens=tuple(tokenize(f"{u} {a}")),
        )
        for u, a in filtered
    ]


def retrieve_relevant_turns(
    user_query: str,
    turns: Sequence[TurnChunk],
    *,
    top_k: int = DEFAULT_RAG_TOP_K,
) -> list[TurnChunk]:
    """Retrieve at most ``top_k`` relevant past turns for ``user_query`` via BM25 RAG.

    Preserves chronological order among the selected turns and always includes the
    most recent turn when ``top_k >= 1`` so immediate follow-ups stay coherent.
    """
    if not turns or top_k <= 0:
        return []
    if len(turns) <= top_k:
        return list(turns)

    query_tokens = tokenize(user_query)
    if not query_tokens:
        return list(turns[-top_k:])

    n_docs = len(turns)
    avg_len = sum(len(t.tokens) for t in turns) / max(1, n_docs)
    k1 = 1.5
    b = 0.75
    df = {term: sum(1 for t in turns if term in t.tokens) for term in set(query_tokens)}

    scored: list[tuple[float, int]] = []
    for idx, turn in enumerate(turns):
        doc_len = len(turn.tokens)
        score = 0.0
        if doc_len > 0:
            for term in query_tokens:
                tf = turn.tokens.count(term)
                if tf == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                denom = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_len)))
                score += idf * ((tf * (k1 + 1.0)) / denom)
        scored.append((score, idx))

    # Always reserve 1 slot for the immediately preceding turn for conversational continuity
    last_idx = len(turns) - 1
    chosen_indices: set[int] = {last_idx}

    for score, idx in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True):
        if len(chosen_indices) >= top_k:
            break
        if score > 0:
            chosen_indices.add(idx)

    # If still below top_k and no lexical matches, only keep the most recent turn
    ordered = sorted(chosen_indices)
    return [turns[i] for i in ordered]


def build_chat_messages(
    system_prompt: str,
    history: Sequence[Mapping[str, object]],
    user_query: str,
    *,
    memory_facts: Sequence[str] | None = None,
    rag_top_k: int = DEFAULT_RAG_TOP_K,
    include_reasoning: bool = False,
    include_errors: bool = False,
    initial_message: str | None = None,
) -> list[dict[str, str]]:
    """Assemble the message list sent to the model using selective memory and RAG.

    - Never sends the initial UI welcome greeting (``initial_message``).
    - Never sends errors or reasoning unless explicitly requested.
    - Injects curated user facts, preferences and constraints into the system prompt.
    - Uses BM25 RAG to include only the most relevant past turns (up to ``rag_top_k``)
      instead of dumping the entire session history on every question.
    - The input ``history`` is never mutated.
    """
    facts = (
        list(memory_facts)
        if memory_facts is not None
        else update_memory_facts(
            [],
            [
                *history,
                {"role": "user", "content": user_query},
            ],
            include_reasoning=include_reasoning,
            include_errors=include_errors,
            initial_message=initial_message,
        )
    )

    effective_system = system_prompt
    if facts:
        facts_block = "\n".join(f"- {fact}" for fact in facts if fact.strip())
        effective_system = (
            f"{system_prompt.rstrip()}\n\n"
            "Fatos, preferências e restrições do usuário (memória seletiva):\n"
            f"{facts_block}"
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": effective_system}]

    turns = index_history_turns(
        history,
        include_reasoning=include_reasoning,
        include_errors=include_errors,
        initial_message=initial_message,
    )
    relevant_turns = retrieve_relevant_turns(user_query, turns, top_k=rag_top_k)

    for turn in relevant_turns:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    messages.append(
        {"role": "user", "content": strip_reasoning(user_query, keep_reasoning=include_reasoning)}
    )
    return messages
