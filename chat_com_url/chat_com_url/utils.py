"""Pure helpers for the web-search chatbot: normalization, session RAG and selective memory.

Kept free of Streamlit/LangChain imports so the data-shaping, chunking, RAG
retrieval and user-fact extraction logic can be unit tested without any network
access.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
    }
)

# Patterns that signal a durable user fact, preference, or constraint.
_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:meu nome e|me chamo|sou o|sou a|eu sou|trabalho com|trabalho na|trabalho no|"
        r"estudo|moro em|tenho \d+\s*anos|minha profissao|minha area|uso o|uso a|utilizo)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:prefiro|gosto de|nao gosto|odeio|quero que|sempre responda|nunca responda|"
        r"responda apenas|responda sempre|responda em|nao use|evite|seja breve|seja detalhado|"
        r"em topicos|em portugues|em ingles|sem jargao|restricao|apenas fontes|foco em|"
        r"lembre(?:-se)? que|considere que|meu objetivo|minha meta|orcamento)\b",
        re.IGNORECASE,
    ),
)

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_RAG_TOP_K = 4


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result, normalized across providers."""

    title: str
    url: str
    content: str


@dataclass(frozen=True, slots=True)
class PageChunk:
    """A chunk of a web page stored in the active session's memory."""

    title: str
    url: str
    content: str
    tokens: tuple[str, ...] = field(default_factory=tuple)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize ``text`` for retrieval (accent-insensitive, no stopwords)."""
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


def normalize_results(raw: Any) -> list[SearchResult]:
    """Normalize the output of a search tool into ``SearchResult`` objects.

    Handles both the ``langchain-tavily`` shape (a mapping with a ``results``
    key) and the legacy plain-list shape. When ``raw_content`` is present alongside
    ``content``, both are preserved so the session memory stores the richest page
    content available.
    """
    if isinstance(raw, Mapping):
        items = raw.get("results") or []
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = raw
    else:
        items = []

    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url or "Fonte").strip()
        snippet = str(item.get("content") or "").strip()
        raw_page = str(item.get("raw_content") or "").strip()
        if raw_page and snippet and snippet not in raw_page:
            content = f"{snippet}\n\n{raw_page}"
        else:
            content = raw_page or snippet
        results.append(SearchResult(title=title, url=url, content=content))

    return results


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping character chunks respecting paragraph/sentence boundaries."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    step = max(1, chunk_size - max(0, overlap))
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned):
            # Prefer breaking at sentence or space boundary.
            boundary = max(cleaned.rfind(". ", start, end), cleaned.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = cleaned[start:end].strip()
        if piece and (not chunks or piece != chunks[-1]):
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(start + step, end - max(0, overlap))
    return chunks


def ingest_pages_into_memory(
    existing_chunks: Sequence[PageChunk],
    results: Sequence[SearchResult],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[PageChunk]:
    """Write the content of searched pages into the active session memory without duplicates."""
    memory = list(existing_chunks)
    seen = {(chunk.url, chunk.content) for chunk in memory}

    for result in results:
        for piece in chunk_text(result.content, chunk_size=chunk_size, overlap=overlap):
            key = (result.url, piece)
            if key in seen:
                continue
            seen.add(key)
            memory.append(
                PageChunk(
                    title=result.title,
                    url=result.url,
                    content=piece,
                    tokens=tuple(tokenize(f"{result.title} {piece}")),
                )
            )
    return memory


def retrieve_relevant_chunks(
    question: str,
    chunks: Sequence[PageChunk],
    *,
    top_k: int = DEFAULT_RAG_TOP_K,
) -> list[SearchResult]:
    """RAG retrieval over session page memory using BM25 scoring.

    Returns at most ``top_k`` relevant ``SearchResult`` chunks for ``question``,
    avoiding sending the entire accumulated page content to the LLM.
    """
    if not chunks or top_k <= 0:
        return []

    query_tokens = tokenize(question)
    if not query_tokens:
        selected = list(chunks[-top_k:])
        return [SearchResult(title=c.title, url=c.url, content=c.content) for c in selected]

    n_docs = len(chunks)
    avg_len = sum(len(c.tokens) for c in chunks) / max(1, n_docs)
    k1 = 1.5
    b = 0.75

    # Document frequencies for query terms
    df: dict[str, int] = {}
    for term in set(query_tokens):
        df[term] = sum(1 for c in chunks if term in c.tokens)

    scored: list[tuple[float, int, PageChunk]] = []
    for idx, chunk in enumerate(chunks):
        doc_len = len(chunk.tokens)
        if doc_len == 0:
            continue
        score = 0.0
        for term in query_tokens:
            tf = chunk.tokens.count(term)
            if tf == 0:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_len)))
            score += idf * ((tf * (k1 + 1.0)) / denom)
        # Small recency tie-breaker so newer pages win when scores tie
        scored.append((score, idx, chunk))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    positive = [item[2] for item in scored if item[0] > 0][:top_k]
    if not positive:
        # Fallback: take the most recent chunks up to top_k
        positive = list(chunks[-top_k:])

    return [SearchResult(title=c.title, url=c.url, content=c.content) for c in positive]


def extract_user_facts(text: str) -> list[str]:
    """Extract only relevant user facts, preferences, and constraints from ``text``.

    Questions without personal facts/preferences/constraints return an empty list,
    keeping casual chit-chat out of long-term session memory.
    """
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


def parse_memory_evaluation(raw_output: str, user_prompt: str = "") -> list[str]:
    """Parse the output of the dedicated memory-evaluator model into clean facts."""
    cleaned = strip_reasoning(raw_output or "", keep_reasoning=False)
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = re.sub(r"^[\s\-*•\d.)]+", "", raw_line).strip()
        if not line:
            continue
        upper = _strip_accents(line.upper())
        if upper in {"NENHUM", "NADA", "NONE", "NULL", "NAO", "N/A"} or upper.startswith("NENHUM"):
            continue
        lines.append(line)

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


def update_memory_facts(
    existing_facts: Sequence[str],
    messages: Sequence[Mapping[str, object]],
    *,
    include_reasoning: bool = False,
    include_errors: bool = False,
    initial_message: str | None = None,
) -> list[str]:
    """Update the list of curated user facts/preferences/constraints from conversation history.

    - Never includes ``initial_message``.
    - Never includes errors unless ``include_errors=True``.
    - Never includes reasoning unless ``include_reasoning=True``.
    - Extracts only relevant user facts, preferences, and constraints.
    """
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
            candidate_list = [f"[Erro registrado]: {content}"]
        elif role == "user":
            candidate_list = extract_user_facts(content)
            if include_reasoning and message.get("reasoning"):
                candidate_list.append(f"[Raciocínio]: {message['reasoning']}")
        elif role == "assistant" and include_reasoning:
            reasoning_text = str(message.get("reasoning") or "").strip()
            if not reasoning_text:
                m = _THINK_BLOCK.search(raw_content)
                reasoning_text = m.group(1).strip() if m else ""
            candidate_list = [f"[Raciocínio]: {reasoning_text}"] if reasoning_text else []
        else:
            candidate_list = []

        for item in candidate_list:
            norm = _strip_accents(item.lower())
            if norm and norm not in seen:
                seen.add(norm)
                facts.append(item)

    return facts


def format_sources(results: Sequence[SearchResult]) -> str:
    """Render the sources as a Markdown numbered list (deduplicated by URL)."""
    if not results:
        return "_Nenhuma fonte encontrada._"

    lines = []
    seen_urls: set[str] = set()
    index = 1
    for result in results:
        key = result.url or result.title
        if key in seen_urls:
            continue
        seen_urls.add(key)
        label = result.title or result.url or "Fonte"
        if result.url:
            lines.append(f"{index}. [{label}]({result.url})")
        else:
            lines.append(f"{index}. {label}")
        index += 1
    return "\n".join(lines)


def build_prompt_variables(
    question: str,
    results: Sequence[SearchResult],
    memory_facts: Sequence[str] = (),
) -> dict[str, str]:
    """Build the input variables for the answer chain.

    Only the RAG-selected ``results`` chunks and the curated ``memory_facts``
    (user facts, preferences and constraints) are sent to the LLM.
    """
    search_results = "\n\n".join(result.content for result in results if result.content)
    seen_urls: list[str] = []
    for result in results:
        if result.url and result.url not in seen_urls:
            seen_urls.append(result.url)
    urls = "\n".join(f"- {url}" for url in seen_urls)
    memory_block = "\n".join(f"- {fact}" for fact in memory_facts if fact.strip())

    return {
        "question": question,
        "memory_facts": memory_block or "Nenhuma preferência ou fato registrado ainda.",
        "search_results": search_results or "Nenhum resultado encontrado.",
        "urls": urls or "Nenhuma URL encontrada.",
    }
