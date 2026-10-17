"""Pure helpers for the web-search chatbot.

Kept free of Streamlit/LangChain imports so the data-shaping logic can be unit
tested without any network access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result, normalized across providers."""

    title: str
    url: str
    content: str


def normalize_results(raw: Any) -> list[SearchResult]:
    """Normalize the output of a search tool into ``SearchResult`` objects.

    Handles both the ``langchain-tavily`` shape (a mapping with a ``results``
    key) and the legacy plain-list shape. Invalid entries are skipped.
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
        content = str(item.get("content") or item.get("raw_content") or "").strip()
        results.append(SearchResult(title=title, url=url, content=content))

    return results


def format_sources(results: Sequence[SearchResult]) -> str:
    """Render the sources as a Markdown numbered list."""
    if not results:
        return "_Nenhuma fonte encontrada._"

    lines = []
    for index, result in enumerate(results, start=1):
        label = result.title or result.url
        lines.append(f"{index}. [{label}]({result.url})")
    return "\n".join(lines)


def build_prompt_variables(question: str, results: Sequence[SearchResult]) -> dict[str, str]:
    """Build the input variables for the answer chain."""
    search_results = "\n\n".join(result.content for result in results if result.content)
    urls = "\n".join(f"- {result.url}" for result in results if result.url)

    return {
        "question": question,
        "search_results": search_results or "Nenhum resultado encontrado.",
        "urls": urls or "Nenhuma URL encontrada.",
    }
