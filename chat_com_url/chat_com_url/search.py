"""Tavily web search integration."""

from __future__ import annotations

from langchain_tavily import TavilySearch

from .utils import SearchResult, normalize_results


def create_search_tool(max_results: int, api_key: str | None = None) -> TavilySearch:
    """Create a Tavily search tool.

    ``TavilySearch`` is the current integration from the ``langchain-tavily``
    package, which replaces the deprecated ``TavilySearchResults``.
    """
    kwargs: dict[str, object] = {"max_results": max_results}
    if api_key:
        kwargs["tavily_api_key"] = api_key
    return TavilySearch(**kwargs)


def run_search(tool: TavilySearch, query: str) -> list[SearchResult]:
    """Run ``query`` against the tool and return normalized results."""
    return normalize_results(tool.invoke({"query": query}))
