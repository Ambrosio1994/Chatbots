"""SearXNG web search integration (keyless, self-hosted metasearch)."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import httpx

from .config import DEFAULT_SEARXNG_BASE_URL
from .utils import SearchResult, normalize_results

DEFAULT_TIMEOUT = 10.0
MAX_PAGE_CHARS = 4000

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def extract_html_text(raw_html: str, *, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Strip HTML tags/scripts and return clean readable text from a web page."""
    if not raw_html:
        return ""
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    without_tags = _TAG_RE.sub(" ", without_scripts)
    unescaped = html.unescape(without_tags)
    cleaned = _WS_RE.sub(" ", unescaped).strip()
    return cleaned[:max_chars]


@dataclass(frozen=True, slots=True)
class SearxNGSearch:
    """Client for a SearXNG instance with JSON format enabled.

    Exposes an ``invoke({"query": ...})`` method compatible with LangChain tools,
    requires no API key, and optionally fetches the target page HTML so the active
    session memory stores rich page content for RAG.
    """

    max_results: int = 3
    base_url: str = DEFAULT_SEARXNG_BASE_URL
    fetch_page_content: bool = True
    timeout: float = DEFAULT_TIMEOUT

    def _fetch_page(self, client: httpx.Client, url: str) -> str:
        if not self.fetch_page_content or not url.startswith(("http://", "https://")):
            return ""
        try:
            response = client.get(url, follow_redirects=True, timeout=4.0)
            content_type = response.headers.get("content-type", "").lower()
            if response.status_code == 200 and ("text/html" in content_type or not content_type):
                return extract_html_text(response.text)
        except httpx.HTTPError:
            return ""
        return ""

    def invoke(self, payload: dict[str, str] | str) -> dict[str, object]:
        """Query SearXNG and return ``{"query": ..., "results": [...]}``."""
        query = payload["query"] if isinstance(payload, dict) else str(payload)
        endpoint = f"{self.base_url.rstrip('/')}/search"
        headers = {"User-Agent": "Chatbots-Portfolio-SearxNGClient/1.0"}

        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.get(
                endpoint,
                params={"q": query, "format": "json"},
            )
            response.raise_for_status()
            data = response.json()

            raw_items = data.get("results") if isinstance(data, dict) else []
            selected: list[dict[str, str]] = []
            for item in raw_items or []:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                title = str(item.get("title") or url).strip()
                snippet = str(item.get("content") or "").strip()
                page_text = self._fetch_page(client, url)
                selected.append(
                    {
                        "title": title,
                        "url": url,
                        "content": snippet,
                        "raw_content": page_text,
                    }
                )
                if len(selected) >= self.max_results:
                    break

        return {"query": query, "results": selected}


def create_search_tool(
    max_results: int,
    base_url: str = DEFAULT_SEARXNG_BASE_URL,
    *,
    fetch_page_content: bool = True,
) -> SearxNGSearch:
    """Create a SearXNG search tool pointing at ``base_url``."""
    return SearxNGSearch(
        max_results=max_results,
        base_url=base_url or DEFAULT_SEARXNG_BASE_URL,
        fetch_page_content=fetch_page_content,
    )


def run_search(tool: SearxNGSearch, query: str) -> list[SearchResult]:
    """Run ``query`` against the tool and return normalized results."""
    return normalize_results(tool.invoke({"query": query}))
