"""Tests for the Tavily search wrapper (no network calls)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_tavily")

from chat_com_url.search import run_search


class FakeTool:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.received: dict[str, str] | None = None

    def invoke(self, payload: dict[str, str]) -> object:
        self.received = payload
        return self.payload


def test_run_search_normalizes_tool_output() -> None:
    tool = FakeTool({"results": [{"url": "https://x.com", "content": "texto"}]})

    results = run_search(tool, "pergunta")

    assert tool.received == {"query": "pergunta"}
    assert len(results) == 1
    assert results[0].url == "https://x.com"
