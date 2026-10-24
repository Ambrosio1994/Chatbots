"""Tests for the SearXNG search integration (no external network calls)."""

from __future__ import annotations

import httpx

from chat_com_url.search import (
    SearxNGSearch,
    create_search_tool,
    extract_html_text,
    run_search,
)


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


def test_extract_html_text_strips_scripts_and_tags() -> None:
    raw = (
        "<html><head><script>evil()</script></head>"
        "<body><h1>Título</h1><p>Texto &amp; mais.</p></body></html>"
    )

    assert extract_html_text(raw) == "Título Texto & mais."


def test_searxng_search_queries_json_endpoint_and_fetches_page(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search":
            assert request.url.params["q"] == "python"
            assert request.url.params["format"] == "json"
            return httpx.Response(
                200,
                json={
                    "query": "python",
                    "results": [
                        {
                            "title": "Python.org",
                            "url": "https://www.python.org/",
                            "content": "Resumo do buscador.",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><p>Conteúdo completo da página oficial do Python.</p></body></html>",
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def mock_client(**kwargs):
        kwargs["transport"] = transport
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", mock_client)

    tool = create_search_tool(max_results=2, base_url="http://searxng.local:8080")
    assert isinstance(tool, SearxNGSearch)

    results = run_search(tool, "python")
    assert len(results) == 1
    assert results[0].title == "Python.org"
    assert results[0].url == "https://www.python.org/"
    assert "Resumo do buscador." in results[0].content
    assert "Conteúdo completo da página oficial do Python." in results[0].content
