"""Tests for the pure helpers in ``chat_com_url.utils``."""

from __future__ import annotations

from chat_com_url.utils import (
    SearchResult,
    build_prompt_variables,
    format_sources,
    normalize_results,
)


def test_normalize_results_handles_langchain_tavily_mapping() -> None:
    raw = {
        "query": "clima",
        "results": [
            {"title": "Previsão", "url": "https://ex.com", "content": "Ensolarado"},
        ],
    }

    results = normalize_results(raw)

    assert results == [SearchResult(title="Previsão", url="https://ex.com", content="Ensolarado")]


def test_normalize_results_handles_legacy_list_shape() -> None:
    raw = [
        {"title": "A", "url": "https://a.com", "content": "a"},
        {"url": "https://b.com", "raw_content": "b"},
    ]

    results = normalize_results(raw)

    assert [result.url for result in results] == ["https://a.com", "https://b.com"]
    assert results[1].title == "https://b.com"
    assert results[1].content == "b"


def test_normalize_results_skips_invalid_entries() -> None:
    raw = {"results": ["not-a-dict", {"url": "https://ok.com", "content": "ok"}]}

    results = normalize_results(raw)

    assert len(results) == 1
    assert results[0].url == "https://ok.com"


def test_normalize_results_handles_unexpected_types() -> None:
    assert normalize_results(None) == []
    assert normalize_results("oops") == []


def test_format_sources_renders_markdown_links() -> None:
    results = [
        SearchResult(title="Docs", url="https://docs.com", content="x"),
        SearchResult(title="", url="https://bare.com", content="y"),
    ]

    markdown = format_sources(results)

    assert markdown == "1. [Docs](https://docs.com)\n2. [https://bare.com](https://bare.com)"


def test_format_sources_without_results() -> None:
    assert format_sources([]) == "_Nenhuma fonte encontrada._"


def test_build_prompt_variables_uses_fallbacks_when_empty() -> None:
    variables = build_prompt_variables("pergunta?", [])

    assert variables["question"] == "pergunta?"
    assert variables["search_results"] == "Nenhum resultado encontrado."
    assert variables["urls"] == "Nenhuma URL encontrada."


def test_build_prompt_variables_joins_results() -> None:
    results = [
        SearchResult(title="A", url="https://a.com", content="conteudo-a"),
        SearchResult(title="B", url="https://b.com", content="conteudo-b"),
    ]

    variables = build_prompt_variables("q", results)

    assert "conteudo-a" in variables["search_results"]
    assert "conteudo-b" in variables["search_results"]
    assert variables["urls"] == "- https://a.com\n- https://b.com"
