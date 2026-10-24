"""Tests for the pure helpers in ``chat_com_url.utils``."""

from __future__ import annotations

from chat_com_url.prompts import INITIAL_MESSAGE
from chat_com_url.utils import (
    SearchResult,
    build_prompt_variables,
    extract_user_facts,
    format_sources,
    ingest_pages_into_memory,
    normalize_results,
    retrieve_relevant_chunks,
    update_memory_facts,
)

OPEN = "<" + "think" + ">"
CLOSE = "<" + "/think" + ">"


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
    assert "Nenhuma preferência" in variables["memory_facts"]


def test_build_prompt_variables_joins_results_and_facts() -> None:
    results = [
        SearchResult(title="A", url="https://a.com", content="conteudo-a"),
        SearchResult(title="B", url="https://b.com", content="conteudo-b"),
    ]

    variables = build_prompt_variables("q", results, ["Prefiro respostas curtas"])

    assert "conteudo-a" in variables["search_results"]
    assert "conteudo-b" in variables["search_results"]
    assert variables["urls"] == "- https://a.com\n- https://b.com"
    assert "- Prefiro respostas curtas" in variables["memory_facts"]


def test_ingest_pages_and_rag_retrieves_only_relevant_chunks() -> None:
    pages = [
        SearchResult(
            title="Python",
            url="https://py.org",
            content="Python 3.12 trouxe melhorias de performance e mensagens de erro mais claras.",
        ),
        SearchResult(
            title="Culinária",
            url="https://chef.com",
            content="O pão de queijo mineiro leva polvilho azedo, queijo curado, leite e óleo.",
        ),
        SearchResult(
            title="Astronomia",
            url="https://espaco.org",
            content="Júpiter é o maior planeta do Sistema Solar e possui dezenas de luas.",
        ),
    ]

    memory = ingest_pages_into_memory([], pages, chunk_size=200)
    # Re-ingesting the same pages does not duplicate chunks
    assert len(ingest_pages_into_memory(memory, pages, chunk_size=200)) == len(memory)

    rag_results = retrieve_relevant_chunks(
        "Quais ingredientes vão no pão de queijo?", memory, top_k=1
    )

    assert len(rag_results) == 1
    assert rag_results[0].url == "https://chef.com"
    assert "polvilho" in rag_results[0].content


def test_selective_memory_extracts_only_user_facts_and_blocks_errors_and_reasoning() -> None:
    assert extract_user_facts("Onde fica Tóquio?") == []

    messages = [
        {"role": "assistant", "content": INITIAL_MESSAGE, "include_in_memory": False},
        {"role": "user", "content": "Me chamo Carlos e tenho restrição a glúten."},
        {"role": "user", "content": "Qual o preço do arroz?"},
        {"role": "assistant", "content": f"{OPEN}pensamento secreto{CLOSE}Custa R$ 25."},
        {"role": "user", "content": "Falha 500", "is_error": True},
    ]

    facts = update_memory_facts([], messages, initial_message=INITIAL_MESSAGE)
    assert facts == ["Me chamo Carlos e tenho restrição a glúten."]

    with_reasoning = update_memory_facts(
        [], messages, include_reasoning=True, initial_message=INITIAL_MESSAGE
    )
    assert any("pensamento secreto" in f for f in with_reasoning)

    with_errors = update_memory_facts(
        [], messages, include_errors=True, initial_message=INITIAL_MESSAGE
    )
    assert any("Falha 500" in f for f in with_errors)
