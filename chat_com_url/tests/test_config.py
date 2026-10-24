"""Tests for environment-based configuration."""

from __future__ import annotations

from chat_com_url.config import DEFAULT_MODEL, DEFAULT_SEARXNG_BASE_URL, Settings


def test_is_configured_with_api_key_or_local_model() -> None:
    assert not Settings(api_key=None).is_configured
    assert Settings(api_key="sk-ant-123").is_configured
    assert Settings(api_key=None, model="ollama/deepseek-r1:1.5b").is_configured


def test_from_env_reads_specific_variables(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080/")
    monkeypatch.setenv("SEARXNG_MAX_RESULTS", "5")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.is_configured
    assert settings.model == DEFAULT_MODEL
    assert settings.searxng_base_url == DEFAULT_SEARXNG_BASE_URL
    assert settings.max_results == 5
