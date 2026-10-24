"""Tests for environment-based configuration."""

from __future__ import annotations

from chat_com_url.config import DEFAULT_MODEL, DEFAULT_SEARXNG_BASE_URL, Settings


def test_is_configured_with_selector_model_or_custom_model_plus_key() -> None:
    assert Settings().is_configured
    assert not Settings(custom_model="claude-haiku-5-5", api_key=None).is_configured
    assert Settings(custom_model="claude-haiku-5-5", api_key="sk-ant-123").is_configured


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
