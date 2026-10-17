"""Tests for environment-based configuration."""

from __future__ import annotations

from chat_com_url.config import DEFAULT_MODEL, Settings


def test_is_configured_requires_both_keys() -> None:
    assert not Settings(api_key="a").is_configured
    assert not Settings(tavily_api_key="t").is_configured
    assert Settings(api_key="a", tavily_api_key="t").is_configured


def test_from_env_reads_specific_variables(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily")
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.is_configured
    assert settings.model == DEFAULT_MODEL
    assert settings.max_results == 5
