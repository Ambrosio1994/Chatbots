"""Tests for environment-based configuration."""

from __future__ import annotations

from chatbasico.config import DEFAULT_MODEL, Settings


def test_from_env_uses_defaults_when_unset(monkeypatch) -> None:
    for variable in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_MAX_TOKENS",
        "ANTHROPIC_TEMPERATURE",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings.from_env()

    assert settings.api_key is None
    assert settings.model == DEFAULT_MODEL
    assert settings.max_tokens == 1024


def test_from_env_reads_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "256")
    monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.7")

    settings = Settings.from_env()

    assert settings.api_key == "secret"
    assert settings.model == "claude-sonnet-5-5"
    assert settings.max_tokens == 256
    assert settings.temperature == 0.7
