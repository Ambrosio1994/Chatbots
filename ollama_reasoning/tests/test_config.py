"""Tests for environment-based configuration."""

from __future__ import annotations

from ollama_reasoning.config import DEFAULT_BASE_URL, DEFAULT_MODEL, Settings


def test_from_env_uses_defaults_when_unset(monkeypatch) -> None:
    for variable in ("OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_TEMPERATURE"):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings.from_env()

    assert settings.model == DEFAULT_MODEL
    assert settings.base_url == DEFAULT_BASE_URL


def test_from_env_reads_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.2")

    settings = Settings.from_env()

    assert settings.model == "llama3.2"
    assert settings.base_url == "http://ollama:11434"
    assert settings.temperature == 0.2
