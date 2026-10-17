"""Application configuration sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-haiku-5-5"

#: Models available in the sidebar. Keep this list in sync with the Anthropic
#: model lineup: https://platform.claude.com/docs/en/about-claude/models/overview
AVAILABLE_MODELS: tuple[str, ...] = (
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
)

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings for the chatbot."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
        )
