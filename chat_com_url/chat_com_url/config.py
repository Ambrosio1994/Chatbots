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

DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_RESULTS = 3


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings for the chatbot."""

    api_key: str | None = None
    tavily_api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    max_results: int = DEFAULT_MAX_RESULTS

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            max_results=int(os.getenv("TAVILY_MAX_RESULTS", str(DEFAULT_MAX_RESULTS))),
        )

    @property
    def is_configured(self) -> bool:
        """Whether both API keys are available."""
        return bool(self.api_key and self.tavily_api_key)
