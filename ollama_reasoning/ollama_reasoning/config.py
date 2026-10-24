"""Application configuration sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "deepseek-r1:1.5b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_HISTORY_TURNS = 5

#: Suggestions shown in the UI. Any model available in the local Ollama
#: instance can be used (see ``ollama list``).
SUGGESTED_MODELS: tuple[str, ...] = (
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "deepseek-r1:8b",
    "llama3.2",
    "qwen2.5",
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings for the chatbot."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = DEFAULT_TEMPERATURE
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS
    include_reasoning_in_memory: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        include_reasoning = os.getenv("OLLAMA_INCLUDE_REASONING_IN_MEMORY", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            max_history_turns=int(
                os.getenv("OLLAMA_MAX_HISTORY_TURNS", str(DEFAULT_MAX_HISTORY_TURNS))
            ),
            include_reasoning_in_memory=include_reasoning,
        )
