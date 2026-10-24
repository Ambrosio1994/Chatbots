"""Application configuration sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-haiku-5-5"
LOCAL_MODEL_PREFIX = "ollama/"

#: Models available in the sidebar. Includes Anthropic Claude models and a
#: keyless local Ollama option so the app can run 100% without API keys.
AVAILABLE_MODELS: tuple[str, ...] = (
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "ollama/deepseek-r1:1.5b",
)

DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_RESULTS = 3
DEFAULT_RAG_TOP_K = 4
DEFAULT_SEARXNG_BASE_URL = "http://localhost:8080"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings for the chatbot."""

    api_key: str | None = None
    searxng_base_url: str = DEFAULT_SEARXNG_BASE_URL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    max_results: int = DEFAULT_MAX_RESULTS
    rag_top_k: int = DEFAULT_RAG_TOP_K
    include_reasoning_in_memory: bool = False
    include_errors_in_memory: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        include_reasoning = os.getenv("CHAT_INCLUDE_REASONING_IN_MEMORY", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        include_errors = os.getenv("CHAT_INCLUDE_ERRORS_IN_MEMORY", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        max_results_raw = (
            os.getenv("SEARXNG_MAX_RESULTS")
            or os.getenv("TAVILY_MAX_RESULTS")
            or str(DEFAULT_MAX_RESULTS)
        )
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            searxng_base_url=(os.getenv("SEARXNG_BASE_URL") or DEFAULT_SEARXNG_BASE_URL).rstrip(
                "/"
            ),
            ollama_base_url=(os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
            model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            max_results=int(max_results_raw),
            rag_top_k=int(os.getenv("CHAT_RAG_TOP_K", str(DEFAULT_RAG_TOP_K))),
            include_reasoning_in_memory=include_reasoning,
            include_errors_in_memory=include_errors,
        )

    @property
    def uses_local_model(self) -> bool:
        """Whether the selected model runs on the local Ollama server (no API key needed)."""
        return self.model.startswith(LOCAL_MODEL_PREFIX)

    @property
    def is_configured(self) -> bool:
        """Whether the search URL and model credentials are available."""
        return bool(self.searxng_base_url and (self.api_key or self.uses_local_model))
