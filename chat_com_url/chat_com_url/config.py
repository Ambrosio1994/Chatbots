"""Application configuration sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_MEMORY_MODEL = "llama3.2:1b"
LOCAL_MODEL_PREFIX = "ollama/"

#: LlamaIndex local models available in the sidebar selector (no API key required).
AVAILABLE_MODELS: tuple[str, ...] = (
    "llama3.2:1b",
    "llama3.2",
    "deepseek-r1:1.5b",
    "qwen2.5",
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
    memory_model: str = DEFAULT_MEMORY_MODEL
    custom_model: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    max_results: int = DEFAULT_MAX_RESULTS
    rag_top_k: int = DEFAULT_RAG_TOP_K

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
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
            memory_model=os.getenv("CHAT_MEMORY_MODEL", DEFAULT_MEMORY_MODEL),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            max_results=int(max_results_raw),
            rag_top_k=int(os.getenv("CHAT_RAG_TOP_K", str(DEFAULT_RAG_TOP_K))),
        )

    @property
    def active_model(self) -> str:
        """Return the custom model when specified, otherwise the selector model."""
        return self.custom_model.strip() or self.model

    @property
    def uses_local_model(self) -> bool:
        """Whether ``active_model`` runs on the local server (no API key needed)."""
        target = self.active_model
        if self.custom_model.strip():
            return target.startswith(LOCAL_MODEL_PREFIX)
        return target in AVAILABLE_MODELS or target.startswith(LOCAL_MODEL_PREFIX)

    @property
    def is_configured(self) -> bool:
        """Whether the search URL and model credentials are available."""
        if not self.searxng_base_url:
            return False
        if self.custom_model.strip() and not self.uses_local_model:
            return bool(self.api_key)
        return bool(self.uses_local_model or self.api_key)
