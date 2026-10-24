"""Application configuration sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_MEMORY_MODEL = "llama3.2:1b"
LOCAL_MODEL_PREFIX = "ollama/"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

#: LlamaIndex local models available in the sidebar selector (no API key required).
AVAILABLE_MODELS: tuple[str, ...] = (
    "llama3.2:1b",
    "llama3.2",
    "deepseek-r1:1.5b",
    "qwen2.5",
)

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RAG_TOP_K = 3


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings for the chatbot."""

    api_key: str | None = None
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_MODEL
    memory_model: str = DEFAULT_MEMORY_MODEL
    custom_model: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    rag_top_k: int = DEFAULT_RAG_TOP_K

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            ollama_base_url=(os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
            model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
            memory_model=os.getenv("CHAT_MEMORY_MODEL", DEFAULT_MEMORY_MODEL),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            rag_top_k=int(os.getenv("CHAT_RAG_TOP_K", str(DEFAULT_RAG_TOP_K))),
        )

    @property
    def active_model(self) -> str:
        """Return the custom model when specified, otherwise the selector model."""
        return self.custom_model.strip() or self.model

    @property
    def uses_local_model(self) -> bool:
        """Whether ``active_model`` is a local LlamaIndex/Ollama model (no API key needed)."""
        target = self.active_model
        if self.custom_model.strip():
            return target.startswith(LOCAL_MODEL_PREFIX)
        return target in AVAILABLE_MODELS or target.startswith(LOCAL_MODEL_PREFIX)

    @property
    def is_configured(self) -> bool:
        """Whether the selected model is ready to run (local model or external model + API key)."""
        if self.custom_model.strip() and not self.uses_local_model:
            return bool(self.api_key)
        return bool(self.uses_local_model or self.api_key)
