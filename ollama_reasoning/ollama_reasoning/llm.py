"""LLM orchestration for the local reasoning chatbot."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from .config import Settings
from .prompts import SYSTEM_PROMPT
from .utils import extract_reasoning


def create_prompt() -> ChatPromptTemplate:
    """Create the chat prompt used for every question."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )


def create_llm(settings: Settings, temperature: float) -> ChatOllama:
    """Create a local Ollama chat model from ``settings``."""
    return ChatOllama(
        model=settings.model,
        base_url=settings.base_url,
        temperature=temperature,
    )


def build_chain(llm: Runnable) -> Runnable:
    """Compose the prompt with the local model."""
    return create_prompt() | llm


def generate_answer(chain: Runnable, question: str) -> tuple[str, str]:
    """Run ``chain`` for ``question`` and return ``(reasoning, answer)``."""
    message = chain.invoke({"input": question})
    reasoning_content = message.additional_kwargs.get("reasoning_content")
    return extract_reasoning(str(message.content), reasoning_content)
