"""LLM orchestration for the web-search chatbot."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .config import Settings
from .prompts import HUMAN_TEMPLATE, SYSTEM_PROMPT
from .utils import SearchResult, build_prompt_variables


def create_prompt() -> ChatPromptTemplate:
    """Create the chat prompt combining the system rules and the template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_TEMPLATE),
        ]
    )


def create_llm(settings: Settings, temperature: float) -> ChatAnthropic:
    """Create the Anthropic chat model from ``settings``."""
    if not settings.api_key:
        raise ValueError("A chave ANTHROPIC_API_KEY não foi encontrada no ambiente.")

    return ChatAnthropic(
        model=settings.model,
        temperature=temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.api_key,
    )


def create_chain(settings: Settings, temperature: float) -> Runnable:
    """Compose prompt → model → text parser into a single runnable."""
    return create_prompt() | create_llm(settings, temperature) | StrOutputParser()


def answer_question(chain: Runnable, question: str, results: Sequence[SearchResult]) -> str:
    """Run the chain and return the model's answer as plain text."""
    return str(chain.invoke(build_prompt_variables(question, results)))
