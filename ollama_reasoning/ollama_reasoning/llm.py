"""LLM orchestration for the local reasoning chatbot."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from ollama import Client

from .config import Settings
from .prompts import SYSTEM_PROMPT
from .utils import ReasoningStreamSplitter, extract_reasoning

THINKING_CAPABILITY = "thinking"
HistoryMessages = Sequence[Mapping[str, str]]


def create_prompt() -> ChatPromptTemplate:
    """Create the chat prompt used for every question, including conversation history."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}"),
        ]
    )


def to_langchain_history(history: HistoryMessages | None) -> list[BaseMessage]:
    """Convert plain ``{"role", "content"}`` mappings into LangChain message objects."""
    if not history:
        return []
    messages: list[BaseMessage] = []
    for item in history:
        role = item.get("role")
        content = str(item.get("content") or "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def supports_thinking(settings: Settings) -> bool:
    """Ask the Ollama server whether ``settings.model`` can expose reasoning.

    Recent Ollama versions return the reasoning in a separate field, which
    ``ChatOllama`` silently drops unless ``reasoning=True``. Enabling it for a
    model without that capability, however, makes the request fail, so we
    check first. Any lookup error falls back to ``False`` (the request itself
    will then surface a meaningful error to the user).
    """
    try:
        info = Client(host=settings.base_url).show(settings.model)
    except Exception:  # noqa: BLE001 - capability probing must never break the app
        return False
    return THINKING_CAPABILITY in (info.capabilities or [])


def create_llm(settings: Settings, temperature: float, reasoning: bool | None = None) -> ChatOllama:
    """Create a local Ollama chat model from ``settings``.

    ``reasoning=True`` asks Ollama to stream the chain of thought separately
    (in ``reasoning_content``); ``None`` keeps the server default.
    """
    return ChatOllama(
        model=settings.model,
        base_url=settings.base_url,
        temperature=temperature,
        reasoning=reasoning,
    )


def build_chain(llm: Runnable) -> Runnable:
    """Compose the prompt with the local model."""
    return create_prompt() | llm


def generate_answer(
    chain: Runnable,
    question: str,
    history: HistoryMessages | None = None,
) -> tuple[str, str]:
    """Run ``chain`` for ``question`` and return ``(reasoning, answer)``."""
    payload: dict[str, object] = {"input": question}
    if history is not None:
        payload["history"] = to_langchain_history(history)
    message = chain.invoke(payload)
    reasoning_content = message.additional_kwargs.get("reasoning_content")
    return extract_reasoning(str(message.content), reasoning_content)


def stream_answer(
    chain: Runnable,
    question: str,
    history: HistoryMessages | None = None,
) -> Iterator[tuple[str, str]]:
    """Stream ``chain`` for ``question`` as ``(kind, text)`` events.

    ``kind`` is ``"reasoning"`` or ``"answer"``, so the UI can render the chain
    of thought and the final answer in separate areas while they are generated.
    """
    payload: dict[str, object] = {"input": question}
    if history is not None:
        payload["history"] = to_langchain_history(history)
    splitter = ReasoningStreamSplitter()
    for chunk in chain.stream(payload):
        additional = getattr(chunk, "additional_kwargs", None) or {}
        yield from splitter.feed(str(chunk.content or ""), additional.get("reasoning_content"))
    yield from splitter.flush()
