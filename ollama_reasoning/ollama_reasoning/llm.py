"""LLM orchestration for the local reasoning chatbot and its memory evaluator."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from ollama import Client

from .config import Settings
from .prompts import MEMORY_EVALUATOR_PROMPT, SYSTEM_PROMPT
from .utils import (
    ReasoningStreamSplitter,
    extract_reasoning,
    extract_user_facts,
    merge_memory_facts,
    parse_memory_evaluation,
)

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


def create_memory_prompt() -> ChatPromptTemplate:
    """Create the prompt for the dedicated memory-evaluator model."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", MEMORY_EVALUATOR_PROMPT),
            ("human", "Mensagem do usuário:\n{user_message}"),
        ]
    )


def to_langchain_history(
    history: HistoryMessages | None,
    memory_facts: Sequence[str] = (),
) -> list[BaseMessage]:
    """Convert plain ``{"role", "content"}`` mappings and curated facts into LangChain messages."""
    messages: list[BaseMessage] = []
    if memory_facts:
        facts_block = "\n".join(f"- {fact}" for fact in memory_facts if fact.strip())
        if facts_block:
            messages.append(
                SystemMessage(
                    content=(
                        "Memória da conversa (fatos, preferências e restrições do usuário):\n"
                        f"{facts_block}"
                    )
                )
            )
    if not history:
        return messages
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
    """Ask the Ollama server whether ``settings.model`` can expose reasoning."""
    try:
        info = Client(host=settings.base_url).show(settings.model)
    except Exception:  # noqa: BLE001 - capability probing must never break the app
        return False
    return THINKING_CAPABILITY in (info.capabilities or [])


def create_llm(settings: Settings, temperature: float, reasoning: bool | None = None) -> ChatOllama:
    """Create a local Ollama chat model from ``settings``."""
    return ChatOllama(
        model=settings.model,
        base_url=settings.base_url,
        temperature=temperature,
        reasoning=reasoning,
    )


def create_memory_llm(settings: Settings) -> ChatOllama:
    """Create the dedicated, separate LLM that evaluates and writes to session memory."""
    return ChatOllama(
        model=settings.memory_model,
        base_url=settings.base_url,
        temperature=0.0,
    )


def evaluate_and_write_memory(
    settings: Settings,
    user_query: str,
    existing_facts: Sequence[str],
    memory_llm: Runnable | None = None,
) -> list[str]:
    """Use a separate model to evaluate whether ``user_query`` should be written to memory.

     Only durable user facts, preferences and constraints are persisted. If the
    message contains none, nothing is invoked or written.
    """
    candidates = extract_user_facts(user_query)
    if not candidates:
        return list(existing_facts)

    try:
        evaluator = memory_llm if memory_llm is not None else create_memory_llm(settings)
        chain = create_memory_prompt() | evaluator
        response = chain.invoke({"user_message": user_query})
        raw_text = str(getattr(response, "content", response) or "")
        extracted = parse_memory_evaluation(raw_text, user_prompt=user_query)
    except Exception:  # noqa: BLE001 - fallback to deterministic extraction on network error
        extracted = candidates

    return merge_memory_facts(existing_facts, extracted)


def build_chain(llm: Runnable) -> Runnable:
    """Compose the prompt with the local model."""
    return create_prompt() | llm


def generate_answer(
    chain: Runnable,
    question: str,
    history: HistoryMessages | None = None,
    memory_facts: Sequence[str] = (),
) -> tuple[str, str]:
    """Run ``chain`` for ``question`` and return ``(reasoning, answer)``."""
    payload: dict[str, object] = {"input": question}
    if history is not None or memory_facts:
        payload["history"] = to_langchain_history(history, memory_facts)
    message = chain.invoke(payload)
    reasoning_content = message.additional_kwargs.get("reasoning_content")
    return extract_reasoning(str(message.content), reasoning_content)


def stream_answer(
    chain: Runnable,
    question: str,
    history: HistoryMessages | None = None,
    memory_facts: Sequence[str] = (),
) -> Iterator[tuple[str, str]]:
    """Stream ``chain`` for ``question`` as ``(kind, text)`` events."""
    payload: dict[str, object] = {"input": question}
    if history is not None or memory_facts:
        payload["history"] = to_langchain_history(history, memory_facts)
    splitter = ReasoningStreamSplitter()
    for chunk in chain.stream(payload):
        additional = getattr(chunk, "additional_kwargs", None) or {}
        yield from splitter.feed(str(chunk.content or ""), additional.get("reasoning_content"))
    yield from splitter.flush()
