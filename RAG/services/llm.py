"""LLM factory — the ONLY place ChatOpenAI is instantiated."""
from langchain_openai import ChatOpenAI

from config.settings import settings


def get_llm(
    temperature: float | None = None,
    streaming: bool = False,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=max_tokens or settings.max_tokens,
        streaming=streaming,
        api_key=settings.openai_api_key,
    )


def get_fallback_llm() -> ChatOpenAI:
    """Cheaper/faster fallback for non-critical extraction paths."""
    return ChatOpenAI(
        model=settings.fallback_model_name,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )


def get_extraction_llm() -> ChatOpenAI:
    """JSON-mode LLM for structured entity/relationship extraction."""
    return ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        max_tokens=settings.max_tokens,
        api_key=settings.openai_api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
