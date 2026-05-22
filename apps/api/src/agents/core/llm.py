from langchain_openai import ChatOpenAI

from core.config import app


def make_llm():
    """Primary generation model with fallback chain."""
    params = dict(
        temperature=app.llm.temperature,
        timeout=app.llm.timeout_seconds,
        max_retries=app.llm.max_retries,
        streaming=app.llm.streaming,
    )
    primary = ChatOpenAI(model=app.models.generation, **params)
    fallback = ChatOpenAI(model=app.models.fallback, **params)
    return primary.with_fallbacks([fallback])
