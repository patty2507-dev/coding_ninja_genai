from langchain_openai import ChatOpenAI

from config import settings

_llm = None


def get_llm() -> ChatOpenAI:
    """
    Returns a cached ChatOpenAI instance.
    streaming=True  — enables token-by-token SSE output
    temperature=0   — deterministic structured output
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
            streaming=True,
        )
    return _llm