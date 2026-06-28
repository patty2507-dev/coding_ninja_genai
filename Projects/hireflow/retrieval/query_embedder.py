from typing import List

from openai import OpenAI

from config import settings

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_query(text: str) -> List[float]:
    """
    Embed a JD or search query into a dense vector.
    Uses same model as ingestion — text-embedding-3-small
    so query and resume vectors are in the same space.
    """
    client = _get_client()

    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )

    return response.data[0].embedding