from typing import List

from openai import OpenAI

from config import settings

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate dense embeddings for a list of texts.
    Uses OpenAI text-embedding-3-small.
    Batches in groups of 100 to stay within API limits.
    Returns list of float vectors.
    """
    client = _get_client()
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]

        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_single(text: str) -> List[float]:
    """
    Embed a single text — used for JD query at retrieval time.
    """
    return embed_texts([text])[0]