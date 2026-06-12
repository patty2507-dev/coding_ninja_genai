from typing import List, Dict, Any

from pinecone import Pinecone

from config import settings

_index = None


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _index = pc.Index(settings.pinecone_index_name)
    return _index


def retrieve_candidates(
    query_embedding: List[float],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone with a dense embedding vector.
    Returns top_k most similar resume chunks.

    We fetch top_k=20 here (more than needed)
    so the reranker has enough candidates to work with.
    Final top-k is applied after reranking.

    Returns list of dicts:
        {
            id, score, candidate_id,
            text, file_name, chunk_index
        }
    """
    index = _get_index()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    candidates = []
    seen_candidate_ids = set()

    for match in results.matches:
        meta = match.metadata or {}
        candidate_id = meta.get("candidate_id", match.id)

        # Deduplicate — keep only best chunk per candidate
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)

        candidates.append({
            "id": match.id,
            "score": round(match.score, 4),
            "candidate_id": candidate_id,
            "text": meta.get("text", ""),
            "file_name": meta.get("file_name", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_total": meta.get("chunk_total", 0),
        })

    print(f"Retrieved {len(candidates)} unique candidate(s) from Pinecone")
    return candidates