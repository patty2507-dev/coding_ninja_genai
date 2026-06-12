from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

_model = None


def _get_model() -> CrossEncoder:
    """
    Load reranker model once and cache it.
    BAAI/bge-reranker-base — free, local, high accuracy.
    First run will download the model (~1GB).
    """
    global _model
    if _model is None:
        print("Loading reranker model (first time may take a moment)...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Reranker model loaded.")
    return _model


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank candidates using cross-encoder model.

    Cross-encoder looks at (query, candidate_text) pair together
    — more accurate than cosine similarity alone.

    Steps:
    1. Build (query, text) pairs
    2. Score each pair
    3. Sort by score descending
    4. Return top_k with rerank_score added

    Returns reranked list of candidate dicts.
    """
    if not candidates:
        return []

    model = _get_model()

    # Build pairs for cross-encoder
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs
    scores = model.predict(pairs)

    # Attach rerank score to each candidate
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 4)

    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # Return top_k
    top_candidates = reranked[:top_k]

    print(f"Reranked {len(candidates)} → top {len(top_candidates)} candidates")
    for i, c in enumerate(top_candidates):
        print(f"  {i+1}. {c['file_name']:<35} rerank_score: {c['rerank_score']}")

    return top_candidates