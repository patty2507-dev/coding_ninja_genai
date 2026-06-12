from typing import List, Dict, Any

from retrieval.query_embedder import embed_query
from retrieval.retriever import retrieve_candidates
from retrieval.reranker import rerank_candidates
from config import settings


def retrieve(
    jd_text: str,
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
        JD text
            → embed (OpenAI text-embedding-3-small)
            → Pinecone vector search (top 20)
            → cross-encoder rerank (BAAI/bge-reranker-base)
            → top_k candidates

    Returns list of ranked candidate dicts:
        {
            id, score, rerank_score,
            candidate_id, text,
            file_name, chunk_index
        }
    """
    if top_k is None:
        top_k = settings.top_k

    print(f"\nRunning retrieval pipeline (top_k={top_k})...")
    print("-" * 40)

    # Step 1 — Embed JD
    print("Step 1: Embedding JD...")
    query_vector = embed_query(jd_text)

    # Step 2 — Pinecone search (fetch 20 for reranker headroom)
    print("Step 2: Searching Pinecone...")
    candidates = retrieve_candidates(
        query_embedding=query_vector,
        top_k=20,
    )

    if not candidates:
        print("No candidates found in Pinecone.")
        return []

    # Step 3 — Rerank
    print("Step 3: Reranking candidates...")
    top_candidates = rerank_candidates(
        query=jd_text,
        candidates=candidates,
        top_k=top_k,
    )

    print(f"\nRetrieval complete — {len(top_candidates)} candidate(s) returned")
    return top_candidates