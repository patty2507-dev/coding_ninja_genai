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


def upsert_chunks(
    candidate_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata_list: List[Dict[str, Any]],
) -> int:
    """
    Upsert resume chunks into Pinecone.
    Vector ID format: {candidate_id}_{chunk_index}
    Each vector carries full metadata for filtering + display.
    Returns count of upserted vectors.
    """
    index = _get_index()

    vectors = []
    for i, (chunk, embedding, meta) in enumerate(
        zip(chunks, embeddings, metadata_list)
    ):
        vector_id = f"{candidate_id}_{i}"

        # Keep metadata clean — Pinecone has 40KB limit per vector
        payload = {
            "candidate_id": candidate_id,
            "chunk_index": i,
            "text": chunk[:1000],  # store first 1000 chars for display
            "file_name": meta.get("file_name", ""),
            "chunk_total": len(chunks),
        }

        # Add any extra metadata passed in
        for key in ["page_label", "total_pages"]:
            if key in meta:
                payload[key] = meta[key]

        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": payload,
        })

    # Upsert in batches of 100
    batch_size = 100
    total_upserted = 0

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i: i + batch_size]
        index.upsert(vectors=batch)
        total_upserted += len(batch)

    print(f"  Upserted {total_upserted} vector(s) for candidate: {candidate_id}")
    return total_upserted