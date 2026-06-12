from typing import Optional
from pathlib import Path

from ingestion.loader import load_single_resume
from ingestion.chunker import chunk_documents
from ingestion.metadata import build_candidate_id
from ingestion.dedup import handle_deduplication
from ingestion.embedder import embed_texts
from ingestion.vector_store import upsert_chunks


def ingest_resume(file_path: str) -> dict:
    """
    Full ingestion pipeline for a single resume PDF.

    Steps:
    1. Load PDF → LlamaIndex Documents
    2. Chunk documents → TextNodes
    3. Extract email → candidate_id
    4. Deduplication → delete old vectors if exists
    5. Embed chunks → dense vectors
    6. Upsert to Pinecone

    Returns summary dict with candidate_id and chunk count.
    """
    file_name = Path(file_path).name
    print(f"\nProcessing: {file_name}")
    print("-" * 40)

    # Step 1 — Load PDF
    documents = load_single_resume(file_path)

    # Step 2 — Chunk
    nodes = chunk_documents(documents)

    # Step 3 — Extract candidate_id from full text
    full_text = " ".join([doc.text for doc in documents])
    candidate_id = build_candidate_id(full_text, file_name)

    # Step 4 — Deduplication
    was_existing = handle_deduplication(candidate_id)

    # Step 5 — Embed all chunks
    chunk_texts = [node.get_content() for node in nodes]
    embeddings = embed_texts(chunk_texts)

    # Step 6 — Build metadata list + upsert
    metadata_list = [node.metadata for node in nodes]
    upsert_chunks(
        candidate_id=candidate_id,
        chunks=chunk_texts,
        embeddings=embeddings,
        metadata_list=metadata_list,
    )

    result = {
        "file_name": file_name,
        "candidate_id": candidate_id,
        "chunks": len(nodes),
        "was_existing": was_existing,
        "status": "success",
    }

    print(f"Done: {file_name} → {len(nodes)} chunks | id: {candidate_id}")
    return result