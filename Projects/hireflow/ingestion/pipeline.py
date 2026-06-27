from pathlib import Path
from ingestion.loader import load_data
from ingestion.chunker import chunk_documents

def ingest_resume(file_path: str) -> dict:
    """
      Steps:
    1. Load PDF → LlamaIndex Documents
    2. Chunk documents → TextNodes
    3. Extract email → candidate_id
    4. Deduplication → delete old vectors if exists
    5. Embed chunks → dense vectors
    6. Upsert to Pinecone
    
    """




    # step 1: load data
    documents = load_data(file_path)
    # step 2: split into chunks
    nodes = chunk_documents(documents)
    # step 3 embeddings

    # step 4 : store in to the vector stores
    
    return nodes