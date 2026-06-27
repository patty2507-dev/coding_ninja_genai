from pathlib import Path
from ingestion.loader import load_data,load_single_resume
from ingestion.chunker import chunk_documents
from ingestion.metadata import build_candidate_id
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

    file_name = Path(file_path).name
    print(f"\nProcessing: {file_name}")
    # step 1: load data
    documents = load_single_resume(file_path)
    # step 2: split into chunks
    nodes = chunk_documents(documents)

    # step 3 : Extract candidate id from full text
    full_text = " ".join([doc.text for doc in documents])
    candidate_id = build_candidate_id(full_text,file_name)
    print(f"  candidate_id: {candidate_id}")
    
    return nodes

def dummy():
    return 'working'