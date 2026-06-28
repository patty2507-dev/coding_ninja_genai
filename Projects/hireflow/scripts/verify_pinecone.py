import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinecone import Pinecone, ServerlessSpec
from config import settings

def verify_pinecone():
    pc = Pinecone(
        api_key=settings.pinecone_api_key,
    )
    existing = [idx.name for idx in pc.list_indexes()]
    print(existing)
    if settings.pinecone_index_name not in existing:
        print(f"creating index: {settings.pinecone_index_name}")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=1536,       # must match embedding model
            metric="cosine",       # cosine similarity for text
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"index created: {settings.pinecone_index_name}")
    else:
        print(f"index already exists: {settings.pinecone_index_name}")
        

if __name__ == "__main__":
    verify_pinecone()
