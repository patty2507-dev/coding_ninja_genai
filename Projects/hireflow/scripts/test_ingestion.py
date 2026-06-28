from ingestion.pipeline import ingest_resume
# python scripts/test_ingestion.py resume/john_doe.pdf
# Load → Chunk → Email extract → Dedup → Embed → Upsert to Pinecone


if __name__ == "__main__":

    file_path =  "resume/Andrew_Green_Resume_27.pdf"
    nodes = ingest_resume(file_path)
    print(f"Created {len(nodes)} chunk(s) from {file_path}")