from ingestion.pipeline import ingest_resume

# python scripts/test_ingestion.py resume/john_doe.pdf


if __name__ == "__main__":

    file_path =  "resume/john_doe.pdf"
    nodes = ingest_resume(file_path)
    print(f"Created {len(nodes)} chunk(s) from {file_path}")