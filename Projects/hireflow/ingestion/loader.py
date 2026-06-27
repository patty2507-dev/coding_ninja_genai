import os
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def load_data(file_path: str) -> List[Document]:
    resume_path = Path(file_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")
    pdf_files = list(resume_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {resume_path}")
    reader = SimpleDirectoryReader(
        input_dir = str(resume_path),
        required_exts = [".pdf"],
    )

    document = reader.load_data()
    return document



def load_single_resume(file_path: str) -> List[Document]:
    """
    Load a single PDF resume file.
    Used by Celery worker — one task per file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    reader = SimpleDirectoryReader(
        input_files=[str(path)],
    )

    documents = reader.load_data()
    print(f"Loaded: {path.name} → {len(documents)} page(s)")
    return documents