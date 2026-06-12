import os
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def load_resumes(resume_dir: str) -> List[Document]:
    """
    Load all PDF files from a directory.
    Returns a list of LlamaIndex Document objects.
    Each document has metadata: file_name, file_path
    """
    resume_path = Path(resume_dir)

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    pdf_files = list(resume_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {resume_dir}")

    print(f"Found {len(pdf_files)} PDF(s) in {resume_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(resume_path),
        required_exts=[".pdf"],
        recursive=False,
    )

    documents = reader.load_data()

    print(f"Loaded {len(documents)} document(s)")
    return documents


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