import os
import zipfile
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()

UPLOAD_DIR = Path(tempfile.gettempdir()) / "hireflow_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


class UploadResponse(BaseModel):
    message: str
    total_files: int
    files: List[str]
    task_ids: List[str]


def _extract_pdfs_from_zip(zip_path: Path, extract_to: Path) -> List[Path]:
    """Extract all PDFs from ZIP into extract_to folder."""
    pdf_files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.lower().endswith(".pdf") and not member.startswith("__"):
                # Flatten — ignore subfolders inside ZIP
                filename = Path(member).name
                target = extract_to / filename
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                pdf_files.append(target)
    return pdf_files


def _process_sync(pdf_paths: List[Path]):
    """Process resumes without Celery — runs in background."""
    from ingestion.pipeline import ingest_resume
    for pdf_path in pdf_paths:
        try:
            ingest_resume(str(pdf_path))
        except Exception as e:
            print(f"Failed: {pdf_path.name} — {e}")


def _dispatch_celery_tasks(pdf_paths: List[Path]) -> List[str]:
    """Dispatch one Celery task per PDF."""
    from ingestion.tasks import ingest_resume_task
    task_ids = []
    for pdf_path in pdf_paths:
        task = ingest_resume_task.delay(str(pdf_path))
        task_ids.append(task.id)
    return task_ids


@router.post("/upload/resumes", response_model=UploadResponse)
async def upload_resumes(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP file containing resume PDFs"),
    use_celery: bool = False,
):
    """
    POST /api/upload/resumes

    Upload a single ZIP file containing resume PDFs.
    System extracts all PDFs and ingests them into Pinecone.

    use_celery=False → background tasks (default, no Redis needed)
    use_celery=True  → Celery workers (requires Redis)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Save uploaded file
    session_dir = UPLOAD_DIR / f"session_{os.getpid()}"
    session_dir.mkdir(exist_ok=True)

    dest = session_dir / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # Extract PDFs
    if file.filename.lower().endswith(".zip"):
        try:
            pdf_paths = _extract_pdfs_from_zip(dest, session_dir)
            dest.unlink(missing_ok=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"ZIP extraction failed: {e}")

    elif file.filename.lower().endswith(".pdf"):
        # Single PDF also accepted
        pdf_paths = [dest]

    else:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Only ZIP or PDF files accepted."
        )

    if not pdf_paths:
        raise HTTPException(
            status_code=400,
            detail="No PDF files found inside the ZIP."
        )

    file_names = [p.name for p in pdf_paths]

    # Dispatch
    if use_celery:
        try:
            task_ids = _dispatch_celery_tasks(pdf_paths)
            return UploadResponse(
                message=f"Dispatched {len(pdf_paths)} resume(s) to Celery workers",
                total_files=len(pdf_paths),
                files=file_names,
                task_ids=task_ids,
            )
        except Exception as e:
            print(f"Celery unavailable ({e}) — falling back to background tasks")

    # Default — background tasks
    background_tasks.add_task(_process_sync, pdf_paths)

    return UploadResponse(
        message=f"Processing {len(pdf_paths)} resume(s) in background",
        total_files=len(pdf_paths),
        files=file_names,
        task_ids=[],
    )


@router.get("/upload/status/{task_id}")
async def get_task_status(task_id: str):
    """GET /api/upload/status/{task_id} — Celery task status check."""
    try:
        from ingestion.celery_app import celery_app
        task = celery_app.AsyncResult(task_id)
        response = {"task_id": task_id, "status": task.status}
        if task.status == "SUCCESS":
            response["result"] = task.result
        elif task.status == "FAILURE":
            response["error"] = str(task.info)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))