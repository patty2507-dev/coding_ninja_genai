from celery import Task
from ingestion.celery_app import celery_app
from ingestion.pipeline import ingest_resume


class IngestionTask(Task):
    """
    Base task class with error handling.
    """
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print(f"Task {task_id} failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        print(f"Task {task_id} retrying: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        print(f"Task {task_id} succeeded: {retval.get('file_name')}")


@celery_app.task(
    bind=True,
    base=IngestionTask,
    max_retries=3,
    default_retry_delay=5,
    name="ingestion.tasks.ingest_resume_task",
)
def ingest_resume_task(self, file_path: str) -> dict:
    """
    Celery task — process a single resume PDF.
    One task per file — runs in a worker process.

    Args:
        file_path: absolute path to the PDF file

    Returns:
        result dict from ingest_resume()
    """
    try:
        result = ingest_resume(file_path)
        return result
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=5 * (self.request.retries + 1))