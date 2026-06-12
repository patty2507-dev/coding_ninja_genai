from celery import Celery
from config import settings

celery_app = Celery(
    "hireflow",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["ingestion.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Retry failed tasks up to 3 times
    task_max_retries=3,
    task_acks_late=True,
    # Result expires after 1 hour
    result_expires=3600,
)