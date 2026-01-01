"""
Celery Application
Background task processing
"""

from celery import Celery

from backend.core.config import settings

# Create Celery app
celery_app = Celery(
    "ocr_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing"""
    print(f"Request: {self.request!r}")


# Import tasks
from backend.tasks import document_tasks  # noqa
