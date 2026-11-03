"""Only features both Redis and SQS support, so the broker can be swapped through the environment alone: no result
backend, no remote control, no events, no ETA tasks."""

from celery import Celery

from segmentation.config import get_settings
from segmentation.contract import JOBS_QUEUE

settings = get_settings()

celery_app = Celery("ml-worker", broker=settings.CELERY_BROKER_URL, include=["segmentation.tasks"])
celery_app.conf.update(
    broker_transport_options=settings.CELERY_BROKER_TRANSPORT_OPTIONS,
    broker_connection_retry_on_startup=True,
    task_default_queue=JOBS_QUEUE,
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    # One GPU, one job: take the next message only when free, and let it be redelivered if the worker dies mid-job.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_enable_remote_control=False,
    worker_send_task_events=False,
    task_send_sent_event=False,
    enable_utc=True,
)
