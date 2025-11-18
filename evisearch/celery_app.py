from __future__ import annotations

from celery import Celery

# configure Celery instance 
# /0 and /1 are Redis database numbers
BROKER_URL = "redis://localhost:6379/0"
BACKEND_URL = "redis://localhost:6379/1"

celery_app = Celery(
    "evisearch",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    # Auto-discover tasks in the 'evisearch.tasks' module
    include=["evisearch.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)