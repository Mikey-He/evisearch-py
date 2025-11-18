from __future__ import annotations

import os  # <--- 1. 添加 import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

BROKER_URL = f"{REDIS_URL}/0"
BACKEND_URL = f"{REDIS_URL}/1"

celery_app = Celery(
    "evisearch",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["evisearch.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)