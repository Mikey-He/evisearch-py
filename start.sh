#!/bin/bash

echo "Starting Gunicorn API server in background..."
gunicorn evisearch.api:app --bind 0.0.0.0:10000 --timeout 300 --workers 1 --worker-class uvicorn.workers.UvicornWorker &

echo "Starting Celery worker in foreground..."
celery -A evisearch.celery_app worker --loglevel=info -P gevent
