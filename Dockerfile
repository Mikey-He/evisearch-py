FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir .

CMD ["gunicorn", "evisearch.api:app", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker"]