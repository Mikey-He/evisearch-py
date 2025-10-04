FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker"]