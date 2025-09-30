FROM python:3.11-slim

# 安装 Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目
COPY . .

# 启动命令
CMD gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT evisearch.api:app