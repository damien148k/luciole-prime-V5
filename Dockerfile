# Luciole Prime V4 — image applicative (services rag et watcher)
# Base CUDA 12.4 : torch >= 2.6 requis par transformers >= 4.52 (CVE-2025-32434).
# RTX A5000 (Ampere, CC 8.6) supportee par cu124.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app \
    HF_HOME=/app/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/models/huggingface \
    EASYOCR_MODULE_PATH=/app/models/easyocr \
    CONFIG_PATH=/app/config/settings.yaml \
    PROMPTS_PATH=/app/config/prompts.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app
COPY requirements.txt ./

RUN python -m pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY evaluation/ ./evaluation/
COPY setup_bge_model.py ./
RUN mkdir -p /app/config /app/data /app/backups /app/models/huggingface /app/models/easyocr

CMD ["python", "-m", "uvicorn", "src.rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
