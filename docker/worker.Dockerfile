# Worker image with two targets:
#   cpu (default) - plain Python image, no CUDA, runs anywhere
#   gpu           - CUDA 12 + cuDNN 9 runtime, which faster-whisper / CTranslate2 4.x require

# ---------- CPU ----------
FROM python:3.11-slim-bookworm AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv

COPY requirements-api.txt requirements-worker.txt ./
RUN pip install --no-cache-dir -r requirements-worker.txt

COPY app ./app
COPY tests ./tests
COPY pytest.ini .

RUN useradd --create-home --uid 1000 appuser && mkdir -p /data /models && chown appuser /data /models
USER appuser

CMD ["python", "-m", "app.worker.main"]

# ---------- GPU ----------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv

COPY requirements-api.txt requirements-worker.txt ./
RUN pip3 install --no-cache-dir -r requirements-worker.txt

COPY app ./app
COPY tests ./tests
COPY pytest.ini .

RUN useradd --create-home --uid 1000 appuser && mkdir -p /data /models && chown appuser /data /models
USER appuser

CMD ["python3", "-m", "app.worker.main"]
