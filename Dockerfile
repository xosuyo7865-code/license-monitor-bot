# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (build essentials minimal)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cache friendly)
COPY requirements_web.txt /app/requirements_web.txt
RUN pip install --upgrade pip && pip install -r requirements_web.txt

# Copy app code
COPY app.py /app/app.py

# Non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Gunicorn with Uvicorn worker
ENV PORT=8000
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "90", "--graceful-timeout", "30"]
