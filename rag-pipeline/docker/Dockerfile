FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies in a separate layer for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY config.py server.py ./

EXPOSE 8000

# Run with uvicorn -- single worker is fine for CPU-bound embedding relay
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
