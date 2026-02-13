# =============================================================================
# Multi-stage Dockerfile for the AI Agents module
# =============================================================================
# Stage 1: Install dependencies in a builder image
# Stage 2: Copy only what is needed into a slim runtime image
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 -- Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System dependencies required by some Python packages during install.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 -- Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="ai-stack-k8s"
LABEL description="AI Agent Workflows -- CrewAI & LangGraph"

# Non-root user for security.
RUN groupadd --gid 1000 agents && \
    useradd --uid 1000 --gid agents --create-home agents

WORKDIR /app

# Copy installed Python packages from the builder stage.
COPY --from=builder /install /usr/local

# Copy application source code.
COPY --chown=agents:agents ../crewai-research-agent/ /app/crewai-research-agent/
COPY --chown=agents:agents ../langgraph-workflow/     /app/langgraph-workflow/

# Create output directory for CrewAI results.
RUN mkdir -p /app/crewai-research-agent/output && \
    chown -R agents:agents /app

USER agents

# Default environment variables (override at runtime).
ENV LLM_BASE_URL="http://localai:8080/v1" \
    LLM_API_KEY="sk-no-key-required" \
    LLM_MODEL_NAME="gpt-3.5-turbo" \
    QDRANT_URL="http://qdrant:6333" \
    QDRANT_COLLECTION="documents" \
    LOG_LEVEL="INFO" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"

# Healthcheck -- verify Python can import the key packages.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import crewai; import langgraph; print('ok')" || exit 1

# Default command runs the LangGraph workflow demo.
CMD ["python", "/app/langgraph-workflow/main.py"]
