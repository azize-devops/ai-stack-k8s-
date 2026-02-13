"""
RAG Pipeline Configuration

Centralized configuration management with environment variable overrides.
All settings are loaded from environment variables with sensible defaults
for Kubernetes in-cluster communication.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (local development)
load_dotenv()


class EmbeddingStrategy(str, Enum):
    """Supported embedding strategies for vector search."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class LogLevel(str, Enum):
    """Application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def _env(key: str, default: str) -> str:
    """Read an environment variable with a default fallback."""
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    """Read an integer environment variable with a default fallback."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings loaded once at startup.

    Attributes:
        localai_url:          Base URL for the LocalAI OpenAI-compatible API.
        ollama_url:           Base URL for the Ollama LLM server.
        qdrant_url:           Base URL for the Qdrant vector database.
        embedding_model:      Model identifier used for generating embeddings.
        llm_model:            Model identifier used for LLM text generation.
        chunk_size:           Maximum number of characters per document chunk.
        chunk_overlap:        Number of overlapping characters between chunks.
        collection_name:      Default Qdrant collection for document storage.
        embedding_strategy:   Vector embedding strategy (dense, sparse, hybrid).
        embedding_dimension:  Dimensionality of the dense embedding vectors.
        top_k:                Default number of candidates for initial retrieval.
        top_n:                Default number of results after re-ranking.
        log_level:            Application-wide logging level.
        document_storage_path: Filesystem path for persistent document storage.
    """

    # --- Service URLs ---
    localai_url: str = field(
        default_factory=lambda: _env(
            "LOCALAI_URL",
            "http://localai.ai-stack.svc.cluster.local:8080/v1",
        )
    )
    ollama_url: str = field(
        default_factory=lambda: _env(
            "OLLAMA_URL",
            "http://ollama.ai-stack.svc.cluster.local:11434",
        )
    )
    qdrant_url: str = field(
        default_factory=lambda: _env(
            "QDRANT_URL",
            "http://qdrant.ai-stack.svc.cluster.local:6333",
        )
    )

    # --- Model configuration ---
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-ada-002")
    )
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "gpt-3.5-turbo")
    )

    # --- Chunking parameters ---
    chunk_size: int = field(
        default_factory=lambda: _env_int("CHUNK_SIZE", 512)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _env_int("CHUNK_OVERLAP", 50)
    )

    # --- Qdrant settings ---
    collection_name: str = field(
        default_factory=lambda: _env("COLLECTION_NAME", "knowledge_base")
    )

    # --- Embedding strategy ---
    embedding_strategy: EmbeddingStrategy = field(default=EmbeddingStrategy.DENSE)
    embedding_dimension: int = field(
        default_factory=lambda: _env_int("EMBEDDING_DIMENSION", 1536)
    )

    # --- Retrieval defaults ---
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 10))
    top_n: int = field(default_factory=lambda: _env_int("TOP_N", 3))

    # --- Logging ---
    log_level: LogLevel = field(default=LogLevel.INFO)

    # --- Storage ---
    document_storage_path: Path = field(
        default_factory=lambda: Path(
            _env("DOCUMENT_STORAGE_PATH", "/data/documents")
        )
    )

    def __post_init__(self) -> None:
        """Resolve enum fields from environment variables after init."""
        # Dataclass is frozen, so we use object.__setattr__ for post-init overrides
        strategy_raw = os.getenv("EMBEDDING_STRATEGY", self.embedding_strategy.value)
        try:
            strategy = EmbeddingStrategy(strategy_raw.lower())
        except ValueError:
            strategy = EmbeddingStrategy.DENSE
        object.__setattr__(self, "embedding_strategy", strategy)

        log_raw = os.getenv("LOG_LEVEL", self.log_level.value)
        try:
            level = LogLevel(log_raw.upper())
        except ValueError:
            level = LogLevel.INFO
        object.__setattr__(self, "log_level", level)

    @property
    def qdrant_host(self) -> str:
        """Extract the hostname from the Qdrant URL."""
        from urllib.parse import urlparse

        parsed = urlparse(self.qdrant_url)
        return parsed.hostname or "localhost"

    @property
    def qdrant_port(self) -> int:
        """Extract the port from the Qdrant URL."""
        from urllib.parse import urlparse

        parsed = urlparse(self.qdrant_url)
        return parsed.port or 6333


# Singleton instance -- import this across the application
settings = Settings()
