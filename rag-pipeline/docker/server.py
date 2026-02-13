"""
RAG Pipeline Server

A production-grade FastAPI application implementing a multi-stage
Retrieval-Augmented Generation pipeline:

    Ingest:  Document -> Chunk -> Embed -> Store (Qdrant)
    Query:   Question -> Embed -> Retrieve -> Re-rank -> Generate (LLM)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import openai
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from config import EmbeddingStrategy, settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=settings.log_level.value,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rag-pipeline")


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Payload for the document ingestion endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw document text to ingest.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to every chunk.",
    )
    collection_name: str | None = Field(
        default=None,
        description="Target Qdrant collection. Falls back to the configured default.",
    )
    chunk_size: int | None = Field(
        default=None,
        ge=64,
        le=8192,
        description="Override the default chunk size for this request.",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        le=512,
        description="Override the default chunk overlap for this request.",
    )


class IngestResponse(BaseModel):
    """Response returned after successful ingestion."""

    collection_name: str
    chunks_created: int
    document_id: str
    latency_ms: dict[str, float]


class QueryRequest(BaseModel):
    """Payload for the RAG query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question.",
    )
    collection_name: str | None = Field(
        default=None,
        description="Qdrant collection to search.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of candidates for initial vector retrieval.",
    )
    top_n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of results to keep after re-ranking.",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source chunks in the response.",
    )


class SourceChunk(BaseModel):
    """A single retrieved chunk returned alongside the generated answer."""

    text: str
    score: float
    rerank_score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """Response from the RAG query endpoint."""

    answer: str
    sources: list[SourceChunk]
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    """Health-check response with dependency statuses."""

    status: str
    version: str
    dependencies: dict[str, str]


class CollectionInfo(BaseModel):
    """Summary information about a single Qdrant collection."""

    name: str
    vectors_count: int
    status: str


class CollectionsResponse(BaseModel):
    """Response listing all Qdrant collections."""

    collections: list[CollectionInfo]


class DeleteCollectionResponse(BaseModel):
    """Response after deleting a Qdrant collection."""

    deleted: bool
    collection_name: str


# ---------------------------------------------------------------------------
# Shared clients (initialized in lifespan)
# ---------------------------------------------------------------------------

_openai_client: openai.OpenAI | None = None
_qdrant_client: QdrantClient | None = None

APP_VERSION = "1.0.0"


def get_openai_client() -> openai.OpenAI:
    """Return the shared OpenAI client, raising if uninitialized."""
    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized")
    return _openai_client


def get_qdrant_client() -> QdrantClient:
    """Return the shared Qdrant client, raising if uninitialized."""
    if _qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")
    return _qdrant_client


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and tear down shared resources."""
    global _openai_client, _qdrant_client  # noqa: PLW0603

    logger.info("Initializing RAG pipeline server v%s", APP_VERSION)
    logger.info("LocalAI URL:        %s", settings.localai_url)
    logger.info("Qdrant URL:         %s", settings.qdrant_url)
    logger.info("Embedding model:    %s", settings.embedding_model)
    logger.info("LLM model:          %s", settings.llm_model)
    logger.info("Embedding strategy: %s", settings.embedding_strategy.value)
    logger.info("Chunk size:         %d", settings.chunk_size)
    logger.info("Chunk overlap:      %d", settings.chunk_overlap)

    # OpenAI-compatible client pointed at LocalAI
    _openai_client = openai.OpenAI(
        base_url=settings.localai_url,
        api_key="not-needed",  # LocalAI does not require an API key
    )

    # Qdrant client
    qdrant_kwargs: dict[str, Any] = {
        "host": settings.qdrant_host,
        "port": settings.qdrant_port,
        "timeout": 30,
    }
    if settings.qdrant_api_key:
        qdrant_kwargs["api_key"] = settings.qdrant_api_key

    _qdrant_client = QdrantClient(**qdrant_kwargs)

    logger.info("RAG pipeline server ready")
    yield

    # Cleanup
    if _qdrant_client is not None:
        _qdrant_client.close()
    logger.info("RAG pipeline server shut down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Pipeline",
    description=(
        "Multi-modal Retrieval-Augmented Generation pipeline with Qdrant vector "
        "search, configurable chunking, re-ranking, and OpenAI-compatible LLM generation."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# CORS middleware
_cors_origins = os.getenv("CORS_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# ---------------------------------------------------------------------------
# API Key Authentication Middleware
# ---------------------------------------------------------------------------

_API_KEY = os.getenv("RAG_API_KEY", "")

_PUBLIC_PATHS = {"/health", "/openapi.json"}


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Validate X-API-Key header on protected endpoints."""
    if _API_KEY and request.url.path not in _PUBLIC_PATHS:
        provided = request.headers.get("X-API-Key", "")
        if provided != _API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _generate_document_id(text: str) -> str:
    """Produce a deterministic document ID from the source text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split *text* into overlapping chunks.

    Uses a simple character-level sliding window. Chunk boundaries are
    adjusted to avoid splitting in the middle of a word when possible.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if overlap >= size:
        overlap = size // 4

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)

        # Try to break at a whitespace boundary (look back up to 20% of chunk_size)
        if end < text_len:
            lookback_limit = max(end - size // 5, start)
            for pos in range(end, lookback_limit, -1):
                if text[pos] in (" ", "\n", "\t"):
                    end = pos
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance the window
        start = end - overlap if end < text_len else text_len

    return chunks


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate dense embeddings for a batch of texts via LocalAI."""
    client = get_openai_client()
    t0 = time.perf_counter()

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(
        "Embedded %d texts in %.1f ms (model=%s)",
        len(texts),
        elapsed,
        settings.embedding_model,
    )

    # Sort by index to guarantee order matches input
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in sorted_data]


def _compute_sparse_vector(text: str, dimension: int = 1536) -> list[float]:
    """Compute a simple TF-IDF-inspired sparse vector."""
    tokens = text.lower().split()
    if not tokens:
        return [0.0] * dimension

    # Term frequency
    tf: dict[str, int] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1

    # Hash-based projection into fixed-size vector
    vector = [0.0] * dimension
    for token, count in tf.items():
        idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dimension
        weight = 1.0 + math.log(count)
        vector[idx] += weight

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector


async def embed_for_strategy(texts: list[str]) -> list[list[float]]:
    """Produce embedding vectors according to the configured strategy."""
    strategy = settings.embedding_strategy

    if strategy == EmbeddingStrategy.DENSE:
        return await embed_texts(texts)

    if strategy == EmbeddingStrategy.SPARSE:
        return [_compute_sparse_vector(t, settings.embedding_dimension) for t in texts]

    # Hybrid: weighted combination of dense + sparse
    dense_vecs = await embed_texts(texts)
    sparse_vecs = [
        _compute_sparse_vector(t, settings.embedding_dimension) for t in texts
    ]

    alpha = 0.7  # dense weight
    combined: list[list[float]] = []
    for dv, sv in zip(dense_vecs, sparse_vecs):
        merged = [alpha * d + (1 - alpha) * s for d, s in zip(dv, sv)]
        # Re-normalize
        norm = math.sqrt(sum(v * v for v in merged))
        if norm > 0:
            merged = [v / norm for v in merged]
        combined.append(merged)

    return combined


def rerank_results(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """Re-rank retrieval candidates using a cross-encoder score simulation."""
    query_tokens = set(query.lower().split())

    for candidate in candidates:
        chunk_text_content: str = candidate.get("text", "")
        chunk_tokens = set(chunk_text_content.lower().split())

        intersection = query_tokens & chunk_tokens
        union = query_tokens | chunk_tokens
        lexical_score = len(intersection) / len(union) if union else 0.0

        vector_score: float = candidate.get("score", 0.0)

        rerank_score = 0.6 * vector_score + 0.4 * lexical_score
        candidate["rerank_score"] = round(rerank_score, 6)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain enough information to answer "
    "the question, say so clearly rather than making up an answer.\n\n"
    "Guidelines:\n"
    "- Be concise and accurate.\n"
    "- Cite the source context when possible.\n"
    "- If multiple sources provide relevant information, synthesize them.\n"
    "- Do not include information that is not supported by the context."
)


def build_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """Assemble the user prompt with retrieved context."""
    numbered_context = "\n\n".join(
        f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        f"Context:\n{numbered_context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


async def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Call the LLM to generate an answer grounded in the retrieved context."""
    client = get_openai_client()
    user_prompt = build_rag_prompt(query, context_chunks)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""


def _ensure_collection(collection_name: str) -> None:
    """Create the Qdrant collection if it does not already exist."""
    qclient = get_qdrant_client()
    try:
        qclient.get_collection(collection_name)
        logger.debug("Collection '%s' already exists", collection_name)
    except UnexpectedResponse:
        logger.info(
            "Creating collection '%s' (dim=%d, distance=Cosine)",
            collection_name,
            settings.embedding_dimension,
        )
        qclient.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )


# ---------------------------------------------------------------------------
# Utility: latency timer
# ---------------------------------------------------------------------------


class LatencyTracker:
    """Simple context-manager-based latency tracker for pipeline stages."""

    def __init__(self) -> None:
        self._timings: dict[str, float] = {}
        self._start: float = 0.0

    def start(self, label: str) -> "LatencyTracker":
        self._current_label = label
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._timings[self._current_label] = round(elapsed_ms, 2)
        return elapsed_ms

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._timings)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check with dependency statuses",
)
async def health_check() -> HealthResponse:
    """Return health status including connectivity to Qdrant and LocalAI."""
    deps: dict[str, str] = {}

    # Check Qdrant
    try:
        qclient = get_qdrant_client()
        qclient.get_collections()
        deps["qdrant"] = "healthy"
    except Exception:
        logger.warning("Qdrant health check failed")
        deps["qdrant"] = "unhealthy"

    # Check LocalAI (lightweight models list call)
    try:
        client = get_openai_client()
        client.models.list()
        deps["localai"] = "healthy"
    except Exception:
        logger.warning("LocalAI health check failed")
        deps["localai"] = "unhealthy"

    overall = "healthy" if all("healthy" == v for v in deps.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=APP_VERSION,
        dependencies=deps,
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Pipeline"],
    summary="Ingest a document into the vector store",
)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """Chunk, embed, and store a document in Qdrant."""
    tracker = LatencyTracker()
    collection = request.collection_name or settings.collection_name
    document_id = _generate_document_id(request.text)

    # --- 1. Chunking -------------------------------------------------------
    tracker.start("chunking")
    chunks = chunk_text(
        request.text,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    tracker.stop()
    logger.info(
        "Document %s chunked into %d segments (size=%s, overlap=%s)",
        document_id,
        len(chunks),
        request.chunk_size or settings.chunk_size,
        request.chunk_overlap or settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document produced zero chunks after splitting.",
        )

    # --- 2. Embedding ------------------------------------------------------
    tracker.start("embedding")
    try:
        vectors = await embed_for_strategy(chunks)
    except Exception:
        logger.exception("Embedding failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding service unavailable",
        )
    tracker.stop()

    # --- 3. Storage --------------------------------------------------------
    tracker.start("storage")
    try:
        _ensure_collection(collection)
        qclient = get_qdrant_client()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "document_id": document_id,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    **request.metadata,
                },
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        qclient.upsert(collection_name=collection, points=points)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Qdrant upsert failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store unavailable",
        )
    tracker.stop()

    logger.info(
        "Ingested document %s into '%s' (%d vectors, latency=%s)",
        document_id,
        collection,
        len(points),
        tracker.timings,
    )

    return IngestResponse(
        collection_name=collection,
        chunks_created=len(chunks),
        document_id=document_id,
        latency_ms=tracker.timings,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Pipeline"],
    summary="Query the RAG pipeline",
)
async def query_pipeline(request: QueryRequest) -> QueryResponse:
    """Execute the full RAG retrieval-generation pipeline."""
    tracker = LatencyTracker()
    collection = request.collection_name or settings.collection_name

    # --- 1. Embed query ----------------------------------------------------
    tracker.start("query_embedding")
    try:
        query_vectors = await embed_for_strategy([request.query])
        query_vector = query_vectors[0]
    except Exception:
        logger.exception("Query embedding failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding service unavailable",
        )
    tracker.stop()

    # --- 2. Retrieve -------------------------------------------------------
    tracker.start("retrieval")
    try:
        qclient = get_qdrant_client()
        search_results = qclient.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=request.top_k,
            with_payload=True,
        )
    except Exception:
        logger.exception("Qdrant search failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector search unavailable",
        )
    tracker.stop()

    if not search_results:
        return QueryResponse(
            answer=(
                "No relevant documents found in the knowledge base. "
                "Please ingest documents before querying."
            ),
            sources=[],
            latency_ms=tracker.timings,
        )

    # Convert Qdrant results to dicts for re-ranking
    candidates: list[dict[str, Any]] = [
        {
            "text": hit.payload.get("text", "") if hit.payload else "",
            "score": hit.score,
            "metadata": {
                k: v
                for k, v in (hit.payload or {}).items()
                if k != "text"
            },
        }
        for hit in search_results
    ]

    # --- 3. Re-rank --------------------------------------------------------
    tracker.start("reranking")
    reranked = rerank_results(request.query, candidates, request.top_n)
    tracker.stop()

    context_chunks = [r["text"] for r in reranked]

    # --- 4. Generate -------------------------------------------------------
    tracker.start("generation")
    try:
        answer = await generate_answer(request.query, context_chunks)
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM generation unavailable",
        )
    tracker.stop()

    sources = (
        [
            SourceChunk(
                text=r["text"],
                score=r["score"],
                rerank_score=r["rerank_score"],
                metadata=r["metadata"],
            )
            for r in reranked
        ]
        if request.include_sources
        else []
    )

    logger.info(
        "Query answered (top_k=%d, top_n=%d, latency=%s)",
        request.top_k,
        request.top_n,
        tracker.timings,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=tracker.timings,
    )


@app.get(
    "/collections",
    response_model=CollectionsResponse,
    tags=["Collections"],
    summary="List all Qdrant collections",
)
async def list_collections() -> CollectionsResponse:
    """Return summary information for every Qdrant collection."""
    try:
        qclient = get_qdrant_client()
        response = qclient.get_collections()
    except Exception:
        logger.exception("Failed to list collections")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store unavailable",
        )

    collections: list[CollectionInfo] = []
    for col in response.collections:
        try:
            info = qclient.get_collection(col.name)
            collections.append(
                CollectionInfo(
                    name=col.name,
                    vectors_count=info.vectors_count or 0,
                    status=info.status.value if info.status else "unknown",
                )
            )
        except Exception:
            collections.append(
                CollectionInfo(
                    name=col.name,
                    vectors_count=0,
                    status="error",
                )
            )

    return CollectionsResponse(collections=collections)


@app.delete(
    "/collections/{name}",
    response_model=DeleteCollectionResponse,
    tags=["Collections"],
    summary="Delete a Qdrant collection",
)
async def delete_collection(name: str) -> DeleteCollectionResponse:
    """Delete a Qdrant collection by name."""
    try:
        qclient = get_qdrant_client()
        result = qclient.delete_collection(collection_name=name)
    except Exception:
        logger.exception("Failed to delete collection '%s'", name)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store unavailable",
        )

    logger.info("Deleted collection '%s' (result=%s)", name, result)
    return DeleteCollectionResponse(deleted=bool(result), collection_name=name)
