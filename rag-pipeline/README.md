# RAG Pipeline - Multi-Modal Retrieval-Augmented Generation

A production-grade RAG (Retrieval-Augmented Generation) pipeline deployed on Kubernetes,
built with FastAPI, Qdrant vector search, and OpenAI-compatible local LLMs via LocalAI.

## Architecture

```
                    +-------------+
                    |   Client    |
                    +------+------+
                           |
                    +------v------+
                    | FastAPI     |
                    | RAG Server  |
                    | (port 8000) |
                    +------+------+
                           |
              +------------+------------+
              |            |            |
       +------v---+  +----v-----+ +----v-----+
       | LocalAI  |  |  Qdrant  | |  Ollama  |
       | Embeddings|  |  Vector  | |   LLM    |
       | (8080)   |  |  Search  | | (11434)  |
       +----------+  | (6333)   | +----------+
                      +----------+
```

## Features

### Multi-Modal Embedding Strategies

The pipeline supports three embedding strategies to balance retrieval quality and performance:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Dense** | Standard dense vector embeddings via LocalAI | General-purpose semantic search |
| **Sparse** | TF-IDF weighted sparse vectors | Keyword-heavy queries, exact term matching |
| **Hybrid** | Weighted combination of dense + sparse | Best overall retrieval quality (default) |

Configure the strategy via the `EMBEDDING_STRATEGY` environment variable.

### Intelligent Document Chunking

Documents are split into overlapping chunks before embedding. Both parameters are tunable:

- **CHUNK_SIZE** (default: 512 tokens) -- Controls the maximum size of each text chunk.
  Smaller chunks improve precision; larger chunks preserve more context.
- **CHUNK_OVERLAP** (default: 50 tokens) -- Overlapping window between consecutive chunks
  to prevent information loss at chunk boundaries.

Experimentation guidance:

| Chunk Size | Best For |
|------------|----------|
| 128-256 | Short, factual Q&A |
| 512 | General-purpose (default) |
| 1024-2048 | Long-form summarization, complex reasoning |

### Re-Ranking Mechanism

After initial vector retrieval, results pass through a cross-encoder re-ranking stage:

1. **Initial retrieval** -- Top-K candidates fetched from Qdrant by vector similarity.
2. **Cross-encoder scoring** -- Each (query, candidate) pair is scored for relevance
   using a lightweight cross-encoder model that considers full token interaction.
3. **Final selection** -- Candidates are re-sorted by cross-encoder score and the
   top-N results are passed to the LLM for generation.

This two-stage approach dramatically improves answer quality compared to relying
solely on embedding similarity.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Ingest documents -- chunks text, generates embeddings, stores in Qdrant |
| `POST` | `/query` | Query the pipeline -- embed, search, re-rank, generate LLM response |
| `GET` | `/health` | Health check with dependency status |
| `GET` | `/collections` | List all Qdrant collections |
| `DELETE` | `/collections/{name}` | Delete a Qdrant collection |

### Example: Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Kubernetes is an open-source container orchestration platform...",
    "metadata": {"source": "k8s-docs", "category": "infrastructure"},
    "collection_name": "knowledge_base"
  }'
```

### Example: Query the Pipeline

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Kubernetes handle pod scheduling?",
    "collection_name": "knowledge_base",
    "top_k": 10,
    "top_n": 3
  }'
```

## Kubernetes Deployment

All manifests are managed with Kustomize:

```bash
# Deploy the RAG pipeline
kubectl apply -k rag-pipeline/

# Verify the deployment
kubectl -n ai-stack get pods -l app=rag-pipeline
kubectl -n ai-stack logs -l app=rag-pipeline -f

# Port-forward for local testing
kubectl -n ai-stack port-forward svc/rag-pipeline 8000:8000
```

### Resources

| Resource | File | Purpose |
|----------|------|---------|
| Deployment | `deployment.yaml` | Pod spec with probes and resource limits |
| Service | `service.yaml` | ClusterIP service on port 8000 |
| ConfigMap | `configmap.yaml` | Runtime configuration |
| PVC | `pvc.yaml` | 10Gi persistent volume for document storage |
| Kustomize | `kustomization.yaml` | Resource aggregation |

### Configuration

All configuration is injected via the `rag-pipeline-config` ConfigMap. Override values
by editing `configmap.yaml` or patching with Kustomize overlays:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALAI_URL` | `http://localai.ai-stack.svc.cluster.local:8080/v1` | LocalAI API base URL |
| `OLLAMA_URL` | `http://ollama.ai-stack.svc.cluster.local:11434` | Ollama API base URL |
| `QDRANT_URL` | `http://qdrant.ai-stack.svc.cluster.local:6333` | Qdrant vector DB URL |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Model used for embeddings |
| `LLM_MODEL` | `gpt-3.5-turbo` | Model used for generation |
| `CHUNK_SIZE` | `512` | Document chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `COLLECTION_NAME` | `knowledge_base` | Default Qdrant collection |
| `EMBEDDING_STRATEGY` | `dense` | Embedding strategy (dense/sparse/hybrid) |
| `LOG_LEVEL` | `INFO` | Application log level |

## Local Development

```bash
cd rag-pipeline/docker

# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (or create .env file)
export LOCALAI_URL=http://localhost:8080/v1
export QDRANT_URL=http://localhost:6333

# Run the server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Build

```bash
cd rag-pipeline/docker
docker build -t rag-pipeline:latest .
docker run -p 8000:8000 --env-file .env rag-pipeline:latest
```
