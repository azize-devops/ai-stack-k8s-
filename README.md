<p align="center">
  <h1 align="center">AI Stack on Kubernetes</h1>
  <p align="center">
    <strong>Agent Workflows, RAG Pipeline & LLM Experimentation</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#architecture">Architecture</a> &bull;
    <a href="#components">Components</a> &bull;
    <a href="#module-details">Modules</a> &bull;
    <a href="docs/architecture.md">Detailed Docs</a>
  </p>
</p>

---

## Overview

A production-grade AI infrastructure platform deployed entirely on Kubernetes, purpose-built for experimenting with multi-agent orchestration, retrieval-augmented generation, LLM benchmarking, and parameter-efficient fine-tuning -- all running on local GPU hardware.

The stack brings together dual LLM backends (LocalAI and Ollama), a Qdrant vector database, a multi-stage RAG pipeline with hybrid embeddings and re-ranking, multi-agent workflows powered by CrewAI and LangGraph, and an AnythingLLM interface -- orchestrated through Helm charts, Kustomize overlays, and raw Kubernetes manifests. Every component communicates over cluster-internal DNS, runs with proper health checks and resource limits, and shares a single NVIDIA GPU via time-slicing.

---

## Architecture

```
                                    +---------------------------+
                                    |        End Users          |
                                    +------------+--------------+
                                                 |
                                    +------------v--------------+
                                    |     NGINX Ingress         |
                                    |  localai.local            |
                                    |  anythingllm.local        |
                                    +---+--------+----------+---+
                                        |        |          |
                         +--------------+   +----v----+  +--v--------------+
                         |                  |         |  |                 |
                +--------v-------+   +------v--+  +--v--v-----------+     |
                |   LocalAI      |   | Qdrant  |  |  AnythingLLM   |     |
                |   (LLM + Emb)  |   | VectorDB|  |  (Chat UI)     |     |
                |   :8080        |   | :6333   |  |  :3001         |     |
                +---^----^-------+   +--^---^--+  +----------------+     |
                    |    |              |   |                             |
                    |    +---------+----+   |                             |
                    |              |        |                             |
              +-----+------+ +----v--------v---+                         |
              |   Ollama   | |  RAG Pipeline    |                        |
              | (LLM Alt)  | |  (FastAPI)       |                        |
              |  :11434    | |  :8000           |                        |
              +-----^------+ +----^-------------+                        |
                    |             |                                       |
                    +------+------+                                      |
                           |                                             |
                +----------v-----------+                                 |
                |   Agent Workflows    <---------------------------------+
                |                      |
                |  +----------------+  |
                |  |  CrewAI        |  |
                |  |  Research Crew |  |
                |  +----------------+  |
                |  +----------------+  |
                |  |  LangGraph     |  |
                |  |  Query Router  |  |
                |  +----------------+  |
                +----------------------+

    Namespace: ai-stack          Storage: Longhorn CSI
    GPU: NVIDIA (time-sliced)    Ingress: NGINX
```

---

## Features

- **Multi-Agent Workflows** -- CrewAI research crew (Researcher / Writer / Reviewer) and LangGraph stateful query-routing graph with conditional RAG retrieval
- **RAG Pipeline** -- Production-grade FastAPI service with configurable chunking, dense / sparse / hybrid embedding strategies, and two-signal re-ranking
- **LLM Benchmarking** -- Systematic comparison of local models (Mistral 7B, LLaMA 3 8B, Phi-3) against cloud baselines (GPT-4o-mini) measuring latency, throughput, quality, and memory
- **QLoRA Fine-Tuning** -- 4-bit NF4 quantized training with LoRA adapters via PEFT + bitsandbytes, optimized for consumer GPUs (6 GB VRAM)
- **Dual LLM Backend** -- LocalAI (OpenAI-compatible API + model gallery) and Ollama (pull-and-run simplicity) running side-by-side
- **Qdrant Vector Database** -- Persistent vector storage with HTTP and gRPC interfaces, backed by Longhorn volumes
- **Kubernetes-Native** -- Helm charts for LocalAI / Qdrant / AnythingLLM, Kustomize for the RAG pipeline, raw manifests for Ollama
- **GPU-Accelerated** -- NVIDIA GPU Operator integration with tolerations and resource limits for GPU time-slicing across workloads

---

## Components

| Component | Type | Port | Deployment | Description |
|:----------|:-----|:----:|:-----------|:------------|
| **LocalAI** | LLM Backend | `8080` | Helm | OpenAI-compatible API serving Mistral, LLaMA, Phi-3. GPU-accelerated with 50Gi model storage |
| **Ollama** | LLM Backend | `11434` | Manifests | Pull-and-run LLM server. Pre-loads TinyLlama and Phi-3 Mini via init container |
| **Qdrant** | Vector DB | `6333` / `6334` | Helm | High-performance vector similarity search. 20Gi persistent storage on Longhorn |
| **AnythingLLM** | Chat UI | `3001` | Helm | Full-featured chat interface wired to LocalAI for inference and Qdrant for RAG |
| **RAG Pipeline** | API Service | `8000` | Kustomize | FastAPI application: ingest, chunk, embed, store, retrieve, re-rank, generate |

### Service Endpoints (in-cluster)

```
LocalAI:      http://localai.ai-stack.svc.cluster.local:8080
Ollama:       http://ollama.ai-stack.svc.cluster.local:11434
Qdrant:       http://qdrant.ai-stack.svc.cluster.local:6333
AnythingLLM:  http://anythingllm.ai-stack.svc.cluster.local:3001
RAG Pipeline: http://rag-pipeline.ai-stack.svc.cluster.local:8000
```

---

## Quick Start

### Prerequisites

- Kubernetes cluster (v1.27+) with NVIDIA GPU Operator installed
- `kubectl`, `helm`, and `kustomize` on your PATH
- Longhorn (or another CSI driver) for persistent volumes
- NGINX Ingress Controller

### Deploy

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-stack-k8s.git
cd ai-stack-k8s

# 2. Configure environment (optional -- sensible defaults are built in)
cp .env.example .env
# Edit .env to add API keys for cloud model comparison

# 3. Deploy the full stack (namespace, Qdrant, LocalAI, Ollama, AnythingLLM, RAG Pipeline)
./deploy-all.sh

# 4. Verify everything is running
kubectl get pods -n ai-stack

# 5. Access the services
#    AnythingLLM UI:  http://anythingllm.local
#    LocalAI API:     http://localai.local/v1/models
#    RAG Pipeline:    kubectl port-forward svc/rag-pipeline 8000:8000 -n ai-stack
```

### Teardown

```bash
./uninstall-all.sh
```

---

## Project Structure

```
ai-stack-k8s/
|
|-- infrastructure/                # Kubernetes infrastructure manifests
|   |-- namespace/
|   |   +-- namespace.yaml         # ai-stack namespace definition
|   |-- localai/
|   |   |-- install.sh             # Helm-based LocalAI installer
|   |   |-- values.yaml            # Helm values (GPU limits, model config)
|   |   +-- ingress.yaml           # NGINX ingress for localai.local
|   |-- ollama/
|   |   |-- deployment.yaml        # Deployment with init container for model pulls
|   |   |-- service.yaml           # ClusterIP service on port 11434
|   |   +-- pvc.yaml               # Longhorn PVC for model storage
|   |-- qdrant/
|   |   |-- install.sh             # Helm-based Qdrant installer
|   |   +-- values.yaml            # Helm values (persistence, resource limits)
|   +-- anythingllm/
|       |-- install.sh             # Helm-based AnythingLLM installer
|       |-- values.yaml            # Helm values (LLM provider, Qdrant connection)
|       +-- ingress.yaml           # NGINX ingress for anythingllm.local
|
|-- rag-pipeline/                  # Retrieval-Augmented Generation service
|   |-- docker/
|   |   |-- Dockerfile             # Container image build
|   |   |-- server.py              # FastAPI app (ingest, query, collections)
|   |   |-- config.py              # Centralized settings with env overrides
|   |   +-- requirements.txt       # Python dependencies
|   |-- configmap.yaml             # Pipeline configuration (models, chunking, etc.)
|   |-- deployment.yaml            # Kubernetes Deployment with health probes
|   |-- service.yaml               # ClusterIP service on port 8000
|   |-- pvc.yaml                   # Document storage volume
|   +-- kustomization.yaml         # Kustomize overlay for the pipeline
|
|-- agents/                        # Multi-agent AI workflows
|   |-- crewai-research-agent/
|   |   |-- agents.py              # Researcher, Writer, Reviewer agent defs
|   |   |-- tasks.py               # Task chain with context wiring
|   |   |-- main.py                # Entry point for CrewAI execution
|   |   +-- config.yaml            # Agent roles, goals, backstories
|   |-- langgraph-workflow/
|   |   |-- graph.py               # StateGraph with conditional routing
|   |   |-- nodes.py               # classify, retrieve, generate, quality nodes
|   |   |-- state.py               # WorkflowState TypedDict schema
|   |   +-- main.py                # Entry point for LangGraph execution
|   |-- docker/
|   |   |-- Dockerfile             # Agent container image
|   |   +-- requirements.txt       # Agent Python dependencies
|   +-- requirements.txt           # Shared agent requirements
|
|-- llm-comparison/                # LLM benchmarking framework
|   |-- compare_models.py          # Benchmark runner (latency, quality, memory)
|   |-- benchmark_results.json     # Latest benchmark output
|   +-- metrics/
|       |-- latency.py             # Request timing + first-token measurement
|       |-- response_quality.py    # Keyword & heuristic quality scoring
|       +-- memory_usage.py        # System memory sampling
|
|-- fine-tuning/                   # Parameter-efficient fine-tuning
|   |-- lora_finetune.py           # QLoRA training script (4-bit NF4 + PEFT)
|   |-- prepare_dataset.py         # Dataset preprocessing utilities
|   |-- evaluate.py                # Post-training evaluation
|   +-- data/
|       +-- sample_dataset.jsonl   # Example instruction-tuning data
|
|-- notebooks/                     # Jupyter notebooks for experimentation
|-- docs/                          # Detailed documentation
|   +-- architecture.md            # System architecture deep-dive
|
|-- deploy-all.sh                  # One-command full stack deployment
|-- uninstall-all.sh               # One-command full stack teardown
|-- .env.example                   # Environment variable template
+-- .gitignore                     # Git ignore rules
```

---

## Tech Stack

| Category | Technologies |
|:---------|:------------|
| **Orchestration** | Kubernetes, Helm, Kustomize |
| **LLM Serving** | LocalAI, Ollama |
| **Vector Search** | Qdrant (HTTP + gRPC) |
| **RAG Framework** | FastAPI, OpenAI Python SDK, Pydantic |
| **Agent Frameworks** | CrewAI, LangGraph, LangChain |
| **Fine-Tuning** | Hugging Face Transformers, PEFT, bitsandbytes |
| **Benchmarking** | Custom Python framework (latency, quality, memory metrics) |
| **Chat Interface** | AnythingLLM |
| **GPU** | NVIDIA GPU Operator, CUDA, GPU time-slicing |
| **Storage** | Longhorn CSI (persistent volumes) |
| **Networking** | NGINX Ingress Controller, ClusterIP services |
| **Languages** | Python 3.12, Bash |

---

## Module Details

### Agents (`agents/`)

Two distinct multi-agent architectures running against the same LLM backends:

**CrewAI Research Crew** -- Three specialized agents (Senior Research Analyst, Technical Writer, Quality Reviewer) collaborating sequentially via a task chain. The Researcher gathers information, the Writer synthesizes it into a structured report, and the Reviewer evaluates accuracy and completeness. All agents share a single `ChatOpenAI` instance pointed at LocalAI.

**LangGraph Query Router** -- A stateful directed graph that classifies incoming queries (factual / analytical / creative), conditionally routes them through RAG retrieval or direct generation, produces an LLM response, and validates output quality with a built-in scoring node. The graph uses conditional edges for dynamic routing and integrates with Qdrant for context retrieval.

### RAG Pipeline (`rag-pipeline/`)

A production-grade Retrieval-Augmented Generation service built with FastAPI and deployed via Kustomize. The pipeline implements:

- **Document ingestion**: text chunking with configurable size/overlap and word-boundary awareness
- **Embedding strategies**: dense (via LocalAI), sparse (TF-IDF hash projection), and hybrid (weighted dense + sparse fusion)
- **Vector storage**: Qdrant collection management with automatic creation and UUID-based point IDs
- **Retrieval + Re-ranking**: top-K vector search followed by a two-signal re-ranker combining cosine similarity with Jaccard lexical overlap
- **Grounded generation**: LLM answers constrained to retrieved context with numbered source citations
- **Full observability**: per-stage latency tracking, structured logging, dependency health checks

### LLM Comparison (`llm-comparison/`)

A systematic benchmarking framework that evaluates multiple LLMs across standardized prompts in both English and Turkish. Measures time-to-first-token (streaming), total latency, tokens-per-second throughput, response quality (keyword coverage + heuristic scoring), and system memory delta. Compares local models (Mistral 7B, LLaMA 3 8B, Phi-3 via LocalAI) against cloud baselines (GPT-4o-mini via OpenAI API). Results are persisted as structured JSON for downstream analysis.

### Fine-Tuning (`fine-tuning/`)

A QLoRA fine-tuning pipeline optimized for consumer-grade NVIDIA GPUs (6 GB VRAM). Uses 4-bit NF4 quantization via bitsandbytes with double quantization enabled, injects LoRA adapters on `q_proj` and `v_proj` attention layers through PEFT, and trains with gradient checkpointing, paged AdamW 8-bit optimizer, and cosine learning rate scheduling. Includes a metrics logger that writes per-step training telemetry to JSONL, and saves adapter weights separately for efficient deployment.

---

## Future Plans (Phase 2)

- **ColQwen2 Visual Embeddings** -- Integrate vision-language embeddings for multi-modal RAG over documents with images, tables, and charts
- **Automated CI/CD with ArgoCD** -- GitOps-driven continuous deployment with automated model promotion pipelines
- **Multi-Node GPU Cluster** -- Scale out to multiple GPU nodes with topology-aware scheduling and distributed inference
- **Model Serving with vLLM** -- Replace LocalAI inference with vLLM for PagedAttention, continuous batching, and significantly higher throughput
- **Prometheus + Grafana Observability** -- Full metrics pipeline with custom dashboards for LLM latency, GPU utilization, and pipeline health
- **SPLADE Sparse Encoder** -- Replace the hash-projection sparse embeddings with a learned sparse model for production-quality hybrid retrieval

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with Kubernetes, Python, and local GPU hardware.
</p>
