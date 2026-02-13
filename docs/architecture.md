# Architecture

Detailed system architecture for the AI Stack on Kubernetes platform.

---

## Table of Contents

- [System Overview](#system-overview)
- [Component Interaction Diagram](#component-interaction-diagram)
- [Data Flow: RAG Pipeline](#data-flow-rag-pipeline)
- [Data Flow: Agent Workflows](#data-flow-agent-workflows)
- [Kubernetes Resource Map](#kubernetes-resource-map)
- [Network Topology](#network-topology)
- [Storage Architecture](#storage-architecture)
- [GPU Sharing Strategy](#gpu-sharing-strategy)

---

## System Overview

The AI Stack is a single-namespace Kubernetes deployment (`ai-stack`) that provides an integrated platform for LLM inference, vector search, RAG, multi-agent orchestration, benchmarking, and fine-tuning. All infrastructure components run as Kubernetes workloads with proper resource limits, health probes, and persistent storage. The platform is designed for a single-node cluster with one NVIDIA GPU shared across workloads via time-slicing.

### Design Principles

1. **Kubernetes-native** -- Every stateful and stateless component is a proper Kubernetes workload with manifests in version control.
2. **OpenAI-compatible API surface** -- LocalAI exposes the `/v1` API, so all upstream consumers (RAG pipeline, agents, benchmarks) use the standard OpenAI Python SDK.
3. **Loose coupling** -- Components communicate over cluster DNS; swapping LocalAI for Ollama (or any OpenAI-compatible server) requires only an environment variable change.
4. **Persistent by default** -- All model weights, vector indices, and document stores are backed by Longhorn PVCs so pod restarts do not lose state.
5. **GPU as a shared resource** -- A single NVIDIA GPU is time-sliced across LocalAI and Ollama using Kubernetes resource limits and tolerations.

---

## Component Interaction Diagram

```
+------------------------------------------------------------------+
|                         Kubernetes Cluster                        |
|  Namespace: ai-stack                                             |
|                                                                  |
|  +------------------+       +------------------+                 |
|  |  NGINX Ingress   |       |  NGINX Ingress   |                 |
|  |  localai.local   |       | anythingllm.local|                 |
|  +--------+---------+       +--------+---------+                 |
|           |                          |                           |
|  +--------v---------+       +--------v---------+                 |
|  |                  |       |                  |                 |
|  |    LocalAI       |       |  AnythingLLM     |                 |
|  |    (Helm)        +<------+  (Helm)          |                 |
|  |                  |  LLM  |                  |                 |
|  |  - Mistral 7B    |  API  |  - Chat UI       |                 |
|  |  - LLaMA 3 8B   |       |  - Doc Upload    |                 |
|  |  - Phi-3         |       |  - Workspace Mgmt|                 |
|  |  - Embeddings    |       |                  |                 |
|  |                  |       +--------+---------+                 |
|  +--+--------+------+                |                           |
|     ^        ^                       | Vector                    |
|     |        |                       | Queries                   |
|     |        |               +-------v---------+                 |
|     |        |               |                 |                 |
|     |        +<--------------+    Qdrant       |                 |
|     |        |  Embed/Search |    (Helm)       |                 |
|     |        |               |                 |                 |
|     |        |               |  - HTTP :6333   |                 |
|     |        |               |  - gRPC :6334   |                 |
|     |        |               |  - 20Gi Storage |                 |
|     |        |               +---^----^--------+                 |
|     |        |                   |    |                          |
|     |  +-----+-------------------+    |                          |
|     |  |     |                        |                          |
|  +--+--v-----+------+         +------++---------+               |
|  |                  |         |                  |               |
|  |  RAG Pipeline    |         |  Agent Workflows |               |
|  |  (Kustomize)     |         |  (Local / K8s)   |               |
|  |                  |         |                  |               |
|  |  - FastAPI :8000 |         |  - CrewAI Crew   |               |
|  |  - Ingest API    |         |  - LangGraph     |               |
|  |  - Query API     |         |    Query Router  |               |
|  |  - Re-ranker     |         |                  |               |
|  +------------------+         +------------------+               |
|                                                                  |
|  +------------------+                                            |
|  |    Ollama        |                                            |
|  |   (Manifests)    |                                            |
|  |                  |                                            |
|  |  - TinyLlama     |                                            |
|  |  - Phi-3 Mini    |                                            |
|  |  - :11434        |                                            |
|  +------------------+                                            |
|                                                                  |
+------------------------------------------------------------------+
         |                |                  |
   +-----v----+   +------v------+   +-------v------+
   | Longhorn  |   | Longhorn    |   | Longhorn     |
   | PVC 50Gi  |   | PVC 20Gi   |   | PVC 10Gi     |
   | (LocalAI  |   | (Qdrant    |   | (AnythingLLM |
   |  Models)  |   |  Vectors)  |   |  Data)       |
   +-----------+   +------------+   +--------------+
```

---

## Data Flow: RAG Pipeline

The RAG pipeline operates in two modes: **ingestion** and **query**.

### Ingestion Flow

```
                     POST /ingest
                         |
                         v
              +----------+----------+
              |  1. Validate Input  |
              |  (Pydantic model)   |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  2. Chunk Document  |
              |                     |
              |  - Sliding window   |
              |  - Word-boundary    |
              |    aware splitting  |
              |  - Configurable     |
              |    size (512) and   |
              |    overlap (50)     |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  3. Generate        |
              |     Embeddings      |
              |                     |
              |  Strategy:          |
              |  - DENSE: LocalAI   |
              |    /v1/embeddings   |
              |  - SPARSE: TF-IDF  |
              |    hash projection  |
              |  - HYBRID: 0.7 *   |
              |    dense + 0.3 *    |
              |    sparse, L2-norm  |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  4. Upsert Vectors  |
              |     to Qdrant       |
              |                     |
              |  - Auto-create      |
              |    collection       |
              |  - UUID point IDs   |
              |  - Cosine distance  |
              |  - Payload: text,   |
              |    doc_id, chunk_ix,|
              |    user metadata    |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  Response:          |
              |  - collection_name  |
              |  - chunks_created   |
              |  - document_id      |
              |  - per-stage        |
              |    latency_ms       |
              +---------------------+
```

### Query Flow

```
                     POST /query
                         |
                         v
              +----------+----------+
              |  1. Embed Query     |
              |                     |
              |  Same strategy as   |
              |  ingestion (dense,  |
              |  sparse, or hybrid) |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  2. Vector Search   |
              |     (Qdrant)        |
              |                     |
              |  - top_k candidates |
              |    (default: 10)    |
              |  - Cosine similarity|
              |  - Returns payload  |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  3. Re-rank         |
              |                     |
              |  Two-signal score:  |
              |  0.6 * vector_sim + |
              |  0.4 * jaccard_lex  |
              |                     |
              |  - Sort descending  |
              |  - Keep top_n       |
              |    (default: 3)     |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  4. LLM Generation  |
              |                     |
              |  - System prompt:   |
              |    answer from      |
              |    context only     |
              |  - Numbered source  |
              |    citations        |
              |  - temp=0.2         |
              |  - max_tokens=1024  |
              +----------+----------+
                         |
                         v
              +----------+----------+
              |  Response:          |
              |  - answer           |
              |  - sources[] with   |
              |    text, score,     |
              |    rerank_score,    |
              |    metadata         |
              |  - per-stage        |
              |    latency_ms       |
              +---------------------+
```

---

## Data Flow: Agent Workflows

### CrewAI Research Crew

```
     +------------------------------------------------------+
     |                    CrewAI Crew                        |
     |                                                      |
     |  +-----------+    +-----------+    +-----------+     |
     |  |           |    |           |    |           |     |
     |  |Researcher +--->+  Writer   +--->+ Reviewer  |     |
     |  |           |    |           |    |           |     |
     |  +-----------+    +-----------+    +-----------+     |
     |       |                |                |            |
     |       v                v                v            |
     |  Research          Draft             Quality         |
     |  findings          report            assessment      |
     |                                                      |
     +---+--------------------+--------------------+--------+
         |                    |                    |
         v                    v                    v
     +---+----+          +---+----+          +----+---+
     | LocalAI |          | LocalAI |          | LocalAI |
     | LLM API |          | LLM API |          | LLM API |
     +---------+          +---------+          +---------+

     Sequential execution with context passing:
       1. Researcher gathers information on the topic
       2. Writer receives research output, drafts report
       3. Reviewer receives both research + draft, evaluates quality
```

### LangGraph Query Router

```
                        START
                          |
                          v
                +---------+---------+
                |  classify_query   |
                |                   |
                |  LLM classifies   |
                |  the query as     |
                |  factual /        |
                |  analytical /     |
                |  creative         |
                |                   |
                |  Decides route:   |
                |  "rag" or "direct"|
                +----+--------+----+
                     |        |
           route="rag"      route="direct"
                     |        |
                     v        |
           +---------+--+     |
           | retrieve_   |     |
           | context     |     |
           |             |     |
           | Qdrant      |     |
           | vector      |     |
           | search      |     |
           | (fallback:  |     |
           |  simulated) |     |
           +------+------+     |
                  |             |
                  +------+------+
                         |
                         v
               +---------+---------+
               | generate_response |
               |                   |
               | LLM generates     |
               | answer using      |
               | context (if RAG)  |
               | or directly       |
               +---------+---------+
                         |
                         v
               +---------+---------+
               |  check_quality    |
               |                   |
               |  LLM self-eval:   |
               |  score 0-100      |
               |  passed if >= 60  |
               +---------+---------+
                         |
                         v
                        END

     State Schema (WorkflowState):
       query, classification, context[],
       response, route, quality_score,
       quality_passed, messages[]
```

---

## Kubernetes Resource Map

All resources live in the `ai-stack` namespace.

### Deployments & StatefulSets

| Workload | Kind | Replicas | Image | Resource Requests | Resource Limits |
|:---------|:-----|:--------:|:------|:------------------|:----------------|
| `localai` | Deployment (Helm) | 1 | `quay.io/go-skynet/local-ai:latest` | 2 CPU, 4Gi RAM | 4 CPU, 8Gi RAM, 1 GPU |
| `ollama` | Deployment | 1 | `ollama/ollama:latest` | 2 CPU, 4Gi RAM | 4 CPU, 8Gi RAM, 1 GPU |
| `qdrant` | StatefulSet (Helm) | 1 | `qdrant/qdrant:latest` | 500m CPU, 1Gi RAM | 1 CPU, 2Gi RAM |
| `anythingllm` | Deployment (Helm) | 1 | `mintplexlabs/anythingllm:latest` | 500m CPU, 1Gi RAM | 1 CPU, 2Gi RAM |
| `rag-pipeline` | Deployment | 1 | `rag-pipeline:latest` | 250m CPU, 512Mi RAM | 1 CPU, 2Gi RAM |

### Services

| Service | Type | Ports | Selector |
|:--------|:-----|:------|:---------|
| `localai` | ClusterIP | `8080/TCP` | `app: localai` |
| `ollama` | ClusterIP | `11434/TCP` | `app: ollama` |
| `qdrant` | ClusterIP | `6333/TCP` (HTTP), `6334/TCP` (gRPC) | `app: qdrant` |
| `anythingllm` | ClusterIP | `3001/TCP` | `app: anythingllm` |
| `rag-pipeline` | ClusterIP | `8000/TCP` | `app: rag-pipeline` |

### Ingress Rules

| Host | Backend Service | Port | Annotations |
|:-----|:----------------|:----:|:------------|
| `localai.local` | `localai` | `8080` | `proxy-body-size: 50m`, timeouts: 300s |
| `anythingllm.local` | `anythingllm` | `3001` | `proxy-body-size: 100m`, timeout: 300s |

### ConfigMaps

| Name | Used By | Key Settings |
|:-----|:--------|:-------------|
| `rag-pipeline-config` | `rag-pipeline` | `LOCALAI_URL`, `QDRANT_URL`, `EMBEDDING_MODEL`, `LLM_MODEL`, `CHUNK_SIZE`, `EMBEDDING_STRATEGY` |

### PersistentVolumeClaims

| PVC Name | Size | StorageClass | Used By | Mount Path |
|:---------|:----:|:-------------|:--------|:-----------|
| `localai-models` | 50Gi | `longhorn` | LocalAI | `/models` |
| `ollama-data` | (Longhorn default) | `longhorn` | Ollama | `/root/.ollama` |
| `qdrant-storage` | 20Gi | `longhorn` | Qdrant | `/qdrant/storage` |
| `anythingllm-data` | 10Gi | `longhorn` | AnythingLLM | (app data) |
| `rag-pipeline-documents` | (Longhorn default) | `longhorn` | RAG Pipeline | `/data/documents` |

---

## Network Topology

```
External Traffic
      |
      v
+-----+------------------+
| NGINX Ingress Controller|
| (LoadBalancer / NodePort)|
+--+------------------+---+
   |                  |
   | localai.local    | anythingllm.local
   |                  |
   v                  v
+---------+    +-----------+
| LocalAI |    | AnythingLLM|
| :8080   |    |   :3001   |
+-+---+---+    +-----+-----+
  ^   ^              |
  |   |              | (calls LocalAI API + Qdrant API)
  |   |              |
  |   |    +---------v---------+
  |   +--->|      Qdrant       |
  |        |  HTTP :6333       |
  |        |  gRPC :6334       |
  |        +---^-------^-------+
  |            |       |
  |   +--------+       +--------+
  |   |                         |
+-+---+------+          +------+-----------+
| RAG Pipeline|          | Agent Workflows  |
|   :8000    |          | (local process   |
|            |          |  or K8s Job)     |
+------+-----+          +---------+--------+
       |                          |
       +-------+    +-------------+
               |    |
          +----v----v----+
          |    Ollama    |
          |   :11434     |
          +--------------+

All service-to-service communication uses Kubernetes DNS:
  <service>.<namespace>.svc.cluster.local:<port>

No service mesh. Plain ClusterIP networking.
TLS termination at the Ingress layer (if configured).
```

### DNS Resolution

All in-cluster communication uses Kubernetes DNS with fully-qualified service names:

```
localai.ai-stack.svc.cluster.local:8080        ->  LocalAI OpenAI-compatible API
ollama.ai-stack.svc.cluster.local:11434         ->  Ollama LLM server
qdrant.ai-stack.svc.cluster.local:6333          ->  Qdrant HTTP API
qdrant.ai-stack.svc.cluster.local:6334          ->  Qdrant gRPC API
anythingllm.ai-stack.svc.cluster.local:3001     ->  AnythingLLM web interface
rag-pipeline.ai-stack.svc.cluster.local:8000    ->  RAG Pipeline FastAPI
```

---

## Storage Architecture

All persistent data uses the **Longhorn** distributed block storage CSI driver with `ReadWriteOnce` access mode.

```
+--------------------------------------------------------------------+
|                    Longhorn Storage Layer                           |
|                                                                    |
|  +-------------------+  +------------------+  +------------------+ |
|  | localai-models    |  | qdrant-storage   |  | anythingllm-data | |
|  | 50 Gi             |  | 20 Gi            |  | 10 Gi            | |
|  |                   |  |                  |  |                  | |
|  | Contents:         |  | Contents:        |  | Contents:        | |
|  | - GGUF model      |  | - Vector indices |  | - Chat history   | |
|  |   weights         |  | - Collection     |  | - Workspace      | |
|  | - Tokenizers      |  |   metadata       |  |   settings       | |
|  | - Model configs   |  | - WAL segments   |  | - Uploaded docs  | |
|  +-------------------+  +------------------+  +------------------+ |
|                                                                    |
|  +-------------------+  +------------------+                       |
|  | ollama-data       |  | rag-pipeline-    |                       |
|  | (default size)    |  | documents        |                       |
|  |                   |  | (default size)   |                       |
|  | Contents:         |  |                  |                       |
|  | - Ollama blobs    |  | Contents:        |                       |
|  | - Model manifests |  | - Ingested raw   |                       |
|  | - TinyLlama       |  |   documents      |                       |
|  | - Phi-3 Mini      |  |                  |                       |
|  +-------------------+  +------------------+                       |
|                                                                    |
+--------------------------------------------------------------------+

Total persistent storage: ~80+ Gi
Access mode: ReadWriteOnce (single node)
Reclaim policy: Retain (data survives PVC deletion)
```

### Storage Sizing Rationale

| Volume | Size | Rationale |
|:-------|:----:|:----------|
| LocalAI Models | 50 Gi | Quantized GGUF models range from 4-8 GB each; 50 Gi supports 5-8 models simultaneously |
| Qdrant Vectors | 20 Gi | 1536-dimensional float32 vectors at ~6 KB each; 20 Gi supports approximately 3 million vectors |
| AnythingLLM | 10 Gi | Chat history, workspace configs, and uploaded documents |
| Ollama Data | Default | TinyLlama (~640 MB) + Phi-3 Mini (~2.3 GB); a few gigabytes suffices |
| RAG Documents | Default | Raw document store for the ingestion pipeline |

---

## GPU Sharing Strategy

The cluster uses a single NVIDIA GPU shared between LocalAI and Ollama via **Kubernetes resource limits and NVIDIA GPU time-slicing**.

### How Time-Slicing Works

```
+---------------------------------------------------------------+
|                     NVIDIA GPU (Physical)                      |
|                                                                |
|  GPU Time-Slicing (NVIDIA Device Plugin)                       |
|  +-----------------------------------------------------------+|
|  |                                                           ||
|  |  Time Slice 1        Time Slice 2        Time Slice 3     ||
|  |  +---------------+   +---------------+   +-----------+    ||
|  |  |   LocalAI     |   |    Ollama     |   |  (idle/   |    ||
|  |  |   Inference   |   |   Inference   |   |  finetune)|    ||
|  |  +---------------+   +---------------+   +-----------+    ||
|  |                                                           ||
|  +-----------------------------------------------------------+|
|                                                                |
|  VRAM is NOT partitioned -- workloads share the full VRAM      |
|  pool. The scheduler time-slices GPU compute cycles.           |
+---------------------------------------------------------------+
```

### Configuration

Both LocalAI and Ollama request `nvidia.com/gpu: 1` in their resource limits and include GPU tolerations:

```yaml
# Applied to both LocalAI and Ollama deployments
resources:
  limits:
    nvidia.com/gpu: "1"

tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

The NVIDIA GPU Operator's device plugin is configured with time-slicing enabled, which advertises the single physical GPU as multiple virtual GPU resources. This allows the Kubernetes scheduler to place both LocalAI and Ollama on the same node, each receiving a `nvidia.com/gpu: 1` allocation that maps to time-shared access to the physical GPU.

### Important Considerations

1. **VRAM contention** -- Time-slicing shares VRAM. If both LocalAI and Ollama load large models simultaneously, out-of-memory errors can occur. Keep model sizes in check or ensure only one backend loads large models at a time.
2. **Throughput impact** -- Concurrent GPU workloads reduce per-workload throughput proportionally. For latency-sensitive inference, consider scaling to zero the unused backend.
3. **Fine-tuning isolation** -- The `fine-tuning/` module is designed to run offline (not as a persistent K8s workload). Schedule fine-tuning jobs when inference workloads are idle to avoid VRAM conflicts.
4. **Monitoring** -- Use `nvidia-smi` or the DCGM exporter to monitor GPU utilization and VRAM consumption per process.

### Resource Budget (Single Node)

| Workload | CPU Request | CPU Limit | Memory Request | Memory Limit | GPU |
|:---------|:------------|:----------|:---------------|:-------------|:----|
| LocalAI | 2 | 4 | 4 Gi | 8 Gi | 1 (time-sliced) |
| Ollama | 2 | 4 | 4 Gi | 8 Gi | 1 (time-sliced) |
| Qdrant | 500m | 1 | 1 Gi | 2 Gi | -- |
| AnythingLLM | 500m | 1 | 1 Gi | 2 Gi | -- |
| RAG Pipeline | 250m | 1 | 512 Mi | 2 Gi | -- |
| **Total** | **5.25** | **11** | **10.5 Gi** | **22 Gi** | **1 physical** |

---

## Deployment Methods Summary

| Component | Method | Tool | Notes |
|:----------|:-------|:-----|:------|
| Namespace | `kubectl apply` | kubectl | Simple YAML manifest |
| LocalAI | Helm install | helm | `go-skynet/local-ai` chart + custom values |
| Ollama | `kubectl apply` | kubectl | Raw manifests (Deployment + Service + PVC) |
| Qdrant | Helm install | helm | `qdrant/qdrant` chart + custom values |
| AnythingLLM | Helm install | helm | `anythingllm/anything-llm` chart + custom values |
| RAG Pipeline | `kubectl apply -k` | kustomize | Kustomization with ConfigMap, PVC, Deployment, Service |
