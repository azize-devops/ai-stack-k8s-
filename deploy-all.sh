#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="ai-stack"

echo "============================================"
echo "  AI Stack - Full Deployment"
echo "  Components: LocalAI, Ollama, Qdrant,"
echo "  AnythingLLM, RAG Pipeline"
echo "============================================"
echo ""

# Step 1: Namespace
echo "[1/6] Creating namespace..."
kubectl apply -f "$SCRIPT_DIR/infrastructure/namespace/namespace.yaml"
echo ""

# Step 2: Qdrant (no GPU dependency, starts fast)
echo "[2/6] Installing Qdrant..."
bash "$SCRIPT_DIR/infrastructure/qdrant/install.sh"
echo ""

# Step 3: LocalAI
echo "[3/6] Installing LocalAI..."
bash "$SCRIPT_DIR/infrastructure/localai/install.sh"
echo ""

# Step 4: Ollama
echo "[4/6] Installing Ollama..."
kubectl apply -f "$SCRIPT_DIR/infrastructure/ollama/pvc.yaml"
kubectl apply -f "$SCRIPT_DIR/infrastructure/ollama/deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/infrastructure/ollama/service.yaml"
echo "Waiting for Ollama to be ready..."
kubectl rollout status deployment/ollama -n "$NAMESPACE" --timeout=10m
echo ""

# Step 5: AnythingLLM
echo "[5/6] Installing AnythingLLM..."
bash "$SCRIPT_DIR/infrastructure/anythingllm/install.sh"
echo ""

# Step 6: RAG Pipeline
echo "[6/6] Deploying RAG Pipeline..."
kubectl apply -k "$SCRIPT_DIR/rag-pipeline/"
kubectl rollout status deployment/rag-pipeline -n "$NAMESPACE" --timeout=5m
echo ""

echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "Services:"
echo "  LocalAI:     http://localai.${NAMESPACE}.svc.cluster.local:8080"
echo "  Ollama:      http://ollama.${NAMESPACE}.svc.cluster.local:11434"
echo "  Qdrant:      http://qdrant.${NAMESPACE}.svc.cluster.local:6333"
echo "  AnythingLLM: http://anythingllm.${NAMESPACE}.svc.cluster.local:3001"
echo "  RAG API:     http://rag-pipeline.${NAMESPACE}.svc.cluster.local:8000"
echo ""
echo "Verify: kubectl get pods -n $NAMESPACE"
