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
echo "[1/8] Creating namespace..."
kubectl apply -f "$SCRIPT_DIR/infrastructure/namespace/namespace.yaml"
echo ""

# Step 2: Resource Quota & LimitRange
echo "[2/8] Applying resource quota and limit range..."
kubectl apply -f "$SCRIPT_DIR/infrastructure/namespace/resource-quota.yaml"
echo ""

# Step 3: Network Policies
echo "[3/8] Applying network policies..."
kubectl apply -f "$SCRIPT_DIR/infrastructure/network-policies/"
echo ""

# Step 4: Qdrant (no GPU dependency, starts fast)
echo "[4/8] Installing Qdrant..."
bash "$SCRIPT_DIR/infrastructure/qdrant/install.sh"
echo ""

# Step 5: LocalAI
echo "[5/8] Installing LocalAI..."
bash "$SCRIPT_DIR/infrastructure/localai/install.sh"
echo ""

# Step 6: Ollama
echo "[6/8] Installing Ollama..."
kubectl apply -f "$SCRIPT_DIR/infrastructure/ollama/pvc.yaml"
kubectl apply -f "$SCRIPT_DIR/infrastructure/ollama/deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/infrastructure/ollama/service.yaml"
echo "Waiting for Ollama to be ready..."
kubectl rollout status deployment/ollama -n "$NAMESPACE" --timeout=10m
echo ""

# Step 7: AnythingLLM
echo "[7/8] Installing AnythingLLM..."
bash "$SCRIPT_DIR/infrastructure/anythingllm/install.sh"
echo ""

# Step 8: RAG Pipeline
echo "[8/8] Deploying RAG Pipeline..."
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
