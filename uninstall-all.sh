#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="ai-stack"

echo "============================================"
echo "  AI Stack - Full Uninstall"
echo "============================================"
echo ""

read -p "This will remove ALL AI Stack components. Continue? (y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

echo ""

# RAG Pipeline
echo "[1/5] Removing RAG Pipeline..."
kubectl delete -k "$SCRIPT_DIR/rag-pipeline/" --ignore-not-found 2>/dev/null || true

# AnythingLLM
echo "[2/5] Removing AnythingLLM..."
helm uninstall anythingllm -n "$NAMESPACE" 2>/dev/null || true
kubectl delete -f "$SCRIPT_DIR/infrastructure/anythingllm/ingress.yaml" --ignore-not-found 2>/dev/null || true

# Ollama
echo "[3/5] Removing Ollama..."
kubectl delete -f "$SCRIPT_DIR/infrastructure/ollama/deployment.yaml" --ignore-not-found 2>/dev/null || true
kubectl delete -f "$SCRIPT_DIR/infrastructure/ollama/service.yaml" --ignore-not-found 2>/dev/null || true
kubectl delete -f "$SCRIPT_DIR/infrastructure/ollama/pvc.yaml" --ignore-not-found 2>/dev/null || true

# LocalAI
echo "[4/5] Removing LocalAI..."
helm uninstall localai -n "$NAMESPACE" 2>/dev/null || true
kubectl delete -f "$SCRIPT_DIR/infrastructure/localai/ingress.yaml" --ignore-not-found 2>/dev/null || true

# Qdrant
echo "[5/5] Removing Qdrant..."
helm uninstall qdrant -n "$NAMESPACE" 2>/dev/null || true

echo ""
read -p "Delete namespace '$NAMESPACE' and all PVCs? (y/N): " confirm_ns
if [[ "$confirm_ns" == "y" || "$confirm_ns" == "Y" ]]; then
  kubectl delete namespace "$NAMESPACE" --ignore-not-found
  echo "Namespace deleted."
else
  echo "Namespace preserved."
fi

echo ""
echo "============================================"
echo "  Uninstall Complete"
echo "============================================"
