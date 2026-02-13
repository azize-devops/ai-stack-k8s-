#!/bin/bash
set -euo pipefail

NAMESPACE="ai-stack"
RELEASE_NAME="anythingllm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing AnythingLLM ==="

# Add Helm repo
helm repo add anythingllm https://mintplex-labs.github.io/anything-llm-helm/ 2>/dev/null || true
helm repo update

# Install or upgrade
helm upgrade --install "$RELEASE_NAME" anythingllm/anything-llm \
  --namespace "$NAMESPACE" \
  --create-namespace \
  --values "$SCRIPT_DIR/values.yaml" \
  --wait \
  --timeout 5m

# Apply ingress
kubectl apply -f "$SCRIPT_DIR/ingress.yaml"

echo "=== AnythingLLM installed successfully ==="
echo "Internal URL: http://anythingllm.${NAMESPACE}.svc.cluster.local:3001"
