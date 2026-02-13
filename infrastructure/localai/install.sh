#!/bin/bash
set -euo pipefail

NAMESPACE="ai-stack"
RELEASE_NAME="localai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing LocalAI ==="

# Add Helm repo
helm repo add go-skynet https://go-skynet.github.io/helm-charts/ 2>/dev/null || true
helm repo update

# Install or upgrade
helm upgrade --install "$RELEASE_NAME" go-skynet/local-ai \
  --namespace "$NAMESPACE" \
  --create-namespace \
  --values "$SCRIPT_DIR/values.yaml" \
  --wait \
  --timeout 10m

# Apply ingress
kubectl apply -f "$SCRIPT_DIR/ingress.yaml"

echo "=== LocalAI installed successfully ==="
echo "Internal URL: http://localai.${NAMESPACE}.svc.cluster.local:8080"
echo "API endpoint: http://localai.${NAMESPACE}.svc.cluster.local:8080/v1"
