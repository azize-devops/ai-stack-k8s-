#!/bin/bash
set -euo pipefail

NAMESPACE="ai-stack"
RELEASE_NAME="qdrant"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing Qdrant Vector Database ==="

# Add Helm repo
helm repo add qdrant https://qdrant.github.io/qdrant-helm 2>/dev/null || true
helm repo update

# Install or upgrade
helm upgrade --install "$RELEASE_NAME" qdrant/qdrant \
  --namespace "$NAMESPACE" \
  --create-namespace \
  --values "$SCRIPT_DIR/values.yaml" \
  --wait \
  --timeout 5m

echo "=== Qdrant installed successfully ==="
echo "HTTP endpoint: http://qdrant.${NAMESPACE}.svc.cluster.local:6333"
echo "gRPC endpoint: http://qdrant.${NAMESPACE}.svc.cluster.local:6334"
