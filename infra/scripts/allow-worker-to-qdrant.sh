#!/bin/bash
set -euo pipefail

# Add a GPU worker's IP to the Qdrant security group so it can reach port 6333.
# Run this AFTER provisioning the GPU worker.
#
# Usage: ./allow-worker-to-qdrant.sh <worker-ip> <qdrant-sg-id> [region]
#
# Example:
#   ./allow-worker-to-qdrant.sh 54.183.100.50 sg-051380f67244de852

WORKER_IP="${1:?Usage: allow-worker-to-qdrant.sh <worker-ip> <qdrant-sg-id> [region]}"
QDRANT_SG="${2:?Provide the Qdrant security group ID}"
REGION="${3:-us-west-1}"

echo "Adding $WORKER_IP/32 to Qdrant SG $QDRANT_SG for port 6333..."

aws ec2 authorize-security-group-ingress \
  --region "$REGION" \
  --group-id "$QDRANT_SG" \
  --protocol tcp \
  --port 6333 \
  --cidr "${WORKER_IP}/32" \
  --tag-specifications "ResourceType=security-group-rule,Tags=[{Key=Name,Value=gpu-worker-qdrant-access}]"

echo "Done. GPU worker at $WORKER_IP can now reach Qdrant on port 6333."
