#!/bin/bash
set -euo pipefail

# Deploy project code to a remote EC2 worker instance.
# Usage: ./deploy-to-worker.sh <worker-ip> [key-path]
#
# Examples:
#   ./deploy-to-worker.sh 13.52.251.41
#   ./deploy-to-worker.sh 13.52.251.41 ~/.ssh/qdrant-test.pem

WORKER_IP="${1:?Usage: deploy-to-worker.sh <worker-ip> [key-path]}"
KEY_PATH="${2:-$HOME/.ssh/qdrant-test.pem}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_DIR="/home/ec2-user/card-oracle-max"

echo "=== Deploying to $WORKER_IP ==="
echo "Project root: $PROJECT_ROOT"
echo "SSH key:      $KEY_PATH"

# Create zip excluding unnecessary files
TMPZIP="/tmp/card-oracle-max-deploy.zip"
echo "Creating deployment archive..."
cd "$PROJECT_ROOT"
zip -rq "$TMPZIP" . \
  -x ".venv/*" \
  -x ".git/*" \
  -x "infra/terraform/*/.terraform/*" \
  -x "infra/terraform/*/.venv/*" \
  -x "infra/terraform/*/terraform.tfstate*" \
  -x "infra/terraform/*/terraform.tfvars" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x ".env" \
  -x "experiment/results/*" \
  -x "*.zip"

SIZE=$(du -h "$TMPZIP" | cut -f1)
echo "Archive size: $SIZE"

# Transfer
echo "Uploading to $WORKER_IP..."
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no "$TMPZIP" "ec2-user@$WORKER_IP:/tmp/deploy.zip"

# Unzip on remote
echo "Extracting on remote..."
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "ec2-user@$WORKER_IP" bash -s <<'REMOTE'
set -euo pipefail
mkdir -p ~/card-oracle-max
cd ~/card-oracle-max
unzip -oq /tmp/deploy.zip
rm /tmp/deploy.zip
echo "Files extracted to ~/card-oracle-max"
ls -la
REMOTE

# Clean up local temp
rm "$TMPZIP"

echo ""
echo "=== Deployment complete ==="
echo "SSH in:  ssh -i $KEY_PATH ec2-user@$WORKER_IP"
echo "Setup:   cd ~/card-oracle-max && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
echo "Run:     python -m src.embeddings.batch_job --index-pattern 2025-02-19 --image-device cuda"
