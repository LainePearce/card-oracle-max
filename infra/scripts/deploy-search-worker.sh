#!/bin/bash
# Deploy code to the GPU search worker instance and restart the service.
#
# Usage:
#   ./infra/scripts/deploy-search-worker.sh <instance-public-ip>
#   ./infra/scripts/deploy-search-worker.sh <instance-public-ip> --skip-pip
#
# The ALB URL is printed by terraform output after `terraform apply`.
# Set GPU_WORKER_URL=http://<alb-dns> in your local .env to route searches through it.

set -euo pipefail

INSTANCE_IP="${1:-}"
SKIP_PIP="${2:-}"
KEY="${SSH_KEY:-$HOME/.ssh/qdrant-test.pem}"
REMOTE="ec2-user@${INSTANCE_IP}"
REMOTE_DIR="/home/ec2-user/card-oracle-max"

if [[ -z "$INSTANCE_IP" ]]; then
  echo "Usage: $0 <instance-public-ip> [--skip-pip]"
  echo ""
  echo "Get the IP from: cd infra/terraform/gpu-worker-search && terraform output"
  exit 1
fi

echo "=== Deploying to GPU search worker at ${INSTANCE_IP} ==="

# ── 1. Push code ──────────────────────────────────────────────────────────────
echo "→ Syncing code..."
rsync -av --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='experiment/results/' \
  --exclude='*.parquet' \
  --exclude='.env' \
  -e "ssh -i ${KEY} -o StrictHostKeyChecking=no" \
  ./ "${REMOTE}:${REMOTE_DIR}/"

# ── 2. Install/update Python dependencies ────────────────────────────────────
if [[ "$SKIP_PIP" != "--skip-pip" ]]; then
  echo "→ Installing Python dependencies..."
  ssh -i "${KEY}" -o StrictHostKeyChecking=no "${REMOTE}" bash <<'REMOTE_CMDS'
    set -euo pipefail
    cd /home/ec2-user/card-oracle-max
    if [[ ! -d .venv ]]; then
      echo "Creating virtualenv..."
      python3.11 -m venv .venv
    fi
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    echo "Dependencies installed."
REMOTE_CMDS
else
  echo "→ Skipping pip install (--skip-pip set)"
fi

# ── 3. Restart the search worker service ─────────────────────────────────────
echo "→ Restarting gpu-search-worker service..."
ssh -i "${KEY}" -o StrictHostKeyChecking=no "${REMOTE}" \
  "sudo systemctl restart gpu-search-worker && sleep 3 && sudo systemctl status gpu-search-worker --no-pager"

# ── 4. Health check ───────────────────────────────────────────────────────────
echo "→ Health check (direct to instance port 8081)..."
sleep 5   # give Gunicorn + CLIP a moment to load
ssh -i "${KEY}" -o StrictHostKeyChecking=no "${REMOTE}" \
  "curl -sf http://localhost:8081/health | python3 -m json.tool" || \
  echo "WARNING: Health check failed — check logs with: ssh -i ${KEY} ${REMOTE} 'sudo journalctl -fu gpu-search-worker'"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "ALB health check (may take ~30s for ALB to register healthy):"
echo "  curl http://\$(cd infra/terraform/gpu-worker-search && terraform output -raw alb_dns)/health"
echo ""
echo "Stream logs:"
echo "  ssh -i ${KEY} ${REMOTE} 'sudo journalctl -fu gpu-search-worker'"
echo ""
echo "Set GPU_WORKER_URL in local .env:"
echo "  GPU_WORKER_URL=http://\$(cd infra/terraform/gpu-worker-search && terraform output -raw alb_dns)"
