#!/bin/bash
# Deploy the DINOv2 embed worker to the GPU fleet, hardened so a box can never
# re-trigger the "CLIP backfill + dino-embed fighting on a no-swap box" incident.
#
# Idempotent — safe to re-run after code changes or on a fresh/replacement box.
# Run from the operator machine in the repo root with the GPU public IPs:
#
#   KEY=~/.ssh/qdrant-test.pem ./infra/scripts/deploy-dino-embed-fleet.sh 1.2.3.4 5.6.7.8 ...
#
# Per box it: rsyncs code (never .env/keys/tfstate), ensures S3_IMAGE_BUCKET +
# transformers, adds an 8GB swapfile (safety net against OOM-freezing sshd),
# MASKS os-backfill/os-orchestrator (the retired CLIP pipeline — mask, not just
# disable, so nothing can start them), and installs + starts dino-embed.
#
# Note: the embed dashboard + seed timer install on GPU worker-0 only — that's
# handled separately, not in this fleet loop.
set -euo pipefail

KEY="${KEY:-$HOME/.ssh/qdrant-test.pem}"
REMOTE=/home/ec2-user/card-oracle-max
SWAP_GB="${SWAP_GB:-8}"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=12)

if [ "$#" -lt 1 ]; then
  echo "usage: KEY=~/.ssh/key.pem $0 <gpu-public-ip> [gpu-public-ip ...]" >&2
  exit 1
fi

for ip in "$@"; do
  echo "=== $ip ==="
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='data' \
    --exclude='.env' --exclude='.terraform' --exclude='*.tfvars' --exclude='*.pem' \
    --exclude='worker_ips.json' --exclude='*.pyc' \
    ./ "ec2-user@$ip:$REMOTE/"

  ssh "${SSH_OPTS[@]}" "ec2-user@$ip" "
    set -e
    cd $REMOTE
    grep -q S3_IMAGE_BUCKET .env || echo 'S3_IMAGE_BUCKET=images-130-sold' >> .env
    .venv/bin/pip install -q 'transformers>=4.45'

    # Swap safety net — a spike pages instead of OOM-freezing sshd.
    if [ ! -f /swapfile ]; then
      sudo dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB * 1024)) status=none
      sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
      echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    fi

    # Retire the CLIP pipeline: mask (not just disable) so it can never start
    # and contend with dino-embed for the single GPU / vCPUs / RAM.
    sudo systemctl disable --now os-backfill os-orchestrator os-dashboard 2>/dev/null || true
    sudo systemctl mask os-backfill os-orchestrator 2>/dev/null || true

    sudo cp infra/systemd/dino-embed.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now dino-embed
    echo -n 'dino-embed: '; systemctl is-active dino-embed
    free -h | grep Swap
  "
done

echo
echo "Deployed dino-embed (swap + CLIP services masked)."
echo "Dashboard + seed timer install on GPU worker-0 only:"
echo "  sudo cp infra/systemd/dino-embed-dashboard.service infra/systemd/dino-embed-seed.{service,timer} /etc/systemd/system/"
echo "  sudo systemctl daemon-reload && sudo systemctl enable --now dino-embed-dashboard dino-embed-seed.timer"
