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

# ── 3. Restart the search worker service (squatter-safe) ─────────────────────
# A stale gunicorn started OUTSIDE the current systemd unit (e.g. a hand-run
# process from early setup) will keep holding port 8081. `systemctl restart`
# then spawns a new process that dies on EADDRINUSE — silently, because the
# old process still answers /health. Result: deploys rsync new code but the
# old code keeps serving for weeks. Guard against it: after restart, assert
# systemd's MainPID is the process actually bound to 8081; if not, kill every
# gunicorn for this app and restart cleanly, then re-assert.
echo "→ Restarting gpu-search-worker service (with squatter check)..."
ssh -i "${KEY}" -o StrictHostKeyChecking=no "${REMOTE}" bash <<'REMOTE_CMDS'
  set -euo pipefail
  PORT=8081

  port_pid() {
    # PID currently listening on $PORT, or empty.
    sudo ss -tlnpH "sport = :${PORT}" 2>/dev/null \
      | grep -oP 'pid=\K[0-9]+' | head -1
  }
  main_pid() {
    systemctl show gpu-search-worker -p MainPID --value
  }

  echo "  restarting..."
  sudo systemctl restart gpu-search-worker
  sleep 4

  MP="$(main_pid)"
  LP="$(port_pid || true)"
  echo "  MainPID=${MP:-0}  port-${PORT}-owner=${LP:-none}"

  if [[ -z "${MP}" || "${MP}" == "0" || "${MP}" != "${LP}" ]]; then
    echo "  ✗ systemd does not own port ${PORT} — a squatter is holding it."
    echo "  killing all gunicorn for tools.gpu_worker_server:app ..."
    sudo systemctl stop gpu-search-worker || true
    sudo pkill -f 'tools.gpu_worker_server:app' || true
    sleep 3
    # Hard-fail if anything still holds the port
    if [[ -n "$(port_pid || true)" ]]; then
      echo "  ✗ port ${PORT} STILL held after pkill — manual intervention needed:"
      sudo ss -tlnp "sport = :${PORT}"
      exit 1
    fi
    echo "  port clear — starting cleanly via systemd..."
    sudo systemctl start gpu-search-worker
    sleep 4
    MP="$(main_pid)"
    LP="$(port_pid || true)"
    echo "  MainPID=${MP:-0}  port-${PORT}-owner=${LP:-none}"
  fi

  if [[ -z "${MP}" || "${MP}" == "0" || "${MP}" != "${LP}" ]]; then
    echo "  ✗ FAILED: systemd MainPID (${MP:-0}) is not the process bound to ${PORT} (${LP:-none})."
    sudo systemctl status gpu-search-worker --no-pager | head -10
    exit 1
  fi
  echo "  ✓ systemd MainPID ${MP} owns port ${PORT}."
  sudo systemctl status gpu-search-worker --no-pager | head -6
REMOTE_CMDS

# ── 4. Health check + code-freshness assertion ───────────────────────────────
# /health alone can't tell new code from old — it returns the same shape either
# way. /search echoes back the request's top_k (a field that ONLY exists in the
# post-2026-06 build), so a tiny top_k probe proves the running process is the
# code we just shipped, not a survivor.
echo "→ Health check + code-freshness probe (direct to port 8081)..."
sleep 5   # give Gunicorn + CLIP a moment to load
ssh -i "${KEY}" -o StrictHostKeyChecking=no "${REMOTE}" bash <<'REMOTE_CMDS' || \
  echo "WARNING: Health/freshness check failed — check: ssh -i ${KEY} ${REMOTE} 'sudo journalctl -fu gpu-search-worker'"
  set -euo pipefail
  PORT=8081
  echo "  /health:"
  curl -sf "http://localhost:${PORT}/health" | python3 -m json.tool

  # Freshness probe: send a search with an explicit top_k and confirm the
  # response echoes it back. We don't care about results — a 422 (bad image)
  # still returns before search, so use a tiny dummy and just check the field
  # is understood. Use a harmless 1x1 transparent PNG.
  PNG='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
  echo "  /search_b64 top_k echo probe:"
  RESP="$(curl -s -X POST "http://localhost:${PORT}/search_b64" \
    -H 'content-type: application/json' \
    -d "{\"image_b64\":\"${PNG}\",\"top_k\":37}")"
  ECHOED="$(echo "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("top_k"))' 2>/dev/null || echo "MISSING")"
  if [[ "$ECHOED" == "37" ]]; then
    echo "  ✓ running code echoes top_k=37 — patched build confirmed live."
  else
    echo "  ✗ top_k not echoed (got: ${ECHOED}). OLD CODE IS SERVING — deploy did not take effect."
    echo "    response head: $(echo "$RESP" | head -c 200)"
    exit 1
  fi
REMOTE_CMDS

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
