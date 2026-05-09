#!/bin/bash
# Deploy the OpenSearch-queue backfill to all 12 GPU workers.
#
# Each worker runs tools/backfill_from_opensearch.py as a systemd service.
# Workers claim jobs from s3://card-oracle-vectors/backfill-v2/queue/ dynamically —
# no worker-index coordination is needed.
#
# Safe to re-run: already-running workers are restarted with latest code.
#
# Usage (from repo root):
#   bash tools/deploy_os_backfill.sh              # deploy + start all 12
#   bash tools/deploy_os_backfill.sh 3 4 5        # deploy + start specific workers only
#
# Monitor:
#   ssh -i ~/.ssh/qdrant-test.pem ec2-user@<ip> 'sudo journalctl -fu os-backfill'

set -euo pipefail

KEY=~/.ssh/qdrant-test.pem
WORKERS=(
  "54.215.57.135"    # worker-0
  "52.53.33.218"     # worker-1
  "3.101.66.241"     # worker-2
  "13.57.215.67"     # worker-3
  "18.145.69.81"     # worker-4
  "18.145.116.26"    # worker-5
  "54.153.95.19"     # worker-6
  "50.18.237.248"    # worker-7
  "13.57.176.92"     # worker-8
  "54.219.76.40"     # worker-9
  "13.56.140.67"     # worker-10
  "54.176.9.174"     # worker-11
)
REMOTE_DIR="/home/ec2-user/card-oracle-max"

# If specific indices provided use them, otherwise deploy all
if [ $# -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("${!WORKERS[@]}")
fi

deploy_worker() {
  local idx=$1
  local ip="${WORKERS[$idx]}"
  local log="/tmp/os_backfill_worker${idx}.log"

  echo "[worker-${idx}] deploying to ${ip}..."

  # 1. Sync latest code (skip .env — credentials already written by Terraform user_data)
  rsync -az --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='.env' \
    --exclude='experiment/results' \
    --exclude='logs' \
    --exclude='*.log' \
    -e "ssh -i ${KEY} -o StrictHostKeyChecking=no" \
    ./ "ec2-user@${ip}:${REMOTE_DIR}/" >> "${log}" 2>&1

  # 2. Ensure venv exists and deps are current
  ssh -i "${KEY}" -o StrictHostKeyChecking=no "ec2-user@${ip}" bash <<REMOTE >> "${log}" 2>&1
    set -e
    cd ${REMOTE_DIR}
    if [ ! -f .venv/bin/python ]; then
      echo "Creating venv..."
      python3.11 -m venv .venv
    fi
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
REMOTE

  # 3. Write / update the os-backfill systemd service
  #    Restarts automatically on failure so spot interruptions recover cleanly.
  ssh -i "${KEY}" -o StrictHostKeyChecking=no "ec2-user@${ip}" bash <<REMOTE >> "${log}" 2>&1
    set -e
    sudo tee /etc/systemd/system/os-backfill.service > /dev/null <<SVCEOF
[Unit]
Description=Card Oracle OpenSearch queue backfill (worker ${idx})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=${REMOTE_DIR}
Environment=PATH=${REMOTE_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${REMOTE_DIR}/.venv/bin/python tools/backfill_from_opensearch.py
Restart=on-failure
RestartSec=30
# Exit cleanly when queue is empty (exit 0) → no restart
SuccessExitStatus=0
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF
    sudo systemctl daemon-reload
REMOTE

  # 4. Start (or restart) the service
  ssh -i "${KEY}" -o StrictHostKeyChecking=no "ec2-user@${ip}" \
    "sudo systemctl restart os-backfill" >> "${log}" 2>&1

  echo "[worker-${idx}] ✓ started. Tail: sudo journalctl -fu os-backfill  (${ip})"
}

# Deploy targets in parallel
pids=()
for idx in "${TARGETS[@]}"; do
  deploy_worker "$idx" &
  pids+=($!)
done

# Collect results
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    : # success already printed inside deploy_worker
  else
    echo "[worker-${TARGETS[$i]}] FAILED — check /tmp/os_backfill_worker${TARGETS[$i]}.log"
    failed=1
  fi
done

echo ""
if [ $failed -eq 0 ]; then
  echo "All ${#TARGETS[@]} worker(s) started."
  echo ""
  echo "Monitor:"
  for idx in "${TARGETS[@]}"; do
    echo "  ssh -i ${KEY} ec2-user@${WORKERS[$idx]} 'sudo journalctl -fu os-backfill'  # worker-${idx}"
  done
  echo ""
  echo "Queue status:"
  echo "  python tools/backfill_dashboard.py"
else
  echo "Some workers failed — see logs above."
  exit 1
fi
