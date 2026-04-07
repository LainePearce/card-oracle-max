#!/bin/bash
# Deploy code to all 12 GPU backfill workers and (re)start the backfill service.
# Run from the repo root: bash tools/deploy_backfill.sh
set -euo pipefail

KEY=~/.ssh/qdrant-test.pem
WORKERS=(
  "54.176.253.45"    # worker-0
  "204.236.180.247"  # worker-1
  "54.219.84.133"    # worker-2
  "13.56.115.249"    # worker-3
  "13.56.212.61"     # worker-4
  "13.57.218.120"    # worker-5
  "54.176.134.82"    # worker-6
  "13.56.151.99"     # worker-7
  "18.144.47.150"    # worker-8
  "3.101.102.118"    # worker-9
  "184.72.25.240"    # worker-10
  "13.56.139.224"    # worker-11
)
REMOTE_DIR="/home/ec2-user/card-oracle-max"

deploy_worker() {
  local idx=$1
  local ip=$2
  local log="/tmp/deploy_worker${idx}.log"

  echo "[worker-${idx}] deploying to ${ip}..."

  # 1. Sync code (never overwrite .env — it has RDS/Qdrant credentials)
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

  # 2. Create venv and install deps if needed
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

  # 3. Rewrite the systemd service to call the multi-phase orchestrator
  ssh -i "${KEY}" -o StrictHostKeyChecking=no "ec2-user@${ip}" bash <<REMOTE >> "${log}" 2>&1
    set -e
    sudo tee /etc/systemd/system/backfill.service > /dev/null <<SVCEOF
[Unit]
Description=Card Oracle RDS embedding backfill (worker ${idx})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=${REMOTE_DIR}
Environment=PATH=${REMOTE_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${REMOTE_DIR}/.venv/bin/python tools/worker_phases.py --worker-index ${idx}
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF
    sudo systemctl daemon-reload
REMOTE

  # 4. Ensure IMAGE_DOWNLOAD_WORKERS=32 is set in .env (add if missing, update if wrong)
  ssh -i "${KEY}" -o StrictHostKeyChecking=no "ec2-user@${ip}" bash <<REMOTE >> "${log}" 2>&1
    set -e
    if grep -q '^IMAGE_DOWNLOAD_WORKERS=' ${REMOTE_DIR}/.env; then
      sed -i 's/^IMAGE_DOWNLOAD_WORKERS=.*/IMAGE_DOWNLOAD_WORKERS=32/' ${REMOTE_DIR}/.env
    else
      echo 'IMAGE_DOWNLOAD_WORKERS=32' >> ${REMOTE_DIR}/.env
    fi
REMOTE

  # 5. Start the backfill service
  ssh -i "${KEY}" -o StrictHostKeyChecking=no "ec2-user@${ip}" \
    "sudo systemctl restart backfill" >> "${log}" 2>&1

  echo "[worker-${idx}] started. Log: ${log}"
}

# Deploy all workers in parallel
pids=()
for i in "${!WORKERS[@]}"; do
  deploy_worker "$i" "${WORKERS[$i]}" &
  pids+=($!)
done

# Wait for all and report
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[worker-${i}] OK"
  else
    echo "[worker-${i}] FAILED -- check /tmp/deploy_worker${i}.log"
    failed=1
  fi
done

if [ $failed -eq 0 ]; then
  echo ""
  echo "All 12 workers deployed. Monitor with:"
  for i in "${!WORKERS[@]}"; do
    echo "  ssh -i ${KEY} ec2-user@${WORKERS[$i]} 'sudo journalctl -fu backfill'  # worker-${i}"
  done
else
  exit 1
fi
