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

# Shared SSH options. ConnectTimeout + BatchMode fail fast on a stale/dead IP
# instead of hanging the parallel deploy forever; ServerAliveInterval kills a
# session that blackholes *after* connecting (e.g. recycled instance now at
# this IP). Without these one unreachable worker stalls the whole `wait`.
#
# LogLevel=ERROR is REQUIRED, not cosmetic: newer OpenSSH clients (recent macOS)
# print a post-quantum key-exchange warning on the ssh channel. rsync uses that
# channel as its protocol pipe and wedges on any extraneous output, so without
# this the rsync step hangs forever with only the warning in the log.
# ServerAlive window is 30s × 10 = 5 min: tolerant of a freshly-booted,
# user-data-loaded instance that's slow to service SSH (60s was far too tight
# and killed rsync against still-initialising boxes), while a genuinely dead
# IP still bails — ConnectTimeout=10 handles the can't-connect-at-all case.
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o LogLevel=ERROR"

WORKERS=(
  "54.67.55.75"      # worker-0
  "18.145.23.246"    # worker-1
  "54.183.190.110"   # worker-2
  "13.56.247.223"    # worker-3
  "3.101.89.26"      # worker-4
  "18.144.49.65"     # worker-5
  "54.241.68.12"     # worker-6
  "52.53.243.136"    # worker-7
  "54.67.142.178"    # worker-8
  "54.241.54.87"     # worker-9
  "18.144.255.159"   # worker-10
  "54.183.184.64"    # worker-11
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
  #    -v so the log shows files flowing (deploy was opaque without it).
  #    Excludes mirror the large local-only / gitignored cruft the worker never
  #    needs. The killers historically:
  #      data/        — multi-GB local datasets (worker reads OpenSearch/S3)
  #      models/      — gitignored; weights pulled at runtime by open-clip / s-t
  #      .terraform/  — ~600 MB AWS provider binary PER terraform module dir
  #                     (×5 dirs ≈ 3 GB; 22k+ files). Worker never runs terraform.
  rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='.env' \
    --exclude='data' \
    --exclude='models' \
    --exclude='.terraform' \
    --exclude='*.tfstate' \
    --exclude='*.tfstate.backup' \
    --exclude='.terraform.lock.hcl' \
    --exclude='experiment/results' \
    --exclude='logs' \
    --exclude='*.log' \
    -e "ssh -i ${KEY} ${SSH_OPTS}" \
    ./ "ec2-user@${ip}:${REMOTE_DIR}/" >> "${log}" 2>&1

  # 2. Ensure venv exists and deps are current
  ssh -i "${KEY}" ${SSH_OPTS} "ec2-user@${ip}" bash <<REMOTE >> "${log}" 2>&1
    set -e
    cd ${REMOTE_DIR}
    if [ ! -f .venv/bin/python ]; then
      echo "Creating venv..."
      python3.11 -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip
    # No --quiet: the torch cold-install is ~15-20 min; without output the log
    # is silent the whole time and indistinguishable from a hang.
    pip install -r requirements.txt
REMOTE

  # 3. Write / update the os-backfill systemd service
  #    Restarts automatically on failure so spot interruptions recover cleanly.
  ssh -i "${KEY}" ${SSH_OPTS} "ec2-user@${ip}" bash <<REMOTE >> "${log}" 2>&1
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
# Nice + reduced CPU weight: the backfill saturates all vCPUs (image decode,
# CLIP preprocessing). Without this it starves sshd — SSH banner-exchange
# times out and the box becomes unmanageable, forcing a reboot to deploy.
# Niced, the worker still gets ~all idle CPU but instantly yields to sshd.
Nice=15
CPUWeight=40
# Memory limits: the worker holds a whole OS page of decoded images in RAM and
# was driving the 16 GB box into swap — swap-thrash makes the box unreachable
# (sshd can't fault its pages back; CPU looks idle ~26%). MemorySwapMax=0 +
# MemoryMax cap the worker's cgroup so it gets OOM-killed cleanly (then
# Restart=on-failure resumes it from checkpoint) instead of swap-killing the
# whole host. MemoryHigh applies reclaim pressure before the hard cap.
MemoryHigh=12G
MemoryMax=13G
MemorySwapMax=0
# No RuntimeMaxSec recycle: the per-page rss= trace showed a flat ~4 GB
# sawtooth (no leak), and a 45-min recycle killed workers mid-job (jobs run
# ~1.3 h) — jobs never completed. MemoryMax above is the real safety net: a
# genuine runaway gets OOM-killed + resumed from checkpoint, no recycle needed.
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
    # enable: WantedBy alone does nothing without the wants/ symlink, so the
    # service never survived a reboot / spot-stop before this. enable creates it.
    sudo systemctl enable os-backfill
REMOTE

  # 4. Start (or restart) the service
  ssh -i "${KEY}" ${SSH_OPTS} "ec2-user@${ip}" \
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

  # ── Dashboard on worker-0 (reproducible — no longer a hand-set-up snowflake) ──
  # Only when worker-0 was part of this deploy. Generates worker_ips.json from
  # the LIVE fleet (public from WORKERS, private collected per host) so the
  # on-w0 dashboard's intra-fleet SSH poll survives terraform recreates with
  # zero manual IP edits, then installs/starts os-dashboard.service.
  # BEST-EFFORT and NON-FATAL: the 12 workers are already deployed and started
  # by this point ("All … started" above). Dashboard setup must never flip the
  # deploy to failed — right after a deploy the boxes are loading CLIP onto the
  # GPU and an SSH banner-exchange can transiently time out under that load.
  # So: disable errexit here, retry the worker-0 step, and only warn on failure.
  dash=0; for t in "${TARGETS[@]}"; do [ "$t" = "0" ] && dash=1; done
  if [ "$dash" = "1" ]; then
    set +e
    W0="${WORKERS[0]}"
    echo "Setting up os-dashboard.service on worker-0 (${W0})... (best-effort)"

    # Collect private IPs; a failed host just gets an empty entry (tolerated).
    pub=""; priv=""
    for idx in "${!WORKERS[@]}"; do
      p=$(ssh -i "${KEY}" ${SSH_OPTS} "ec2-user@${WORKERS[$idx]}" \
            "hostname -I | awk '{print \$1}'" 2>/dev/null)
      pub="${pub}\"${WORKERS[$idx]}\","
      priv="${priv}\"${p}\","
    done
    pub="[${pub%,}]"; priv="[${priv%,}]"

    dash_ok=0
    for attempt in 1 2 3; do
      ssh -i "${KEY}" ${SSH_OPTS} "ec2-user@${W0}" bash <<REMOTE >/dev/null 2>&1
        set -e
        cat > ${REMOTE_DIR}/worker_ips.json <<JSON
{"public": ${pub}, "private": ${priv}}
JSON
        sudo tee /etc/systemd/system/os-dashboard.service > /dev/null <<SVCEOF
[Unit]
Description=Card Oracle backfill dashboard (worker-0)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=${REMOTE_DIR}
Environment=PATH=${REMOTE_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${REMOTE_DIR}/.venv/bin/python tools/backfill_dashboard.py --port 8080
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF
        sudo systemctl daemon-reload
        sudo systemctl enable os-dashboard >/dev/null 2>&1 || true
        sudo systemctl restart os-dashboard
REMOTE
      if [ $? -eq 0 ]; then dash_ok=1; break; fi
      echo "  dashboard setup attempt ${attempt} failed (box likely busy) — retrying in 20s..."
      sleep 20
    done
    if [ "$dash_ok" = "1" ]; then
      echo "  ✓ os-dashboard.service running → http://${W0}:8080"
    else
      echo "  ⚠ dashboard setup failed after 3 tries — WORKERS ARE FINE, this is non-fatal."
      echo "    Re-run just the dashboard later:  bash tools/deploy_os_backfill.sh 0"
      echo "    Or use the local dashboard:        python tools/backfill_dashboard.py"
    fi
    set -e
  else
    echo "(worker-0 not in targets — skipping os-dashboard setup)"
  fi
  echo ""
  echo "Monitor:"
  for idx in "${TARGETS[@]}"; do
    echo "  ssh -i ${KEY} ec2-user@${WORKERS[$idx]} 'sudo journalctl -fu os-backfill'  # worker-${idx}"
  done
  echo ""
  echo "Dashboard:"
  echo "  http://${WORKERS[0]}:8080            # on worker-0 (os-dashboard.service)"
  echo "  python tools/backfill_dashboard.py   # or run locally"
else
  echo "Some workers failed — see logs above."
  exit 1
fi
