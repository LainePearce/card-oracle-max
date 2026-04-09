#!/usr/bin/env bash
# Deploy code to the daily GPU worker and start the timer.
#
# Usage:
#   ./tools/deploy_daily_worker.sh <worker-ip>
#   ./tools/deploy_daily_worker.sh $(cd infra/terraform/daily-worker && terraform output -raw public_ip)
#
# Prerequisites:
#   - Instance is running; private key matches Terraform key_pair_name (set DAILY_WORKER_SSH_KEY if not in ssh-agent)
#   - Terraform has been applied (user_data bootstrap complete — check /var/log/user-data.log)
set -euo pipefail

WORKER_IP="${1:-}"
if [[ -z "$WORKER_IP" ]]; then
    echo "Usage: $0 <worker-ip>"
    echo "       $0 \$(cd infra/terraform/daily-worker && terraform output -raw public_ip)"
    exit 1
fi

SSH_USER="ec2-user"
# Optional: path to .pem for the EC2 key pair (must match terraform key_pair_name), e.g. ~/.ssh/qdrant-test.pem
SSH_ARGS=( -o StrictHostKeyChecking=no -o ConnectTimeout=10 )
[[ -n "${DAILY_WORKER_SSH_KEY:-}" ]] && SSH_ARGS+=( -i "$DAILY_WORKER_SSH_KEY" )
# rsync -e must be one argv; paths without spaces are fine with *
RSYNC_RSH="ssh ${SSH_ARGS[*]}"
REMOTE="$SSH_USER@$WORKER_IP"
REMOTE_DIR="/home/ec2-user/card-oracle-max"

echo "=== Deploying daily worker to $WORKER_IP ==="

# --- 1. Sync code ---
echo "[1/5] Syncing code..."
rsync -av --progress \
    -e "$RSYNC_RSH" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='.env' \
    --exclude='logs/' \
    --exclude='experiment/results/' \
    ./ "$REMOTE:$REMOTE_DIR/"

# --- 2. Create / update virtualenv ---
echo "[2/5] Installing Python dependencies..."
ssh "${SSH_ARGS[@]}" "$REMOTE" bash <<'REMOTE_SCRIPT'
set -euo pipefail
cd ~/card-oracle-max

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    python3.11 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "Dependencies installed."
REMOTE_SCRIPT

# --- 3. Reload systemd (picks up any service/timer file changes) ---
echo "[3/5] Reloading systemd..."
ssh "${SSH_ARGS[@]}" "$REMOTE" "sudo systemctl daemon-reload"

# --- 4. Ensure timer is enabled and running ---
echo "[4/5] Enabling daily-update.timer..."
ssh "${SSH_ARGS[@]}" "$REMOTE" bash <<'REMOTE_SCRIPT'
sudo systemctl enable daily-update.timer
sudo systemctl start  daily-update.timer
sudo systemctl status daily-update.timer --no-pager
REMOTE_SCRIPT

# --- 5. Verify with a dry-run test ---
echo "[5/5] Running dry-run sanity check (encode only, no writes)..."
ssh "${SSH_ARGS[@]}" "$REMOTE" bash <<'REMOTE_SCRIPT'
set -euo pipefail
cd ~/card-oracle-max
source .venv/bin/activate
python tools/daily_update.py --lookback-days 1 --dry-run
echo "Dry-run passed."
REMOTE_SCRIPT

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Timer status:"
ssh "${SSH_ARGS[@]}" "$REMOTE" "sudo systemctl list-timers daily-update.timer --no-pager"
echo ""
echo "Next scheduled run is shown above."
echo "To trigger immediately:  ssh $REMOTE 'sudo systemctl start daily-update.service'"
echo "To follow logs:          ssh $REMOTE 'sudo journalctl -fu daily-update.service'"
echo "To check completion:     aws s3 ls s3://\$S3_VECTOR_BUCKET/checkpoints/daily- --recursive"
