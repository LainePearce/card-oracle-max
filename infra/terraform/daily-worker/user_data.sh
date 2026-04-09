#!/bin/bash
set -euo pipefail
exec > /var/log/user-data.log 2>&1

echo "=== Daily embedding worker bootstrap ==="

# --- 1. System dependencies ---
echo "Installing system packages..."
dnf install -y python3.11 python3.11-pip python3.11-devel git gcc gcc-c++ unzip

# --- 2. Verify GPU ---
echo "Checking GPU..."
nvidia-smi || echo "WARNING: nvidia-smi not ready — drivers are pre-installed on AMI"

# --- 3. Working directory ---
mkdir -p /home/ec2-user/card-oracle-max/logs
chown -R ec2-user:ec2-user /home/ec2-user/card-oracle-max

# --- 4. Write .env ---
cat > /home/ec2-user/card-oracle-max/.env <<ENVEOF
# Qdrant
QDRANT_HOST=${qdrant_host}
QDRANT_PORT=${qdrant_port}
QDRANT_API_KEY=${qdrant_api_key}
QDRANT_USE_GRPC=true
QDRANT_COLLECTION=cards
QDRANT_SINGLE_NODE=false

# RDS primary
RDS_HOST=${rds_host}
RDS_PORT=3306
RDS_USER=${rds_user}
RDS_PASSWORD=${rds_password}
RDS_DATABASE=${rds_database}

# RDS secondary (empty = disabled)
RDS2_HOST=${rds2_host}
RDS2_PORT=3306
RDS2_USER=${rds2_user}
RDS2_PASSWORD=${rds2_password}
RDS2_DATABASE=${rds2_database}

# S3 vector store
S3_VECTOR_BUCKET=${s3_bucket}
S3_VECTOR_PREFIX=${s3_prefix}
S3_CHECKPOINT_PREFIX=checkpoints

# Pipeline
DEVELOPMENT_STAGE=1
AWS_REGION=us-west-1
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE_IMAGE=256
IMAGE_DOWNLOAD_WORKERS=16

# Daily update config
DAILY_LOOKBACK_DAYS=${lookback_days}
ENVEOF

chown ec2-user:ec2-user /home/ec2-user/card-oracle-max/.env
chmod 600 /home/ec2-user/card-oracle-max/.env

# --- 5. Systemd service (oneshot — runs to completion then exits) ---
cat > /etc/systemd/system/daily-update.service <<SVCEOF
[Unit]
Description=Card Oracle daily embedding update (${lookback_days}-day lookback)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ec2-user
WorkingDirectory=/home/ec2-user/card-oracle-max
Environment=PATH=/home/ec2-user/card-oracle-max/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ec2-user/card-oracle-max/.venv/bin/python tools/daily_update.py \
    --lookback-days ${lookback_days}
StandardOutput=journal
StandardError=journal

# If the job fails, surface it in the journal but let the timer retry tomorrow.
# For alerting wire OnFailure= to a notify unit if needed.

[Install]
WantedBy=multi-user.target
SVCEOF

# --- 6. Systemd timer (02:00 UTC daily) ---
# Persistent=true means if the instance was off at 02:00 it catches up on next boot.
cat > /etc/systemd/system/daily-update.timer <<TIMEREOF
[Unit]
Description=Run Card Oracle daily embedding update at 02:00 UTC

[Timer]
OnCalendar=*-*-* 02:00:00 UTC
Persistent=true
Unit=daily-update.service

[Install]
WantedBy=timers.target
TIMEREOF

systemctl daemon-reload

# Timer is enabled (auto-starts on boot) but NOT triggered now.
# Code must be deployed and venv installed before the first run.
# Enable is done here so after code deploy + venv setup it will fire automatically.
systemctl enable daily-update.timer

echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. Deploy code:    ./tools/deploy_daily_worker.sh <this-ip>"
echo "  2. Check timer:    ssh ec2-user@<ip> 'sudo systemctl status daily-update.timer'"
echo "  3. Manual test:    ssh ec2-user@<ip> 'sudo systemctl start daily-update.service'"
echo "  4. Follow logs:    ssh ec2-user@<ip> 'sudo journalctl -fu daily-update.service'"
echo "  5. Timer fires:    daily at 02:00 UTC automatically"
