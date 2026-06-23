#!/bin/bash
set -euo pipefail
exec > /var/log/user-data.log 2>&1

echo "=== image-archive CPU worker bootstrap (worker ${worker_index}) ==="

# --- 1. System deps (no CUDA/NVIDIA — this is CPU/IO only) ---
dnf install -y python3.11 python3.11-pip python3.11-devel git gcc gcc-c++ \
  libjpeg-turbo-devel zlib-devel unzip

# --- 2. Working dir ---
mkdir -p /home/ec2-user/card-oracle-max/logs
chown -R ec2-user:ec2-user /home/ec2-user/card-oracle-max

# --- 3. Minimal .env (image archival needs only OpenSearch read + the two S3 buckets) ---
cat > /home/ec2-user/card-oracle-max/.env <<ENVEOF
S3_VECTOR_BUCKET=${s3_vector_bucket}
S3_IMAGE_BUCKET=${s3_image_bucket}
S3_VECTOR_PREFIX=vectors

OPENSEARCH_HOST=${os_host}
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true
OPENSEARCH_USE_IAM=false
OPENSEARCH_USER=${os_user}
OPENSEARCH_PASSWORD=${os_password}

AWS_REGION=us-west-1
WORKER_INDEX=${worker_index}
ENVEOF
chown ec2-user:ec2-user /home/ec2-user/card-oracle-max/.env
chmod 600 /home/ec2-user/card-oracle-max/.env

# --- 4. systemd unit (started after code is deployed) ---
cat > /etc/systemd/system/image-archive.service <<SVCEOF
[Unit]
Description=Card Oracle image-archive worker (CPU)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/card-oracle-max
ExecStart=/home/ec2-user/card-oracle-max/.venv/bin/python -u tools/image_archive_worker.py --workers ${download_workers} --loop
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
# Not enabled on boot — the deploy script installs code + venv, then starts it.

echo "=== bootstrap complete (worker ${worker_index}) ==="
echo "Next: run infra/scripts/deploy-image-archive-fleet.sh from the operator machine."
