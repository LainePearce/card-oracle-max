#!/bin/bash
set -euo pipefail
exec > /var/log/user-data.log 2>&1

echo "=== GPU worker bootstrap (worker ${worker_index}) ==="

# --- 1. Install system dependencies ---
# The Deep Learning AMI ships with NVIDIA drivers, CUDA, and PyTorch pre-installed.
# Do NOT reinstall nvidia-driver-latest-dkms -- it conflicts with AMI drivers.
echo "Installing system packages..."
dnf install -y python3.11 python3.11-pip python3.11-devel git gcc gcc-c++ unzip

# --- 2. Verify GPU ---
echo "Checking GPU..."
nvidia-smi || echo "WARNING: nvidia-smi not ready yet -- drivers are pre-installed on AMI"

# --- 3. Create working directory ---
mkdir -p /home/ec2-user/card-oracle-max/logs
chown -R ec2-user:ec2-user /home/ec2-user/card-oracle-max

# --- 4. Write .env file ---
cat > /home/ec2-user/card-oracle-max/.env <<ENVEOF
# Qdrant -- 3-node production cluster (internal NLB, gRPC)
QDRANT_HOST=${qdrant_host}
QDRANT_PORT=${qdrant_port}
QDRANT_API_KEY=${qdrant_api_key}
QDRANT_USE_GRPC=true
QDRANT_COLLECTION=cards
QDRANT_SINGLE_NODE=false

# OpenSearch -- extant cluster (read-only, experiment control)
OPENSEARCH_HOST=${os_host}
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true
OPENSEARCH_USE_IAM=false
OPENSEARCH_USER=${os_user}
OPENSEARCH_PASSWORD=${os_password}

# OpenSearch -- new self-managed cluster (Stage 2; not used in Stage 1 backfill)
OPENSEARCH_DOCS_HOST=${os_docs_host}
OPENSEARCH_DOCS_PORT=9200
OPENSEARCH_DOCS_USE_SSL=false
OPENSEARCH_DOCS_USER=${os_docs_user}
OPENSEARCH_DOCS_PASSWORD=${os_docs_password}

# RDS MySQL -- primary (current data source)
RDS_HOST=${rds_host}
RDS_PORT=3306
RDS_USER=${rds_user}
RDS_PASSWORD=${rds_password}
RDS_DATABASE=${rds_database}

# RDS MySQL -- secondary (older DB; empty = disabled, primary wins on duplicate IDs)
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
# AWS_PROFILE intentionally omitted -- EC2 instances use IAM instance profile, not named profiles
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE_IMAGE=256

# Image download concurrency -- 16 threads keeps A10G (g5.xlarge) saturated
IMAGE_DOWNLOAD_WORKERS=16

# Worker identity (used in checkpoint naming)
WORKER_INDEX=${worker_index}
ENVEOF

chown ec2-user:ec2-user /home/ec2-user/card-oracle-max/.env
chmod 600 /home/ec2-user/card-oracle-max/.env

# --- 5. Write systemd service ---
# Runs the embedding backfill: RDS scroll -> CLIP + MiniLM encode -> S3 + Qdrant.
# Checkpointed every 10k rows so spot interruptions resume cleanly.
cat > /etc/systemd/system/backfill.service <<SVCEOF
[Unit]
Description=Card Oracle RDS embedding backfill (worker ${worker_index})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/card-oracle-max
Environment=PATH=/home/ec2-user/card-oracle-max/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ec2-user/card-oracle-max/.venv/bin/python tools/worker_phases.py \
  --worker-index ${worker_index}
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
# Service is NOT enabled on boot -- start it manually after deploying code.

echo "=== Bootstrap complete -- worker ${worker_index} ==="
echo ""
echo "Next steps:"
echo "  1. Sync code:    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.venv' ./ ec2-user@<ip>:~/card-oracle-max/"
echo "  2. SSH in:       ssh -i ~/.ssh/qdrant-test.pem ec2-user@<ip>"
echo "  3. Create venv:  cd ~/card-oracle-max && python3.11 -m venv .venv"
echo "  4. Install deps: source .venv/bin/activate && pip install -r requirements.txt"
echo "  5. Start job:    sudo systemctl start backfill && sudo journalctl -fu backfill"
echo "  6. Verify S3:    aws s3 ls s3://${s3_bucket}/${s3_prefix}/image/ --recursive | tail -5"
