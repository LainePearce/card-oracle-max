#!/bin/bash
set -euo pipefail
exec > /var/log/user-data.log 2>&1

echo "=== GPU search worker bootstrap ==="

# ── 1. System packages ────────────────────────────────────────────────────────
# Deep Learning AMI ships with NVIDIA drivers, CUDA, and PyTorch pre-installed.
# Do NOT reinstall nvidia-driver-latest-dkms — it conflicts with AMI drivers.
echo "Installing system packages..."
dnf install -y python3.11 python3.11-pip python3.11-devel git gcc gcc-c++ unzip

# ── 2. Verify GPU ─────────────────────────────────────────────────────────────
echo "Checking GPU..."
nvidia-smi || echo "WARNING: nvidia-smi not ready yet -- drivers are pre-installed on AMI"

# ── 3. Working directory ──────────────────────────────────────────────────────
mkdir -p /home/ec2-user/card-oracle-max/logs
chown -R ec2-user:ec2-user /home/ec2-user/card-oracle-max

# ── 4. Write .env ─────────────────────────────────────────────────────────────
cat > /home/ec2-user/card-oracle-max/.env <<ENVEOF
# Qdrant
QDRANT_HOST=${qdrant_host}
QDRANT_PORT=${qdrant_port}
QDRANT_API_KEY=${qdrant_api_key}
QDRANT_USE_GRPC=false
QDRANT_COLLECTION=cards

# OpenSearch — document enrichment
OPENSEARCH_HOST=${opensearch_host}
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true
OPENSEARCH_USE_IAM=false
OPENSEARCH_USER=${opensearch_user}
OPENSEARCH_PASSWORD=${opensearch_password}

# S3
S3_VECTOR_BUCKET=${s3_vector_bucket}
AWS_REGION=us-west-1

# Worker server
WORKER_PORT=${worker_port}
EMBEDDING_DEVICE=cuda
BATCH_SIZE=8
BATCH_TIMEOUT_MS=20
ENVEOF

chown ec2-user:ec2-user /home/ec2-user/card-oracle-max/.env
chmod 600 /home/ec2-user/card-oracle-max/.env

# ── 5. Systemd service ────────────────────────────────────────────────────────
# Runs gpu_worker_server.py under Gunicorn:
#   - 1 process (CUDA is not fork-safe; single process owns the GPU)
#   - 8 threads (concurrent HTTP requests share the CLIP singleton)
#   - Dynamic batching daemon coalesces requests into GPU batches

cat > /etc/systemd/system/gpu-search-worker.service <<SVCEOF
[Unit]
Description=Card Oracle GPU Search Worker (CLIP + Qdrant)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/card-oracle-max
Environment=PATH=/home/ec2-user/card-oracle-max/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ec2-user/card-oracle-max/.venv/bin/gunicorn \
    --workers 1 \
    --threads 8 \
    --worker-class gthread \
    --bind 0.0.0.0:${worker_port} \
    --timeout 120 \
    --access-logfile /home/ec2-user/card-oracle-max/logs/gunicorn-access.log \
    --error-logfile /home/ec2-user/card-oracle-max/logs/gunicorn-error.log \
    "tools.gpu_worker_server:app"
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
# Service is NOT auto-started — start manually after code is deployed.

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. Deploy code:"
echo "     rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.venv' ./ ec2-user@<public-ip>:~/card-oracle-max/"
echo "  2. SSH in:"
echo "     ssh -i ~/.ssh/<key>.pem ec2-user@<public-ip>"
echo "  3. Create venv and install deps:"
echo "     cd ~/card-oracle-max"
echo "     python3.11 -m venv .venv"
echo "     source .venv/bin/activate"
echo "     pip install -r requirements.txt"
echo "  4. Start the search worker:"
echo "     sudo systemctl start gpu-search-worker"
echo "     sudo journalctl -fu gpu-search-worker"
echo "  5. Enable on reboot (optional):"
echo "     sudo systemctl enable gpu-search-worker"
echo "  6. Health check (via ALB):"
echo "     curl http://<alb-dns>/health"
