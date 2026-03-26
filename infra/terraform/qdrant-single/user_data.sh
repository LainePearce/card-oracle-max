#!/bin/bash
set -euo pipefail
exec > /var/log/user-data.log 2>&1

echo "=== Qdrant test instance bootstrap ==="

# --- 1. Mount NVMe instance store ---
DEVICE="/dev/nvme1n1"
MOUNT="/mnt/qdrant-storage"

echo "Mounting NVMe instance store..."
mkfs.ext4 -F "$DEVICE"
mkdir -p "$MOUNT"
mount "$DEVICE" "$MOUNT"
chown -R ec2-user:ec2-user "$MOUNT"
echo "$DEVICE $MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
echo "NVMe mounted at $MOUNT"

# Create Qdrant data directories
mkdir -p "$MOUNT/data" "$MOUNT/snapshots"
chown -R ec2-user:ec2-user "$MOUNT"

# --- 2. Install Docker ---
echo "Installing Docker..."
dnf install -y docker
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# --- 3. Write Qdrant config ---
cat > /tmp/qdrant-config.yaml <<'QDRANT_CONFIG'
storage:
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/snapshots
  optimizers:
    default_segment_number: 2
    indexing_threshold_kb: 50000
    memmap_threshold_kb: 50000
  wal:
    wal_capacity_mb: 64
    wal_segments_ahead: 0

service:
  host: "0.0.0.0"
  http_port: 6333
  grpc_port: 6334
  enable_tls: false

log_level: INFO
QDRANT_CONFIG

# --- 4. Start Qdrant container ---
echo "Starting Qdrant ${qdrant_version}..."
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$MOUNT/data:/qdrant/storage" \
  -v "$MOUNT/snapshots:/qdrant/snapshots" \
  -v "/tmp/qdrant-config.yaml:/qdrant/config/production.yaml:ro" \
  -e "QDRANT__SERVICE__API_KEY=${qdrant_api_key}" \
  --ulimit nofile=65535:65535 \
  "qdrant/qdrant:${qdrant_version}"

# --- 5. Wait for Qdrant to be ready ---
echo "Waiting for Qdrant to start..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "Qdrant is healthy!"
    break
  fi
  echo "  attempt $i/30..."
  sleep 2
done

echo "=== Bootstrap complete ==="
