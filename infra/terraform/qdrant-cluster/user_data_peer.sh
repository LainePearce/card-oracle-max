#!/bin/bash
set -euo pipefail
exec >> /var/log/user-data.log 2>&1

echo "=== Qdrant cluster PEER node bootstrap ($(date -u)) ==="
echo "    Seed URI: http://${seed_private_ip}:6335"

# --- 1. Mount NVMe instance store (idempotent) ---
DEVICE="/dev/nvme1n1"
MOUNT="/mnt/qdrant-storage"

echo "Checking NVMe instance store..."
if ! blkid "$DEVICE" | grep -q ext4; then
  echo "No ext4 filesystem found — formatting $DEVICE"
  mkfs.ext4 -F "$DEVICE"
else
  echo "ext4 filesystem already present on $DEVICE — skipping format"
fi

mkdir -p "$MOUNT"
if ! mountpoint -q "$MOUNT"; then
  echo "Mounting $DEVICE at $MOUNT..."
  mount "$DEVICE" "$MOUNT"
  chown -R ec2-user:ec2-user "$MOUNT"
  echo "Mounted."
else
  echo "$MOUNT already mounted — skipping"
fi

# Add fstab entry only once
if ! grep -q "$DEVICE" /etc/fstab; then
  echo "$DEVICE $MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
fi

mkdir -p "$MOUNT/data" "$MOUNT/snapshots" "$MOUNT/config"
chown -R ec2-user:ec2-user "$MOUNT"
echo "NVMe ready at $MOUNT"

# --- 2. Install Docker (idempotent) ---
if ! command -v docker &>/dev/null; then
  echo "Installing Docker..."
  dnf install -y docker
  systemctl enable docker
  systemctl start docker
  usermod -aG docker ec2-user
else
  echo "Docker already installed — ensuring it is running"
  systemctl start docker || true
fi

# --- 3. Write Qdrant config to persistent NVMe storage ---
cat > "$MOUNT/config/qdrant.yaml" <<'QDRANT_CONFIG'
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

cluster:
  enabled: true
  p2p:
    port: 6335

log_level: INFO
QDRANT_CONFIG

# --- 4. Start Qdrant peer node (idempotent) ---
MY_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)

if docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
  echo "Qdrant container already running — skipping start"
elif docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
  echo "Qdrant container exists but stopped — starting it"
  docker start qdrant
else
  echo "Starting Qdrant ${qdrant_version} (peer, uri=http://$MY_IP:6335, bootstrap=http://${seed_private_ip}:6335)..."
  docker run -d \
    --name qdrant \
    --restart unless-stopped \
    -p 6333:6333 \
    -p 6334:6334 \
    -p 6335:6335 \
    -v "$MOUNT/data:/qdrant/storage" \
    -v "$MOUNT/snapshots:/qdrant/snapshots" \
    -v "$MOUNT/config/qdrant.yaml:/qdrant/config/production.yaml:ro" \
    -e "QDRANT__SERVICE__API_KEY=${qdrant_api_key}" \
    --ulimit nofile=65535:65535 \
    "qdrant/qdrant:${qdrant_version}" \
    ./qdrant --uri "http://$MY_IP:6335" --bootstrap "http://${seed_private_ip}:6335"
fi

# --- 5. Wait for peer to be healthy ---
echo "Waiting for Qdrant peer to start..."
for i in $(seq 1 40); do
  if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "Qdrant peer is healthy!"
    break
  fi
  echo "  attempt $i/40..."
  sleep 3
done

echo "=== Peer bootstrap complete ==="
