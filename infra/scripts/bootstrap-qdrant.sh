#!/bin/bash
# Install Docker and start Qdrant on an EC2 instance.
# Assumes NVMe is already mounted at /mnt/qdrant-storage (run mount-nvme.sh first).
# Requires: QDRANT_API_KEY env var, optional QDRANT_VERSION (default: v1.8.2).
set -euo pipefail

MOUNT="/mnt/qdrant-storage"
QDRANT_VERSION="${QDRANT_VERSION:-v1.8.2}"

if [ -z "${QDRANT_API_KEY:-}" ]; then
    echo "ERROR: QDRANT_API_KEY environment variable must be set"
    exit 1
fi

if ! mountpoint -q "$MOUNT"; then
    echo "ERROR: $MOUNT is not mounted. Run mount-nvme.sh first."
    exit 1
fi

# Install Docker if not present
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    sudo dnf install -y docker
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ec2-user
    echo "Docker installed. You may need to log out and back in for group changes."
fi

# Create data directories
mkdir -p "$MOUNT/data" "$MOUNT/snapshots"

# Stop existing container if running
if docker ps -q -f name=qdrant | grep -q .; then
    echo "Stopping existing Qdrant container..."
    docker stop qdrant
    docker rm qdrant
fi

# Start Qdrant
echo "Starting Qdrant $QDRANT_VERSION..."
docker run -d \
    --name qdrant \
    --restart unless-stopped \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "$MOUNT/data:/qdrant/storage" \
    -v "$MOUNT/snapshots:/qdrant/snapshots" \
    -e "QDRANT__SERVICE__API_KEY=$QDRANT_API_KEY" \
    -e "QDRANT__LOG_LEVEL=INFO" \
    --ulimit nofile=65535:65535 \
    "qdrant/qdrant:$QDRANT_VERSION"

# Wait for health
echo "Waiting for Qdrant..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        echo "Qdrant is healthy at http://localhost:6333"
        exit 0
    fi
    sleep 2
done

echo "ERROR: Qdrant did not become healthy within 60 seconds"
exit 1
