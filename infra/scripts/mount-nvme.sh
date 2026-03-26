#!/bin/bash
# Mount NVMe instance store on r6gd instances.
# Run once on instance startup. Idempotent — safe to re-run.
set -euo pipefail

DEVICE="/dev/nvme1n1"
MOUNT="/mnt/qdrant-storage"

if mountpoint -q "$MOUNT" 2>/dev/null; then
    echo "Already mounted at $MOUNT"
    exit 0
fi

mkfs.ext4 -F "$DEVICE"
mkdir -p "$MOUNT"
mount "$DEVICE" "$MOUNT"
chown -R ec2-user:ec2-user "$MOUNT"

# Add to fstab if not already present
if ! grep -q "$MOUNT" /etc/fstab; then
    echo "$DEVICE $MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
fi

echo "NVMe mounted at $MOUNT"
