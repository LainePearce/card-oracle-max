#!/bin/bash
# Publish the image-archive code as a tarball to S3 so ASG instances can
# self-bootstrap at boot (the launch-template user_data downloads + extracts it).
#
# Run from the repo root whenever the worker code changes:
#   ./infra/scripts/publish-image-archive-code.sh
#
# Then either let new ASG instances pick it up on launch, or force a rolling
# refresh:  aws autoscaling start-instance-refresh --auto-scaling-group-name image-archive-asg
set -euo pipefail

BUCKET="${S3_VECTOR_BUCKET:-card-oracle-vectors}"
KEY="${CODE_KEY:-deploy/image-archive-code.tar.gz}"

TAR="$(mktemp /tmp/iac-code-XXXXXX.tar.gz)"
trap 'rm -f "$TAR"' EXIT

# Only the bits the worker needs — no .git/.venv/data/secrets.
tar czf "$TAR" \
  --exclude='__pycache__' --exclude='*.pyc' \
  tools src infra/systemd requirements-image-archive.txt

aws s3 cp "$TAR" "s3://$BUCKET/$KEY"
echo "Published $(du -h "$TAR" | cut -f1) → s3://$BUCKET/$KEY"
