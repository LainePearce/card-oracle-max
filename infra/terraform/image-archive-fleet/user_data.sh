#!/bin/bash
set -euo pipefail
exec > /var/log/user-data.log 2>&1

echo "=== image-archive ASG bootstrap ==="

# --- 1. System deps (CPU/IO only — no CUDA) ---
dnf install -y python3.11 python3.11-pip python3.11-devel git gcc gcc-c++ \
  libjpeg-turbo-devel zlib-devel unzip tar

# --- 2. AWS CLI (AL2023 does not ship it; needed to fetch the code tarball) ---
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
AWS=/usr/local/bin/aws

# --- 3. Swap safety net (so a memory spike pages instead of OOM-freezing sshd) ---
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=8192 status=none
  chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# --- 4. Fetch code (instance role has GetObject on the vector bucket) ---
APP=/home/ec2-user/card-oracle-max
mkdir -p "$APP"
$AWS s3 cp "s3://${s3_vector_bucket}/${code_key}" /tmp/code.tar.gz
tar xzf /tmp/code.tar.gz -C "$APP"

# --- 5. .env (image archival needs OpenSearch read + the two buckets) ---
cat > "$APP/.env" <<ENVEOF
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
ENVEOF
chmod 600 "$APP/.env"

# --- 6. venv + lean deps ---
python3.11 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q -U pip
"$APP/.venv/bin/pip" install -q -r "$APP/requirements-image-archive.txt"
chown -R ec2-user:ec2-user "$APP"

# --- 7. systemd: hourly incremental archiver (steady-state) ---
# Installs the repo's incremental units and runs them on an hourly timer instead
# of the continuous backfill loop. To re-drain history instead, bump worker_count
# and swap this for image-archive.service (--loop).
cp "$APP/infra/systemd/image-archive-incremental.service" /etc/systemd/system/
cp "$APP/infra/systemd/image-archive-incremental.timer"   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now image-archive-incremental.timer

echo "=== bootstrap complete — hourly incremental archiver enabled ==="
