#!/bin/bash
# Deploy the image-archive code to the CPU fleet and start the worker service.
#
# Run from the operator machine in the repo root, passing the fleet's public IPs
# (terraform output public_ips). Idempotent — safe to re-run after code changes.
#
#   KEY=~/.ssh/qdrant-test.pem ./infra/scripts/deploy-image-archive-fleet.sh 1.2.3.4 5.6.7.8 ...
#
# Writes worker_ips.json (public + private) so the image dashboard polls this
# fleet instead of the GPU workers — copy it to whichever host runs the dashboard.
set -euo pipefail

KEY="${KEY:-$HOME/.ssh/qdrant-test.pem}"
REMOTE=/home/ec2-user/card-oracle-max
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10)

if [ "$#" -lt 1 ]; then
  echo "usage: KEY=~/.ssh/key.pem $0 <public-ip> [public-ip ...]" >&2
  exit 1
fi

PUBLIC=("$@")
PRIVATE=()

for ip in "${PUBLIC[@]}"; do
  echo "=== deploying $ip ==="
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='.venv-local' \
    --exclude='data' --exclude='*.pyc' \
    --exclude='.terraform' --exclude='*.tfstate*' --exclude='*.tfvars' \
    ./ "ec2-user@$ip:$REMOTE/"

  ssh "${SSH_OPTS[@]}" "ec2-user@$ip" "
    set -e
    cd $REMOTE
    test -d .venv || python3.11 -m venv .venv
    .venv/bin/pip install -q -U pip
    .venv/bin/pip install -q -r requirements-image-archive.txt
    sudo systemctl enable --now image-archive
    echo -n 'service: '; systemctl is-active image-archive
  "

  priv="$(ssh "${SSH_OPTS[@]}" "ec2-user@$ip" "hostname -I | awk '{print \$1}'")"
  PRIVATE+=("$priv")
done

# Emit worker_ips.json for image_archive_dashboard.py (it prefers this over hardcoded IPs).
pub_json=$(printf '"%s",' "${PUBLIC[@]}"  | sed 's/,$//')
priv_json=$(printf '"%s",' "${PRIVATE[@]}" | sed 's/,$//')
echo "{\"public\": [$pub_json], \"private\": [$priv_json]}" > worker_ips.json

echo
echo "Deployed to ${#PUBLIC[@]} workers. Wrote worker_ips.json."
echo "Copy worker_ips.json to the dashboard host so it polls this fleet:"
echo "  scp -i $KEY worker_ips.json ec2-user@${PUBLIC[0]}:$REMOTE/"
