#!/usr/bin/env bash
#
# Apply the images-130-sold lifecycle policy.
#
# Originals are large and rarely re-read after ingest, so they transition to
# STANDARD_IA after 30 days. The 512/256 serving variants stay in Standard (hot
# for the search UI/CDN, and below the 128 KB IA minimum-billable size so IA
# would cost MORE, not less). One rule per marketplace source, because keys are
# namespaced images/{source}/original/... and S3 prefix filters can't wildcard
# the middle segment.
#
# put-bucket-lifecycle-configuration REPLACES the entire config, so this script
# is the single source of truth. Re-run it after introducing a new source.
# It backs up the live config to /tmp before applying.
#
# Usage:
#   ./infra/scripts/apply-image-lifecycle.sh
#   IA_DAYS=45 SOURCES="ebay pwcc pris gold heri ms" ./infra/scripts/apply-image-lifecycle.sh
set -euo pipefail

BUCKET="${IMAGE_BUCKET:-images-130-sold}"
REGION="${AWS_REGION:-us-west-1}"
IA_DAYS="${IA_DAYS:-30}"
read -r -a SOURCES <<< "${SOURCES:-ebay pwcc pris gold heri ms other}"

echo "Backing up current lifecycle config to /tmp/lifecycle-backup.json ..."
aws s3api get-bucket-lifecycle-configuration --bucket "$BUCKET" --region "$REGION" \
  > /tmp/lifecycle-backup.json 2>/dev/null \
  && echo "  backed up" || echo "  (no existing lifecycle config)"

rules=""
for src in "${SOURCES[@]}"; do
  rules+="{\"ID\":\"original-to-ia-${src}\",\"Filter\":{\"Prefix\":\"images/${src}/original/\"},\"Status\":\"Enabled\",\"Transitions\":[{\"Days\":${IA_DAYS},\"StorageClass\":\"STANDARD_IA\"}]},"
done
# Plus a hygiene rule: clean up incomplete multipart uploads bucket-wide.
rules+="{\"ID\":\"abort-incomplete-multipart\",\"Filter\":{\"Prefix\":\"\"},\"Status\":\"Enabled\",\"AbortIncompleteMultipartUpload\":{\"DaysAfterInitiation\":7}}"

printf '{"Rules":[%s]}' "$rules" > /tmp/lifecycle.json
python3 -m json.tool /tmp/lifecycle.json > /dev/null   # validate before applying

aws s3api put-bucket-lifecycle-configuration \
  --bucket "$BUCKET" --region "$REGION" \
  --lifecycle-configuration file:///tmp/lifecycle.json

echo "Applied lifecycle to s3://${BUCKET} — originals -> STANDARD_IA @ ${IA_DAYS}d for: ${SOURCES[*]}"
aws s3api get-bucket-lifecycle-configuration --bucket "$BUCKET" --region "$REGION" \
  --query "Rules[].{ID:ID,Prefix:Filter.Prefix,IA_days:Transitions[0].Days}" --output table
