#!/usr/bin/env bash
#
# Finish the image CDN on the EXISTING CloudFront distribution that fronts the
# images-130-sold bucket (OAC + bucket GetObject policy are already in place
# and serving). Idempotent — re-run freely; every step is a no-op when already
# correct. Run from an admin laptop.
#
# What it does:
#   1. ACM (us-east-1): reuse an ISSUED cert covering img.130point.com (exact
#      or *.130point.com SAN); otherwise request one (DNS validation) and
#      print the validation CNAME to add in external DNS.
#   2. Bucket policy: add s3:ListBucket for the distribution so a missing key
#      returns a true 404 (without it S3 answers 403, which CloudFront caches
#      and the app can't distinguish from an access problem).
#   3. Distribution: custom error responses (403/404 -> 404, 60s TTL so a
#      listing archived after a miss shows up fast), SimpleCORS response
#      headers (canvas use), and — when the cert is ISSUED — the
#      img.130point.com alias + cert (TLS 1.2, SNI).
#   4. Prints the DNS CNAME to add (the public 130point.com zone is not in
#      Route 53 in this account) and verifies GET caching (Miss -> Hit).
#
# Usage:
#   ./infra/scripts/setup-image-cdn.sh
set -euo pipefail

DIST_ID="${DIST_ID:-E2QPA9BVUAG3OV}"
BUCKET="${BUCKET:-images-130-sold}"
HOST="${CDN_HOST:-img.130point.com}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
DIST_ARN="arn:aws:cloudfront::${ACCOUNT}:distribution/${DIST_ID}"
CF_DOMAIN=$(aws cloudfront get-distribution --id "$DIST_ID" --query "Distribution.DomainName" --output text)
SIMPLE_CORS_POLICY="60669652-455b-4ae9-85a4-c4c02393f86c"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

echo "== distribution $DIST_ID ($CF_DOMAIN) -> s3://$BUCKET =="

# ── 1. Certificate ───────────────────────────────────────────────────────────
echo "== ACM (us-east-1): looking for an ISSUED cert covering $HOST =="
CERT_ARN=$(aws acm list-certificates --region us-east-1 --certificate-statuses ISSUED \
  --query "CertificateSummaryList[].CertificateArn" --output text | tr '\t' '\n' | while read -r arn; do
    [[ -z "$arn" ]] && continue
    names=$(aws acm describe-certificate --region us-east-1 --certificate-arn "$arn" \
      --query "[Certificate.DomainName, Certificate.SubjectAlternativeNames[]]" --output text | tr '\t' '\n')
    parent="${HOST#*.}"
    host_re=$(printf '%s' "$HOST" | sed 's/\./\\./g'); parent_re=$(printf '%s' "$parent" | sed 's/\./\\./g')
    if echo "$names" | grep -qxE "${host_re}|\*\.${parent_re}"; then echo "$arn"; break; fi
  done)

if [[ -n "${CERT_ARN:-}" ]]; then
  echo "cert covers $HOST: $CERT_ARN"
else
  echo "no issued cert covers $HOST — checking for a pending request"
  CERT_ARN=$(aws acm list-certificates --region us-east-1 --certificate-statuses PENDING_VALIDATION \
    --query "CertificateSummaryList[?DomainName=='${HOST}'].CertificateArn | [0]" --output text)
  if [[ -z "$CERT_ARN" || "$CERT_ARN" == "None" ]]; then
    CERT_ARN=$(aws acm request-certificate --region us-east-1 --domain-name "$HOST" \
      --validation-method DNS --query CertificateArn --output text)
    echo "requested: $CERT_ARN"; sleep 5
  fi
  aws acm describe-certificate --region us-east-1 --certificate-arn "$CERT_ARN" \
    --query "Certificate.DomainValidationOptions[0].ResourceRecord.{name:Name,type:Type,value:Value}" --output table
  echo ">>> Add that validation CNAME in the 130point.com DNS zone, wait for status ISSUED"
  echo ">>> (aws acm describe-certificate ... --query Certificate.Status), then re-run this script."
  CERT_ARN=""
fi

# ── 2. Bucket policy: ListBucket -> real 404s for missing keys ───────────────
echo "== bucket policy: s3:ListBucket for the distribution =="
aws s3api get-bucket-policy --bucket "$BUCKET" --query Policy --output text > "$WORK/policy.json"
python3 - "$WORK/policy.json" "$BUCKET" "$DIST_ARN" <<'PY'
import json, sys
path, bucket, dist_arn = sys.argv[1:]
pol = json.load(open(path))
sid = "AllowCloudFrontOACListBucket"
if not any(s.get("Sid") == sid for s in pol["Statement"]):
    pol["Statement"].append({
        "Sid": sid, "Effect": "Allow",
        "Principal": {"Service": "cloudfront.amazonaws.com"},
        "Action": "s3:ListBucket",
        "Resource": f"arn:aws:s3:::{bucket}",
        "Condition": {"StringEquals": {"AWS:SourceArn": dist_arn}},
    })
    json.dump(pol, open(path, "w"))
    print("added ListBucket statement")
else:
    print("ListBucket statement already present")
PY
aws s3api put-bucket-policy --bucket "$BUCKET" --policy "file://$WORK/policy.json"

# ── 3. Distribution config ───────────────────────────────────────────────────
echo "== distribution: error responses, CORS headers, alias + cert =="
aws cloudfront get-distribution-config --id "$DIST_ID" > "$WORK/dist.json"
ETAG=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ETag'])" "$WORK/dist.json")
CHANGED=$(python3 - "$WORK/dist.json" "$WORK/cfg.json" "$HOST" "${CERT_ARN:-}" "$SIMPLE_CORS_POLICY" <<'PY'
import json, sys
src, dst, host, cert, cors = sys.argv[1:]
cfg = json.load(open(src))["DistributionConfig"]
before = json.dumps(cfg, sort_keys=True)

errors = [{"ErrorCode": c, "ResponsePagePath": "", "ResponseCode": "404", "ErrorCachingMinTTL": 60}
          for c in (403, 404)]
cfg["CustomErrorResponses"] = {"Quantity": len(errors), "Items": errors}

cfg["DefaultCacheBehavior"]["ResponseHeadersPolicyId"] = cors

if cert:
    aliases = set(cfg.get("Aliases", {}).get("Items") or [])
    aliases.add(host)
    cfg["Aliases"] = {"Quantity": len(aliases), "Items": sorted(aliases)}
    cfg["ViewerCertificate"] = {
        "ACMCertificateArn": cert, "SSLSupportMethod": "sni-only",
        "MinimumProtocolVersion": "TLSv1.2_2021",
        "CloudFrontDefaultCertificate": False,
    }

json.dump(cfg, open(dst, "w"))
print("yes" if json.dumps(cfg, sort_keys=True) != before else "no")
PY
)
if [[ "$CHANGED" == "yes" ]]; then
  aws cloudfront update-distribution --id "$DIST_ID" --if-match "$ETAG" \
    --distribution-config "file://$WORK/cfg.json" --query "Distribution.Status" --output text
  echo "distribution updated — propagating (5–10 min); existing serving continues meanwhile"
else
  echo "distribution already matches desired config"
fi

# ── 4. DNS + verification ────────────────────────────────────────────────────
echo
echo "== DNS record to add in the 130point.com zone (not Route 53 in this account) =="
echo "  $HOST  CNAME  $CF_DOMAIN"
echo
echo "== verify: GET twice via $CF_DOMAIN (expect Miss then Hit) =="
KEY=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix images/ebay/256/ --max-keys 1 --query "Contents[0].Key" --output text)
for i in 1 2; do
  curl -s -o /dev/null -w "  %{http_code}  %{size_download}B  " "https://${CF_DOMAIN}/${KEY}"
  curl -sI "https://${CF_DOMAIN}/${KEY}" | grep -i "^x-cache" || echo
done
echo "== verify: missing key returns 404 (not 403) =="
curl -s -o /dev/null -w "  %{http_code}\n" "https://${CF_DOMAIN}/images/ebay/256/does-not-exist.jpg"
if getent hosts "$HOST" >/dev/null 2>&1 || host "$HOST" >/dev/null 2>&1; then
  echo "== verify via $HOST =="
  curl -s -o /dev/null -w "  %{http_code} https://${HOST}/${KEY}\n" "https://${HOST}/${KEY}" || true
fi
echo
echo "App URL template:  https://${HOST}/images/{source}/{256|512|original}/{_id}.jpg"
echo "Fallback chain:    CDN -> listing galleryURL -> placeholder (404 = never archived)"
