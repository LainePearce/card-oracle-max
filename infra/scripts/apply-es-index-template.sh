#!/bin/bash
# Apply the card-oracle index template to the new self-managed Elasticsearch cluster.
# Run this once after the cluster is healthy (green), before starting backfill.
#
# The template covers all marketplace index naming conventions:
#
#   eBay:      20??-??-??       YYYY-MM-DD
#   Pristine:  20??-??-pris     YYYY-MM-pris
#   Fanatics:  20??-??-pwcc     YYYY-MM-pwcc
#   Heritage:  20??-heri        YYYY-heri
#   MySlabs:   20??-ms          YYYY-ms
#   Goldin:    20??-gold        YYYY-gold
#
# Usage:
#   OPENSEARCH_DOCS_HOST=<nlb-dns-or-ip> \
#   OPENSEARCH_DOCS_USER=elastic \
#   OPENSEARCH_DOCS_PASSWORD=<password> \
#   ./apply-es-index-template.sh
#
# If security is disabled (xpack.security.enabled=false), USER/PASSWORD are optional.

set -euo pipefail

HOST="${OPENSEARCH_DOCS_HOST:?Set OPENSEARCH_DOCS_HOST}"
PORT="${OPENSEARCH_DOCS_PORT:-9200}"
USER="${OPENSEARCH_DOCS_USER:-}"
PASS="${OPENSEARCH_DOCS_PASSWORD:-}"

BASE="http://${HOST}:${PORT}"

# Build curl auth args only if credentials are set
AUTH_ARGS=""
if [[ -n "$USER" && -n "$PASS" ]]; then
  AUTH_ARGS="-u ${USER}:${PASS}"
fi

echo "→ Checking cluster health at ${BASE}..."
curl -sf $AUTH_ARGS "${BASE}/_cluster/health?pretty"

echo ""
echo "→ Applying index template..."
curl -sf $AUTH_ARGS -X PUT "${BASE}/_index_template/card-oracle-template" \
  -H "Content-Type: application/json" \
  -d @- <<'TEMPLATE'
{
  "index_patterns": [
    "20??-??-??",
    "20??-??-pris",
    "20??-??-pwcc",
    "20??-heri",
    "20??-ms",
    "20??-gold"
  ],
  "priority": 100,
  "template": {
    "settings": {
      "number_of_shards":   3,
      "number_of_replicas": 1,
      "refresh_interval":   "30s",
      "codec":              "best_compression",
      "analysis": {
        "normalizer": {
          "lowercase_normalizer": {
            "type":   "custom",
            "filter": ["lowercase", "asciifolding"]
          }
        }
      }
    },
    "mappings": {
      "dynamic": false,
      "properties": {
        "id":          { "type": "long" },
        "itemId":      { "type": "keyword" },
        "source":      { "type": "keyword" },
        "globalId":    { "type": "keyword" },
        "title": {
          "type": "text",
          "analyzer": "standard",
          "fields": {
            "keyword": { "type": "keyword", "ignore_above": 512 }
          }
        },
        "galleryURL":           { "type": "keyword", "ignore_above": 2048 },
        "itemURL":              { "type": "keyword", "ignore_above": 2048 },
        "saleType":             { "type": "keyword" },
        "currentPrice":         { "type": "double" },
        "currentPriceCurrency": { "type": "keyword" },
        "salePrice":            { "type": "double" },
        "shippingServiceCost":  { "type": "double" },
        "bidCount":             { "type": "integer" },
        "endTime":              { "type": "date" },
        "BestOfferPrice":       { "type": "double" },
        "BestOfferCurrency":    { "type": "keyword" },
        "itemSpecifics": {
          "type": "object",
          "properties": {
            "brand":        { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "player":       { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "genre":        { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "country":      { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "set":          { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "cardNumber":   { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "subset":       { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "parallel":     { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "serialNumber": { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "year":         { "type": "integer" },
            "graded":       { "type": "boolean" },
            "grader":       { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "grade":        { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "type":         { "type": "keyword", "normalizer": "lowercase_normalizer" },
            "autographed":  { "type": "boolean" },
            "team":         { "type": "keyword", "normalizer": "lowercase_normalizer" }
          }
        }
      }
    }
  }
}
TEMPLATE

echo ""
echo "→ Template applied. Verifying..."
curl -sf $AUTH_ARGS "${BASE}/_index_template/card-oracle-template?pretty" \
  | python3 -m json.tool | head -30

echo ""
echo "→ Listing existing indices (if any)..."
curl -sf $AUTH_ARGS "${BASE}/_cat/indices?v&s=index" || echo "(no indices yet)"

echo ""
echo "Done. Cluster is ready for backfill."
