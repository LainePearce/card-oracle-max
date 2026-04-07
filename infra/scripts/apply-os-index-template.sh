#!/bin/bash
# Apply the card-oracle index template to the new OpenSearch cluster.
# Run this once after the cluster is healthy, before starting backfill.
#
# Usage:
#   OPENSEARCH_DOCS_HOST=<nlb-dns> \
#   OPENSEARCH_DOCS_USER=admin \
#   OPENSEARCH_DOCS_PASSWORD=<password> \
#   ./apply-os-index-template.sh

set -euo pipefail

HOST="${OPENSEARCH_DOCS_HOST:?Set OPENSEARCH_DOCS_HOST}"
PORT="${OPENSEARCH_DOCS_PORT:-9200}"
USER="${OPENSEARCH_DOCS_USER:-admin}"
PASS="${OPENSEARCH_DOCS_PASSWORD:?Set OPENSEARCH_DOCS_PASSWORD}"

BASE="http://${HOST}:${PORT}"
AUTH="-u ${USER}:${PASS}"

echo "→ Checking cluster health..."
curl -sf $AUTH "${BASE}/_cluster/health?pretty"

echo ""
echo "→ Applying index template..."
curl -sf $AUTH -X PUT "${BASE}/_index_template/card-oracle-template" \
  -H "Content-Type: application/json" \
  -d @- <<'TEMPLATE'
{
  "index_patterns": ["2*", "*-gold", "*-pwcc", "*-pris", "*-ms", "*-heri", "*-rea", "*-veriswap", "*-cardhobby"],
  "priority": 100,
  "template": {
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 1,
      "refresh_interval": "30s",
      "index.codec": "best_compression",
      "index.knn": false,
      "analysis": {
        "normalizer": {
          "lowercase_normalizer": {
            "type": "custom",
            "filter": ["lowercase", "asciifolding"]
          }
        }
      }
    },
    "mappings": {
      "dynamic": false,
      "_source": {
        "excludes": []
      },
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
        "galleryURL":  { "type": "keyword", "ignore_above": 2048 },
        "itemURL":     { "type": "keyword", "ignore_above": 2048 },
        "saleType":    { "type": "keyword" },
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
curl -sf $AUTH "${BASE}/_index_template/card-oracle-template?pretty" | python3 -m json.tool | head -20

echo ""
echo "Done. Cluster is ready for backfill."
