#!/bin/bash
# warmup_all.sh — Warm up KNN graphs for all indices (newest first)
# Usage: bash warmup_all.sh [start_index_name]
# Example: bash warmup_all.sh 2025-08-01

OS_AUTH="es130point-vector-master:9BTUsLKu*IVzt3GF"
OS_HOST="https://search-es130point-vector-h3eaau7mwcpyhgynkthfcwzeje.aos.us-west-1.on.aws"
START_FROM="${1:-}"

INDICES=$(curl -s -u "$OS_AUTH" "$OS_HOST/_cat/indices?h=index" \
  | grep -v '^\.' \
  | grep -v '^goldin-live' \
  | grep -v '^heritage-live' \
  | grep -v '^pristine-live' \
  | grep -v '^run$' \
  | sort -r)

TOTAL=$(echo "$INDICES" | wc -l | tr -d ' ')
COUNT=0
SKIP=true

if [ -z "$START_FROM" ]; then
  SKIP=false
fi

for INDEX in $INDICES; do
  COUNT=$((COUNT + 1))

  if [ "$SKIP" = true ]; then
    if [ "$INDEX" = "$START_FROM" ]; then
      SKIP=false
    else
      echo "[$(date '+%H:%M:%S')] [$COUNT/$TOTAL] $INDEX: SKIPPED"
      continue
    fi
  fi

  RESULT=$(curl -s -o /dev/null -w "%{http_code}" -u "$OS_AUTH" \
    "$OS_HOST/_plugins/_knn/warmup/$INDEX")
  echo "[$(date '+%H:%M:%S')] [$COUNT/$TOTAL] $INDEX: $RESULT"
done

echo ""
echo "[$(date '+%H:%M:%S')] Done. All $TOTAL indices warmed."
