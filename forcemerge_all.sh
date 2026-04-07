#!/bin/bash
# forcemerge_all.sh — Force merge all KNN indices to 1 segment per shard
# Processes newest indices first (2026 -> 2008)
# Usage: bash forcemerge_all.sh [start_index_name]
# Example: bash forcemerge_all.sh 2025-08-01

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

  echo "[$(date '+%H:%M:%S')] [$COUNT/$TOTAL] $INDEX: starting merge..."

  # Fire force merge with 30s HTTP timeout (merge continues server-side)
  curl -s -o /dev/null -m 30 -u "$OS_AUTH" -X POST \
    "$OS_HOST/${INDEX}/_forcemerge?max_num_segments=1"

  # Wait for all active merges to finish before starting next index
  WAIT_COUNT=0
  while true; do
    MERGING=$(curl -s -u "$OS_AUTH" "$OS_HOST/_cat/nodes?h=merges.current" \
      | awk '{s+=$1} END {print s}')

    if [ "$MERGING" -eq 0 ] 2>/dev/null; then
      break
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $((WAIT_COUNT % 12)) -eq 0 ]; then
      echo "[$(date '+%H:%M:%S')] [$COUNT/$TOTAL] $INDEX: still merging ($MERGING active)..."
    fi
    sleep 5
  done

  # Verify segment count
  SEGMENTS=$(curl -s -u "$OS_AUTH" \
    "$OS_HOST/_cat/segments/${INDEX}?h=segment" | sort -u | wc -l | tr -d ' ')

  echo "[$(date '+%H:%M:%S')] [$COUNT/$TOTAL] $INDEX: done ($SEGMENTS segments)"
done

echo ""
echo "[$(date '+%H:%M:%S')] All $TOTAL indices merged."
