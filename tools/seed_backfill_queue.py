#!/usr/bin/env python3
"""
Seed the S3 backfill queue with jobs for all eBay and non-eBay indices.

For each index the script:
  1. Queries OpenSearch for the endTime distribution (date_histogram aggregation).
  2. Computes dynamic window boundaries so each of the NUM_WORKERS workers
     receives approximately equal document counts.
  3. Writes NUM_WORKERS job JSON files to:
       s3://{S3_VECTOR_BUCKET}/backfill-v2/queue/{job_id}.json

Idempotent — already-queued, active, or complete jobs are skipped.

Usage
-----
    # Preview without writing (prints planned jobs)
    python tools/seed_backfill_queue.py --dry-run

    # Seed queue for priority-1 indices only (Oct 2025 – Feb 2026 + Aug-Sep 2025)
    python tools/seed_backfill_queue.py --priority 1 2

    # Full queue (all eBay + non-eBay)
    python tools/seed_backfill_queue.py

Priority bands
--------------
    1 — Oct 2025 – Feb 2026  (2–10% Qdrant coverage, highest volume)
    2 — Aug–Sep 2025         (10% coverage)
    3 — Dec 2024             (0% coverage, older images)
    4 — All other date indices (partial coverage, gap fill)
    5 — Non-eBay marketplaces (PWCC, Pristine, Goldin, MySlabs, Heritage)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import boto3
from loguru import logger

from src.ingestion.opensearch_reader import classify_index, get_opensearch_client

# ── Config ────────────────────────────────────────────────────────────────────

S3_BUCKET        = os.environ["S3_VECTOR_BUCKET"]
S3_QUEUE_PREFIX  = "backfill-v2/queue"
S3_ACTIVE_PREFIX = "backfill-v2/active"
S3_COMP_PREFIX   = "backfill-v2/complete"
S3_FAIL_PREFIX   = "backfill-v2/failed"

NUM_WORKERS = 3

# Indices to exclude from queueing (live/run indices, system indices)
EXCLUDE_PATTERNS = re.compile(r"^(\.|run$|.*-live$)")

# Pause between aggregation queries to avoid hammering OS (seconds)
AGG_QUERY_PAUSE = 0.2


# ── Priority ──────────────────────────────────────────────────────────────────

def _priority(index_name: str) -> int:
    """Return processing priority (lower = sooner)."""
    c = classify_index(index_name)
    if c["index_type"] == "ebay-dated":
        d = index_name  # YYYY-MM-DD
        if "2025-10-01" <= d <= "2026-02-28":
            return 1
        if "2025-08-01" <= d <= "2025-09-30":
            return 2
        if "2024-12-01" <= d <= "2024-12-31":
            return 3
        return 4
    # Non-eBay marketplaces — process after eBay gaps are filled
    return 5


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def _job_already_handled(s3, job_id: str) -> bool:
    """Return True if this job is already queued, active, complete, or failed."""
    for prefix in (S3_QUEUE_PREFIX, S3_ACTIVE_PREFIX, S3_COMP_PREFIX, S3_FAIL_PREFIX):
        if _s3_exists(s3, f"{prefix}/{job_id}.json"):
            return True
    return False


# ── Window computation ────────────────────────────────────────────────────────

def _compute_windows(
    os_client,
    index_name: str,
    num_workers: int,
    classification: dict,
) -> list[dict] | None:
    """
    Query OpenSearch for the endTime distribution and return a list of
    {ts_start, ts_end, doc_count_estimate} dicts — one per worker.

    Returns None if the index has no documents.
    """
    # Choose bucket granularity based on index span
    if classification["index_type"] == "ebay-dated":
        interval = "1h"
        agg_key  = "by_hour"
    else:
        # Monthly / annual non-eBay indices → split by day
        interval = "1d"
        agg_key  = "by_day"

    try:
        resp = os_client.search(
            index=index_name,
            body={
                "size": 0,
                "aggs": {
                    agg_key: {
                        "date_histogram": {
                            "field":    "endTime",
                            "fixed_interval" if interval == "1h" else "calendar_interval": interval,
                            "format":   "yyyy-MM-dd HH:mm:ss",
                            "min_doc_count": 1,
                        }
                    }
                },
            },
            request_timeout=30,
        )
    except Exception as exc:
        logger.warning("Aggregation failed for {}: {}", index_name, exc)
        return None

    buckets = resp["aggregations"][agg_key]["buckets"]
    if not buckets:
        return None

    total = sum(b["doc_count"] for b in buckets)
    if total == 0:
        return None

    target_per_worker = total / num_workers

    # Find bucket-boundary cutoffs where cumulative crosses N * target
    cutoffs: list[str] = []
    cumulative = 0
    for i, b in enumerate(buckets):
        cumulative += b["doc_count"]
        if len(cutoffs) < num_workers - 1:
            threshold = target_per_worker * (len(cutoffs) + 1)
            if cumulative >= threshold and i + 1 < len(buckets):
                # Use the START of the next bucket as the boundary
                cutoffs.append(buckets[i + 1]["key_as_string"])

    # Build (ts_start, ts_end) pairs.
    # If the index has too few distinct buckets to produce num_workers cutoffs,
    # fall back gracefully to however many splits we could compute.
    actual_workers = len(cutoffs) + 1   # always at least 1
    boundaries = [None] + cutoffs + [None]
    windows = []
    for w in range(actual_workers):
        ts_start = boundaries[w]
        ts_end   = boundaries[w + 1]
        # Estimate count
        est = total // actual_workers
        windows.append({
            "ts_start":          ts_start,
            "ts_end":            ts_end,
            "doc_count_estimate": est,
        })

    return windows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Seed S3 backfill queue")
    ap.add_argument("--priority", type=int, nargs="+",
                    help="Only queue indices matching these priority levels (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned jobs without writing to S3")
    ap.add_argument("--include-pattern", default=None,
                    help="Only queue indices whose name contains this string")
    args = ap.parse_args()

    allowed_priorities = set(args.priority) if args.priority else None

    logger.info("Connecting to OpenSearch...")
    os_client = get_opensearch_client()

    logger.info("Connecting to S3 (bucket={})...", S3_BUCKET)
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))

    # ── Discover all indices ──────────────────────────────────────────────────
    logger.info("Listing all OpenSearch indices...")
    cat_resp = os_client.cat.indices(
        format="json",
        h="index,docs.count",
        params={"expand_wildcards": "open"},
    )

    indices = []
    for row in cat_resp:
        name  = row.get("index", "")
        count = int(row.get("docs.count", 0) or 0)
        if EXCLUDE_PATTERNS.match(name):
            continue
        if count == 0:
            continue
        if args.include_pattern and args.include_pattern not in name:
            continue
        c = classify_index(name)
        if c["index_type"] == "unknown":
            continue
        p = _priority(name)
        if allowed_priorities and p not in allowed_priorities:
            continue
        indices.append({
            "name":     name,
            "count":    count,
            "priority": p,
            "classify": c,
        })

    indices.sort(key=lambda x: (x["priority"], x["name"]))
    logger.info("Found {} eligible indices to process", len(indices))

    # ── Queue jobs ────────────────────────────────────────────────────────────
    queued = skipped_exists = skipped_empty = 0

    for idx_info in indices:
        name  = idx_info["name"]
        prio  = idx_info["priority"]
        c     = idx_info["classify"]

        logger.debug("Processing {} (priority={}, type={})", name, prio, c["index_type"])
        time.sleep(AGG_QUERY_PAUSE)

        # Compute dynamic windows
        windows = _compute_windows(os_client, name, NUM_WORKERS, c)
        if not windows:
            logger.debug("  Skipping {} — empty or agg failed", name)
            skipped_empty += 1
            continue

        for w_id, win in enumerate(windows):
            # Sanitize index name for use in job_id (replace / with -)
            safe_name = name.replace("/", "-")
            job_id = f"{safe_name}-w{w_id}"

            # Check if already handled
            if not args.dry_run and _job_already_handled(s3, job_id):
                logger.debug("  Job {} already exists — skipping", job_id)
                skipped_exists += 1
                continue

            job = {
                "job_id":            job_id,
                "index_name":        name,
                "index_type":        c["index_type"],
                "marketplace":       c["marketplace"],
                "has_item_specifics": c["has_item_specifics"],
                "worker_id":         w_id,
                "ts_start":          win["ts_start"],   # None = start of index
                "ts_end":            win["ts_end"],     # None = end of index
                "doc_count_estimate": win["doc_count_estimate"],
                "priority":          prio,
                "created_at":        datetime.now(timezone.utc).isoformat(),
            }

            if args.dry_run:
                ts_s = win["ts_start"] or "START"
                ts_e = win["ts_end"]   or "END"
                logger.info("  [DRY] {} | w{} | {} → {} | ~{:,} docs",
                            name, w_id, ts_s, ts_e, win["doc_count_estimate"])
            else:
                key = f"{S3_QUEUE_PREFIX}/{job_id}.json"
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=json.dumps(job, indent=2).encode(),
                    ContentType="application/json",
                )
                logger.info("  Queued {} (w{}, ~{:,} docs, priority={})",
                            job_id, w_id, win["doc_count_estimate"], prio)
                queued += 1

    logger.info("Done. queued={} skipped_exists={} skipped_empty={}",
                queued, skipped_exists, skipped_empty)


if __name__ == "__main__":
    main()
