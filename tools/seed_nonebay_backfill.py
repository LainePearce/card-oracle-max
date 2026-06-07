#!/usr/bin/env python3
"""
Non-eBay marketplace backfill seeder (2025/2026).

The per-day seeder (seed_daily_backfill.py) only handles eBay-dated
YYYY-MM-DD indices. Non-eBay marketplaces use different naming
(YYYY-MM-pris/pwcc, YYYY-heri/ms/gold/heritage) and were originally
backfilled by seed_backfill_queue.py — but that run (May 2026) predated
the shard-key fix (commit 68d5718), so every job for a given type wrote
to the same {type}/all/NNNN.parquet and clobbered the others. Result:
~9% coverage. The fix is now in vector_store.shard_key (job_id in the
key), so re-running with current code won't collide.

This seeder:
  - Discovers non-eBay OS indices in the requested years.
  - Splits each into NUM_WORKERS windows by endTime histogram.
  - Seeds jobs with marketplace/index_type/has_item_specifics/partition
    from classify_index (which now also handles the `heritage` alias).

Job IDs use the form "{index}_w{NN}" (e.g. 2025-01-pris_w00). These do
NOT match the orchestrator's YYYY-MM-DD date regex, so os-orchestrator
ignores them entirely — non-eBay verify/push is handled manually via
audit_nonebay_coverage.py + repopulate_qdrant --index-type <t>.

Idempotent: indices whose window-00 job is already queued/active/
complete/failed are skipped. Use --force to re-seed (clear old records
first — see the runbook in the chat).

Usage:
  python tools/seed_nonebay_backfill.py --dry-run
  python tools/seed_nonebay_backfill.py --years 2025 2026
  python tools/seed_nonebay_backfill.py --include pris pwcc   # subset of types
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from loguru import logger
from opensearchpy.exceptions import NotFoundError

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.ingestion.opensearch_reader import get_opensearch_client, classify_index


S3_BUCKET        = os.environ["S3_VECTOR_BUCKET"]
S3_QUEUE_PREFIX  = "backfill-v2/queue"
S3_ACTIVE_PREFIX = "backfill-v2/active"
S3_COMP_PREFIX   = "backfill-v2/complete"
S3_FAIL_PREFIX   = "backfill-v2/failed"

NUM_WORKERS = 12

# Non-eBay marketplace index patterns.
_MONTHLY = re.compile(r"^(\d{4})-\d{2}-(pris|pwcc)$")
_ANNUAL  = re.compile(r"^(\d{4})-(heri|heritage|ms|gold)$")

# Non-eBay priority band — well above eBay's P1/P2/P3 (1000-3xxx) so
# non-eBay only runs when eBay work is drained. 4000 base.
PRIORITY_BASE = 4_000


def _index_year(name: str) -> str | None:
    m = _MONTHLY.match(name) or _ANNUAL.match(name)
    return m.group(1) if m else None


def s3_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def index_already_seeded(s3, index_name: str) -> bool:
    # Old campaign used "{index}-w0"; new seeder uses "{index}_w00". Check both.
    for jid in (f"{index_name}_w00", f"{index_name}-w0"):
        for prefix in (S3_QUEUE_PREFIX, S3_ACTIVE_PREFIX,
                       S3_COMP_PREFIX,  S3_FAIL_PREFIX):
            if s3_exists(s3, f"{prefix}/{jid}.json"):
                return True
    return False


def compute_windows(os_client, index_name: str, n_workers: int) -> list[dict] | None:
    """Daily endTime histogram, split into n_workers equal-doc windows."""
    try:
        resp = os_client.search(
            index = index_name,
            body  = {
                "size": 0,
                "aggs": {"by_day": {"date_histogram": {
                    "field": "endTime", "calendar_interval": "1d",
                    "format": "yyyy-MM-dd HH:mm:ss", "min_doc_count": 1,
                }}},
            },
            request_timeout=30,
        )
    except NotFoundError:
        return None
    except Exception as exc:
        logger.warning("Aggregation failed for {}: {}", index_name, exc)
        return None

    buckets = resp["aggregations"]["by_day"]["buckets"]
    if not buckets:
        return None
    total = sum(b["doc_count"] for b in buckets)
    if total == 0:
        return None

    target = total / n_workers
    cutoffs: list[str] = []
    cumulative = 0
    for i, b in enumerate(buckets):
        cumulative += b["doc_count"]
        if len(cutoffs) < n_workers - 1:
            if cumulative >= target * (len(cutoffs) + 1) and i + 1 < len(buckets):
                cutoffs.append(buckets[i + 1]["key_as_string"])

    actual_n   = len(cutoffs) + 1
    boundaries = [None] + cutoffs + [None]
    return [
        {"ts_start": boundaries[w], "ts_end": boundaries[w + 1],
         "doc_count_estimate": total // actual_n}
        for w in range(actual_n)
    ]


def seed_index(s3, os_client, index_name: str, offset: int,
               include_types: set[str] | None, dry_run: bool) -> tuple[int, str]:
    c = classify_index(index_name)
    if include_types and c["index_type"] not in include_types:
        return (0, "skip:type-filtered")
    if index_already_seeded(s3, index_name):
        return (0, "skip:already-handled")

    windows = compute_windows(os_client, index_name, NUM_WORKERS)
    if not windows:
        return (0, "skip:no-docs-or-index")

    priority = PRIORITY_BASE + offset
    seeded = 0
    for w_idx, window in enumerate(windows):
        job_id = f"{index_name}_w{w_idx:02d}"
        job = {
            "job_id":             job_id,
            "priority":           priority,
            "block":              "P-nonebay",
            "index_name":         index_name,
            "index_type":         c["index_type"],
            "marketplace":        c["marketplace"],
            "has_item_specifics": c["has_item_specifics"],
            "partition":          c["partition"],
            "ts_start":           window["ts_start"],
            "ts_end":             window["ts_end"],
            "doc_count_estimate": window["doc_count_estimate"],
            "created_at":         datetime.now(timezone.utc).isoformat(),
        }
        if dry_run:
            logger.info("[dry-run] would seed {} (mkt={}, type={}, prio={})",
                        job_id, c["marketplace"], c["index_type"], priority)
        else:
            s3.put_object(Bucket=S3_BUCKET,
                          Key=f"{S3_QUEUE_PREFIX}/{job_id}.json",
                          Body=json.dumps(job).encode())
        seeded += 1
    return (seeded, "seeded")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", default=["2025", "2026"])
    ap.add_argument("--include", nargs="+", default=None,
                    help="Only seed these index_types (pris pwcc heri ms gold).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    years         = set(args.years)
    include_types = set(args.include) if args.include else None

    s3        = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))
    os_client = get_opensearch_client()

    logger.info("Discovering non-eBay indices for years {}...", sorted(years))
    cat = os_client.cat.indices(format="json", h="index,docs.count",
                                params={"expand_wildcards": "open"})
    indices = sorted(
        row["index"] for row in cat
        if _index_year(row.get("index", "")) in years
    )
    logger.info("  {} non-eBay indices in scope", len(indices))

    totals = {"seeded_jobs": 0, "seeded_indices": 0}
    status_counts: dict[str, int] = {}

    for offset, idx in enumerate(indices):
        n, status = seed_index(s3, os_client, idx, offset, include_types, args.dry_run)
        status_counts[status] = status_counts.get(status, 0) + 1
        totals["seeded_jobs"] += n
        if status == "seeded":
            totals["seeded_indices"] += 1
            logger.info("  {} → seeded {} jobs", idx, n)
        else:
            logger.info("  {} → {}", idx, status)

    logger.info("──────────────────────────────────────────")
    logger.info("Seeded {} jobs across {} indices",
                totals["seeded_jobs"], totals["seeded_indices"])
    for k in sorted(status_counts):
        logger.info("  {:24s} {}", k, status_counts[k])


if __name__ == "__main__":
    main()
