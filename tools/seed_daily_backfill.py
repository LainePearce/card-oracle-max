#!/usr/bin/env python3
"""
Per-day backfill seeder with date-block priorities.

Generates 12 windows per eBay-dated day across three priority blocks:

  P1 (priority base 1_000): 2026-05-08 → today      — zero S3 coverage
  P2 (priority base 2_000): 2026-05-07 → 2026-01-01 — partial S3 coverage
  P3 (priority base 3_000): 2025-12-31 → 2025-01-01 — backfill 2025

Within each block, days are seeded in DESC order (latest first). The priority
number = block_base + offset_from_block_start. With 12 windows per day, all
12 windows for the current day are claimable before any next-day window —
satisfies the "all workers on the same day at once" requirement, because
workers always claim the lowest priority number available.

Idempotent: dates that already have a queued/active/complete/failed job for
window 00 are skipped entirely. Dates whose OS index does not exist or has
zero docs are also skipped.

Usage:
    python tools/seed_daily_backfill.py                       # all three blocks
    python tools/seed_daily_backfill.py --blocks 1            # only P1
    python tools/seed_daily_backfill.py --blocks 1 2          # P1 + P2
    python tools/seed_daily_backfill.py --dry-run --blocks 1  # preview
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import boto3
from loguru import logger
from opensearchpy.exceptions import NotFoundError

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.ingestion.opensearch_reader import (
    get_opensearch_client, classify_index,
)


S3_BUCKET        = os.environ["S3_VECTOR_BUCKET"]
S3_QUEUE_PREFIX  = "backfill-v2/queue"
S3_ACTIVE_PREFIX = "backfill-v2/active"
S3_COMP_PREFIX   = "backfill-v2/complete"
S3_FAIL_PREFIX   = "backfill-v2/failed"

NUM_WORKERS = 12

# Priority block bases. Lower = higher urgency. 1_000 spacing gives room for
# ~1_000 days per block. The block label rides along on the job for telemetry.
BLOCK_BASES = {1: 1_000, 2: 2_000, 3: 3_000}


@dataclass(frozen=True)
class Block:
    label:      str
    block_id:   int          # 1, 2, 3
    start_date: date         # inclusive (earlier date)
    end_date:   date         # inclusive (later date)


def get_blocks(today: date) -> list[Block]:
    """Three priority blocks per the audit-driven plan."""
    return [
        Block("P1", 1, date(2026, 5,  8), today),
        Block("P2", 2, date(2026, 1,  1), date(2026, 5, 7)),
        Block("P3", 3, date(2025, 1,  1), date(2025, 12, 31)),
    ]


def block_dates_descending(b: Block) -> list[date]:
    """All dates in the block, latest first."""
    days = (b.end_date - b.start_date).days + 1
    return [b.end_date - timedelta(days=i) for i in range(days)]


def s3_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def date_already_seeded(s3, index_name: str) -> bool:
    """Check if window 00 for this date is already in any state."""
    job_id = f"{index_name}_w00"
    for prefix in (S3_QUEUE_PREFIX, S3_ACTIVE_PREFIX,
                   S3_COMP_PREFIX,  S3_FAIL_PREFIX):
        if s3_exists(s3, f"{prefix}/{job_id}.json"):
            return True
    return False


def compute_day_windows(os_client, index_name: str,
                        n_workers: int) -> list[dict] | None:
    """
    Hourly date_histogram, then split into n_workers windows of roughly
    equal doc_count. Returns a list of {ts_start, ts_end, doc_count_estimate}
    or None if the index has no docs.
    """
    try:
        resp = os_client.search(
            index = index_name,
            body  = {
                "size": 0,
                "aggs": {
                    "by_hour": {
                        "date_histogram": {
                            "field":          "endTime",
                            "fixed_interval": "1h",
                            "format":         "yyyy-MM-dd HH:mm:ss",
                            "min_doc_count":  1,
                        }
                    }
                },
            },
            request_timeout=30,
        )
    except NotFoundError:
        return None
    except Exception as exc:
        logger.warning("Aggregation failed for {}: {}", index_name, exc)
        return None

    buckets = resp["aggregations"]["by_hour"]["buckets"]
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
            threshold = target * (len(cutoffs) + 1)
            if cumulative >= threshold and i + 1 < len(buckets):
                cutoffs.append(buckets[i + 1]["key_as_string"])

    actual_n  = len(cutoffs) + 1
    boundaries = [None] + cutoffs + [None]
    out = []
    for w in range(actual_n):
        out.append({
            "ts_start":           boundaries[w],
            "ts_end":              boundaries[w + 1],
            "doc_count_estimate":  total // actual_n,
        })
    return out


def seed_day(s3, os_client, index_name: str, block: Block,
             day_offset: int, dry_run: bool) -> tuple[int, str]:
    """
    Seed one day. Returns (jobs_seeded, status).

    status ∈ {"seeded", "skip:already-handled",
              "skip:no-os-index", "skip:no-docs",
              "skip:zero-windows"}
    """
    if date_already_seeded(s3, index_name):
        return (0, "skip:already-handled")

    windows = compute_day_windows(os_client, index_name, NUM_WORKERS)
    if windows is None:
        return (0, "skip:no-os-index")
    if not windows:
        return (0, "skip:zero-windows")

    priority = BLOCK_BASES[block.block_id] + day_offset
    seeded = 0

    for w_idx, window in enumerate(windows):
        job_id = f"{index_name}_w{w_idx:02d}"
        job = {
            "job_id":              job_id,
            "priority":            priority,
            "block":               block.label,
            "block_id":            block.block_id,
            "day_offset":          day_offset,
            "index_name":          index_name,
            "index_type":          "ebay-dated",
            "partition":           index_name,
            "ts_start":            window["ts_start"],
            "ts_end":              window["ts_end"],
            "doc_count_estimate":  window["doc_count_estimate"],
            "created_at":          datetime.now(timezone.utc).isoformat(),
        }
        if dry_run:
            logger.info("[dry-run] would seed {} priority={}", job_id, priority)
        else:
            s3.put_object(
                Bucket = S3_BUCKET,
                Key    = f"{S3_QUEUE_PREFIX}/{job_id}.json",
                Body   = json.dumps(job).encode(),
            )
        seeded += 1

    return (seeded, "seeded")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", type=int, nargs="+",
                    default=[1, 2, 3], choices=[1, 2, 3],
                    help="Priority blocks to seed (default: all three).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned jobs without writing to S3.")
    ap.add_argument("--limit-days", type=int, default=None,
                    help="Only seed the first N days of each block (testing).")
    args = ap.parse_args()

    s3        = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))
    os_client = get_opensearch_client()
    today     = date.today()

    total_jobs        = 0
    total_day_attempts = 0
    status_counts: dict[str, int] = {}

    for b in get_blocks(today):
        if b.block_id not in args.blocks:
            continue
        logger.info("── Block {} ({}): {} → {} ─────────────",
                    b.block_id, b.label, b.end_date, b.start_date)

        dates = block_dates_descending(b)
        if args.limit_days:
            dates = dates[:args.limit_days]

        block_seeded = 0
        for offset, d in enumerate(dates):
            n, status = seed_day(s3, os_client, d.isoformat(),
                                  b, day_offset=offset, dry_run=args.dry_run)
            status_counts[status] = status_counts.get(status, 0) + 1
            total_day_attempts += 1
            block_seeded += n
            total_jobs    += n
            if status == "seeded":
                logger.info("  {} → seeded {} jobs (priority {})",
                            d.isoformat(), n,
                            BLOCK_BASES[b.block_id] + offset)
            else:
                logger.info("  {} → {}", d.isoformat(), status)

        logger.info("Block {} total: {} jobs seeded", b.label, block_seeded)

    logger.info("──────────────────────────────────────────")
    logger.info("Grand total: {} jobs across {} day-attempts",
                total_jobs, total_day_attempts)
    for k in sorted(status_counts):
        logger.info("  {:30s} {}", k, status_counts[k])


if __name__ == "__main__":
    main()
