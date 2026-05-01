#!/usr/bin/env python3
"""
Non-eBay backfill worker — covers Jan 1 2025 to today in monthly windows,
newest first. Workers 6–11 (passed as --worker-index 6 through 11) each own
a slice of every month, advancing month by month together.

Sources covered: Fanatics/PWCC, Pristine, Goldin, MySlabs, Heritage.
In the salesdata table these appear as source_feed values:
  PWCC, FANATICS, PRISTINE, GOLDIN, MYSLABS, HERITAGE

Task IDs use the scheme  nonebay-m{YYYY}{MM}w{N}  to avoid collision
with the eBay cleanup task IDs (m{YYYY}{MM}w{N}).

Usage:
    python tools/worker_nonebay.py --worker-index <6-11>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from loguru import logger


# ── Date range ──────────────────────────────────────────────────────────────────

BACKFILL_START = date(2025, 1, 1)
BACKFILL_END   = date.today() + timedelta(days=1)   # exclusive — includes today

# Workers 6–11 are assigned to this script.
# Internally we index them 0–5 so that slice assignment is consistent.
FIRST_WORKER_INDEX = 6
N_WORKERS          = 6

# SQL filter passed as --extra-where to rds_batch_job.
# source_feed values are stored in mixed/lowercase in the DB (e.g. 'pwcc', 'MySlabs').
# Exclude 'ebay'/'EBAY' and NULL/empty/'NA' rows; everything else is a non-eBay source.
NONEBAY_WHERE = (
    "LOWER(source_feed) NOT IN ('ebay', 'na', '') AND source_feed IS NOT NULL"
)


# ── Task generation ─────────────────────────────────────────────────────────────

def _monthly_windows() -> list[tuple[date, date]]:
    """
    Generate (start, end) windows, one per calendar month, newest-first.
    Each window is [month_start, next_month_start) clipped to [BACKFILL_START, BACKFILL_END).
    """
    windows: list[tuple[date, date]] = []
    cur_end = BACKFILL_END
    while cur_end > BACKFILL_START:
        last_day = cur_end - timedelta(days=1)
        month_start = date(last_day.year, last_day.month, 1)
        win_start = max(BACKFILL_START, month_start)
        windows.append((win_start, cur_end))
        cur_end = month_start
    return windows   # already sorted newest-first


def _split_range(start: date, end: date, n: int) -> list[tuple[date, date]]:
    """Split [start, end) into n roughly equal sub-ranges."""
    total = (end - start).days
    base  = total // n
    extra = total % n
    slices: list[tuple[date, date]] = []
    cur = start
    for i in range(n):
        days = base + (1 if i < extra else 0)
        nxt  = cur + timedelta(days=days)
        slices.append((cur, min(nxt, end)))
        cur = nxt
    return slices


def _build_tasks() -> list[tuple[date, date, str]]:
    """
    Full task list ordered so all N_WORKERS slices of the newest month
    come first, then all slices of the next month, etc.
    """
    tasks: list[tuple[date, date, str]] = []
    for win_start, win_end in _monthly_windows():
        month_tag = win_start.strftime("%Y%m")
        slices = _split_range(win_start, win_end, N_WORKERS)
        for w_idx, (s, e) in enumerate(slices):
            if (e - s).days == 0:
                continue   # skip zero-day slices
            cid = f"nonebay-m{month_tag}w{w_idx}"
            tasks.append((s, e, cid))
    return tasks


ALL_TASKS = _build_tasks()


# ── S3 helpers ──────────────────────────────────────────────────────────────────

def _s3():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))


def _key(prefix: str, cid: str, suffix: str) -> str:
    return f"{prefix}/{cid}{suffix}"


def is_complete(bucket: str, prefix: str, cid: str) -> bool:
    try:
        _s3().head_object(Bucket=bucket, Key=_key(prefix, cid, "-complete.json"))
        return True
    except Exception:
        return False


def is_claimed(bucket: str, prefix: str, cid: str) -> bool:
    try:
        _s3().head_object(Bucket=bucket, Key=_key(prefix, cid, "-claimed.json"))
        return True
    except Exception:
        return False


def claim(bucket: str, prefix: str, cid: str, worker_idx: int) -> bool:
    """Write a claim marker; brief pause then verify (idempotent, safe if two workers race)."""
    try:
        _s3().put_object(
            Bucket=bucket,
            Key=_key(prefix, cid, "-claimed.json"),
            Body=json.dumps({
                "worker":     worker_idx + FIRST_WORKER_INDEX,
                "claimed_at": datetime.now(timezone.utc).isoformat(),
            }).encode(),
        )
        time.sleep(1)
        return True
    except Exception:
        return False


def mark_complete(bucket: str, prefix: str, cid: str, start: date, end: date) -> None:
    _s3().put_object(
        Bucket=bucket,
        Key=_key(prefix, cid, "-complete.json"),
        Body=json.dumps({
            "completed": True,
            "start":     str(start),
            "end":       str(end),
        }).encode(),
    )


# ── Task execution ──────────────────────────────────────────────────────────────

def run_task(start: date, end: date, cid: str) -> int:
    cmd = [
        sys.executable, "-m", "src.embeddings.rds_batch_job",
        "--start-date",              str(start),
        "--end-date",                str(end),
        "--batch-size",              "256",
        "--checkpoint-id",           cid,
        "--image-device",            "cuda",
        "--extra-where",             NONEBAY_WHERE,
        # Secondary RDS (old eBay DB) lacks source_feed column — query it
        # without any filter so gap rows are still captured and deduped by ID.
        "--no-secondary-extra-where",
    ]
    logger.info("Running: {}", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(_ROOT)).returncode


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Non-eBay backfill worker (workers 6–11)"
    )
    p.add_argument(
        "--worker-index", type=int, required=True,
        help="Global worker index (6–11).",
    )
    args = p.parse_args()

    idx = args.worker_index
    if idx < FIRST_WORKER_INDEX or idx >= FIRST_WORKER_INDEX + N_WORKERS:
        logger.error(
            "worker-index must be {}-{}, got {}",
            FIRST_WORKER_INDEX, FIRST_WORKER_INDEX + N_WORKERS - 1, idx,
        )
        sys.exit(1)

    # Translate global index (6–11) to local index (0–5) for slice assignment
    local_idx = idx - FIRST_WORKER_INDEX

    bucket = os.environ.get("S3_VECTOR_BUCKET", "")
    if not bucket:
        logger.error("S3_VECTOR_BUCKET not set")
        sys.exit(1)

    prefix = os.environ.get("S3_CHECKPOINT_PREFIX", "checkpoints")

    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    # Stagger startup to reduce S3 listing races
    startup_delay = local_idx * 2
    if startup_delay:
        logger.info("Staggered startup: sleeping {}s", startup_delay)
        time.sleep(startup_delay)

    # Rotate task list so each worker starts at its own slice of the newest month
    rotated = ALL_TASKS[local_idx % len(ALL_TASKS):] + ALL_TASKS[:local_idx % len(ALL_TASKS)]

    logger.info(
        "Non-eBay worker {} (local {}) starting — {} total tasks across {} monthly windows",
        idx, local_idx, len(ALL_TASKS), len(_monthly_windows()),
    )
    logger.info("Date range: {} → {}", BACKFILL_START, BACKFILL_END - timedelta(days=1))
    logger.info("Filter: {}", NONEBAY_WHERE)

    ran = 0
    for start, end, cid in rotated:
        if is_complete(bucket, prefix, cid):
            logger.debug("  {} already complete — skip", cid)
            continue

        if is_claimed(bucket, prefix, cid):
            logger.debug("  {} claimed by another worker — skip", cid)
            continue

        claim(bucket, prefix, cid, local_idx)

        # Re-check: another worker may have completed it in the claim window
        if is_complete(bucket, prefix, cid):
            logger.info("  {} completed by another worker just now — skip", cid)
            continue

        logger.info("=" * 60)
        logger.info("Claimed  {} → {}  [{}]", start, end, cid)
        logger.info("=" * 60)

        exit_code = run_task(start, end, cid)
        if exit_code != 0:
            logger.error("Task {} failed (exit {}). Exiting.", cid, exit_code)
            sys.exit(exit_code)

        mark_complete(bucket, prefix, cid, start, end)
        logger.info("Task {} complete ✓", cid)
        ran += 1

    if ran == 0:
        logger.info("Worker {}: all tasks already claimed or complete.", idx)
    else:
        logger.info("Worker {}: finished {} tasks.", idx, ran)


if __name__ == "__main__":
    main()
