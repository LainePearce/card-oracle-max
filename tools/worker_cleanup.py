#!/usr/bin/env python3
"""
Dynamic cleanup worker — covers the full backfill range in monthly windows,
newest first. All 12 workers process the same monthly window simultaneously
(each taking 1/12th of that month's days), then advance to the previous month.

Task IDs use the scheme  m{YYYY}{MM}w{N}  (e.g. m202603w0 = March 2026, worker 0).
Each task has an S3 claim marker and a completion marker so workers never
double-process a slice across restarts.

Usage:
    python tools/worker_cleanup.py --worker-index <0-11>
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
BACKFILL_END   = date(2026, 4, 8)   # exclusive upper bound (last priority phase end)
N_WORKERS      = 12


# ── Task generation ─────────────────────────────────────────────────────────────

def _monthly_windows() -> list[tuple[date, date]]:
    """
    Generate (start, end) windows, one per calendar month, newest-first.
    Each window is [month_start, next_month_start) clipped to [BACKFILL_START, BACKFILL_END).
    """
    windows: list[tuple[date, date]] = []
    cur_end = BACKFILL_END
    while cur_end > BACKFILL_START:
        # Walk backward one day to find the month we're currently ending in
        last_day = cur_end - timedelta(days=1)
        month_start = date(last_day.year, last_day.month, 1)
        win_start = max(BACKFILL_START, month_start)
        windows.append((win_start, cur_end))
        cur_end = month_start   # exclusive end for the next (earlier) window
    return windows              # already sorted newest-first


def _split_range(start: date, end: date, n: int) -> list[tuple[date, date]]:
    """Split [start, end) into n roughly equal sub-ranges (mirrors worker_phases.py)."""
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
    Full task list ordered so that all N_WORKERS slices of the newest month
    come first, then all slices of the next month, etc.

    This means 12 workers each claiming one slice will naturally all work on
    the same (newest) month before any of them touch the previous month.
    """
    tasks: list[tuple[date, date, str]] = []
    for win_start, win_end in _monthly_windows():
        month_tag = win_start.strftime("%Y%m")   # e.g. "202603" for March 2026
        slices = _split_range(win_start, win_end, N_WORKERS)
        for w_idx, (s, e) in enumerate(slices):
            if (e - s).days == 0:
                continue   # skip zero-day slices (can happen when month < N_WORKERS days)
            cid = f"m{month_tag}w{w_idx}"
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
    """Write a claim marker; brief pause then continue (idempotent, safe if two workers race)."""
    try:
        _s3().put_object(
            Bucket=bucket,
            Key=_key(prefix, cid, "-claimed.json"),
            Body=json.dumps({
                "worker":     worker_idx,
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
        Body=json.dumps({"completed": True, "start": str(start), "end": str(end)}).encode(),
    )


# ── Task execution ──────────────────────────────────────────────────────────────

EBAY_WHERE = "LOWER(source_feed) = 'ebay'"


def run_task(start: date, end: date, cid: str) -> int:
    cmd = [
        sys.executable, "-m", "src.embeddings.rds_batch_job",
        "--start-date",    str(start),
        "--end-date",      str(end),
        "--batch-size",    "256",
        "--checkpoint-id", cid,
        "--image-device",  "cuda",
        "--extra-where",   EBAY_WHERE,
    ]
    logger.info("Running: {}", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(_ROOT)).returncode


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker-index", type=int, required=True)
    args = p.parse_args()
    idx = args.worker_index

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

    # Stagger startup so workers don't all race for the same first task
    startup_delay = idx * 2
    if startup_delay:
        logger.info("Staggered startup: sleeping {}s", startup_delay)
        time.sleep(startup_delay)

    # Rotate the task list so each worker's "first" task is its own monthly slice.
    # Workers 0–11 each start at their respective slice of the newest month,
    # then collectively advance month by month.
    rotated = ALL_TASKS[idx % len(ALL_TASKS):] + ALL_TASKS[:idx % len(ALL_TASKS)]

    logger.info(
        "Cleanup worker {} starting — {} total tasks across {} monthly windows",
        idx, len(ALL_TASKS), len(_monthly_windows()),
    )

    ran = 0
    for start, end, cid in rotated:
        if is_complete(bucket, prefix, cid):
            logger.debug("  {} already complete — skip", cid)
            continue

        if is_claimed(bucket, prefix, cid):
            logger.debug("  {} claimed by another worker — skip", cid)
            continue

        claim(bucket, prefix, cid, idx)

        # Re-check: another worker may have completed it in the window between checks
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
