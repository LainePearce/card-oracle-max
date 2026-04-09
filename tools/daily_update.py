#!/usr/bin/env python3
"""
Daily routine embedding update for Card Oracle.

Processes the last N days of RDS data (default: 2 days) to catch
any late-arriving records. Runs as a systemd timer at 02:00 UTC daily
on a single dedicated GPU worker.

Each calendar day is treated as an independent unit:
  - Before processing a day, an S3 completion marker is checked.
  - If the marker exists the day is skipped (idempotent — safe to re-run).
  - After successful processing the marker is written.

The 2-day default lookback means today's run processes yesterday AND
the day before, catching any records that arrived with a pipeline delay.

Usage (called by systemd daily-update.timer):
    python tools/daily_update.py

Manual invocation:
    # Process default lookback (yesterday + day before)
    python tools/daily_update.py

    # Process a specific date only
    python tools/daily_update.py --date 2026-04-08

    # Extend lookback window (e.g. after a holiday/outage)
    python tools/daily_update.py --lookback-days 7

    # Dry run — encode but write nothing
    python tools/daily_update.py --dry-run

The script exits 0 when all days in the window are complete.
If any day fails (rds_batch_job exits non-zero) it exits non-zero so
systemd OnFailure= can alert and the next timer run will retry.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from loguru import logger

# ── Configuration ──────────────────────────────────────────────────────────────

# How many days back to process by default.
# 2 catches yesterday + the day before (handles late-arriving RDS data).
DEFAULT_LOOKBACK_DAYS = 2

# Batch size passed to rds_batch_job — one day of data fits comfortably in RAM
# at this mini-batch granularity.
BATCH_SIZE = 256


# ── S3 completion markers ──────────────────────────────────────────────────────

def _marker_key(day: date) -> str:
    prefix = os.environ.get("S3_CHECKPOINT_PREFIX", "checkpoints")
    return f"{prefix}/daily-{day}-complete.json"


def _s3_client():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))


def is_day_complete(bucket: str, day: date) -> bool:
    try:
        _s3_client().head_object(Bucket=bucket, Key=_marker_key(day))
        return True
    except Exception:
        return False


def mark_day_complete(bucket: str, day: date, processed: int) -> None:
    key = _marker_key(day)
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps({
            "completed":  True,
            "date":       str(day),
            "processed":  processed,
            "job":        "daily-update",
        }).encode(),
    )
    logger.info("Completion marker → s3://{}/{}", bucket, key)


# ── Day runner ─────────────────────────────────────────────────────────────────

def run_day(day: date, dry_run: bool = False) -> int:
    """
    Invoke rds_batch_job for a single calendar day.
    start_date is inclusive; end_date is exclusive (next day).
    Returns the process exit code.
    """
    start = str(day)
    end   = str(day + timedelta(days=1))
    checkpoint_id = f"daily-{day}"

    cmd = [
        sys.executable, "-m", "src.embeddings.rds_batch_job",
        "--start-date",    start,
        "--end-date",      end,
        "--batch-size",    str(BATCH_SIZE),
        "--checkpoint-id", checkpoint_id,
        "--image-device",  "cuda",
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Executing: {}", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_ROOT))
    return result.returncode


# ── Main ───────────────────────────────────────────────────────────────────────

def build_window(args: argparse.Namespace) -> list[date]:
    """Return list of calendar dates to process, oldest first."""
    if args.date:
        target = date.fromisoformat(args.date)
        return [target]

    today  = date.today()
    # Process from (today - lookback_days) up to (but not including) today.
    # e.g. lookback=2 on 2026-04-09 → [2026-04-07, 2026-04-08]
    return [
        today - timedelta(days=i)
        for i in range(args.lookback_days, 0, -1)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Daily routine embedding update")
    p.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help=f"Number of past days to process (default: {DEFAULT_LOOKBACK_DAYS}). "
             "Increase after an outage to backfill missed days.",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Process a single specific date (YYYY-MM-DD). Overrides --lookback-days.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Encode vectors but skip all S3 and Qdrant writes.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-process days even if S3 completion marker exists.",
    )
    args = p.parse_args()

    bucket = os.environ.get("S3_VECTOR_BUCKET", "")
    if not bucket:
        logger.error("S3_VECTOR_BUCKET not set")
        sys.exit(1)

    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    window = build_window(args)
    logger.info(
        "Daily update — processing {} day(s): {} → {}",
        len(window), window[0], window[-1],
    )

    failed_days: list[date] = []

    for day in window:
        logger.info("=" * 60)
        logger.info("Processing {}", day)
        logger.info("=" * 60)

        if not args.force and is_day_complete(bucket, day):
            logger.info("{} already complete — skipping", day)
            continue

        exit_code = run_day(day, dry_run=args.dry_run)

        if exit_code != 0:
            logger.error("{} failed (exit {})", day, exit_code)
            failed_days.append(day)
            # Continue to remaining days rather than aborting — a single bad day
            # (e.g. no data) shouldn't block subsequent days from being processed.
            continue

        if not args.dry_run:
            mark_day_complete(bucket, day, processed=0)  # rds_batch_job logs count internally

        logger.info("{} complete ✓", day)

    logger.info("=" * 60)
    if failed_days:
        logger.error("Daily update finished with {} failed day(s): {}", len(failed_days), failed_days)
        sys.exit(1)
    else:
        logger.info("Daily update complete — all {} day(s) processed ✓", len(window))


if __name__ == "__main__":
    main()
