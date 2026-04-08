#!/usr/bin/env python3
"""
Multi-phase backfill orchestrator for a single GPU worker.

Phases run in priority order (most recent first, then working backwards in
3-month windows). Each phase's date range is split evenly across N_WORKERS.
A completion marker written to S3 prevents re-running a finished phase if
the worker process restarts.

Usage (called by systemd backfill.service):
    python tools/worker_phases.py --worker-index 0

The script exits 0 when all phases are complete.
If a phase fails (non-zero exit from rds_batch_job), it exits non-zero so
systemd Restart=on-failure will retry from the failed phase.
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

N_WORKERS = 12

# Phases in execution order — most recent data first, then work backwards.
# Each entry is (inclusive_start, exclusive_end).
PHASES: list[tuple[date, date]] = [
    (date(2026, 1, 1),  date(2026, 4, 8)),   # Phase 1: 2026 Q1  ← current priority
    (date(2025, 10, 1), date(2026, 1, 1)),   # Phase 2: 2025 Q4
    (date(2025, 7, 1),  date(2025, 10, 1)),  # Phase 3: 2025 Q3
    (date(2025, 4, 1),  date(2025, 7, 1)),   # Phase 4: 2025 Q2
    (date(2025, 1, 1),  date(2025, 4, 1)),   # Phase 5: 2025 Q1
]


# ── Date splitting ─────────────────────────────────────────────────────────────

def split_range(start: date, end: date, n: int) -> list[tuple[date, date]]:
    """Split [start, end) into n roughly equal sub-ranges."""
    total = (end - start).days
    base  = total // n
    extra = total % n
    ranges: list[tuple[date, date]] = []
    cur = start
    for i in range(n):
        days = base + (1 if i < extra else 0)
        nxt  = cur + timedelta(days=days)
        ranges.append((cur, min(nxt, end)))
        cur  = nxt
    return ranges


# ── S3 completion markers ─────────────────────────────────────────────────────

def _marker_key(worker_idx: int, phase_num: int) -> str:
    prefix = os.environ.get("S3_CHECKPOINT_PREFIX", "checkpoints")
    return f"{prefix}/backfill-w{worker_idx}-phase{phase_num}-complete.json"


def _s3_client():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))


def is_phase_complete(bucket: str, worker_idx: int, phase_num: int) -> bool:
    try:
        _s3_client().head_object(Bucket=bucket, Key=_marker_key(worker_idx, phase_num))
        return True
    except Exception:
        return False


def mark_phase_complete(
    bucket: str, worker_idx: int, phase_num: int, start: date, end: date
) -> None:
    key = _marker_key(worker_idx, phase_num)
    _s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps({
            "completed": True,
            "worker": worker_idx,
            "phase": phase_num,
            "start": str(start),
            "end": str(end),
        }).encode(),
    )
    logger.info("Completion marker → s3://{}/{}", bucket, key)


# ── Phase runner ───────────────────────────────────────────────────────────────

def run_phase(worker_idx: int, phase_num: int, start: date, end: date) -> int:
    """Invoke rds_batch_job for one phase sub-range. Returns exit code."""
    cmd = [
        sys.executable, "-m", "src.embeddings.rds_batch_job",
        "--start-date",    str(start),
        "--end-date",      str(end),
        "--batch-size",    "256",
        "--checkpoint-id", f"backfill-w{worker_idx}-phase{phase_num}",
        "--image-device",  "cuda",
    ]
    logger.info("Executing: {}", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_ROOT))
    return result.returncode


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Multi-phase GPU backfill orchestrator")
    p.add_argument(
        "--worker-index", type=int, required=True,
        help=f"0-based worker index (0–{N_WORKERS - 1})",
    )
    args = p.parse_args()
    idx  = args.worker_index

    if not 0 <= idx < N_WORKERS:
        logger.error("--worker-index must be 0–{}", N_WORKERS - 1)
        sys.exit(1)

    bucket = os.environ.get("S3_VECTOR_BUCKET", "")
    if not bucket:
        logger.error("S3_VECTOR_BUCKET not set")
        sys.exit(1)

    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    logger.info("Worker {} starting — {} phases to run", idx, len(PHASES))

    for phase_num, (phase_start, phase_end) in enumerate(PHASES, 1):
        ranges       = split_range(phase_start, phase_end, N_WORKERS)
        start, end   = ranges[idx]

        logger.info("=" * 60)
        logger.info(
            "Phase {}/{}: {} → {} | worker {}: {} → {}",
            phase_num, len(PHASES), phase_start, phase_end, idx, start, end,
        )
        logger.info("=" * 60)

        if is_phase_complete(bucket, idx, phase_num):
            logger.info("Phase {} already complete — skipping", phase_num)
            continue

        exit_code = run_phase(idx, phase_num, start, end)

        if exit_code != 0:
            logger.error(
                "Phase {} failed (exit {}). systemd will retry.", phase_num, exit_code
            )
            sys.exit(exit_code)

        mark_phase_complete(bucket, idx, phase_num, start, end)
        logger.info("Phase {} complete ✓", phase_num)

    logger.info("All {} phases complete for worker {}. Exiting.", len(PHASES), idx)


if __name__ == "__main__":
    main()
