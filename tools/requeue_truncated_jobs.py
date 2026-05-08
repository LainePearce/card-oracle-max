#!/usr/bin/env python3
"""
Identify completed backfill jobs that were truncated by the scroll-cursor
expiry bug and move them back to the queue for reprocessing.

Background
----------
Before the search_after fix, jobs used OpenSearch scroll cursors with a
5-minute TTL.  CLIP encoding took longer than 5 minutes between pages, so
the cursor expired silently after ~11 pages (~5,500 docs).  The job was then
marked 'complete' even though only a fraction of its window was processed.

Detection heuristic
-------------------
A job is considered truncated if ALL of the following are true:
  - Its checkpoint exists and shows  scrolled  <=  SCROLLED_CEILING
  - Its job definition shows          doc_count_estimate  >  ESTIMATE_FLOOR
  - The ratio  scrolled / doc_count_estimate  <  COVERAGE_THRESHOLD

Defaults:
  SCROLLED_CEILING     = 6,000    (just above the 11-page scroll cap)
  ESTIMATE_FLOOR       = 10,000   (ignore genuinely tiny jobs)
  COVERAGE_THRESHOLD   = 0.50     (less than 50% coverage → truncated)

You can also pass  --requeue-all  to re-queue every completed job regardless
of stats (safe because workers skip Qdrant-present IDs, just slower).

Usage
-----
    # Preview — show truncated jobs without making any changes
    python tools/requeue_truncated_jobs.py --dry-run

    # Re-queue truncated jobs only (recommended first pass)
    python tools/requeue_truncated_jobs.py

    # Re-queue ALL completed jobs (brute-force safety net)
    python tools/requeue_truncated_jobs.py --requeue-all

    # Only look at P1 priority jobs
    python tools/requeue_truncated_jobs.py --priority 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import boto3
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────

S3_BUCKET        = os.environ["S3_VECTOR_BUCKET"]
S3_QUEUE_PREFIX  = "backfill-v2/queue"
S3_COMP_PREFIX   = "backfill-v2/complete"
S3_CKPT_PREFIX   = "backfill-v2/checkpoints"

# Truncation detection thresholds (tunable via CLI)
SCROLLED_CEILING    = 6_000    # max docs a scroll-limited job could process
ESTIMATE_FLOOR      = 10_000   # ignore jobs this small (they may be genuinely tiny)
COVERAGE_THRESHOLD  = 0.50     # fraction of estimate — below this = truncated


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _list_keys(s3, prefix: str) -> list[str]:
    pag = s3.get_paginator("list_objects_v2")
    keys = []
    for page in pag.paginate(Bucket=S3_BUCKET, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _read_json(s3, key: str) -> dict | None:
    try:
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return json.loads(body)
    except Exception:
        return None


def _move_to_queue(s3, job: dict, complete_key: str, dry_run: bool) -> None:
    """Copy job JSON from complete/ to queue/ and delete the complete record."""
    queue_key = f"{S3_QUEUE_PREFIX}/{job['job_id']}.json"
    if dry_run:
        logger.info("  [DRY] would re-queue: {}", job["job_id"])
        return
    # Write to queue (re-uses original job dict — no mutation)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=queue_key,
        Body=json.dumps(job, indent=2).encode(),
        ContentType="application/json",
    )
    # Remove from complete
    s3.delete_object(Bucket=S3_BUCKET, Key=complete_key)
    logger.info("  Re-queued: {}", job["job_id"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Re-queue truncated backfill jobs")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Show what would happen without making changes")
    ap.add_argument("--requeue-all",  action="store_true",
                    help="Re-queue every completed job regardless of stats")
    ap.add_argument("--priority",     type=int, nargs="+",
                    help="Only consider jobs matching these priority levels")
    ap.add_argument("--scrolled-ceiling", type=int, default=SCROLLED_CEILING,
                    help=f"Max scrolled count to consider truncated (default {SCROLLED_CEILING})")
    ap.add_argument("--estimate-floor",   type=int, default=ESTIMATE_FLOOR,
                    help=f"Min doc_count_estimate to inspect (default {ESTIMATE_FLOOR})")
    ap.add_argument("--coverage-threshold", type=float, default=COVERAGE_THRESHOLD,
                    help=f"Fraction threshold below which job is truncated (default {COVERAGE_THRESHOLD})")
    args = ap.parse_args()

    allowed_priorities = set(args.priority) if args.priority else None

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))

    logger.info("Listing completed jobs...")
    complete_keys = _list_keys(s3, S3_COMP_PREFIX)
    logger.info("Found {} completed jobs", len(complete_keys))

    requeued = skipped_ok = skipped_no_ckpt = skipped_priority = 0

    for comp_key in sorted(complete_keys):
        job = _read_json(s3, comp_key)
        if not job:
            logger.warning("Could not read job: {}", comp_key)
            continue

        job_id   = job.get("job_id", "")
        priority = job.get("priority", 99)
        estimate = job.get("doc_count_estimate", 0)

        # Priority filter
        if allowed_priorities and priority not in allowed_priorities:
            skipped_priority += 1
            continue

        # --requeue-all: skip stats check entirely
        if args.requeue_all:
            _move_to_queue(s3, job, comp_key, args.dry_run)
            requeued += 1
            continue

        # Small jobs: don't bother inspecting — probably were genuinely tiny
        if estimate < args.estimate_floor:
            logger.debug("Skipping {} — estimate {} below floor", job_id, estimate)
            skipped_ok += 1
            continue

        # Load checkpoint to get actual scrolled count
        ckpt_key = f"{S3_CKPT_PREFIX}/{job_id}.json"
        ckpt = _read_json(s3, ckpt_key)
        if not ckpt:
            # No checkpoint: job may have been processed without one (very old run)
            # or failed before writing a checkpoint.  Re-queue conservatively.
            logger.debug("No checkpoint for {} — re-queueing conservatively", job_id)
            _move_to_queue(s3, job, comp_key, args.dry_run)
            requeued += 1
            skipped_no_ckpt += 1
            continue

        scrolled = ckpt.get("stats", {}).get("scrolled", 0)
        coverage = scrolled / estimate if estimate else 1.0

        truncated = (
            scrolled <= args.scrolled_ceiling
            and coverage < args.coverage_threshold
        )

        if truncated:
            logger.info(
                "TRUNCATED {} | priority={} scrolled={:,} estimate={:,} coverage={:.1%}",
                job_id, priority, scrolled, estimate, coverage,
            )
            _move_to_queue(s3, job, comp_key, args.dry_run)
            requeued += 1
        else:
            logger.debug(
                "OK        {} | scrolled={:,} estimate={:,} coverage={:.1%}",
                job_id, scrolled, estimate, coverage,
            )
            skipped_ok += 1

    logger.info(
        "Done.  requeued={}  skipped_ok={}  skipped_no_ckpt={}  skipped_priority={}",
        requeued, skipped_ok, skipped_no_ckpt, skipped_priority,
    )
    if args.dry_run and requeued:
        logger.info("(dry-run — no changes made)")


if __name__ == "__main__":
    main()
