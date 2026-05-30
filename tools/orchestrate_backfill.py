#!/usr/bin/env python3
"""
Main loop for the per-day backfill workflow.

Designed to run continuously on worker-0 in tmux. Polls S3 for completed
backfill jobs, runs the verifier on each day where all windows have
terminated (no queued / no active), and — when the verifier returns
"complete" or "remediation_attempted" — kicks repopulate_qdrant.py with
--partition <date> to push that day's image vectors from S3 to Qdrant.

Stateless across restarts. State lives in S3:

  backfill-v2/queue       — pending jobs (workers claim from here)
  backfill-v2/active      — claimed-but-not-yet-complete jobs
  backfill-v2/complete    — completed jobs
  backfill-v2/failed      — failed jobs
  backfill-v2/verified    — per-day verifier summary (this script writes)
  backfill-v2/qdrant-pushed — per-day "vectors pushed to Qdrant" marker
                              (this script writes)

Operations on each tick:
  1. Optionally seed initial queue (--seed-on-start)
  2. Find dates ready for verification (complete jobs exist; no
     queued/active jobs remain).
  3. For each such date, if not already verified, run verify_day.
  4. If verifier returns complete or remediation_attempted, and the date
     isn't already qdrant-pushed, call repopulate_qdrant.py
     --partition <date> --index-type ebay-dated.
  5. Print progress summary.

Usage:
  python tools/orchestrate_backfill.py --seed-on-start
  python tools/orchestrate_backfill.py --no-seed     # restart, queue already seeded
  python tools/orchestrate_backfill.py --once        # one tick then exit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import boto3
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from tools.verify_day_backfill import verify_day, VerifyResult


S3_BUCKET           = os.environ["S3_VECTOR_BUCKET"]
S3_QUEUE_PFX        = "backfill-v2/queue"
S3_ACTIVE_PFX       = "backfill-v2/active"
S3_COMP_PFX         = "backfill-v2/complete"
S3_FAIL_PFX         = "backfill-v2/failed"
S3_VERIFIED_PFX     = "backfill-v2/verified"
S3_QDRANT_PUSHED_PFX = "backfill-v2/qdrant-pushed"

POLL_INTERVAL_S = 60
REPOPULATE_PY   = ROOT / "tools" / "repopulate_qdrant.py"


def _list_dates_in_prefix(s3, prefix: str) -> dict[str, int]:
    """Return {date: count} of objects under {prefix}/ keyed by date.

    Handles two naming conventions used in backfill-v2/:
      - Job-state prefixes (queue/active/complete/failed): YYYY-MM-DD_wNN.json
      - Per-day prefixes  (verified/qdrant-pushed):        YYYY-MM-DD.json

    Returns a count per date so callers can ask "how many windows per
    date are in this prefix?" — for per-day prefixes that's just 1 per
    date but counting it the same way keeps the STATUS line honest.
    """
    counts: dict[str, int] = defaultdict(int)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if not name.endswith(".json"):
                continue
            if "_w" in name:
                date = name.split("_w", 1)[0]
            else:
                date = name.removesuffix(".json")
            if len(date) == 10 and date[4] == "-" and date[7] == "-":
                counts[date] += 1
    return counts


def _s3_key_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def find_dates_ready_to_verify(s3) -> list[str]:
    """Dates where complete > 0 AND queued == 0 AND active == 0."""
    queued   = _list_dates_in_prefix(s3, S3_QUEUE_PFX)
    active   = _list_dates_in_prefix(s3, S3_ACTIVE_PFX)
    complete = _list_dates_in_prefix(s3, S3_COMP_PFX)

    ready = []
    for d, c in complete.items():
        if queued.get(d, 0) == 0 and active.get(d, 0) == 0 and c > 0:
            ready.append(d)
    return sorted(ready, reverse=True)   # latest first


def push_day_to_qdrant(date_str: str) -> bool:
    """Subprocess repopulate_qdrant.py for this partition. Returns True on success."""
    cmd = [
        sys.executable, str(REPOPULATE_PY),
        "--vector-type",     "image",
        "--index-type",      "ebay-dated",
        "--partition",       date_str,
        "--no-payload-flag",     # has_image already set; skip the extra set_payload pass
        "--checkpoint-every", "10",
        # Per-batch HNSW commit time is roughly fixed (~300-600ms regardless of
        # batch size, dominated by wait=True flush). 2000 → ~4x fewer round
        # trips than the default 500, ~4x faster per partition. Each batch is
        # still atomic so larger batch ≠ larger failure blast radius for our
        # idempotent update_vectors use case.
        "--batch-size",      "2000",
    ]
    logger.info("Qdrant push: starting subprocess for {}", date_str)
    logger.debug("cmd: {}", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, cwd=ROOT,
            capture_output=True, text=True, timeout=60 * 60,
        )
    except subprocess.TimeoutExpired:
        logger.error("Qdrant push for {} timed out after 1h", date_str)
        return False

    if result.returncode != 0:
        logger.error("Qdrant push for {} failed (rc={}): {}",
                     date_str, result.returncode,
                     (result.stderr or result.stdout)[-500:])
        return False

    # Tail of stdout is the "DONE" line — log it for the operator
    tail = "\n".join((result.stdout or "").rstrip().splitlines()[-5:])
    logger.info("Qdrant push for {} done. Tail:\n{}", date_str, tail)
    return True


def mark_qdrant_pushed(s3, date_str: str, ok: bool, summary: dict) -> None:
    key = f"{S3_QDRANT_PUSHED_PFX}/{date_str}.json"
    s3.put_object(
        Bucket = S3_BUCKET,
        Key    = key,
        Body   = json.dumps({
            "date":       date_str,
            "ok":         ok,
            "pushed_at":  datetime.now(timezone.utc).isoformat(),
            "summary":    summary,
        }).encode(),
    )


def run_tick(s3) -> dict:
    """One iteration. Returns counters."""
    counts = {
        "verified_this_tick":     0,
        "qdrant_pushed_this_tick": 0,
        "ready_dates":             0,
        "still_in_flight_dates":   0,
        "verifier_failures":       0,
        "push_failures":           0,
    }

    in_flight = _list_dates_in_prefix(s3, S3_QUEUE_PFX)
    in_flight_active = _list_dates_in_prefix(s3, S3_ACTIVE_PFX)
    all_in_flight = set(in_flight) | set(in_flight_active)
    counts["still_in_flight_dates"] = len(all_in_flight)

    ready = find_dates_ready_to_verify(s3)
    counts["ready_dates"] = len(ready)

    for date_str in ready:
        verify_key = f"{S3_VERIFIED_PFX}/{date_str}.json"
        already_verified = _s3_key_exists(s3, verify_key)
        if not already_verified:
            try:
                result: VerifyResult = verify_day(date_str, s3=s3)
                counts["verified_this_tick"] += 1
                logger.info("verify {}: {} ({:.2f}%, {} missing)",
                            date_str, result.status,
                            result.coverage_pct, result.missing_count)
            except Exception as e:
                counts["verifier_failures"] += 1
                logger.error("verify {} raised: {}: {}",
                             date_str, type(e).__name__, e)
                continue
        else:
            # Already verified — load the summary to decide on push
            obj = s3.get_object(Bucket=S3_BUCKET, Key=verify_key)
            result_dict = json.loads(obj["Body"].read())
            result = VerifyResult(**{k: v for k, v in result_dict.items()
                                     if k in VerifyResult.__dataclass_fields__})

        if result.status not in ("complete", "remediation_attempted"):
            logger.info("skip qdrant push for {}: status={}",
                        date_str, result.status)
            continue

        push_marker = f"{S3_QDRANT_PUSHED_PFX}/{date_str}.json"
        if _s3_key_exists(s3, push_marker):
            continue

        ok = push_day_to_qdrant(date_str)
        mark_qdrant_pushed(s3, date_str, ok, asdict(result))
        if ok:
            counts["qdrant_pushed_this_tick"] += 1
        else:
            counts["push_failures"] += 1

    return counts


def print_top_level_status(s3, counters: dict, started_at: datetime) -> None:
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    queued    = sum(_list_dates_in_prefix(s3, S3_QUEUE_PFX).values())
    active    = sum(_list_dates_in_prefix(s3, S3_ACTIVE_PFX).values())
    completed = sum(_list_dates_in_prefix(s3, S3_COMP_PFX).values())
    failed    = sum(_list_dates_in_prefix(s3, S3_FAIL_PFX).values())
    verified_dates = _list_dates_in_prefix(s3, S3_VERIFIED_PFX)
    pushed_dates   = _list_dates_in_prefix(s3, S3_QDRANT_PUSHED_PFX)

    logger.info("[STATUS] elapsed={:.0f}s  jobs: q={} a={} c={} f={}  "
                "dates: verified={} pushed_to_qdrant={}",
                elapsed, queued, active, completed, failed,
                len(verified_dates), len(pushed_dates))
    logger.info("[TICK ]  ready_dates={} verified_this_tick={} "
                "qdrant_pushed_this_tick={} in_flight_dates={} "
                "verifier_failures={} push_failures={}",
                counters["ready_dates"],
                counters["verified_this_tick"],
                counters["qdrant_pushed_this_tick"],
                counters["still_in_flight_dates"],
                counters["verifier_failures"],
                counters["push_failures"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-on-start", action="store_true",
                    help="Run tools/seed_daily_backfill.py before the first tick.")
    ap.add_argument("--blocks", type=int, nargs="+",
                    default=[1, 2, 3], choices=[1, 2, 3],
                    help="Priority blocks to seed (if --seed-on-start).")
    ap.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_S)
    ap.add_argument("--once", action="store_true",
                    help="Run one tick and exit.")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))

    if args.seed_on_start:
        seed_cmd = [
            sys.executable, str(ROOT / "tools" / "seed_daily_backfill.py"),
            "--blocks", *[str(b) for b in args.blocks],
        ]
        logger.info("Seeding queue: {}", " ".join(seed_cmd))
        r = subprocess.run(seed_cmd, cwd=ROOT)
        if r.returncode != 0:
            logger.error("seed_daily_backfill exited rc={}", r.returncode)
            sys.exit(1)

    started_at = datetime.now(timezone.utc)
    logger.info("Orchestrator started at {} (poll={}s, once={})",
                started_at.isoformat(), args.poll_interval, args.once)

    while True:
        try:
            counters = run_tick(s3)
            print_top_level_status(s3, counters, started_at)
        except KeyboardInterrupt:
            logger.info("Interrupted — exiting.")
            return
        except Exception as e:
            logger.exception("Tick failed: {}: {}", type(e).__name__, e)

        if args.once:
            return

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
