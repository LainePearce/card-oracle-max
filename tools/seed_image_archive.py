#!/usr/bin/env python3
"""
Seed the 2026 image-archive queue.

Queues every 2026 eBay-dated day (today back to 2026-01-01) into
image-archive/queue, newest first, so the fleet downloads recent sales before
older ones. Idempotent — days already queued/active/complete are skipped, so
it's safe to re-run as new days accrue.

    python tools/seed_image_archive.py
    python tools/seed_image_archive.py --start 2026-01-01 --end 2026-06-20
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from tools.image_archive_common import s3_client, seed_date, reap_stale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--reap-stale-minutes", type=float, default=0,
                    help="Also re-queue active days idle longer than this "
                         "(recovers dead/spot-killed workers). 0 = off.")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).date()
    end   = datetime.fromisoformat(args.end).date()
    s3 = s3_client()

    seeded = skipped = 0
    d = end
    while d >= start:                      # newest first
        if seed_date(s3, d.isoformat()):
            seeded += 1
        else:
            skipped += 1
        d -= timedelta(days=1)

    logger.info("Image-archive queue seeded: {} new days, {} already present "
                "({} → {})", seeded, skipped, start, end)

    if args.reap_stale_minutes > 0:
        reaped = reap_stale(s3, args.reap_stale_minutes)
        logger.info("Reaped {} stale active days back to queue", reaped)


if __name__ == "__main__":
    main()
