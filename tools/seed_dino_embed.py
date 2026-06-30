#!/usr/bin/env python3
"""
Seed the DINOv2 embed queue from archived days.

A day is embeddable once its images are in S3 — i.e. it has an
image-archive/complete marker. This queues those days (default: 2026, newest
first) into dino-embed/queue, skipping any already queued/active/embedded.
Idempotent and safe to re-run (e.g. from a timer) as archival completes more
days.

    python tools/seed_dino_embed.py                          # all archived 2026 days
    python tools/seed_dino_embed.py --start 2026-01-01 --end 2026-12-31
    python tools/seed_dino_embed.py --reap-stale-minutes 30  # also recover dead workers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from tools.dino_embed_common import s3_client, seed_date, reap_stale, list_dates, ARCHIVE_COMPLETE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--reap-stale-minutes", type=float, default=0)
    args = ap.parse_args()

    s3 = s3_client()
    # Filter by YEAR prefix, not full-string range, so non-eBay index-name
    # markers (e.g. "2025-gold", "2026-04-pwcc") are included alongside dated
    # eBay days — string range comparison mis-sorts the letter suffixes.
    y0, y1 = args.start[:4], args.end[:4]
    archived = [d for d in list_dates(s3, ARCHIVE_COMPLETE) if y0 <= d[:4] <= y1]
    archived.sort(reverse=True)   # newest first

    seeded = skipped = 0
    for d in archived:
        if seed_date(s3, d):
            seeded += 1
        else:
            skipped += 1

    logger.info("DINOv2 embed queue: {} new days seeded, {} already present "
                "(from {} archived days in {}..{})",
                seeded, skipped, len(archived), args.start, args.end)

    if args.reap_stale_minutes > 0:
        logger.info("Reaped {} stale active days", reap_stale(s3, args.reap_stale_minutes))


if __name__ == "__main__":
    main()
