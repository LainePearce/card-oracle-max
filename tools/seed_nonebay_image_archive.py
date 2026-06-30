#!/usr/bin/env python3
"""
Seed non-eBay marketplace indices into the image-archive queue.

eBay archival is date-partitioned (YYYY-MM-DD); the non-eBay marketplaces use
named indices instead — monthly {YYYY}-{MM}-pris / {YYYY}-{MM}-pwcc and annual
{YYYY}-heri/heritage/ms/gold. This seeds each such index (for the requested
years) as one image-archive job. The worker is index-name-agnostic — it queries
index=<job> and archives every doc's galleryURL exactly like an eBay day — and
DINOv2 embed follows automatically via the dino-embed-seed timer.

Caveats (carried over from the CLIP-era non-eBay backfill):
  - Heritage and one Goldin tier 403 their image CDN → those images are
    unfetchable regardless of pipeline.
  - Pristine listings often have galleryURL="N/A" (no image) → unembeddable.

Usage:
  python tools/seed_nonebay_image_archive.py
  python tools/seed_nonebay_image_archive.py --years 2025 2026 --include pwcc pris
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from src.ingestion.opensearch_reader import get_opensearch_client
from tools.image_archive_common import s3_client, seed_date

# Non-eBay index naming (mirrors seed_nonebay_backfill.py).
_MONTHLY = re.compile(r"^(\d{4})-\d{2}-(pris|pwcc)$")
_ANNUAL  = re.compile(r"^(\d{4})-(heri|heritage|ms|gold)$")


def discover_nonebay(os_client, years: set[str], include: set[str] | None) -> list[str]:
    cat = os_client.cat.indices(h="index", format="json", request_timeout=60)
    out = []
    for row in cat:
        n = row["index"]
        m = _MONTHLY.match(n) or _ANNUAL.match(n)
        if not m:
            continue
        year, typ = m.group(1), m.group(2)
        if year not in years:
            continue
        if include and typ not in include:
            continue
        out.append(n)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", default=["2025", "2026"])
    ap.add_argument("--include", nargs="+", default=None,
                    help="Subset of types: pris pwcc heri heritage ms gold")
    args = ap.parse_args()

    os_client = get_opensearch_client()
    indices = discover_nonebay(os_client, set(args.years),
                               set(args.include) if args.include else None)
    logger.info("Discovered {} non-eBay indices: {}", len(indices), indices)

    s3 = s3_client()
    seeded = skipped = 0
    for n in indices:
        if seed_date(s3, n):
            seeded += 1
            logger.info("  seeded {}", n)
        else:
            skipped += 1

    logger.info("Non-eBay image-archive queue: {} new, {} already present "
                "({} indices total)", seeded, skipped, len(indices))


if __name__ == "__main__":
    main()
