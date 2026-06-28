#!/usr/bin/env python3
"""
Audit DINOv2 embed coverage for a year against image archival + Qdrant.

Reconciles three layers per day:
  archived   image-archive/complete/{date}.json   -> 'archived' (images in S3)
  embedded   dino-embed/complete/{date}.json       -> 'embedded' (vectors written)
  index      cards_dinov2 Qdrant points (collection total)

Flags every gap so you can confirm a year is fully embedded before moving on:
  - archived days with NO dino-embed complete marker (still queued/active/missing)
  - days where embedded < archived (partial)
  - the embedded total vs the live Qdrant point count

Usage:
  python tools/audit_dino_embed.py --year 2026
  python tools/audit_dino_embed.py --year 2026 --csv /tmp/dino_2026.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from tools.dino_embed_common import (
    s3_client, QUEUE_BUCKET, QUEUE, ACTIVE, COMPLETE, ARCHIVE_COMPLETE, list_dates,
)
from tools.dino_embed_worker import COLLECTION


def _read(s3, prefix, d):
    try:
        return d, json.loads(s3.get_object(Bucket=QUEUE_BUCKET, Key=f"{prefix}/{d}.json")["Body"].read())
    except Exception:
        return d, {}


def _read_all(s3, prefix, dates):
    out = {}
    if dates:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for d, m in ex.map(lambda d: _read(s3, prefix, d), dates):
                out[d] = m
    return out


def qdrant_points() -> int | None:
    try:
        from src.ingestion.qdrant_writer import get_qdrant_client
        info = get_qdrant_client().get_collection(COLLECTION)
        return info.points_count
    except Exception as e:
        logger.warning("Could not read Qdrant '{}' point count: {}", COLLECTION, e)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", default="2026")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    yr = args.year

    s3 = s3_client()
    arch_dates  = [d for d in list_dates(s3, ARCHIVE_COMPLETE) if d.startswith(yr)]
    emb_dates   = [d for d in list_dates(s3, COMPLETE) if d.startswith(yr)]
    queued      = {d for d in list_dates(s3, QUEUE)  if d.startswith(yr)}
    active      = {d for d in list_dates(s3, ACTIVE) if d.startswith(yr)}

    archived = _read_all(s3, ARCHIVE_COMPLETE, arch_dates)
    embedded = _read_all(s3, COMPLETE, emb_dates)

    rows = []
    sum_arch = sum_emb = 0
    n_complete = n_partial = n_missing = 0

    for d in sorted(arch_dates, reverse=True):
        a = archived[d].get("archived", 0)
        sum_arch += a
        if d in embedded:
            e = embedded[d].get("embedded", 0)
            sum_emb += e
            gap = a - e
            if gap <= 0:
                status = "complete"
                n_complete += 1
            else:
                status = f"PARTIAL (-{gap:,})"
                n_partial += 1
        else:
            e, gap = 0, a
            status = ("active" if d in active else "queued" if d in queued else "**NOT EMBEDDED**")
            n_missing += 1
        rows.append({"date": d, "archived": a, "embedded": e, "gap": gap, "status": status})

    print(f"\nDINOv2 embed audit — {yr}  (archived days: {len(arch_dates)})\n")
    print(f"{'Date':12} {'Archived':>12} {'Embedded':>12} {'Gap':>10}  Status")
    print("-" * 70)
    for r in rows:
        print(f"{r['date']:12} {r['archived']:>12,} {r['embedded']:>12,} {r['gap']:>10,}  {r['status']}")
    print("-" * 70)
    print(f"{'TOTAL':12} {sum_arch:>12,} {sum_emb:>12,} {sum_arch - sum_emb:>10,}")

    qp = qdrant_points()
    print()
    print(f"Days complete:            {n_complete}")
    print(f"Days partial:             {n_partial}")
    print(f"Days archived-not-embed:  {n_missing}  ({'in-flight' if (active or queued) else 'GAP'})")
    print(f"Total archived images:    {sum_arch:,}")
    print(f"Total embedded vectors:   {sum_emb:,}")
    if qp is not None:
        print(f"Qdrant '{COLLECTION}' points: {qp:,}  (delta vs embedded: {qp - sum_emb:+,})")

    verdict = ("READY — fully embedded" if n_partial == 0 and n_missing == 0
               else "NOT READY — gaps above")
    print(f"\nVerdict for {yr}: {verdict}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
