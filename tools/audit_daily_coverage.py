#!/usr/bin/env python3
"""
Per-day image-vector coverage audit: OS vs S3.

For each YYYY-MM-DD eBay-dated index from --start to --end:
  - count OS docs (total and with galleryURL)
  - count S3 image-vector rows under that partition
  - report gap (OS-with-image − S3) and gap %

S3 row counts come from parquet footer metadata only (range-request to S3,
not full file download) so this is fast — ~100ms per shard, ~1.5 min total
for a 5-month range with typical shard counts.

Use cases:
  - Detect daily worker gaps since 2026 cutover.
  - Spot dates where backfill failed and never got retried.
  - Quantify "we're behind by X vectors as of <date>".

Usage:
  python tools/audit_daily_coverage.py
  python tools/audit_daily_coverage.py --start 2026-01-01 --end 2026-05-28
  python tools/audit_daily_coverage.py --csv /tmp/coverage.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pyarrow.fs as pafs
import pyarrow.parquet as pq
from opensearchpy.exceptions import NotFoundError

from src.embeddings.vector_store    import S3VectorStore
from src.ingestion.opensearch_reader import get_opensearch_client

VTYPE  = "image"
MODEL  = "clip-vit-l-14"
PARAMS = "v2-fp16-224px-sqpad"


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def os_counts(osclient, index_name: str) -> tuple[int, int, bool]:
    """Return (total_docs, docs_with_galleryURL, exists)."""
    try:
        total = osclient.count(index=index_name, body={"query": {"match_all": {}}})["count"]
    except NotFoundError:
        return (0, 0, False)
    except Exception:
        return (0, 0, False)
    if total == 0:
        return (0, 0, True)
    with_img = osclient.count(index=index_name, body={
        "query": {"exists": {"field": "galleryURL"}}
    })["count"]
    return (total, with_img, True)


def s3_row_count(s3_fs: pafs.S3FileSystem, bucket: str, keys: list[str]) -> int:
    """Sum row counts across shards using parquet footer metadata only."""
    total = 0
    for key in keys:
        try:
            md = pq.read_metadata(f"{bucket}/{key}", filesystem=s3_fs)
            total += md.num_rows
        except Exception as e:
            print(f"  warning: failed reading metadata for {key}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2026-01-01",
                   help="Start date YYYY-MM-DD (default 2026-01-01).")
    p.add_argument("--end",   default=date.today().isoformat(),
                   help="End date YYYY-MM-DD (default today).")
    p.add_argument("--csv",   default=None,
                   help="Also write per-day rows to this CSV path.")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-1"))
    args = p.parse_args()

    start = datetime.fromisoformat(args.start).date()
    end   = datetime.fromisoformat(args.end).date()
    if end < start:
        print(f"end ({end}) before start ({start})", file=sys.stderr)
        sys.exit(1)

    osclient = get_opensearch_client()
    store = S3VectorStore(
        bucket=os.environ["S3_VECTOR_BUCKET"],
        prefix=os.environ.get("S3_VECTOR_PREFIX", "vectors"),
    )
    s3_fs = pafs.S3FileSystem(region=args.region)

    rows = []
    print(f"Auditing {start} → {end} ({(end - start).days + 1} days)")
    print()
    print(f"{'Date':12} {'OS docs':>12} {'OS w/img':>12} {'S3 vecs':>12} "
          f"{'Gap':>12}  {'Gap %':>7}  Note")
    print("-" * 96)

    sum_os_total = 0
    sum_os_img   = 0
    sum_s3       = 0
    n_missing_index = 0
    n_full_coverage = 0
    n_partial       = 0
    n_zero_s3_nonzero_os = 0

    for d in daterange(start, end):
        idx = d.isoformat()
        os_total, os_img, exists = os_counts(osclient, idx)

        keys = store.list_shards(VTYPE, MODEL, PARAMS,
                                 index_type="ebay-dated", partition=idx)
        s3_n = s3_row_count(s3_fs, store.bucket, keys) if keys else 0

        if not exists:
            note = "no OS index"
            n_missing_index += 1
            gap = -s3_n
            gap_pct = 0.0
        else:
            gap = os_img - s3_n
            gap_pct = (100.0 * gap / os_img) if os_img > 0 else 0.0
            if os_img == 0:
                note = "OS has no img docs"
            elif gap <= 0:
                note = "complete"
                n_full_coverage += 1
            elif s3_n == 0:
                note = "**NO S3 SHARDS**"
                n_zero_s3_nonzero_os += 1
            else:
                note = f"partial ({len(keys)} shards)"
                n_partial += 1

        sum_os_total += os_total
        sum_os_img   += os_img
        sum_s3       += s3_n

        rows.append({
            "date": idx, "os_total": os_total, "os_with_img": os_img,
            "s3_vectors": s3_n, "gap": gap, "gap_pct": round(gap_pct, 2),
            "shards": len(keys), "note": note,
        })

        print(f"{idx:12} {os_total:>12,} {os_img:>12,} {s3_n:>12,} "
              f"{gap:>12,}  {gap_pct:>6.2f}%  {note}", flush=True)

    overall_gap = sum_os_img - sum_s3
    overall_pct = (100.0 * overall_gap / sum_os_img) if sum_os_img > 0 else 0.0
    print("-" * 96)
    print(f"{'TOTAL':12} {sum_os_total:>12,} {sum_os_img:>12,} {sum_s3:>12,} "
          f"{overall_gap:>12,}  {overall_pct:>6.2f}%")
    print()
    print(f"Days with complete S3 coverage:       {n_full_coverage}")
    print(f"Days with partial S3 coverage:        {n_partial}")
    print(f"Days with OS docs but ZERO S3 shards: {n_zero_s3_nonzero_os}  (critical)")
    print(f"Days where OS index does not exist:   {n_missing_index}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
