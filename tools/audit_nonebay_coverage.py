#!/usr/bin/env python3
"""
Non-eBay marketplace image-vector coverage audit.

Unlike eBay-dated indices (YYYY-MM-DD, one S3 partition per day), non-eBay
marketplaces are stored with index_type = the suffix (pris/pwcc/heri/ms/gold)
and ALL dates dumped under a single partition "all". So per-source-index
coverage can't be read from S3 prefixes — we read the `index_name` column
inside the parquet shards and bucket by it.

For each non-eBay OS index in the requested year range:
  - OS docs total + with galleryURL (the image-eligible denominator)
  - S3 image vectors attributed to that index_name (from parquet scan)
  - coverage % and gap

Usage:
  python tools/audit_nonebay_coverage.py                 # 2025+2026 (default)
  python tools/audit_nonebay_coverage.py --years 2025 2026
  python tools/audit_nonebay_coverage.py --csv /tmp/nonebay.csv
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pyarrow.parquet as pq
from opensearchpy.exceptions import NotFoundError

from src.embeddings.vector_store    import S3VectorStore
from src.ingestion.opensearch_reader import get_opensearch_client

VTYPE  = "image"
MODEL  = "clip-vit-l-14"
PARAMS = "v2-fp16-224px-sqpad"

# index_type path segments that are non-eBay (everything except ebay-dated +
# the "unknown" bucket, which catches misclassified names like 2026-heritage).
NONEBAY_TYPES = ["pris", "pwcc", "heri", "ms", "gold", "heritage", "unknown"]

# Recognise a non-eBay OS index name and pull its year.
#   YYYY-MM-pris / YYYY-MM-pwcc      (monthly)
#   YYYY-heri / YYYY-ms / YYYY-gold / YYYY-heritage   (annual)
_MONTHLY = re.compile(r"^(\d{4})-\d{2}-(pris|pwcc)$")
_ANNUAL  = re.compile(r"^(\d{4})-(heri|ms|gold|heritage)$")


def _os_index_year(name: str) -> str | None:
    m = _MONTHLY.match(name) or _ANNUAL.match(name)
    return m.group(1) if m else None


def scan_s3_by_index_name(store: S3VectorStore, years: set[str]) -> dict[str, int]:
    """Scan all non-eBay image shards, count vectors per OS index_name."""
    counts: dict[str, int] = defaultdict(int)
    t0 = time.time()
    total_rows = 0

    for itype in NONEBAY_TYPES:
        # list_shards with index_type but no partition → all partitions of that type
        keys = store.list_shards(VTYPE, MODEL, PARAMS, index_type=itype)
        if not keys:
            continue
        print(f"  scanning {itype}: {len(keys)} shards", flush=True)
        for i, key in enumerate(keys):
            try:
                raw = store._s3.get_object(Bucket=store.bucket, Key=key)["Body"].read()
                table = pq.read_table(io.BytesIO(raw), columns=["index_name"])
            except Exception as e:
                print(f"    WARN shard {key}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            for nm in table["index_name"].to_pylist():
                counts[nm] += 1
                total_rows += 1
            if (i + 1) % 25 == 0:
                print(f"    {itype}: {i+1}/{len(keys)} shards, "
                      f"{total_rows:,} rows ({time.time()-t0:.0f}s)", flush=True)

    print(f"  S3 scan done: {total_rows:,} non-eBay vectors across "
          f"{len(counts)} distinct index_names ({time.time()-t0:.1f}s)", flush=True)
    return dict(counts)


def os_counts_for(osclient, index_name: str) -> tuple[int, int, bool]:
    """(total_docs, docs_with_REAL_galleryURL, exists).

    "Real" = galleryURL exists AND is not the literal "N/A" placeholder.
    Non-eBay marketplaces (notably pristine) store galleryURL="N/A" when
    there is no image — those docs are unembeddable, so counting them in
    the denominator wildly overstates the achievable gap. The embeddable
    denominator excludes them.
    """
    try:
        total = osclient.count(index=index_name, body={"query": {"match_all": {}}})["count"]
    except NotFoundError:
        return (0, 0, False)
    except Exception:
        return (0, 0, False)
    if total == 0:
        return (0, 0, True)
    with_img = osclient.count(index=index_name, body={
        "query": {"bool": {
            "must":     [{"exists": {"field": "galleryURL"}}],
            "must_not": [{"term": {"galleryURL": "N/A"}}],
        }}
    })["count"]
    return (total, with_img, True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs="+", default=["2025", "2026"])
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    years = set(args.years)

    osclient = get_opensearch_client()
    store = S3VectorStore(
        bucket=os.environ["S3_VECTOR_BUCKET"],
        prefix=os.environ.get("S3_VECTOR_PREFIX", "vectors"),
    )

    # ── Discover non-eBay OS indices in the requested years ───────────────
    print("Listing non-eBay OS indices...", flush=True)
    cat = osclient.cat.indices(format="json", h="index,docs.count",
                               params={"expand_wildcards": "open"})
    target_indices = []
    for row in cat:
        nm = row.get("index", "")
        yr = _os_index_year(nm)
        if yr and yr in years:
            target_indices.append(nm)
    target_indices.sort()
    print(f"  {len(target_indices)} non-eBay indices in years {sorted(years)}",
          flush=True)

    # ── Scan S3 once, bucket by index_name ────────────────────────────────
    print("\nScanning S3 non-eBay image shards (reads index_name column)...",
          flush=True)
    s3_by_index = scan_s3_by_index_name(store, years)

    # ── Compare ───────────────────────────────────────────────────────────
    print()
    print(f"{'OS Index':18} {'OS docs':>10} {'embeddable':>10} {'S3 vecs':>10} "
          f"{'Gap':>10} {'Gap %':>7}  Note")
    print("  (embeddable = docs with a real galleryURL; excludes 'N/A' placeholders)")
    print("-" * 88)

    rows = []
    sum_os = sum_img = sum_s3 = 0
    zero_cov = []

    for idx in target_indices:
        os_total, os_img, exists = os_counts_for(osclient, idx)
        s3_n = s3_by_index.get(idx, 0)
        denom = os_img if os_img > 0 else os_total
        gap = denom - s3_n
        gap_pct = (100.0 * gap / denom) if denom > 0 else 0.0

        if s3_n == 0 and denom > 0:
            note = "**ZERO S3 — not backfilled**"
            zero_cov.append(idx)
        elif gap <= 0:
            note = "complete"
        elif gap_pct >= 10:
            note = f"**LOW ({gap_pct:.0f}% missing)**"
        else:
            note = "partial"

        sum_os  += os_total
        sum_img += os_img
        sum_s3  += s3_n
        rows.append({"index": idx, "os_total": os_total, "os_with_img": os_img,
                     "s3_vectors": s3_n, "gap": gap, "gap_pct": round(gap_pct, 2),
                     "note": note})
        print(f"{idx:18} {os_total:>10,} {os_img:>10,} {s3_n:>10,} "
              f"{gap:>10,} {gap_pct:>6.1f}%  {note}", flush=True)

    overall_denom = sum_img if sum_img > 0 else sum_os
    overall_gap = overall_denom - sum_s3
    overall_pct = (100.0 * overall_gap / overall_denom) if overall_denom > 0 else 0.0
    print("-" * 88)
    print(f"{'TOTAL':18} {sum_os:>10,} {sum_img:>10,} {sum_s3:>10,} "
          f"{overall_gap:>10,} {overall_pct:>6.1f}%")
    print()
    if zero_cov:
        print(f"Indices with ZERO S3 coverage ({len(zero_cov)}):")
        for idx in zero_cov:
            print(f"  {idx}")
    else:
        print("No indices with zero coverage.")

    # Flag any S3 index_names that look like target years but aren't in OS
    # (orphan vectors — index deleted or renamed since backfill).
    orphans = [nm for nm in s3_by_index
               if (_os_index_year(nm) in years) and nm not in set(target_indices)]
    if orphans:
        print(f"\nOrphan S3 index_names (vectors exist, OS index gone): {len(orphans)}")
        for nm in sorted(orphans):
            print(f"  {nm}: {s3_by_index[nm]:,} vectors")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
