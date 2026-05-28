#!/usr/bin/env python3
"""
Per-day backfill verifier.

For a given YYYY-MM-DD eBay-dated index:

  1. count OS docs with galleryURL  (the should-have universe)
  2. count S3 image vectors stored under that partition  (the do-have)
  3. compute coverage %
  4. if coverage < target_pct, scroll OS to enumerate every os_id with
     galleryURL, scan S3 parquet shards for os_ids already covered, compute
     the missing set, and write it to:
         s3://$S3_VECTOR_BUCKET/backfill-v2/remediation/{date}_missing.json
  5. write a per-day summary to:
         s3://$S3_VECTOR_BUCKET/backfill-v2/verified/{date}.json

Status returned:

  complete             — coverage ≥ target_pct (98 by default)
  remediation_attempted— coverage in [soft_floor, target_pct), missing-id
                         list written
  manual_review        — coverage < soft_floor (90 by default), missing-id
                         list written and operator should investigate
  no_os_index          — OS index doesn't exist (skip silently)

Usable both as a CLI:
    python tools/verify_day_backfill.py 2026-05-28

And as a module the orchestrator imports:
    from tools.verify_day_backfill import verify_day, VerifyResult
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import boto3
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from loguru import logger
from opensearchpy.exceptions import NotFoundError

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.embeddings.vector_store    import S3VectorStore
from src.ingestion.opensearch_reader import get_opensearch_client


S3_BUCKET            = os.environ["S3_VECTOR_BUCKET"]
S3_REMEDIATION_PFX   = "backfill-v2/remediation"
S3_VERIFIED_PFX      = "backfill-v2/verified"

VECTOR_TYPE  = "image"
MODEL        = "clip-vit-l-14"
PARAMS       = "v2-fp16-224px-sqpad"

# Default thresholds. Configurable per call.
TARGET_PCT     = 98.0
SOFT_FLOOR_PCT = 90.0

# Scroll-time settings
OS_SCROLL_PAGE = 10_000
OS_SCROLL_TTL  = "5m"


@dataclass
class VerifyResult:
    date:              str
    status:            str           # "complete" | "remediation_attempted"
                                     # | "manual_review" | "no_os_index"
    os_docs_with_img:  int
    s3_vectors:        int
    coverage_pct:      float
    missing_count:     int
    target_pct:        float
    soft_floor_pct:    float
    remediation_key:   str | None    # S3 key if missing-ids written
    elapsed_s:         float


def _s3_count_vectors(s3_fs: pafs.S3FileSystem, bucket: str,
                      keys: list[str]) -> int:
    total = 0
    for key in keys:
        try:
            md = pq.read_metadata(f"{bucket}/{key}", filesystem=s3_fs)
            total += md.num_rows
        except Exception as e:
            logger.warning("metadata read failed for {}: {}: {}",
                           key, type(e).__name__, e)
    return total


def _os_ids_with_galleryurl(os_client, index_name: str) -> set[str]:
    """Scroll the OS index for every doc with galleryURL — return _id set."""
    body = {
        "query":   {"exists": {"field": "galleryURL"}},
        "_source": False,
        "size":    OS_SCROLL_PAGE,
        "sort":    ["_doc"],
    }
    ids: set[str] = set()
    resp = os_client.search(index=index_name, body=body, scroll=OS_SCROLL_TTL)
    scroll_id = resp.get("_scroll_id")
    try:
        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                ids.add(h["_id"])
            resp = os_client.scroll(scroll_id=scroll_id, scroll=OS_SCROLL_TTL)
            scroll_id = resp.get("_scroll_id")
    finally:
        if scroll_id:
            try:
                os_client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass
    return ids


def _s3_os_ids(store: S3VectorStore, keys: list[str]) -> set[str]:
    """Scan parquet shards (os_id column only) and union all os_ids."""
    ids: set[str] = set()
    for key in keys:
        try:
            obj = store._s3.get_object(Bucket=store.bucket, Key=key)
            import io
            table = pq.read_table(io.BytesIO(obj["Body"].read()),
                                  columns=["os_id"])
            ids.update(table["os_id"].to_pylist())
        except Exception as e:
            logger.warning("shard read failed for {}: {}: {}",
                           key, type(e).__name__, e)
    return ids


def _write_remediation(s3, date_str: str, missing: Iterable[str]) -> str:
    key = f"{S3_REMEDIATION_PFX}/{date_str}_missing.json"
    body = {
        "date":         date_str,
        "missing_ids":  sorted(missing),
        "count":        len(list(missing)) if not isinstance(missing, list) else len(missing),
        "written_at":   datetime.now(timezone.utc).isoformat(),
    }
    # Re-sort to a list so the count is consistent
    sorted_missing       = sorted(missing)
    body["missing_ids"]  = sorted_missing
    body["count"]        = len(sorted_missing)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(body).encode())
    return key


def _write_verified_summary(s3, result: VerifyResult) -> str:
    key = f"{S3_VERIFIED_PFX}/{result.date}.json"
    body = {
        **asdict(result),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(body).encode())
    return key


def verify_day(date_str:        str,
               s3              = None,
               os_client       = None,
               store: S3VectorStore = None,
               target_pct:     float = TARGET_PCT,
               soft_floor_pct: float = SOFT_FLOOR_PCT,
               aws_region:     str = None) -> VerifyResult:
    """
    Run the verification flow for one YYYY-MM-DD date. Returns VerifyResult.

    Clients (s3, os_client, store) may be passed in for re-use; built from
    env vars if omitted.
    """
    t0 = datetime.now(timezone.utc)
    aws_region = aws_region or os.environ.get("AWS_REGION", "us-west-1")
    s3        = s3        or boto3.client("s3", region_name=aws_region)
    os_client = os_client or get_opensearch_client()
    store     = store     or S3VectorStore(
        bucket = S3_BUCKET,
        prefix = os.environ.get("S3_VECTOR_PREFIX", "vectors"),
    )
    s3_fs = pafs.S3FileSystem(region=aws_region)

    # ── 1. OS count ───────────────────────────────────────────────────
    try:
        os_count = os_client.count(
            index = date_str,
            body  = {"query": {"exists": {"field": "galleryURL"}}},
        )["count"]
    except NotFoundError:
        return VerifyResult(
            date=date_str, status="no_os_index",
            os_docs_with_img=0, s3_vectors=0, coverage_pct=0.0,
            missing_count=0, target_pct=target_pct,
            soft_floor_pct=soft_floor_pct, remediation_key=None,
            elapsed_s=(datetime.now(timezone.utc) - t0).total_seconds(),
        )

    # ── 2. S3 count ───────────────────────────────────────────────────
    keys = store.list_shards(VECTOR_TYPE, MODEL, PARAMS,
                             index_type="ebay-dated", partition=date_str)
    s3_count = _s3_count_vectors(s3_fs, store.bucket, keys) if keys else 0

    pct = (100.0 * s3_count / os_count) if os_count > 0 else 100.0

    if os_count == 0:
        return VerifyResult(
            date=date_str, status="complete",
            os_docs_with_img=0, s3_vectors=s3_count, coverage_pct=100.0,
            missing_count=0, target_pct=target_pct,
            soft_floor_pct=soft_floor_pct, remediation_key=None,
            elapsed_s=(datetime.now(timezone.utc) - t0).total_seconds(),
        )

    # ── Decide branch ─────────────────────────────────────────────────
    if pct >= target_pct:
        result = VerifyResult(
            date=date_str, status="complete",
            os_docs_with_img=os_count, s3_vectors=s3_count, coverage_pct=pct,
            missing_count=0, target_pct=target_pct,
            soft_floor_pct=soft_floor_pct, remediation_key=None,
            elapsed_s=(datetime.now(timezone.utc) - t0).total_seconds(),
        )
        _write_verified_summary(s3, result)
        return result

    # Below target — enumerate missing
    logger.info("{}  pct={:.2f}% < target {:.0f}%, finding missing ids...",
                date_str, pct, target_pct)
    os_ids = _os_ids_with_galleryurl(os_client, date_str)
    s3_ids = _s3_os_ids(store, keys)
    missing = os_ids - s3_ids
    rem_key = _write_remediation(s3, date_str, missing) if missing else None

    status = ("remediation_attempted"
              if pct >= soft_floor_pct
              else "manual_review")
    result = VerifyResult(
        date=date_str, status=status,
        os_docs_with_img=os_count, s3_vectors=s3_count, coverage_pct=pct,
        missing_count=len(missing), target_pct=target_pct,
        soft_floor_pct=soft_floor_pct, remediation_key=rem_key,
        elapsed_s=(datetime.now(timezone.utc) - t0).total_seconds(),
    )
    _write_verified_summary(s3, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("--target-pct", type=float, default=TARGET_PCT)
    ap.add_argument("--soft-floor-pct", type=float, default=SOFT_FLOOR_PCT)
    args = ap.parse_args()

    result = verify_day(args.date,
                        target_pct=args.target_pct,
                        soft_floor_pct=args.soft_floor_pct)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
