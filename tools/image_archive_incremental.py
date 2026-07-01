#!/usr/bin/env python3
"""
Hourly incremental image archiver — the steady-state successor to the backfill fleet.

Re-scans the most recent OpenSearch days (default: today + yesterday) and archives
only images NOT already in S3, by diffing each day's OS docs against the day's
existing manifest (image-archive/manifests/{date}.jsonl.gz). New images are
downloaded in 1600/512/256, appended to the manifest, and the day's complete
marker is refreshed. Cheap to run every hour: only genuinely-new listings are
fetched — nothing already archived is re-downloaded.

Runs as a single scheduled instance (hourly systemd timer), replacing the
continuous backfill fleet once the 2025/2026 backlog is done.

Usage:
  python tools/image_archive_incremental.py                 # today + yesterday
  python tools/image_archive_incremental.py --days 3 --workers 12
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from opensearchpy.exceptions import NotFoundError

from tools.poc_image_archive import fetch_window, process_one
from tools.poc_common import source_for_index
from src.ingestion.opensearch_reader import get_opensearch_client
from tools.image_archive_common import (
    s3_client, mark_complete, QUEUE_BUCKET, IMAGE_BUCKET, MANIFESTS,
)


def read_manifest(s3, date_str: str) -> list[dict]:
    """Load the day's existing manifest rows (empty list if none yet)."""
    try:
        body = s3.get_object(Bucket=QUEUE_BUCKET,
                             Key=f"{MANIFESTS}/{date_str}.jsonl.gz")["Body"].read()
    except s3.exceptions.ClientError:
        return []
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
        for line in gz:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_manifest(s3, date_str: str, rows: list[dict]) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in rows:
            gz.write((json.dumps(r) + "\n").encode())
    s3.put_object(Bucket=QUEUE_BUCKET,
                  Key=f"{MANIFESTS}/{date_str}.jsonl.gz", Body=buf.getvalue())


def incremental_day(os_client, s3, date_str: str, workers: int) -> tuple[int, int]:
    """Archive only images missing from the manifest for one day.
    Returns (new_archived, os_total)."""
    existing = read_manifest(s3, date_str)
    done = {r["os_id"] for r in existing}

    try:
        docs = list(fetch_window(os_client, date_str, None))
    except NotFoundError:
        return 0, 0

    todo = [(oid, src) for oid, src in docs if oid not in done]
    if not todo:
        return 0, len(docs)

    source = source_for_index(date_str)
    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, oid, src, IMAGE_BUCKET, source): oid
                for oid, src in todo}
        for f in as_completed(futs):
            r = f.result()
            if r and "_error" not in r:
                new_rows.append(r)

    if new_rows:
        write_manifest(s3, date_str, existing + new_rows)
        total = len(existing) + len(new_rows)
        mark_complete(s3, date_str, {
            "archived": total, "os_total": len(docs),
            "failed": len(docs) - total, "incremental": True,
        })
    return len(new_rows), len(docs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=2,
                    help="How many recent days to re-scan (today going back). Default 2.")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    s3 = s3_client()
    os_client = get_opensearch_client()
    today = date.today()

    total_new = 0
    for i in range(args.days):
        d = (today - timedelta(days=i)).isoformat()
        t0 = time.time()
        new, os_total = incremental_day(os_client, s3, d, args.workers)
        total_new += new
        logger.info("{}: +{} new images ({} docs in OS) in {:.0f}s",
                    d, new, os_total, time.time() - t0)

    logger.info("Incremental run complete — {} new images across last {} day(s)",
                total_new, args.days)


if __name__ == "__main__":
    main()
