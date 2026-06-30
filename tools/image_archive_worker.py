#!/usr/bin/env python3
"""
Image-archive fleet worker.

Loops: atomically claim a 2026 day from the image-archive queue, download every
card's s-l1600 image, build 512/256 variants, upload all three to the images
bucket, upload a gzipped per-day manifest (for the later DINOv2 embed), and
mark the day complete with counts. Runs continuously as a systemd service on
each worker; exits when the queue is empty (unless --loop).

Reuses poc_image_archive.process_one (pixel-capped, guarded, never raises) so
one bad image can't take down a worker.

    python tools/image_archive_worker.py --workers 12
    python tools/image_archive_worker.py --workers 12 --loop   # poll for new days
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from opensearchpy.exceptions import NotFoundError

from src.ingestion.opensearch_reader import get_opensearch_client
from tools.poc_image_archive import fetch_window, process_one
from tools.poc_common import source_for_index
from tools.image_archive_common import (
    s3_client, claim_next, mark_complete, release, update_active,
    QUEUE_BUCKET, IMAGE_BUCKET, MANIFESTS,
)


def archive_date(os_client, s3, date_str: str, workers: int) -> tuple[list[dict], dict]:
    """Archive one day. Returns (manifest_rows, stats). Writes intra-day progress
    to the active marker every 5k images so the dashboard can show a live bar."""
    try:
        docs = list(fetch_window(os_client, date_str, None))
    except NotFoundError:
        return [], {"archived": 0, "failed": 0, "os_total": 0, "note": "no OS index"}

    os_total = len(docs)
    update_active(s3, date_str, {"os_total": os_total, "archived": 0, "failed": 0})

    # Namespace S3 keys by marketplace (ebay for YYYY-MM-DD, else the suffix) so
    # an OS _id reused across indexes/marketplaces can't clash.
    source = source_for_index(date_str)

    rows: list[dict] = []
    ok = fail = 0
    bytes_total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, oid, src, IMAGE_BUCKET, source): oid for oid, src in docs}
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r is None or "_error" in r:
                fail += 1
            else:
                rows.append(r)
                ok += 1
                bytes_total += sum(r.get("sizes", {}).values())
            if (i + 1) % 5000 == 0:
                logger.info("  [{}] {}/{}  ok={} fail={}",
                            date_str, i + 1, os_total, ok, fail)
                update_active(s3, date_str,
                              {"os_total": os_total, "archived": ok, "failed": fail})

    return rows, {"archived": ok, "failed": fail, "os_total": os_total,
                  "bytes": bytes_total}


def upload_manifest(s3, date_str: str, rows: list[dict]) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in rows:
            gz.write((json.dumps(r) + "\n").encode())
    s3.put_object(Bucket=QUEUE_BUCKET, Key=f"{MANIFESTS}/{date_str}.jsonl.gz",
                  Body=buf.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12,
                    help="Concurrent image downloads per day.")
    ap.add_argument("--loop", action="store_true",
                    help="Keep polling for new days instead of exiting on empty queue.")
    ap.add_argument("--idle-sleep", type=int, default=120)
    args = ap.parse_args()

    s3 = s3_client()
    os_client = get_opensearch_client()
    logger.info("Image-archive worker started — images → s3://{}", IMAGE_BUCKET)

    while True:
        date_str = claim_next(s3)
        if date_str is None:
            if args.loop:
                logger.info("Queue empty — sleeping {}s", args.idle_sleep)
                time.sleep(args.idle_sleep)
                continue
            logger.info("Queue empty — exiting")
            return

        logger.info("Claimed {}", date_str)
        t0 = time.time()
        try:
            rows, stats = archive_date(os_client, s3, date_str, args.workers)
            if rows:
                upload_manifest(s3, date_str, rows)
            stats["seconds"] = round(time.time() - t0, 1)
            mark_complete(s3, date_str, stats)
            logger.info("Done {} — archived {}/{} ({} failed) in {:.0f}s",
                        date_str, stats["archived"], stats["os_total"],
                        stats["failed"], stats["seconds"])
        except Exception as e:
            logger.error("{} failed ({}: {}) — releasing back to queue",
                         date_str, type(e).__name__, e)
            release(s3, date_str)
            time.sleep(5)


if __name__ == "__main__":
    main()
