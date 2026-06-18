#!/usr/bin/env python3
"""
POC component 1 — archive card images to S3 in three sizes.

Reads the eBay-dated OpenSearch window, fetches each card's gallery image at
full size (s-l1600), produces 512px and 256px JPEG variants, and uploads all
three to S3. Writes a manifest (one row per card) carrying the S3 keys, the
recorded byte sizes (for the cost report), and the source doc fields needed to
build the Qdrant payload later.

This is the foundation for serving images from S3/CDN instead of hot-fetching
source URLs, AND it lets every later re-embed read images from S3 instead of
re-downloading from (increasingly dead) source URLs.

Pilot:
    python tools/poc_image_archive.py --date 2026-06-01 --limit 50000 \
        --out data/poc/manifest_2026-06-01.jsonl
Full window (per day):
    python tools/poc_image_archive.py --date 2026-06-03 \
        --out data/poc/manifest_2026-06-03.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from PIL import Image

from src.ingestion.opensearch_reader import get_opensearch_client
from src.ingestion.qdrant_writer import os_id_to_qdrant_id
from tools.poc_common import (
    image_key, upsize_ebay_url, make_s3, put_image, encode_jpeg,
    RESIZE_DIMS,
)

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS   = 50_000_000   # reject decompression bombs before decoding
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
_local = threading.local()


def _s3():
    if not hasattr(_local, "s3"):
        _local.s3 = make_s3()
    return _local.s3


def fetch_window(os_client, date_str: str, limit: int | None):
    """Scroll one eBay-dated index, yielding source docs with a gallery image."""
    body = {
        "size": 1000,
        "query": {"bool": {
            "must": [{"exists": {"field": "galleryURL"}}],
            "must_not": [{"term": {"galleryURL": "N/A"}}],
        }},
        "_source": ["id", "itemId", "title", "galleryURL", "source", "globalId",
                    "saleType", "currentPrice", "currentPriceCurrency",
                    "itemSpecifics"],
    }
    resp = os_client.search(index=date_str, body=body, scroll="5m", request_timeout=60)
    sid = resp.get("_scroll_id")
    n = 0
    while True:
        hits = resp["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            yield h["_id"], h["_source"]
            n += 1
            if limit and n >= limit:
                return
        resp = os_client.scroll(scroll_id=sid, scroll="5m", request_timeout=60)
        sid = resp.get("_scroll_id")
        if sid is None:
            break


def download(url: str) -> bytes | None:
    try:
        with requests.get(url, headers=DOWNLOAD_HEADERS, timeout=20,
                          stream=True, verify=True) as r:
            if r.status_code != 200:
                return None
            buf = io.BytesIO()
            for chunk in r.iter_content(65536):
                buf.write(chunk)
                if buf.tell() > MAX_DOWNLOAD_BYTES:
                    return None
            data = buf.getvalue()
        return data if len(data) >= 500 else None
    except Exception:
        return None


def process_one(os_id: str, src: dict, bucket: str) -> dict | None:
    """Download s-l1600, build 3 variants, upload to S3, return a manifest row.

    Never raises: download/decode/resize/upload are all guarded so one bad image
    or a transient S3 error can't crash a pool thread (and thence the run). A
    pixel cap rejects decompression bombs BEFORE decoding so a single giant
    image can't exhaust RAM across 12+ concurrent threads. Returns None for a
    skip, {"_error": ...} for a failure worth surfacing, else the manifest row.
    """
    url = upsize_ebay_url(str(src.get("galleryURL") or "").strip())
    if not url:
        return None
    raw = download(url)
    if raw is None:
        return None
    try:
        img = Image.open(io.BytesIO(raw))           # lazy — header only
        if (img.width or 0) * (img.height or 0) > MAX_IMAGE_PIXELS:
            return None
        img = img.convert("RGB")                     # forces decode

        s3 = _s3()
        keys: dict[str, str] = {}
        sizes: dict[str, int] = {}

        orig_bytes = encode_jpeg(img, quality=92)
        keys["original"] = image_key(os_id, "original")
        put_image(s3, bucket, keys["original"], orig_bytes)
        sizes["original"] = len(orig_bytes)

        for name, dim in RESIZE_DIMS.items():
            v = img.copy()
            v.thumbnail((dim, dim), Image.LANCZOS)
            b = encode_jpeg(v, quality=88)
            keys[name] = image_key(os_id, name)
            put_image(s3, bucket, keys[name], b)
            sizes[name] = len(b)

        return {
            "os_id":       os_id,
            "qdrant_id":   str(os_id_to_qdrant_id(os_id)),
            "gallery_url": url,
            "s3_keys":     keys,
            "sizes":       sizes,
            "source_doc":  src,
        }
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="eBay-dated OS index, YYYY-MM-DD.")
    ap.add_argument("--limit", type=int, default=None, help="Cap N cards (pilot).")
    ap.add_argument("--out", required=True, help="Manifest JSONL output path.")
    ap.add_argument("--bucket", default=os.environ.get("S3_IMAGE_BUCKET")
                    or os.environ.get("S3_VECTOR_BUCKET"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if not args.bucket:
        logger.error("No S3 bucket — set S3_IMAGE_BUCKET (or S3_VECTOR_BUCKET).")
        sys.exit(1)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os_client = get_opensearch_client()
    docs = list(fetch_window(os_client, args.date, args.limit))
    logger.info("{}: {} cards with gallery image → archiving to s3://{}",
                args.date, len(docs), args.bucket)

    ok = fail = 0
    err_samples: list[str] = []
    with open(out_path, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, oid, src, args.bucket): oid for oid, src in docs}
        for i, f in enumerate(as_completed(futs)):
            row = f.result()
            if row is None:
                fail += 1
            elif "_error" in row:
                fail += 1
                if len(err_samples) < 5:
                    err_samples.append(row["_error"])
            else:
                fout.write(json.dumps(row) + "\n")
                ok += 1
            if (i + 1) % 1000 == 0:
                logger.info("  {}/{}  ok={} fail={}", i + 1, len(docs), ok, fail)
                if ok == 0 and err_samples:
                    logger.error("All uploads failing — sample errors: {}", err_samples)

    logger.info("─" * 60)
    if err_samples:
        logger.warning("Sample failures: {}", err_samples)
    logger.info("Archived {} cards ({} failed) → manifest {}", ok, fail, out_path)
    logger.info("Each card stored as original/512/256 under s3://{}/images/ebay/",
                args.bucket)


if __name__ == "__main__":
    main()
