#!/usr/bin/env python3
"""
Daily dual-embed worker — CLIP + DINOv2 from the S3 image archive, single pass.

Steady-state successor to the RDS-based daily CLIP job. For each recent day
(default: last 2) plus the current non-eBay indices, reads the image-archive
manifest, loads each archived image from S3 ONCE, and encodes it with both
backbones in the same pass:

  - CLIP ViT-L/14 (768-d, "image") + MiniLM specifics (384-d)  -> `cards`
  - DINOv2-large @512 (1024-d, "image_dinov2")                 -> `cards_dinov2`

Vectors are written to the S3 vector store (durable) before the Qdrant upsert,
per the S3-before-Qdrant invariant. Per-day completion markers under
daily-dual/complete make it idempotent and safe to re-run.

Usage:
  python tools/daily_dual_embed.py                    # last 2 days + current non-eBay
  python tools/daily_dual_embed.py --days 3
  python tools/daily_dual_embed.py --date 2026-07-01  # one specific index/day
  python tools/daily_dual_embed.py --no-nonebay --force
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from qdrant_client.models import PointStruct

from src.embeddings.image_encoder import ImageEncoder
from src.embeddings.text_encoder import TextEncoder, format_specifics
from src.embeddings.vector_store import S3VectorStore, VectorRecord, SHARD_SIZE
from src.embeddings.rds_batch_job import (
    encode_pil_batch, IMAGE_MODEL_ID, IMAGE_PARAMS, TEXT_MODEL_ID, TEXT_PARAMS,
)
from src.ingestion.qdrant_writer import get_qdrant_client, extract_payload, build_point
from tools.poc_common import get_image_pil, source_for_index
from tools.dino_embed_worker import (
    _point_id, ensure_collection,
    COLLECTION as DINO_COLLECTION, VEC_NAME as DINO_VEC,
    MODEL_ID as DINO_MODEL, PARAMS as DINO_PARAMS, DINO_SIZE,
)
from tools.eval_retrieval_at_scale import build_encoder
from tools.eval_parallel_discrimination import _pick_device
from tools.dino_embed_common import s3_client, QUEUE_BUCKET, IMAGE_BUCKET, ARCHIVE_MANIFESTS
from tools.image_archive_incremental import recent_nonebay_indices

CARDS_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cards")
MARKER_PREFIX    = "daily-dual/complete"


# ── Per-day markers (idempotency) ───────────────────────────────────────────────

def _marker_key(date_str: str) -> str:
    return f"{MARKER_PREFIX}/{date_str}.json"


def is_complete(s3, date_str: str) -> bool:
    try:
        s3.head_object(Bucket=QUEUE_BUCKET, Key=_marker_key(date_str))
        return True
    except Exception:
        return False


def mark_complete(s3, date_str: str, stats: dict) -> None:
    s3.put_object(Bucket=QUEUE_BUCKET, Key=_marker_key(date_str),
                  Body=json.dumps(stats).encode())


# ── Manifest + image loading ────────────────────────────────────────────────────

def read_manifest(s3, date_str: str) -> list[dict] | None:
    try:
        raw = s3.get_object(Bucket=QUEUE_BUCKET,
                            Key=f"{ARCHIVE_MANIFESTS}/{date_str}.jsonl.gz")["Body"].read()
    except Exception:
        return None
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        for line in gz:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_one(s3, r):
    key = r["s3_keys"].get("512")
    if not key:
        return None
    try:
        return get_image_pil(s3, IMAGE_BUCKET, key)
    except Exception:
        return None


# ── Dual embed one day/index ────────────────────────────────────────────────────

def embed_day(date_str, clip_enc, text_enc, dino_encode, qdrant, store,
              s3, pool, batch) -> dict:
    rows = read_manifest(s3, date_str)
    if not rows:
        return {"embedded": 0, "total": 0, "note": "no manifest"}

    source = source_for_index(date_str)
    # eBay keeps the historical "ebay-dated" S3 index_type; non-eBay uses source.
    itype  = "ebay-dated" if source == "ebay" else source
    job_id = f"daily-{date_str}"
    total  = len(rows)

    img_buf: list[VectorRecord] = []
    spec_buf: list[VectorRecord] = []
    dino_buf: list[VectorRecord] = []
    img_n = spec_n = dino_n = 0
    embedded = 0

    def _drain(buf, shard_n, final=False):
        while len(buf) >= SHARD_SIZE:
            store.write_shard(buf[:SHARD_SIZE], shard_n)
            shard_n += 1
            del buf[:SHARD_SIZE]
        if final and buf:
            store.write_shard(buf, shard_n)
            shard_n += 1
        return shard_n

    for i in range(0, total, batch):
        chunk = rows[i:i + batch]
        raw = list(pool.map(lambda r: load_one(s3, r), chunk))
        keep = [(r, im) for r, im in zip(chunk, raw) if im is not None]
        if not keep:
            continue
        sub  = [r for r, _ in keep]
        imgs = [im for _, im in keep]

        # CLIP: pad-to-square then batched GPU encode (matches the cards backfill).
        clip_vecs = encode_pil_batch([clip_enc._pad_to_square(im) for im in imgs], clip_enc)
        # DINOv2: build_encoder pads/resizes to 512 internally (sqpad params).
        dino_vecs = dino_encode(imgs)
        # Specifics text -> MiniLM (fall back to title so every row gets a vector).
        specs = []
        for r in sub:
            doc = r["source_doc"]
            t = format_specifics(doc.get("itemSpecifics") or {}) \
                or (doc.get("title") or "").lower().strip()
            specs.append(t)
        spec_vecs = text_enc.encode_batch(specs)

        cards_pts, dino_pts = [], []
        for r, iv, sv, dv in zip(sub, clip_vecs, spec_vecs, dino_vecs):
            qid = r["qdrant_id"]
            pid = _point_id(qid)
            payload = extract_payload(r["source_doc"], doc_id=r["os_id"])
            payload["img_512"] = r["s3_keys"].get("512", "")
            payload["img_256"] = r["s3_keys"].get("256", "")
            common = dict(os_id=r["os_id"], qdrant_id=qid, index_name=date_str,
                          index_type=itype, job_id=job_id)

            pt = build_point(pid, payload, iv, sv)   # cards: image + specifics
            if pt is not None:
                cards_pts.append(pt)
            if dv is not None:
                dino_pts.append(PointStruct(id=pid, vector={DINO_VEC: dv.tolist()},
                                            payload=payload))
            if iv is not None:
                img_buf.append(VectorRecord(**common, vector=iv.tolist(),
                    vector_type="image", model_id=IMAGE_MODEL_ID,
                    params_hash=IMAGE_PARAMS, source_url=r.get("gallery_url", "")))
            if sv is not None:
                spec_buf.append(VectorRecord(**common, vector=sv.tolist(),
                    vector_type="specifics", model_id=TEXT_MODEL_ID,
                    params_hash=TEXT_PARAMS, source_url=""))
            if dv is not None:
                dino_buf.append(VectorRecord(**common, vector=dv.tolist(),
                    vector_type="image", model_id=DINO_MODEL,
                    params_hash=DINO_PARAMS, source_url=r.get("gallery_url", "")))

        # S3 (durable) before Qdrant.
        img_n  = _drain(img_buf,  img_n)
        spec_n = _drain(spec_buf, spec_n)
        dino_n = _drain(dino_buf, dino_n)
        qdrant.upsert(collection_name=CARDS_COLLECTION,  points=cards_pts, wait=True)
        qdrant.upsert(collection_name=DINO_COLLECTION,   points=dino_pts,  wait=True)

        embedded += len(sub)
        if (i // batch) % 20 == 0:
            logger.info("  [{}] {}/{} dual-embedded", date_str, embedded, total)

    _drain(img_buf,  img_n,  final=True)
    _drain(spec_buf, spec_n, final=True)
    _drain(dino_buf, dino_n, final=True)
    return {"embedded": embedded, "total": total}


def build_index_list(os_client_getter, s3, args) -> list[str]:
    if args.date:
        return [args.date]
    today = date.today()
    idx = [(today - timedelta(days=i)).isoformat() for i in range(args.days)]
    if not args.no_nonebay:
        idx += recent_nonebay_indices(os_client_getter(), today)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=2, help="Recent eBay days to embed.")
    ap.add_argument("--date", help="Embed one specific index/day, then exit.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--download-workers", type=int, default=16)
    ap.add_argument("--no-nonebay", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-embed even if marker exists.")
    args = ap.parse_args()

    s3 = s3_client()
    store = S3VectorStore(bucket=QUEUE_BUCKET,
                          prefix=os.environ.get("S3_VECTOR_PREFIX", "vectors"))
    qdrant = get_qdrant_client()
    ensure_collection(qdrant)   # cards_dinov2 (cards already exists)

    device = _pick_device()
    logger.info("Loading CLIP ViT-L/14 + DINOv2 @{} on {} …", DINO_SIZE, device)
    clip_enc = ImageEncoder(model_name="ViT-L/14", pretrained="openai", device=device)
    clip_enc._load()
    text_enc = TextEncoder(device="cpu")
    dino_encode, dim = build_encoder("dinov2", DINO_SIZE, True, device)
    assert dim == 1024, dim
    pool = ThreadPoolExecutor(max_workers=args.download_workers)

    from src.ingestion.opensearch_reader import get_opensearch_client
    indices = build_index_list(get_opensearch_client, s3, args)
    logger.info("Dual-embed targets: {}", indices)

    for idx in indices:
        if not args.force and is_complete(s3, idx):
            logger.info("{} already dual-embedded — skipping", idx)
            continue
        t0 = time.time()
        stats = embed_day(idx, clip_enc, text_enc, dino_encode, qdrant, store,
                          s3, pool, args.batch)
        stats["seconds"] = round(time.time() - t0, 1)
        if stats.get("total", 0) > 0:
            mark_complete(s3, idx, stats)
        logger.info("Done {} — dual-embedded {}/{} in {:.0f}s → cards + {}",
                    idx, stats["embedded"], stats["total"], stats["seconds"],
                    DINO_COLLECTION)


if __name__ == "__main__":
    main()
