#!/usr/bin/env python3
"""
DINOv2 embed fleet worker (GPU).

Loops: claim an archived day from the dino-embed queue, read its image-archive
manifest, embed every 512px image with DINOv2-large @512 (fp16) into 1024-d
vectors, persist them to the S3 vector store, upsert into the cards_dinov2
Qdrant collection (named vector image_dinov2 + payload), and mark the day done.

The DINOv2 model loads once per process and is reused across days. Images are
read from S3 (no source re-fetch). On crash the day re-queues and re-embeds
idempotently (deterministic S3 keys + upsert-on-id).

    python tools/dino_embed_worker.py --batch 32 --loop
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
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from qdrant_client.models import (
    Distance, HnswConfigDiff, OptimizersConfigDiff, PointStruct,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType, VectorParams,
)

from src.embeddings.vector_store import S3VectorStore, VectorRecord, SHARD_SIZE
from src.ingestion.qdrant_writer import get_qdrant_client, extract_payload
from tools.poc_common import get_image_pil, image_key
from tools.eval_retrieval_at_scale import build_encoder
from tools.eval_parallel_discrimination import _pick_device
from tools.dino_embed_common import (
    s3_client, claim_next, mark_complete, update_active, release,
    QUEUE_BUCKET, IMAGE_BUCKET, ARCHIVE_MANIFESTS,
)

COLLECTION = os.environ.get("QDRANT_DINOV2_COLLECTION", "cards_dinov2")
VEC_NAME   = "image_dinov2"
MODEL_ID   = "dinov2-large"
PARAMS     = "512px-fp16-sqpad"
DINO_SIZE  = 512


def ensure_collection(client) -> None:
    try:
        if any(c.name == COLLECTION for c in client.get_collections().collections):
            return
        single = os.environ.get("QDRANT_SINGLE_NODE", "false").lower() in ("true", "1", "yes")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={VEC_NAME: VectorParams(
                size=1024, distance=Distance.COSINE, on_disk=True,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200, on_disk=True))},
            quantization_config=ScalarQuantization(scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8, quantile=0.99, always_ram=True)),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=50_000,
                                                   memmap_threshold=50_000),
            on_disk_payload=True,
            shard_number=1 if single else 6,
            replication_factor=1 if single else 2,
        )
        logger.info("Created collection '{}' (image_dinov2 1024-d)", COLLECTION)
    except Exception as e:
        # Concurrent create from another worker is fine — collection now exists.
        logger.info("ensure_collection: {} ({}) — assuming it exists", COLLECTION, e)


def read_manifest(s3, date_str: str) -> list[dict] | None:
    key = f"{ARCHIVE_MANIFESTS}/{date_str}.jsonl.gz"
    try:
        raw = s3.get_object(Bucket=QUEUE_BUCKET, Key=key)["Body"].read()
    except Exception:
        return None
    rows: list[dict] = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        for line in gz:
            rows.append(json.loads(line))
    return rows


def load_images(pool, s3, rows):
    from PIL import Image
    grey = Image.new("RGB", (224, 224), (114, 114, 114))

    def _one(r):
        key = r["s3_keys"].get("512") or image_key(r["os_id"], "512")
        try:
            return get_image_pil(s3, IMAGE_BUCKET, key)
        except Exception:
            return grey

    return list(pool.map(_one, rows))


def embed_day(date_str, encode, client, store, s3, pool, batch) -> dict:
    rows = read_manifest(s3, date_str)
    if rows is None:
        return {"embedded": 0, "total": 0, "note": "no manifest"}
    total = len(rows)
    job_id = f"dino-{date_str}"
    update_active(s3, date_str, {"total": total, "vectors": 0})

    shard_buf: list[VectorRecord] = []
    shard_num = 0
    done = 0
    for i in range(0, total, batch):
        chunk = rows[i:i + batch]
        vecs = encode(load_images(pool, s3, chunk))
        points = []
        for r, v in zip(chunk, vecs):
            vl = v.tolist()
            shard_buf.append(VectorRecord(
                os_id=r["os_id"], qdrant_id=r["qdrant_id"],
                index_name=date_str, index_type="ebay-dated",
                vector=vl, vector_type="image",
                model_id=MODEL_ID, params_hash=PARAMS,
                job_id=job_id, source_url=r.get("gallery_url", "")))
            payload = extract_payload(r["source_doc"], doc_id=r["os_id"])
            payload["img_512"] = r["s3_keys"].get("512", "")
            payload["img_256"] = r["s3_keys"].get("256", "")
            points.append(PointStruct(id=int(r["qdrant_id"]),
                                      vector={VEC_NAME: vl}, payload=payload))
        # S3 (durable) before Qdrant; flush whole shards as they fill.
        while len(shard_buf) >= SHARD_SIZE:
            store.write_shard(shard_buf[:SHARD_SIZE], shard_num)
            shard_num += 1
            shard_buf = shard_buf[SHARD_SIZE:]
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        done += len(chunk)
        if (i // batch) % 20 == 0:
            update_active(s3, date_str, {"total": total, "vectors": done})
            logger.info("  [{}] {}/{} embedded", date_str, done, total)
    if shard_buf:
        store.write_shard(shard_buf, shard_num)
    return {"embedded": done, "total": total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--download-workers", type=int, default=16)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--idle-sleep", type=int, default=120)
    args = ap.parse_args()

    s3 = s3_client()
    store = S3VectorStore(bucket=QUEUE_BUCKET,
                          prefix=os.environ.get("S3_VECTOR_PREFIX", "vectors"))
    client = get_qdrant_client()
    ensure_collection(client)

    device = _pick_device()
    logger.info("Loading DINOv2 @{} on {} …", DINO_SIZE, device)
    encode, dim = build_encoder("dinov2", DINO_SIZE, True, device)
    assert dim == 1024, dim
    pool = ThreadPoolExecutor(max_workers=args.download_workers)
    logger.info("DINOv2 embed worker ready → collection '{}'", COLLECTION)

    while True:
        date_str = claim_next(s3)
        if date_str is None:
            if args.loop:
                time.sleep(args.idle_sleep)
                continue
            logger.info("Queue empty — exiting")
            return
        logger.info("Claimed {}", date_str)
        t0 = time.time()
        try:
            stats = embed_day(date_str, encode, client, store, s3, pool, args.batch)
            stats["seconds"] = round(time.time() - t0, 1)
            mark_complete(s3, date_str, stats)
            logger.info("Done {} — embedded {}/{} in {:.0f}s",
                        date_str, stats["embedded"], stats["total"], stats["seconds"])
        except Exception as e:
            logger.error("{} failed ({}: {}) — releasing", date_str, type(e).__name__, e)
            release(s3, date_str)
            time.sleep(5)


if __name__ == "__main__":
    main()
