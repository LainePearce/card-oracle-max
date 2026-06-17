#!/usr/bin/env python3
"""
POC component 2 — embed each archived image with all three backbones.

Reads the manifest from poc_image_archive.py, pulls each card's 512px image
from S3, and embeds it with CLIP (224/fp16), DINOv2 (512/fp16) and DINOv3
(512/fp32). Each batch of images is downloaded once and run through all three
encoders in sequence, so only the model weights co-reside on the GPU (one
encoder's activations live at a time) — fits a 16GB T4.

Vectors are written to the S3 vector store (the durable, reusable copy) with
distinct (model_id, params) keys per backbone, exactly like production backfill.
poc_load_qdrant.py then loads them into the POC collection.

    python tools/poc_triple_embed.py --manifest data/poc/manifest_2026-06-01.jsonl \
        --date 2026-06-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger

from src.embeddings.vector_store import S3VectorStore, VectorRecord, SHARD_SIZE
from tools.poc_common import (
    POC_ENCODERS, DINO_IMAGE_SIZE, make_s3, get_image_pil, image_key, poc_job_id,
)
from tools.eval_retrieval_at_scale import build_encoder
from tools.eval_parallel_discrimination import _pick_device


def load_batch_images(s3, bucket: str, rows: list[dict]):
    """Load the 512px variant for a batch; substitute grey on failure (kept aligned)."""
    from PIL import Image
    grey = Image.new("RGB", (224, 224), (114, 114, 114))
    out = []
    for r in rows:
        key = r["s3_keys"].get("512") or image_key(r["os_id"], "512")
        try:
            out.append(get_image_pil(s3, bucket, key))
        except Exception:
            out.append(grey)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--date", required=True, help="Partition (YYYY-MM-DD).")
    ap.add_argument("--bucket", default=os.environ.get("S3_IMAGE_BUCKET")
                    or os.environ.get("S3_VECTOR_BUCKET"))
    ap.add_argument("--vector-bucket", default=os.environ.get("S3_VECTOR_BUCKET"))
    ap.add_argument("--vector-prefix", default=os.environ.get("S3_VECTOR_PREFIX", "vectors"))
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / args.manifest)]
    logger.info("Embedding {} cards with {} backbones", len(rows), len(POC_ENCODERS))

    device = _pick_device()
    s3 = make_s3()
    store = S3VectorStore(bucket=args.vector_bucket, prefix=args.vector_prefix)
    job_id = poc_job_id(args.date)

    # Load all three encoders once (weights co-resident; activations sequential).
    encoders = {}
    for spec in POC_ENCODERS:
        logger.info("Loading {} ({}) …", spec.encoder, spec.vector_name)
        enc, dim = build_encoder(spec.encoder, DINO_IMAGE_SIZE, spec.fp16, device)
        assert dim == spec.dim, f"{spec.encoder} dim {dim} != expected {spec.dim}"
        encoders[spec.vector_name] = enc

    # Per-backbone shard buffers.
    buffers: dict[str, list[VectorRecord]] = {s.vector_name: [] for s in POC_ENCODERS}
    shard_num: dict[str, int] = {s.vector_name: 0 for s in POC_ENCODERS}

    def flush(vector_name: str, force: bool = False) -> None:
        buf = buffers[vector_name]
        if buf and (force or len(buf) >= SHARD_SIZE):
            store.write_shard(buf, shard_num[vector_name])
            logger.info("  wrote {} shard {:04d} ({} vecs)",
                        vector_name, shard_num[vector_name], len(buf))
            shard_num[vector_name] += 1
            buffers[vector_name] = []

    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        pils = load_batch_images(s3, args.bucket, chunk)
        for spec in POC_ENCODERS:
            vecs = encoders[spec.vector_name](pils)
            for r, v in zip(chunk, vecs):
                buffers[spec.vector_name].append(VectorRecord(
                    os_id=r["os_id"], qdrant_id=r["qdrant_id"],
                    index_name=args.date, index_type="ebay-dated",
                    vector=v.tolist(), vector_type="image",
                    model_id=spec.model_id, params_hash=spec.params,
                    job_id=job_id, source_url=r.get("gallery_url", ""),
                ))
            flush(spec.vector_name)
        if (i + args.batch) % (args.batch * 25) == 0:
            logger.info("  embedded {}/{}", i + args.batch, len(rows))

    for spec in POC_ENCODERS:
        flush(spec.vector_name, force=True)

    logger.info("─" * 60)
    logger.info("Done — 3 vector sets persisted to s3://{}/{} (job {})",
                args.vector_bucket, args.vector_prefix, job_id)
    for spec in POC_ENCODERS:
        logger.info("  {:12} model_id={} params={}",
                    spec.vector_name, spec.model_id, spec.params)


if __name__ == "__main__":
    main()
