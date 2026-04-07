#!/usr/bin/env python3
"""
Qdrant → S3 Vector Export Tool.

Scrolls all points from a Qdrant collection and writes their vectors to S3
as Parquet shards, using the S3VectorStore format defined in CLAUDE.md §16.

Run this before ANY Qdrant infrastructure change (cluster migration, collection
recreation, node replacement) so vectors can be repopulated from S3 without
re-embedding.

The exported data can be reloaded into any Qdrant instance via:
    python -m src.embeddings.vector_store repopulate \\
        --bucket <bucket> --prefix vectors \\
        --vector-type image --model clip-vit-l-14 \\
        --params v2-fp16-224px-sqpad --index-type qdrant-export \\
        --qdrant-host <host> --collection cards

Usage:
    # Dry run (validate read path, no S3 writes)
    python tools/qdrant_to_s3_export.py --dry-run

    # Full export — image AND specifics vectors
    python tools/qdrant_to_s3_export.py \\
        --collection cards \\
        --s3-bucket card-oracle-vectors \\
        --s3-prefix vectors \\
        --vector-type both

    # Export only image vectors
    python tools/qdrant_to_s3_export.py --vector-type image

    # Resume from checkpoint
    python tools/qdrant_to_s3_export.py \\
        --checkpoint-id export-2026-04-06 \\
        --vector-type both

Environment variables (loaded from .env):
    QDRANT_HOST          Qdrant host      (default: localhost)
    QDRANT_PORT          Qdrant REST port (default: 6333)
    QDRANT_API_KEY       Qdrant API key
    QDRANT_COLLECTION    collection name  (default: cards)

    S3_VECTOR_BUCKET     S3 bucket name
    S3_VECTOR_PREFIX     S3 key prefix    (default: vectors)
    AWS_REGION           AWS region       (default: us-west-1)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.embeddings.vector_store import S3VectorStore, VectorRecord

# ── Configuration ─────────────────────────────────────────────────────────────

QDRANT_HOST       = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cards")

S3_BUCKET = os.environ.get("S3_VECTOR_BUCKET", "")
S3_PREFIX = os.environ.get("S3_VECTOR_PREFIX", "vectors")

# Metadata for vectors currently in Qdrant (produced by rds_batch_job.py)
IMAGE_MODEL_ID    = "clip-vit-l-14"
IMAGE_PARAMS_HASH = "v2-fp16-224px-sqpad"   # square-pad + FP16 CLIP ViT-L/14
SPECS_MODEL_ID    = "minilm-l6-v2"
SPECS_PARAMS_HASH = "v1-mean-256tok"        # mean pooling, 256-token MiniLM

# Export index type — used in S3 key since per-date info is not in payload
EXPORT_INDEX_TYPE = "qdrant-export"
EXPORT_PARTITION  = "all"

SCROLL_LIMIT      = 1000     # points per Qdrant scroll page
SHARD_SIZE        = 10_000   # vectors per S3 Parquet file
CHECKPOINT_EVERY  = 10_000   # save checkpoint every N points


# ── Qdrant source field → S3 index_type mapping ───────────────────────────────

def source_to_index_type(source: str) -> str:
    """Map payload 'source' value to S3 index_type string."""
    src = (source or "").lower().strip()
    if src in ("ebay", "ebay_uk", "ebay_us", "ebay-gb", "ebay-us", ""):
        return "ebay-dated"
    if "pwcc" in src or "fanatics" in src:
        return "pwcc"
    if "goldin" in src or "gold" in src:
        return "gold"
    if "pristine" in src or "pris" in src:
        return "pris"
    if "heritage" in src or "heri" in src:
        return "heritage"
    if "myslabs" in src or "ms" in src:
        return "ms"
    return "unknown"


# ── Scroll + export ───────────────────────────────────────────────────────────

def export_to_s3(
    collection:   str,
    vector_types: list[str],   # ["image"], ["specifics"], or ["image", "specifics"]
    store:        S3VectorStore | None,
    checkpoint_id: str,
    dry_run:      bool,
    resume_offset: str | None = None,
) -> dict:
    """
    Scroll all points from Qdrant and write vectors to S3.

    Returns a summary dict with counts.
    """
    from qdrant_client import QdrantClient

    kwargs: dict = {
        "url":         f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        "prefer_grpc": False,
        "timeout":     60,
    }
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    client = QdrantClient(**kwargs)

    # Verify collection exists
    try:
        info = client.get_collection(collection)
        total_points = info.points_count
        logger.info("Collection '{}': {:,} points total", collection, total_points)
    except Exception as e:
        logger.error("Cannot access collection '{}': {}", collection, e)
        sys.exit(1)

    # Buffers: one per (vector_type, index_type) — we use one flat export bucket
    # Key: vector_type ("image"|"specifics")
    buffers:     dict[str, list[VectorRecord]] = {vt: [] for vt in vector_types}
    shard_nums:  dict[str, int] = {vt: 0 for vt in vector_types}
    total_written: dict[str, int] = {vt: 0 for vt in vector_types}
    s3_keys_written: list[str] = []

    points_scrolled = 0
    offset = resume_offset    # None = start from beginning; str UUID = resume

    t_start = time.perf_counter()
    t_last_log = t_start

    logger.info("Starting export — vector_types={}, dry_run={}", vector_types, dry_run)

    while True:
        # Scroll a page of points
        try:
            records, next_offset = client.scroll(
                collection_name=collection,
                offset=offset,
                limit=SCROLL_LIMIT,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as e:
            logger.error("Qdrant scroll failed at offset={}: {}", offset, e)
            raise

        if not records:
            break

        for point in records:
            points_scrolled += 1

            # point.vector is dict[str, list[float]] for named vectors
            vec_dict = point.vector if isinstance(point.vector, dict) else {}
            payload  = point.payload or {}
            os_id    = str(payload.get("os_id", point.id))
            qdrant_id = str(point.id)
            source   = payload.get("source", "")
            idx_type = source_to_index_type(source)
            specs_src = str(payload.get("specifics_source", "ebay"))

            # Build gallery URL for source_url (best effort from payload)
            gallery_url = str(payload.get("gallery_url", ""))

            for vt in vector_types:
                vec = vec_dict.get(vt)
                if vec is None:
                    continue   # this point has no vector of this type — skip

                model_id    = IMAGE_MODEL_ID    if vt == "image" else SPECS_MODEL_ID
                params_hash = IMAGE_PARAMS_HASH if vt == "image" else SPECS_PARAMS_HASH

                rec = VectorRecord(
                    os_id        = os_id,
                    qdrant_id    = qdrant_id,
                    index_name   = EXPORT_INDEX_TYPE,  # no per-date info in payload
                    index_type   = EXPORT_INDEX_TYPE,
                    vector       = vec if isinstance(vec, list) else list(vec),
                    vector_type  = vt,
                    model_id     = model_id,
                    params_hash  = params_hash,
                    source_url   = gallery_url if vt == "image" else "",
                    specifics_src= specs_src,
                )
                buffers[vt].append(rec)

                # Flush shard when full
                if len(buffers[vt]) >= SHARD_SIZE:
                    key = _flush_shard(buffers[vt], shard_nums[vt], store, dry_run)
                    if key:
                        s3_keys_written.append(key)
                    total_written[vt] += len(buffers[vt])
                    shard_nums[vt] += 1
                    buffers[vt] = []

        # Checkpoint
        if store and points_scrolled % CHECKPOINT_EVERY == 0:
            _save_checkpoint(store, checkpoint_id, {
                "offset":          str(next_offset) if next_offset else None,
                "points_scrolled": points_scrolled,
                "shard_nums":      shard_nums,
                "total_written":   total_written,
            }, dry_run)

        # Progress log
        now = time.perf_counter()
        if now - t_last_log > 15:
            elapsed = now - t_start
            rate = points_scrolled / elapsed if elapsed > 0 else 0
            pct = (points_scrolled / total_points * 100) if total_points else 0
            logger.info(
                "Progress: {:,}/{:,} ({:.1f}%) @ {:.0f} pts/s | written={}",
                points_scrolled, total_points, pct, rate,
                {vt: total_written[vt] for vt in vector_types},
            )
            t_last_log = now

        offset = next_offset
        if next_offset is None:
            break   # end of collection

    # Flush remaining buffers
    for vt in vector_types:
        if buffers[vt]:
            key = _flush_shard(buffers[vt], shard_nums[vt], store, dry_run)
            if key:
                s3_keys_written.append(key)
            total_written[vt] += len(buffers[vt])
            shard_nums[vt] += 1
            buffers[vt] = []

    elapsed = time.perf_counter() - t_start
    summary = {
        "points_scrolled": points_scrolled,
        "vectors_written": total_written,
        "s3_files":        len(s3_keys_written),
        "elapsed_s":       round(elapsed, 1),
    }
    return summary


def _flush_shard(
    records:   list[VectorRecord],
    shard_num: int,
    store:     S3VectorStore | None,
    dry_run:   bool,
) -> str | None:
    """Write a shard to S3. Returns S3 key written, or None on dry-run."""
    if dry_run or store is None:
        vt = records[0].vector_type if records else "?"
        logger.debug("[dry-run] Would write shard {} for '{}' ({} records)",
                     shard_num, vt, len(records))
        return None
    key = store.write_shard(records, shard_num)
    logger.debug("S3 shard written: {} ({} records)", key, len(records))
    return key


def _save_checkpoint(store, checkpoint_id, state, dry_run):
    if dry_run or store is None:
        return
    try:
        store.save_checkpoint(checkpoint_id, state)
    except Exception as e:
        logger.warning("Failed to save checkpoint: {}", e)


def _load_checkpoint(store, checkpoint_id) -> dict | None:
    if store is None:
        return None
    try:
        return store.load_checkpoint(checkpoint_id)
    except Exception:
        return None


# ── S3 cleanup: delete incompatible v1 data ───────────────────────────────────

def delete_incompatible_s3_data(bucket: str, dry_run: bool) -> int:
    """
    Delete v1-fp16-224px image and v1 specifics/pwcc|ebay-dated data from S3.

    These were encoded without square-padding and used RDS integer id as os_id
    — both incompatible with the current system.

    v2-fp16-224px-sqpad data is KEPT (correct encoding).
    """
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))

    # Prefixes of data to delete (v1 encoding — wrong params AND wrong os_id)
    delete_prefixes = [
        "vectors/image/clip-vit-l-14/v1-fp16-224px/",
        "vectors/specifics/minilm-l6-v2/v1-mean-256tok/ebay-dated/",
        "vectors/specifics/minilm-l6-v2/v1-mean-256tok/pwcc/",
    ]

    paginator = s3.get_paginator("list_objects_v2")
    to_delete = []
    for prefix in delete_prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                to_delete.append(obj["Key"])

    if not to_delete:
        logger.info("No incompatible objects found — S3 already clean")
        return 0

    logger.info("{} incompatible objects to delete", len(to_delete))
    if dry_run:
        for k in to_delete[:5]:
            logger.info("  [dry-run] would delete: {}", k)
        if len(to_delete) > 5:
            logger.info("  ... and {} more", len(to_delete) - 5)
        return len(to_delete)

    # Delete in batches of 1000 (S3 delete_objects limit)
    deleted = 0
    for i in range(0, len(to_delete), 1000):
        batch = to_delete[i:i + 1000]
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch]},
        )
        deleted += len(batch)
        logger.info("Deleted {}/{} objects", deleted, len(to_delete))

    logger.info("Cleanup complete — {} incompatible objects removed", deleted)
    return deleted


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qdrant vectors to S3 Parquet shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--collection",    default=QDRANT_COLLECTION)
    parser.add_argument("--s3-bucket",     default=S3_BUCKET)
    parser.add_argument("--s3-prefix",     default=S3_PREFIX)
    parser.add_argument("--vector-type",   choices=["image", "specifics", "both"],
                        default="both")
    parser.add_argument("--checkpoint-id", default=None,
                        help="Checkpoint ID for resume. Auto-generated if not set.")
    parser.add_argument("--cleanup",       action="store_true",
                        help="Delete incompatible v1 data from S3 before export")
    parser.add_argument("--cleanup-only",  action="store_true",
                        help="Delete incompatible v1 data and exit (no export)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Validate without writing to S3")
    args = parser.parse_args()

    # Validate S3 bucket
    if not args.s3_bucket and not args.dry_run and not args.cleanup_only:
        logger.error("S3_VECTOR_BUCKET not set. Pass --s3-bucket or set env var.")
        sys.exit(1)

    # Build store
    store = S3VectorStore(args.s3_bucket, args.s3_prefix) if args.s3_bucket else None

    # ── Cleanup incompatible data ─────────────────────────────────────────────
    if args.cleanup or args.cleanup_only:
        logger.info("Removing incompatible v1 data from s3://{}/{} ...",
                    args.s3_bucket, args.s3_prefix)
        n = delete_incompatible_s3_data(args.s3_bucket, args.dry_run)
        logger.info("Cleanup: {} objects {}",
                    n, "would be deleted (dry-run)" if args.dry_run else "deleted")
        if args.cleanup_only:
            return

    # ── Export ───────────────────────────────────────────────────────────────
    vector_types = (
        ["image", "specifics"] if args.vector_type == "both"
        else [args.vector_type]
    )

    # Generate or use checkpoint ID
    checkpoint_id = args.checkpoint_id or (
        "qdrant-export-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )

    # Resume from checkpoint if it exists
    resume_offset = None
    if args.checkpoint_id and store:
        ckpt = _load_checkpoint(store, checkpoint_id)
        if ckpt:
            resume_offset = ckpt.get("offset")
            logger.info("Resuming from checkpoint: offset={}, scrolled so far={}",
                        resume_offset, ckpt.get("points_scrolled", 0))
        else:
            logger.info("No checkpoint found for '{}' — starting fresh", checkpoint_id)

    logger.info("─" * 60)
    logger.info("Qdrant → S3 Export")
    logger.info("  Qdrant:       {}:{}/{}", QDRANT_HOST, QDRANT_PORT, args.collection)
    logger.info("  S3:           s3://{}/{}", args.s3_bucket, args.s3_prefix)
    logger.info("  Vector types: {}", vector_types)
    logger.info("  Checkpoint:   {}", checkpoint_id)
    logger.info("  Dry run:      {}", args.dry_run)
    logger.info("─" * 60)

    try:
        summary = export_to_s3(
            collection    = args.collection,
            vector_types  = vector_types,
            store         = store,
            checkpoint_id = checkpoint_id,
            dry_run       = args.dry_run,
            resume_offset = resume_offset,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted — partial export saved. Resume with --checkpoint-id {}", checkpoint_id)
        sys.exit(130)

    logger.info("─" * 60)
    logger.info("Export complete:")
    logger.info("  Points scrolled:  {:,}", summary["points_scrolled"])
    for vt, count in summary["vectors_written"].items():
        logger.info("  {} vectors:    {:,}", vt, count)
    logger.info("  S3 files written: {}", summary["s3_files"])
    logger.info("  Elapsed:          {}s", summary["elapsed_s"])
    logger.info("─" * 60)

    if not args.dry_run and summary["s3_files"] > 0:
        logger.info("To reload into Qdrant:")
        for vt in vector_types:
            model  = IMAGE_MODEL_ID    if vt == "image" else SPECS_MODEL_ID
            params = IMAGE_PARAMS_HASH if vt == "image" else SPECS_PARAMS_HASH
            logger.info(
                "  python -m src.embeddings.vector_store repopulate "
                "--bucket {} --prefix {} --vector-type {} "
                "--model {} --params {} --index-type {} "
                "--qdrant-host <host> --collection {}",
                args.s3_bucket, args.s3_prefix, vt, model, params,
                EXPORT_INDEX_TYPE, args.collection,
            )


if __name__ == "__main__":
    main()
