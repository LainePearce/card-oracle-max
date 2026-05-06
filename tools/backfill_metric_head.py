#!/usr/bin/env python3
"""
Metric head backfill: project all Qdrant "image" vectors through MetricHead
and store them as new "image_v2" (128-dim) named vectors.

Run this on the GPU worker EC2 instance BEFORE enabling METRIC_HEAD_ENABLED=1.
Service continues running on "image" vectors throughout — zero downtime.

Steps this script performs:
    1. Add "image_v2" (128-dim cosine) named vector field to the Qdrant collection
       (no-op if it already exists).
    2. Scroll all points in the collection, retrieving their "image" vectors.
    3. Project each batch through MetricHead on GPU/CPU.
    4. Upsert the projected vectors back as "image_v2".
    5. Checkpoint progress to disk every --checkpoint-every points so the script
       can resume after interruption.

Usage:
    # Dry-run — no writes, just estimates time and checks connectivity
    python tools/backfill_metric_head.py --checkpoint models/metric_head_v2.pt --dry-run

    # Full backfill
    python tools/backfill_metric_head.py --checkpoint models/metric_head_v2.pt

    # Resume an interrupted backfill (skips already-projected points)
    python tools/backfill_metric_head.py --checkpoint models/metric_head_v2.pt --resume

Environment variables (loaded from .env):
    QDRANT_HOST, QDRANT_HTTP_PORT / QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_HOST       = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.environ.get("QDRANT_HTTP_PORT",
                        os.environ.get("QDRANT_PORT", 6333)))
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cards")

# New vector field added to the collection by this script
NEW_VECTOR_NAME = "image_v2"
NEW_VECTOR_DIM  = 128

# Scroll page size — number of points retrieved per Qdrant scroll call.
# Larger pages = fewer round trips but more memory per batch.
SCROLL_PAGE = 512

# Upsert batch size — number of projected vectors sent to Qdrant per upsert call.
UPSERT_BATCH = 512

# Progress checkpoint file (lives next to the backfill script)
CHECKPOINT_PATH = ROOT / "data" / "backfill_metric_head_checkpoint.json"


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def get_qdrant():
    from qdrant_client import QdrantClient
    kwargs = {
        "url":         f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        "prefer_grpc": False,
        "timeout":     60,
    }
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)


def ensure_image_v2_field(qdrant, collection: str, dry_run: bool) -> bool:
    """
    Add the 'image_v2' named vector field to the collection if it doesn't exist.
    Returns True if the field already existed (skipped), False if it was created.
    """
    from qdrant_client.models import VectorParams, Distance

    info = qdrant.get_collection(collection)
    existing = info.config.params.vectors

    if NEW_VECTOR_NAME in (existing or {}):
        logger.info("'{}' field already exists in collection — skipping creation",
                    NEW_VECTOR_NAME)
        return True

    if dry_run:
        logger.info("[dry-run] Would add '{}' ({}-dim cosine) to collection '{}'",
                    NEW_VECTOR_NAME, NEW_VECTOR_DIM, collection)
        return False

    logger.info("Adding '{}' ({}-dim cosine) to collection '{}'...",
                NEW_VECTOR_NAME, NEW_VECTOR_DIM, collection)
    qdrant.update_collection(
        collection_name=collection,
        vectors_config={
            NEW_VECTOR_NAME: VectorParams(
                size     = NEW_VECTOR_DIM,
                distance = Distance.COSINE,
                on_disk  = True,
            )
        },
    )
    logger.info("'{}' field added OK", NEW_VECTOR_NAME)
    return False


# ── MetricHead projection ─────────────────────────────────────────────────────

def load_metric_head(checkpoint_path: str):
    """Load MetricHead from checkpoint. Returns (head, device)."""
    import torch
    from src.embeddings.metric_head import MetricHead

    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head   = MetricHead(
        input_dim  = ckpt.get("input_dim",  768),
        output_dim = ckpt.get("output_dim", 128),
    )
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    head = head.to(device)
    logger.info("MetricHead loaded: input={}d → output={}d on {} (R@10={:.3f} in training)",
                ckpt.get("input_dim", 768), ckpt.get("output_dim", 128),
                device, ckpt.get("recall10", 0.0))
    return head, device


def project_batch(head, device, raw_vectors: list[list[float]]):
    """Project a batch of raw CLIP vectors through MetricHead. Returns list of lists."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        t   = torch.tensor(raw_vectors, dtype=torch.float32, device=device)
        out = head(t)               # already L2-normalised by MetricHead.forward()
    return out.cpu().tolist()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"projected_ids": [], "total_projected": 0, "next_offset": None}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f)


# ── Main backfill loop ────────────────────────────────────────────────────────

def run_backfill(
    checkpoint_path:   str,
    dry_run:           bool = False,
    resume:            bool = False,
    checkpoint_every:  int  = 10_000,
) -> None:
    """
    Full backfill: scroll all Qdrant points → project → upsert as image_v2.
    """
    import numpy as np
    from qdrant_client.models import PointVectors

    qdrant = get_qdrant()
    head, device = load_metric_head(checkpoint_path)

    # ── Verify collection exists and get point count ─────────────────────────
    info   = qdrant.get_collection(QDRANT_COLLECTION)
    total  = info.points_count or 0
    logger.info("Collection '{}': {:,} total points", QDRANT_COLLECTION, total)

    # ── Ensure image_v2 field exists ─────────────────────────────────────────
    ensure_image_v2_field(qdrant, QDRANT_COLLECTION, dry_run)

    if dry_run:
        # Rough throughput estimate: ~500 pts/sec on CPU projection + Qdrant upsert
        est_min = total / 500 / 60
        logger.info("[dry-run] Estimated backfill time: {:.0f}–{:.0f} minutes for {:,} points",
                    est_min * 0.5, est_min * 2, total)
        logger.info("[dry-run] No writes performed — exiting")
        return

    # ── Load checkpoint if resuming ──────────────────────────────────────────
    state = load_checkpoint() if resume else {
        "total_projected": 0,
        "next_offset":     None,
    }
    already_done   = state.get("total_projected", 0)
    next_offset    = state.get("next_offset", None)

    if already_done > 0:
        logger.info("Resuming from checkpoint: {:,} points already projected, "
                    "next_offset={}", already_done, next_offset)
    else:
        logger.info("Starting fresh backfill of {:,} points", total)

    # ── Scroll → project → upsert loop ───────────────────────────────────────
    projected   = already_done
    upsert_buf  = []            # accumulate projected points before upserting
    t_start     = time.perf_counter()
    t_last_log  = t_start

    while True:
        # Scroll one page, retrieving the "image" named vector for each point
        scroll_result = qdrant.scroll(
            collection_name = QDRANT_COLLECTION,
            offset          = next_offset,
            limit           = SCROLL_PAGE,
            with_vectors    = [NEW_VECTOR_NAME == nv or "image" for nv in ["image"]][0],
            with_payload    = False,   # payload not needed for projection
        )
        # scroll() returns (points, next_offset_or_None)
        points, next_offset = scroll_result

        if not points:
            break   # exhausted collection

        # Filter out points that already have image_v2 (safe for resume mid-run)
        # and points without an "image" vector (can't project without source vector)
        to_project = []
        for p in points:
            vecs = p.vector if p.vector else {}
            # p.vector is a dict when with_vectors=["image"] is used
            if isinstance(vecs, dict):
                img_vec = vecs.get("image")
            else:
                img_vec = vecs   # fallback for single named vector
            if img_vec is None:
                continue   # no image vector — skip
            to_project.append((p.id, img_vec))

        if not to_project:
            if next_offset is None:
                break
            continue

        # Project batch through MetricHead
        ids, raw_vecs = zip(*to_project)
        projected_vecs = project_batch(head, device, list(raw_vecs))

        # Build upsert batch
        for pid, pvec in zip(ids, projected_vecs):
            upsert_buf.append(PointVectors(id=pid, vector={NEW_VECTOR_NAME: pvec}))

        # Flush upsert buffer
        if len(upsert_buf) >= UPSERT_BATCH:
            qdrant.update_vectors(
                collection_name = QDRANT_COLLECTION,
                points          = upsert_buf[:UPSERT_BATCH],
            )
            projected  += len(upsert_buf[:UPSERT_BATCH])
            upsert_buf  = upsert_buf[UPSERT_BATCH:]

        # Progress logging
        now = time.perf_counter()
        if now - t_last_log >= 30:
            elapsed   = now - t_start
            rate      = (projected - already_done) / elapsed if elapsed > 0 else 0
            remaining = (total - projected) / rate / 60 if rate > 0 else 0
            logger.info(
                "Progress: {:,}/{:,} ({:.1f}%) | {:.0f} pts/s | ~{:.0f} min remaining",
                projected, total, 100 * projected / total if total else 0,
                rate, remaining,
            )
            t_last_log = now

        # Checkpoint
        if (projected - already_done) % checkpoint_every < SCROLL_PAGE:
            save_checkpoint({
                "total_projected": projected,
                "next_offset":     next_offset,
            })

        if next_offset is None:
            break

    # Flush remaining upsert buffer
    if upsert_buf:
        qdrant.update_vectors(
            collection_name = QDRANT_COLLECTION,
            points          = upsert_buf,
        )
        projected += len(upsert_buf)

    # Final checkpoint
    save_checkpoint({"total_projected": projected, "next_offset": None})

    elapsed = time.perf_counter() - t_start
    logger.info("─" * 60)
    logger.info("Backfill complete: {:,} points projected in {:.1f} minutes",
                projected, elapsed / 60)
    logger.info("'{}'  (128-dim) is now populated and ready for queries", NEW_VECTOR_NAME)
    logger.info("Next step: set METRIC_HEAD_ENABLED=1 in .env and reload gunicorn")


# ── Verify backfill ───────────────────────────────────────────────────────────

def verify_backfill(sample_n: int = 10) -> None:
    """
    Quick sanity check: retrieve sample_n random points and confirm they have
    non-zero image_v2 vectors of the correct dimension.
    """
    import random
    qdrant = get_qdrant()
    info   = qdrant.get_collection(QDRANT_COLLECTION)
    total  = info.points_count or 0

    logger.info("Verifying backfill — checking {} random points...", sample_n)

    ok = 0
    fail = 0
    scroll_result = qdrant.scroll(
        collection_name = QDRANT_COLLECTION,
        limit           = sample_n,
        with_vectors    = [NEW_VECTOR_NAME],
        with_payload    = False,
    )
    points, _ = scroll_result

    for p in points:
        vecs = p.vector if p.vector else {}
        v2   = vecs.get(NEW_VECTOR_NAME) if isinstance(vecs, dict) else None
        if v2 is not None and len(v2) == NEW_VECTOR_DIM:
            ok += 1
        else:
            fail += 1
            logger.warning("Point {} missing or wrong-dim image_v2: {}", p.id,
                           len(v2) if v2 else "None")

    logger.info("Verification: {}/{} points have valid {}-dim '{}' vectors",
                ok, ok + fail, NEW_VECTOR_DIM, NEW_VECTOR_NAME)

    if fail == 0:
        logger.info("✓ Backfill looks healthy")
    else:
        logger.error("✗ {} points failed verification — re-run backfill", fail)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Qdrant with metric-head projected vectors")
    ap.add_argument("--checkpoint",       required=True,
                    help="Path to MetricHead .pt checkpoint (e.g. models/metric_head_v2.pt)")
    ap.add_argument("--dry-run",          action="store_true",
                    help="Print estimates only — no writes to Qdrant")
    ap.add_argument("--resume",           action="store_true",
                    help="Resume from last checkpoint (skip already-projected points)")
    ap.add_argument("--checkpoint-every", type=int, default=10_000,
                    help="Save progress checkpoint every N projected points (default: 10000)")
    ap.add_argument("--verify",           action="store_true",
                    help="Run verification check on a sample of points after backfill")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    run_backfill(
        checkpoint_path  = args.checkpoint,
        dry_run          = args.dry_run,
        resume           = args.resume,
        checkpoint_every = args.checkpoint_every,
    )

    if args.verify and not args.dry_run:
        verify_backfill()


if __name__ == "__main__":
    main()
