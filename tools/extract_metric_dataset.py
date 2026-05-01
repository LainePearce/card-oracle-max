#!/usr/bin/env python3
"""
Extract a labeled dataset of CLIP image vectors from Qdrant for metric head training.

Runs on EC2 (requires VPC access to the Qdrant NLB).
Output is a Parquet file that can be copied to the local machine for training.

Steps:
  1. Scroll Qdrant fetching image vectors + identity payload fields
  2. Filter: has_image=True, specifics_source="ebay", essential fields non-empty
  3. Build Tier-1/2/3 identity keys per point (incorporating parallel resolution)
  4. Keep only identity keys that appear ≥ MIN_POSITIVES times (triplet loss needs pairs)
  5. Cap at MAX_PER_KEY per key (prevents imbalance from hyper-common cards)
  6. Write data/metric_dataset.parquet

Usage (on EC2 worker):
    cd ~/card-oracle-max && source .venv/bin/activate
    python tools/extract_metric_dataset.py
    # Large-scale run (~1M training points, scroll 7M, cap 200 per key):
    python tools/extract_metric_dataset.py --sample 7000000 --min-positives 2 --max-per-key 200
    # Original small test run:
    python tools/extract_metric_dataset.py --sample 200000 --min-positives 2 --max-per-key 50

Copy result to local:
    scp ec2-user@<ip>:~/card-oracle-max/data/metric_dataset.parquet ./data/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq


# ── Config ───────────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE        = 7_000_000  # total points to scroll from Qdrant (~1M selected)
DEFAULT_MIN_POSITIVES = 2          # minimum instances per identity key (for pairs)
DEFAULT_MAX_PER_KEY   = 200        # cap per key to avoid imbalance from hyper-common cards

# Payload fields to retrieve (vectors always requested)
PAYLOAD_FIELDS = [
    "has_image", "specifics_source",
    "player", "set", "card_number", "year", "brand", "genre",
    "parallel", "graded", "grader", "grade",
]

# Essential fields — point is skipped if any are empty
ESSENTIAL_FIELDS = {"player", "set", "card_number"}

# Parallel DB path (built by build_parallel_db.py)
PARALLEL_DB_PATH = _ROOT / "data" / "parallel_db.json"
PARALLEL_TOK_PATH = _ROOT / "data" / "parallel_tokens.json"

DATA_DIR = _ROOT / "data"


# ── Parallel resolution ─────────────────────────────────────────────────────────

def _load_parallel_db() -> tuple[dict, dict]:
    """Load the parallel dictionary and token index if they exist."""
    db, tokens = {}, {}
    if PARALLEL_DB_PATH.exists():
        with open(PARALLEL_DB_PATH, encoding="utf-8") as f:
            db = json.load(f)
        logger.info("Parallel DB loaded — {:,} entries", len(db))
    else:
        logger.warning("Parallel DB not found at {} — parallel resolution disabled", PARALLEL_DB_PATH)
    if PARALLEL_TOK_PATH.exists():
        with open(PARALLEL_TOK_PATH, encoding="utf-8") as f:
            tokens = json.load(f)
    return db, tokens


def resolve_parallel(payload: dict, parallel_db: dict, token_index: dict) -> str:
    """
    Return the resolved parallel label for a Qdrant payload:
      1. Use payload['parallel'] directly if it's a known DB entry
      2. Skip (return '') — title not available in Qdrant payload
    Returns '' for unresolvable / absent parallels.
    """
    raw = (payload.get("parallel") or "").lower().strip()
    if raw and raw in parallel_db:
        return raw
    # If raw value exists but isn't in DB, return it anyway (may be rare variant)
    if raw and len(raw) >= 2:
        return raw
    return ""


# ── Identity keys ────────────────────────────────────────────────────────────────

def _clean(v) -> str:
    if v is None:
        return ""
    return str(v).lower().strip()


def make_identity_keys(payload: dict, parallel_db: dict, token_index: dict) -> dict:
    """
    Return a dict of identity keys at each tier:
      tier3: player|set|card_number             (base card identity)
      tier2: player|set|card_number|parallel     (variant)
      tier1: player|set|card_number|parallel|grader|grade  (exact graded copy)

    Returns {} if essential fields are missing.
    """
    player  = _clean(payload.get("player"))
    card_set = _clean(payload.get("set"))
    card_num = _clean(payload.get("card_number"))

    if not all([player, card_set, card_num]):
        return {}

    # Reject generic noise values
    noise = {"unknown", "n/a", "na", "null", "none", "", "-"}
    if any(v in noise for v in [player, card_set, card_num]):
        return {}

    parallel = resolve_parallel(payload, parallel_db, token_index)
    graded   = bool(payload.get("graded", False))
    grader   = _clean(payload.get("grader")) if graded else ""
    grade    = _clean(payload.get("grade"))  if graded else ""

    tier3 = f"{player}|{card_set}|{card_num}"
    tier2 = f"{tier3}|{parallel}"           if parallel else tier3
    tier1 = f"{tier2}|{grader}|{grade}"     if (grader and grade) else tier2

    return {"tier3": tier3, "tier2": tier2, "tier1": tier1}


# ── Qdrant client ────────────────────────────────────────────────────────────────

def get_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    host = os.environ["QDRANT_HOST"]
    port = int(os.environ.get("QDRANT_HTTP_PORT", 6333))
    key  = os.environ["QDRANT_API_KEY"]
    col  = os.environ.get("QDRANT_COLLECTION", "cards")
    prefer_grpc = os.environ.get("QDRANT_USE_GRPC", "false").lower() == "true"

    client = QdrantClient(
        host=host, port=port, api_key=key,
        https=False, prefer_grpc=prefer_grpc, timeout=30,
    )
    return client, col


# ── Pass 1 checkpoint ────────────────────────────────────────────────────────────

def _checkpoint_path(data_dir: Path) -> Path:
    return data_dir / "metric_dataset_pass1.json"


def _save_pass1_checkpoint(
    data_dir:     Path,
    selected_ids: list[str],
    id_to_meta:   dict[str, dict],
) -> None:
    """Persist Pass 1 results so Pass 2 can resume after a connection failure."""
    cp = _checkpoint_path(data_dir)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump({"selected_ids": selected_ids, "id_to_meta": id_to_meta}, f)
    size_mb = cp.stat().st_size / 1e6
    logger.info("Pass 1 checkpoint saved → {} ({:.1f} MB)", cp.name, size_mb)


def _load_pass1_checkpoint(data_dir: Path) -> tuple[list[str], dict[str, dict]] | None:
    """Load a saved Pass 1 checkpoint if it exists. Returns None if absent."""
    cp = _checkpoint_path(data_dir)
    if not cp.exists():
        return None
    logger.info("Loading Pass 1 checkpoint from {} ...", cp.name)
    with open(cp, encoding="utf-8") as f:
        d = json.load(f)
    selected_ids = d["selected_ids"]
    id_to_meta   = d["id_to_meta"]
    logger.info("  {:,} selected IDs, {:,} meta entries loaded", len(selected_ids), len(id_to_meta))
    return selected_ids, id_to_meta


# ── Main extraction ───────────────────────────────────────────────────────────────

def extract(
    sample_size:  int,
    min_positives: int,
    max_per_key:  int,
    resume:       bool = False,
) -> Path:
    parallel_db, token_index = _load_parallel_db()

    t0 = time.perf_counter()

    # ── Pass 1: scroll for payload only (or load from checkpoint) ────────────
    if resume:
        cp = _load_pass1_checkpoint(DATA_DIR)
        if cp is None:
            logger.error("--resume specified but no checkpoint found at {}",
                         _checkpoint_path(DATA_DIR))
            sys.exit(1)
        selected_ids, id_to_meta = cp
    else:
        client, col = get_qdrant()
        logger.info("Connected to Qdrant — collection '{}'", col)
        logger.info("Pass 1: scrolling {:,} points for payload + identity labels ...", sample_size)

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        base_filter = Filter(must=[
            FieldCondition(key="has_image",         match=MatchValue(value=True)),
            FieldCondition(key="specifics_source",  match=MatchValue(value="ebay")),
        ])

        key_to_ids: dict[str, list[str]] = defaultdict(list)
        id_to_meta: dict[str, dict]      = {}

        scrolled = 0
        offset   = None
        page_sz  = 2000

        log_interval = 100_000 if sample_size >= 1_000_000 else 10_000

        while scrolled < sample_size:
            batch_limit = min(page_sz, sample_size - scrolled)
            results, offset = client.scroll(
                collection_name=col,
                scroll_filter=base_filter,
                limit=batch_limit,
                offset=offset,
                with_payload=PAYLOAD_FIELDS,
                with_vectors=False,
            )
            if not results:
                logger.info("Scroll exhausted after {:,} points", scrolled)
                break

            for pt in results:
                p = pt.payload or {}
                keys = make_identity_keys(p, parallel_db, token_index)
                if not keys:
                    continue
                sid = str(pt.id)
                key_to_ids[keys["tier3"]].append(sid)
                id_to_meta[sid] = {
                    "id":       sid,
                    "tier3":    keys["tier3"],
                    "tier2":    keys["tier2"],
                    "tier1":    keys["tier1"],
                    "player":   _clean(p.get("player")),
                    "set":      _clean(p.get("set")),
                    "card_num": _clean(p.get("card_number")),
                    "parallel": _clean(p.get("parallel")),
                    "graded":   bool(p.get("graded", False)),
                    "grader":   _clean(p.get("grader")),
                    "grade":    _clean(p.get("grade")),
                    "genre":    _clean(p.get("genre")),
                    "brand":    _clean(p.get("brand")),
                }

            scrolled += len(results)
            if scrolled % log_interval == 0:
                elapsed = time.perf_counter() - t0
                rate = scrolled / elapsed if elapsed > 0 else 0
                logger.info(
                    "  scrolled {:>9,}/{:,} ({:.0f}/s) | labelled {:>8,} | unique keys {:>7,} | {:.1f}s",
                    scrolled, sample_size, rate, len(id_to_meta), len(key_to_ids), elapsed,
                )

            if offset is None:
                break

        logger.info(
            "Pass 1 done — {:,} labelled points, {:,} unique Tier-3 identity keys",
            len(id_to_meta), len(key_to_ids),
        )

        # ── Filter: keep only keys with enough positives ──────────────────────
        selected_ids: list[str] = []
        for key, ids in key_to_ids.items():
            if len(ids) < min_positives:
                continue
            selected_ids.extend(ids[:max_per_key])

        logger.info(
            "After filtering (min_positives={}, max_per_key={}): {:,} points selected",
            min_positives, max_per_key, len(selected_ids),
        )

        if not selected_ids:
            logger.error("No points passed filters — try a larger sample or lower min_positives")
            sys.exit(1)

        # Save checkpoint so Pass 2 can resume if the connection drops
        _save_pass1_checkpoint(DATA_DIR, selected_ids, id_to_meta)

    # ── Pass 2: fetch image vectors (fresh connection to avoid timeout) ───────
    logger.info("Pass 2: fetching image vectors for {:,} selected points ...", len(selected_ids))

    # Always create a fresh client for Pass 2 — the Pass 1 connection may have
    # timed out after a long scroll (gRPC DEADLINE_EXCEEDED on reuse).
    client2, col = get_qdrant()

    CHUNK       = 512     # points per Qdrant retrieve call
    WRITE_CHUNK = 50_000  # rows per parquet flush — keeps peak RAM bounded (~200 MB)

    log_every_chunks = 100 if len(selected_ids) >= 500_000 else 10
    MAX_RETRIES = 3

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "metric_dataset.parquet"

    # Define schema once so all row-groups are consistent
    schema = pa.schema([
        pa.field("id",        pa.string()),
        pa.field("tier3",     pa.string()),
        pa.field("tier2",     pa.string()),
        pa.field("tier1",     pa.string()),
        pa.field("player",    pa.string()),
        pa.field("card_set",  pa.string()),
        pa.field("card_num",  pa.string()),
        pa.field("parallel",  pa.string()),
        pa.field("graded",    pa.bool_()),
        pa.field("grader",    pa.string()),
        pa.field("grade",     pa.string()),
        pa.field("genre",     pa.string()),
        pa.field("brand",     pa.string()),
        pa.field("image_vec", pa.list_(pa.float32())),
    ])

    def _flush(writer: pq.ParquetWriter, vecs: list, metas: list) -> int:
        """Write one row-group and return number of rows written."""
        table = pa.table({
            "id":        pa.array([m["id"]       for m in metas], type=pa.string()),
            "tier3":     pa.array([m["tier3"]    for m in metas], type=pa.string()),
            "tier2":     pa.array([m["tier2"]    for m in metas], type=pa.string()),
            "tier1":     pa.array([m["tier1"]    for m in metas], type=pa.string()),
            "player":    pa.array([m["player"]   for m in metas], type=pa.string()),
            "card_set":  pa.array([m["set"]      for m in metas], type=pa.string()),
            "card_num":  pa.array([m["card_num"] for m in metas], type=pa.string()),
            "parallel":  pa.array([m["parallel"] for m in metas], type=pa.string()),
            "graded":    pa.array([m["graded"]   for m in metas], type=pa.bool_()),
            "grader":    pa.array([m["grader"]   for m in metas], type=pa.string()),
            "grade":     pa.array([m["grade"]    for m in metas], type=pa.string()),
            "genre":     pa.array([m["genre"]    for m in metas], type=pa.string()),
            "brand":     pa.array([m["brand"]    for m in metas], type=pa.string()),
            "image_vec": pa.array(vecs,          type=pa.list_(pa.float32())),
        }, schema=schema)
        writer.write_table(table)
        return len(vecs)

    total_written  = 0
    chunk_vectors: list[list[float]] = []
    chunk_meta:    list[dict]        = []

    with pq.ParquetWriter(str(out_path), schema, compression="snappy") as writer:
        for i in range(0, len(selected_ids), CHUNK):
            chunk_ids = selected_ids[i:i + CHUNK]
            int_ids   = [int(sid) if sid.isdigit() else sid for sid in chunk_ids]

            # Retry with reconnect on timeout / connection error
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    points = client2.retrieve(
                        collection_name=col,
                        ids=int_ids,
                        with_vectors=["image"],
                        with_payload=False,
                    )
                    break
                except Exception as exc:
                    if attempt >= MAX_RETRIES:
                        logger.error("Chunk at offset {:,} failed after {} retries: {}", i, MAX_RETRIES, exc)
                        raise
                    logger.warning("Chunk timeout (attempt {}/{}), reconnecting...", attempt, MAX_RETRIES)
                    time.sleep(2)
                    client2, col = get_qdrant()  # fresh connection

            for pt in points:
                if pt.vector and "image" in pt.vector:
                    sid = str(pt.id)
                    if sid in id_to_meta:
                        chunk_vectors.append(pt.vector["image"])
                        chunk_meta.append(id_to_meta[sid])

            # Flush to disk every WRITE_CHUNK rows — frees RAM immediately
            if len(chunk_vectors) >= WRITE_CHUNK:
                n_flushed = _flush(writer, chunk_vectors, chunk_meta)
                total_written += n_flushed
                logger.info("  flushed {:,} rows → parquet (total written: {:,})",
                            n_flushed, total_written)
                chunk_vectors.clear()
                chunk_meta.clear()

            chunk_num = i // CHUNK
            if chunk_num % log_every_chunks == 0:
                pct = 100 * (i + len(chunk_ids)) / len(selected_ids)
                elapsed2 = time.perf_counter() - t0
                logger.info("  fetched {:>8,}/{:,} ({:.1f}%) | {:.1f}s",
                            i + len(chunk_ids), len(selected_ids), pct, elapsed2)

        # Flush any remaining rows
        if chunk_vectors:
            total_written += _flush(writer, chunk_vectors, chunk_meta)
            chunk_vectors.clear()
            chunk_meta.clear()

    n = total_written
    logger.info("Pass 2 done — {:,} vectors written", n)

    elapsed = time.perf_counter() - t0
    logger.info("─" * 60)
    logger.info("Dataset written → {} ({:,} points, {:.1f}s)", out_path.name, n, elapsed)
    logger.info("  File size: {:.1f} MB", out_path.stat().st_size / 1e6)
    logger.info("─" * 60)

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract metric head training dataset from Qdrant")
    ap.add_argument("--sample",        type=int, default=DEFAULT_SAMPLE,
                    help=f"Points to scroll from Qdrant (default {DEFAULT_SAMPLE:,}). "
                         f"Use ~7M to yield ~1M selected training points.")
    ap.add_argument("--min-positives", type=int, default=DEFAULT_MIN_POSITIVES,
                    help=f"Min instances per identity key (default {DEFAULT_MIN_POSITIVES})")
    ap.add_argument("--max-per-key",   type=int, default=DEFAULT_MAX_PER_KEY,
                    help=f"Max instances per identity key (default {DEFAULT_MAX_PER_KEY})")
    ap.add_argument("--resume",        action="store_true",
                    help="Skip Pass 1 and load from data/metric_dataset_pass1.json checkpoint. "
                         "Use after a Pass 2 failure to avoid re-scrolling.")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    extract(args.sample, args.min_positives, args.max_per_key, resume=args.resume)


if __name__ == "__main__":
    main()
