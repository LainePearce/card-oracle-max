#!/usr/bin/env python3
"""
Repopulate the Qdrant `cards` collection from the S3 vector store.

Why this rewrite (May 2026):
  Three prior repopulate attempts each landed only ~4% of expected image
  vectors (~1.9M of ~51M). Audit confirmed the cluster was healthy, the
  vector dimensions matched (768-d), and isolated `wait=True` upserts
  succeeded cleanly. The actual causes were:

  1. The script used `client.upsert(payload={"os_id": ...})` which silently
     REPLACES the existing payload on a point — it does not merge. Every
     re-load wiped has_image, source, type, brand, player, … on the points
     it touched, breaking production search filters (qdrant_search.py
     Arm 1 hard-filters on has_image=True).

  2. `wait=False` was used at batch level. Server-side validation errors,
     queue-overflow rejections, and per-point write failures were silently
     swallowed by the client.

  3. No checkpointing. Runs that died (OOM, killed terminal, spot
     interruption) restarted at shard 0 and died at roughly the same place
     each time, so progress was capped at ~5% of shards regardless of how
     many attempts were made.

What this version does differently:
  - `update_vectors()` instead of `upsert()`: touches only the named vector
    slot, leaves payload untouched. Requires the point to already exist —
    fine for the recovery case (audit showed 0% missing point IDs).
  - `wait=True` per batch: validation / queue / write errors surface
    immediately as Python exceptions. Throughput drops to ~10-20k pts/s
    instead of ~50k pts/s — acceptable for a one-time recovery.
  - Per-shard checkpoint to S3 (via S3VectorStore.save_checkpoint).
    Restartable; an OOM mid-run means re-running picks up where we left off.
  - After update_vectors, set_payload (MERGE semantics) restores
    has_image=True on each newly-vectorized point so the search Arm 1
    filter returns them. This fixes payload as a side effect.
  - Streaming: one shard's parquet in memory at a time. No giant Python
    sets like the earlier diag scripts.

Run image and specifics separately:
    python tools/repopulate_qdrant.py --vector-type image
    python tools/repopulate_qdrant.py --vector-type specifics

Run inside tmux/screen so a closed SSH session doesn't kill it:
    tmux new -s repop
    python tools/repopulate_qdrant.py --vector-type image 2>&1 | tee /tmp/repop_image.log
    # ctrl-b d to detach; `tmux attach -t repop` to reattach
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qdrant_client.models import PointVectors

from src.embeddings.vector_store import S3VectorStore
from src.ingestion.qdrant_writer import get_qdrant_client, COLLECTION_NAME


# vector_type → (model_id, params_hash) — must match the worker's
# IMAGE_MODEL_ID/IMAGE_PARAMS and TEXT_MODEL_ID/TEXT_PARAMS constants.
DEFAULTS = {
    "image":     ("clip-vit-l-14", "v2-fp16-224px-sqpad"),
    "specifics": ("minilm-l6-v2",  "v1-mean-256tok"),
}

# Bumped after the upsert→update_vectors fix; older checkpoint files use
# the prior buggy code path and must not be reused.
CHECKPOINT_JOB_PREFIX = "repopulate_v2"

# Payload fields that are definitionally true for points reloaded from
# a given vector_type's S3 shards. Merged in via set_payload after
# update_vectors so existing rich payload (player/brand/set/etc.) is
# preserved untouched.
PAYLOAD_FLAG_BY_TYPE = {
    "image":     {"has_image": True},
    "specifics": {},   # specifics_source already carries this signal; no flag needed
}


def _to_qdrant_id(raw):
    """qdrant_ids are stored as strings in parquet; numeric ones must be int for Qdrant."""
    try:
        return int(raw)
    except (ValueError, TypeError):
        return raw   # UUID5 fallback path — use as-is


def main() -> None:
    p = argparse.ArgumentParser(description="Repopulate Qdrant from S3 vectors.")
    p.add_argument("--vector-type", choices=["image", "specifics"], required=True)
    p.add_argument("--model",       default=None,
                   help="Override the model_id; defaults are vector-type aware.")
    p.add_argument("--params",      default=None,
                   help="Override the params_hash; defaults are vector-type aware.")
    p.add_argument("--collection",  default=COLLECTION_NAME)
    p.add_argument("--batch-size",  type=int, default=500,
                   help="Points per Qdrant update_vectors call (default 500). "
                        "Smaller is more failure-localised under wait=True.")
    p.add_argument("--progress-every", type=int, default=30,
                   help="Seconds between progress lines (default 30).")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Save checkpoint every N shards (default 25).")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore the existing checkpoint and start from shard 0.")
    p.add_argument("--no-payload-flag", action="store_true",
                   help="Skip the set_payload(has_image=True) merge — vectors only.")
    args = p.parse_args()

    default_model, default_params = DEFAULTS[args.vector_type]
    if args.model  is None: args.model  = default_model
    if args.params is None: args.params = default_params

    job_id = f"{CHECKPOINT_JOB_PREFIX}_{args.vector_type}"
    payload_flag = PAYLOAD_FLAG_BY_TYPE[args.vector_type]

    store  = S3VectorStore(
        bucket=os.environ["S3_VECTOR_BUCKET"],
        prefix=os.environ.get("S3_VECTOR_PREFIX", "vectors"),
    )
    client = get_qdrant_client()

    print(f"[repopulate] {args.vector_type} → Qdrant '{args.collection}' "
          f"(model={args.model}, params={args.params})", flush=True)

    # ── Resume from checkpoint ────────────────────────────────────────────
    ckpt = None if args.fresh else store.load_checkpoint(job_id)
    completed: set[str] = set(ckpt.get("completed", [])) if ckpt else set()
    total_already       = ckpt.get("vectors_loaded", 0) if ckpt else 0

    if args.fresh:
        print(f"[repopulate] --fresh: ignoring any existing checkpoint", flush=True)
    elif ckpt:
        print(f"[repopulate] resuming: {len(completed):,} shards already done, "
              f"{total_already:,} vectors loaded in prior sessions", flush=True)
    else:
        print(f"[repopulate] no checkpoint — starting from scratch", flush=True)

    # ── Enumerate shards ──────────────────────────────────────────────────
    keys = store.list_shards(args.vector_type, args.model, args.params)
    todo = [k for k in keys if k not in completed]
    print(f"[repopulate] {len(keys):,} total shards, "
          f"{len(todo):,} to process this session, "
          f"{len(completed):,} already done", flush=True)

    if not todo:
        print(f"[repopulate] nothing to do — checkpoint says all shards complete", flush=True)
        return

    # ── Process ───────────────────────────────────────────────────────────
    t0          = time.time()
    last_print  = t0
    total       = total_already
    errors:     list[tuple[str, str]] = []   # (shard_key, error_repr)

    for shard_idx, key in enumerate(todo):
        try:
            table  = store.read_shard(key)
            n_rows = len(table)

            for start in range(0, n_rows, args.batch_size):
                batch = table.slice(start, args.batch_size)
                vecs: list[PointVectors] = []
                ids:  list[int | str]    = []
                for i in range(len(batch)):
                    pid = _to_qdrant_id(batch["qdrant_id"][i].as_py())
                    ids.append(pid)
                    vecs.append(PointVectors(
                        id=pid,
                        vector={args.vector_type: batch["vector"][i].as_py()},
                    ))

                # Step 1: set vector slot. Does NOT touch payload.
                client.update_vectors(
                    collection_name=args.collection,
                    points=vecs,
                    wait=True,
                )

                # Step 2: merge payload flag (e.g. has_image=True) so the
                # search-path filter actually returns these points.
                # set_payload has merge semantics — existing keys preserved.
                if payload_flag and not args.no_payload_flag:
                    client.set_payload(
                        collection_name=args.collection,
                        payload=payload_flag,
                        points=ids,
                        wait=True,
                    )

                total += len(vecs)

            completed.add(key)

            # Checkpoint cadence: every N shards + on final shard.
            if (shard_idx + 1) % args.checkpoint_every == 0 \
                    or (shard_idx + 1) == len(todo):
                store.save_checkpoint(job_id, {
                    "completed":      sorted(completed),
                    "vectors_loaded": total,
                    "last_shard":     key,
                    "errors":         errors[-50:],  # cap to avoid unbounded growth
                    "updated_at":     time.time(),
                })

            now = time.time()
            if now - last_print >= args.progress_every:
                elapsed = now - t0
                rate = (total - total_already) / elapsed if elapsed > 0 else 0
                pct = 100 * (shard_idx + 1) / len(todo)
                print(f"[repopulate] shard {shard_idx+1:,}/{len(todo):,} ({pct:.1f}%)  "
                      f"loaded={total:,}  rate={rate:,.0f}/s  "
                      f"elapsed={elapsed:.0f}s  "
                      f"errors={len(errors)}", flush=True)
                last_print = now

        except Exception as e:
            # Don't lose progress on one bad shard — log and continue.
            # The shard is NOT added to `completed`, so a re-run retries it.
            err_repr = f"{type(e).__name__}: {e}"
            errors.append((key, err_repr))
            print(f"[repopulate] ERROR on shard {key}: {err_repr}", flush=True)

    # Final checkpoint
    store.save_checkpoint(job_id, {
        "completed":      sorted(completed),
        "vectors_loaded": total,
        "last_shard":     None,
        "errors":         errors[-50:],
        "updated_at":     time.time(),
        "finished_at":    time.time(),
    })

    elapsed = time.time() - t0
    rate = (total - total_already) / elapsed if elapsed > 0 else 0
    print(f"\n[repopulate] DONE: {total:,} total vectors loaded "
          f"({len(completed):,}/{len(keys):,} shards complete) "
          f"in {elapsed:.1f}s session ({rate:,.0f}/s)", flush=True)
    if errors:
        print(f"[repopulate] {len(errors)} shard(s) failed — re-run to retry; "
              f"first 10 errors:", flush=True)
        for k, err in errors[:10]:
            print(f"    {k}: {err}", flush=True)


if __name__ == "__main__":
    main()
