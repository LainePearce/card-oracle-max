#!/usr/bin/env python3
"""
2b — Restore search-filter payload on Population B from OpenSearch.

Population B is the ~51M image-bearing points whose payload is currently
{os_id, has_image} (the patched repopulate's stub form). They have image
vectors and the has_image flag, but lack the other filter fields the search
path uses (source, type, brand, player, set, card_number, year, team, genre,
specifics_source). Until restored, qdrant_search.py Arms 2 and 3 silently
exclude them from any filtered query.

Strategy:
  1. Iterate S3 image shards (skipping any in checkpoint).
  2. For each shard, read (os_id, index_name, qdrant_id).
  3. Group by index_name, then OS _mget in chunks of 500 ids per index.
  4. For each found doc, run extract_payload to derive the full payload,
     then keep only the search-filter subset (so we don't overwrite the
     OS-scroll-original Population A rich payload with a partial copy on
     any overlap).
  5. Use the REST batch endpoint to issue multiple set_payload operations
     in one call — per-point payload, ~500 ops/call. Raw requests to avoid
     any qdrant-client wire-format surprises.
  6. Checkpoint per shard so the run is restartable.

Run inside tmux. Expected wall-clock ~3-4h, dominated by OS _mget latency.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.embeddings.vector_store import S3VectorStore
from src.ingestion.opensearch_reader import get_opensearch_client
from src.ingestion.qdrant_writer       import extract_payload, COLLECTION_NAME


# Search-filter subset — what qdrant_search.py + fusion.py actually filter on.
# Keep this narrow; do NOT include sale_type/current_price (display fields)
# or specifics_confidence (we don't have it for OS-scroll docs).
SEARCH_FILTER_KEYS = (
    "source", "type",
    "brand", "player", "set", "card_number", "year", "team", "genre",
    "specifics_source",
)

CHECKPOINT_JOB = "restore_search_payload_image_v1"
S3_VECTOR_TYPE = "image"
S3_MODEL       = "clip-vit-l-14"
S3_PARAMS      = "v2-fp16-224px-sqpad"

OS_MGET_CHUNK     = 500    # ids per OS _mget call
QDRANT_BATCH_OPS  = 500    # set_payload ops per Qdrant batch call


def _to_qdrant_id(raw):
    try:
        return int(raw)
    except (ValueError, TypeError):
        return raw


def filtered_payload(doc_source: dict, os_id: str) -> dict:
    """Run extract_payload, keep only search-filter keys with non-empty values."""
    full = extract_payload(doc_source, doc_id=os_id)
    out  = {}
    for k in SEARCH_FILTER_KEYS:
        v = full.get(k)
        # Skip empty strings, None, empty lists — set_payload merges, so
        # leaving them out preserves anything already set on the point.
        if v is None or v == "" or v == []:
            continue
        out[k] = v
    return out


def qdrant_batch_set_payload(qdrant_url: str, headers: dict, collection: str,
                             ops_payloads: list[tuple[int | str, dict]]) -> int:
    """
    POST /collections/{name}/points/batch with N set_payload operations,
    one per point. Returns number of ops submitted. Raises on non-200.
    """
    if not ops_payloads:
        return 0
    body = {
        "operations": [
            {"set_payload": {"payload": pl, "points": [pid]}}
            for pid, pl in ops_payloads
        ]
    }
    url = f"{qdrant_url}/collections/{collection}/points/batch?wait=true"
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Qdrant batch failed {resp.status_code}: {resp.text[:300]}")
    j = resp.json()
    if j.get("status") != "ok":
        raise RuntimeError(f"Qdrant batch status={j.get('status')}: {j}")
    return len(ops_payloads)


def main() -> None:
    store    = S3VectorStore(
        bucket = os.environ["S3_VECTOR_BUCKET"],
        prefix = os.environ.get("S3_VECTOR_PREFIX", "vectors"),
    )
    osclient = get_opensearch_client()

    qdrant_url = f"http://{os.environ['QDRANT_HOST']}:6333"
    qdrant_headers = {"content-type": "application/json"}
    if os.environ.get("QDRANT_API_KEY"):
        qdrant_headers["api-key"] = os.environ["QDRANT_API_KEY"]

    # ── Resume from checkpoint ────────────────────────────────────────────
    ckpt = store.load_checkpoint(CHECKPOINT_JOB)
    completed: set[str] = set(ckpt.get("completed", [])) if ckpt else set()
    total_restored      = ckpt.get("points_restored", 0) if ckpt else 0
    total_not_found     = ckpt.get("os_not_found",    0) if ckpt else 0

    if ckpt:
        print(f"[restore] resuming: {len(completed):,} shards done, "
              f"{total_restored:,} payloads restored, "
              f"{total_not_found:,} OS-not-found previously", flush=True)
    else:
        print("[restore] no checkpoint — starting fresh", flush=True)

    keys = store.list_shards(S3_VECTOR_TYPE, S3_MODEL, S3_PARAMS)
    todo = [k for k in keys if k not in completed]
    print(f"[restore] {len(keys):,} total shards, {len(todo):,} to process, "
          f"{len(completed):,} already done", flush=True)

    if not todo:
        print("[restore] nothing to do — checkpoint says all shards complete")
        return

    t0          = time.time()
    last_print  = t0
    not_found   = total_not_found

    # Accumulator across shards so we can flush a near-full batch
    pending_ops: list[tuple[int | str, dict]] = []

    def flush_pending():
        nonlocal pending_ops
        if not pending_ops:
            return 0
        n = qdrant_batch_set_payload(qdrant_url, qdrant_headers,
                                     COLLECTION_NAME, pending_ops)
        pending_ops = []
        return n

    for shard_idx, key in enumerate(todo):
        try:
            table = store.read_shard(key)

            # Group by index_name for efficient OS _mget batching
            by_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for i in range(len(table)):
                os_id = table["os_id"][i].as_py()
                idx   = table["index_name"][i].as_py()
                qid   = table["qdrant_id"][i].as_py()
                by_index[idx].append((os_id, qid))

            for index_name, items in by_index.items():
                for chunk_start in range(0, len(items), OS_MGET_CHUNK):
                    chunk = items[chunk_start:chunk_start + OS_MGET_CHUNK]
                    body = {"docs": [{"_index": index_name, "_id": os_id}
                                     for os_id, _ in chunk]}
                    resp = osclient.mget(body=body)

                    for doc, (os_id, qid_raw) in zip(resp["docs"], chunk):
                        if not doc.get("found"):
                            not_found += 1
                            continue
                        pl = filtered_payload(doc.get("_source", {}), os_id)
                        if not pl:
                            continue
                        pid = _to_qdrant_id(qid_raw)
                        pending_ops.append((pid, pl))

                        if len(pending_ops) >= QDRANT_BATCH_OPS:
                            total_restored += flush_pending()

            completed.add(key)

            # Checkpoint cadence: every 25 shards + final
            if (shard_idx + 1) % 25 == 0 or (shard_idx + 1) == len(todo):
                # Flush any pending ops before checkpointing so the count
                # in the checkpoint reflects what's actually persisted.
                total_restored += flush_pending()
                store.save_checkpoint(CHECKPOINT_JOB, {
                    "completed":       sorted(completed),
                    "points_restored": total_restored,
                    "os_not_found":    not_found,
                    "updated_at":      time.time(),
                })

            now = time.time()
            if now - last_print >= 30:
                elapsed = now - t0
                rate = (total_restored - (ckpt or {}).get("points_restored", 0)) \
                       / elapsed if elapsed > 0 else 0
                pct = 100 * (shard_idx + 1) / len(todo)
                print(f"[restore] shard {shard_idx+1:,}/{len(todo):,} ({pct:.1f}%)  "
                      f"restored={total_restored:,}  "
                      f"os_not_found={not_found:,}  "
                      f"rate={rate:,.0f}/s  elapsed={elapsed:.0f}s",
                      flush=True)
                last_print = now

        except Exception as e:
            print(f"[restore] ERROR on shard {key}: {type(e).__name__}: {e}",
                  flush=True)

    # Final flush + checkpoint
    total_restored += flush_pending()
    store.save_checkpoint(CHECKPOINT_JOB, {
        "completed":       sorted(completed),
        "points_restored": total_restored,
        "os_not_found":    not_found,
        "updated_at":      time.time(),
        "finished_at":     time.time(),
    })

    elapsed = time.time() - t0
    print()
    print(f"[restore] DONE in {elapsed:.0f}s")
    print(f"  shards complete:        {len(completed):,}/{len(keys):,}")
    print(f"  payloads restored:      {total_restored:,}")
    print(f"  os_id not found in OS:  {not_found:,}  "
          f"(expected for deleted listings)")


if __name__ == "__main__":
    main()
