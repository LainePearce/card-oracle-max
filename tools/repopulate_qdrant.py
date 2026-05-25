#!/usr/bin/env python3
"""
Repopulate the Qdrant `cards` collection from the S3 vector store.

Uses the worker's proven gRPC connection (get_qdrant_client → QDRANT_HOST/PORT
from .env). Prints periodic progress so multi-hour runs are observable.

Reads both old shallow-path shards (legacy: …/{partition}/{NNNN}.parquet) and
new job-unique-path shards (…/{partition}/{job_id}/{NNNN}.parquet) — iter_vectors
lists by prefix recursively, so both formats are picked up transparently.

Run image and specifics separately; each upserts only its own named vector
and leaves the other (if any) untouched on the point.

Usage:
    python tools/repopulate_qdrant.py --vector-type image
    python tools/repopulate_qdrant.py --vector-type specifics
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

from qdrant_client.models import PointStruct

from src.embeddings.vector_store import S3VectorStore
from src.ingestion.qdrant_writer import get_qdrant_client, COLLECTION_NAME


DEFAULTS = {
    # vector_type → (model_id, params_hash) — must match the worker's
    # IMAGE_MODEL_ID/IMAGE_PARAMS and TEXT_MODEL_ID/TEXT_PARAMS constants.
    "image":     ("clip-vit-l-14", "v2-fp16-224px-sqpad"),
    "specifics": ("minilm-l6-v2",  "v1-mean-256tok"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Repopulate Qdrant from S3 vectors.")
    p.add_argument("--vector-type", choices=["image", "specifics"], required=True)
    p.add_argument("--model",       default=None,
                   help="Override the model_id; defaults are vector-type aware.")
    p.add_argument("--params",      default=None,
                   help="Override the params_hash; defaults are vector-type aware.")
    p.add_argument("--collection",  default=COLLECTION_NAME)
    p.add_argument("--batch-size",  type=int, default=1000,
                   help="Points per Qdrant upsert call.")
    p.add_argument("--progress-every", type=int, default=30,
                   help="Seconds between progress lines (default 30).")
    args = p.parse_args()

    default_model, default_params = DEFAULTS[args.vector_type]
    if args.model  is None: args.model  = default_model
    if args.params is None: args.params = default_params

    store = S3VectorStore(
        bucket=os.environ["S3_VECTOR_BUCKET"],
        prefix=os.environ.get("S3_VECTOR_PREFIX", "vectors"),
    )
    client = get_qdrant_client()

    print(f"[repopulate] {args.vector_type} → Qdrant '{args.collection}' "
          f"(model={args.model}, params={args.params})", flush=True)

    t0 = time.time()
    last = t0
    total = 0

    for table in store.iter_vectors(
        args.vector_type, args.model, args.params,
        columns=["os_id", "qdrant_id", "vector"],
    ):
        for start in range(0, len(table), args.batch_size):
            batch = table.slice(start, args.batch_size)
            points = []
            for i in range(len(batch)):
                raw_id = batch["qdrant_id"][i].as_py()
                # qdrant_ids stored as strings; numeric ones must be int for Qdrant
                try:
                    pid = int(raw_id)
                except (ValueError, TypeError):
                    pid = raw_id   # UUID5 fallback path — use as-is
                points.append(PointStruct(
                    id=pid,
                    vector={args.vector_type: batch["vector"][i].as_py()},
                    payload={"os_id": batch["os_id"][i].as_py()},
                ))
            client.upsert(
                collection_name=args.collection,
                points=points,
                wait=False,
            )
            total += len(points)

        now = time.time()
        if now - last >= args.progress_every:
            elapsed = now - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f"[repopulate] {total:,} points loaded — "
                  f"{elapsed:.0f}s elapsed, {rate:,.0f}/s", flush=True)
            last = now

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"[repopulate] DONE: {total:,} points loaded in {elapsed:.1f}s "
          f"({rate:,.0f}/s)", flush=True)


if __name__ == "__main__":
    main()
