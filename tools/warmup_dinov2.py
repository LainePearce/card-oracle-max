#!/usr/bin/env python3
"""
Warm the cards_dinov2 collection's disk-backed data into OS page cache.

With always_ram=False, DINOv2's INT8-quantized vectors and on-disk HNSW links
are read from NVMe on demand — a cold collection means slow first queries.
This fires batches of random-unit-vector ANN searches (each query fans out to
every shard on every node), walking the HNSW graph broadly so the hot
structures get paged in cluster-wide. Watch the per-batch p50 fall and level
off — flat p50 = warm.

Read-only and safe: page cache is evictable; this cannot destabilize a node.

Run from worker-0 (needs QDRANT_* in .env). Uses the legacy REST
/points/search endpoint (server 1.8.2; the 1.18 client's .search is gone).

    python tools/warmup_dinov2.py                       # 2000 queries, conc 8
    python tools/warmup_dinov2.py --queries 5000 --concurrency 12
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import requests
from loguru import logger

QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = int(os.environ.get("QDRANT_HTTP_PORT", os.environ.get("QDRANT_PORT", 6333)))
API_KEY     = os.environ.get("QDRANT_API_KEY", "")
COLLECTION  = os.environ.get("QDRANT_DINOV2_COLLECTION", "cards_dinov2")
VEC_NAME    = "image_dinov2"
DIM         = 1024
URL         = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION}/points/search"
HEADERS     = {"Content-Type": "application/json", **({"api-key": API_KEY} if API_KEY else {})}


def random_unit_vector() -> list[float]:
    v = [random.gauss(0, 1) for _ in range(DIM)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def one_query(hnsw_ef: int, top_k: int) -> float | None:
    body = {
        "vector": {"name": VEC_NAME, "vector": random_unit_vector()},
        "limit": top_k,
        "with_payload": False,
        "params": {"hnsw_ef": hnsw_ef, "quantization": {"rescore": True}},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(URL, json=body, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return (time.perf_counter() - t0) * 1000
    except Exception as e:
        logger.warning("query failed: {}", e)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--batch", type=int, default=100, help="Queries per progress report.")
    ap.add_argument("--hnsw-ef", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    logger.info("Warming '{}' via {} — {} queries, concurrency {}, ef {}",
                COLLECTION, QDRANT_HOST, args.queries, args.concurrency, args.hnsw_ef)

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        while done < args.queries:
            n = min(args.batch, args.queries - done)
            lat = [l for l in ex.map(lambda _: one_query(args.hnsw_ef, args.top_k),
                                     range(n)) if l is not None]
            done += n
            if lat:
                lat.sort()
                p50 = statistics.median(lat)
                p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
                logger.info("{}/{}  p50 {:.0f}ms  p95 {:.0f}ms  ok {}/{}",
                            done, args.queries, p50, p95, len(lat), n)
            else:
                logger.error("{}/{}  batch fully failed", done, args.queries)

    logger.info("Warm-up complete — flat p50 across the last few batches means warm.")


if __name__ == "__main__":
    main()
