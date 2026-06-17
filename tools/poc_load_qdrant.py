#!/usr/bin/env python3
"""
POC component 3 — create cards_backbone_poc and load the three vector sets.

Creates an isolated Qdrant collection with three named image vectors
(image 768-d CLIP, image_dinov2 1024-d, image_dinov3 1024-d) and loads it from
the S3 vector store written by poc_triple_embed.py.

Load is streamed per named vector so it scales past the pilot without holding
everything in RAM:
  Pass A  upsert points with payload + the CLIP `image` vector
  Pass B  update_vectors `image_dinov2`
  Pass C  update_vectors `image_dinov3`

Payload (identity fields + the S3 image keys for UI rendering) comes from the
archive manifest. Point IDs are the production uint64 os_id mapping.

Target the local single-node Qdrant (QDRANT_HOST=localhost with the dev
docker-compose) for "locally hosted" testing, or the cluster.

    python tools/poc_load_qdrant.py --manifest data/poc/manifest_2026-06-01.jsonl \
        --dates 2026-06-01
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
from qdrant_client.models import (
    Distance, HnswConfigDiff, OptimizersConfigDiff, PointStruct, PointVectors,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType, VectorParams,
)

from src.embeddings.vector_store import S3VectorStore
from src.ingestion.qdrant_writer import get_qdrant_client, extract_payload
from tools.poc_common import POC_COLLECTION, POC_ENCODERS


def create_collection(client) -> None:
    if any(c.name == POC_COLLECTION for c in client.get_collections().collections):
        logger.info("Collection '{}' already exists — leaving as-is", POC_COLLECTION)
        return
    vectors = {
        s.vector_name: VectorParams(
            size=s.dim, distance=Distance.COSINE, on_disk=True,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200, on_disk=True),
        )
        for s in POC_ENCODERS
    }
    client.create_collection(
        collection_name=POC_COLLECTION,
        vectors_config=vectors,
        quantization_config=ScalarQuantization(scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8, quantile=0.99, always_ram=True)),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000,
                                               memmap_threshold=20_000),
        on_disk_payload=True,
        shard_number=1,
        replication_factor=1,
    )
    logger.info("Created '{}' with named vectors: {}",
                POC_COLLECTION, [s.vector_name for s in POC_ENCODERS])


def build_payloads(manifests: list[str]) -> dict[str, dict]:
    """os_id -> Qdrant payload (identity fields + S3 image keys + title)."""
    out: dict[str, dict] = {}
    for mpath in manifests:
        for line in open(ROOT / mpath):
            r = json.loads(line)
            src = r["source_doc"]
            payload = extract_payload(src, doc_id=r["os_id"])
            payload["title"]    = str(src.get("title", ""))
            payload["img_orig"] = r["s3_keys"].get("original", "")
            payload["img_512"]  = r["s3_keys"].get("512", "")
            payload["img_256"]  = r["s3_keys"].get("256", "")
            out[r["os_id"]] = payload
    return out


def _spec(vector_name: str):
    return next(s for s in POC_ENCODERS if s.vector_name == vector_name)


def load_clip_points(client, store, dates, payloads, batch_size) -> int:
    spec = _spec("image")
    total = 0
    points: list[PointStruct] = []
    for d in dates:
        for table in store.iter_vectors("image", spec.model_id, spec.params,
                                        index_type="ebay-dated", partition=d,
                                        columns=["os_id", "qdrant_id", "vector"]):
            for i in range(len(table)):
                os_id = table["os_id"][i].as_py()
                pl = payloads.get(os_id)
                if pl is None:
                    continue
                points.append(PointStruct(
                    id=int(table["qdrant_id"][i].as_py()),
                    vector={"image": table["vector"][i].as_py()},
                    payload=pl,
                ))
                if len(points) >= batch_size:
                    client.upsert(collection_name=POC_COLLECTION, points=points, wait=True)
                    total += len(points)
                    points = []
    if points:
        client.upsert(collection_name=POC_COLLECTION, points=points, wait=True)
        total += len(points)
    logger.info("Pass A (image/CLIP): upserted {} points", total)
    return total


def load_extra_vectors(client, store, vector_name, dates, batch_size) -> int:
    spec = _spec(vector_name)
    total = 0
    pv: list[PointVectors] = []
    for d in dates:
        for table in store.iter_vectors("image", spec.model_id, spec.params,
                                        index_type="ebay-dated", partition=d,
                                        columns=["qdrant_id", "vector"]):
            for i in range(len(table)):
                pv.append(PointVectors(
                    id=int(table["qdrant_id"][i].as_py()),
                    vector={vector_name: table["vector"][i].as_py()},
                ))
                if len(pv) >= batch_size:
                    client.update_vectors(collection_name=POC_COLLECTION, points=pv, wait=True)
                    total += len(pv)
                    pv = []
    if pv:
        client.update_vectors(collection_name=POC_COLLECTION, points=pv, wait=True)
        total += len(pv)
    logger.info("Pass ({}): updated {} vectors", vector_name, total)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--dates", nargs="+", required=True, help="Partitions (YYYY-MM-DD).")
    ap.add_argument("--vector-bucket", default=os.environ.get("S3_VECTOR_BUCKET"))
    ap.add_argument("--vector-prefix", default=os.environ.get("S3_VECTOR_PREFIX", "vectors"))
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    client = get_qdrant_client()
    store  = S3VectorStore(bucket=args.vector_bucket, prefix=args.vector_prefix)

    create_collection(client)
    logger.info("Loading payloads from {} manifest(s) …", len(args.manifest))
    payloads = build_payloads(args.manifest)
    logger.info("  {} payloads", len(payloads))

    load_clip_points(client, store, args.dates, payloads, args.batch_size)
    load_extra_vectors(client, store, "image_dinov2", args.dates, args.batch_size)
    load_extra_vectors(client, store, "image_dinov3", args.dates, args.batch_size)

    info = client.get_collection(POC_COLLECTION)
    logger.info("─" * 60)
    logger.info("'{}' loaded — {} points", POC_COLLECTION, info.points_count)


if __name__ == "__main__":
    main()
