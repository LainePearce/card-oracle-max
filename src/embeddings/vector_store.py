"""S3 vector persistence layer.

S3 is the PRIMARY durable store for all generated vectors.
Vectors are written to S3 BEFORE being loaded into Qdrant.
Qdrant is a queryable index built from S3, not the source of truth.

If Qdrant is lost or needs rebuilding, repopulate from S3 — no re-embedding needed.

See CLAUDE.md Section 16 for full specification.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

SHARD_SIZE  = 100_000   # target vectors per parquet file
COMPRESSION = "snappy"


@dataclass
class VectorRecord:
    os_id:         str
    qdrant_id:     str
    index_name:    str
    index_type:    str
    vector:        list[float]
    vector_type:   str            # "image" | "specifics"
    model_id:      str
    params_hash:   str
    source_url:    str  = ""
    specifics_src: str  = "ebay"


class S3VectorStore:
    """
    Primary durable store for all generated vectors.

    Write vectors here BEFORE loading into Qdrant.
    All keys are fully deterministic — re-runs overwrite the same key,
    enabling idempotent backfill.
    """

    def __init__(self, bucket: str, prefix: str = "vectors") -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._s3    = boto3.client("s3")

    # ── Key construction ──────────────────────────────────────────────────────

    def key_prefix(
        self,
        vector_type: str,
        model_id:    str,
        params_hash: str,
        index_type:  str,
        partition:   str,
    ) -> str:
        return "/".join([
            self.prefix,
            vector_type.lower(),
            model_id.lower().replace(" ", "-"),
            params_hash.lower(),
            index_type.lower().replace(" ", "-"),
            partition.lower(),
        ])

    def shard_key(self, prefix: str, shard_num: int) -> str:
        return f"{prefix}/{shard_num:04d}.parquet"

    @staticmethod
    def partition_for_index(index_name: str, index_type: str) -> str:
        """eBay dated → full date string. Non-eBay → 'all'."""
        if index_type == "ebay-dated":
            return index_name
        return "all"

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_shard(self, records: list[VectorRecord], shard_num: int) -> str:
        """
        Serialise records to Parquet and upload to S3.
        Returns the S3 key written.
        RAISES on failure — do NOT proceed to Qdrant upsert if this raises.
        """
        if not records:
            raise ValueError("Cannot write empty shard")

        r0     = records[0]
        part   = self.partition_for_index(r0.index_name, r0.index_type)
        prefix = self.key_prefix(
            r0.vector_type, r0.model_id, r0.params_hash, r0.index_type, part
        )
        key = self.shard_key(prefix, shard_num)

        now = datetime.now(timezone.utc)
        table = pa.table({
            "os_id":         [r.os_id         for r in records],
            "qdrant_id":     [r.qdrant_id      for r in records],
            "index_name":    [r.index_name     for r in records],
            "index_type":    [r.index_type     for r in records],
            "vector":        [r.vector         for r in records],
            "vector_type":   [r.vector_type    for r in records],
            "model_id":      [r.model_id       for r in records],
            "params_hash":   [r.params_hash    for r in records],
            "created_at":    [now              for _ in records],
            "source_url":    [r.source_url     for r in records],
            "specifics_src": [r.specifics_src  for r in records],
        })

        buf = io.BytesIO()
        pq.write_table(table, buf, compression=COMPRESSION)
        buf.seek(0)

        self._s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
        logger.debug("S3 shard written: s3://{}/{} ({} vectors)", self.bucket, key, len(records))
        return key

    # ── Read / Repopulate ─────────────────────────────────────────────────────

    def list_shards(
        self,
        vector_type:  str,
        model_id:     str,
        params_hash:  str,
        index_type:   str | None = None,
        partition:    str | None = None,
    ) -> list[str]:
        """List all shard keys matching the given attributes."""
        parts = [self.prefix, vector_type, model_id, params_hash]
        if index_type:
            parts.append(index_type)
            if partition:
                parts.append(partition)
        prefix = "/".join(parts) + "/"

        keys = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])
        return sorted(keys)

    def read_shard(self, key: str) -> pa.Table:
        obj = self._s3.get_object(Bucket=self.bucket, Key=key)
        return pq.read_table(io.BytesIO(obj["Body"].read()))

    def iter_vectors(
        self,
        vector_type:  str,
        model_id:     str,
        params_hash:  str,
        index_type:   str | None = None,
        partition:    str | None = None,
        columns:      list[str] | None = None,
    ):
        """Iterate over shard tables. Use columns=["os_id","qdrant_id","vector"] for fast reload."""
        keys = self.list_shards(vector_type, model_id, params_hash, index_type, partition)
        for key in keys:
            raw = self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            yield pq.read_table(io.BytesIO(raw), columns=columns)

    # ── Repopulate Qdrant from S3 ─────────────────────────────────────────────

    def repopulate_qdrant(
        self,
        qdrant_client,              # QdrantClient instance
        vector_type:  str,          # "image" | "specifics"
        model_id:     str,          # e.g. "clip-vit-l-14"
        params_hash:  str,          # e.g. "v2-fp16-224px-sqpad"
        collection:   str = "cards",
        index_type:   str | None = None,    # None = all index types
        partition:    str | None = None,    # None = all partitions
        batch_size:   int = 1_000,
    ) -> int:
        """
        Reload vectors from S3 into Qdrant. Safe to re-run — upsert is idempotent
        on qdrant_id. Returns total points loaded.

        Reads only the columns needed for Qdrant upsert (os_id, qdrant_id, vector)
        to minimise memory usage and S3 download size.
        """
        from qdrant_client.models import PointStruct

        total = 0
        for table in self.iter_vectors(
            vector_type, model_id, params_hash,
            index_type=index_type,
            partition=partition,
            columns=["os_id", "qdrant_id", "vector"],
        ):
            for batch_start in range(0, len(table), batch_size):
                batch = table.slice(batch_start, batch_size)
                points = []
                for i in range(len(batch)):
                    raw_id = batch["qdrant_id"][i].as_py()
                    # qdrant_ids are stored as strings; convert numeric ones to int
                    # (Qdrant accepts uint64 or UUID strings, not plain numeric strings)
                    try:
                        point_id = int(raw_id)
                    except (ValueError, TypeError):
                        point_id = raw_id  # UUID string — use as-is
                    points.append(PointStruct(
                        id=point_id,
                        vector={vector_type: batch["vector"][i].as_py()},
                        payload={"os_id": batch["os_id"][i].as_py()},
                    ))
                qdrant_client.upsert(
                    collection_name=collection,
                    points=points,
                    wait=False,
                )
                total += len(points)
                logger.debug("Upserted {:,} points to '{}'", total, collection)

        logger.info("repopulate_qdrant: {:,} points loaded from S3 → '{}'", total, collection)
        return total

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, job_id: str, state: dict) -> None:
        import json
        key = f"checkpoints/{job_id}.json"
        self._s3.put_object(
            Bucket=self.bucket, Key=key,
            Body=json.dumps(state).encode(),
        )

    def load_checkpoint(self, job_id: str) -> dict | None:
        import json
        key = f"checkpoints/{job_id}.json"
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(obj["Body"].read())
        except self._s3.exceptions.NoSuchKey:
            return None


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli_repopulate(args) -> None:
    """CLI: reload vectors from S3 → Qdrant."""
    from qdrant_client import QdrantClient

    store = S3VectorStore(args.bucket, args.prefix)

    kwargs: dict = {
        "url":         f"http://{args.qdrant_host}:{args.qdrant_port}",
        "prefer_grpc": False,
        "timeout":     60,
    }
    if args.qdrant_api_key:
        kwargs["api_key"] = args.qdrant_api_key

    client = QdrantClient(**kwargs)

    logger.info("Repopulating '{}' from s3://{}/{}", args.collection, args.bucket, args.prefix)
    n = store.repopulate_qdrant(
        qdrant_client = client,
        vector_type   = args.vector_type,
        model_id      = args.model,
        params_hash   = args.params,
        collection    = args.collection,
        index_type    = args.index_type or None,
        partition     = args.partition or None,
        batch_size    = args.batch_size,
    )
    logger.info("Done — {:,} points loaded", n)


def _cli_list(args) -> None:
    """CLI: list S3 shards for a given model/params."""
    store = S3VectorStore(args.bucket, args.prefix)
    keys = store.list_shards(
        vector_type = args.vector_type,
        model_id    = args.model,
        params_hash = args.params,
        index_type  = args.index_type or None,
        partition   = args.partition or None,
    )
    for k in keys:
        print(k)
    print(f"\nTotal: {len(keys)} shards")


if __name__ == "__main__":
    import argparse, os, sys
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(description="S3VectorStore CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared S3 args
    def add_s3_args(p):
        p.add_argument("--bucket",  required=True)
        p.add_argument("--prefix",  default="vectors")
        p.add_argument("--vector-type", dest="vector_type",
                       choices=["image", "specifics"], required=True)
        p.add_argument("--model",   required=True,
                       help="e.g. clip-vit-l-14")
        p.add_argument("--params",  required=True,
                       help="e.g. v2-fp16-224px-sqpad")
        p.add_argument("--index-type", default="",
                       help="Filter by index_type (optional)")
        p.add_argument("--partition", default="",
                       help="Filter by partition (optional)")

    # repopulate subcommand
    rep = sub.add_parser("repopulate", help="Reload S3 vectors → Qdrant")
    add_s3_args(rep)
    rep.add_argument("--qdrant-host", default=os.environ.get("QDRANT_HOST", "localhost"))
    rep.add_argument("--qdrant-port", type=int, default=int(os.environ.get("QDRANT_PORT", 6333)))
    rep.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY", ""))
    rep.add_argument("--collection",  default=os.environ.get("QDRANT_COLLECTION", "cards"))
    rep.add_argument("--batch-size",  type=int, default=1_000)
    rep.set_defaults(func=_cli_repopulate)

    # list subcommand
    lst = sub.add_parser("list", help="List S3 shards")
    add_s3_args(lst)
    lst.set_defaults(func=_cli_list)

    args = parser.parse_args()
    args.func(args)
