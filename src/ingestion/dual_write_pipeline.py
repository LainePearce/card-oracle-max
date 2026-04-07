"""Dual-write backfill pipeline: RDS → Qdrant + new Elasticsearch simultaneously.

Each batch of RDS rows is:
  1. Embedded (image via CLIP ViT-L/14, specifics via MiniLM)
  2. Written to S3 vector store (mandatory before Qdrant)
  3. Upserted to Qdrant (vector index)
  4. Bulk indexed into new Elasticsearch (document store, no vectors)

Supports TWO RDS source databases:
  - Primary RDS   (RDS_HOST / RDS_*)   — current database
  - Secondary RDS (RDS2_HOST / RDS2_*) — older database (pre-cutover)

Because cutover dates between the two databases are not exact, both are queried
for every partition. Results are merged and deduplicated by document `id` — the
primary RDS record wins on collision (it is considered more authoritative).

Invoke via:
  python -m src.ingestion.dual_write_pipeline \
    --where "endTime >= '2025-01-01' AND endTime < '2025-05-01'" \
    [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from typing import Iterator

from loguru import logger
from dotenv import load_dotenv

load_dotenv()


BATCH_SIZE       = 500    # docs per pipeline cycle (embed + write)
CHECKPOINT_EVERY = 10_000 # save S3 checkpoint after N docs
LOG_EVERY        = 1_000


# ── RDS helpers ───────────────────────────────────────────────────────────────

def _make_rds_connection(
    host: str, port: int, user: str, password: str, database: str
):
    """Open a single MySQL/RDS connection."""
    import pymysql
    return pymysql.connect(
        host=host, port=port, user=user, password=password,
        database=database, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
    )


def get_rds_connections() -> tuple:
    """
    Return (primary_conn, secondary_conn | None).

    Primary RDS is required.
    Secondary RDS (RDS2_*) is optional — if RDS2_HOST is not set, returns None
    for the secondary connection and only primary data is used.
    """
    primary = _make_rds_connection(
        host     = os.environ["RDS_HOST"],
        port     = int(os.environ.get("RDS_PORT", 3306)),
        user     = os.environ["RDS_USER"],
        password = os.environ["RDS_PASSWORD"],
        database = os.environ["RDS_DATABASE"],
    )

    secondary = None
    if os.environ.get("RDS2_HOST"):
        try:
            secondary = _make_rds_connection(
                host     = os.environ["RDS2_HOST"],
                port     = int(os.environ.get("RDS2_PORT", 3306)),
                user     = os.environ["RDS2_USER"],
                password = os.environ["RDS2_PASSWORD"],
                database = os.environ["RDS2_DATABASE"],
            )
            logger.info("Secondary RDS connected ({})", os.environ["RDS2_HOST"])
        except Exception as e:
            logger.warning("Secondary RDS connection failed — skipping: {}", e)

    return primary, secondary


def _iter_rows_from_conn(conn, where_clause: str) -> Iterator[dict]:
    """
    Stream rows from a single RDS connection matching the WHERE clause.
    Each row is expected to have at minimum: id, gallery_url, item_specifics,
    index_name, index_type, has_item_specifics.
    """
    from src.ingestion.rds_reader import iter_rows
    yield from iter_rows(conn, where_clause=where_clause)


def iter_merged_rows(
    primary_conn,
    secondary_conn,
    where_clause: str,
) -> Iterator[tuple[dict, str]]:
    """
    Yield (row, source) tuples merged from primary and secondary RDS.

    Strategy:
      1. Stream ALL rows from primary → build a seen-id set, yield each row.
      2. Stream ALL rows from secondary → yield only if id not already seen.

    This means primary rows always win on collision and secondary fills gaps.
    'source' is "primary" or "secondary" — logged for observability.

    Memory note: seen_ids holds one integer per unique document. At 80M docs
    that's ~640MB for 64-bit ints. Acceptable on a g4dn.xlarge (64GB RAM).
    If memory is a concern, replace with a Bloom filter from pybloom_live.
    """
    seen_ids: set[int] = set()

    logger.info("Reading primary RDS...")
    for row in _iter_rows_from_conn(primary_conn, where_clause):
        seen_ids.add(int(row["id"]))
        yield row, "primary"

    if secondary_conn is None:
        logger.info("No secondary RDS configured — using primary only")
        return

    logger.info(
        "Reading secondary RDS (deduplication against {:,} primary IDs)...",
        len(seen_ids),
    )
    secondary_new = 0
    for row in _iter_rows_from_conn(secondary_conn, where_clause):
        doc_id = int(row["id"])
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            secondary_new += 1
            yield row, "secondary"

    logger.info("Secondary RDS contributed {:,} unique rows", secondary_new)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(where_clause: str, dry_run: bool = False) -> None:
    from src.ingestion.qdrant_writer import get_qdrant_client, build_point
    from src.ingestion.new_os_writer import get_es_client, bulk_index
    from src.ingestion.rds_reader import build_os_document
    from src.embeddings.image_encoder import ImageEncoder
    from src.embeddings.text_encoder import TextEncoder, format_specifics
    from src.embeddings.vector_store import S3VectorStore, VectorRecord
    from src.schema.collection import COLLECTION_NAME

    # ── Connections ──────────────────────────────────────────────────────────
    logger.info("Connecting to RDS sources...")
    primary_conn, secondary_conn = get_rds_connections()

    logger.info("Connecting to Qdrant...")
    qdrant = get_qdrant_client()

    logger.info("Connecting to Elasticsearch...")
    es_client = get_es_client()

    vector_store = S3VectorStore(
        bucket=os.environ["S3_VECTOR_BUCKET"],
        prefix=os.environ.get("S3_VECTOR_PREFIX", "vectors"),
    )

    # ── Encoders ─────────────────────────────────────────────────────────────
    device = os.environ.get("EMBEDDING_DEVICE", "cuda")
    logger.info("Loading encoders on {}...", device)
    image_encoder = ImageEncoder(device=device)
    text_encoder  = TextEncoder(device="cpu")

    # ── Counters ─────────────────────────────────────────────────────────────
    stats = {
        "rows_read":        0,
        "rows_primary":     0,
        "rows_secondary":   0,
        "embedded":         0,
        "qdrant_upserted":  0,
        "es_indexed":       0,
        "s3_shards":        0,
        "errors":           0,
    }
    shard_num = 0
    t_start   = time.perf_counter()

    if dry_run:
        logger.warning("DRY RUN — no writes to Qdrant, Elasticsearch, or S3")
    logger.info("Starting dual-write pipeline. WHERE: {}", where_clause or "(all rows)")

    batch_rows: list[dict] = []

    # ── Flush function ────────────────────────────────────────────────────────
    def flush_batch(rows: list[dict]) -> None:
        nonlocal shard_num

        if not rows:
            return

        # Step 1: Encode
        gallery_urls    = [r.get("gallery_url") for r in rows]
        specifics_texts = [
            format_specifics(r.get("item_specifics") or {})
            for r in rows
        ]

        image_vecs_raw = image_encoder.encode_batch(gallery_urls)
        specifics_vecs = text_encoder.encode_batch(specifics_texts)

        # Step 2: S3 (always before Qdrant)
        image_records     = []
        specifics_records = []

        for i, row in enumerate(rows):
            img_vec  = image_vecs_raw[i]
            spec_vec = specifics_vecs[i] if specifics_vecs is not None else None
            src_flag = "ebay" if row.get("has_item_specifics") else "none"

            if img_vec is not None:
                image_records.append(VectorRecord(
                    os_id         = str(row["id"]),
                    qdrant_id     = str(row["id"]),
                    index_name    = row.get("index_name", "unknown"),
                    index_type    = row.get("index_type", "ebay-dated"),
                    vector        = img_vec.tolist(),
                    vector_type   = "image",
                    model_id      = "clip-vit-l-14",
                    params_hash   = "v1-fp16-224px",
                    source_url    = row.get("gallery_url", ""),
                    specifics_src = src_flag,
                ))
            if spec_vec is not None and row.get("has_item_specifics"):
                specifics_records.append(VectorRecord(
                    os_id         = str(row["id"]),
                    qdrant_id     = str(row["id"]),
                    index_name    = row.get("index_name", "unknown"),
                    index_type    = row.get("index_type", "ebay-dated"),
                    vector        = spec_vec.tolist(),
                    vector_type   = "specifics",
                    model_id      = "minilm-l6-v2",
                    params_hash   = "v1-mean-256tok",
                    source_url    = "",
                    specifics_src = src_flag,
                ))

        if not dry_run:
            if image_records:
                vector_store.write_shard(image_records, shard_num)
            if specifics_records:
                vector_store.write_shard(specifics_records, shard_num)
            shard_num += 1
            stats["s3_shards"] += 1

        # Step 3: Qdrant upsert
        if not dry_run:
            from qdrant_client.models import PointStruct
            points = []
            for i, row in enumerate(rows):
                img_vec  = image_vecs_raw[i]
                spec_vec = specifics_vecs[i] if specifics_vecs is not None else None

                if img_vec is None and spec_vec is None:
                    continue

                vectors: dict = {}
                if img_vec is not None:
                    vectors["image"] = img_vec.tolist()
                if spec_vec is not None and row.get("has_item_specifics"):
                    vectors["specifics"] = spec_vec.tolist()

                points.append(PointStruct(
                    id=int(row["id"]),
                    vector=vectors,
                    payload=build_point(row),
                ))

            if points:
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=False,
                )
                stats["qdrant_upserted"] += len(points)

        # Step 4: Elasticsearch bulk index
        if not dry_run:
            es_docs = [
                (row["index_name"], str(row["id"]), build_os_document(row))
                for row in rows
            ]
            result = bulk_index(es_client, es_docs)
            stats["es_indexed"] += result["indexed"]
            if result["errors"]:
                stats["errors"] += result["errors"]
                logger.warning("{} ES bulk index errors in this batch", result["errors"])

        stats["embedded"] += sum(1 for v in image_vecs_raw if v is not None)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for row, source in iter_merged_rows(primary_conn, secondary_conn, where_clause):
        batch_rows.append(row)
        stats["rows_read"] += 1
        stats[f"rows_{source}"] += 1

        if len(batch_rows) >= BATCH_SIZE:
            flush_batch(batch_rows)
            batch_rows.clear()

        if stats["rows_read"] % LOG_EVERY == 0:
            elapsed = time.perf_counter() - t_start
            rate    = stats["rows_read"] / elapsed if elapsed > 0 else 0
            logger.info(
                "rows={:,} (pri={:,} sec={:,})  embedded={:,}  qdrant={:,}  "
                "es={:,}  rate={:.0f}/s  errors={}",
                stats["rows_read"], stats["rows_primary"], stats["rows_secondary"],
                stats["embedded"], stats["qdrant_upserted"],
                stats["es_indexed"], rate, stats["errors"],
            )

    # Flush final partial batch
    if batch_rows:
        flush_batch(batch_rows)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Pipeline complete.\n"
        "  rows_total={:,}  primary={:,}  secondary={:,}\n"
        "  qdrant={:,}  es={:,}  s3_shards={}\n"
        "  errors={}  elapsed={:.1f}s",
        stats["rows_read"], stats["rows_primary"], stats["rows_secondary"],
        stats["qdrant_upserted"], stats["es_indexed"],
        stats["s3_shards"], stats["errors"], elapsed,
    )

    primary_conn.close()
    if secondary_conn:
        secondary_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-write backfill: RDS (primary + secondary) → Qdrant + Elasticsearch"
    )
    parser.add_argument(
        "--where",
        default="",
        help="SQL WHERE clause fragment, e.g. \"endTime >= '2025-01-01' AND endTime < '2025-05-01'\"",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Embed and log but do not write to Qdrant, Elasticsearch, or S3",
    )
    args = parser.parse_args()
    run(where_clause=args.where, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
