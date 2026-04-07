"""RDS → OpenSearch migration orchestrator.

Reads from MySQL RDS, transforms rows to OpenSearch documents,
and bulk-indexes into the new self-managed cluster.

Usage:
    # Full migration
    python -m src.ingestion.rds_migration

    # Specific source feed
    python -m src.ingestion.rds_migration --where "source_feed = 'EBAY'"

    # Date range
    python -m src.ingestion.rds_migration --where "endTime >= '2025-01-01'"

    # Dry run (transform only, no writes)
    python -m src.ingestion.rds_migration --dry-run --max-rows 1000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.rds_reader import (
    count_rows,
    get_rds_connection,
    scroll_rds,
    transform_row,
)
from src.ingestion.new_os_writer import (
    bulk_index,
    get_new_os_client,
)


BATCH_SIZE = 1000
CHECKPOINT_EVERY = 50_000
LOG_EVERY = 5000


def run_migration(
    where: str | None = None,
    table: str = "salesdata",
    batch_size: int = BATCH_SIZE,
    dry_run: bool = False,
    max_rows: int | None = None,
    checkpoint_bucket: str | None = None,
    job_id: str | None = None,
) -> dict:
    """
    Main migration pipeline: RDS → transform → OpenSearch bulk index.

    Returns stats dict with counts.
    """
    if job_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        job_id = f"rds-migration-{ts}"

    logger.info(
        "Starting migration job_id={} dry_run={} max_rows={} where={}",
        job_id, dry_run, max_rows, where,
    )

    # Connect to RDS
    conn = get_rds_connection()

    # Count total for progress
    total = count_rows(conn, table, where)
    if max_rows:
        total = min(total, max_rows)
    logger.info("Total rows to process: {}", total)

    # Connect to new OpenSearch (if not dry run)
    os_client = get_new_os_client() if not dry_run else None

    # Stats
    stats = {
        "processed": 0,
        "indexed": 0,
        "skipped": 0,
        "errors": 0,
        "indices_seen": set(),
    }

    # Batch buffer
    batch: list[tuple[str, str, dict]] = []
    pipeline_start = time.perf_counter()

    for row in scroll_rds(conn, table, where, batch_size):
        # Check max_rows limit
        if max_rows and stats["processed"] >= max_rows:
            break

        # Transform row
        try:
            index_name, doc_id, doc_body = transform_row(row)
            stats["indices_seen"].add(index_name)
        except Exception as e:
            logger.error("Transform failed for row id={}: {}", row.get("id"), e)
            stats["errors"] += 1
            continue

        batch.append((index_name, doc_id, doc_body))
        stats["processed"] += 1

        # Flush batch
        if len(batch) >= batch_size:
            if not dry_run:
                result = bulk_index(os_client, batch)
                stats["indexed"] += result["indexed"]
                stats["errors"] += result["errors"]
            else:
                stats["indexed"] += len(batch)
            batch = []

            # Log progress
            if stats["processed"] % LOG_EVERY == 0:
                elapsed = time.perf_counter() - pipeline_start
                rate = stats["processed"] / max(elapsed, 0.01)
                remaining = (total - stats["processed"]) / max(rate, 0.01)
                logger.info(
                    "Progress: {}/{} ({:.1f}%) | indexed={} errors={} | "
                    "{:.0f} rows/s | ETA {:.0f}m | indices={}",
                    stats["processed"], total,
                    100 * stats["processed"] / max(total, 1),
                    stats["indexed"], stats["errors"],
                    rate, remaining / 60,
                    len(stats["indices_seen"]),
                )

            # Checkpoint
            if (
                stats["processed"] % CHECKPOINT_EVERY == 0
                and checkpoint_bucket
                and not dry_run
            ):
                _save_checkpoint(checkpoint_bucket, job_id, stats)

    # Flush remaining
    if batch:
        if not dry_run:
            result = bulk_index(os_client, batch)
            stats["indexed"] += result["indexed"]
            stats["errors"] += result["errors"]
        else:
            stats["indexed"] += len(batch)

    conn.close()

    elapsed = time.perf_counter() - pipeline_start
    stats["indices_seen"] = list(stats["indices_seen"])
    stats["elapsed_s"] = round(elapsed, 1)
    stats["rate"] = round(stats["processed"] / max(elapsed, 0.01), 1)

    logger.info("Migration complete in {:.1f}s: {}", elapsed, {
        k: v for k, v in stats.items() if k != "indices_seen"
    })
    logger.info("Indices created: {}", len(stats["indices_seen"]))

    return stats


def _save_checkpoint(bucket: str, job_id: str, stats: dict) -> None:
    """Save progress checkpoint to S3."""
    import boto3

    s3 = boto3.client("s3")
    key = f"checkpoints/{job_id}.json"
    body = json.dumps({
        "processed": stats["processed"],
        "indexed": stats["indexed"],
        "errors": stats["errors"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
    logger.info("Checkpoint saved at {} rows", stats["processed"])


def main():
    parser = argparse.ArgumentParser(description="Migrate data from RDS to self-managed OpenSearch")
    parser.add_argument(
        "--where", default=None,
        help="SQL WHERE clause to filter rows (e.g. \"source_feed = 'EBAY'\")",
    )
    parser.add_argument("--table", default="salesdata")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Transform only, no writes")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--checkpoint-bucket", default=None, help="S3 bucket for checkpoints")
    parser.add_argument("--job-id", default=None)

    args = parser.parse_args()

    # Validate env vars
    required = ["RDS_HOST", "RDS_USER", "RDS_PASSWORD", "RDS_DATABASE"]
    if not args.dry_run:
        required.append("OPENSEARCH_DOCS_HOST")

    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        logger.error("Missing required environment variables: {}", ", ".join(missing))
        raise SystemExit(1)

    stats = run_migration(
        where=args.where,
        table=args.table,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        max_rows=args.max_rows,
        checkpoint_bucket=args.checkpoint_bucket,
        job_id=args.job_id,
    )


if __name__ == "__main__":
    main()
