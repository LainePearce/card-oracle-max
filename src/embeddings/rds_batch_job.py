"""AWS Batch / standalone GPU worker entry point for RDS-sourced embedding generation.

Reads documents from RDS MySQL via scroll, generates CLIP (image) and
MiniLM (text specifics) embeddings, writes vectors to S3 as the primary
durable store, then upserts into Qdrant.

The pipeline is idempotent: re-running with the same --where clause
overwrites the same S3 keys and re-upserts the same Qdrant point IDs.

Usage (GPU instance)
--------------------
    # Full date range backfill
    python -m src.embeddings.rds_batch_job \\
        --where "endTime >= '2026-01-01' AND endTime < '2026-04-06'" \\
        --image-device cuda

    # Dry run (no writes to S3 or Qdrant)
    python -m src.embeddings.rds_batch_job \\
        --where "endTime >= '2026-01-01' AND endTime < '2026-04-06'" \\
        --dry-run

    # Skip S3 (write to Qdrant only — use when S3 already populated)
    python -m src.embeddings.rds_batch_job \\
        --where "endTime >= '2026-01-01' AND endTime < '2026-04-06'" \\
        --no-s3

    # Resume from checkpoint
    python -m src.embeddings.rds_batch_job \\
        --where "endTime >= '2026-01-01' AND endTime < '2026-04-06'" \\
        --checkpoint-id my-job-2026-01

Environment variables (loaded from .env)
-----------------------------------------
    # Primary RDS (required)
    RDS_HOST, RDS_PORT, RDS_USER, RDS_PASSWORD, RDS_DATABASE

    # Secondary RDS (optional — used to fill gaps; primary wins on duplicate IDs)
    RDS2_HOST, RDS2_PORT, RDS2_USER, RDS2_PASSWORD, RDS2_DATABASE

    # Qdrant
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION

    # S3
    S3_VECTOR_BUCKET, S3_VECTOR_PREFIX

    # Misc
    RDS_TABLE           table name (default: salesdata)
    EMBEDDING_DEVICE    cuda | cpu (overridden by --image-device)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow `python -m src.embeddings.rds_batch_job` from the repo root.
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

# ── Project imports (after path setup) ───────────────────────────────────────
from src.embeddings.image_encoder import ImageEncoder
from src.embeddings.text_encoder import TextEncoder, format_specifics
from src.embeddings.vector_store import S3VectorStore, VectorRecord
from src.ingestion.qdrant_writer import (
    build_point,
    extract_payload,
    get_qdrant_client,
    os_id_to_qdrant_id,
    upsert_batch,
    UPSERT_BATCH_SIZE,
)
from src.ingestion.rds_reader import (
    count_rows_range,
    determine_index_name,
    get_rds_connection,
    scroll_rds,
    transform_row,
)
from src.ingestion.opensearch_reader import classify_index

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_MODEL_ID  = "clip-vit-l-14"
IMAGE_PARAMS    = "v2-fp16-224px-sqpad"   # v2: square-pad before centre-crop
TEXT_MODEL_ID   = "minilm-l6-v2"
TEXT_PARAMS     = "v1-mean-256tok"
RDS_TABLE       = os.environ.get("RDS_TABLE", "salesdata")

# How many rows to accumulate before flushing a shard to S3 + Qdrant
SHARD_FLUSH_SIZE = 10_000

# How many rows to accumulate before saving a checkpoint
CHECKPOINT_EVERY = 10_000

# Thread pool size for parallel image downloading (download only — no GPU in threads)
# Tune via IMAGE_DOWNLOAD_WORKERS env var. 16 saturates a g5.xlarge; raise to 32 on
# faster network (e.g. jumbo-frame VPC or if images are served from us-west-1).
IMAGE_DOWNLOAD_WORKERS = int(os.environ.get("IMAGE_DOWNLOAD_WORKERS", 16))


# ── Secondary RDS (optional gap-fill) ────────────────────────────────────────

def _has_secondary_rds() -> bool:
    return bool(os.environ.get("RDS2_HOST"))


def get_secondary_rds_connection():
    """Connect to the optional secondary RDS instance. Returns None if not configured."""
    if not _has_secondary_rds():
        return None

    import pymysql
    import pymysql.cursors

    return pymysql.connect(
        host=os.environ["RDS2_HOST"],
        port=int(os.environ.get("RDS2_PORT", 3306)),
        user=os.environ["RDS2_USER"],
        password=os.environ["RDS2_PASSWORD"],
        database=os.environ["RDS2_DATABASE"],
        charset="utf8mb4",
        connect_timeout=30,
    )


def iter_merged_rows(
    primary_conn,
    secondary_conn,
    start_date: str,
    end_date: str,
    extra_where: str | None = None,
    resume_date: str | None = None,
):
    """
    Yield rows from primary RDS day-by-day, then fill gaps from secondary.
    Primary wins on duplicate IDs.

    Yields: (row_dict, source_label)
    """
    seen_ids: set[int] = set()

    logger.info("Streaming from primary RDS ({} → {})...", start_date, end_date)
    for row in scroll_rds(
        primary_conn, RDS_TABLE,
        start_date=start_date,
        end_date=end_date,
        extra_where=extra_where,
        resume_date=resume_date,
    ):
        seen_ids.add(int(row["id"]))
        yield row, "primary"

    if secondary_conn is None:
        return

    logger.info(
        "Primary complete ({} rows). Checking secondary RDS for gaps...",
        len(seen_ids),
    )
    gap_count = 0
    for row in scroll_rds(
        secondary_conn, RDS_TABLE,
        start_date=start_date,
        end_date=end_date,
        extra_where=extra_where,
        resume_date=resume_date,
    ):
        if int(row["id"]) not in seen_ids:
            gap_count += 1
            seen_ids.add(int(row["id"]))
            yield row, "secondary"

    logger.info("Secondary added {} gap rows", gap_count)


# ── Image downloading (parallel) ──────────────────────────────────────────────

def _download_one(args):
    """Worker fn: (index, url, encoder) → (index, PIL image or None).
    Downloads only — no GPU work inside threads.
    """
    import io as _io
    from PIL import Image as PILImage

    idx, url, encoder = args
    if not url:
        return idx, None
    try:
        import httpx as _httpx
        resp = _httpx.get(
            url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; card-oracle/1.0)"},
        )
        resp.raise_for_status()
        img = PILImage.open(_io.BytesIO(resp.content)).convert("RGB")
        img = encoder._pad_to_square(img)   # apply padding before GPU batch
        return idx, img
    except Exception as e:
        logger.debug("Image download failed at index {} ({}): {}", idx, url[:60], e)
        return idx, None


def download_images_parallel(urls: list[str | None], encoder: ImageEncoder) -> list:
    """
    Download images in parallel threads. Returns list of PIL images or None.
    GPU encoding is NOT done here — threads handle I/O only.
    """
    results = [None] * len(urls)
    tasks   = [(i, url, encoder) for i, url in enumerate(urls) if url]

    with ThreadPoolExecutor(max_workers=IMAGE_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_download_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, img = future.result()
                results[idx] = img
            except Exception as e:
                logger.warning("Download worker error at index {}: {}", idx, e)

    return results


def encode_pil_batch(pil_images: list, encoder: ImageEncoder) -> list:
    """
    GPU-batch encode a list of PIL images in a single forward pass.
    Much more efficient than encoding one at a time inside threads.
    Returns list of np.ndarray or None, same length as input.
    """
    import torch

    encoder._load()
    results = [None] * len(pil_images)
    valid   = [(i, img) for i, img in enumerate(pil_images) if img is not None]

    if not valid:
        return results

    # Preprocess all images → stack into a single batch tensor
    tensors = torch.stack([encoder._preprocess(img) for _, img in valid])
    tensors = tensors.to(encoder._device)

    with torch.no_grad():
        if encoder._device_str == "cuda":
            with torch.cuda.amp.autocast():
                vecs = encoder._model.encode_image(tensors)
        else:
            vecs = encoder._model.encode_image(tensors)

    vecs = vecs / vecs.norm(dim=-1, keepdim=True)   # L2 normalise
    vecs_np = vecs.cpu().float().numpy()

    for batch_i, (orig_i, _) in enumerate(valid):
        results[orig_i] = vecs_np[batch_i]

    return results


# ── S3 shard buffer ───────────────────────────────────────────────────────────

class ShardBuffer:
    """
    Accumulates VectorRecords per (vector_type, index_type, partition) partition
    and flushes to S3 when the buffer reaches the flush threshold.
    """

    def __init__(
        self,
        store: S3VectorStore | None,
        flush_size: int = SHARD_FLUSH_SIZE,
        dry_run: bool = False,
    ):
        self._store      = store
        self._flush_size = flush_size
        self._dry_run    = dry_run
        # key: (vector_type, index_type, partition) → list[VectorRecord]
        self._buffers: dict[tuple, list[VectorRecord]] = defaultdict(list)
        # key: same tuple → next shard number (0-based)
        self._shard_nums: dict[tuple, int] = defaultdict(int)

    def add(self, record: VectorRecord) -> list[str]:
        """Add a record. Returns list of S3 keys written (may be empty)."""
        key = (record.vector_type, record.index_type,
               S3VectorStore.partition_for_index(record.index_name, record.index_type))
        self._buffers[key].append(record)

        if len(self._buffers[key]) >= self._flush_size:
            return self._flush(key)
        return []

    def flush_all(self) -> list[str]:
        """Flush all remaining buffers. Call at end of job."""
        keys_written = []
        for key in list(self._buffers.keys()):
            if self._buffers[key]:
                keys_written.extend(self._flush(key))
        return keys_written

    def _flush(self, key: tuple) -> list[str]:
        records = self._buffers.pop(key, [])
        if not records:
            return []

        shard_num = self._shard_nums[key]
        self._shard_nums[key] = shard_num + 1

        if self._dry_run or self._store is None:
            logger.info(
                "[dry-run] Would write S3 shard: {} records, key={}, shard={}",
                len(records), key, shard_num,
            )
            return []

        try:
            s3_key = self._store.write_shard(records, shard_num)
            logger.debug("S3 shard written: {} ({} records)", s3_key, len(records))
            return [s3_key]
        except Exception as e:
            logger.error("S3 write failed for key={} shard={}: {}", key, shard_num, e)
            raise  # do not proceed to Qdrant if S3 fails

    def get_shard_state(self) -> dict:
        """Serialisable state for checkpointing."""
        return {str(k): v for k, v in self._shard_nums.items()}

    def restore_shard_state(self, state: dict) -> None:
        for k_str, num in state.items():
            k = tuple(k_str.strip("()").replace("'", "").split(", "))
            self._shard_nums[tuple(k)] = num


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    """Main entry point for the RDS batch embedding job."""

    # ── Connections ──────────────────────────────────────────────────────────
    logger.info("Connecting to primary RDS...")
    primary_conn = get_rds_connection()

    secondary_conn = None
    if _has_secondary_rds():
        logger.info("Connecting to secondary RDS ({})...", os.environ.get("RDS2_HOST"))
        secondary_conn = get_secondary_rds_connection()

    logger.info("Connecting to Qdrant ({})...", os.environ.get("QDRANT_HOST", "localhost"))
    qdrant = get_qdrant_client()

    # ── S3 store ─────────────────────────────────────────────────────────────
    s3_store: S3VectorStore | None = None
    if not args.no_s3 and not args.dry_run:
        bucket = os.environ.get("S3_VECTOR_BUCKET", "")
        prefix = os.environ.get("S3_VECTOR_PREFIX", "vectors")
        if not bucket:
            logger.warning(
                "S3_VECTOR_BUCKET not set — vectors will NOT be written to S3. "
                "Pass --no-s3 to suppress this warning."
            )
        else:
            s3_store = S3VectorStore(bucket=bucket, prefix=prefix)
            logger.info("S3 vector store: s3://{}/{}", bucket, prefix)

    # ── Checkpoint (resume support) ──────────────────────────────────────────
    checkpoint_id = args.checkpoint_id
    resume_date: str | None = None

    if checkpoint_id and s3_store:
        cp = s3_store.load_checkpoint(checkpoint_id)
        if cp:
            resume_date = cp.get("last_completed_date")
            if resume_date:
                logger.info(
                    "Resuming from checkpoint '{}': skipping up to (but not including) {}",
                    checkpoint_id, resume_date,
                )

    # ── Encoders (lazy-loaded) ───────────────────────────────────────────────
    device = args.image_device or os.environ.get("EMBEDDING_DEVICE", "cuda")
    image_encoder = ImageEncoder(
        model_name="ViT-L/14",
        pretrained="openai",
        device=device,
    )
    text_encoder = TextEncoder(device="cpu")  # CPU is fine for MiniLM

    logger.info("Image encoder: ViT-L/14 on {}. Text encoder: MiniLM on cpu.", device)

    # ── Count rows for progress logging ─────────────────────────────────────
    # Use resume_date (not start_date) so the count reflects rows that will
    # actually be streamed, giving an accurate ETA when resuming mid-range.
    try:
        count_from = resume_date or args.start_date
        total = count_rows_range(
            primary_conn, RDS_TABLE,
            count_from, args.end_date,
            extra_where=args.extra_where,
        )
        logger.info("Rows to process (primary): {:,}", total)
    except Exception as e:
        logger.warning("Could not count rows: {} — proceeding without ETA", e)
        total = None

    # ── Shard buffers ────────────────────────────────────────────────────────
    img_buffer  = ShardBuffer(s3_store, args.shard_size, args.dry_run)
    spec_buffer = ShardBuffer(s3_store, args.shard_size, args.dry_run)

    # ── Process rows ─────────────────────────────────────────────────────────
    processed    = 0
    skipped_novec= 0
    start_time   = time.perf_counter()
    qdrant_batch : list = []

    row_stream = iter_merged_rows(
        primary_conn, secondary_conn,
        start_date=args.start_date,
        end_date=args.end_date,
        extra_where=args.extra_where,
        resume_date=resume_date,
    )

    # We accumulate a mini-batch for parallel image downloading
    mini_batch: list[dict] = []

    def flush_mini_batch(batch: list[dict]) -> None:
        nonlocal processed, skipped_novec

        if not batch:
            return

        # --- Extract docs ---
        urls         = [row.get("galleryURL") or "" for row in batch]
        specs_texts  = []
        transformed  = []

        for row in batch:
            try:
                idx_name, doc_id, doc = transform_row(row)
                transformed.append((idx_name, doc_id, doc))
                # Prefer structured itemSpecifics; fall back to title so rows
                # with no structured data still get a specifics vector.
                specs_text = format_specifics(doc.get("itemSpecifics") or {})
                if not specs_text:
                    specs_text = (doc.get("title") or "").lower().strip()
                specs_texts.append(specs_text)
            except Exception as e:
                logger.warning("transform_row failed for id={}: {}", row.get("id"), e)
                transformed.append(None)
                specs_texts.append("")

        # --- Image vectors: download in parallel, encode in one GPU batch ---
        pil_images = download_images_parallel(urls, image_encoder)
        image_vecs = encode_pil_batch(pil_images, image_encoder)

        # --- Specifics vectors (batched CPU inference) ---
        specifics_vecs = text_encoder.encode_batch(specs_texts)

        # --- Build points and S3 records ---
        for i, row in enumerate(batch):
            if transformed[i] is None:
                skipped_novec += 1
                continue

            idx_name, doc_id, doc = transformed[i]
            image_vec   = image_vecs[i]
            spec_vec    = specifics_vecs[i]

            if image_vec is None and spec_vec is None:
                skipped_novec += 1
                logger.warning(
                    "Skipping id={} — no image (url={}) and no specifics/title text",
                    doc_id, (urls[i] or "")[:60],
                )
                continue

            qdrant_id = os_id_to_qdrant_id(doc_id)
            classification = classify_index(idx_name)
            index_type = classification.get("index_type", "unknown")
            # Normalise index_type to match S3 key naming convention
            index_type_key = idx_name if classification.get("marketplace") != "ebay" else index_type
            # Use a consistent slug for the index_type path component
            index_type_slug = {
                "ebay_dated":         "ebay-dated",
                "marketplace_suffix": idx_name.rsplit("-", 1)[-1],  # suffix
                "unknown":            "unknown",
            }.get(classification.get("index_type", "unknown"), "unknown")

            # --- S3 records ---
            common = dict(
                os_id=doc_id,
                qdrant_id=str(qdrant_id),
                index_name=idx_name,
                index_type=index_type_slug,
            )

            if image_vec is not None:
                img_rec = VectorRecord(
                    **common,
                    vector=image_vec.tolist(),
                    vector_type="image",
                    model_id=IMAGE_MODEL_ID,
                    params_hash=IMAGE_PARAMS,
                    source_url=urls[i],
                    specifics_src="ebay"
                    if classification.get("has_item_specifics")
                    else "none",
                )
                img_buffer.add(img_rec)

            if spec_vec is not None:
                spec_rec = VectorRecord(
                    **common,
                    vector=spec_vec.tolist(),
                    vector_type="specifics",
                    model_id=TEXT_MODEL_ID,
                    params_hash=TEXT_PARAMS,
                    source_url="",
                    specifics_src="ebay"
                    if classification.get("has_item_specifics")
                    else "none",
                )
                spec_buffer.add(spec_rec)

            # --- Qdrant payload ---
            has_specs = classification.get("has_item_specifics", False)
            payload = extract_payload(
                doc,
                doc_id=doc_id,
                specifics_source="ebay" if has_specs else "none",
            )

            point = build_point(qdrant_id, payload, image_vec, spec_vec)
            if point is not None:
                qdrant_batch.append(point)

            processed += 1

        # --- Flush Qdrant batch when large enough ---
        if len(qdrant_batch) >= UPSERT_BATCH_SIZE:
            _flush_qdrant(qdrant_batch, qdrant, args)
            qdrant_batch.clear()

        # --- Progress log ---
        elapsed = time.perf_counter() - start_time
        rate    = processed / elapsed if elapsed > 0 else 0
        eta_str = ""
        if total:
            remaining = total - processed
            eta_secs  = remaining / rate if rate > 0 else 0
            eta_str   = f" | ETA ~{eta_secs/60:.1f}min"
        logger.info(
            "Processed {:,} rows | {:.0f} rows/s | skipped {} | {:,} in Qdrant queue{}",
            processed, rate, skipped_novec, len(qdrant_batch), eta_str,
        )

        # --- Checkpoint (every CHECKPOINT_EVERY rows) ---
        if checkpoint_id and s3_store and processed % CHECKPOINT_EVERY < args.batch_size:
            # Derive the date of the last row processed for resume
            last_end_time = str(batch[-1].get("endTime", ""))[:10]  # YYYY-MM-DD
            s3_store.save_checkpoint(checkpoint_id, {
                "last_completed_date": last_end_time,
                "processed": processed,
                "start_date": args.start_date,
                "end_date": args.end_date,
            })
            logger.debug("Checkpoint saved: last_completed_date={}", last_end_time)

    # ── Stream rows into mini-batches ────────────────────────────────────────
    for row, _source in row_stream:
        mini_batch.append(row)
        if len(mini_batch) >= args.batch_size:
            flush_mini_batch(mini_batch)
            mini_batch.clear()

    # Flush remainder
    if mini_batch:
        flush_mini_batch(mini_batch)
        mini_batch.clear()

    # ── Final Qdrant flush ───────────────────────────────────────────────────
    if qdrant_batch:
        _flush_qdrant(qdrant_batch, qdrant, args)
        qdrant_batch.clear()

    # ── Final S3 shard flush ────────────────────────────────────────────────
    logger.info("Flushing remaining S3 shard buffers...")
    img_buffer.flush_all()
    spec_buffer.flush_all()

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    logger.info("─" * 60)
    logger.info("Job complete.")
    logger.info("  Rows processed : {:,}", processed)
    logger.info("  Rows skipped   : {:,}", skipped_novec)
    logger.info("  Elapsed        : {:.1f}s ({:.0f} rows/s)", elapsed, processed / elapsed if elapsed > 0 else 0)
    if args.dry_run:
        logger.info("  Mode           : DRY RUN — no writes performed")

    # Close connections
    try:
        primary_conn.close()
        if secondary_conn:
            secondary_conn.close()
    except Exception:
        pass


def _flush_qdrant(batch: list, qdrant, args: argparse.Namespace) -> None:
    """Upsert a batch of points to Qdrant, skipping if dry-run."""
    if args.dry_run:
        logger.info("[dry-run] Would upsert {} points to Qdrant", len(batch))
        return
    try:
        upsert_batch(qdrant, batch)
    except Exception as e:
        logger.error("Qdrant upsert failed: {}", e)
        raise


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate CLIP + MiniLM embeddings from RDS and load into S3 + Qdrant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--start-date",
        required=True,
        help='Inclusive start date YYYY-MM-DD. Rows with endTime >= this date are processed.',
    )
    p.add_argument(
        "--end-date",
        required=True,
        help='Exclusive end date YYYY-MM-DD. Rows with endTime < this date are processed.',
    )
    p.add_argument(
        "--extra-where",
        default=None,
        dest="extra_where",
        help='Optional extra SQL filter ANDed into every daily band query (e.g. "source_feed = \'EBAY\'")',
    )
    # Kept for backward compat but ignored if --start-date/--end-date are supplied
    p.add_argument("--where", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--image-device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device for CLIP image encoding (default: cuda if available)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Rows per mini-batch for image encoding (default: 100)",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        help="Vector records per S3 parquet shard (default: 10,000)",
    )
    p.add_argument(
        "--checkpoint-id",
        default=None,
        help="Checkpoint ID for resume support (e.g. job-2026-01). "
             "State is saved to S3 checkpoints/<id>.json.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Process rows and encode vectors but skip all writes (S3 + Qdrant).",
    )
    p.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip S3 writes (upsert to Qdrant only). Not recommended for production.",
    )
    p.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (default: from QDRANT_COLLECTION env or 'cards')",
    )

    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    # Override collection from CLI
    if args.collection:
        os.environ["QDRANT_COLLECTION"] = args.collection

    # Configure loguru — JSON to file + pretty to stderr
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    log_file = _ROOT / "logs" / "rds_batch_job_{time}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), level="DEBUG", rotation="500 MB", serialize=False)

    logger.info("=" * 60)
    logger.info("RDS Batch Embedding Job")
    logger.info("  Start date: {}", args.start_date)
    logger.info("  End date:   {}", args.end_date)
    if args.extra_where:
        logger.info("  Extra WHERE:{}", args.extra_where)
    logger.info("  Device:     {}", args.image_device or "auto")
    logger.info("  Batch size: {}", args.batch_size)
    logger.info("  Dry run:    {}", args.dry_run)
    logger.info("  Skip S3:    {}", args.no_s3)
    logger.info("=" * 60)

    run(args)
