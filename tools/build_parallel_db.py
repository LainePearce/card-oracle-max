#!/usr/bin/env python3
"""
Build the parallel variant dictionary from RDS sales data.

One-off script — run once, no auto-update triggers.
Streams rows from salesdata where ItemSpecifics contains parallel data,
parses and aggregates into a canonical lookup table used by the metric
projection head and title-based parallel resolution.

Output:
    data/parallel_db.json        — canonical lookup: value → {count, brands, genres, tokens, samples}
    data/parallel_tokens.json    — inverted index: token → [parallel values] (for title scanning)
    data/parallel_db.parquet     — full frequency table (Snappy-compressed)
    data/parallel_db_meta.json   — build provenance and top-10 summary

Usage:
    python tools/build_parallel_db.py
    python tools/build_parallel_db.py --max-rows 5000000
    python tools/build_parallel_db.py --min-count 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from loguru import logger

import pymysql
import pymysql.cursors

from src.ingestion.rds_reader import get_rds_connection, parse_item_specifics


# ── Config ───────────────────────────────────────────────────────────────────────

DEFAULT_MAX_ROWS  = 3_000_000   # max parallel-containing rows to scan from RDS
DEFAULT_MIN_COUNT = 5           # minimum occurrences to write to output
BATCH_SIZE        = 50_000      # rows per MySQL round-trip

DATA_DIR = _ROOT / "data"

# Tokens that appear in nearly every parallel value — useless for disambiguation
STOPWORDS = {
    "", "a", "an", "and", "card", "cards", "edition", "non",
    "not", "of", "or", "the", "version", "with",
}


# ── Normalisation ────────────────────────────────────────────────────────────────

def normalise_parallel(val: str | None) -> str | None:
    """
    Lowercase, strip, collapse whitespace. Return None for noise / empty values.
    Reject values that can't be real parallel names (too short, too long, pure noise).
    """
    if not val:
        return None
    v = str(val).lower().strip()
    v = re.sub(r"\s+", " ", v)

    # Strip literal "[none]" / "(none)" bracket wrappers sellers sometimes submit
    v = re.sub(r"^\[none\]$|^\(none\)$|^\[null\]$|^\(null\)$", "none", v)

    # Reject known non-values
    if v in ("", "null", "none", "n/a", "na", "no", "yes", "true", "false",
             "-", "base", "regular", "standard", "normal", "plain",
             "[none]", "(none)", "[null]", "(null)"):
        return None

    # Reject values that are implausibly short or very long
    if len(v) < 2 or len(v) > 120:
        return None

    # Reject values that are purely numeric (serial number, not parallel name)
    if re.fullmatch(r"[\d/\-\s]+", v):
        return None

    return v


def tokenise(parallel: str) -> list[str]:
    """
    Split a parallel value into searchable tokens for the inverted index.
    Splits on whitespace, slash, hyphen, and underscore.
    """
    raw_tokens = re.split(r"[\s/\-_]+", parallel.lower())
    return [
        t for t in raw_tokens
        if t and t not in STOPWORDS and len(t) >= 3
    ]


# ── Aggregation ──────────────────────────────────────────────────────────────────

def build_parallel_db(max_rows: int, min_count: int) -> dict:
    """
    Stream rows from the RDS salesdata table where ItemSpecifics contains
    parallel data. Parse ItemSpecifics for each row, extract parallel +
    brand + genre + source_feed. Aggregate into frequency tables.

    Uses id-based pagination (WHERE id > last_id … LIMIT N) so each batch
    uses the primary key index — fast even on very large tables.

    Returns a dict keyed by normalised parallel value with aggregated metadata.
    """
    conn = get_rds_connection()
    logger.info("Connected to RDS — scanning up to {:,} rows", max_rows)

    table = os.environ.get("RDS_SALES_TABLE", "salesdata")

    # Aggregation containers
    parallel_counts: Counter             = Counter()
    brand_counts:    dict[str, Counter]  = defaultdict(Counter)
    genre_counts:    dict[str, Counter]  = defaultdict(Counter)
    source_counts:   dict[str, Counter]  = defaultdict(Counter)
    title_samples:   dict[str, list]     = defaultdict(list)

    scanned = 0
    matched = 0
    last_id = 0
    batch_n = 0
    t_start = time.perf_counter()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            while matched < max_rows:
                cursor.execute(
                    f"""
                    SELECT id, ItemSpecifics, source_feed, title
                    FROM {table}
                    WHERE id > %s
                      AND ItemSpecifics IS NOT NULL
                      AND ItemSpecifics LIKE %s
                    ORDER BY id
                    LIMIT {BATCH_SIZE}
                    """,
                    (last_id, "%arallel%"),
                )
                rows = cursor.fetchall()
                if not rows:
                    logger.info("Reached end of table after {:,} batches", batch_n)
                    break

                batch_n += 1
                scanned += len(rows)
                last_id  = rows[-1]["id"]

                for row in rows:
                    specs = parse_item_specifics(row.get("ItemSpecifics"))
                    p     = normalise_parallel(specs.get("parallel"))
                    if not p:
                        continue

                    brand  = (specs.get("brand",  "") or "").lower().strip() or "unknown"
                    genre  = (specs.get("genre",  "") or "").lower().strip() or "unknown"
                    source = (row.get("source_feed") or "unknown").lower().strip()
                    title  = (row.get("title") or "")[:200]

                    parallel_counts[p] += 1
                    brand_counts[p][brand]   += 1
                    genre_counts[p][genre]   += 1
                    source_counts[p][source] += 1
                    matched += 1

                    if len(title_samples[p]) < 5 and title:
                        title_samples[p].append(title)

                elapsed = time.perf_counter() - t_start
                rate    = matched / elapsed if elapsed > 0 else 0
                logger.info(
                    "Batch {:>4,} | id {:>12,} | scanned {:>10,} | matched {:>9,} | "
                    "unique {:>7,} | {:.0f} rows/s",
                    batch_n, last_id, scanned, matched,
                    len(parallel_counts), rate,
                )
    finally:
        conn.close()

    logger.info(
        "Scan complete — {:,} matching rows scanned, {:,} unique parallel values found",
        matched, len(parallel_counts),
    )

    # Build final output, filtered to min_count, ordered by frequency
    db: dict = {}
    for p, count in parallel_counts.most_common():
        if count < min_count:
            continue
        db[p] = {
            "count":   count,
            "brands":  dict(brand_counts[p].most_common(15)),
            "genres":  dict(genre_counts[p].most_common(10)),
            "sources": dict(source_counts[p].most_common(10)),
            "tokens":  tokenise(p),
            "samples": title_samples.get(p, []),
        }

    return db


# ── Token index ──────────────────────────────────────────────────────────────────

def build_token_index(db: dict) -> dict[str, list[str]]:
    """
    Build an inverted index: token → [parallel values that contain that token].
    Parallel values within each token list are ordered by frequency (highest first)
    so the most common match is tried first during title scanning.
    """
    index: dict[str, list[str]] = defaultdict(list)

    # Sort by count descending so most-common parallels appear first in token lists
    sorted_pairs = sorted(db.items(), key=lambda x: x[1]["count"], reverse=True)

    for p, _ in sorted_pairs:
        for token in db[p]["tokens"]:
            if p not in index[token]:
                index[token].append(p)

    return dict(index)


# ── Output writers ───────────────────────────────────────────────────────────────

def write_parquet(db: dict, path: Path) -> None:
    """Write frequency table as a Snappy-compressed Parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not installed — skipping parquet output")
        return

    if not db:
        logger.warning("Empty db — skipping parquet output")
        return

    rows = [
        {
            "parallel":    p,
            "count":       data["count"],
            "top_brand":   max(data["brands"], key=data["brands"].get)  if data["brands"]  else "",
            "top_genre":   max(data["genres"], key=data["genres"].get)  if data["genres"]  else "",
            "top_source":  max(data["sources"], key=data["sources"].get) if data["sources"] else "",
            "n_tokens":    len(data["tokens"]),
            "tokens_str":  " ".join(data["tokens"]),
        }
        for p, data in db.items()
    ]

    table = pa.table({
        "parallel":   pa.array([r["parallel"]   for r in rows], type=pa.string()),
        "count":      pa.array([r["count"]      for r in rows], type=pa.int64()),
        "top_brand":  pa.array([r["top_brand"]  for r in rows], type=pa.string()),
        "top_genre":  pa.array([r["top_genre"]  for r in rows], type=pa.string()),
        "top_source": pa.array([r["top_source"] for r in rows], type=pa.string()),
        "n_tokens":   pa.array([r["n_tokens"]   for r in rows], type=pa.int32()),
        "tokens_str": pa.array([r["tokens_str"] for r in rows], type=pa.string()),
    })

    pq.write_table(table, str(path), compression="snappy")
    logger.info("Parquet written  → {} ({:,} rows)", path.name, len(rows))


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build parallel variant dictionary from RDS")
    ap.add_argument(
        "--max-rows", type=int, default=DEFAULT_MAX_ROWS,
        help=f"Max parallel-containing rows to scan (default {DEFAULT_MAX_ROWS:,})",
    )
    ap.add_argument(
        "--min-count", type=int, default=DEFAULT_MIN_COUNT,
        help=f"Minimum occurrences to include in output (default {DEFAULT_MIN_COUNT})",
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    DATA_DIR.mkdir(exist_ok=True)

    logger.info(
        "Starting parallel DB build — max_rows={:,}  min_count={}",
        args.max_rows, args.min_count,
    )
    t0 = time.perf_counter()

    db = build_parallel_db(args.max_rows, args.min_count)

    if not db:
        logger.error("No parallel values found — check RDS connection and table contents")
        sys.exit(1)

    logger.info(
        "{:,} unique parallel values with ≥{} occurrences",
        len(db), args.min_count,
    )

    token_index = build_token_index(db)
    logger.info("{:,} unique tokens in inverted index", len(token_index))

    # File paths
    json_path    = DATA_DIR / "parallel_db.json"
    tokens_path  = DATA_DIR / "parallel_tokens.json"
    parquet_path = DATA_DIR / "parallel_db.parquet"
    meta_path    = DATA_DIR / "parallel_db_meta.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    logger.info("JSON written     → {} ({:,} entries)", json_path.name, len(db))

    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump(token_index, f, ensure_ascii=False, indent=2)
    logger.info("Token index      → {} ({:,} tokens)", tokens_path.name, len(token_index))

    write_parquet(db, parquet_path)

    elapsed = round(time.perf_counter() - t0, 1)

    top_10 = [
        {"parallel": p, "count": db[p]["count"], "top_genre": max(db[p]["genres"], key=db[p]["genres"].get) if db[p]["genres"] else ""}
        for p in list(db)[:10]
    ]

    meta = {
        "built_at":         datetime.now(timezone.utc).isoformat(),
        "max_rows_scanned": args.max_rows,
        "min_count":        args.min_count,
        "unique_parallels": len(db),
        "unique_tokens":    len(token_index),
        "top_10":           top_10,
        "elapsed_s":        elapsed,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata         → {}", meta_path.name)

    logger.info("─" * 60)
    logger.info("Top 10 parallel values by frequency:")
    for entry in top_10:
        logger.info("  {:>8,}  {}  ({})", entry["count"], entry["parallel"], entry["top_genre"])
    logger.info("─" * 60)
    logger.info("Done in {:.1f}s  →  data/", elapsed)


if __name__ == "__main__":
    main()
