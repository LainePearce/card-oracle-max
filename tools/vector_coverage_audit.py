#!/usr/bin/env python3
"""
Vector Coverage Audit
=====================
Compares rows in both RDS SQL databases against actual vectors in Qdrant
to identify coverage gaps by month and source.

Outputs:
  1. Per-source summary   — SQL rows vs Qdrant points per source_feed
  2. Monthly breakdown    — SQL rows per (month, source) with S3 checkpoint status
  3. Gap summary          — months/sources with < 95% estimated coverage
  4. Qdrant collection    — total counts, named vector presence, collection health

Usage:
    python tools/vector_coverage_audit.py
    python tools/vector_coverage_audit.py --start 2025-01-01 --end 2026-04-18
    python tools/vector_coverage_audit.py --json audit_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

# ── Config ────────────────────────────────────────────────────────────────────

AUDIT_START  = date(2025, 1, 1)
AUDIT_END    = date.today() + timedelta(days=1)   # exclusive, includes today
RDS_TABLE    = os.environ.get("RDS_TABLE", "salesdata")

# Source categories for report grouping
EBAY_SOURCES    = {"ebay", "EBAY"}
NONEBAY_SOURCES = {"PWCC", "FANATICS", "PRISTINE", "GOLDIN", "MYSLABS",
                   "HERITAGE", "CARDHOBBY", "REA", "VERISWAP"}


# ── Database helpers ──────────────────────────────────────────────────────────

def _connect_primary():
    import pymysql
    return pymysql.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", 3306)),
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        database=os.environ["RDS_DATABASE"],
        charset="utf8mb4",
        connect_timeout=30,
        read_timeout=600,
    )


def _connect_secondary():
    import pymysql
    if not os.environ.get("RDS2_HOST"):
        return None
    return pymysql.connect(
        host=os.environ["RDS2_HOST"],
        port=int(os.environ.get("RDS2_PORT", 3306)),
        user=os.environ["RDS2_USER"],
        password=os.environ["RDS2_PASSWORD"],
        database=os.environ["RDS2_DATABASE"],
        charset="utf8mb4",
        connect_timeout=30,
        read_timeout=600,
    )


def _month_range(start: date, end: date):
    """Yield (month_start, month_end) pairs covering [start, end)."""
    cur = date(start.year, start.month, 1)
    while cur < end:
        nxt = date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        yield max(cur, start), min(nxt, end)
        cur = nxt


def _daily_ranges(m_start: date, m_end: date):
    """Yield 1-day sub-ranges within [m_start, m_end) for chunked fallback.
    Daily chunks keep per-query row counts to ~300-500k even on peak months."""
    from datetime import timedelta
    cur = m_start
    while cur < m_end:
        nxt = min(cur + timedelta(days=1), m_end)
        yield cur, nxt
        cur = nxt


# ── Cleanup slot reconstruction (mirrors worker_cleanup.py) ──────────────────
# These constants + functions exactly replicate the slot-generation logic in
# worker_cleanup.py so we can map every calendar date to its cleanup slot ID
# without importing from the tools directory.

CLEANUP_BACKFILL_START = date(2025, 1, 1)
CLEANUP_BACKFILL_END   = date(2026, 4, 8)   # exclusive upper bound (matches worker_cleanup.py)
CLEANUP_N_WORKERS      = 12


def _cleanup_monthly_windows() -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cur_end = CLEANUP_BACKFILL_END
    while cur_end > CLEANUP_BACKFILL_START:
        last_day    = cur_end - timedelta(days=1)
        month_start = date(last_day.year, last_day.month, 1)
        win_start   = max(CLEANUP_BACKFILL_START, month_start)
        windows.append((win_start, cur_end))
        cur_end = month_start
    return windows


def _cleanup_split_range(start: date, end: date, n: int) -> list[tuple[date, date]]:
    total = (end - start).days
    base  = total // n
    extra = total % n
    slices: list[tuple[date, date]] = []
    cur = start
    for i in range(n):
        days = base + (1 if i < extra else 0)
        nxt  = cur + timedelta(days=days)
        slices.append((cur, min(nxt, end)))
        cur = nxt
    return slices


def _build_cleanup_tasks() -> list[tuple[date, date, str]]:
    tasks: list[tuple[date, date, str]] = []
    for win_start, win_end in _cleanup_monthly_windows():
        month_tag = win_start.strftime("%Y%m")
        slices    = _cleanup_split_range(win_start, win_end, CLEANUP_N_WORKERS)
        for w_idx, (s, e) in enumerate(slices):
            if (e - s).days == 0:
                continue
            cid = f"m{month_tag}w{w_idx}"
            tasks.append((s, e, cid))
    return tasks


_CLEANUP_TASKS: list[tuple[date, date, str]] = _build_cleanup_tasks()


def build_date_to_slot() -> dict[str, str]:
    """Map every calendar date in the cleanup backfill range to its slot ID."""
    d2s: dict[str, str] = {}
    for s, e, cid in _CLEANUP_TASKS:
        cur = s
        while cur < e:
            d2s[str(cur)] = cid
            cur += timedelta(days=1)
    return d2s


def build_slot_ranges() -> dict[str, tuple[date, date]]:
    """Map slot ID → (start, end) date range."""
    return {cid: (s, e) for s, e, cid in _CLEANUP_TASKS}


def _has_source_feed(conn) -> bool:
    """Check whether the salesdata table has a source_feed column."""
    import pymysql.cursors
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(f"SELECT source_feed FROM {RDS_TABLE} LIMIT 1")
            cur.fetchall()
        return True
    except Exception:
        return False


def query_rds_counts(conn, label: str, start: date, end: date) -> dict:
    """
    Query monthly + source breakdown of rows, one calendar month at a time.
    Per-month queries avoid the long-running GROUP BY timeout on large tables.
    Returns {(YYYY-MM, source_feed): {total, with_image, with_specifics}} dict.
    """
    import pymysql.cursors

    has_sf = _has_source_feed(conn)
    logger.info("[{}] source_feed column present: {}", label, has_sf)

    result: dict[tuple, dict] = {}
    total_buckets = 0

    for m_start, m_end in _month_range(start, end):
        s_str = m_start.strftime("%Y-%m-%d")
        e_str = m_end.strftime("%Y-%m-%d")
        month = m_start.strftime("%Y-%m")

        if has_sf:
            sql = f"""
                SELECT
                    UPPER(COALESCE(source_feed, 'UNKNOWN')) AS source_feed,
                    COUNT(*)                                AS total_rows,
                    SUM(CASE WHEN galleryURL IS NOT NULL AND galleryURL != ''
                             THEN 1 ELSE 0 END)             AS rows_with_image,
                    SUM(CASE WHEN ItemSpecifics IS NOT NULL
                                  AND ItemSpecifics != ''
                                  AND ItemSpecifics != 'NULL'
                             THEN 1 ELSE 0 END)             AS rows_with_specifics
                FROM {RDS_TABLE}
                WHERE endTime >= '{s_str} 00:00:00'
                  AND endTime  < '{e_str} 00:00:00'
                GROUP BY source_feed
            """
        else:
            # Old schema without source_feed — count all rows as UNKNOWN
            sql = f"""
                SELECT
                    'UNKNOWN'                               AS source_feed,
                    COUNT(*)                                AS total_rows,
                    SUM(CASE WHEN galleryURL IS NOT NULL AND galleryURL != ''
                             THEN 1 ELSE 0 END)             AS rows_with_image,
                    0                                       AS rows_with_specifics
                FROM {RDS_TABLE}
                WHERE endTime >= '{s_str} 00:00:00'
                  AND endTime  < '{e_str} 00:00:00'
            """

        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                t0 = time.time()
                cur.execute(sql)
                rows = cur.fetchall()
                elapsed = time.time() - t0

            for r in rows:
                key = (month, r["source_feed"])
                result[key] = {
                    "total_rows":          int(r["total_rows"] or 0),
                    "rows_with_image":     int(r["rows_with_image"] or 0),
                    "rows_with_specifics": int(r["rows_with_specifics"] or 0),
                }
                total_buckets += 1

            month_total = sum(r["total_rows"] for r in rows)
            logger.info("[{}] {} → {:,} rows in {:.1f}s", label, month, month_total, elapsed)

        except Exception as e:
            logger.warning("[{}] {} full-month query failed ({}); retrying in daily chunks…", label, month, e)
            conn.ping(reconnect=True)
            # ── Weekly fallback ─────────────────────────────────────────────
            day_buckets: dict[str, dict] = {}   # source_feed → aggregated counts
            any_week_ok = False
            for w_start, w_end in _daily_ranges(m_start, m_end):
                ws = w_start.strftime("%Y-%m-%d")
                we = w_end.strftime("%Y-%m-%d")
                if has_sf:
                    wsql = f"""
                        SELECT
                            UPPER(COALESCE(source_feed, 'UNKNOWN')) AS source_feed,
                            COUNT(*)                                AS total_rows,
                            SUM(CASE WHEN galleryURL IS NOT NULL AND galleryURL != ''
                                     THEN 1 ELSE 0 END)             AS rows_with_image,
                            SUM(CASE WHEN ItemSpecifics IS NOT NULL
                                          AND ItemSpecifics != ''
                                          AND ItemSpecifics != 'NULL'
                                     THEN 1 ELSE 0 END)             AS rows_with_specifics
                        FROM {RDS_TABLE}
                        WHERE endTime >= '{ws} 00:00:00'
                          AND endTime  < '{we} 00:00:00'
                        GROUP BY source_feed
                    """
                else:
                    wsql = f"""
                        SELECT 'UNKNOWN' AS source_feed,
                               COUNT(*) AS total_rows,
                               SUM(CASE WHEN galleryURL IS NOT NULL AND galleryURL != ''
                                        THEN 1 ELSE 0 END) AS rows_with_image,
                               0 AS rows_with_specifics
                        FROM {RDS_TABLE}
                        WHERE endTime >= '{ws} 00:00:00'
                          AND endTime  < '{we} 00:00:00'
                    """
                try:
                    with conn.cursor(pymysql.cursors.DictCursor) as cur:
                        wt0 = time.time()
                        cur.execute(wsql)
                        wrows = cur.fetchall()
                        welapsed = time.time() - wt0
                    for r in wrows:
                        sf = r["source_feed"]
                        if sf not in day_buckets:
                            day_buckets[sf] = {"total_rows": 0, "rows_with_image": 0, "rows_with_specifics": 0}
                        day_buckets[sf]["total_rows"]          += int(r["total_rows"] or 0)
                        day_buckets[sf]["rows_with_image"]     += int(r["rows_with_image"] or 0)
                        day_buckets[sf]["rows_with_specifics"] += int(r["rows_with_specifics"] or 0)
                    week_total = sum(r["total_rows"] for r in wrows)
                    logger.info("[{}]   {} day {} → {:,} rows in {:.1f}s",
                                label, month, ws, week_total, welapsed)
                    any_week_ok = True
                except Exception as we_exc:
                    logger.warning("[{}]   {} day {} failed: {}", label, month, ws, we_exc)
                    conn.ping(reconnect=True)

            if any_week_ok:
                for sf, counts in day_buckets.items():
                    key = (month, sf)
                    result[key] = counts
                    total_buckets += 1
                month_grand = sum(c["total_rows"] for c in day_buckets.values())
                logger.info("[{}] {} → {:,} rows (reassembled from daily chunks)", label, month, month_grand)
            else:
                logger.error("[{}] {} → ALL daily chunks failed; month excluded from report", label, month)

    logger.info("[{}] Complete — {} month×source buckets", label, total_buckets)
    return result


def query_rds_total_by_source(conn, label: str, start: date, end: date) -> dict:
    """
    Aggregate monthly counts into a per-source total dict.
    Derived from query_rds_counts() so no extra DB queries needed.
    """
    monthly = query_rds_counts.__wrapped_monthly if hasattr(query_rds_counts, "__wrapped_monthly") else None
    # Just aggregate from what we already have — caller passes monthly data
    return {}  # populated externally from monthly data


# ── Daily RDS query (EBAY only) ──────────────────────────────────────────────

def _connect_primary_short():
    """RDS connection with a 30-second read timeout for per-day COUNT queries.
    Each query touches ~300-600k rows via the endTime index; 30s is generous."""
    import pymysql
    return pymysql.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", 3306)),
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        database=os.environ["RDS_DATABASE"],
        charset="utf8mb4",
        connect_timeout=15,
        read_timeout=30,   # kill any single query that takes longer than 30s
    )


def query_rds_daily_ebay(start: date, end: date) -> dict[str, int]:
    """
    Query primary RDS for EBAY-only row counts per calendar day in [start, end).
    Opens a fresh short-timeout connection for each day so a hanging query
    never blocks progress for more than 30 seconds.
    Returns {YYYY-MM-DD: row_count}; -1 sentinel on query failure.
    """
    import pymysql.cursors

    result: dict[str, int] = {}
    cur = start
    total_days = (end - start).days

    logger.info("Querying RDS eBay daily counts for {} days ({} → {})…",
                total_days, start, end - timedelta(days=1))

    while cur < end:
        nxt  = cur + timedelta(days=1)
        ds   = str(cur)
        ds_n = str(nxt)

        sql = (
            f"SELECT COUNT(*) AS cnt FROM {RDS_TABLE} "
            f"WHERE endTime >= '{ds} 00:00:00' AND endTime < '{ds_n} 00:00:00' "
            f"AND UPPER(COALESCE(source_feed, 'EBAY')) = 'EBAY'"
        )

        count = -1
        for attempt in range(3):
            conn = None
            try:
                t0   = time.time()
                conn = _connect_primary_short()
                with conn.cursor(pymysql.cursors.DictCursor) as c:
                    c.execute(sql)
                    row = c.fetchone()
                count = int(row["cnt"] or 0)
                elapsed = time.time() - t0
                logger.info("  {} → {:,} rows  ({:.1f}s)", ds, count, elapsed)
                break
            except Exception as exc:
                logger.warning("  {} attempt {}/3 failed: {}", ds, attempt + 1, exc)
                if attempt == 2:
                    logger.error("  {} — giving up, marking as error", ds)
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

        result[ds] = count
        done = len(result)
        if done % 30 == 0:
            logger.info("  ── {}/{} days done ──", done, total_days)

        cur = nxt

    logger.info("Daily RDS query complete — {} dates", len(result))
    return result


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def query_qdrant(collection: str) -> dict:
    """
    Query Qdrant for:
      - Total point count
      - Points per source (using payload filter + count)
      - Named vector presence (image vs specifics)
      - Collection info (vectors config, indexed count)
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    # Use the same client factory as the workers — ensures gRPC/TLS config matches
    from src.ingestion.qdrant_writer import get_qdrant_client

    logger.info("Connecting to Qdrant ({}:{}, grpc={})…",
                os.environ.get("QDRANT_HOST", "localhost"),
                os.environ.get("QDRANT_PORT", 6333),
                os.environ.get("QDRANT_USE_GRPC", "false"))
    # Use a long-timeout client for audit — exact counts on 15M+ points can take minutes.
    from qdrant_client import QdrantClient as _QC
    host    = os.environ.get("QDRANT_HOST", "localhost")
    port    = int(os.environ.get("QDRANT_PORT", 6333))
    api_key = os.environ.get("QDRANT_API_KEY") or None
    use_grpc = os.environ.get("QDRANT_USE_GRPC", "false").lower() in ("true", "1")
    client = _QC(url=f"http://{host}:{port}", api_key=api_key,
                 prefer_grpc=use_grpc, timeout=300)

    # ── Collection info ────────────────────────────────────────────
    info = client.get_collection(collection)
    total_points   = info.points_count
    indexed_points = info.indexed_vectors_count  # may differ during active indexing
    vectors_config = list(info.config.params.vectors_config.keys()) if \
        hasattr(info.config.params, "vectors_config") and info.config.params.vectors_config else []

    logger.info("Qdrant collection '{}': {:,} total points, {:,} indexed",
                collection, total_points, indexed_points or 0)

    # ── Counts by source using payload filter ──────────────────────
    # Map from RDS source_feed (uppercase) to Qdrant payload source (lowercase)
    sources_to_check = [
        "ebay", "ebay_uk", "ebay_us",
        "pwcc", "fanatics", "fanatics_collect",
        "pristine", "goldin", "myslabs", "heritage",
        "cardhobby", "rea", "veriswap",
        "na", "",
    ]

    # With payload indexes on source/type/has_image, exact counts are now fast (index lookup).
    source_counts: dict[str, int] = {}
    for src in sources_to_check:
        filt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=src))])
        try:
            cnt = client.count(collection_name=collection, count_filter=filt, exact=True)
            if cnt.count > 0:
                source_counts[src] = cnt.count
                logger.info("  source='{}': {:,} points", src, cnt.count)
        except Exception as e:
            logger.warning("Could not count source='{}': {}", src, e)

    # ── Count points with image vector vs specifics only ──────────
    try:
        has_image_cnt = client.count(
            collection_name=collection,
            count_filter=Filter(must=[FieldCondition(key="has_image", match=MatchValue(value=True))]),
            exact=True,
        ).count
        logger.info("  has_image=True: {:,} points", has_image_cnt)
    except Exception as e:
        logger.warning("Could not count has_image=True: {}", e)
        has_image_cnt = None

    try:
        no_image_cnt = client.count(
            collection_name=collection,
            count_filter=Filter(must=[FieldCondition(key="has_image", match=MatchValue(value=False))]),
            exact=True,
        ).count
    except Exception:
        no_image_cnt = None

    # ── Count by type (sold vs catalogue) ─────────────────────────
    type_counts: dict[str, int] = {}
    for t in ("sold", "catalogue"):
        try:
            filt = Filter(must=[FieldCondition(key="type", match=MatchValue(value=t))])
            cnt = client.count(collection_name=collection, count_filter=filt, exact=True)
            type_counts[t] = cnt.count
            logger.info("  type='{}': {:,} points", t, cnt.count)
        except Exception:
            pass

    return {
        "total_points":    total_points,
        "indexed_points":  indexed_points,
        "vectors_config":  vectors_config,
        "source_counts":   source_counts,
        "has_image_count": has_image_cnt,
        "no_image_count":  no_image_cnt,
        "type_counts":     type_counts,
    }


# ── S3 checkpoint helpers ─────────────────────────────────────────────────────

def fetch_full_checkpoint_sets(bucket: str, prefix: str) -> tuple[set, set]:
    """
    Scan S3 checkpoints once and return:
      complete_slots — set of slot IDs (e.g. "m202501w3") with a -complete.json marker
      daily_dates    — set of date strings (e.g. "2026-04-15") with a daily-complete marker
    """
    import boto3
    from botocore.exceptions import ClientError

    complete_slots: set[str] = set()
    daily_dates:    set[str] = set()

    if not bucket:
        logger.warning("S3_VECTOR_BUCKET not set — checkpoint data unavailable")
        return complete_slots, daily_dates

    s3        = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))
    paginator = s3.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
            for obj in page.get("Contents", []):
                fname = obj["Key"].split("/")[-1]
                if not fname.endswith("-complete.json"):
                    continue
                stem = fname[: -len("-complete.json")]

                # Cleanup slot:  m{YYYYMM}w{N}-complete.json
                if stem and stem[0] == "m" and len(stem) >= 8 and stem[1:7].isdigit():
                    complete_slots.add(stem)

                # Daily ingestion:  daily-YYYY-MM-DD-complete.json  (stem = "daily-YYYY-MM-DD")
                elif stem.startswith("daily-") and len(stem) == len("daily-2026-01-01"):
                    daily_dates.add(stem[6:])   # strip "daily-" prefix → YYYY-MM-DD

    except ClientError as e:
        logger.error("S3 error scanning checkpoints: {}", e)

    logger.info("S3: {} complete cleanup slots, {} daily-complete dates",
                len(complete_slots), len(daily_dates))
    return complete_slots, daily_dates


# ── Day-by-day audit ──────────────────────────────────────────────────────────

def run_daily_audit(args) -> None:
    """
    Day-by-day coverage audit.  For each calendar day in [start, end):
      - Query primary RDS for eBay-only row count
      - Check S3 for a completed cleanup slot covering that date
      - Check S3 for a daily-ingestion complete marker for that date
      - Flag the day as OK, GAP, NO_DATA, or ERROR
    """
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end) + timedelta(days=1)   # make exclusive

    # Build slot → date mappings (replicated from worker_cleanup.py)
    date_to_slot = build_date_to_slot()
    slot_ranges  = build_slot_ranges()

    # S3 checkpoint scan
    bucket = os.environ.get("S3_VECTOR_BUCKET", "")
    ckpt   = os.environ.get("S3_CHECKPOINT_PREFIX", "checkpoints")
    if not args.skip_s3:
        complete_slots, daily_dates = fetch_full_checkpoint_sets(bucket, ckpt)
    else:
        logger.info("Skipping S3 checkpoint scan (--skip-s3)")
        complete_slots, daily_dates = set(), set()

    # RDS daily counts — opens one fresh 30s-timeout connection per day internally
    rds_daily: dict[str, int] = {}
    if not args.skip_primary:
        logger.info("Primary RDS: {} (30s per-query timeout)", os.environ.get("RDS_HOST", "?"))
        try:
            rds_daily = query_rds_daily_ebay(start, end)
        except Exception as exc:
            logger.error("Daily RDS query loop failed: {}", exc)
    else:
        logger.info("Skipping primary RDS (--skip-primary)")

    # ── Build per-day result rows ─────────────────────────────────────────────
    rows = []
    cur  = start
    while cur < end:
        ds        = str(cur)
        rds_count = rds_daily.get(ds, 0)
        slot_id   = date_to_slot.get(ds)          # None for dates outside cleanup range
        cleanup_ok = slot_id in complete_slots if slot_id else False
        daily_ok   = ds in daily_dates
        covered    = cleanup_ok or daily_ok

        if rds_count == -1:
            status = "ERROR"
        elif rds_count == 0:
            status = "NO_DATA"
        elif covered:
            status = "OK"
        else:
            status = "GAP"

        rows.append({
            "date":       ds,
            "rds_rows":   rds_count,
            "slot_id":    slot_id or "—",
            "cleanup_ok": cleanup_ok,
            "daily_ok":   daily_ok,
            "covered":    covered,
            "status":     status,
        })
        cur += timedelta(days=1)

    # ── Print table ───────────────────────────────────────────────────────────
    W = 88
    print()
    print("═" * W)
    print(f"  DAY-BY-DAY COVERAGE AUDIT  —  {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Range: {start} → {end - timedelta(days=1)}   |   RDS: eBay rows only")
    print("═" * W)
    print(f"  {'Date':<12} {'RDS eBay':>10}  {'Slot':<14} {'Cleanup':>9} {'Daily':>7}  Status")
    print("─" * W)

    gap_rows = []
    for r in rows:
        cleanup_flag = "✓" if r["cleanup_ok"] else "✗"
        daily_flag   = "✓" if r["daily_ok"]   else "—"
        rds_fmt      = f"{r['rds_rows']:,}" if r["rds_rows"] >= 0 else "ERROR"
        suffix       = "  ⚠ GAP" if r["status"] == "GAP" else \
                       "  ✗ ERR" if r["status"] == "ERROR" else ""
        if r["status"] == "GAP":
            gap_rows.append(r)
        print(f"  {r['date']:<12} {rds_fmt:>10}  {r['slot_id']:<14} "
              f"{cleanup_flag:>9} {daily_flag:>7}  {r['status']}{suffix}")

    print("─" * W)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_count      = sum(1 for r in rows if r["status"] == "OK")
    gap_count     = len(gap_rows)
    no_data_count = sum(1 for r in rows if r["status"] == "NO_DATA")
    error_count   = sum(1 for r in rows if r["status"] == "ERROR")
    total_gap_rows = sum(r["rds_rows"] for r in gap_rows if r["rds_rows"] > 0)

    print(f"\n  SUMMARY   ({len(rows)} days audited)")
    print(f"  ✓ Covered:             {ok_count:>5}")
    print(f"  ⚠ GAP (needs work):   {gap_count:>5}   ({total_gap_rows:,} total eBay rows unaccounted)")
    print(f"  — No eBay data:        {no_data_count:>5}")
    print(f"  ✗ Query errors:        {error_count:>5}")

    if gap_rows:
        print(f"\n  ── GAPS REQUIRING ATTENTION ──────────────────────────────────────────────")
        print(f"  {'Date':<12} {'RDS Rows':>10}  {'Cleanup Slot':<15} Slot Range")
        print("  " + "─" * 64)
        for r in gap_rows:
            sr        = slot_ranges.get(r["slot_id"])
            slot_info = f"{sr[0]}→{sr[1]}" if sr else "(outside cleanup range)"
            print(f"  {r['date']:<12} {r['rds_rows']:>10,}  {r['slot_id']:<15} {slot_info}")
    else:
        print(f"\n  ✓ No gaps detected — all days with eBay data are covered.")

    print("═" * W)
    print()

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.json:
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "audit_range":  {"start": str(start), "end": str(end - timedelta(days=1))},
            "mode":         "daily",
            "summary": {
                "total_days":    len(rows),
                "covered_days":  ok_count,
                "gap_days":      gap_count,
                "gap_rds_rows":  total_gap_rows,
                "no_data_days":  no_data_count,
                "error_days":    error_count,
            },
            "days": rows,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Daily audit JSON written to {}", args.json)


def fetch_s3_checkpoint_summary(bucket: str, prefix: str) -> dict:
    """
    Fetch a summary of S3 checkpoint markers to determine which months are confirmed complete.
    Returns {month_YYYYMM: {ebay_complete_slices, cleanup_complete_slices, nonebay_complete_slices}}.
    """
    import boto3
    from botocore.exceptions import ClientError

    if not bucket:
        logger.warning("S3_VECTOR_BUCKET not set — skipping S3 checkpoint check")
        return {}

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))
    logger.info("Scanning S3 checkpoints in s3://{}/{}/…", bucket, prefix)

    # Count complete markers per task type per month
    ebay_phase_complete:   dict[str, int] = defaultdict(int)  # month → count
    cleanup_complete:      dict[str, int] = defaultdict(int)
    nonebay_complete:      dict[str, int] = defaultdict(int)
    gap_complete_count = 0
    daily_complete:        set[str]       = set()

    paginator = s3.get_paginator("list_objects_v2")
    total_keys = 0
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
            for obj in page.get("Contents", []):
                key   = obj["Key"]
                fname = key.split("/")[-1]
                total_keys += 1

                if not fname.endswith("-complete.json"):
                    continue

                # eBay phase: backfill-w{N}-phase{P}-complete.json
                if fname.startswith("backfill-w") and "-phase" in fname:
                    ebay_phase_complete["all"] = ebay_phase_complete["all"] + 1

                # Cleanup: m{YYYYMM}w{N}-complete.json  (no prefix)
                elif fname[0] == "m" and len(fname) > 10 and fname[1:7].isdigit():
                    month = fname[1:7]   # YYYYMM
                    cleanup_complete[month] += 1

                # Non-eBay: nonebay-m{YYYYMM}w{N}-complete.json
                elif fname.startswith("nonebay-m"):
                    month = fname[9:15]  # YYYYMM after "nonebay-m"
                    nonebay_complete[month] += 1

                # Gap fill: gap-*-complete.json
                elif fname.startswith("gap-"):
                    gap_complete_count += 1

                # Daily: daily-{YYYY-MM-DD}-complete.json
                elif fname.startswith("daily-") and len(fname) == len("daily-2026-01-01-complete.json"):
                    day = fname[6:16]  # YYYY-MM-DD
                    daily_complete.add(day)

    except ClientError as e:
        logger.error("S3 error: {}", e)
        return {}

    logger.info("S3 scan: {:,} total keys, {} daily complete, {} cleanup months, {} nonebay months",
                total_keys, len(daily_complete), len(cleanup_complete), len(nonebay_complete))

    return {
        "ebay_phase_markers":   dict(ebay_phase_complete),
        "cleanup_complete":     dict(cleanup_complete),
        "nonebay_complete":     dict(nonebay_complete),
        "gap_complete_count":   gap_complete_count,
        "daily_complete_count": len(daily_complete),
        "daily_complete_dates": sorted(daily_complete)[-10:],  # last 10 for report
        "total_checkpoint_keys": total_keys,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(
    primary_monthly:   dict,
    secondary_monthly: dict | None,
    primary_totals:    dict,
    secondary_totals:  dict | None,
    qdrant:            dict,
    s3_checkpoints:    dict,
    start:             date,
    end:               date,
) -> dict:
    """Assemble the full audit report."""

    # Merge primary + secondary monthly counts (secondary fills gaps)
    merged_monthly: dict[tuple, dict] = {}
    all_keys = set(primary_monthly.keys())
    if secondary_monthly:
        all_keys |= set(secondary_monthly.keys())

    for key in sorted(all_keys):
        p = primary_monthly.get(key, {"total_rows": 0, "rows_with_image": 0, "rows_with_specifics": 0})
        s = (secondary_monthly or {}).get(key, {"total_rows": 0, "rows_with_image": 0, "rows_with_specifics": 0})
        # Primary wins; secondary only adds rows not in primary (gap fill)
        # For counting purposes, use max of the two (conservative estimate)
        merged_monthly[key] = {
            "primary_rows":      p["total_rows"],
            "secondary_rows":    s["total_rows"],
            "combined_rows":     max(p["total_rows"], s["total_rows"]),
            "rows_with_image":   max(p["rows_with_image"], s["rows_with_image"]),
            "rows_with_specifics": max(p["rows_with_specifics"], s["rows_with_specifics"]),
        }

    # Merge primary + secondary totals by source
    all_sources = set(primary_totals.keys())
    if secondary_totals:
        all_sources |= set(secondary_totals.keys())

    merged_sources: dict[str, dict] = {}
    for src in all_sources:
        p = primary_totals.get(src, {"total_rows": 0, "rows_with_image": 0})
        s = (secondary_totals or {}).get(src, {"total_rows": 0, "rows_with_image": 0})
        merged_sources[src] = {
            "primary_rows":   p["total_rows"],
            "secondary_rows": s["total_rows"],
            "combined_rows":  max(p["total_rows"], s["total_rows"]),
            "rows_with_image": max(p["rows_with_image"], s["rows_with_image"]),
        }

    # ── Qdrant source mapping (RDS uppercase → Qdrant lowercase variants) ──
    # Map RDS source names to Qdrant payload source counts
    rds_to_qdrant_source: dict[str, list[str]] = {
        "EBAY":      ["ebay", "ebay_uk", "ebay_us"],
        "PWCC":      ["pwcc"],
        "FANATICS":  ["fanatics", "fanatics_collect"],
        "PRISTINE":  ["pristine"],
        "GOLDIN":    ["goldin"],
        "MYSLABS":   ["myslabs"],
        "HERITAGE":  ["heritage"],
        "CARDHOBBY": ["cardhobby"],
        "REA":       ["rea"],
        "VERISWAP":  ["veriswap"],
        "UNKNOWN":   ["na", ""],
    }

    qdrant_src_counts = qdrant.get("source_counts", {})

    source_summary = []
    for rds_src in sorted(merged_sources.keys()):
        qdrant_keys = rds_to_qdrant_source.get(rds_src, [rds_src.lower()])
        q_count = sum(qdrant_src_counts.get(k, 0) for k in qdrant_keys)
        d = merged_sources[rds_src]
        sql_count = d["combined_rows"]
        pct = round(q_count / sql_count * 100, 1) if sql_count > 0 else 0.0
        source_summary.append({
            "source":          rds_src,
            "sql_rows":        sql_count,
            "sql_with_image":  d["rows_with_image"],
            "qdrant_points":   q_count,
            "coverage_pct":    pct,
            "gap":             max(0, sql_count - q_count),
        })

    source_summary.sort(key=lambda x: -x["sql_rows"])

    # ── Monthly breakdown ─────────────────────────────────────────
    monthly_breakdown = []
    cleanup_complete  = s3_checkpoints.get("cleanup_complete", {})
    nonebay_complete  = s3_checkpoints.get("nonebay_complete", {})

    # Group merged monthly by month (sum across sources)
    monthly_totals: dict[str, dict] = defaultdict(lambda: {
        "total_rows": 0, "rows_with_image": 0, "ebay_rows": 0, "nonebay_rows": 0
    })
    for (month, src), counts in merged_monthly.items():
        m = monthly_totals[month]
        m["total_rows"]    += counts["combined_rows"]
        m["rows_with_image"] += counts["rows_with_image"]
        if src.upper() in ("EBAY",) or src.upper().startswith("EBAY"):
            m["ebay_rows"] += counts["combined_rows"]
        else:
            m["nonebay_rows"] += counts["combined_rows"]

    for month in sorted(monthly_totals.keys()):
        m    = monthly_totals[month]
        yyyymm = month.replace("-", "")

        cleanup_slices  = cleanup_complete.get(yyyymm, 0)
        nonebay_slices  = nonebay_complete.get(yyyymm, 0)
        # Max possible slices per month = 12 workers for cleanup, 6 for non-eBay
        ebay_status   = "✓ complete" if cleanup_slices >= 12 else \
                        f"partial ({cleanup_slices}/12)" if cleanup_slices > 0 else "pending"
        nonebay_status = "✓ complete" if nonebay_slices >= 6 else \
                         f"partial ({nonebay_slices}/6)" if nonebay_slices > 0 else "pending"

        monthly_breakdown.append({
            "month":           month,
            "total_sql_rows":  m["total_rows"],
            "ebay_rows":       m["ebay_rows"],
            "nonebay_rows":    m["nonebay_rows"],
            "rows_with_image": m["rows_with_image"],
            "ebay_checkpoint": ebay_status,
            "nonebay_checkpoint": nonebay_status,
            "cleanup_slices":  cleanup_slices,
            "nonebay_slices":  nonebay_slices,
        })

    # ── Overall totals ─────────────────────────────────────────────
    total_sql_rows     = sum(d["combined_rows"] for d in merged_sources.values())
    total_sql_image    = sum(d["rows_with_image"] for d in merged_sources.values())
    total_qdrant       = qdrant.get("total_points", 0)
    overall_coverage   = round(total_qdrant / total_sql_rows * 100, 1) if total_sql_rows > 0 else 0.0

    # ── Gaps ──────────────────────────────────────────────────────
    gaps = [s for s in source_summary if s["coverage_pct"] < 95 and s["sql_rows"] > 1000]

    return {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "audit_range":    {"start": str(start), "end": str(end - timedelta(days=1))},
        "overall": {
            "total_sql_rows":        total_sql_rows,
            "total_sql_with_image":  total_sql_image,
            "total_qdrant_points":   total_qdrant,
            "overall_coverage_pct":  overall_coverage,
            "total_gap":             max(0, total_sql_rows - total_qdrant),
        },
        "qdrant_collection": {
            "total_points":    qdrant.get("total_points"),
            "indexed_points":  qdrant.get("indexed_points"),
            "vectors_config":  qdrant.get("vectors_config"),
            "has_image_count": qdrant.get("has_image_count"),
            "no_image_count":  qdrant.get("no_image_count"),
            "type_counts":     qdrant.get("type_counts"),
        },
        "source_summary":    source_summary,
        "monthly_breakdown": monthly_breakdown,
        "s3_checkpoints":    s3_checkpoints,
        "gaps":              gaps,
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    W = 80
    def sep(c="─"): print(c * W)
    def hdr(t):
        print(f"\n{'─'*3} {t} {'─'*(W-5-len(t))}")

    print()
    sep("═")
    print(f"  VECTOR COVERAGE AUDIT  —  {report['generated_at'][:19].replace('T',' ')} UTC")
    print(f"  Range: {report['audit_range']['start']} → {report['audit_range']['end']}")
    sep("═")

    # Overall
    o = report["overall"]
    hdr("OVERALL SUMMARY")
    print(f"  SQL rows (combined DBs):  {o['total_sql_rows']:>14,}")
    print(f"  SQL rows with image URL:  {o['total_sql_with_image']:>14,}")
    print(f"  Qdrant points:            {o['total_qdrant_points']:>14,}")
    print(f"  Overall coverage:         {o['overall_coverage_pct']:>13.1f}%")
    print(f"  Estimated gap:            {o['total_gap']:>14,}")

    # Qdrant collection
    hdr("QDRANT COLLECTION")
    q = report["qdrant_collection"]
    print(f"  Named vectors:     {q['vectors_config']}")
    print(f"  Has image:         {(q['has_image_count'] or 0):>12,}")
    print(f"  No image (text):   {(q['no_image_count'] or 0):>12,}")
    print(f"  By type:           {q['type_counts']}")

    # Source summary
    hdr("COVERAGE BY SOURCE")
    print(f"  {'Source':<16} {'SQL Rows':>12} {'With Image':>11} {'Qdrant':>12} {'Coverage':>9} {'Gap':>10}")
    sep()
    for s in report["source_summary"]:
        flag = "⚠️ " if s["coverage_pct"] < 80 else ("△ " if s["coverage_pct"] < 95 else "  ")
        print(f"  {flag}{s['source']:<14} {s['sql_rows']:>12,} {s['sql_with_image']:>11,} "
              f"{s['qdrant_points']:>12,} {s['coverage_pct']:>8.1f}% {s['gap']:>10,}")

    # Monthly breakdown
    hdr("MONTHLY BREAKDOWN (SQL rows + S3 checkpoint status)")
    print(f"  {'Month':<8} {'Total SQL':>10} {'eBay':>10} {'Non-eBay':>10}  {'eBay chk':<18} {'Non-eBay chk':<18}")
    sep()
    for m in report["monthly_breakdown"]:
        print(f"  {m['month']:<8} {m['total_sql_rows']:>10,} {m['ebay_rows']:>10,} "
              f"{m['nonebay_rows']:>10,}  {m['ebay_checkpoint']:<18} {m['nonebay_checkpoint']:<18}")

    # S3 checkpoint summary
    hdr("S3 CHECKPOINT SUMMARY")
    s3 = report.get("s3_checkpoints") or {}
    def _fmt_int(v):
        return f"{v:,}" if isinstance(v, int) else str(v) if v is not None else "—"
    print(f"  Total checkpoint keys:  {_fmt_int(s3.get('total_checkpoint_keys')):>8}")
    print(f"  Daily complete markers: {_fmt_int(s3.get('daily_complete_count')):>8}")
    print(f"  Gap-fill completions:   {_fmt_int(s3.get('gap_complete_count')):>8}")
    print(f"  Latest daily dates:     {', '.join(s3.get('daily_complete_dates', []) or []) or '—'}")

    # Gaps
    if report["gaps"]:
        hdr("⚠️  GAPS REQUIRING ATTENTION (>1000 SQL rows, <95% coverage)")
        for g in report["gaps"]:
            print(f"  {g['source']:<16}  {g['gap']:>10,} missing  ({g['coverage_pct']:.1f}% coverage)")
    else:
        hdr("✓ NO MAJOR GAPS DETECTED")
        print("  All sources with >1000 rows are at ≥95% coverage.")

    sep("═")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vector coverage audit")
    parser.add_argument("--start", default=str(AUDIT_START), help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=str(AUDIT_END - timedelta(days=1)), help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--json",  metavar="FILE", help="Also write full report as JSON to FILE")
    parser.add_argument("--daily", action="store_true",
                        help="Day-by-day audit: RDS eBay rows vs S3 checkpoint coverage per date")
    parser.add_argument("--skip-secondary", action="store_true", help="Skip secondary RDS query")
    parser.add_argument("--skip-s3",        action="store_true", help="Skip S3 checkpoint scan")
    parser.add_argument("--skip-primary",   action="store_true", help="Skip primary RDS query (use empty data)")
    parser.add_argument("--skip-qdrant",    action="store_true", help="Skip Qdrant query (use empty data)")
    args = parser.parse_args()

    # ── Daily mode: day-by-day coverage audit ─────────────────────────────────
    if args.daily:
        run_daily_audit(args)
        return

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end) + timedelta(days=1)  # make exclusive

    # ── Primary RDS ────────────────────────────────────────────────
    primary_monthly: dict = {}
    if not args.skip_primary:
        logger.info("Connecting to primary RDS ({})…", os.environ.get("RDS_HOST", "?"))
        try:
            primary_conn    = _connect_primary()
            primary_monthly = query_rds_counts(primary_conn, "primary", start, end)
            primary_conn.close()
        except Exception as e:
            logger.error("Primary RDS failed: {}", e)
    else:
        logger.info("Skipping primary RDS (--skip-primary)")

    # Derive per-source totals from monthly data (no extra query needed)
    primary_totals: dict[str, dict] = defaultdict(lambda: {"total_rows": 0, "rows_with_image": 0})
    for (month, src), v in primary_monthly.items():
        primary_totals[src]["total_rows"]      += v["total_rows"]
        primary_totals[src]["rows_with_image"] += v["rows_with_image"]
    primary_totals = dict(primary_totals)

    # ── Secondary RDS ──────────────────────────────────────────────
    secondary_monthly = None
    secondary_totals  = None
    if not args.skip_secondary and os.environ.get("RDS2_HOST"):
        logger.info("Connecting to secondary RDS ({})…", os.environ.get("RDS2_HOST", "?"))
        try:
            sec_conn          = _connect_secondary()
            secondary_monthly = query_rds_counts(sec_conn, "secondary", start, end)
            sec_conn.close()
            # Derive totals
            sec_totals_tmp: dict[str, dict] = defaultdict(lambda: {"total_rows": 0, "rows_with_image": 0})
            for (month, src), v in secondary_monthly.items():
                sec_totals_tmp[src]["total_rows"]      += v["total_rows"]
                sec_totals_tmp[src]["rows_with_image"] += v["rows_with_image"]
            secondary_totals = dict(sec_totals_tmp)
        except Exception as e:
            logger.warning("Secondary RDS failed (non-fatal): {}", e)
    elif not os.environ.get("RDS2_HOST"):
        logger.info("Secondary RDS not configured — skipping")

    # ── Qdrant ────────────────────────────────────────────────────
    collection = os.environ.get("QDRANT_COLLECTION", "cards")
    qdrant_data: dict = {"total_points": 0, "source_counts": {}}
    if not args.skip_qdrant:
        logger.info("Querying Qdrant collection '{}'…", collection)
        try:
            qdrant_data = query_qdrant(collection)
        except Exception as e:
            logger.error("Qdrant query failed: {}", e)
    else:
        logger.info("Skipping Qdrant query (--skip-qdrant)")

    # ── S3 checkpoints ────────────────────────────────────────────
    s3_checkpoints: dict = {}
    if not args.skip_s3:
        bucket = os.environ.get("S3_VECTOR_BUCKET", "")
        prefix = os.environ.get("S3_CHECKPOINT_PREFIX", "checkpoints")
        s3_checkpoints = fetch_s3_checkpoint_summary(bucket, prefix)

    # ── Build + print report ──────────────────────────────────────
    report = build_report(
        primary_monthly   = primary_monthly,
        secondary_monthly = secondary_monthly,
        primary_totals    = primary_totals,
        secondary_totals  = secondary_totals,
        qdrant            = qdrant_data,
        s3_checkpoints    = s3_checkpoints,
        start             = start,
        end               = end,
    )

    print_report(report)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Full JSON report written to {}", args.json)


if __name__ == "__main__":
    main()
