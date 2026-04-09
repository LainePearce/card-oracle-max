"""MySQL RDS reader for populating the new self-managed OpenSearch cluster.

Reads from the salesdata table, transforms flat MySQL rows into the
OpenSearch document format (nested itemSpecifics), and determines the
target index name based on source_feed + date.

The RDS instance is in a different VPC/region — connect via public endpoint.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Generator

import pymysql
import pymysql.cursors
from loguru import logger


# ── MySQL connection ─────────────────────────────────────────────

BATCH_SIZE = 1000  # rows per fetch from server-side cursor


def get_rds_connection() -> pymysql.Connection:
    """Connect to RDS MySQL using environment variables."""
    return pymysql.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", 3306)),
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        database=os.environ["RDS_DATABASE"],
        charset="utf8mb4",
        connect_timeout=30,
        read_timeout=300,
    )


# ── Source feed → index name mapping ─────────────────────────────

# Maps source_feed values from RDS to the marketplace suffix used in index names.
# eBay rows use the endTime date as the index name (YYYY-MM-DD).
# Non-eBay rows use the marketplace-specific pattern.

MARKETPLACE_INDEX_MAP = {
    "EBAY":       "ebay",         # → YYYY-MM-DD
    "GOLDIN":     "gold",         # → YYYY-gold
    "PWCC":       "pwcc",         # → YYYY-MM-pwcc
    "FANATICS":   "pwcc",         # → YYYY-MM-pwcc (same as PWCC)
    "PRISTINE":   "pris",         # → YYYY-MM-pris
    "MYSLABS":    "ms",           # → YYYY-ms
    "HERITAGE":   "heri",         # → YYYY-heri
    "CARDHOBBY":  "cardhobby",    # → YYYY-cardhobby
    "REA":        "rea",          # → YYYY-rea
    "VERISWAP":   "veriswap",     # → YYYY-veriswap
}


def determine_index_name(source_feed: str, end_time: datetime | str) -> str:
    """
    Determine the OpenSearch index name for a row based on its source and date.

    eBay: YYYY-MM-DD (daily indices)
    Monthly marketplaces (PWCC, Pristine): YYYY-MM-suffix
    Yearly marketplaces (Goldin, MySlabs, Heritage, etc.): YYYY-suffix
    """
    feed = (source_feed or "").strip().upper()
    suffix = MARKETPLACE_INDEX_MAP.get(feed)

    if suffix is None:
        logger.warning("Unknown source_feed '{}', defaulting to ebay pattern", source_feed)
        suffix = "ebay"

    # Parse date
    if isinstance(end_time, str):
        try:
            dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.strptime(end_time[:10], "%Y-%m-%d")
    elif isinstance(end_time, datetime):
        dt = end_time
    else:
        raise ValueError(f"Cannot parse endTime: {end_time}")

    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")

    if suffix == "ebay":
        return f"{year}-{month}-{day}"
    elif suffix in ("pwcc", "pris"):
        # Monthly indices: YYYY-MM-suffix
        return f"{year}-{month}-{suffix}"
    else:
        # Yearly indices: YYYY-suffix
        return f"{year}-{suffix}"


# ── ItemSpecifics parsing ────────────────────────────────────────

# The ItemSpecifics column in RDS is a mediumtext field.
# Expected format: JSON string like {"brand": "Topps", "player": "...", ...}
# Falls back to empty dict if unparseable.

# ── Raw eBay attribute name → normalized field name mapping ──────
# eBay ItemSpecifics come as {"Raw Name": ["value"]} where values are arrays.
# Multiple raw names can map to the same normalized field.

RAW_KEY_MAP: dict[str, str] = {
    # brand / manufacturer
    "Manufacturer":                     "brand",
    "Brand":                            "brand",
    # player
    "Player/Athlete":                   "player",
    "Player":                           "player",
    "Character":                        "player",
    # genre / sport
    "Sport":                            "genre",
    "Game":                             "genre",
    # country
    "Country/Region of Manufacture":    "country",
    "Country":                          "country",
    # set
    "Set":                              "set",
    # card number
    "Card Number":                      "cardNumber",
    # subset
    "Card Name":                        "subset",
    "Insert Set":                       "subset",
    # parallel
    "Parallel/Variety":                 "parallel",
    "Parallel":                         "parallel",
    # serial number
    "Serial Number":                    "serialNumber",
    # year
    "Year Manufactured":                "year",
    "Season":                           "year",
    "Year":                             "year",
    # graded
    "Graded":                           "graded",
    "Professional Grader":              "grader",
    "Grader":                           "grader",
    "Grade":                            "grade",
    "Certification Number":             "grade",
    # type
    "Type":                             "type",
    # autographed
    "Autographed":                      "autographed",
    "Autograph":                        "autographed",
    # team
    "Team":                             "team",
    # features (mapped to subset if no subset found)
    "Features":                         "_features",
    # league (not directly mapped but useful context)
    "League":                           "_league",
}

# Fields we output in the normalized itemSpecifics object
ITEM_SPECIFICS_FIELDS = [
    "brand", "player", "genre", "country", "set", "cardNumber",
    "subset", "parallel", "serialNumber", "year", "graded",
    "grader", "grade", "type", "autographed", "team",
]


def parse_item_specifics(raw: str | None) -> dict:
    """
    Parse the ItemSpecifics mediumtext column into a normalized dict.

    Supports two input formats:

    Array format (native RDS / eBay API response):
        [{"type":"STRING","name":"Player/Athlete","value":"Derek Jeter"}, ...]

    Dict format (legacy / pre-processed):
        {"Player/Athlete": ["Derek Jeter"], "Set": ["2023 Topps Finest"], ...}

    Output: {"player": "derek jeter", "set": "2023 topps finest", ...}
    """
    if not raw or raw.strip() in ("", "NULL", "null", "None"):
        return {}

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}

    # Handle array format: [{"type":"STRING","name":"Set","value":"..."}, ...]
    # This is the native RDS format from the eBay API response.
    if isinstance(parsed, list):
        parsed = {
            item["name"]: item["value"]
            for item in parsed
            if isinstance(item, dict) and "name" in item and item.get("value") not in (None, "")
        }
    elif not isinstance(parsed, dict):
        return {}

    # First pass: map raw keys to normalized field names
    mapped: dict[str, str] = {}
    features_val = None

    for raw_key, raw_val in parsed.items():
        # Extract scalar value from array
        if isinstance(raw_val, list):
            val = raw_val[0] if raw_val else None
        else:
            val = raw_val

        if val is None or str(val).strip() == "":
            continue

        val_str = str(val).strip()
        normalized_key = RAW_KEY_MAP.get(raw_key)

        if normalized_key is None:
            continue

        if normalized_key == "_features":
            features_val = val_str
            continue
        if normalized_key == "_league":
            continue

        # Don't overwrite if already set (first match wins)
        if normalized_key not in mapped:
            mapped[normalized_key] = val_str

    # Use Features as fallback for subset if subset is empty
    if "subset" not in mapped and features_val:
        mapped["subset"] = features_val

    # Second pass: type coercion
    result = {}
    for field in ITEM_SPECIFICS_FIELDS:
        val = mapped.get(field)
        if val is None:
            continue

        if field == "year":
            # Extract 4-digit year from strings like "2023"
            year_match = re.search(r"(19[5-9]\d|20[0-2]\d)", val)
            if year_match:
                result[field] = int(year_match.group(1))
        elif field in ("graded", "autographed"):
            result[field] = _to_bool(val)
        else:
            result[field] = val

    return result


def _to_bool(val) -> bool:
    """Convert various truthy representations to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).lower().strip()
    return s in ("true", "1", "yes", "y")


# ── Row → OpenSearch document transformation ─────────────────────

def transform_row(row: dict) -> tuple[str, str, dict]:
    """
    Transform a MySQL row into an OpenSearch document.

    Returns: (index_name, doc_id, document_body)

    The document structure matches the extant OpenSearch mapping
    (CLAUDE.md Section 18) minus vector fields.
    """
    source_feed = row.get("source_feed") or "EBAY"
    end_time = row.get("endTime")

    index_name = determine_index_name(source_feed, end_time)
    # OpenSearch uses itemId as _id (not the RDS integer id).
    # Must match so Qdrant os_id → OpenSearch mget enrichment works.
    doc_id = str(row.get("itemId") or row["id"])

    # Parse itemSpecifics from mediumtext
    item_specifics = parse_item_specifics(row.get("ItemSpecifics"))

    # Build document matching the OpenSearch mapping
    doc = {
        "id": row["id"],
        "itemId": str(row.get("itemId", "")),
        "source": source_feed.upper(),
        "globalId": row.get("globaId", ""),   # note: typo in RDS schema
        "title": row.get("title", ""),
        "galleryURL": row.get("galleryURL", ""),
        "itemURL": row.get("viewItemURL", ""),   # RDS: viewItemURL → OS: itemURL
        "saleType": row.get("saleType", ""),
        "currentPrice": _to_float(row.get("currentPrice")),
        "currentPriceCurrency": row.get("currentPriceCurrency", ""),
        "salePrice": _to_float(row.get("BestOfferPrice")),  # map to salePrice
        "salePriceCurrency": row.get("BestOfferCurrency", ""),
        "shippingServiceCost": _to_float(row.get("shippingServiceCost")),
        "bidCount": _to_int(row.get("bidCount")),
        "endTime": _format_datetime(end_time),
        "startTime": _format_datetime(row.get("startTime")),
        "BestOfferPrice": _to_float(row.get("BestOfferPrice")),
        "BestOfferCurrency": row.get("BestOfferCurrency", ""),
        "cloud": _to_int(row.get("cloud")),
        "itemSpecifics": item_specifics,
    }

    return index_name, doc_id, doc


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _format_datetime(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    return str(val)


# ── Scroll reader ────────────────────────────────────────────────

def scroll_rds(
    conn: pymysql.Connection,
    table: str = "salesdata",
    start_date: str | None = None,
    end_date: str | None = None,
    extra_where: str | None = None,
    batch_size: int = BATCH_SIZE,
    resume_date: str | None = None,
) -> Generator[dict, None, None]:
    """
    Stream rows from the RDS salesdata table using daily date bands.

    Issues one small query per calendar day (e.g.
    ``endTime >= '2025-01-01 00:00:00' AND endTime < '2025-01-02 00:00:00'``).
    Each daily query is fast because the result set is small (~50k–200k rows),
    avoiding the multi-minute MySQL sort that a single multi-million-row
    ORDER BY query would require.

    Args:
        conn:        MySQL connection.
        table:       Table name (default: salesdata).
        start_date:  Inclusive start date string "YYYY-MM-DD".
        end_date:    Exclusive end date string "YYYY-MM-DD".
        extra_where: Additional SQL filter ANDed into every daily query.
        batch_size:  Rows per ``fetchmany()`` call within a day.
        resume_date: Skip all days strictly before this date "YYYY-MM-DD".
                     Used for checkpoint-based resume.

    Yields: dicts of raw MySQL rows.
    """
    from datetime import date, timedelta

    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required for scroll_rds")

    day = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    resume = date.fromisoformat(resume_date) if resume_date else None

    while day < end:
        next_day = day + timedelta(days=1)

        # Skip days already processed (checkpoint resume)
        if resume and day < resume:
            day = next_day
            continue

        band = (
            f"endTime >= '{day} 00:00:00' AND endTime < '{next_day} 00:00:00'"
        )
        where_clause = f"({extra_where}) AND {band}" if extra_where else band

        query = f"SELECT * FROM {table} WHERE {where_clause} ORDER BY id"

        logger.debug("Date band: {} → {}", day, next_day)
        with conn.cursor(pymysql.cursors.SSDictCursor) as cursor:
            cursor.execute(query)
            day_rows: list[dict] = []
            while True:
                chunk = cursor.fetchmany(batch_size)
                if not chunk:
                    break
                day_rows.extend(chunk)

        logger.info("Fetched {:,} rows for {}", len(day_rows), day)
        yield from day_rows

        day = next_day


def count_rows(
    conn: pymysql.Connection,
    table: str = "salesdata",
    where: str | None = None,
) -> int:
    """Count total rows for progress estimation."""
    query = f"SELECT COUNT(*) as cnt FROM {table}"
    if where:
        query += f" WHERE {where}"

    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
        return result["cnt"]


def count_rows_range(
    conn: pymysql.Connection,
    table: str,
    start_date: str,
    end_date: str,
    extra_where: str | None = None,
) -> int:
    """Count rows in a date range for progress estimation."""
    where = f"endTime >= '{start_date} 00:00:00' AND endTime < '{end_date} 00:00:00'"
    if extra_where:
        where = f"({extra_where}) AND {where}"
    return count_rows(conn, table, where)
