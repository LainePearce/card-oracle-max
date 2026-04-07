"""Scroll reader for the extant OpenSearch cluster (read-only).

Connects to the live AOS cluster in us-west-1.
Scrolls across indices using the full naming taxonomy:

  eBay:      YYYY-MM-DD         (e.g. 2025-03-15)
  Pristine:  YYYY-MM-pris       (e.g. 2025-03-pris)
  Fanatics:  YYYY-MM-pwcc       (e.g. 2025-03-pwcc)
  Heritage:  YYYY-heri          (e.g. 2025-heri)
  MySlabs:   YYYY-ms            (e.g. 2025-ms)
  Goldin:    YYYY-gold          (e.g. 2025-gold)

imageVector and textVector are excluded from _source — embeddings must be re-generated.

CRITICAL: This codebase NEVER writes to the extant OpenSearch cluster.
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from typing import Generator

from loguru import logger
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3


SCROLL_PAGE_SIZE = 1_000
SCROLL_TIMEOUT = "5m"

SCROLL_SOURCE_FIELDS = [
    "id", "itemId", "source", "globalId",
    "title", "galleryURL",
    "saleType", "currentPrice", "currentPriceCurrency",
    "endTime",
    "itemSpecifics",
]

# ── Index classification patterns ─────────────────────────────────────────────

# eBay: YYYY-MM-DD  (must match before the broader marketplace patterns)
_EBAY_RE     = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Pristine / Fanatics: YYYY-MM-<suffix>
_YYYYMM_RE   = re.compile(r"^(\d{4}-\d{2})-(pris|pwcc)$")
# Heritage / MySlabs / Goldin: YYYY-<suffix>
_YYYY_RE     = re.compile(r"^(\d{4})-(heri|ms|gold)$")

_MARKETPLACE_NAMES: dict[str, str] = {
    "pris": "pristine",
    "pwcc": "fanatics_collect",
    "heri": "heritage",
    "ms":   "myslabs",
    "gold": "goldin",
}


def classify_index(index_name: str) -> dict:
    """
    Classify an index by its naming convention.

    Returns dict with:
      marketplace:        "ebay" | "pristine" | "fanatics_collect" | "heritage" | "myslabs" | "goldin" | "unknown"
      has_item_specifics: True for eBay only
      index_type:         short string used as S3 partition key and Qdrant specifics_source
      partition:          date fragment used for S3 key construction
    """
    if _EBAY_RE.match(index_name):
        return {
            "marketplace":        "ebay",
            "has_item_specifics": True,
            "index_type":         "ebay-dated",
            "partition":          index_name,          # YYYY-MM-DD
        }

    m = _YYYYMM_RE.match(index_name)
    if m:
        ym, suffix = m.group(1), m.group(2)
        return {
            "marketplace":        _MARKETPLACE_NAMES[suffix],
            "has_item_specifics": False,
            "index_type":         suffix,
            "partition":          ym,                  # YYYY-MM
        }

    m = _YYYY_RE.match(index_name)
    if m:
        yr, suffix = m.group(1), m.group(2)
        return {
            "marketplace":        _MARKETPLACE_NAMES[suffix],
            "has_item_specifics": False,
            "index_type":         suffix,
            "partition":          yr,                  # YYYY
        }

    return {
        "marketplace":        "unknown",
        "has_item_specifics": False,
        "index_type":         "unknown",
        "partition":          "all",
    }


# ── Client ────────────────────────────────────────────────────────────────────

def get_opensearch_client() -> OpenSearch:
    """Connect to the live AOS cluster using env-configured auth."""
    host    = os.environ["OPENSEARCH_HOST"]
    port    = int(os.environ.get("OPENSEARCH_PORT", 443))
    use_iam = os.environ.get("OPENSEARCH_USE_IAM", "false").lower() == "true"

    if use_iam:
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            os.environ.get("AWS_REGION", "us-west-1"),
            "es",
            session_token=credentials.token,
        )
    else:
        auth = (os.environ["OPENSEARCH_USER"], os.environ["OPENSEARCH_PASSWORD"])

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )


# ── Index discovery ───────────────────────────────────────────────────────────

def discover_indices(client: OpenSearch) -> list[dict]:
    """
    List all indices on the extant cluster and classify them.
    Returns sorted list of dicts with name, doc_count, store_size, and classification.
    """
    response = client.cat.indices(format="json", h="index,docs.count,store.size")
    classified = []
    for row in response:
        name = row["index"]
        if name.startswith("."):
            continue   # skip system indices
        info = classify_index(name)
        classified.append({
            "index":      name,
            "doc_count":  int(row.get("docs.count") or 0),
            "store_size": row.get("store.size", ""),
            **info,
        })
    return sorted(classified, key=lambda x: x["index"])


# ── Date index helpers ────────────────────────────────────────────────────────

def date_index_list(start: date, end: date) -> str:
    """
    Build a comma-separated list of YYYY-MM-DD eBay index names for a date range.
    Avoids wildcards that could match non-eBay indices.
    """
    indices: list[str] = []
    current = start
    while current <= end:
        indices.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return ",".join(indices)


# ── Scroll ────────────────────────────────────────────────────────────────────

def scroll_index(
    client: OpenSearch,
    index_pattern: str,
) -> Generator[dict, None, None]:
    """
    Scroll through all documents in indices matching index_pattern.

    index_pattern examples:
      "2025-03-15"                    — single eBay date index
      "2025-03-15,2025-03-16"        — explicit list (from date_index_list())
      "2025-03-pris"                  — single Pristine monthly index
      "2025-gold"                     — single Goldin yearly index
      "*"                             — all indices (full backfill)

    Yields raw OpenSearch hit dicts: {"_id": ..., "_index": ..., "_source": {...}}
    """
    resp = client.search(
        index=index_pattern,
        scroll=SCROLL_TIMEOUT,
        size=SCROLL_PAGE_SIZE,
        body={"query": {"match_all": {}}, "_source": SCROLL_SOURCE_FIELDS},
    )
    scroll_id = resp["_scroll_id"]
    hits      = resp["hits"]["hits"]
    total     = resp["hits"]["total"]["value"]
    logger.info("Scrolling {} — {} total documents", index_pattern, total)

    while hits:
        yield from hits
        resp      = client.scroll(scroll_id=scroll_id, scroll=SCROLL_TIMEOUT)
        scroll_id = resp["_scroll_id"]
        hits      = resp["hits"]["hits"]

    client.clear_scroll(scroll_id=scroll_id)
