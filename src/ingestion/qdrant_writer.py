"""Bulk upsert writer for Qdrant.

Extracts payload from OpenSearch documents or RDS-sourced dicts,
normalises field values (lowercase + strip to match OpenSearch
lowercase_normalizer), and upserts points in batches.

ID strategy
-----------
Qdrant requires uint64 or UUID point IDs. The RDS/OpenSearch
numeric `id` field (MySQL BIGINT / ES long) is used directly as
a uint64 where possible. String-based IDs fall back to UUID5.

Usage (standalone)
------------------
    from src.ingestion.qdrant_writer import get_qdrant_client, build_point, upsert_batch

    client = get_qdrant_client()
    point  = build_point(qdrant_id, payload, image_vec, specifics_vec)
    upsert_batch(client, [point])
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

UPSERT_BATCH_SIZE = 1_000
UPSERT_WAIT       = False   # async upsert for backfill throughput
CHECKPOINT_EVERY  = 10_000

# Deterministic namespace for UUID5 fallback IDs
_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "cards")


# ── Client factory ─────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """
    Build a QdrantClient from environment variables.

    Required env vars:
        QDRANT_HOST         hostname or IP  (default: localhost)
        QDRANT_PORT         HTTP port       (default: 6333)
    Optional:
        QDRANT_API_KEY      API key
        QDRANT_USE_GRPC     "true" to prefer gRPC (default: false)
        QDRANT_COLLECTION   collection name (default: cards)
    """
    host     = os.environ.get("QDRANT_HOST", "localhost")
    port     = int(os.environ.get("QDRANT_PORT", 6333))
    api_key  = os.environ.get("QDRANT_API_KEY") or None
    use_grpc = os.environ.get("QDRANT_USE_GRPC", "false").lower() in ("true", "1")

    # Always use an explicit URL so the client does not assume HTTPS on remote hosts.
    url = f"http://{host}:{port}"
    logger.debug("Connecting to Qdrant at {} (grpc={})", url, use_grpc)

    return QdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=use_grpc,
        timeout=60,
    )


# ── ID conversion ──────────────────────────────────────────────────────────────

def os_id_to_qdrant_id(doc_id: str | int) -> int | str:
    """
    Map an OpenSearch / RDS document ID to a Qdrant point ID.

    Preference order:
      1. If `doc_id` is already an int → use as uint64 directly.
      2. If `doc_id` is a numeric string → cast to int.
      3. Otherwise → deterministic UUID5 from the string.
    """
    if isinstance(doc_id, int):
        return doc_id
    try:
        return int(doc_id)
    except (ValueError, TypeError):
        return str(uuid.uuid5(_UUID_NAMESPACE, str(doc_id)))


# ── Payload building ───────────────────────────────────────────────────────────

def _lower(val) -> str:
    """Lowercase + strip a value, returning '' for None/empty."""
    if val is None:
        return ""
    return str(val).lower().strip()


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).lower().strip()
    return s in ("true", "1", "yes", "y")


def extract_payload(
    doc: dict,
    doc_id: str | None = None,
    specifics_source: str = "ebay",
    specifics_confidence: float | None = None,
    specifics_ref_ids: list | None = None,
    specifics_model: str | None = None,
    is_catalogue: bool = False,
) -> dict:
    """
    Build a Qdrant payload dict from a document.

    Accepts both:
      - OpenSearch scroll hits: pass the raw `_source` dict as `doc`
        and the `_id` as `doc_id`.
      - RDS-derived documents: as produced by `rds_reader.transform_row()`.
        In this case `doc_id` is the string ID from the RDS row.

    All string itemSpecifics values are lowercased to mirror
    OpenSearch's lowercase_normalizer behaviour.
    """
    specs = doc.get("itemSpecifics") or {}

    # --- Core identity ---
    effective_id = doc_id if doc_id is not None else str(doc.get("id", ""))

    payload: dict = {
        # Join key back to OpenSearch / Elasticsearch
        "os_id":     effective_id,
        "item_id":   str(doc.get("itemId", "")),
        "type":      "catalogue" if is_catalogue else "sold",
        "has_image": bool(doc.get("galleryURL", "").strip()),
        "source":    _lower(doc.get("source")),
        "global_id": _lower(doc.get("globalId")),
        "active":    True,

        # --- itemSpecifics (all lowercased) ---
        "brand":         _lower(specs.get("brand")),
        "player":        _lower(specs.get("player")),
        "genre":         _lower(specs.get("genre")),
        "country":       _lower(specs.get("country")),
        "set":           _lower(specs.get("set")),
        "card_number":   _lower(specs.get("cardNumber") or specs.get("card_number")),
        "subset":        _lower(specs.get("subset")),
        "parallel":      _lower(specs.get("parallel")),
        "serial_number": _lower(specs.get("serialNumber") or specs.get("serial_number")),
        "year":          int(specs["year"]) if specs.get("year") else None,
        "graded":        _to_bool(specs.get("graded", False)),
        "grader":        _lower(specs.get("grader")),
        "grade":         _lower(specs.get("grade")),
        "card_type":     _lower(specs.get("type")),   # renamed to avoid clash
        "autographed":   _to_bool(specs.get("autographed", False)),
        "team":          _lower(specs.get("team")),

        # --- Sale context ---
        "sale_type":     _lower(doc.get("saleType")),
        "current_price": _to_float(doc.get("currentPrice")),
        "currency":      _lower(doc.get("currentPriceCurrency")),

        # --- Provenance ---
        "specifics_source":     specifics_source,
        "specifics_confidence": specifics_confidence,
        "specifics_ref_ids":    specifics_ref_ids,
        "specifics_model":      specifics_model,

        # --- System ---
        "image_is_pseudo":           False,
        "pseudo_vector_sample_size": 0,
        "indexed_at":                datetime.utcnow().isoformat(),
    }

    return payload


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Point construction ─────────────────────────────────────────────────────────

def build_point(
    qdrant_id:     int | str,
    payload:       dict,
    image_vec:     Optional[np.ndarray],
    specifics_vec: Optional[np.ndarray],
) -> PointStruct | None:
    """
    Build a Qdrant PointStruct.

    Returns None if neither vector is available (point cannot be indexed).
    A point with only an image vector (no specifics) is valid.
    A point with only a specifics vector (no image) is valid.
    """
    vectors: dict[str, list[float]] = {}

    if image_vec is not None:
        vectors["image"] = image_vec.tolist()
    if specifics_vec is not None:
        vectors["specifics"] = specifics_vec.tolist()

    if not vectors:
        return None

    return PointStruct(
        id=qdrant_id,
        vector=vectors,
        payload=payload,
    )


# ── Batch upsert ───────────────────────────────────────────────────────────────

def upsert_batch(
    client:          QdrantClient,
    points:          list[PointStruct],
    collection_name: str = COLLECTION_NAME,
    wait:            bool = UPSERT_WAIT,
) -> None:
    """
    Upsert a batch of PointStructs into Qdrant.

    Logs the batch size and any errors. Raises on failure — callers
    should NOT fall back silently; a Qdrant write failure must be logged
    to the dead-letter queue.
    """
    if not points:
        return

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=wait,
    )
    logger.debug("Qdrant upsert: {} points → '{}'", len(points), collection_name)


def upsert_in_batches(
    client:          QdrantClient,
    points:          list[PointStruct],
    batch_size:      int = UPSERT_BATCH_SIZE,
    collection_name: str = COLLECTION_NAME,
    wait:            bool = UPSERT_WAIT,
) -> int:
    """
    Upsert a large list of points in batches of `batch_size`.
    Returns the total number of points upserted.
    """
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        upsert_batch(client, batch, collection_name=collection_name, wait=wait)
        total += len(batch)
    return total
