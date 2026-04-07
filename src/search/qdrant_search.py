"""Qdrant query patterns for card search.

Simple image vector search using qdrant-client v1.8.x API.
No prefetch/RRF — single-arm image search for experiment comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import NamedTuple

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams, QuantizationSearchParams


SEARCH_PARAMS_STANDARD = SearchParams(
    hnsw_ef=128,
    exact=False,
    quantization=QuantizationSearchParams(rescore=True),
)

SEARCH_PARAMS_HARD = SearchParams(
    hnsw_ef=256,
    exact=False,
    quantization=QuantizationSearchParams(rescore=True),
)

SEARCH_PARAMS_EXACT = SearchParams(exact=True)


@dataclass
class QdrantResult:
    os_id: str
    score: float
    payload: dict


class QueryTimings(NamedTuple):
    query_ms: float


def image_search(
    client: QdrantClient,
    image_vec: list[float],
    collection: str = "cards",
    top_k: int = 20,
    is_hard_query: bool = False,
) -> tuple[list[QdrantResult], QueryTimings]:
    """
    Search Qdrant by image vector similarity.
    Returns (results, timings) — compatible with qdrant-client v1.8.x.
    """
    params = SEARCH_PARAMS_HARD if is_hard_query else SEARCH_PARAMS_STANDARD

    t0 = time.perf_counter()
    hits = client.search(
        collection_name=collection,
        query_vector=("image", image_vec),
        limit=top_k,
        with_payload=True,
        search_params=params,
    )
    query_ms = (time.perf_counter() - t0) * 1000

    results = [
        QdrantResult(
            os_id=h.payload.get("os_id", str(h.id)),
            score=h.score,
            payload=h.payload,
        )
        for h in hits
    ]
    return results, QueryTimings(query_ms=query_ms)


def specifics_search(
    client: QdrantClient,
    specifics_vec: list[float],
    collection: str = "cards",
    top_k: int = 20,
) -> tuple[list[QdrantResult], QueryTimings]:
    """Search Qdrant by specifics (text) vector similarity."""
    t0 = time.perf_counter()
    hits = client.search(
        collection_name=collection,
        query_vector=("specifics", specifics_vec),
        limit=top_k,
        with_payload=True,
        search_params=SEARCH_PARAMS_STANDARD,
    )
    query_ms = (time.perf_counter() - t0) * 1000

    results = [
        QdrantResult(
            os_id=h.payload.get("os_id", str(h.id)),
            score=h.score,
            payload=h.payload,
        )
        for h in hits
    ]
    return results, QueryTimings(query_ms=query_ms)
