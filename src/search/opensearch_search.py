"""OpenSearch KNN search queries — control system for experiment.

Queries the extant AOS cluster using imageVector KNN.
post_filter is applied AFTER KNN candidate selection — this intentionally
causes recall degradation under selective filters (documented weakness).

NOTE — embedding model for OS queries:
  The extant cluster's imageVector field was indexed using Xenova/clip-vit-base-patch32,
  which is OpenAI CLIP ViT-B/32 exported to ONNX (512-dim, L2-normalised).
  Query vectors sent to this module MUST be produced by the equivalent open_clip model:
    ImageEncoder(model_name="ViT-B-32", pretrained="openai")
  Using any other model will produce a dimension or distribution mismatch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import NamedTuple

from opensearchpy import OpenSearch


@dataclass
class OSResult:
    doc_id: str
    score: float
    source: dict


class QueryTimings(NamedTuple):
    query_ms: float


def opensearch_knn_search(
    client: OpenSearch,
    image_vec: list[float],
    index_pattern: str = "*",
    filters: dict | None = None,
    top_k: int = 20,
) -> tuple[list[OSResult], QueryTimings]:
    """
    Execute a KNN search against the live OpenSearch cluster.
    post_filter applied after KNN — measures recall degradation under filtering.
    Returns (results, timings).
    """
    query: dict = {
        "size": top_k,
        "query": {
            "knn": {
                "imageVector": {
                    "vector": image_vec,
                    "k": top_k,
                }
            }
        },
        "_source": [
            "id", "itemId", "source", "globalId",
            "galleryURL", "saleType", "currentPrice",
            "itemSpecifics",
        ],
    }

    if filters:
        query["post_filter"] = build_os_filter(filters)

    t0 = time.perf_counter()
    response = client.search(index=index_pattern, body=query)
    query_ms = (time.perf_counter() - t0) * 1000

    results = [
        OSResult(
            doc_id=hit["_id"],
            score=hit.get("_score", 0.0),
            source=hit["_source"],
        )
        for hit in response["hits"]["hits"]
    ]
    return results, QueryTimings(query_ms=query_ms)


def build_os_filter(criteria: dict) -> dict:
    """Build an OpenSearch bool filter from itemSpecifics criteria."""
    field_map = {
        "set":         "itemSpecifics.set",
        "card_number": "itemSpecifics.cardNumber",
        "grader":      "itemSpecifics.grader",
        "grade":       "itemSpecifics.grade",
        "graded":      "itemSpecifics.graded",
        "genre":       "itemSpecifics.genre",
        "player":      "itemSpecifics.player",
        "brand":       "itemSpecifics.brand",
        "parallel":    "itemSpecifics.parallel",
        "year":        "itemSpecifics.year",
        "autographed": "itemSpecifics.autographed",
        "team":        "itemSpecifics.team",
        "source":      "source",
        "global_id":   "globalId",
    }
    must_clauses = []
    for key, value in criteria.items():
        os_field = field_map.get(key, key)
        if isinstance(value, bool):
            must_clauses.append({"term": {os_field: value}})
        elif isinstance(value, str):
            must_clauses.append({"term": {os_field: value.lower().strip()}})
        elif isinstance(value, (int, float)):
            must_clauses.append({"term": {os_field: value}})
    return {"bool": {"must": must_clauses}}
