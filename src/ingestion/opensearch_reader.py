"""Scroll reader for the extant OpenSearch cluster (read-only).

Connects to the live AOS cluster in us-west-1.
Scrolls across per-date indices (YYYY-MM-DD) and marketplace suffix indices.
imageVector and textVector are excluded from _source — embeddings must be re-generated.

CRITICAL: This codebase NEVER writes to the extant OpenSearch cluster.
"""

SCROLL_PAGE_SIZE = 1_000
SCROLL_TIMEOUT = "5m"

SCROLL_SOURCE_FIELDS = [
    "id", "itemId", "source", "globalId",
    "title", "galleryURL",
    "saleType", "currentPrice", "currentPriceCurrency",
    "endTime",
    "itemSpecifics",
]
