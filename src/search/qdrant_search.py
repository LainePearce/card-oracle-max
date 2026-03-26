"""Qdrant query patterns for card search.

Implements three-arm prefetch with RRF fusion:
  Arm 1: image ANN (has_image=True)
  Arm 2: specifics ANN (all cards)
  Arm 3: specifics ANN (has_image=False catalogue only)
"""

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
