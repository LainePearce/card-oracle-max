#!/usr/bin/env python3
"""
2c — Spot-check Population A.

Population A: ~18.5M points written by the OS-scroll backfill with full
rich payload (has_image=true + source + type + ...). They're disjoint from
the 51M S3 image-shard set that audit_image.py probed, so we don't yet know
whether their image-vector slots are populated.

Sampling strategy — use an indexed filter to target Population A directly:

    must     = [has_image == true]
    must_not = [IsEmpty(source)]

The patched repopulate's set_payload only added has_image=true to Population
B; it never touched source/type/etc. So `source` is set only on Population A.
This filter returns A directly without scrolling through B's millions of
stubs in segment order (which is why the prior version of this script hung).

Decision rule after running:
  - have image vec ≈ 100%: A is healthy, leave alone, move to 2a/2b.
  - have image vec materially <100%: A has its own vector-missing problem;
    needs separate remediation (re-embed from galleryURL).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    IsEmptyCondition, PayloadField,
)


COLLECTION    = os.environ.get("QDRANT_COLLECTION", "cards")
SAMPLE_TARGET = 2_000


def main() -> None:
    client = QdrantClient(
        url=f"http://{os.environ['QDRANT_HOST']}:6333",
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=False, timeout=300, check_compatibility=False,
    )

    pop_a_filter = Filter(
        must=[FieldCondition(key="has_image", match=MatchValue(value=True))],
        must_not=[IsEmptyCondition(is_empty=PayloadField(key="source"))],
    )

    sampled:        list = []
    offset               = None
    page                 = 0

    print(f"Scrolling Population A directly "
          f"(has_image=true AND source != empty), target {SAMPLE_TARGET:,}...",
          flush=True)

    while len(sampled) < SAMPLE_TARGET:
        pts, offset = client.scroll(
            collection_name = COLLECTION,
            scroll_filter   = pop_a_filter,
            limit           = 500,
            offset          = offset,
            with_payload    = True,
            with_vectors    = ["image"],
        )
        page += 1
        if not pts:
            print(f"  page {page}: 0 results — scroll exhausted", flush=True)
            break
        sampled.extend(pts)
        print(f"  page {page}: +{len(pts):,}  total {len(sampled):,}",
              flush=True)
        if offset is None:
            break

    sampled = sampled[:SAMPLE_TARGET]
    n = len(sampled)

    if n == 0:
        print("\nNo Population A points found. "
              "Either the filter is wrong or A doesn't exist in the form we expected.")
        return

    have_img = sum(
        1 for p in sampled
        if isinstance(p.vector, dict) and p.vector.get("image") is not None
    )
    print()
    print(f"Sampled {n:,} Population A points")
    print(f"  with image vector slot populated:    {have_img:,}  ({100*have_img/n:.1f}%)")
    print(f"  without:                             {n-have_img:,}  ({100*(n-have_img)/n:.1f}%)")
    print()

    payload_keys: Counter = Counter()
    sources:      Counter = Counter()
    types_:       Counter = Counter()
    spec_srcs:    Counter = Counter()
    n_keys_dist:  Counter = Counter()
    for p in sampled:
        pl = p.payload or {}
        payload_keys.update(pl.keys())
        sources.update([pl.get("source", "<missing>")])
        types_.update([pl.get("type", "<missing>")])
        spec_srcs.update([pl.get("specifics_source", "<missing>")])
        n_keys_dist[len(pl)] += 1

    print("Payload-key count distribution (rich payload should be 20+ keys):")
    for k in sorted(n_keys_dist):
        c = n_keys_dist[k]
        print(f"  {k:3d} keys: {c:,}  ({100*c/n:.1f}%)")
    print()
    print("Top 15 payload keys:")
    for k, c in payload_keys.most_common(15):
        print(f"  {k:30s}  {c:,}  ({100*c/n:.1f}%)")
    print()
    print("Top 'source' values:")
    for k, c in sources.most_common(10):
        print(f"  {repr(k):30s}  {c:,}  ({100*c/n:.1f}%)")
    print()
    print("Top 'type' values:")
    for k, c in types_.most_common(10):
        print(f"  {repr(k):20s}  {c:,}  ({100*c/n:.1f}%)")
    print()
    print("Top 'specifics_source' values:")
    for k, c in spec_srcs.most_common(10):
        print(f"  {repr(k):20s}  {c:,}  ({100*c/n:.1f}%)")


if __name__ == "__main__":
    main()
