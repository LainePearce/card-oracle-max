#!/usr/bin/env python3
"""
2c — Spot-check Population A.

Population A: ~18.5M points whose has_image=true flag predates the recovery,
written by the OS-scroll backfill with full rich payload. We never sampled
them directly (they're disjoint from the S3 image-shard set that audit_image.py
probed), so we don't actually know whether their image-vector slots are
populated.

Sampling strategy: scroll points with has_image=true and classify client-side
by payload size — anything with >2 payload keys is Population A (rich payload),
2 or fewer is Population B (stub form: {os_id, has_image}). Pull image vector
along with payload and check if the slot is populated.

Decision rule after running:
  - have image vec ≈ 100%: A is healthy, leave alone, move to 2a/2b.
  - have image vec materially <100%: A has the same vector-missing pattern
    as B used to have; a separate recovery is needed (re-embed from
    galleryURL, or find image vectors at a different S3 prefix).
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
from qdrant_client.models import Filter, FieldCondition, MatchValue


COLLECTION    = os.environ.get("QDRANT_COLLECTION", "cards")
SAMPLE_TARGET = 2_000


def main() -> None:
    client = QdrantClient(
        url=f"http://{os.environ['QDRANT_HOST']}:6333",
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=False, timeout=300, check_compatibility=False,
    )

    has_image_filter = Filter(must=[
        FieldCondition(key="has_image", match=MatchValue(value=True))
    ])

    sampled_pop_a:    list = []
    total_scrolled    = 0
    pop_a_seen        = 0
    pop_b_seen        = 0
    offset            = None

    print(f"Scrolling has_image=true, collecting up to {SAMPLE_TARGET:,} "
          f"Population A samples (rich payload)...", flush=True)

    while len(sampled_pop_a) < SAMPLE_TARGET:
        pts, offset = client.scroll(
            collection_name  = COLLECTION,
            scroll_filter    = has_image_filter,
            limit            = 2_000,
            offset           = offset,
            with_payload     = True,
            with_vectors     = ["image"],
        )
        if not pts:
            break
        total_scrolled += len(pts)
        for p in pts:
            n_keys = len(p.payload or {})
            if n_keys > 2:
                sampled_pop_a.append(p)
                pop_a_seen += 1
                if len(sampled_pop_a) >= SAMPLE_TARGET:
                    break
            else:
                pop_b_seen += 1
        if offset is None:
            break

    n = len(sampled_pop_a)
    if n == 0:
        print("No Population A points found in scroll window — "
              "all sampled has_image=true points have stub payload.")
        print(f"  total scrolled: {total_scrolled:,}, all classified as Population B")
        return

    have_img = sum(
        1 for p in sampled_pop_a
        if isinstance(p.vector, dict) and p.vector.get("image") is not None
    )

    print()
    print(f"Sampled {n:,} Population A points (payload has >2 keys)")
    print(f"  with image vector slot populated:    {have_img:,}  ({100*have_img/n:.1f}%)")
    print(f"  without:                             {n-have_img:,}  ({100*(n-have_img)/n:.1f}%)")
    print()
    print(f"Population mix in {total_scrolled:,} scrolled has_image=true points:")
    print(f"  Population A (rich payload):  {pop_a_seen:,}  ({100*pop_a_seen/total_scrolled:.1f}%)")
    print(f"  Population B (stub payload):  {pop_b_seen:,}  ({100*pop_b_seen/total_scrolled:.1f}%)")
    print()

    payload_keys: Counter = Counter()
    sources:      Counter = Counter()
    types_:       Counter = Counter()
    spec_srcs:    Counter = Counter()
    for p in sampled_pop_a:
        pl = p.payload or {}
        payload_keys.update(pl.keys())
        sources.update([pl.get("source", "<missing>")])
        types_.update([pl.get("type", "<missing>")])
        spec_srcs.update([pl.get("specifics_source", "<missing>")])

    print("Population A payload-key frequency (top 25):")
    for k, c in payload_keys.most_common(25):
        print(f"  {k:30s}  {c:,}  ({100*c/n:.1f}%)")
    print()
    print("Population A 'source' values (top 10):")
    for k, c in sources.most_common(10):
        print(f"  {repr(k):30s}  {c:,}  ({100*c/n:.1f}%)")
    print()
    print("Population A 'type' values:")
    for k, c in types_.most_common(10):
        print(f"  {repr(k):20s}  {c:,}  ({100*c/n:.1f}%)")
    print()
    print("Population A 'specifics_source' values:")
    for k, c in spec_srcs.most_common(10):
        print(f"  {repr(k):20s}  {c:,}  ({100*c/n:.1f}%)")


if __name__ == "__main__":
    main()
