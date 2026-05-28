#!/usr/bin/env python3
"""
2a — Fix-up Population C.

Population C: ~500K points where an image vector slot is populated but no
has_image=true payload flag is set, so the search Arm 1 filter excludes them.
The patched repopulate's set_payload should have covered these but missed
some — likely because their image vectors came from a path other than the
S3 image shards (daily worker, dual_write_pipeline, etc.).

Strategy (two-pass for safety — Qdrant scroll consistency under concurrent
updates is undefined, so we collect first, then write):

  Pass 1 — scroll has_image != true, retrieve image vector, collect ids
           where the vector slot is populated.
  Pass 2 — batched set_payload({"has_image": True}, points=batch).

Population D (no image vector) is identified and left alone.

Idempotent — safe to re-run. If has_image=true is already set, the filter
in pass 1 excludes the point so it's never touched.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


COLLECTION = os.environ.get("QDRANT_COLLECTION", "cards")
BATCH      = 500
SCROLL     = 2_000


def main() -> None:
    client = QdrantClient(
        url=f"http://{os.environ['QDRANT_HOST']}:6333",
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=False, timeout=300, check_compatibility=False,
    )

    no_flag_filter = Filter(must_not=[
        FieldCondition(key="has_image", match=MatchValue(value=True))
    ])

    # ── Pass 1: collect ids of image-bearing points missing the flag ─────
    print("Pass 1: scrolling has_image != true, collecting image-bearing ids...",
          flush=True)
    t0 = time.time()

    ids_to_fix:    list = []     # have image vec slot — Population C
    no_img_count  = 0            # no image vec — Population D
    total_scrolled = 0
    offset = None

    while True:
        pts, offset = client.scroll(
            collection_name = COLLECTION,
            scroll_filter   = no_flag_filter,
            limit           = SCROLL,
            offset          = offset,
            with_payload    = False,
            with_vectors    = ["image"],
        )
        if not pts:
            break
        total_scrolled += len(pts)
        for p in pts:
            if isinstance(p.vector, dict) and p.vector.get("image") is not None:
                ids_to_fix.append(p.id)
            else:
                no_img_count += 1
        if total_scrolled % 20_000 == 0 or offset is None:
            print(f"  scrolled={total_scrolled:,}  to_fix={len(ids_to_fix):,}  "
                  f"no_img={no_img_count:,}  ({time.time()-t0:.0f}s)", flush=True)
        if offset is None:
            break

    print()
    print(f"Pass 1 done in {time.time()-t0:.1f}s")
    print(f"  Total scrolled (has_image != true):  {total_scrolled:,}")
    print(f"  Population C (will fix):             {len(ids_to_fix):,}")
    print(f"  Population D (no image vec):         {no_img_count:,}")

    if not ids_to_fix:
        print("\nNothing to fix.")
        return

    # ── Pass 2: batched set_payload({"has_image": True}) ─────────────────
    print(f"\nPass 2: batched set_payload on {len(ids_to_fix):,} ids "
          f"(batch={BATCH}, wait=True)...", flush=True)
    t1 = time.time()
    fixed   = 0
    errors  = 0

    for start in range(0, len(ids_to_fix), BATCH):
        chunk = ids_to_fix[start:start + BATCH]
        try:
            client.set_payload(
                collection_name = COLLECTION,
                payload         = {"has_image": True},
                points          = chunk,
                wait            = True,
            )
            fixed += len(chunk)
        except Exception as e:
            errors += 1
            print(f"  ERROR on batch {start//BATCH}: {type(e).__name__}: {e}",
                  flush=True)

        if (start // BATCH + 1) % 50 == 0:
            elapsed = time.time() - t1
            rate    = fixed / elapsed if elapsed > 0 else 0
            print(f"  fixed={fixed:,}/{len(ids_to_fix):,} "
                  f"({100*fixed/len(ids_to_fix):.1f}%)  "
                  f"rate={rate:,.0f}/s  elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t1
    print()
    print(f"Pass 2 done in {elapsed:.1f}s")
    print(f"  Fixed:           {fixed:,}")
    print(f"  Batch errors:    {errors}")
    print()
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
