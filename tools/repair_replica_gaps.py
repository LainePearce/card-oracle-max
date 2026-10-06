#!/usr/bin/env python3
"""
Repair replica divergence in a Qdrant collection without re-embedding.

Background (2026-09/10): under write load, node-to-node forwards on the
Qdrant cluster intermittently time out ("Healthcheck timeout 2000ms exceeded").
With write_consistency_factor=1 the upsert is acknowledged by the replica that
succeeded, the failed replica is NOT always marked Dead, and the replicas
silently diverge: cards_dinov2 on node-2 was found ~411k points short of its
peers. Reads that land on the short replica make those listings invisible.

This tool:
  1. enumerates point ids from the image-archive manifests of the given
     indices (eBay days / non-eBay indices);
  2. finds ids absent from at least one replica via a retrieve with
     `consistency=all` (returns only points present on EVERY replica);
  3. for each missing id, reads the full point (vector + payload) directly
     from each node in turn until one has it;
  4. re-upserts it through the NLB with wait=True — the write reaches every
     replica, filling the gap — and re-verifies.

No embedding, no shard streaming, no cluster config ops. Idempotent: run it
until `remaining` is 0. Run from a box with S3 + Qdrant access (worker-0).

    python tools/repair_replica_gaps.py --collection cards_dinov2 \
        --days 2026-09-01:2026-09-30 --nonebay --dry-run
    python tools/repair_replica_gaps.py --collection cards_dinov2 \
        --days 2026-09-01:2026-09-30 --nonebay
    python tools/repair_replica_gaps.py --collection cards --index 2026-09-14
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import httpx
from loguru import logger

from tools.dino_embed_common import s3_client, QUEUE_BUCKET, ARCHIVE_MANIFESTS

NLB   = f"http://{os.environ['QDRANT_HOST']}:{os.environ.get('QDRANT_HTTP_PORT', 6333)}"
NODES = [n for n in os.environ.get(
    "QDRANT_NODE_IPS", "172.31.0.41,172.31.7.154,172.31.6.110").split(",") if n]
HEADERS = {"Content-Type": "application/json",
           **({"api-key": os.environ["QDRANT_API_KEY"]} if os.environ.get("QDRANT_API_KEY") else {})}


def manifest_ids(s3, index_name: str) -> list:
    try:
        raw = s3.get_object(Bucket=QUEUE_BUCKET,
                            Key=f"{ARCHIVE_MANIFESTS}/{index_name}.jsonl.gz")["Body"].read()
    except Exception:
        logger.warning("{}: no manifest", index_name)
        return []
    ids = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        for line in gz:
            if line.strip():
                q = json.loads(line)["qdrant_id"]
                ids.append(int(q) if q.isdigit() else q)
    return ids


def retrieve(c: httpx.Client, base: str, coll: str, ids: list, *, consistency=None,
             with_vector=False, with_payload=False) -> dict:
    url = f"{base}/collections/{coll}/points" + (f"?consistency={consistency}" if consistency else "")
    body = {"ids": ids, "with_vector": with_vector, "with_payload": with_payload}
    for attempt in range(1, 6):
        try:
            r = c.post(url, json=body, headers=HEADERS)
            r.raise_for_status()
            return {str(p["id"]): p for p in r.json()["result"]}
        except Exception as e:
            if attempt == 5:
                raise
            logger.warning("retrieve via {} failed ({}), retry {}/5", base, e, attempt)
            time.sleep(2 * attempt)
    return {}


def find_missing(c, coll, ids, batch=1000):
    """Ids absent from at least one replica (present everywhere == consistency=all)."""
    missing = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        everywhere = retrieve(c, NLB, coll, chunk, consistency="all")
        missing += [x for x in chunk if str(x) not in everywhere]
    return missing


def fetch_full(c, coll, ids) -> list:
    """Read vector+payload for ids from whichever node holds them."""
    found: dict = {}
    for ip in NODES:
        want = [x for x in ids if str(x) not in found]
        if not want:
            break
        got = retrieve(c, f"http://{ip}:6333", coll, want, with_vector=True, with_payload=True)
        found.update(got)
    return [found[str(x)] for x in ids if str(x) in found]


def upsert(c, coll, points) -> None:
    body = {"points": [{"id": p["id"], "vector": p["vector"], "payload": p.get("payload") or {}}
                       for p in points]}
    delay = 2.0
    for attempt in range(1, 7):
        try:
            r = c.put(f"{NLB}/collections/{coll}/points?wait=true", json=body, headers=HEADERS)
            r.raise_for_status()
            return
        except Exception as e:
            if attempt == 6:
                raise
            logger.warning("upsert {} pts failed ({}), retry {}/6 in {:.0f}s", len(points), e, attempt, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default="cards_dinov2")
    ap.add_argument("--days", help="eBay day range START:END (inclusive, YYYY-MM-DD)")
    ap.add_argument("--index", nargs="*", default=[], help="explicit index names")
    ap.add_argument("--nonebay", action="store_true",
                    help="also include the current-year non-eBay indices")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true", help="count only, no writes")
    args = ap.parse_args()

    indices = list(args.index)
    if args.days:
        a, b = (date.fromisoformat(x) for x in args.days.split(":"))
        indices += [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]
    if args.nonebay:
        y = date.today().year
        indices += [f"{y}-{m:02d}-{t}" for m in range(1, 13) for t in ("pris", "pwcc")]
        indices += [f"{y}-{t}" for t in ("gold", "ms", "heritage", "heri")]
    if not indices:
        ap.error("give --days, --index and/or --nonebay")

    s3 = s3_client()
    coll = args.collection
    total_missing = total_fixed = total_remaining = 0
    with httpx.Client(timeout=180) as c:
        for idx in indices:
            ids = manifest_ids(s3, idx)
            if not ids:
                continue
            missing = find_missing(c, coll, ids)
            total_missing += len(missing)
            logger.info("{}: {:,} ids, {:,} missing from ≥1 replica", idx, len(ids), len(missing))
            if args.dry_run or not missing:
                continue
            fixed = 0
            for i in range(0, len(missing), args.batch):
                chunk = missing[i:i + args.batch]
                pts = fetch_full(c, coll, chunk)
                if len(pts) < len(chunk):
                    logger.warning("{}: {} ids not found on ANY node (never written) — skipped",
                                   idx, len(chunk) - len(pts))
                if pts:
                    upsert(c, coll, pts)
                    fixed += len(pts)
            remaining = len(find_missing(c, coll, missing))
            total_fixed += fixed
            total_remaining += remaining
            logger.info("{}: re-upserted {:,}; remaining missing after verify: {:,}", idx, fixed, remaining)

    logger.info("DONE — missing {:,} | re-upserted {:,} | still missing {:,}{}",
                total_missing, total_fixed, total_remaining,
                "  (dry run)" if args.dry_run else "")


if __name__ == "__main__":
    main()
