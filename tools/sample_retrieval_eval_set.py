#!/usr/bin/env python3
"""
Build an at-scale retrieval evaluation set: a large catalog CORPUS plus a
held-out QUERY set with identity ground truth.

This is the real-world test the synthetic parallel eval can't give: a query
image is searched against a big catalog, and we ask "is the correct card in the
top-K". Run identically for CLIP / DINOv2 / DINOv3 it tells us which encoder
actually retrieves better at scale, not just within a 67-group toy set.

Ground-truth construction:
  - identity_key = player|year|set|card_number|parallel   (the exact card)
  - design_key   = player|year|set|card_number            (card ignoring parallel)
  - For each identity with >= 2 images from DISTINCT gallery URLs (distinct
    photos — never the same photo, which would be a trivial exact match), one
    image becomes a QUERY and the rest go into the CORPUS. Every query is
    therefore guaranteed a same-identity answer somewhere in the corpus.
  - The corpus is then padded with single-image identities as realistic
    distractors up to --corpus-size.

The evaluator (eval_retrieval_at_scale.py) reports Recall@K at BOTH design and
identity granularity; the gap = parallel confusion at scale.

Usage:
    python tools/sample_retrieval_eval_set.py --corpus-size 30000 --queries 2000 \
        --out data/retrieval_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from src.ingestion.opensearch_reader import get_opensearch_client
from tools.sample_parallel_eval_set import (
    norm, norm_number, _clean, design_key, parallel_label, download_image,
)


def scroll_pool(os_client, index_pattern: str, pool_size: int) -> list[dict]:
    body = {
        "size": 1000,
        "query": {"bool": {
            "must": [
                {"exists": {"field": "galleryURL"}},
                {"exists": {"field": "itemSpecifics.player"}},
                {"exists": {"field": "itemSpecifics.set"}},
                {"exists": {"field": "itemSpecifics.cardNumber"}},
            ],
            "must_not": [{"term": {"galleryURL": "N/A"}}],
        }},
        "_source": ["id", "title", "galleryURL", "itemSpecifics"],
    }
    out: list[dict] = []
    resp = os_client.search(index=index_pattern, body=body, scroll="3m",
                            request_timeout=60)
    sid = resp.get("_scroll_id")
    while True:
        hits = resp["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            src   = h["_source"]
            specs = src.get("itemSpecifics") or {}
            url   = _clean(src.get("galleryURL"))
            dk    = design_key(specs)
            if not url or url.lower() == "n/a" or dk is None:
                continue
            par = parallel_label(specs)
            out.append({
                "os_id":       h["_id"],
                "gallery_url": url,
                "design_key":  dk,
                "identity_key": f"{dk}|{par}",
                "parallel":    par,
                "player":      _clean(specs.get("player")),
                "year":        _clean(specs.get("year")),
                "set":         _clean(specs.get("set")),
                "card_number": _clean(specs.get("cardNumber")),
            })
        if len(out) >= pool_size:
            break
        resp = os_client.scroll(scroll_id=sid, scroll="3m", request_timeout=60)
        sid  = resp.get("_scroll_id")
        if sid is None:
            break
    logger.info("Scrolled {} candidate cards from {}", len(out), index_pattern)
    return out


def plan_split(pool: list[dict], corpus_size: int, n_queries: int,
               max_per_identity: int) -> tuple[list[dict], list[dict]]:
    """Choose query + corpus cards. Queries get a same-identity corpus answer."""
    # De-dup to distinct gallery URLs per identity (distinct photos only).
    by_identity: dict[str, dict[str, dict]] = defaultdict(dict)
    for c in pool:
        by_identity[c["identity_key"]].setdefault(c["gallery_url"], c)

    queries: list[dict] = []
    corpus:  list[dict] = []
    leftover_singletons: list[dict] = []

    for ik, by_url in by_identity.items():
        cards = list(by_url.values())
        if len(cards) >= 2 and len(queries) < n_queries:
            q, *rest = cards
            q["role"] = "query"
            queries.append(q)
            for c in rest[: max_per_identity]:
                c["role"] = "corpus"
                corpus.append(c)
        else:
            for c in cards[: max_per_identity]:
                leftover_singletons.append(c)

    # Pad corpus with distractors up to target size.
    for c in leftover_singletons:
        if len(corpus) >= corpus_size:
            break
        c["role"] = "corpus"
        corpus.append(c)

    return queries, corpus


def download_all(cards: list[dict], img_dir: Path, workers: int) -> list[dict]:
    def _one(c: dict) -> dict | None:
        dest = img_dir / f"{c['os_id'].replace('/', '_')}.jpg"
        if dest.exists() or download_image(c["gallery_url"], dest):
            c["image_file"] = str(dest.relative_to(img_dir.parent))
            return c
        return None

    ok: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, c) for c in cards]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r is not None:
                ok.append(r)
            if (i + 1) % 2000 == 0:
                logger.info("  downloaded {}/{} ({} ok)", i + 1, len(cards), len(ok))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-size", type=int, default=30000)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--out", default="data/retrieval_eval")
    ap.add_argument("--index", default="2026-*,2025-*")
    ap.add_argument("--pool", type=int, default=300000)
    ap.add_argument("--max-per-identity", type=int, default=4)
    ap.add_argument("--download-workers", type=int, default=24)
    args = ap.parse_args()

    out_dir = ROOT / args.out
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    os_client = get_opensearch_client()
    pool = scroll_pool(os_client, args.index, args.pool)
    queries, corpus = plan_split(pool, args.corpus_size, args.queries,
                                 args.max_per_identity)
    logger.info("Planned {} queries + {} corpus cards — downloading …",
                len(queries), len(corpus))

    queries = download_all(queries, img_dir, args.download_workers)
    corpus  = download_all(corpus,  img_dir, args.download_workers)

    # Post-download validation: keep only queries whose identity (or at least
    # design) still has a surviving corpus image — otherwise they're unanswerable.
    corpus_identities = {c["identity_key"] for c in corpus}
    corpus_designs    = {c["design_key"]   for c in corpus}
    kept_queries = [q for q in queries if q["design_key"] in corpus_designs]
    answerable_identity = sum(1 for q in kept_queries
                              if q["identity_key"] in corpus_identities)

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for c in corpus + kept_queries:
            f.write(json.dumps(c) + "\n")

    logger.info("─" * 60)
    logger.info("Retrieval eval set written:")
    logger.info("  corpus images:           {}", len(corpus))
    logger.info("  queries (design-answerable):   {}", len(kept_queries))
    logger.info("  queries (identity-answerable): {}", answerable_identity)
    logger.info("  distinct corpus identities:    {}", len(corpus_identities))
    logger.info("  distinct corpus designs:       {}", len(corpus_designs))
    logger.info("  manifest: {}", manifest_path)


if __name__ == "__main__":
    main()
