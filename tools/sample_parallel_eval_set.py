#!/usr/bin/env python3
"""
Build a parallel-discrimination evaluation set.

The hard problem for trading-card image search is the "same card, different
parallel" case: identical player/set/card-number, differing only in foil/colour/
texture (base vs silver prizm vs gold refractor /10 …). CLIP is trained to be
invariant to exactly those low-level cues, so parallels collapse together. This
sampler builds the data to MEASURE that, and eval_parallel_discrimination.py
scores CLIP vs DINOv2/v3 on it.

Structure we sample from the extant (read-only) OpenSearch cluster:
  - group cards by DESIGN KEY = (player, year, set, card_number)
  - keep design groups that contain >= 2 DISTINCT parallels
  - keep only parallels that have >= MIN_PER_PARALLEL real images
    (so a "same-parallel but different listing" positive always exists for a
     leave-one-out nearest-neighbour test)

Each design group is therefore a clean closed-world discrimination test: every
candidate shares the same design, so the ONLY thing distinguishing them is the
parallel's appearance.

Ground truth (the parallel label) is seller-entered itemSpecifics.parallel and
imperfect; the evaluator reports aggregate rates, not per-card pass/fail.

Usage:
    python tools/sample_parallel_eval_set.py --groups 80 --out data/parallel_eval
    python tools/sample_parallel_eval_set.py --groups 40 --index "2026-*" --pool 40000
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from src.ingestion.opensearch_reader import get_opensearch_client

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}
MAX_IMG_BYTES = 10 * 1024 * 1024

_PUNCT = re.compile(r"[^\w\s]")
_WS    = re.compile(r"\s+")


def norm(s) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        s = s[0] if s else ""
    s = str(s).lower().strip()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def norm_number(s) -> str:
    s = norm(s).replace(" ", "")
    return re.sub(r"0+(\d)", r"\1", s)


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        v = v[0] if v else ""
    return str(v).strip()


def design_key(specs: dict) -> str | None:
    player = norm(specs.get("player"))
    cset   = norm(specs.get("set"))
    cnum   = norm_number(specs.get("cardNumber"))
    year   = norm(specs.get("year"))
    if not (player and cset and cnum):
        return None
    return f"{player}|{year}|{cset}|{cnum}"


def parallel_label(specs: dict) -> str:
    p = norm(specs.get("parallel"))
    return p if p else "base"


def scroll_pool(os_client, index_pattern: str, pool_size: int) -> list[dict]:
    """Scroll recent indices collecting cards that have the identity fields."""
    body = {
        "size": 1000,
        "query": {"bool": {
            "must": [
                {"exists": {"field": "galleryURL"}},
                {"exists": {"field": "itemSpecifics.player"}},
                {"exists": {"field": "itemSpecifics.set"}},
                {"exists": {"field": "itemSpecifics.cardNumber"}},
                {"exists": {"field": "itemSpecifics.parallel"}},
            ],
            "must_not": [{"term": {"galleryURL": "N/A"}}],
        }},
        "_source": ["id", "title", "galleryURL", "itemSpecifics"],
    }
    out: list[dict] = []
    resp = os_client.search(index=index_pattern, body=body, scroll="2m",
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
            out.append({
                "os_id":       h["_id"],
                "index":       h["_index"],
                "gallery_url": url,
                "design_key":  dk,
                "parallel":    parallel_label(specs),
                "player":      _clean(specs.get("player")),
                "year":        _clean(specs.get("year")),
                "set":         _clean(specs.get("set")),
                "card_number": _clean(specs.get("cardNumber")),
            })
        if len(out) >= pool_size:
            break
        resp = os_client.scroll(scroll_id=sid, scroll="2m", request_timeout=60)
        sid  = resp.get("_scroll_id")
        if sid is None:
            break
    logger.info("Scrolled {} candidate cards from {}", len(out), index_pattern)
    return out


def select_groups(pool: list[dict], n_groups: int, min_per_parallel: int,
                  max_parallels: int, per_parallel_cap: int) -> list[dict]:
    """Pick design groups with >=2 parallels, each with >=min_per_parallel imgs."""
    by_design: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in pool:
        by_design[c["design_key"]][c["parallel"]].append(c)

    selected: list[dict] = []
    for dk, parallels in by_design.items():
        good = {p: cs[:per_parallel_cap] for p, cs in parallels.items()
                if len(cs) >= min_per_parallel}
        if len(good) < 2:
            continue
        kept_parallels = dict(list(good.items())[:max_parallels])
        for cards in kept_parallels.values():
            selected.extend(cards)
        if len({c["design_key"] for c in selected}) >= n_groups:
            break
    return selected


def download_image(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, headers=DOWNLOAD_HEADERS, timeout=20,
                          stream=True, verify=True) as r:
            if r.status_code != 200:
                return False
            buf = io.BytesIO()
            for chunk in r.iter_content(65536):
                buf.write(chunk)
                if buf.tell() > MAX_IMG_BYTES:
                    return False
            data = buf.getvalue()
        if len(data) < 500:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        logger.debug("download failed {}: {}", url[:60], e)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", type=int, default=80, help="Target design groups.")
    ap.add_argument("--out", default="data/parallel_eval")
    ap.add_argument("--index", default="2026-*,2025-12-*",
                    help="OS index pattern to sample from.")
    ap.add_argument("--pool", type=int, default=40000,
                    help="Candidate cards to scroll before grouping.")
    ap.add_argument("--min-per-parallel", type=int, default=2)
    ap.add_argument("--max-parallels", type=int, default=4)
    ap.add_argument("--per-parallel-cap", type=int, default=3)
    args = ap.parse_args()

    out_dir = ROOT / args.out
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    os_client = get_opensearch_client()
    pool = scroll_pool(os_client, args.index, args.pool)
    selected = select_groups(pool, args.groups, args.min_per_parallel,
                             args.max_parallels, args.per_parallel_cap)
    logger.info("Selected {} cards across {} design groups — downloading …",
                len(selected), len({c['design_key'] for c in selected}))

    manifest: list[dict] = []
    ok = 0
    for i, c in enumerate(selected):
        img_path = img_dir / f"{c['os_id'].replace('/', '_')}.jpg"
        if download_image(c["gallery_url"], img_path):
            c["image_file"] = str(img_path.relative_to(out_dir))
            manifest.append(c)
            ok += 1
        if (i + 1) % 50 == 0:
            logger.info("  {}/{} downloaded ({} ok)", i + 1, len(selected), ok)
        time.sleep(0.02)

    # Re-validate groups survive image-download losses: keep only design groups
    # that still have >= 2 parallels each with >= min_per_parallel images.
    by_design: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in manifest:
        by_design[c["design_key"]][c["parallel"]].append(c)
    final: list[dict] = []
    for dk, parallels in by_design.items():
        good = {p: cs for p, cs in parallels.items() if len(cs) >= args.min_per_parallel}
        if len(good) < 2:
            continue
        for cards in good.values():
            final.extend(cards)

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for c in final:
            f.write(json.dumps(c) + "\n")

    n_groups   = len({c["design_key"] for c in final})
    n_parallel = len({(c["design_key"], c["parallel"]) for c in final})
    logger.info("─" * 60)
    logger.info("Parallel eval set written: {} cards", len(final))
    logger.info("  design groups:        {}", n_groups)
    logger.info("  (design,parallel) cells: {}", n_parallel)
    logger.info("  avg parallels/group:  {:.1f}", n_parallel / max(1, n_groups))
    logger.info("  manifest: {}", manifest_path)
    logger.info("  images:   {}", img_dir)


if __name__ == "__main__":
    main()
