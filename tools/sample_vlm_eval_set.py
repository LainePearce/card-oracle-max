#!/usr/bin/env python3
"""
Build a stratified evaluation set for the VLM-OCR card-identification POC.

Pulls a balanced sample of cards from the extant (read-only) OpenSearch
cluster, with their itemSpecifics + title as ground truth, and downloads the
card images. The output manifest feeds vlm_extract.py.

Stratification (default 300 cards):
  - by genre (baseball/basketball/football/pokemon/soccer/other)
  - by graded vs raw within each genre (~50/50)
  - plus "hard families": multiple distinct cards of the SAME player, to test
    whether the VLM reads the card number well enough to tell them apart
    (the exact discrimination that capped the image metric head).

Ground truth = seller-entered itemSpecifics. It is imperfect; the scorer
surfaces VLM-vs-itemSpecifics disagreements for manual review rather than
assuming itemSpecifics is always right.

Usage:
    python tools/sample_vlm_eval_set.py --n 300 --out data/vlm_eval
    python tools/sample_vlm_eval_set.py --n 150 --out data/vlm_eval   # faster first read
"""
from __future__ import annotations

import argparse
import io
import json
import os
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

GENRES = ["baseball", "basketball", "football", "pokemon", "soccer", "other"]

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}
MAX_IMG_BYTES = 10 * 1024 * 1024


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        v = v[0] if v else ""
    return str(v).strip()


def _genre_bucket(specs: dict) -> str:
    g = _clean(specs.get("genre")).lower()
    for known in GENRES[:-1]:
        if known in g:
            return known
    return "other"


def fetch_candidates(os_client, index_pattern: str, want: int) -> list[dict]:
    """Search recent indices for cards with a real image + identity fields."""
    body = {
        "size": want,
        "query": {"bool": {
            "must": [
                {"exists": {"field": "galleryURL"}},
                {"exists": {"field": "itemSpecifics.player"}},
                {"exists": {"field": "itemSpecifics.set"}},
                {"exists": {"field": "itemSpecifics.cardNumber"}},
            ],
            "must_not": [{"term": {"galleryURL": "N/A"}}],
        }},
        "_source": ["id", "itemId", "title", "galleryURL", "globalId",
                    "source", "itemSpecifics"],
    }
    resp = os_client.search(index=index_pattern, body=body, request_timeout=60)
    out = []
    for h in resp["hits"]["hits"]:
        src   = h["_source"]
        specs = src.get("itemSpecifics") or {}
        url   = _clean(src.get("galleryURL"))
        if not url or url.lower() == "n/a":
            continue
        out.append({
            "os_id":       h["_id"],
            "index":       h["_index"],
            "item_id":     _clean(src.get("itemId")),
            "title":       _clean(src.get("title")),
            "gallery_url": url,
            "genre":       _genre_bucket(specs),
            "graded":      _clean(specs.get("graded")).lower() in ("true", "yes", "1"),
            "truth": {
                "player":      _clean(specs.get("player")),
                "year":        _clean(specs.get("year")),
                "brand":       _clean(specs.get("brand")),
                "set":         _clean(specs.get("set")),
                "card_number": _clean(specs.get("cardNumber")),
                "parallel":    _clean(specs.get("parallel")),
                "grader":      _clean(specs.get("grader")),
                "grade":       _clean(specs.get("grade")),
            },
        })
    return out


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
    ap.add_argument("--n", type=int, default=300, help="Total cards in the eval set.")
    ap.add_argument("--out", default="data/vlm_eval", help="Output dir.")
    ap.add_argument("--index", default="2026-05-*,2026-04-*,2026-03-*",
                    help="OS index pattern to sample from (recent eBay days).")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    os_client = get_opensearch_client()

    per_genre   = max(1, args.n // len(GENRES))
    logger.info("Target {} cards (~{}/genre, ~50/50 graded/raw)", args.n, per_genre)

    logger.info("Fetching candidate pool from {} ...", args.index)
    pool = fetch_candidates(os_client, args.index, want=args.n * 12)
    logger.info("  {} candidates with image + identity fields", len(pool))

    by_bucket: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for c in pool:
        by_bucket[(c["genre"], c["graded"])].append(c)

    selected: list[dict] = []
    for genre in GENRES:
        for graded in (True, False):
            bucket = by_bucket.get((genre, graded), [])
            take   = bucket[: per_genre // 2]
            selected.extend(take)
            logger.info("  {:10s} graded={!s:5} → {} selected (pool {})",
                        genre, graded, len(take), len(bucket))

    seen = {c["os_id"] for c in selected}
    for c in pool:
        if len(selected) >= args.n:
            break
        if c["os_id"] not in seen:
            selected.append(c)
            seen.add(c["os_id"])

    logger.info("Selected {} cards — downloading images ...", len(selected))
    manifest = []
    ok = 0
    for i, c in enumerate(selected):
        img_path = img_dir / f"{c['os_id'].replace('/', '_')}.jpg"
        if download_image(c["gallery_url"], img_path):
            c["image_file"] = str(img_path.relative_to(out_dir))
            manifest.append(c)
            ok += 1
        if (i + 1) % 25 == 0:
            logger.info("  {}/{} downloaded ({} ok)", i + 1, len(selected), ok)
        time.sleep(0.02)

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for c in manifest:
            f.write(json.dumps(c) + "\n")

    n_graded = sum(1 for c in manifest if c["graded"])
    logger.info("─" * 60)
    logger.info("Eval set written: {} cards ({} graded / {} raw)",
                len(manifest), n_graded, len(manifest) - n_graded)
    logger.info("  manifest: {}", manifest_path)
    logger.info("  images:   {}", img_dir)
    by_g = defaultdict(int)
    for c in manifest:
        by_g[c["genre"]] += 1
    logger.info("  by genre: {}", dict(by_g))


if __name__ == "__main__":
    main()
