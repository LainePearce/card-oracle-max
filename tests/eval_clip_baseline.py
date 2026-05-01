"""Baseline CLIP Recall Evaluation — Qdrant vector quality benchmark.

Measures Recall@1 / @5 / @10 / @20 for the existing CLIP ViT-L/14 image
embeddings stored in Qdrant, without any metric projection head applied.
This establishes the baseline against which a trained projection head will
be compared.

How it works
------------
1. Scroll N * oversample_factor candidate points from Qdrant (has_image=true,
   specifics_source="ebay", active=true).
2. Build an identity key for each point from its payload:
   (brand, set, card_number, year, player) — all five must be non-blank.
3. For each candidate, scroll Qdrant with a payload filter to find every OTHER
   point sharing the same identity key (ground truth set). Skip singletons
   (no second image of this card in the corpus — nothing to recall).
4. Retrieve the stored CLIP image vector for each query point.
5. Run a Qdrant ANN search with that vector, exclude the query point itself,
   and check whether any of the top-K results appear in the ground truth set.
6. Report Recall@K, mean first-hit rank, and breakdowns by ground truth size,
   genre, and graded/raw condition.

No images are downloaded and no GPU is required. Everything is self-contained
within Qdrant and .env credentials.

Usage
-----
    # From project root:
    python tests/eval_clip_baseline.py

    # Custom options:
    python tests/eval_clip_baseline.py \\
        --sample-size 500 \\
        --oversample-factor 4 \\
        --gt-cap 500 \\
        --output-dir tests/results/ \\
        --seed 42

Output
------
    tests/results/clip_baseline_YYYYMMDD_HHMMSS.json   — full per-query results
    tests/results/clip_baseline_YYYYMMDD_HHMMSS_summary.json — aggregated metrics
    Printed table to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Optional

# ── Path setup — allow running from any working directory ──────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from src.ingestion.qdrant_writer import get_qdrant_client


# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION = os.environ.get("QDRANT_COLLECTION", "cards")

# Fields that must ALL be present and non-blank for a point to be included.
REQUIRED_PAYLOAD_FIELDS = ["brand", "player", "set", "card_number", "year"]

# Values that are treated as blank regardless of the field.
_BLANK_VALUES = {"", "none", "null", "n/a", "unknown", "undefined", "0"}

# Recall ranks to evaluate.
RECALL_KS = [1, 5, 10, 20]

# Ground truth scroll page size — Qdrant returns at most this many per page.
_GT_PAGE_SIZE = 250


# ── Identity key ──────────────────────────────────────────────────────────────

def build_identity_key(payload: dict) -> str | None:
    """
    Build a normalised identity string from the five essential fields.
    Returns None if any field is missing or blank — the point is excluded.

    Two cards are "the same" if they share the same identity key.
    Grading condition (grader/grade) is intentionally excluded so that a
    raw copy and a PSA 10 copy of the same card count as correct matches.
    """
    parts = []
    for field in REQUIRED_PAYLOAD_FIELDS:
        raw = payload.get(field)
        if raw is None:
            return None
        val = str(raw).lower().strip()
        if val in _BLANK_VALUES:
            return None
        parts.append(val)
    return "|".join(parts)


# ── Ground truth lookup ───────────────────────────────────────────────────────

def _identity_filter(payload: dict, exclude_id: int | str) -> Filter:
    """Build a Qdrant payload filter matching all five essential fields."""
    conditions = []
    for field in REQUIRED_PAYLOAD_FIELDS:
        val = str(payload.get(field, "")).lower().strip()
        conditions.append(FieldCondition(key=field, match=MatchValue(value=val)))
    conditions.append(FieldCondition(key="has_image", match=MatchValue(value=True)))
    return Filter(must=conditions)


def get_ground_truth(
    qdrant: QdrantClient,
    payload: dict,
    exclude_id: int | str,
    cap: int = 500,
) -> set[str]:
    """
    Scroll Qdrant for all points sharing the same identity key as `payload`,
    excluding the query point itself.

    `cap` prevents very popular cards (e.g. Charizard Base Set) from dominating
    the ground truth lookup time. 500 is plenty for recall evaluation.
    """
    gt: set[str] = set()
    identity_filter = _identity_filter(payload, exclude_id)
    offset = None

    while len(gt) < cap:
        page, next_offset = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=identity_filter,
            limit=_GT_PAGE_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for p in page:
            if str(p.id) != str(exclude_id):
                gt.add(str(p.id))
        if next_offset is None or len(page) < _GT_PAGE_SIZE:
            break
        offset = next_offset

    return gt


# ── Sampling ──────────────────────────────────────────────────────────────────

_CANDIDATE_FILTER = Filter(
    must=[
        FieldCondition(key="has_image",        match=MatchValue(value=True)),
        FieldCondition(key="specifics_source",  match=MatchValue(value="ebay")),
        FieldCondition(key="active",            match=MatchValue(value=True)),
    ]
)


def scroll_candidates(
    qdrant: QdrantClient,
    target: int,
    oversample_factor: int,
    seed: int,
) -> list:
    """
    Scroll `target * oversample_factor` candidate points from Qdrant,
    filter to those with valid identity keys, shuffle, and return up to
    `target` points.

    Oversample because:
      - ~20-30% will be singletons (only one image of that card)
      - Some identity keys will be incomplete (missing essential fields)
    """
    fetch = target * oversample_factor
    logger.info(
        "Scrolling up to {:,} candidate points from Qdrant (filter: "
        "has_image=true, specifics_source=ebay, active=true) …",
        fetch,
    )

    all_points = []
    offset = None
    page_size = min(1_000, fetch)

    while len(all_points) < fetch:
        remaining = fetch - len(all_points)
        page, next_offset = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=_CANDIDATE_FILTER,
            limit=min(page_size, remaining),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(page)
        if next_offset is None or len(page) == 0:
            break
        offset = next_offset

    logger.info("Scrolled {:,} raw candidates", len(all_points))

    # Filter to those with valid identity keys
    valid = [p for p in all_points if build_identity_key(p.payload) is not None]
    logger.info(
        "{:,} / {:,} have complete identity keys ({:.1f}%)",
        len(valid),
        len(all_points),
        100 * len(valid) / max(len(all_points), 1),
    )

    random.seed(seed)
    random.shuffle(valid)
    return valid[:target]


# ── Vector retrieval ──────────────────────────────────────────────────────────

def retrieve_vectors(
    qdrant: QdrantClient,
    ids: list[int | str],
    batch_size: int = 100,
) -> dict[str, list[float]]:
    """
    Retrieve the stored "image" CLIP vector for each point ID.
    Returns a dict: str(point_id) → vector (list[float]).
    Points without an image vector are silently skipped.
    """
    id_to_vec: dict[str, list[float]] = {}

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        points = qdrant.retrieve(
            collection_name=COLLECTION,
            ids=batch_ids,
            with_vectors=True,
            with_payload=False,
        )
        for p in points:
            vectors = p.vector
            if isinstance(vectors, dict):
                vec = vectors.get("image")
            else:
                vec = vectors  # flat vector (single-vector collection)
            if vec is not None:
                id_to_vec[str(p.id)] = vec

    return id_to_vec


# ── ANN search ────────────────────────────────────────────────────────────────

def search_similar(
    qdrant: QdrantClient,
    vec: list[float],
    query_id: int | str,
    top_k: int = 21,
) -> list[str]:
    """
    Run a Qdrant ANN search using the raw CLIP image vector (no projection,
    no score threshold). Returns up to top_k-1 result IDs, excluding the
    query point itself.
    """
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=("image", vec),
        limit=top_k,
        with_payload=False,
        with_vectors=False,
    )
    return [str(r.id) for r in results if str(r.id) != str(query_id)]


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> int:
    """1 if any of the top-k retrieved IDs appear in ground_truth, else 0."""
    return int(any(r in ground_truth for r in retrieved[:k]))


def first_hit_rank(retrieved: list[str], ground_truth: set[str]) -> int | None:
    """1-based rank of the first correct result, or None if none found in top-20."""
    for i, r in enumerate(retrieved[:20]):
        if r in ground_truth:
            return i + 1
    return None


# ── Reporting ─────────────────────────────────────────────────────────────────

def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def print_table(label: str, rows: list[dict], ks: list[int] = RECALL_KS) -> None:
    """Pretty-print a breakdown table to stdout."""
    header = f"{'Group':<28}" + "".join(f"  R@{k:<5}" for k in ks) + "  MeanRank  n"
    print(f"\n{label}")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for row in rows:
        n = row["n"]
        line = f"{row['label']:<28}"
        for k in ks:
            v = _pct(row.get(f"recall_{k}", 0), n)
            line += f"  {v:<7}"
        mr = row.get("mean_rank")
        line += f"  {mr:<9}" if mr else "  n/a      "
        line += f"  {n}"
        print(line)
    print("─" * len(header))


def aggregate(query_results: list[dict], group_key: str, group_label_fn) -> list[dict]:
    """Aggregate recall metrics by a grouping key."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in query_results:
        groups[group_label_fn(r)].append(r)

    rows = []
    for label in sorted(groups):
        grp = groups[label]
        n = len(grp)
        row = {"label": label, "n": n}
        for k in RECALL_KS:
            row[f"recall_{k}"] = sum(r[f"recall_{k}"] for r in grp)
        ranks = [r["first_hit_rank"] for r in grp if r["first_hit_rank"] is not None]
        row["mean_rank"] = f"{mean(ranks):.1f}" if ranks else None
        rows.append(row)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(
    sample_size: int = 1_000,
    oversample_factor: int = 3,
    gt_cap: int = 500,
    output_dir: str = "tests/results",
    seed: int = 42,
) -> dict:
    """
    Run the full baseline evaluation and return a summary dict.
    Also writes per-query and summary JSON to output_dir.
    """
    out_dir = _ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("CLIP Baseline Recall Evaluation")
    logger.info("  Collection:  {}", COLLECTION)
    logger.info("  Sample size: {:,}", sample_size)
    logger.info("  Oversample:  {}×", oversample_factor)
    logger.info("  GT cap:      {:,}", gt_cap)
    logger.info("  Seed:        {}", seed)
    logger.info("=" * 60)

    qdrant = get_qdrant_client()

    # ── Collection info ───────────────────────────────────────────────────────
    info = qdrant.get_collection(COLLECTION)
    total_points = info.points_count
    logger.info("Collection '{}' has {:,} points", COLLECTION, total_points)

    # ── Phase 1: Sample candidates ────────────────────────────────────────────
    t0 = time.perf_counter()
    candidates = scroll_candidates(qdrant, sample_size, oversample_factor, seed)
    logger.info(
        "Sampled {:,} candidates in {:.1f}s", len(candidates), time.perf_counter() - t0
    )

    if not candidates:
        logger.error("No valid candidates found — check collection and filters.")
        sys.exit(1)

    # ── Phase 2: Retrieve stored CLIP vectors ─────────────────────────────────
    logger.info("Retrieving stored image vectors …")
    t0 = time.perf_counter()
    ids = [p.id for p in candidates]
    id_to_vec = retrieve_vectors(qdrant, ids)
    missing_vec = sum(1 for p in candidates if str(p.id) not in id_to_vec)
    if missing_vec:
        logger.warning("{:,} candidates had no image vector — skipping", missing_vec)
    logger.info(
        "Retrieved {:,} vectors in {:.1f}s",
        len(id_to_vec),
        time.perf_counter() - t0,
    )

    # ── Phase 3: Build ground truth ───────────────────────────────────────────
    logger.info("Building ground truth for {:,} candidates …", len(candidates))
    t0 = time.perf_counter()

    ground_truths: dict[str, set[str]] = {}
    singletons = 0

    for i, point in enumerate(candidates):
        pid = str(point.id)
        if pid not in id_to_vec:
            continue
        gt = get_ground_truth(qdrant, point.payload, point.id, cap=gt_cap)
        if not gt:
            singletons += 1
        else:
            ground_truths[pid] = gt
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = (len(candidates) - i - 1) / rate
            logger.info(
                "  Ground truth: {:,}/{:,} — {:,} singletons skipped — "
                "ETA {:.0f}s",
                i + 1,
                len(candidates),
                singletons,
                remaining,
            )

    gt_elapsed = time.perf_counter() - t0
    logger.info(
        "Ground truth done in {:.1f}s — {:,} queries, {:,} singletons excluded",
        gt_elapsed,
        len(ground_truths),
        singletons,
    )

    if not ground_truths:
        logger.error(
            "All candidates were singletons — no ground truth to evaluate against."
        )
        sys.exit(1)

    # ── Phase 4: ANN search and recall computation ────────────────────────────
    logger.info("Running ANN searches for {:,} queries …", len(ground_truths))
    t0 = time.perf_counter()

    query_results: list[dict] = []
    search_latencies: list[float] = []

    eligible = [p for p in candidates if str(p.id) in ground_truths]

    for i, point in enumerate(eligible):
        pid = str(point.id)
        vec = id_to_vec[pid]
        gt  = ground_truths[pid]

        t_search = time.perf_counter()
        retrieved = search_similar(qdrant, vec, point.id, top_k=21)
        search_latencies.append((time.perf_counter() - t_search) * 1000)

        fhr = first_hit_rank(retrieved, gt)

        payload = point.payload
        result = {
            "id":               pid,
            "identity_key":     build_identity_key(payload),
            "gt_size":          len(gt),
            "retrieved_top20":  retrieved[:20],
            "genre":            str(payload.get("genre") or "unknown").lower().strip() or "unknown",
            "graded":           bool(payload.get("graded", False)),
            "grader":           str(payload.get("grader") or "").lower().strip(),
            "grade":            str(payload.get("grade") or "").lower().strip(),
            "brand":            str(payload.get("brand") or "").lower().strip(),
            "first_hit_rank":   fhr,
        }
        for k in RECALL_KS:
            result[f"recall_{k}"] = recall_at_k(retrieved, gt, k)

        query_results.append(result)

        if (i + 1) % 100 == 0:
            logger.info("  Searched {:,}/{:,}", i + 1, len(eligible))

    search_elapsed = time.perf_counter() - t0
    logger.info(
        "ANN searches done in {:.1f}s — {:.1f}ms median latency",
        search_elapsed,
        median(search_latencies) if search_latencies else 0,
    )

    # ── Phase 5: Aggregate and report ─────────────────────────────────────────
    n = len(query_results)
    if n == 0:
        logger.error("No evaluated queries — cannot compute metrics.")
        sys.exit(1)

    overall: dict = {"n": n}
    for k in RECALL_KS:
        hits = sum(r[f"recall_{k}"] for r in query_results)
        overall[f"recall_{k}"] = hits
        overall[f"recall_{k}_pct"] = round(100 * hits / n, 2)

    ranks = [r["first_hit_rank"] for r in query_results if r["first_hit_rank"] is not None]
    overall["mean_first_hit_rank"]   = round(mean(ranks), 2)   if ranks else None
    overall["median_first_hit_rank"] = round(median(ranks), 2) if ranks else None
    overall["pct_with_hit_in_top20"] = round(100 * len(ranks) / n, 2)

    # Breakdown: by ground truth size bucket
    def gt_bucket(r: dict) -> str:
        s = r["gt_size"]
        if s <= 4:  return "2–4 matches"
        if s <= 20: return "5–20 matches"
        return "21+ matches"

    by_gt_size = aggregate(query_results, "gt_size", gt_bucket)

    # Breakdown: by genre
    def genre_label(r: dict) -> str:
        g = r["genre"]
        return g if g in ("pokemon", "football", "basketball", "baseball") else "other"

    by_genre = aggregate(query_results, "genre", genre_label)

    # Breakdown: graded vs raw
    def graded_label(r: dict) -> str:
        return "graded" if r["graded"] else "raw"

    by_condition = aggregate(query_results, "graded", graded_label)

    # Breakdown: by grader (graded cards only)
    graded_results = [r for r in query_results if r["graded"]]
    def grader_label(r: dict) -> str:
        g = r["grader"]
        return g if g in ("psa", "bgs", "cgc", "sgc") else "other_grader"

    by_grader = aggregate(graded_results, "grader", grader_label) if graded_results else []

    # ── Print report ──────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("CLIP Baseline Recall Evaluation — Results")
    print(f"Collection: {COLLECTION}  |  Evaluated queries: {n:,}")
    print(f"Singletons excluded: {singletons:,}  |  Timestamp: {timestamp}")
    print(sep)

    print("\nOverall")
    print("─" * 40)
    for k in RECALL_KS:
        hits = overall[f"recall_{k}"]
        pct  = overall[f"recall_{k}_pct"]
        print(f"  Recall@{k:<3}  {pct:>6.1f}%  ({hits:,}/{n:,})")
    print(f"\n  Mean first-hit rank:    {overall['mean_first_hit_rank']}")
    print(f"  Median first-hit rank:  {overall['median_first_hit_rank']}")
    print(f"  Any hit in top-20:      {overall['pct_with_hit_in_top20']}%")
    print(f"\n  Median ANN latency:     {median(search_latencies):.1f}ms")
    print(f"  p95 ANN latency:        {sorted(search_latencies)[int(0.95*len(search_latencies))]:.1f}ms")

    print_table("By ground truth size", by_gt_size)
    print_table("By genre",             by_genre)
    print_table("By condition",         by_condition)
    if by_grader:
        print_table("By grader (graded cards only)", by_grader)

    # GT size distribution
    print("\nGround truth size distribution")
    print("─" * 40)
    buckets = defaultdict(int)
    for r in query_results:
        s = r["gt_size"]
        if s <= 4:    buckets["2–4"]   += 1
        elif s <= 20: buckets["5–20"]  += 1
        elif s <= 50: buckets["21–50"] += 1
        else:         buckets["51+"]   += 1
    for label, count in sorted(buckets.items()):
        print(f"  {label:<8}  {count:>5,}  ({100*count/n:.1f}%)")

    print(f"\n{sep}\n")

    # ── Write results ─────────────────────────────────────────────────────────
    per_query_path = out_dir / f"clip_baseline_{timestamp}.json"
    summary_path   = out_dir / f"clip_baseline_{timestamp}_summary.json"

    summary = {
        "timestamp":        timestamp,
        "collection":       COLLECTION,
        "total_points":     total_points,
        "sample_size":      sample_size,
        "evaluated_queries": n,
        "singletons_excluded": singletons,
        "overall":          overall,
        "by_gt_size":       by_gt_size,
        "by_genre":         by_genre,
        "by_condition":     by_condition,
        "by_grader":        by_grader,
        "latency_ms": {
            "median": round(median(search_latencies), 2) if search_latencies else None,
            "p95":    round(sorted(search_latencies)[int(0.95 * len(search_latencies))], 2)
                      if search_latencies else None,
            "p99":    round(sorted(search_latencies)[int(0.99 * len(search_latencies))], 2)
                      if search_latencies else None,
        },
        "config": {
            "oversample_factor": oversample_factor,
            "gt_cap":            gt_cap,
            "seed":              seed,
            "recall_ks":         RECALL_KS,
        },
    }

    # Per-query file: omit retrieved_top20 to keep size manageable
    per_query_slim = [
        {k: v for k, v in r.items() if k != "retrieved_top20"}
        for r in query_results
    ]

    with open(per_query_path, "w") as f:
        json.dump(per_query_slim, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Per-query results → {}", per_query_path)
    logger.info("Summary           → {}", summary_path)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLIP baseline recall evaluation against Qdrant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=1_000,
        help="Number of query points to evaluate.",
    )
    p.add_argument(
        "--oversample-factor",
        type=int,
        default=3,
        help="Scroll this many × sample_size candidates to account for singletons "
             "and incomplete identity keys.",
    )
    p.add_argument(
        "--gt-cap",
        type=int,
        default=500,
        help="Maximum ground truth size per query (prevents very popular cards "
             "from dominating ground truth lookup time).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="tests/results",
        help="Directory to write JSON result files (relative to project root).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate shuffling (reproducibility).",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Override QDRANT_COLLECTION env var.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.collection:
        os.environ["QDRANT_COLLECTION"] = args.collection
        # Reload the module-level constant
        import importlib
        import src.ingestion.qdrant_writer as _qw
        importlib.reload(_qw)

    run_evaluation(
        sample_size=args.sample_size,
        oversample_factor=args.oversample_factor,
        gt_cap=args.gt_cap,
        output_dir=args.output_dir,
        seed=args.seed,
    )
