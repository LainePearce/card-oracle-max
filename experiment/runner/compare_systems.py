"""Quick comparative search test: Qdrant ANN vs OpenSearch KNN.

Samples N random documents from a date range, uses their images as queries,
runs both systems side-by-side, and reports overlap, latency, and rank correlation.

Usage:
    python -m experiment.runner.compare_systems \
        --index-pattern "2025-02-17,2025-02-18,2025-02-19,2025-02-20,2025-02-21" \
        --num-queries 100 \
        --top-k 20 \
        --image-device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def sample_query_docs(
    os_client,
    index_pattern: str,
    num_queries: int,
    seed: int = 42,
) -> list[dict]:
    """
    Sample random documents from OpenSearch to use as queries.
    Each doc must have a galleryURL (for image encoding) and itemSpecifics.
    """
    from src.ingestion.opensearch_reader import scroll_index

    logger.info("Sampling {} query documents from {}", num_queries, index_pattern)

    # Collect candidates with galleryURL
    candidates = []
    for hit in scroll_index(os_client, index_pattern):
        src = hit["_source"]
        if src.get("galleryURL") and src.get("itemSpecifics"):
            candidates.append(hit)
        # Collect 10x candidates for random sampling
        if len(candidates) >= num_queries * 10:
            break

    rng = random.Random(seed)
    sampled = rng.sample(candidates, min(num_queries, len(candidates)))
    logger.info("Sampled {} documents (from {} candidates)", len(sampled), len(candidates))
    return sampled


def encode_query(hit: dict, image_encoder, text_encoder) -> tuple[list[float] | None, list[float] | None]:
    """Encode a single document's image and specifics for use as a query."""
    from src.embeddings.text_encoder import format_specifics

    src = hit["_source"]

    # Image embedding
    image_vec = image_encoder.encode_url(src.get("galleryURL"))

    # Specifics embedding
    specs = src.get("itemSpecifics") or {}
    text = format_specifics(specs)
    specifics_vec = text_encoder.encode(text) if text else None

    return (
        image_vec.tolist() if image_vec is not None else None,
        specifics_vec.tolist() if specifics_vec is not None else None,
    )


def run_comparison(
    index_pattern: str,
    num_queries: int = 100,
    top_k: int = 20,
    image_device: str = "cuda",
    output_dir: str | None = None,
    seed: int = 42,
) -> dict:
    """
    Run a side-by-side comparison of Qdrant ANN vs OpenSearch KNN.

    For each sampled query:
      1. Encode the document's image with CLIP
      2. Search Qdrant using image_search()
      3. Search OpenSearch using opensearch_knn_search()
      4. Compare result sets: overlap, rank correlation, latency

    Returns summary statistics dict.
    """
    from src.embeddings.image_encoder import ImageEncoder
    from src.embeddings.text_encoder import TextEncoder
    from src.ingestion.opensearch_reader import get_opensearch_client
    from src.ingestion.qdrant_writer import get_qdrant_client
    from src.search.qdrant_search import image_search as qdrant_image_search
    from src.search.opensearch_search import opensearch_knn_search

    # Initialize clients
    os_client = get_opensearch_client()
    qdrant_client = get_qdrant_client()
    collection = os.environ.get("QDRANT_COLLECTION", "cards")

    # Initialize encoders
    logger.info("Loading encoders...")
    # Qdrant encoder: ViT-L/14 openai → 768-dim (matches Qdrant collection)
    image_encoder = ImageEncoder(device=image_device)
    # OpenSearch encoder: ViT-B-32 openai → 512-dim
    # The extant OS cluster was indexed with Xenova/clip-vit-base-patch32, which is
    # OpenAI CLIP ViT-B/32 exported to ONNX. Weights are identical to open_clip
    # ViT-B-32 pretrained="openai" — same vectors, different runtime.
    os_image_encoder = ImageEncoder(
        model_name="ViT-B-32",
        pretrained="openai",
        device=image_device,
    )
    text_encoder = TextEncoder(device="cpu")

    # Sample query documents
    query_docs = sample_query_docs(os_client, index_pattern, num_queries, seed)

    if not query_docs:
        logger.error("No query documents found")
        return {}

    # Run queries
    qdrant_latencies = []
    os_latencies = []
    overlaps = []
    rank_correlations = []
    query_details = []

    for i, hit in enumerate(query_docs):
        src = hit["_source"]
        doc_id = hit["_id"]

        # Encode query image — two encoders for the two systems
        image_vec, specifics_vec = encode_query(hit, image_encoder, text_encoder)
        if image_vec is None:
            logger.warning("Skipping doc {} — image encoding failed", doc_id)
            continue
        # 512-dim vector for OpenSearch (indexed with ViT-B-32)
        os_image_vec = os_image_encoder.encode_url(src.get("galleryURL"))
        if os_image_vec is None:
            logger.warning("Skipping doc {} — OS image encoding failed", doc_id)
            continue

        # Search Qdrant
        try:
            qdrant_results, qdrant_timings = qdrant_image_search(
                qdrant_client,
                image_vec,
                collection=collection,
                top_k=top_k,
            )
            qdrant_latencies.append(qdrant_timings.query_ms)
            qdrant_ids = [r.os_id for r in qdrant_results]
        except Exception as e:
            logger.warning("Qdrant query failed for doc {}: {}", doc_id, e)
            continue

        # Search OpenSearch
        try:
            os_results, os_timings = opensearch_knn_search(
                os_client,
                os_image_vec.tolist(),
                index_pattern=index_pattern,
                top_k=top_k,
            )
            os_latencies.append(os_timings.query_ms)
            os_ids = [r.doc_id for r in os_results]
        except Exception as e:
            logger.warning("OpenSearch query failed for doc {}: {}", doc_id, e)
            continue

        # Compute overlap
        qdrant_set = set(qdrant_ids[:top_k])
        os_set = set(os_ids[:top_k])
        overlap = len(qdrant_set & os_set) / top_k if top_k > 0 else 0.0
        overlaps.append(overlap)

        # Compute Spearman rank correlation on shared items
        shared = qdrant_set & os_set
        if len(shared) >= 3:
            qdrant_ranks = {id_: rank for rank, id_ in enumerate(qdrant_ids)}
            os_ranks = {id_: rank for rank, id_ in enumerate(os_ids)}
            qdrant_r = [qdrant_ranks[s] for s in shared]
            os_r = [os_ranks[s] for s in shared]
            rho = _spearman_rho(qdrant_r, os_r)
            rank_correlations.append(rho)
        else:
            rank_correlations.append(None)

        # Store per-query detail
        detail = {
            "query_doc_id": doc_id,
            "query_item_id": src.get("itemId"),
            "query_title": src.get("title", "")[:100],
            "qdrant_top5": qdrant_ids[:5],
            "os_top5": os_ids[:5],
            "overlap": overlap,
            "rank_correlation": rank_correlations[-1],
            "qdrant_ms": qdrant_timings.query_ms,
            "os_ms": os_timings.query_ms,
        }
        query_details.append(detail)

        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: {}/{} queries | avg overlap={:.2f} | qdrant p50={:.1f}ms | os p50={:.1f}ms",
                i + 1, len(query_docs),
                statistics.mean(overlaps),
                statistics.median(qdrant_latencies),
                statistics.median(os_latencies),
            )

    # Compute summary statistics
    valid_rhos = [r for r in rank_correlations if r is not None]

    summary = {
        "metadata": {
            "index_pattern": index_pattern,
            "num_queries": len(query_details),
            "top_k": top_k,
            "seed": seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "overlap": {
            "mean": _safe_mean(overlaps),
            "median": _safe_median(overlaps),
            "min": min(overlaps) if overlaps else None,
            "max": max(overlaps) if overlaps else None,
            "std": _safe_stdev(overlaps),
        },
        "rank_correlation": {
            "mean": _safe_mean(valid_rhos),
            "median": _safe_median(valid_rhos),
            "n_computable": len(valid_rhos),
        },
        "latency_qdrant_ms": _percentile_stats(qdrant_latencies),
        "latency_opensearch_ms": _percentile_stats(os_latencies),
        "latency_speedup": {
            "median_ratio": (
                _safe_median(os_latencies) / _safe_median(qdrant_latencies)
                if qdrant_latencies and os_latencies and _safe_median(qdrant_latencies) > 0
                else None
            ),
        },
    }

    # Print report
    _print_report(summary)

    # Save results
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "comparison_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(out / "comparison_details.json", "w") as f:
            json.dump(query_details, f, indent=2, default=str)
        logger.info("Results saved to {}", output_dir)

    return summary


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two rank lists."""
    n = len(x)
    if n < 2:
        return 0.0
    # Convert to ranks
    x_ranks = _to_ranks(x)
    y_ranks = _to_ranks(y)
    d_sq = sum((xr - yr) ** 2 for xr, yr in zip(x_ranks, y_ranks))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def _to_ranks(values: list[float]) -> list[float]:
    """Convert values to 1-based ranks (average rank for ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _percentile_stats(values: list[float]) -> dict:
    """Compute p50, p95, p99, min, max, mean for a list of values."""
    if not values:
        return {"p50": None, "p95": None, "p99": None, "min": None, "max": None, "mean": None}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "p50": sorted_vals[int(n * 0.50)],
        "p95": sorted_vals[min(int(n * 0.95), n - 1)],
        "p99": sorted_vals[min(int(n * 0.99), n - 1)],
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": statistics.mean(values),
        "count": n,
    }


def _print_report(summary: dict) -> None:
    """Print a human-readable comparison report."""
    print("\n" + "=" * 70)
    print("  QDRANT vs OPENSEARCH — COMPARATIVE SEARCH REPORT")
    print("=" * 70)

    meta = summary["metadata"]
    print(f"\n  Queries:        {meta['num_queries']}")
    print(f"  Top-K:          {meta['top_k']}")
    print(f"  Index pattern:  {meta['index_pattern']}")

    print("\n--- Result Overlap ---")
    ov = summary["overlap"]
    print(f"  Mean overlap:   {ov['mean']:.3f}" if ov['mean'] is not None else "  Mean overlap:   N/A")
    print(f"  Median overlap: {ov['median']:.3f}" if ov['median'] is not None else "  Median overlap: N/A")
    print(f"  Min / Max:      {ov['min']:.3f} / {ov['max']:.3f}" if ov['min'] is not None else "  Min / Max:      N/A")

    print("\n--- Rank Correlation (Spearman rho on shared results) ---")
    rc = summary["rank_correlation"]
    print(f"  Mean rho:       {rc['mean']:.3f}" if rc['mean'] is not None else "  Mean rho:       N/A")
    print(f"  Computable:     {rc['n_computable']} / {meta['num_queries']}")

    print("\n--- Latency (ms) ---")
    ql = summary["latency_qdrant_ms"]
    ol = summary["latency_opensearch_ms"]
    speedup = summary["latency_speedup"]

    header = f"  {'':15s} {'Qdrant':>10s} {'OpenSearch':>12s}"
    print(header)
    print(f"  {'p50':15s} {ql['p50']:10.1f} {ol['p50']:12.1f}" if ql['p50'] is not None and ol['p50'] is not None else "")
    print(f"  {'p95':15s} {ql['p95']:10.1f} {ol['p95']:12.1f}" if ql['p95'] is not None and ol['p95'] is not None else "")
    print(f"  {'p99':15s} {ql['p99']:10.1f} {ol['p99']:12.1f}" if ql['p99'] is not None and ol['p99'] is not None else "")
    print(f"  {'mean':15s} {ql['mean']:10.1f} {ol['mean']:12.1f}" if ql['mean'] is not None and ol['mean'] is not None else "")

    if speedup.get("median_ratio") is not None:
        print(f"\n  Median speedup: {speedup['median_ratio']:.2f}x (OpenSearch / Qdrant)")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare Qdrant ANN vs OpenSearch KNN")
    parser.add_argument(
        "--index-pattern", required=True,
        help="OpenSearch index pattern (e.g. '2025-02-17,2025-02-18')",
    )
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--image-device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results (default: experiment/results/<timestamp>/)",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"experiment/results/{ts}"

    summary = run_comparison(
        index_pattern=args.index_pattern,
        num_queries=args.num_queries,
        top_k=args.top_k,
        image_device=args.image_device,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
