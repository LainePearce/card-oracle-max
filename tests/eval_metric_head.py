"""
Evaluate the metric head against the baseline CLIP vectors.

Runs entirely locally — no live system access required.
Inputs:
    data/metric_dataset.parquet      — produced by tools/extract_metric_dataset.py
    models/metric_head_v1.pt         — produced by src/embeddings/metric_head.py

Outputs:
    tests/results/metric_head/
        eval_report.json             — full numeric results
        eval_summary.txt             — human-readable comparison
        recall_curve.png             — Recall@K plot (if matplotlib available)

Usage:
    .venv-local/bin/python tests/eval_metric_head.py
    .venv-local/bin/python tests/eval_metric_head.py --checkpoint models/metric_head_v1.pt
    .venv-local/bin/python tests/eval_metric_head.py --n-queries 2000 --tier tier2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch

from src.embeddings.metric_head import MetricHead, load_metric_head, recall_at_k

RESULTS_DIR = _ROOT / "tests" / "results" / "metric_head"


# ── Helpers ───────────────────────────────────────────────────────────────────────

def load_eval_data(parquet_path: Path, tier: str = "tier3"):
    """
    Load vectors and identity labels from parquet.
    Returns: (vectors: np.ndarray, labels: np.ndarray, label_strs: list[str])
    """
    table  = pq.read_table(str(parquet_path), columns=["image_vec", tier, "genre", "player", "card_set"])
    raw    = table.column("image_vec").to_pylist()
    vecs   = np.array(raw, dtype=np.float32)
    strs   = table.column(tier).to_pylist()
    unique = sorted(set(strs))
    k2i    = {k: i for i, k in enumerate(unique)}
    labels = np.array([k2i[s] for s in strs], dtype=np.int64)
    genres = table.column("genre").to_pylist()
    return vecs, labels, unique, genres


def apply_head_batched(head: MetricHead, vectors: np.ndarray, batch: int = 2048,
                       device: str = "cpu") -> np.ndarray:
    """Apply metric head to all vectors in batches."""
    head.eval()
    out_parts = []
    t = torch.tensor(vectors, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(t), batch):
            out_parts.append(head(t[i:i+batch].to(device)).cpu().numpy())
    return np.vstack(out_parts) if out_parts else np.zeros((0, head.output_dim))


def recall_by_genre(embeddings: np.ndarray, labels: np.ndarray,
                    genres: list[str], k: int = 10, n_per_genre: int = 500) -> dict:
    """Compute Recall@K broken down by genre."""
    genre_groups: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(genres):
        genre_groups[(g or "unknown").lower()].append(i)

    results = {}
    for genre, idxs in sorted(genre_groups.items()):
        if len(idxs) < 20:
            continue
        idxs_arr = np.array(idxs)
        sub_emb  = embeddings[idxs_arr]
        sub_lbl  = labels[idxs_arr]
        r = recall_at_k(sub_emb, sub_lbl, k_values=[k], n_queries=min(n_per_genre, len(idxs_arr)))
        results[genre] = r[k]

    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


def hard_negative_rejection_rate(embeddings: np.ndarray, labels: np.ndarray,
                                  n_queries: int = 500, seed: int = 42) -> float:
    """
    For each query, check whether the nearest neighbour shares the same label.
    Higher = better (more often the nearest neighbour is the same card).
    """
    rng  = np.random.RandomState(seed)
    unique, counts = np.unique(labels, return_counts=True)
    valid = unique[counts >= 2]
    valid_idx = np.where(np.isin(labels, valid))[0]
    q_idx = rng.choice(valid_idx, size=min(n_queries, len(valid_idx)), replace=False)

    correct = 0
    for qi in q_idx:
        sims = embeddings[qi] @ embeddings.T
        sims[qi] = -np.inf
        nn = np.argmax(sims)
        if labels[nn] == labels[qi]:
            correct += 1

    return correct / len(q_idx)


def top10_hard_keys(embeddings: np.ndarray, labels: np.ndarray,
                    label_strs: list[str], n: int = 10) -> list[dict]:
    """
    Find the N identity keys where baseline struggles most (low Recall@10).
    Returns list of {key, count, recall10}.
    Useful for diagnosing where the metric head helps or doesn't.
    """
    unique, counts = np.unique(labels, return_counts=True)
    large_classes  = unique[counts >= 5][:200]  # cap to keep tractable

    results = []
    for cls in large_classes:
        cls_idx  = np.where(labels == cls)[0][:50]  # sample 50 per class
        cls_emb  = embeddings[cls_idx]
        cls_lbl  = labels[cls_idx]
        r = recall_at_k(cls_emb, embeddings, k_values=[10], n_queries=len(cls_idx))
        # Actually compute per-class by doing recall within the full set
        # Simpler: count how many top-10 for each cls_idx are same class
        hits = 0
        for qi in cls_idx:
            sims = cls_emb[qi] @ embeddings.T
            sims[qi] = -np.inf
            top10 = np.argpartition(sims, -10)[-10:]
            hits += (labels[top10] == cls).sum()
        recall = hits / (len(cls_idx) * 10)
        results.append({
            "key":     label_strs[cls],
            "count":   int(counts[unique.tolist().index(cls)]),
            "recall10": round(recall, 3),
        })

    results.sort(key=lambda x: x["recall10"])
    return results[:n]


# ── Main evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    dataset_path:    Path,
    checkpoint_path: Path | None,
    n_queries:       int,
    tier:            str,
    device:          str,
) -> dict:
    print(f"\n{'='*70}")
    print("METRIC HEAD EVALUATION")
    print(f"{'='*70}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading dataset from {dataset_path.name} ...")
    vecs, labels, label_strs, genres = load_eval_data(dataset_path, tier=tier)
    n_pts     = len(vecs)
    n_classes = len(label_strs)
    vec_dim   = vecs.shape[1]

    unique, counts = np.unique(labels, return_counts=True)
    multi_class = (counts >= 2).sum()
    avg_per_key = counts[counts >= 2].mean() if multi_class else 0

    print(f"  Points:           {n_pts:,}")
    print(f"  Identity keys:    {n_classes:,}  (using {tier})")
    print(f"  Keys with ≥2 pts: {multi_class:,}  (avg {avg_per_key:.1f}/key)")
    print(f"  Vector dim:       {vec_dim}")
    print()

    k_values = [1, 5, 10, 20]

    # ── Baseline: raw CLIP vectors ────────────────────────────────────────────
    print("Computing baseline Recall@K (raw CLIP embeddings) ...")
    t0       = time.perf_counter()
    baseline = recall_at_k(vecs, labels, k_values=k_values, n_queries=n_queries)
    base_t   = time.perf_counter() - t0
    base_nn  = hard_negative_rejection_rate(vecs, labels, n_queries=min(500, n_queries))

    print(f"  R@1={baseline[1]:.3f}  R@5={baseline[5]:.3f}  "
          f"R@10={baseline[10]:.3f}  R@20={baseline[20]:.3f}  "
          f"NN-match={base_nn:.3f}  ({base_t:.1f}s)")

    # ── Genre breakdown (baseline) ────────────────────────────────────────────
    base_by_genre = recall_by_genre(vecs, labels, genres)

    # ── Metric head: projected vectors ────────────────────────────────────────
    head_results = {}
    head_by_genre = {}

    if checkpoint_path and checkpoint_path.exists():
        print(f"\nLoading metric head from {checkpoint_path.name} ...")
        head = load_metric_head(checkpoint_path)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        print(f"  Trained epoch: {ckpt.get('epoch', '?')}")
        print(f"  Checkpoint R@10 (training eval): {ckpt.get('recall10', '?')}")
        print(f"  Config: {ckpt.get('config', {})}")

        print(f"\nProjecting {n_pts:,} vectors through metric head ...")
        t0       = time.perf_counter()
        proj_vecs = apply_head_batched(head, vecs, device=device)
        proj_t   = time.perf_counter() - t0
        print(f"  Done — output dim={proj_vecs.shape[1]}  ({proj_t:.1f}s)")

        print("\nComputing metric head Recall@K ...")
        t0         = time.perf_counter()
        head_recall = recall_at_k(proj_vecs, labels, k_values=k_values, n_queries=n_queries)
        head_t     = time.perf_counter() - t0
        head_nn    = hard_negative_rejection_rate(proj_vecs, labels, n_queries=min(500, n_queries))

        print(f"  R@1={head_recall[1]:.3f}  R@5={head_recall[5]:.3f}  "
              f"R@10={head_recall[10]:.3f}  R@20={head_recall[20]:.3f}  "
              f"NN-match={head_nn:.3f}  ({head_t:.1f}s)")

        head_results  = head_recall
        head_by_genre = recall_by_genre(proj_vecs, labels, genres)
        head_nn_rate  = head_nn
    else:
        if checkpoint_path:
            print(f"\nNo checkpoint found at {checkpoint_path} — showing baseline only")
        head_results  = {}
        head_by_genre = {}
        head_nn_rate  = None

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'COMPARISON':^70}")
    print(f"{'─'*70}")
    print(f"{'Metric':<20} {'Baseline CLIP':>15} {'Metric Head':>14} {'Delta':>10}")
    print(f"{'─'*70}")

    for k in k_values:
        b    = baseline[k]
        h    = head_results.get(k, None)
        d    = f"{h - b:+.3f}" if h is not None else "—"
        hstr = f"{h:.3f}" if h is not None else "—"
        marker = " ★" if h is not None and (h - b) >= 0.05 else ""
        print(f"  Recall@{k:<13} {b:>15.3f} {hstr:>14} {d:>10}{marker}")

    print(f"  {'NN match rate':<18} {base_nn:>15.3f} {head_nn_rate if head_nn_rate is not None else '—':>14}")
    print(f"{'─'*70}")

    # ── Genre breakdown ───────────────────────────────────────────────────────
    if base_by_genre:
        print(f"\n{'Recall@10 by genre':}")
        print(f"  {'Genre':<30} {'Baseline':>10} {'Head':>10} {'Δ':>8}")
        print(f"  {'─'*58}")
        all_genres = sorted(set(base_by_genre) | set(head_by_genre))
        for g in all_genres:
            b = base_by_genre.get(g, 0)
            h = head_by_genre.get(g, None)
            d = f"{h - b:+.3f}" if h is not None else "—"
            hstr = f"{h:.3f}" if h is not None else "—"
            print(f"  {g:<30} {b:>10.3f} {hstr:>10} {d:>8}")

    print(f"\n{'='*70}\n")

    # ── Build report dict ─────────────────────────────────────────────────────
    report = {
        "dataset":    str(dataset_path),
        "tier":       tier,
        "n_points":   n_pts,
        "n_classes":  n_classes,
        "vec_dim":    vec_dim,
        "n_queries":  n_queries,
        "baseline": {
            **{f"recall@{k}": baseline[k] for k in k_values},
            "nn_match_rate": base_nn,
            "by_genre": base_by_genre,
        },
        "metric_head": {
            **{f"recall@{k}": head_results.get(k) for k in k_values},
            "nn_match_rate": head_nn_rate,
            "by_genre": head_by_genre,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        } if head_results else None,
        "delta": {
            f"recall@{k}": round(head_results[k] - baseline[k], 4)
            for k in k_values if k in head_results
        } if head_results else None,
    }

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate metric head vs baseline CLIP")
    ap.add_argument("--dataset",    default="data/metric_dataset.parquet")
    ap.add_argument("--checkpoint", default="models/metric_head_v1.pt",
                    help="Path to metric head checkpoint (omit to show baseline only)")
    ap.add_argument("--n-queries",  type=int, default=1000)
    ap.add_argument("--tier",       default="tier3",
                    choices=["tier3", "tier2", "tier1"],
                    help="Identity tier to use for evaluation labels")
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--no-plot",    action="store_true",
                    help="Skip Recall@K plot even if matplotlib is available")
    args = ap.parse_args()

    dataset_path    = _ROOT / args.dataset
    checkpoint_path = _ROOT / args.checkpoint if args.checkpoint else None

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run tools/extract_metric_dataset.py first (on EC2), then copy data/metric_dataset.parquet here.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = evaluate(dataset_path, checkpoint_path, args.n_queries, args.tier, args.device)

    # Save report
    report_path = RESULTS_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path}")

    # Summary text
    summary_path = RESULTS_DIR / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write("METRIC HEAD EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset:   {dataset_path.name}\n")
        f.write(f"Points:    {report['n_points']:,}\n")
        f.write(f"Classes:   {report['n_classes']:,}\n")
        f.write(f"Tier:      {report['tier']}\n\n")
        f.write("Baseline CLIP:\n")
        for k in [1, 5, 10, 20]:
            f.write(f"  Recall@{k:<3} = {report['baseline'][f'recall@{k}']:.4f}\n")
        if report.get("metric_head"):
            f.write("\nMetric Head:\n")
            for k in [1, 5, 10, 20]:
                v = report["metric_head"].get(f"recall@{k}")
                if v is not None:
                    d = report["delta"].get(f"recall@{k}", 0)
                    f.write(f"  Recall@{k:<3} = {v:.4f}  ({d:+.4f})\n")
    print(f"Summary saved → {summary_path}")

    # Optional plot
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            k_vals = [1, 5, 10, 20]
            base_r = [report["baseline"][f"recall@{k}"] for k in k_vals]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(k_vals, base_r, "o-", label="Baseline CLIP (768-dim)", color="steelblue")
            if report.get("metric_head"):
                head_r = [report["metric_head"].get(f"recall@{k}", 0) for k in k_vals]
                ax.plot(k_vals, head_r, "s--", label="Metric Head (128-dim)", color="darkorange")
            ax.set_xlabel("K")
            ax.set_ylabel("Recall@K")
            ax.set_title("Recall@K: Baseline CLIP vs Metric Head")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(k_vals)
            ax.set_ylim(0, 1)
            plot_path = RESULTS_DIR / "recall_curve.png"
            plt.tight_layout()
            plt.savefig(str(plot_path), dpi=150)
            print(f"Plot saved → {plot_path}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
