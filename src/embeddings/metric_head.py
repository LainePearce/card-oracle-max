"""
Metric projection head for card identity disambiguation.

Trained on top of frozen CLIP ViT-L/14 (768-dim) image vectors to produce
a 128-dim embedding space where the same card clusters tightly regardless of
grade, lighting, angle, or parallel variant.

Architecture:
    768-dim CLIP vector
        → Linear(768, 512) + GELU + LayerNorm
        → Linear(512, 256) + GELU + LayerNorm
        → Linear(256, 128) + L2-normalise
    = 128-dim card identity embedding (unit sphere)

Loss: Online Hard Triplet Loss (semi-hard negative mining)
    For each anchor in a batch, selects the hardest positive (furthest same-key)
    and the hardest semi-hard negative (closest different-key beyond the margin).
    This is more stable than random triplet sampling and handles class imbalance well.

Training data:
    data/metric_dataset.parquet — produced by tools/extract_metric_dataset.py
    Identity label: tier3 key  player|set|card_number  (base card identity)

Usage (local, CPU):
    from src.embeddings.metric_head import MetricHead, train_metric_head
    head = train_metric_head("data/metric_dataset.parquet", epochs=30)
    # or from CLI:
    python src/embeddings/metric_head.py --epochs 30 --output models/metric_head_v1.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

MODELS_DIR = _ROOT / "models"


# ── Architecture ──────────────────────────────────────────────────────────────────

class MetricHead(nn.Module):
    """
    Small MLP projection head: 768 → 512 → 256 → 128 (L2-normalised).

    Designed to be trained on top of a frozen CLIP ViT-L/14 backbone.
    Input vectors should be pre-normalised (as CLIP always outputs them).
    Output vectors are L2-normalised to the unit sphere — compatible with
    cosine similarity scoring in Qdrant.
    """

    INPUT_DIM  = 768
    OUTPUT_DIM = 128

    def __init__(self, input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — pre-normalised CLIP vectors
        Returns:
            (batch, output_dim) — L2-normalised card identity embeddings
        """
        projected = self.net(x)
        return F.normalize(projected, dim=-1)


# ── Dataset ───────────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Dataset that yields (vectors, label_indices) per batch.
    Online hard triplet mining happens inside the training loop — this dataset
    just provides the raw vectors and integer class labels.

    The label is the integer encoding of the tier3 identity key so that
    same-card vectors share a label and can be detected for mining.
    """

    def __init__(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        self.vectors = torch.tensor(vectors, dtype=torch.float32)
        self.labels  = torch.tensor(labels,  dtype=torch.long)
        assert len(self.vectors) == len(self.labels)

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int):
        return self.vectors[idx], self.labels[idx]


def load_dataset(parquet_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load parquet dataset. Returns:
        vectors    (N, 768) float32 ndarray
        labels     (N,)     int ndarray  (integer-encoded tier3 keys)
        label_strs list of unique tier3 strings (index → string mapping)
    """
    import pyarrow.parquet as pq

    # Stream row-groups into a preallocated array. The previous implementation
    # used .to_pylist() + np.array() on the whole vector column, which
    # materialises N×768 floats as Python objects (~24B each + per-row list
    # overhead): ~35-40GB transient for a 1.76M-point dataset — OOM-froze two
    # g5 workers before training ever started. Arrow→NumPy via .values avoids
    # Python objects entirely; per-row-group conversion bounds the transient
    # to one row group. Peak RSS ≈ the final (N,768) float32 array (~5.4GB).
    pf      = pq.ParquetFile(str(parquet_path))
    n_total = pf.metadata.num_rows

    vectors    = np.empty((n_total, 768), dtype=np.float32)
    tier3_strs: list[str] = []
    row = 0
    for rg in range(pf.num_row_groups):
        tbl  = pf.read_row_group(rg, columns=["image_vec", "tier3"])
        col  = tbl.column("image_vec").combine_chunks()
        # (FixedSize)List<float> → flat numpy → reshape: no Python objects
        flat = col.values.to_numpy(zero_copy_only=False)
        n_rg = len(col)
        vectors[row:row + n_rg] = flat.reshape(n_rg, -1).astype(np.float32, copy=False)
        tier3_strs.extend(tbl.column("tier3").to_pylist())
        row += n_rg
    assert row == n_total, f"row-group rows {row} != metadata rows {n_total}"

    # Encode tier3 strings → integers
    unique_keys = sorted(set(tier3_strs))
    key_to_int  = {k: i for i, k in enumerate(unique_keys)}
    labels      = np.array([key_to_int[k] for k in tier3_strs], dtype=np.int64)

    return vectors, labels, unique_keys


# ── Loss ──────────────────────────────────────────────────────────────────────────

def online_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
    margin:     float = 0.3,
) -> torch.Tensor:
    """
    Online Hard Triplet Loss with semi-hard negative mining.

    For each anchor:
      Positive:  hardest (furthest) embedding with the same label
      Negative:  hardest semi-hard negative — closest embedding with a different
                 label that is still further than the positive distance + margin
                 Falls back to the hardest negative if no semi-hard exists.

    Args:
        embeddings: (N, D) L2-normalised embeddings
        labels:     (N,)   integer class labels
        margin:     triplet loss margin (default 0.3)

    Returns:
        scalar loss (mean over valid triplets)
    """
    # Pairwise cosine distances: D[i,j] = 1 - cos(e_i, e_j)
    # Since embeddings are L2-normalised, cos(a,b) = a·b  →  D = 1 - a@b^T
    dots = torch.mm(embeddings, embeddings.t()).clamp(-1, 1)
    dists = 1.0 - dots  # shape (N, N)

    # Masks
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (N,N) same class
    eye       = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask  = labels_eq & ~eye         # same class, not self
    neg_mask  = ~labels_eq               # different class

    losses = []
    for i in range(len(embeddings)):
        pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
        neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        # Hardest positive
        d_pos = dists[i][pos_indices].max()

        # Semi-hard negatives: distance > d_pos AND distance < d_pos + margin
        d_negs    = dists[i][neg_indices]
        semi_hard = d_negs[(d_negs > d_pos) & (d_negs < d_pos + margin)]
        if len(semi_hard) > 0:
            d_neg = semi_hard.min()
        else:
            d_neg = d_negs.min()    # hard negative fallback

        loss = F.relu(d_pos - d_neg + margin)
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, requires_grad=True)

    return torch.stack(losses).mean()


# ── Recall@K evaluation ──────────────────────────────────────────────────────────

def recall_at_k(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    k_values:   list[int] = (1, 5, 10, 20),
    n_queries:  int = 1000,
    seed:       int = 42,
) -> dict[int, float]:
    """
    Compute Recall@K: fraction of queries where at least one true positive
    appears in the top-K retrieved results (excluding self).

    Uses brute-force cosine similarity — fine for <200k vectors on CPU.
    """
    rng   = np.random.RandomState(seed)
    n     = len(embeddings)
    n_q   = min(n_queries, n)

    # Filter: only query from classes with ≥2 members (need at least 1 positive)
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes  = unique[counts >= 2]
    valid_mask     = np.isin(labels, valid_classes)
    valid_idx      = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return {k: 0.0 for k in k_values}

    query_idx = rng.choice(valid_idx, size=min(n_q, len(valid_idx)), replace=False)

    recalls = {k: 0 for k in k_values}
    max_k   = max(k_values)

    # Batch cosine similarity computation
    batch = 512
    hits  = {k: 0 for k in k_values}
    total = 0

    for start in range(0, len(query_idx), batch):
        q_idx  = query_idx[start:start + batch]
        q_embs = embeddings[q_idx]                        # (B, D)
        sims   = q_embs @ embeddings.T                    # (B, N)

        # Exclude self: set diagonal (query against itself) to -inf
        for bi, gi in enumerate(q_idx):
            sims[bi, gi] = -np.inf

        # Top-K retrieved (descending similarity)
        top_k_idx = np.argpartition(sims, -max_k, axis=1)[:, -max_k:]
        top_k_idx = top_k_idx[np.arange(len(q_idx))[:, None],
                               np.argsort(sims[np.arange(len(q_idx))[:, None], top_k_idx], axis=1)[:, ::-1]]

        q_labels = labels[q_idx]   # (B,)
        for bi in range(len(q_idx)):
            true_label = q_labels[bi]
            retrieved  = labels[top_k_idx[bi]]
            for k in k_values:
                if true_label in retrieved[:k]:
                    hits[k] += 1
            total += 1

    return {k: hits[k] / total for k in k_values}


# ── Training loop ─────────────────────────────────────────────────────────────────

def train_metric_head(
    dataset_path:    str | Path,
    epochs:          int    = 30,
    batch_size:      int    = 512,
    lr:              float  = 3e-4,
    margin:          float  = 0.3,
    dropout:         float  = 0.1,
    val_fraction:    float  = 0.15,
    eval_every:      int    = 5,
    output_path:     Optional[str | Path] = None,
    seed:            int    = 42,
    epoch_sample:    Optional[int] = None,
    init_checkpoint: Optional[str | Path] = None,
) -> MetricHead:
    """
    Train the MetricHead on the extracted dataset.

    Args:
        dataset_path:  Path to metric_dataset.parquet
        epochs:        Training epochs
        batch_size:    Batch size (online mining within each batch)
        lr:            AdamW learning rate
        margin:        Triplet loss margin
        dropout:       Dropout rate in the head
        val_fraction:  Fraction of identity keys held out for evaluation
        eval_every:    Evaluate Recall@K every N epochs
        output_path:   Where to save the best checkpoint (default: models/metric_head_v1.pt)
        seed:          Random seed for reproducibility
        epoch_sample:    If set, randomly sample this many training points per epoch.
                         Useful for large datasets (>200k) to keep each epoch fast.
                         e.g. epoch_sample=150_000 trains on a fresh 150k subset each epoch.
                         None = use all training points every epoch.
        init_checkpoint: Optional path to a saved checkpoint (.pt). If provided, the
                         model is warm-started from that checkpoint's weights before
                         training begins. Useful for fine-tuning the best checkpoint
                         from a previous run with a lower learning rate.

    Returns:
        Best MetricHead (by Recall@10 on val set)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_path = Path(dataset_path)
    if output_path is None:
        output_path = MODELS_DIR / "metric_head_v1.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    logger.info("Loading dataset from {}", dataset_path)
    vectors, labels, label_strs = load_dataset(dataset_path)
    n_total   = len(vectors)
    n_classes = len(label_strs)
    vec_dim   = vectors.shape[1]
    mem_gb    = vectors.nbytes / 1e9
    logger.info("{:,} points | {:,} identity classes | {}-dim vectors | {:.2f} GB RAM",
                n_total, n_classes, vec_dim, mem_gb)

    # ── Train/val split on identity keys (not on points) ─────────────────────
    unique_classes = np.unique(labels)
    rng            = np.random.RandomState(seed)
    rng.shuffle(unique_classes)
    n_val          = max(1, int(len(unique_classes) * val_fraction))
    val_classes    = set(unique_classes[:n_val].tolist())
    train_classes  = set(unique_classes[n_val:].tolist())

    train_mask = np.array([l in train_classes for l in labels])
    val_mask   = np.array([l in val_classes   for l in labels])

    train_vecs   = vectors[train_mask]
    train_labels = labels[train_mask]
    val_vecs     = vectors[val_mask]
    val_labels   = labels[val_mask]

    logger.info("Train: {:,} points ({:,} classes) | Val: {:,} points ({:,} classes)",
                len(train_vecs), len(train_classes), len(val_vecs), len(val_classes))

    # ── Baseline recall (before training) ────────────────────────────────────
    # For large datasets, cap baseline evaluation at 50k points for speed.
    # This gives a representative estimate without taking many minutes.
    BASELINE_CAP = 50_000
    logger.info("Computing baseline Recall@K (raw CLIP vectors) ...")
    if n_total > BASELINE_CAP:
        bl_idx  = rng.choice(n_total, BASELINE_CAP, replace=False)
        bl_vecs = np.vstack([train_vecs, val_vecs]) if len(val_vecs) else train_vecs
        bl_labs = np.concatenate([train_labels, val_labels]) if len(val_labels) else train_labels
        bl_vecs = bl_vecs[bl_idx]
        bl_labs = bl_labs[bl_idx]
        logger.info("  (using {:,}-point sample for baseline — full dataset is {:,})", BASELINE_CAP, n_total)
    else:
        bl_vecs = np.vstack([train_vecs, val_vecs]) if len(val_vecs) else train_vecs
        bl_labs = np.concatenate([train_labels, val_labels]) if len(val_labels) else train_labels

    baseline = recall_at_k(bl_vecs, bl_labs, k_values=[1, 5, 10, 20])
    logger.info("Baseline  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}  R@20={:.3f}",
                baseline[1], baseline[5], baseline[10], baseline[20])

    # ── Model & optimiser ─────────────────────────────────────────────────────
    # cuda (EC2 g5 workers) → mps (local Mac dev) → cpu. The original line
    # checked mps-else-cpu only, which silently trained on CPU on the GPU
    # workers — ~10-20x slower with the A10G sitting idle.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Training on {}", device)

    model = MetricHead(input_dim=vec_dim, output_dim=128, dropout=dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr/20)

    best_recall10 = baseline[10]
    best_state    = None
    history       = []

    # Warm-start: load weights AFTER best_recall10 is set so we correctly
    # require the fine-tuned model to beat the checkpoint, not just the baseline.
    if init_checkpoint is not None:
        ckpt = torch.load(str(init_checkpoint), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        init_r10 = ckpt.get("recall10", baseline[10])
        logger.info("Warm-started from {} (R@10={:.3f})", Path(init_checkpoint).name, init_r10)
        best_recall10 = init_r10   # must beat the checkpoint to save a new one

    if epoch_sample:
        effective_n = min(epoch_sample, len(train_vecs))
        logger.info(
            "Training {:,} epochs | epoch_sample={:,} of {:,} train | batch={} | lr={} | margin={}",
            epochs, effective_n, len(train_vecs), batch_size, lr, margin,
        )
    else:
        logger.info("Training {:,} epochs | batch={} | lr={} | margin={}",
                    epochs, batch_size, lr, margin)
    logger.info("─" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        t_ep         = time.perf_counter()

        # ── Per-epoch sampling for large datasets ─────────────────────────────
        if epoch_sample and epoch_sample < len(train_vecs):
            ep_idx    = rng.choice(len(train_vecs), epoch_sample, replace=False)
            ep_vecs   = train_vecs[ep_idx]
            ep_labels = train_labels[ep_idx]
            ep_ds     = TripletDataset(ep_vecs, ep_labels)
            loader    = DataLoader(ep_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=0, pin_memory=False)
        else:
            ep_ds  = TripletDataset(train_vecs, train_labels)
            loader = DataLoader(ep_ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)

        for vecs_batch, labels_batch in loader:
            vecs_batch   = vecs_batch.to(device)
            labels_batch = labels_batch.to(device)

            projected = model(vecs_batch)
            loss      = online_hard_triplet_loss(projected, labels_batch, margin=margin)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        log_row = {"epoch": epoch, "loss": mean_loss}

        # ── Eval ──────────────────────────────────────────────────────────────
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()

            # For large val sets, cap at 30k to keep eval tractable.
            # Recall@K needs enough coverage to be meaningful — 30k is sufficient.
            VAL_EVAL_CAP   = 30_000
            TRAIN_EVAL_CAP = 15_000

            with torch.no_grad():
                # Apply head to val vectors (capped subset)
                if len(val_vecs) > VAL_EVAL_CAP:
                    val_eval_idx = rng.choice(len(val_vecs), VAL_EVAL_CAP, replace=False)
                    val_eval_vecs   = val_vecs[val_eval_idx]
                    val_eval_labels = val_labels[val_eval_idx]
                else:
                    val_eval_vecs   = val_vecs
                    val_eval_labels = val_labels

                val_t   = torch.tensor(val_eval_vecs, dtype=torch.float32, device=device)
                val_emb = []
                for i in range(0, len(val_t), 1024):
                    val_emb.append(model(val_t[i:i+1024]).cpu().numpy())
                val_emb = np.vstack(val_emb) if val_emb else np.zeros((0, 128))

                # Apply head to train vectors (capped subset for diversity)
                n_train_sample = min(TRAIN_EVAL_CAP, len(train_vecs))
                idx_sample     = rng.choice(len(train_vecs), n_train_sample, replace=False)
                tr_t    = torch.tensor(train_vecs[idx_sample], dtype=torch.float32, device=device)
                tr_emb  = []
                for i in range(0, len(tr_t), 1024):
                    tr_emb.append(model(tr_t[i:i+1024]).cpu().numpy())
                tr_emb  = np.vstack(tr_emb) if tr_emb else np.zeros((0, 128))

                # Combined embeddings for Recall@K
                all_emb    = np.vstack([tr_emb, val_emb]) if len(val_emb) else tr_emb
                all_labels = np.concatenate([train_labels[idx_sample], val_eval_labels]) if len(val_emb) else train_labels[idx_sample]

            r = recall_at_k(all_emb, all_labels, k_values=[1, 5, 10, 20],
                            n_queries=2000)
            log_row.update(r)

            flag = ""
            if r[10] > best_recall10:
                best_recall10 = r[10]
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save({
                    "state_dict":  best_state,
                    "input_dim":   vec_dim,
                    "output_dim":  128,
                    "epoch":       epoch,
                    "recall10":    best_recall10,
                    "baseline":    baseline,
                    "config":      {"lr": lr, "margin": margin, "dropout": dropout, "batch_size": batch_size},
                }, str(output_path))
                flag = "  ★ new best"

            ep_time = time.perf_counter() - t_ep
            logger.info(
                "Epoch {:>3}/{} | loss={:.4f} | R@1={:.3f} R@5={:.3f} R@10={:.3f} R@20={:.3f} | {:.1f}s{}",
                epoch, epochs, mean_loss, r[1], r[5], r[10], r[20], ep_time, flag,
            )
        else:
            ep_time = time.perf_counter() - t_ep
            logger.info("Epoch {:>3}/{} | loss={:.4f} | {:.1f}s", epoch, epochs, mean_loss, ep_time)

        history.append(log_row)

    logger.info("─" * 70)
    logger.info("Training complete")
    logger.info("Baseline   R@10 = {:.3f}", baseline[10])
    logger.info("Best model R@10 = {:.3f}  (Δ = {:+.3f})", best_recall10, best_recall10 - baseline[10])
    if best_state:
        logger.info("Best checkpoint saved → {}", output_path)

    # Load best weights into model before returning
    if best_state:
        model.load_state_dict(best_state)

    # Write training history JSON
    history_path = output_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump({"baseline": baseline, "history": history,
                   "best_recall10": best_recall10}, f, indent=2)
    logger.info("History written → {}", history_path.name)

    return model


# ── Inference helper (for using the trained head at query time) ────────────────

def load_metric_head(path: str | Path) -> MetricHead:
    """Load a saved MetricHead checkpoint for inference."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    head = MetricHead(
        input_dim  = checkpoint.get("input_dim",  768),
        output_dim = checkpoint.get("output_dim", 128),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head


def project_vector(
    head:       MetricHead,
    clip_vector: np.ndarray,
    device:     str = "cpu",
) -> np.ndarray:
    """Apply the metric head to a single CLIP vector. Returns (128,) float32."""
    t = torch.tensor(clip_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = head(t.to(device)).squeeze(0)
    return out.cpu().numpy()


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train metric head on CLIP image vectors")
    ap.add_argument("--dataset",      default="data/metric_dataset.parquet",
                    help="Path to metric_dataset.parquet")
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch-size",   type=int,   default=512)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--margin",       type=float, default=0.3)
    ap.add_argument("--dropout",      type=float, default=0.1)
    ap.add_argument("--eval-every",   type=int,   default=5)
    ap.add_argument("--epoch-sample", type=int,   default=None,
                    help="Per-epoch training subset size for large datasets "
                         "(e.g. 150000). None = use full training set each epoch.")
    ap.add_argument("--output",       default="models/metric_head_v1.pt",
                    help="Where to save the best checkpoint")
    ap.add_argument("--init-checkpoint", default=None,
                    help="Warm-start from this checkpoint before training "
                         "(fine-tuning a previous best model)")
    args = ap.parse_args()

    try:
        from loguru import logger as _log
        _log.remove()
        _log.add(sys.stderr, level="INFO", colorize=True,
                 format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    except ImportError:
        pass

    train_metric_head(
        dataset_path    = args.dataset,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        margin          = args.margin,
        dropout         = args.dropout,
        eval_every      = args.eval_every,
        output_path     = args.output,
        epoch_sample    = args.epoch_sample,
        init_checkpoint = args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
