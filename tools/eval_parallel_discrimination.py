#!/usr/bin/env python3
"""
Score image encoders on parallel discrimination: CLIP (production, ViT-L/14,
768-d) vs DINOv2 / DINOv3.

Reads the eval set from sample_parallel_eval_set.py. Every card belongs to a
DESIGN GROUP (same player/year/set/card_number); within a group, cards differ
only by parallel. We measure whether each encoder's embeddings separate the
parallels.

Metrics, per encoder:
  - Parallel R@1 (within-group, leave-one-out): for each card, rank the OTHER
    cards in its design group by cosine sim; is the top-1 the SAME parallel?
    Reported against the CHANCE baseline (what you'd get by random ordering,
    given the group's parallel mix). This is the headline number.
  - Parallel mAP (within-group): ranking quality for same-parallel retrieval.
  - Δ cosine = mean cos(same-parallel pair) − mean cos(different-parallel pair),
    over within-group pairs. Bigger Δ = the encoder pushes parallels apart.
  - Design R@1 (global sanity): nearest neighbour across ALL cards is the same
    design? Both encoders should score high; confirms the eval set is sane.

Fairness: every encoder sees the SAME local images with the SAME pad-to-square
(grey-114) that production CLIP uses; only the backbone + resize differ.

Encoders:
  clip     open_clip ViT-L/14 openai (768-d) — production path, via ImageEncoder
  dinov2   facebook/dinov2-large (1024-d) via transformers AutoModel
  dinov3   facebook/dinov3-vitl16-pretrain-lvd1689m (gated; may need HF login
           and a recent transformers) — skipped gracefully if it won't load

Usage:
    python tools/eval_parallel_discrimination.py --eval-dir data/parallel_eval \
        --models clip dinov2
    python tools/eval_parallel_discrimination.py --models clip dinov2 dinov3 \
        --dino-image-size 518          # test the resolution lever on DINO
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger

from src.embeddings.image_encoder import ImageEncoder

DINO_MODEL_IDS = {
    "dinov2": "facebook/dinov2-large",
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_images(eval_dir: Path, manifest: list[dict]):
    from PIL import Image
    imgs = []
    for c in manifest:
        p = eval_dir / c["image_file"]
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def embed_clip(pils: list, batch: int) -> np.ndarray:
    enc = ImageEncoder()   # ViT-L/14 openai, 768-d, pad-to-square + L2-norm
    out = []
    for i in range(0, len(pils), batch):
        out.append(enc.encode_batch_pil(pils[i:i + batch]))
    return np.vstack(out).astype(np.float32)


def embed_hf_vision(model_id: str, pils: list, batch: int,
                    image_size: int | None, device: str,
                    fp16: bool = False) -> np.ndarray:
    """Generic DINOv2/v3 (or any HF vision backbone) encoder.

    Applies the SAME grey-114 pad-to-square as production CLIP, then the model's
    own processor, then L2-normalises the CLS/pooler embedding.

    Runs in fp32 by default. DINOv3 overflows to NaN in fp16 (its norm layers
    are fp16-unstable); the eval set is tiny so fp32 costs nothing meaningful.
    Pass fp16=True only for a backbone you've confirmed is stable in half.
    """
    import torch
    from transformers import AutoModel, AutoImageProcessor

    dtype = torch.float16 if fp16 else torch.float32
    proc  = AutoImageProcessor.from_pretrained(model_id)
    if image_size is not None and hasattr(proc, "size"):
        proc.size = {"shortest_edge": image_size}
        if hasattr(proc, "crop_size"):
            proc.crop_size = {"height": image_size, "width": image_size}
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).eval().to(device)

    pad = ImageEncoder._pad_to_square
    out = []
    for i in range(0, len(pils), batch):
        chunk  = [pad(p) for p in pils[i:i + batch]]
        inputs = proc(images=chunk, return_tensors="pt").to(device)
        if fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}
        with torch.no_grad():
            res = model(**inputs)
        emb = getattr(res, "pooler_output", None)
        if emb is None:
            emb = res.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        out.append(emb.cpu().numpy())
    embs = np.vstack(out).astype(np.float32)
    if not np.isfinite(embs).all():
        logger.error("{} produced non-finite embeddings (NaN/Inf) — results "
                     "for this model are invalid", model_id)
    return embs


def average_precision(relevant: np.ndarray, order: np.ndarray) -> float:
    """AP for a binary relevance vector reordered by `order` (best-first)."""
    rel = relevant[order]
    if rel.sum() == 0:
        return 0.0
    hits = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    precision_at_hits = hits[rel == 1] / ranks[rel == 1]
    return float(precision_at_hits.mean())


def score_encoder(emb: np.ndarray, manifest: list[dict]) -> dict:
    design   = np.array([c["design_key"] for c in manifest])
    parallel = np.array([c["parallel"]   for c in manifest])

    groups: dict[str, list[int]] = defaultdict(list)
    for i, dk in enumerate(design):
        groups[dk].append(i)

    par1_hits = par1_total = 0
    chance_sum = 0.0
    ap_list: list[float] = []
    same_sims: list[float] = []
    diff_sims: list[float] = []

    for dk, idx in groups.items():
        idx = np.array(idx)
        if len(idx) < 2:
            continue
        sub = emb[idx]
        sims = sub @ sub.T
        plabels = parallel[idx]

        for r in range(len(idx)):
            same_mask = plabels == plabels[r]
            same_mask[r] = False
            if same_mask.sum() == 0:
                continue   # no same-parallel positive for this query
            cand_mask = np.ones(len(idx), dtype=bool)
            cand_mask[r] = False

            s = sims[r].copy()
            s[r] = -np.inf
            top = int(np.argmax(s))
            par1_hits  += int(plabels[top] == plabels[r])
            par1_total += 1
            chance_sum += same_mask.sum() / cand_mask.sum()

            order = np.argsort(-sims[r][cand_mask])
            ap_list.append(average_precision(same_mask[cand_mask].astype(int), order))

        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                (same_sims if plabels[a] == plabels[b] else diff_sims).append(float(sims[a, b]))

    # Global design R@1 sanity check
    sims_all = emb @ emb.T
    np.fill_diagonal(sims_all, -np.inf)
    nn = np.argmax(sims_all, axis=1)
    design_r1 = float(np.mean(design[nn] == design))

    return {
        "dim":             int(emb.shape[1]),
        "parallel_r1":     100.0 * par1_hits / par1_total if par1_total else 0.0,
        "parallel_chance": 100.0 * chance_sum / par1_total if par1_total else 0.0,
        "parallel_map":    100.0 * float(np.mean(ap_list)) if ap_list else 0.0,
        "delta_cos":       (float(np.mean(same_sims)) - float(np.mean(diff_sims)))
                           if same_sims and diff_sims else 0.0,
        "same_cos":        float(np.mean(same_sims)) if same_sims else 0.0,
        "diff_cos":        float(np.mean(diff_sims)) if diff_sims else 0.0,
        "design_r1":       100.0 * design_r1,
        "queries":         par1_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default="data/parallel_eval")
    ap.add_argument("--models", nargs="+", default=["clip", "dinov2"],
                    choices=["clip", "dinov2", "dinov3"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dino-image-size", type=int, default=None,
                    help="Override DINO resize (e.g. 518) to test the resolution lever.")
    ap.add_argument("--fp16", action="store_true",
                    help="Run DINO backbones in fp16 (faster; DINOv3 is NaN-unstable in fp16).")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    eval_dir = ROOT / args.eval_dir
    manifest = [json.loads(l) for l in open(eval_dir / "manifest.jsonl")]
    n_groups = len({c["design_key"] for c in manifest})
    logger.info("Loaded {} cards / {} design groups", len(manifest), n_groups)

    pils   = load_images(eval_dir, manifest)
    device = _pick_device()
    logger.info("Device: {}", device)

    results: dict[str, dict] = {}
    for m in args.models:
        logger.info("Embedding with {} …", m)
        try:
            if m == "clip":
                emb = embed_clip(pils, args.batch)
            else:
                emb = embed_hf_vision(DINO_MODEL_IDS[m], pils, args.batch,
                                      args.dino_image_size, device, fp16=args.fp16)
        except Exception as e:
            logger.error("{} failed to load/run ({}: {}) — skipping",
                         m, type(e).__name__, e)
            continue
        results[m] = score_encoder(emb, manifest)
        logger.info("  {} done (dim={})", m, results[m]["dim"])

    print(f"\nParallel-discrimination eval — {len(manifest)} cards, {n_groups} design groups")
    print(f"(queries = cards that have a same-parallel sibling in their group)\n")
    print(f"{'model':9} {'dim':>5} {'par_R@1':>8} {'chance':>7} {'lift':>6} "
          f"{'par_mAP':>8} {'Δcos':>7} {'same':>6} {'diff':>6} {'design_R@1':>10}")
    print("-" * 84)
    for m, s in results.items():
        lift = s["parallel_r1"] - s["parallel_chance"]
        print(f"{m:9} {s['dim']:>5} {s['parallel_r1']:>7.1f}% {s['parallel_chance']:>6.1f}% "
              f"{lift:>+5.1f} {s['parallel_map']:>7.1f}% {s['delta_cos']:>7.3f} "
              f"{s['same_cos']:>6.3f} {s['diff_cos']:>6.3f} {s['design_r1']:>9.1f}%")
    print()
    print("Reading it: par_R@1 vs chance is the parallel-discrimination signal —")
    print("a parallel-BLIND encoder lands near chance (lift ~0); higher lift and")
    print("higher Δcos mean the encoder actually separates parallels. design_R@1")
    print("should be high for all (sanity that the images/embeddings are good).")

    if args.json:
        out = ROOT / args.json
        out.write_text(json.dumps(results, indent=2))
        logger.info("Wrote {}", out)


if __name__ == "__main__":
    main()
