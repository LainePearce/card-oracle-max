#!/usr/bin/env python3
"""
At-scale retrieval bake-off: CLIP vs DINOv2 vs DINOv3.

Reads the corpus+query set from sample_retrieval_eval_set.py. For each encoder
it embeds the whole corpus and the held-out queries, runs EXACT cosine nearest
-neighbour search (corpus on GPU), and reports Recall@K at two granularities:

  design   = right card ignoring parallel   (player/year/set/card_number)
  identity = right card INCLUDING parallel   (… | parallel)

The identity−design gap is parallel confusion at scale: how often the encoder
retrieves the right design but the wrong parallel.

Exact search (not ANN) is deliberate — it measures the ENCODER's intrinsic
retrieval quality, isolated from Qdrant HNSW/quantisation recall loss (which is
roughly encoder-independent).

Embedding is streamed from disk in batches so a 30k+ corpus never has to live
in RAM as decoded images. DINO backbones run fp32 by default (DINOv3 is NaN
-unstable in fp16); CLIP uses its production fp16 path.

Usage:
    python tools/eval_retrieval_at_scale.py --eval-dir data/retrieval_eval \
        --models clip dinov2 dinov3 --dino-image-size 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger

from src.embeddings.image_encoder import ImageEncoder
from tools.eval_parallel_discrimination import DINO_MODEL_IDS, _pick_device

K_VALUES = [1, 5, 10, 20, 50]


def build_encoder(name: str, image_size: int | None, fp16: bool, device: str):
    """Return a callable encode(pils)->np.ndarray (L2-normalised), model loaded once."""
    if name == "clip":
        enc = ImageEncoder()          # ViT-L/14 openai, 768-d, prod fp16 path
        return lambda pils: enc.encode_batch_pil(pils), 768

    import torch
    from transformers import AutoModel, AutoImageProcessor

    model_id = DINO_MODEL_IDS[name]
    dtype = torch.float16 if fp16 else torch.float32
    proc  = AutoImageProcessor.from_pretrained(model_id)
    if image_size is not None and hasattr(proc, "size"):
        proc.size = {"shortest_edge": image_size}
        if hasattr(proc, "crop_size"):
            proc.crop_size = {"height": image_size, "width": image_size}
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).eval().to(device)
    pad = ImageEncoder._pad_to_square

    def encode(pils):
        chunk  = [pad(p) for p in pils]
        inputs = proc(images=chunk, return_tensors="pt").to(device)
        if fp16:
            inputs = {k: (v.half() if v.dtype == torch.float32 else v)
                      for k, v in inputs.items()}
        with torch.no_grad():
            res = model(**inputs)
        emb = getattr(res, "pooler_output", None)
        if emb is None:
            emb = res.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy()

    dim = 1024
    return encode, dim


def embed_rows(rows: list[dict], eval_dir: Path, encode, batch: int) -> np.ndarray:
    from PIL import Image
    gray = Image.new("RGB", (224, 224), (114, 114, 114))
    out, bad = [], 0
    t0 = time.time()
    for i in range(0, len(rows), batch):
        pics = []
        for r in rows[i:i + batch]:
            try:
                pics.append(Image.open(eval_dir / r["image_file"]).convert("RGB"))
            except Exception:
                pics.append(gray)
                bad += 1
        out.append(encode(pics))
        if (i + batch) % (batch * 50) == 0:
            rate = (i + batch) / max(1e-6, time.time() - t0)
            logger.info("    {}/{} embedded ({:.0f} img/s)", i + batch, len(rows), rate)
    if bad:
        logger.warning("    {} images failed to decode (used grey placeholder)", bad)
    return np.vstack(out).astype(np.float32)


def evaluate(query_emb: np.ndarray, corpus_emb: np.ndarray,
             q_design, q_identity, c_design, c_identity, device: str) -> dict:
    import torch
    corpus_t = torch.from_numpy(corpus_emb).to(device)
    c_design  = np.asarray(c_design)
    c_identity = np.asarray(c_identity)
    corpus_design_set   = set(c_design.tolist())
    corpus_identity_set = set(c_identity.tolist())

    maxk = max(K_VALUES)
    design_hits   = {k: 0 for k in K_VALUES}
    identity_hits = {k: 0 for k in K_VALUES}
    design_rr = identity_rr = 0.0
    n_design = n_identity = 0

    B = 256
    for s in range(0, len(query_emb), B):
        qb = torch.from_numpy(query_emb[s:s + B]).to(device)
        sims = qb @ corpus_t.T
        topk = torch.topk(sims, k=maxk, dim=1).indices.cpu().numpy()
        for r in range(topk.shape[0]):
            gi = s + r
            ranked_design   = c_design[topk[r]]
            ranked_identity = c_identity[topk[r]]

            if q_design[gi] in corpus_design_set:
                n_design += 1
                match = np.where(ranked_design == q_design[gi])[0]
                if len(match):
                    first = match[0]
                    design_rr += 1.0 / (first + 1)
                    for k in K_VALUES:
                        if first < k:
                            design_hits[k] += 1
            if q_identity[gi] in corpus_identity_set:
                n_identity += 1
                match = np.where(ranked_identity == q_identity[gi])[0]
                if len(match):
                    first = match[0]
                    identity_rr += 1.0 / (first + 1)
                    for k in K_VALUES:
                        if first < k:
                            identity_hits[k] += 1

    return {
        "n_design_queries":   n_design,
        "n_identity_queries": n_identity,
        "design_recall":   {k: 100.0 * design_hits[k] / n_design if n_design else 0.0
                            for k in K_VALUES},
        "identity_recall": {k: 100.0 * identity_hits[k] / n_identity if n_identity else 0.0
                            for k in K_VALUES},
        "design_mrr":   100.0 * design_rr / n_design if n_design else 0.0,
        "identity_mrr": 100.0 * identity_rr / n_identity if n_identity else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default="data/retrieval_eval")
    ap.add_argument("--models", nargs="+", default=["clip", "dinov2", "dinov3"],
                    choices=["clip", "dinov2", "dinov3"])
    ap.add_argument("--dino-image-size", type=int, default=512)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    eval_dir = ROOT / args.eval_dir
    rows = [json.loads(l) for l in open(eval_dir / "manifest.jsonl")]
    corpus = [r for r in rows if r["role"] == "corpus"]
    queries = [r for r in rows if r["role"] == "query"]
    logger.info("Corpus {} / queries {}", len(corpus), len(queries))

    c_design   = [r["design_key"]   for r in corpus]
    c_identity = [r["identity_key"] for r in corpus]
    q_design   = [r["design_key"]   for r in queries]
    q_identity = [r["identity_key"] for r in queries]

    device = _pick_device()
    logger.info("Device: {}", device)

    results: dict[str, dict] = {}
    for m in args.models:
        logger.info("=== {} ===", m)
        try:
            encode, dim = build_encoder(m, args.dino_image_size, args.fp16, device)
            t0 = time.time()
            logger.info("  embedding corpus …")
            cemb = embed_rows(corpus, eval_dir, encode, args.batch)
            logger.info("  embedding queries …")
            qemb = embed_rows(queries, eval_dir, encode, args.batch)
            if not (np.isfinite(cemb).all() and np.isfinite(qemb).all()):
                logger.error("  {} produced non-finite embeddings — skipping", m)
                continue
            res = evaluate(qemb, cemb, q_design, q_identity,
                           c_design, c_identity, device)
            res["dim"] = dim
            res["embed_seconds"] = round(time.time() - t0, 1)
            results[m] = res
            logger.info("  done in {:.0f}s", res["embed_seconds"])
        except Exception as e:
            logger.error("  {} failed ({}: {}) — skipping", m, type(e).__name__, e)

    print(f"\nAt-scale retrieval — corpus={len(corpus)}, queries={len(queries)}")
    print("DESIGN = right card ignoring parallel | IDENTITY = incl. parallel\n")
    hdr = f"{'model':9} {'dim':>5} " + " ".join(f"R@{k:<3}" for k in K_VALUES) + "  MRR"
    for gran in ("design", "identity"):
        print(f"── {gran.upper()} recall ─────────────────────────────────────────")
        print(hdr)
        for m, s in results.items():
            rec = s[f"{gran}_recall"]
            cells = " ".join(f"{rec[k]:5.1f}" for k in K_VALUES)
            print(f"{m:9} {s['dim']:>5} {cells}  {s[f'{gran}_mrr']:5.1f}")
        print()
    if results:
        print("Parallel confusion (design R@1 − identity R@1):")
        for m, s in results.items():
            gap = s["design_recall"][1] - s["identity_recall"][1]
            print(f"  {m:9} {gap:+5.1f} pp")

    if args.json:
        out = ROOT / args.json
        out.write_text(json.dumps(results, indent=2))
        logger.info("Wrote {}", out)


if __name__ == "__main__":
    main()
