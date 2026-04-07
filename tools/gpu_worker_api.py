#!/usr/bin/env python3
"""
GPU worker vector encoding API.

Run this on the GPU EC2 worker to offload CLIP encoding from the local search UI.
The local search_ui_server.py calls this if GPU_WORKER_URL is set in its .env.

Usage (on the GPU worker EC2 instance):
    cd ~/card-oracle-max
    source .venv/bin/activate
    python tools/gpu_worker_api.py

    # Or run on a specific port / bind address:
    GPU_API_PORT=8081 GPU_API_HOST=0.0.0.0 python tools/gpu_worker_api.py

On your local machine, set:
    GPU_WORKER_URL=http://<ec2-public-ip>:8081

Security note:
    This server has no authentication. Run it behind a security group rule that
    only allows access from your local IP. Do NOT expose it to the internet.

Endpoints:
    GET  /health              → {"status":"ok","gpu":true,"model":"ViT-L/14","dim":768}
    POST /encode              → {"image_url": "..."} → {"vector":[...],"dim":768,"elapsed_ms":N}
    POST /encode_batch        → {"image_urls": [...], "max":N} → {"vectors":[...],"dim":768}
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from flask import Flask, jsonify, request
from loguru import logger

GPU_API_HOST  = os.environ.get("GPU_API_HOST", "0.0.0.0")
GPU_API_PORT  = int(os.environ.get("GPU_API_PORT", 8081))
DEVICE        = os.environ.get("EMBEDDING_DEVICE", "cuda")
MODEL_NAME    = os.environ.get("CLIP_MODEL", "ViT-L/14")
PRETRAINED    = "openai"
MAX_BATCH     = 50

app = Flask(__name__)

# ── Encoder — loaded at startup ────────────────────────────────────────────
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from src.embeddings.image_encoder import ImageEncoder
        logger.info("Loading CLIP {} on {}...", MODEL_NAME, DEVICE)
        _encoder = ImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
        # Trigger model load with a dummy call so first real request isn't slow
        _encoder._load()
        logger.info("CLIP model ready")
    return _encoder


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    import torch
    enc   = get_encoder()
    # Get embedding dim by checking the loaded model
    dim   = 768 if "L/14" in MODEL_NAME else 512
    return jsonify({
        "status":  "ok",
        "gpu":     torch.cuda.is_available(),
        "device":  DEVICE,
        "model":   MODEL_NAME,
        "pretrained": PRETRAINED,
        "dim":     dim,
    })


@app.route("/encode", methods=["POST"])
def encode_single():
    """
    Encode a single image URL.

    Request body:
        {"image_url": "https://..."}

    Response:
        {"vector": [...768 floats...], "dim": 768, "elapsed_ms": 42.3}
    """
    data      = request.get_json(force=True) or {}
    image_url = (data.get("image_url") or "").strip()

    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    t0  = time.perf_counter()
    enc = get_encoder()
    vec = enc.encode_url(image_url)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if vec is None:
        return jsonify({"error": f"Failed to encode image: {image_url}"}), 422

    return jsonify({
        "vector":     vec.tolist(),
        "dim":        len(vec),
        "elapsed_ms": round(elapsed_ms, 1),
        "model":      MODEL_NAME,
    })


@app.route("/encode_batch", methods=["POST"])
def encode_batch():
    """
    Encode a batch of image URLs.

    Request body:
        {"image_urls": ["https://...", ...], "max": 50}

    Response:
        {
          "vectors": [
            {"url": "...", "vector": [...], "ok": true},
            {"url": "...", "vector": null, "ok": false, "error": "..."},
            ...
          ],
          "dim": 768
        }
    """
    data       = request.get_json(force=True) or {}
    image_urls = data.get("image_urls") or []
    max_items  = min(int(data.get("max", MAX_BATCH)), MAX_BATCH)
    image_urls = image_urls[:max_items]

    if not image_urls:
        return jsonify({"error": "image_urls must be a non-empty list"}), 400

    enc = get_encoder()
    results = []
    for url in image_urls:
        vec = enc.encode_url(url)
        if vec is not None:
            results.append({"url": url, "vector": vec.tolist(), "ok": True})
        else:
            results.append({"url": url, "vector": None, "ok": False, "error": "encode failed"})

    dim = len(results[0]["vector"]) if results and results[0]["vector"] else 768
    return jsonify({"vectors": results, "dim": dim})


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("─" * 50)
    logger.info("Card Oracle GPU Worker API")
    logger.info("  Model:  {} ({})", MODEL_NAME, PRETRAINED)
    logger.info("  Device: {}", DEVICE)
    logger.info("  Listen: http://{}:{}", GPU_API_HOST, GPU_API_PORT)
    logger.info("─" * 50)
    logger.info("Loading model at startup to avoid first-request delay...")
    get_encoder()  # pre-load
    logger.info("Ready. Set GPU_WORKER_URL=http://<this-ip>:{} on your local machine.", GPU_API_PORT)
    app.run(host=GPU_API_HOST, port=GPU_API_PORT, debug=False, threaded=True)
