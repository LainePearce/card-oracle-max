#!/usr/bin/env python3
"""
GPU Worker Search Server.

Runs on the EC2 GPU instance. Exposes search endpoints that own the complete
vector search pipeline:

  1. Receive image (URL download OR pre-resized base64 bytes)
  2. CLIP ViT-L/14 encode on GPU (FP16) via dynamic batching daemon
  3. Qdrant ANN search (via internal VPC IP — low latency)
  4. OpenSearch mget to enrich results with full documents
  5. Return combined results to the calling client

Production flow (preferred):
  The client (search_ui_server.py) downloads the image at native CDN resolution
  and base64-encodes the raw bytes (no pre-resize).  The GPU worker pads to
  square and lets CLIP's own Resize(224, BICUBIC) do the downscale — matching
  the backfill pipeline exactly.  CDN download latency is removed from the GPU
  hot path; the base64 payload is ~30–80 KB over the internal ALB connection.

Legacy flow:
  POST /search with image_url — GPU worker downloads the image itself.

Endpoints:
    GET  /health        — liveness check
    POST /encode        — encode image URL → vector only (for diagnostics)
    POST /encode_b64    — encode base64 image → vector only (for diagnostics)
    POST /search        — full pipeline via URL download → enriched results
    POST /search_b64    — full pipeline via pre-resized base64 → enriched results

Usage (on GPU instance):
    # Production — Gunicorn with gthread worker (recommended):
    source /home/ec2-user/card-oracle-max/.venv/bin/activate
    cd /home/ec2-user/card-oracle-max
    nohup gunicorn \
        --workers 1 \
        --threads 8 \
        --worker-class gthread \
        --bind 0.0.0.0:8081 \
        --timeout 120 \
        "tools.gpu_worker_server:app" \
        > /tmp/gpu-worker.log 2>&1 &

    # Development — Flask dev server (single process, no batching advantage):
    nohup python tools/gpu_worker_server.py > /tmp/gpu-worker.log 2>&1 &

Batching:
    Concurrent /search requests are coalesced into GPU batches automatically.
    Tune via env vars BATCH_SIZE (default 8) and BATCH_TIMEOUT_MS (default 20).
    One request thread downloads the image; the batch daemon runs the GPU forward
    pass on all pending images together, then resolves each caller.

Environment variables (loaded from .env):
    QDRANT_HOST         Qdrant host              (default: localhost)
    QDRANT_PORT         Qdrant REST port         (default: 6333)
    QDRANT_API_KEY      Qdrant API key
    QDRANT_COLLECTION   collection name          (default: cards)

    OPENSEARCH_HOST     OpenSearch cluster host
    OPENSEARCH_USER     Basic auth username
    OPENSEARCH_PASSWORD Basic auth password
    OPENSEARCH_INDEX    index pattern to search  (default: *,-*.*,-*-live*)

    WORKER_PORT         port to listen on        (default: 8081)
    EMBEDDING_DEVICE    cuda | cpu               (default: cuda)
"""

from __future__ import annotations

import base64
import io
import os
import queue
import sys
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from loguru import logger

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────

QDRANT_HOST       = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cards")

OS_HOST    = os.environ.get("OPENSEARCH_HOST", "")
OS_USE_SSL = os.environ.get("OPENSEARCH_USE_SSL", "true").lower() == "true"
OS_INDEX   = os.environ.get("OPENSEARCH_INDEX", "*,-*.*,-*-live*")
OS_BASE    = f"{'https' if OS_USE_SSL else 'http'}://{OS_HOST}" if OS_HOST else ""

_user = os.environ.get("OPENSEARCH_USER", "")
_pass = os.environ.get("OPENSEARCH_PASSWORD", "")
_encoded = base64.b64encode(f"{_user}:{_pass}".encode()).decode() if _user else ""
OS_HEADERS = {
    "Content-Type": "application/json",
    **({"Authorization": f"Basic {_encoded}"} if _encoded else {}),
}

EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cuda")
WORKER_PORT      = int(os.environ.get("WORKER_PORT", 8081))

# Batch encoding config — tune these for throughput vs latency trade-off
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", 8))
BATCH_TIMEOUT_MS = float(os.environ.get("BATCH_TIMEOUT_MS", 20)) / 1000

# ── App ───────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# CLIP encoder — loaded once at startup, shared across requests
_encoder      = None
_encoder_lock = threading.Lock()


def get_encoder():
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                from src.embeddings.image_encoder import ImageEncoder
                logger.info("Loading CLIP ViT-L/14 on {}...", EMBEDDING_DEVICE)
                _encoder = ImageEncoder(
                    model_name="ViT-L/14",
                    pretrained="openai",
                    device=EMBEDDING_DEVICE,
                )
                # Warm up — avoids latency on first real request
                import numpy as np
                _encoder._load()
                logger.info("CLIP ViT-L/14 ready ({}-dim)", _encoder._embedding_dim())
    return _encoder


# Qdrant client — created once and reused
_qdrant      = None
_qdrant_lock = threading.Lock()


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        with _qdrant_lock:
            if _qdrant is None:
                from qdrant_client import QdrantClient
                kwargs = {
                    "url":         f"http://{QDRANT_HOST}:{QDRANT_PORT}",
                    "prefer_grpc": False,
                    "timeout":     30,
                }
                if QDRANT_API_KEY:
                    kwargs["api_key"] = QDRANT_API_KEY
                _qdrant = QdrantClient(**kwargs)
                logger.info("Qdrant client connected to {}:{}", QDRANT_HOST, QDRANT_PORT)
    return _qdrant


# ── Batch encoding daemon ─────────────────────────────────────────────────────
# Concurrent /search requests enqueue their PIL images here. The daemon
# drains the queue every BATCH_TIMEOUT_MS (or when BATCH_SIZE is reached)
# and runs one GPU forward pass for all pending images, then resolves each
# caller's result queue. This converts serial ~159ms single-image encodes
# into batched ~185ms encodes shared across up to BATCH_SIZE concurrent requests.

_batch_queue: queue.Queue = queue.Queue()


def _batch_daemon() -> None:
    """Background daemon: coalesces encode requests into GPU batches."""
    while True:
        items = []

        # Block until at least one item arrives
        try:
            items.append(_batch_queue.get(timeout=1.0))
        except queue.Empty:
            continue

        # Drain up to BATCH_SIZE within the timeout window
        deadline = time.perf_counter() + BATCH_TIMEOUT_MS
        while len(items) < BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                items.append(_batch_queue.get(timeout=remaining))
            except queue.Empty:
                break

        # Single GPU forward pass for all items in this batch
        try:
            encoder = get_encoder()
            pil_images = [item["img"] for item in items]
            vecs = encoder.encode_batch_pil(pil_images)   # shape (n, dim)
            for item, vec in zip(items, vecs):
                item["result_q"].put(("ok", vec))
            if len(items) > 1:
                logger.debug("Batch encoded {} images together", len(items))
        except Exception as e:
            logger.error("Batch encode error: {}", e)
            for item in items:
                item["result_q"].put(("error", str(e)))


def _encode_via_batch(pil_image) -> "np.ndarray":
    """Submit a PIL image to the batch daemon and block until encoded."""
    result_q: queue.Queue = queue.Queue()
    _batch_queue.put({"img": pil_image, "result_q": result_q})
    status, payload = result_q.get(timeout=30)
    if status == "error":
        raise RuntimeError(payload)
    return payload


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    encoder = get_encoder()
    qdrant  = get_qdrant()

    qdrant_ok = False
    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        qdrant_ok = True
        points = info.points_count
    except Exception as e:
        points = f"error: {e}"

    return jsonify({
        "status":          "ok",
        "device":          EMBEDDING_DEVICE,
        "embedding_dim":   encoder._embedding_dim(),
        "qdrant":          "ok" if qdrant_ok else "error",
        "qdrant_points":   points,
        "opensearch":      OS_BASE or "(not configured)",
    })


@app.route("/encode", methods=["POST"])
def encode():
    """
    Encode a single image URL to a vector.
    Used for diagnostics and backward compat with the local UI server's
    GPU_WORKER_URL=/encode fallback path.
    """
    data      = request.get_json(force=True) or {}
    image_url = data.get("image_url", "").strip()
    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    encoder = get_encoder()
    vec = encoder.encode_url(image_url)
    if vec is None:
        return jsonify({"error": f"Failed to encode: {image_url}"}), 422

    return jsonify({"vector": vec.tolist(), "dim": len(vec)})


@app.route("/search", methods=["POST"])
def search():
    """
    Full pipeline: image URL → GPU encode → Qdrant ANN → OpenSearch enrichment.

    Request JSON:
        image_url   str   URL of the card image to use as query
        top_k       int   number of results (default 20)
        hnsw_ef     int   HNSW ef_search parameter (default 128, use 256 for hard queries)

    Response JSON:
        query_image_url   str   echoed back
        qdrant_results    list  [{qdrant_id, os_id, score, payload, doc}, ...]
        total             int
        encode_ms         int   encoding latency
        qdrant_ms         int   Qdrant search latency
        enrich_ms         int   OpenSearch enrichment latency
    """
    import time

    data      = request.get_json(force=True) or {}
    image_url = data.get("image_url", "").strip()
    top_k     = int(data.get("top_k", 20))
    hnsw_ef   = int(data.get("hnsw_ef", 128))

    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    # ── Step 1: Download image + GPU encode (via batch daemon) ───────────────
    t0 = time.perf_counter()
    try:
        from PIL import Image as PILImage
        from src.embeddings.image_encoder import DOWNLOAD_HEADERS, DOWNLOAD_TIMEOUT

        resp = httpx.get(
            image_url,
            timeout=DOWNLOAD_TIMEOUT,
            follow_redirects=True,
            headers=DOWNLOAD_HEADERS,
        )
        resp.raise_for_status()
        pil_img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to download image: {e}"}), 422

    try:
        vec = _encode_via_batch(pil_img)
    except Exception as e:
        return jsonify({"error": f"Failed to encode image: {e}"}), 422

    encode_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("Encoded {} in {}ms ({}-dim)", image_url[:60], encode_ms, len(vec))

    # ── Step 2: Qdrant ANN search ─────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        from qdrant_client.models import SearchParams, QuantizationSearchParams

        qdrant = get_qdrant()
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=("image", vec.tolist()),
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(
                hnsw_ef=hnsw_ef,
                exact=False,
                quantization=QuantizationSearchParams(rescore=True),
            ),
        )
    except Exception as e:
        logger.error("Qdrant search error: {}", e)
        return jsonify({"error": f"Qdrant search failed: {e}"}), 500

    qdrant_ms = int((time.perf_counter() - t1) * 1000)
    logger.info("Qdrant returned {} hits in {}ms", len(hits), qdrant_ms)

    # Build result list and gather os_ids for enrichment
    qdrant_results = []
    os_ids = []
    for h in hits:
        os_id = str(h.payload.get("os_id", h.id))
        qdrant_results.append({
            "qdrant_id": str(h.id),
            "os_id":     os_id,
            "score":     round(float(h.score), 4),
            "payload":   h.payload,
            "doc":       {},
        })
        os_ids.append(os_id)

    # ── Step 3: OpenSearch enrichment ─────────────────────────────────────────
    t2 = time.perf_counter()
    if os_ids and OS_BASE:
        ids_query = {
            "size": len(os_ids),
            "query": {"ids": {"values": os_ids}},
            "_source": [
                "id", "itemId", "title", "galleryURL", "itemURL",
                "saleType", "currentPrice", "currentPriceCurrency",
                "endTime", "globalId", "source", "itemSpecifics",
            ],
        }
        try:
            with httpx.Client(timeout=15, verify=False) as c:
                r = c.post(
                    f"{OS_BASE}/{OS_INDEX}/_search",
                    json=ids_query,
                    headers=OS_HEADERS,
                )
            os_docs = {
                h["_id"]: h["_source"]
                for h in r.json().get("hits", {}).get("hits", [])
            }
            for item in qdrant_results:
                item["doc"] = os_docs.get(item["os_id"], {})

            matched = sum(1 for item in qdrant_results if item["doc"])
            logger.info("OS enrichment: {}/{} docs matched in {}ms",
                        matched, len(os_ids), int((time.perf_counter() - t2) * 1000))
        except Exception as e:
            logger.warning("OpenSearch enrichment failed: {}", e)

    enrich_ms = int((time.perf_counter() - t2) * 1000)

    return jsonify({
        "query_image_url": image_url,
        "qdrant_results":  qdrant_results,
        "total":           len(qdrant_results),
        "encode_ms":       encode_ms,
        "qdrant_ms":       qdrant_ms,
        "enrich_ms":       enrich_ms,
    })


@app.route("/encode_b64", methods=["POST"])
def encode_b64():
    """
    Encode a base64 image to a vector (diagnostics / testing).

    Request JSON:
        image_b64   str   Base64-encoded image bytes (any format PIL can read)

    Response JSON:
        vector      list[float]
        dim         int
        image_size  str   e.g. "180x224" — size BEFORE square padding
    """
    data      = request.get_json(force=True) or {}
    image_b64 = data.get("image_b64", "").strip()
    if not image_b64:
        return jsonify({"error": "image_b64 is required"}), 400

    try:
        from PIL import Image as PILImage
        img_bytes = base64.b64decode(image_b64)
        pil_img   = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_size = f"{pil_img.width}x{pil_img.height}"
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {e}"}), 400

    try:
        vec = _encode_via_batch(pil_img)
    except Exception as e:
        return jsonify({"error": f"Failed to encode image: {e}"}), 422

    return jsonify({"vector": vec.tolist(), "dim": len(vec), "image_size": orig_size})


@app.route("/search_b64", methods=["POST"])
def search_b64():
    """
    Full pipeline: pre-resized base64 image → GPU encode → Qdrant ANN → OS enrichment.

    The caller (search_ui_server.py) is responsible for:
      - Downloading the image from the CDN at native resolution (no pre-resize)
      - Base64-encoding the raw image bytes

    This worker then:
      - Decodes base64 → PIL image (at native CDN resolution, typically 300–600px)
      - Pads to square with RGB(114,114,114) fill at native resolution
      - Runs CLIP's _preprocess (Resize(224, BICUBIC) → CenterCrop → Normalize)
        — identical to the rds_batch_job backfill pipeline
      - GPU-encodes via the batching daemon
      - Searches Qdrant ANN
      - Enriches results from OpenSearch

    Request JSON:
        image_b64   str   Base64-encoded image bytes at native CDN resolution (no pre-resize)
        top_k       int   number of results (default 20)
        hnsw_ef     int   HNSW ef_search parameter (default 128)

    Response JSON:
        image_size        str   "WxH" of the received image (before padding)
        qdrant_results    list  [{qdrant_id, os_id, score, payload, doc}, ...]
        total             int
        encode_ms         int   decode + GPU encode latency
        qdrant_ms         int   Qdrant search latency
        enrich_ms         int   OpenSearch enrichment latency
    """
    import time

    data      = request.get_json(force=True) or {}
    image_b64 = data.get("image_b64", "").strip()
    top_k     = int(data.get("top_k", 20))
    hnsw_ef   = int(data.get("hnsw_ef", 128))

    if not image_b64:
        return jsonify({"error": "image_b64 is required"}), 400

    # ── Step 1: Decode base64 + GPU encode (via batch daemon) ────────────────
    t0 = time.perf_counter()
    try:
        from PIL import Image as PILImage
        img_bytes = base64.b64decode(image_b64)
        pil_img   = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_size = f"{pil_img.width}x{pil_img.height}"
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {e}"}), 400

    try:
        # encode_batch_pil applies _pad_to_square at native resolution, then
        # CLIP's Resize(224, BICUBIC) — identical to the backfill pipeline
        vec = _encode_via_batch(pil_img)
    except Exception as e:
        return jsonify({"error": f"Failed to encode image: {e}"}), 422

    encode_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("B64 search: decoded {}px image in {}ms ({}-dim)", orig_size, encode_ms, len(vec))

    # ── Step 2: Qdrant ANN search ─────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        from qdrant_client.models import SearchParams, QuantizationSearchParams

        qdrant = get_qdrant()
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=("image", vec.tolist()),
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(
                hnsw_ef=hnsw_ef,
                exact=False,
                quantization=QuantizationSearchParams(rescore=True),
            ),
        )
    except Exception as e:
        logger.error("Qdrant search error: {}", e)
        return jsonify({"error": f"Qdrant search failed: {e}"}), 500

    qdrant_ms = int((time.perf_counter() - t1) * 1000)
    logger.info("Qdrant returned {} hits in {}ms", len(hits), qdrant_ms)

    qdrant_results = []
    os_ids = []
    for h in hits:
        os_id = str(h.payload.get("os_id", h.id))
        qdrant_results.append({
            "qdrant_id": str(h.id),
            "os_id":     os_id,
            "score":     round(float(h.score), 4),
            "payload":   h.payload,
            "doc":       {},
        })
        os_ids.append(os_id)

    # ── Step 3: OpenSearch enrichment ─────────────────────────────────────────
    t2 = time.perf_counter()
    if os_ids and OS_BASE:
        ids_query = {
            "size": len(os_ids),
            "query": {"ids": {"values": os_ids}},
            "_source": [
                "id", "itemId", "title", "galleryURL", "itemURL",
                "saleType", "currentPrice", "currentPriceCurrency",
                "endTime", "globalId", "source", "itemSpecifics",
            ],
        }
        try:
            with httpx.Client(timeout=15, verify=False) as c:
                r = c.post(
                    f"{OS_BASE}/{OS_INDEX}/_search",
                    json=ids_query,
                    headers=OS_HEADERS,
                )
            os_docs = {
                h["_id"]: h["_source"]
                for h in r.json().get("hits", {}).get("hits", [])
            }
            for item in qdrant_results:
                item["doc"] = os_docs.get(item["os_id"], {})

            matched = sum(1 for item in qdrant_results if item["doc"])
            logger.info("OS enrichment: {}/{} docs matched in {}ms",
                        matched, len(os_ids), int((time.perf_counter() - t2) * 1000))
        except Exception as e:
            logger.warning("OpenSearch enrichment failed: {}", e)

    enrich_ms = int((time.perf_counter() - t2) * 1000)

    return jsonify({
        "image_size":     orig_size,
        "qdrant_results": qdrant_results,
        "total":          len(qdrant_results),
        "encode_ms":      encode_ms,
        "qdrant_ms":      qdrant_ms,
        "enrich_ms":      enrich_ms,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def _start_services() -> None:
    """Pre-load CLIP and start the batch daemon. Called at module load when
    running under Gunicorn (imported as a module) or directly via __main__."""
    logger.info("─" * 60)
    logger.info("Card Oracle — GPU Worker Server")
    logger.info("  Port:         {}", WORKER_PORT)
    logger.info("  Device:       {}", EMBEDDING_DEVICE)
    logger.info("  Qdrant:       {}:{} / {}", QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION)
    logger.info("  OpenSearch:   {}", OS_BASE or "(not configured)")
    logger.info("  Batch size:   {}", BATCH_SIZE)
    logger.info("  Batch timeout: {}ms", int(BATCH_TIMEOUT_MS * 1000))
    logger.info("─" * 60)

    # Start the batch encoding daemon before pre-loading CLIP
    t = threading.Thread(target=_batch_daemon, daemon=True, name="batch-encoder")
    t.start()
    logger.info("Batch encoder daemon started")

    # Pre-load CLIP so the first request is not slow
    logger.info("Pre-loading CLIP ViT-L/14...")
    get_encoder()
    logger.info("GPU worker ready")


# Start services when imported by Gunicorn (module-level call)
_start_services()


if __name__ == "__main__":
    # Development only — use Gunicorn in production (see docstring)
    app.run(host="0.0.0.0", port=WORKER_PORT, debug=False, threaded=True)
