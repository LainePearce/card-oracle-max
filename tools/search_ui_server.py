#!/usr/bin/env python3
"""
Card Oracle local search UI server.

Serves the search UI HTML and proxies API calls to OpenSearch and Qdrant.

Flow:
  1. User text searches → proxied to OpenSearch simple_query_string
  2. User clicks image → image URL sent to /api/vector-search
  3. Server encodes image with CLIP ViT-L/14 (locally or via GPU worker)
  4. Server queries Qdrant for top-K visual matches
  5. Server fetches full documents from OpenSearch by returned IDs
  6. Combined results returned to browser

Usage:
    cd /path/to/card-oracle-max
    source .venv/bin/activate
    python tools/search_ui_server.py

    # Optional: offload CLIP encoding to the GPU worker
    GPU_WORKER_URL=http://<ec2-ip>:8081 python tools/search_ui_server.py

Configuration (env vars, loaded from .env):
    OPENSEARCH_HOST         extant OpenSearch cluster hostname
    OPENSEARCH_AUTH_HEADER  "Basic <base64>" auth header (see note below)
    OPENSEARCH_USER / OPENSEARCH_PASSWORD  alternative: basic auth credentials
    OPENSEARCH_INDEX        index pattern to search  (default: *,-*.*,-*-live*)

    QDRANT_HOST             Qdrant host           (default: localhost)
    QDRANT_PORT             Qdrant HTTP port      (default: 6333)
    QDRANT_API_KEY          Qdrant API key        (optional)
    QDRANT_COLLECTION       collection name       (default: cards)

    GPU_WORKER_URL          http://host:port of GPU worker encoding API (optional)
                            If unset, CLIP encoding runs locally on CPU.
    UI_PORT                 server port           (default: 8080)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
from pathlib import Path

import httpx
from flask import Flask, jsonify, request, send_from_directory
from loguru import logger
from dotenv import load_dotenv

# Allow imports from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────

OS_HOST        = os.environ.get("OPENSEARCH_HOST", "search-es130point-vector-h3eaau7mwcpyhgynkthfcwzeje.aos.us-west-1.on.aws")
OS_PORT        = int(os.environ.get("OPENSEARCH_PORT", 443))
OS_USE_SSL     = os.environ.get("OPENSEARCH_USE_SSL", "true").lower() == "true"
OS_INDEX       = os.environ.get("OPENSEARCH_INDEX", "*,-*.*,-*-live*")

# Auth: prefer pre-built header (supports any auth scheme), fall back to user/pass
_AUTH_HEADER = os.environ.get("OPENSEARCH_AUTH_HEADER", "")
if not _AUTH_HEADER:
    _user = os.environ.get("OPENSEARCH_USER", "")
    _pass = os.environ.get("OPENSEARCH_PASSWORD", "")
    if _user and _pass:
        _encoded = base64.b64encode(f"{_user}:{_pass}".encode()).decode()
        _AUTH_HEADER = f"Basic {_encoded}"

OS_BASE = f"{'https' if OS_USE_SSL else 'http'}://{OS_HOST}"
OS_HEADERS = {
    "Content-Type": "application/json",
    **({"Authorization": _AUTH_HEADER} if _AUTH_HEADER else {}),
}

QDRANT_HOST       = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cards")

GPU_WORKER_URL = os.environ.get("GPU_WORKER_URL", "").rstrip("/")
UI_PORT        = int(os.environ.get("UI_PORT", 8080))

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(Path(__file__).parent))

# CLIP encoder — loaded lazily in a thread-safe way on first vector-search call
_encoder       = None
_encoder_lock  = threading.Lock()


def get_encoder():
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                from src.embeddings.image_encoder import ImageEncoder
                logger.info("Loading CLIP ViT-L/14 on CPU for local encoding...")
                _encoder = ImageEncoder(model_name="ViT-L/14", pretrained="openai", device="cpu")
    return _encoder


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "search_ui.html")


@app.route("/api/health")
def health():
    """Check connectivity to OpenSearch and Qdrant."""
    status = {"opensearch": "unknown", "qdrant": "unknown", "gpu_worker": None}

    # OpenSearch
    try:
        with httpx.Client(timeout=5, verify=False) as c:
            r = c.get(f"{OS_BASE}/_cluster/health", headers=OS_HEADERS)
            status["opensearch"] = r.json().get("status", "error")
    except Exception as e:
        status["opensearch"] = f"error: {e}"

    # Qdrant
    try:
        qdrant_headers = {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else {}
        with httpx.Client(timeout=5) as c:
            r = c.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/healthz", headers=qdrant_headers)
            status["qdrant"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    # GPU worker (optional)
    if GPU_WORKER_URL:
        try:
            with httpx.Client(timeout=3) as c:
                r = c.get(f"{GPU_WORKER_URL}/health")
                status["gpu_worker"] = r.json()
        except Exception as e:
            status["gpu_worker"] = f"error: {e}"

    return jsonify(status)


@app.route("/api/text-search")
def text_search():
    """
    Proxy a text search to OpenSearch.
    Mirrors the query structure from the example curl command.

    Query params:
        q          search string (simple_query_string against title)
        from       pagination offset  (default: 0)
        size       result count        (default: 50)
        date_from  ISO date YYYY-MM-DD (optional)
        date_to    ISO date YYYY-MM-DD (optional)
    """
    q         = request.args.get("q", "").strip()
    from_val  = int(request.args.get("from", 0))
    size      = min(int(request.args.get("size", 50)), 1000)
    date_from = request.args.get("date_from", "")
    date_to   = request.args.get("date_to", "")
    sort      = request.args.get("sort", "newest")   # "newest" | "oldest"

    # Build bool query — must clause only if there is a query string
    must: list[dict] = []
    if q:
        must.append({
            "simple_query_string": {
                "fields": ["title"],
                "query": q,
                "default_operator": "and",
            }
        })

    # Date range filter
    filter_clauses: list[dict] = []
    if date_from or date_to:
        range_val: dict = {}
        if date_from:
            range_val["gte"] = date_from
        if date_to:
            range_val["lte"] = date_to + " 23:59:59"
        filter_clauses.append({"range": {"endTime": range_val}})

    query_body: dict = {
        "track_scores": True,
        "from": from_val,
        "size": size,
        "query": {
            "bool": {
                "must": must if must else [{"match_all": {}}],
                **({"filter": filter_clauses} if filter_clauses else {}),
            }
        },
        "sort": [{"endTime": "asc" if sort == "oldest" else "desc"}],
        "_source": [
            "id", "itemId", "title", "galleryURL", "itemURL",
            "saleType", "currentPrice", "currentPriceCurrency",
            "endTime", "globalId", "source", "itemSpecifics",
        ],
    }

    url = f"{OS_BASE}/{OS_INDEX}/_search"
    logger.debug("OS query → {} | body: {}", url, query_body)
    try:
        with httpx.Client(timeout=30, verify=False) as c:
            resp = c.post(url, json=query_body, headers=OS_HEADERS)
        if resp.status_code >= 400:
            logger.error("OpenSearch returned {} for {}\nBody: {}", resp.status_code, url, resp.text[:500])
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        logger.error("OpenSearch text-search error: {}", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/vector-search", methods=["POST"])
def vector_search():
    """
    Vector search: encode image URL → Qdrant ANN → fetch OS docs by returned IDs.

    Request body:
        image_url   URL of the card image to use as query
        top_k       number of Qdrant results to return (default: 20)

    Response:
        query_image_url   the URL that was encoded
        qdrant_results    list of {qdrant_id, os_id, score, payload, doc}
                          doc = full OpenSearch source document
    """
    data      = request.get_json(force=True) or {}
    image_url = data.get("image_url", "").strip()
    top_k     = int(data.get("top_k", 20))

    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    # ── Delegate to GPU worker if configured ─────────────────────────────────
    # Production flow: download image here (close to CDN), resize to 224px long
    # edge, base64-encode, and POST pre-resized bytes to /search_b64.
    # This removes CDN download latency from the GPU worker's hot path and
    # sends only ~15–25 KB over the internal VPC connection instead of having
    # the GPU worker fetch the full-resolution image from eBay's CDN.
    if GPU_WORKER_URL:
        try:
            # Download image at native CDN resolution (no pre-resize).
            # The GPU worker pads to square and lets CLIP's own Resize(224, BICUBIC)
            # do the downscale — exactly matching the backfill pipeline in rds_batch_job.py.
            dl_headers = {"User-Agent": "Mozilla/5.0 (compatible; card-oracle/1.0)"}
            with httpx.Client(timeout=30, follow_redirects=True, verify=False) as c:
                dl = c.get(image_url, headers=dl_headers)
                dl.raise_for_status()

            # Base64-encode the raw image bytes (no PIL decode needed here)
            image_b64 = base64.b64encode(dl.content).decode()
            image_size_kb = len(dl.content) / 1024

            logger.info(
                "Downloaded image {:.1f}KB — posting to GPU worker",
                image_size_kb,
            )

            with httpx.Client(timeout=120) as c:
                r = c.post(
                    f"{GPU_WORKER_URL}/search_b64",
                    json={"image_b64": image_b64, "top_k": top_k},
                )
                r.raise_for_status()
                result = r.json()
                # Stitch query_image_url back in for the UI (it's not in the b64 response)
                result.setdefault("query_image_url", image_url)
                logger.info(
                    "GPU worker b64 search: {} results | encode={}ms qdrant={}ms enrich={}ms",
                    result.get("total", 0),
                    result.get("encode_ms", "?"),
                    result.get("qdrant_ms", "?"),
                    result.get("enrich_ms", "?"),
                )
                return jsonify(result)
        except Exception as e:
            logger.warning("GPU worker search failed, falling back to local pipeline: {}", e)

    # ── Local fallback pipeline (no GPU worker) ───────────────────────────────

    # Step 1: Local CPU encoding
    encoder = get_encoder()
    vec = encoder.encode_url(image_url)
    if vec is None:
        return jsonify({"error": f"Failed to encode image: {image_url}"}), 422
    image_vec = vec.tolist()
    logger.info("Local CPU encoded image ({}-dim)", len(image_vec))

    # Step 2: Qdrant ANN search
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import SearchParams, QuantizationSearchParams

        _use_grpc = os.environ.get("QDRANT_USE_GRPC", "false").lower() in ("true", "1")
        if _use_grpc:
            qdrant_kwargs: dict = {
                "host":        QDRANT_HOST,
                "grpc_port":   QDRANT_PORT,
                "prefer_grpc": True,
                "https":       False,
            }
        else:
            qdrant_kwargs = {
                "url":         f"http://{QDRANT_HOST}:{QDRANT_PORT}",
                "prefer_grpc": False,
            }
        if QDRANT_API_KEY:
            qdrant_kwargs["api_key"] = QDRANT_API_KEY

        qdrant = QdrantClient(**qdrant_kwargs)
        result = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=image_vec,
            using="image",
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(
                hnsw_ef=128,
                exact=False,
                quantization=QuantizationSearchParams(rescore=True),
            ),
        )
        hits = result.points
    except Exception as e:
        logger.error("Qdrant search error: {}", e)
        return jsonify({"error": f"Qdrant search failed: {e}"}), 500

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

    # Step 3: OpenSearch enrichment
    if os_ids:
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
            with httpx.Client(timeout=20, verify=False) as c:
                r = c.post(
                    f"{OS_BASE}/{OS_INDEX}/_search",
                    json=ids_query,
                    headers=OS_HEADERS,
                )
            os_docs = {
                h["_id"]: h["_source"]
                for h in r.json().get("hits", {}).get("hits", [])
            }
            for res in qdrant_results:
                res["doc"] = os_docs.get(res["os_id"], {})
        except Exception as e:
            logger.warning("OpenSearch enrichment failed: {}", e)

    return jsonify({
        "query_image_url": image_url,
        "qdrant_results":  qdrant_results,
        "total":           len(qdrant_results),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("─" * 60)
    logger.info("Card Oracle Search UI")
    logger.info("  UI:          http://localhost:{}", UI_PORT)
    logger.info("  OpenSearch:  {}", OS_BASE)
    logger.info("  Qdrant:      {}:{}", QDRANT_HOST, QDRANT_PORT)
    logger.info("  GPU worker:  {}", GPU_WORKER_URL or "(disabled — local CPU encoding)")
    logger.info("─" * 60)
    logger.info("CLIP model will load on first vector search (~10-30s on CPU)")

    app.run(host="0.0.0.0", port=UI_PORT, debug=False, threaded=True)
