#!/usr/bin/env python3
"""
POC component 4 — local 3-way search server (CLIP | DINOv2 | DINOv3).

Loads all three backbones, takes an uploaded query image, embeds it with each,
runs three searches against the cards_backbone_poc Qdrant collection (one per
named vector), and returns the three ranked result lists side by side for human
comparison. Result thumbnails are served as presigned URLs from the images
bucket so nothing in the bucket has to be public.

Uses the legacy client.search + NamedVector API (not query_points) so it works
against a Qdrant 1.8.2 cluster as well as a recent local single-node Qdrant.

Run on the worker (or wherever QDRANT_HOST resolves), then tunnel 8088:
    .venv/bin/python tools/poc_search_server.py --port 8088
    # locally:  ssh -i key.pem -L 8088:localhost:8088 ec2-user@<worker>
    # browse:   http://localhost:8088
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from PIL import Image

import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from tools.poc_common import POC_COLLECTION, POC_ENCODERS, DINO_IMAGE_SIZE, make_s3, image_key
from tools.eval_retrieval_at_scale import build_encoder
from tools.eval_parallel_discrimination import _pick_device

app = FastAPI(title="Backbone POC search")
STATE: dict = {}
TOP_K = 20
PRESIGN_TTL = 3600
IMAGE_BUCKET = os.environ.get("S3_IMAGE_BUCKET") or os.environ.get("S3_VECTOR_BUCKET")
QDRANT_URL = f"http://{os.environ.get('QDRANT_HOST', 'localhost')}:6333"
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None


def _qsearch(vector_name: str, vec: list, limit: int) -> list[dict]:
    """Legacy REST /points/search — works against the 1.8.2 server.

    The 1.18 client dropped the high-level .search method, and query_points
    targets a /points/query endpoint that 1.8.2 doesn't have; this classic
    search endpoint exists in every Qdrant version.
    """
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    body = {"vector": {"name": vector_name, "vector": vec},
            "limit": limit, "with_payload": True}
    r = requests.post(f"{QDRANT_URL}/collections/{POC_COLLECTION}/points/search",
                      json=body, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["result"]


@app.on_event("startup")
def _startup() -> None:
    device = _pick_device()
    logger.info("Loading 3 backbones on {} …", device)
    STATE["encoders"] = {}
    for spec in POC_ENCODERS:
        enc, _ = build_encoder(spec.encoder, DINO_IMAGE_SIZE, spec.fp16, device)
        STATE["encoders"][spec.vector_name] = enc
    STATE["s3"] = make_s3()
    logger.info("Ready — collection '{}'", POC_COLLECTION)


def _presign(payload: dict) -> str:
    key = payload.get("img_256") or image_key(payload.get("os_id", ""), "256")
    try:
        return STATE["s3"].generate_presigned_url(
            "get_object", Params={"Bucket": IMAGE_BUCKET, "Key": key}, ExpiresIn=PRESIGN_TTL)
    except Exception:
        return ""


def _result(h: dict) -> dict:
    p = h.get("payload") or {}
    return {
        "score":       round(float(h.get("score", 0.0)), 4),
        "player":      p.get("player", ""),
        "set":         p.get("set", ""),
        "card_number": p.get("card_number", ""),
        "parallel":    p.get("parallel", ""),
        "year":        p.get("year"),
        "title":       (p.get("title", "") or "")[:90],
        "thumb":       _presign(p),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(ROOT / "tools" / "poc_search_ui.html"))


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "collection": POC_COLLECTION,
            "vectors": [s.vector_name for s in POC_ENCODERS]}


@app.post("/search")
async def search(image: UploadFile = File(...)) -> JSONResponse:
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    out: dict[str, list] = {}
    for spec in POC_ENCODERS:
        vec = STATE["encoders"][spec.vector_name]([pil])[0].tolist()
        hits = _qsearch(spec.vector_name, vec, TOP_K)
        out[spec.vector_name] = [_result(h) for h in hits]
    return JSONResponse({"results": out})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
