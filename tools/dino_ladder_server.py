#!/usr/bin/env python3
"""
DINOv2 score-ladder tuning UI (local).

CLIP's adaptive search ladder (0.99 → 0.95 phase 1, → 0.85 floor) is calibrated
to CLIP's cosine-score distribution and does not transfer to DINOv2. This tool
collects the data needed to build the DINOv2 equivalent: you run real query
images against the search worker's /search_b64_dino endpoint, mark which
results are actually good matches, and the analysis view derives suggested
ladder thresholds (phase-1 start/stop, floor) from the labeled score
distributions.

Flow:
  1. Paste an image URL (eBay CDN etc.) or upload a local image
  2. Server downloads/receives it, base64s it, and queries the search worker:
     /search_b64_dino (labelable) and optionally /search_b64 (CLIP reference)
  3. In the UI, click the last good result ("boundary") — everything above is
     marked good, below bad; individual cards can be toggled after
  4. Save — labels append to a local JSONL
  5. Analysis tab aggregates all labeled queries: good/bad score histograms,
     threshold sweep (precision / recall), and suggested ladder values

Usage (from the repo root on any machine that can reach the search worker):
    source .venv/bin/activate
    python tools/dino_ladder_server.py

Configuration (env, .env is loaded):
    DINO_WORKER_URL   search worker base URL (default: http://13.57.253.55:8081)
    LADDER_UI_PORT    port to listen on      (default: 8090)
    LADDER_LABELS     labels JSONL path
                      (default: experiment/results/dino_ladder/labels.jsonl)
    S3_IMAGE_BUCKET   image-archive bucket for thumbnail fallback
                      (default: images-130-sold; needs local AWS credentials)

Thumbnails: result cards render doc.galleryURL first; when that's missing or
the CDN image is dead (common for older sold listings), the UI falls back to
/api/thumb/<os_id>, which serves the 256px variant from our S3 image archive.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
from flask import Flask, Response, jsonify, request, send_from_directory
from loguru import logger
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from tools.poc_common import image_key

WORKER_URL  = os.environ.get("DINO_WORKER_URL", "http://13.57.253.55:8081").rstrip("/")
UI_PORT     = int(os.environ.get("LADDER_UI_PORT", 8090))
LABELS_PATH = Path(os.environ.get("LADDER_LABELS",
                                  ROOT / "experiment/results/dino_ladder/labels.jsonl"))
IMAGE_BUCKET = os.environ.get("S3_IMAGE_BUCKET", "images-130-sold")
# Source-namespaced key layout: images/{source}/{variant}/{os_id}.jpg
_THUMB_SOURCES = ("ebay", "pris", "pwcc", "gold", "ms", "heri", "other")

DL_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; card-oracle/1.0)"}

app = Flask(__name__, static_folder=str(Path(__file__).parent))


@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "dino_ladder_ui.html")


@app.route("/api/health")
def health():
    try:
        with httpx.Client(timeout=8) as c:
            r = c.get(f"{WORKER_URL}/health")
            worker = r.json()
    except Exception as e:
        worker = {"status": f"error: {e}"}
    return jsonify({"worker_url": WORKER_URL, "worker": worker,
                    "labels_path": str(LABELS_PATH),
                    "labeled_queries": _count_labeled()})


def _count_labeled() -> int:
    if not LABELS_PATH.exists():
        return 0
    with LABELS_PATH.open() as f:
        return sum(1 for line in f if line.strip())


@app.route("/api/query", methods=["POST"])
def query():
    """
    Run one query image through the worker.

    Body: { image_url | image_b64, top_k, score_floor, run_clip }
    Returns: { query_id, image_ref, dino: <worker response>, clip: <worker response>|null }
    """
    data        = request.get_json(force=True) or {}
    image_url   = (data.get("image_url") or "").strip()
    image_b64   = (data.get("image_b64") or "").strip()
    top_k       = int(data.get("top_k", 100))
    score_floor = float(data.get("score_floor", 0.0))
    run_clip    = bool(data.get("run_clip", True))

    if image_url:
        try:
            with httpx.Client(timeout=30, follow_redirects=True, verify=False) as c:
                dl = c.get(image_url, headers=DL_HEADERS)
                dl.raise_for_status()
            image_b64 = base64.b64encode(dl.content).decode()
        except Exception as e:
            return jsonify({"error": f"Failed to download image: {e}"}), 422
        image_ref = image_url
    elif image_b64:
        image_ref = "upload"
    else:
        return jsonify({"error": "image_url or image_b64 is required"}), 400

    query_id = hashlib.sha1(image_b64.encode()).hexdigest()[:16]
    out: dict = {"query_id": query_id, "image_ref": image_ref}

    try:
        with httpx.Client(timeout=180) as c:
            r = c.post(f"{WORKER_URL}/search_b64_dino",
                       json={"image_b64": image_b64, "top_k": top_k,
                             "score_floor": score_floor})
            r.raise_for_status()
            out["dino"] = r.json()
            if run_clip:
                r = c.post(f"{WORKER_URL}/search_b64",
                           json={"image_b64": image_b64, "top_k": top_k})
                r.raise_for_status()
                out["clip"] = r.json()
            else:
                out["clip"] = None
    except Exception as e:
        logger.error("Worker query failed: {}", e)
        return jsonify({"error": f"Worker query failed: {e}"}), 502

    logger.info("query {} — dino {} hits{}", query_id,
                out["dino"].get("total"),
                f", clip {out['clip'].get('total')} hits" if out.get("clip") else "")
    return jsonify(out)


@app.route("/api/label", methods=["POST"])
def label():
    """
    Persist labels for one query.

    Body: { query_id, image_ref, results: [{os_id, score, good}] }
    good must be true/false; unlabeled results should be omitted by the client.
    """
    data    = request.get_json(force=True) or {}
    results = data.get("results") or []
    if not data.get("query_id") or not results:
        return jsonify({"error": "query_id and non-empty results are required"}), 400

    record = {
        "query_id":  data["query_id"],
        "image_ref": data.get("image_ref", ""),
        "ts":        int(time.time()),
        "results":   [{"os_id": r["os_id"], "score": float(r["score"]),
                       "good": bool(r["good"])} for r in results],
    }
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    n_good = sum(1 for r in record["results"] if r["good"])
    logger.info("labels saved: {} — {} good / {} bad", data["query_id"],
                n_good, len(record["results"]) - n_good)
    return jsonify({"ok": True, "labeled_queries": _count_labeled()})


_s3 = None
_thumb_cache: dict[str, str | None] = {}   # os_id → resolved S3 key (None = miss)


def _get_s3():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client("s3")
    return _s3


@app.route("/api/thumb/<os_id>")
def thumb(os_id: str):
    """Serve the archived 256px image for a result whose CDN thumbnail is
    missing/dead. ?src= hints the marketplace (payload index_type); other
    sources are tried after it since the client can't always know."""
    hint = request.args.get("src", "").strip().lower()
    hint = "ebay" if hint in ("ebay-dated", "") else ("heri" if hint == "heritage" else hint)
    sources = [hint] + [s for s in _THUMB_SOURCES if s != hint]

    if os_id in _thumb_cache:
        key = _thumb_cache[os_id]
        if key is None:
            return Response(status=404)
        obj = _get_s3().get_object(Bucket=IMAGE_BUCKET, Key=key)
        return Response(obj["Body"].read(), mimetype="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})

    for src in sources:
        key = image_key(os_id, "256", src)
        try:
            obj = _get_s3().get_object(Bucket=IMAGE_BUCKET, Key=key)
            _thumb_cache[os_id] = key
            return Response(obj["Body"].read(), mimetype="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})
        except Exception as e:
            # A plain missing key is the normal miss path; anything else
            # (NoSuchBucket, AccessDenied, NoCredentials, region) means the
            # proxy itself is broken — surface it instead of blanket-404ing.
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                continue
            logger.warning("thumb S3 error ({}): {} — bucket={} key={}",
                           code or type(e).__name__, e, IMAGE_BUCKET, key)
            return Response(status=502)
    _thumb_cache[os_id] = None
    return Response(status=404)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated percentile of an ascending list; p in [0, 1]."""
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@app.route("/api/analysis")
def analysis():
    """
    Aggregate all labeled queries into ladder-threshold suggestions.

    Semantics mirror the CLIP ladder in gpu_worker_server.py:
      phase1_start ~ CLIP 0.99: near-exact band          → p90 of good scores
      phase1_stop  ~ CLIP 0.95: end of confident band    → p50 of good scores
      floor        ~ CLIP 0.85: minimum plausible match  → p05 of good scores
    Rounded to 0.005. The sweep table shows precision/recall at each threshold
    so the suggestions can be sanity-checked (and hand-adjusted) against the
    actual separation between good and bad scores.
    """
    good: list[float] = []
    bad:  list[float] = []
    n_queries = 0
    # Re-labeling the same query appends a newer record; last record wins.
    latest: dict[str, dict] = {}
    if LABELS_PATH.exists():
        with LABELS_PATH.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    latest[rec["query_id"]] = rec
    for rec in latest.values():
        n_queries += 1
        for r in rec["results"]:
            (good if r["good"] else bad).append(r["score"])

    if not good:
        return jsonify({"n_queries": n_queries, "n_good": 0, "n_bad": len(bad),
                        "message": "No good-labeled results yet — label some queries first."})

    good.sort(), bad.sort()

    def rnd(v: float) -> float:
        return round(round(v / 0.005) * 0.005, 3)

    suggested = {
        "phase1_start": rnd(_percentile(good, 0.90)),
        "phase1_stop":  rnd(_percentile(good, 0.50)),
        "floor":        rnd(_percentile(good, 0.05)),
    }

    lo = min(good[0], bad[0] if bad else good[0])
    hi = max(good[-1], bad[-1] if bad else good[-1])
    sweep = []
    t = max(0.0, round(lo - 0.02, 2))
    while t <= hi + 0.02 and t <= 1.0:
        g_keep = sum(1 for s in good if s >= t)
        b_keep = sum(1 for s in bad if s >= t)
        kept = g_keep + b_keep
        sweep.append({"threshold": round(t, 2),
                      "precision": round(g_keep / kept, 3) if kept else None,
                      "recall":    round(g_keep / len(good), 3),
                      "good_kept": g_keep, "bad_kept": b_keep})
        t = round(t + 0.02, 2)

    buckets = [{"lo": round(b / 20, 2),
                "good": sum(1 for s in good if b / 20 <= s < (b + 1) / 20),
                "bad":  sum(1 for s in bad if b / 20 <= s < (b + 1) / 20)}
               for b in range(20)]

    return jsonify({
        "n_queries": n_queries, "n_good": len(good), "n_bad": len(bad),
        "good_percentiles": {p: round(_percentile(good, q), 4) for p, q in
                             [("p05", .05), ("p25", .25), ("p50", .5),
                              ("p75", .75), ("p90", .9), ("p95", .95)]},
        "bad_percentiles":  {p: round(_percentile(bad, q), 4) for p, q in
                             [("p50", .5), ("p90", .9), ("p95", .95)]} if bad else {},
        "suggested_ladder": suggested,
        "sweep": sweep,
        "histogram": buckets,
    })


if __name__ == "__main__":
    logger.info("─" * 60)
    logger.info("DINOv2 ladder tuning UI")
    logger.info("  UI:      http://localhost:{}", UI_PORT)
    logger.info("  Worker:  {}", WORKER_URL)
    logger.info("  Labels:  {}", LABELS_PATH)
    logger.info("─" * 60)
    app.run(host="127.0.0.1", port=UI_PORT, debug=False, threaded=True)
