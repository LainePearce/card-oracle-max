#!/usr/bin/env python3
"""
GPU worker script: backfill Qdrant from OpenSearch.

Reads jobs from the S3 queue seeded by seed_backfill_queue.py.
For each job:
  1. Scroll OpenSearch for the assigned index / time window.
  2. Batch-check Qdrant — skip IDs already present.
  3. Fetch images for missing IDs (async, with eBay size fallback).
  4. CLIP-encode successful fetches on GPU.
  5. Write parquet shards to S3 (durable store — mandatory before Qdrant).
  6. Upsert image (+specifics for eBay) vectors to Qdrant.

Rate-limiting safeguards
------------------------
  OpenSearch : 500 docs/page, 300 ms inter-page pause (halved at peak hours)
  Qdrant     : 100-point upsert batches, 500 ms pause (doubled at peak hours)
               Pauses 5 min if Qdrant p99 latency > 200 ms
  eBay CDN   : up to 64 concurrent async image fetches per worker,
               capped at 24 concurrent connections per host (i.ebayimg.com),
               over one persistent aiohttp session for the worker's lifetime
  Kill switch: workers exit cleanly if s3://BUCKET/backfill-v2/STOP exists
  Peak hours : 09:00–15:00 UTC — all OS/Qdrant pauses doubled

eBay image URL fallback chain
------------------------------
  If URL ends in l1200.<ext>:
    1. Try original URL
    2. Try l1000.<ext>
    3. Try l800.<ext>
    4. Try l600.<ext>
  Skip only after all four attempts fail.

Usage
-----
    # On each GPU worker (worker_id read from WORKER_INDEX env var):
    python tools/backfill_from_opensearch.py

    # Override worker ID (useful for testing):
    python tools/backfill_from_opensearch.py --worker-id 1

    # Dry run (scroll + Qdrant check only, no writes):
    python tools/backfill_from_opensearch.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import aiohttp
import boto3
import numpy as np
from loguru import logger

from src.embeddings.image_encoder import ImageEncoder
from src.embeddings.text_encoder import TextEncoder, format_specifics
from src.embeddings.vector_store import S3VectorStore, VectorRecord
from src.ingestion.opensearch_reader import (
    classify_index,
    get_opensearch_client,
    SCROLL_SOURCE_FIELDS,
)
from src.ingestion.qdrant_writer import (
    build_point,
    extract_payload,
    get_qdrant_client,
    os_id_to_qdrant_id,
    upsert_batch,
    COLLECTION_NAME,
)

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_MODEL_ID = "clip-vit-l-14"
IMAGE_PARAMS   = "v2-fp16-224px-sqpad"
TEXT_MODEL_ID  = "minilm-l6-v2"
TEXT_PARAMS    = "v1-mean-256tok"

S3_BUCKET       = os.environ["S3_VECTOR_BUCKET"]
S3_PREFIX       = os.environ.get("S3_VECTOR_PREFIX", "vectors")
S3_QUEUE_PREFIX = "backfill-v2/queue"
S3_ACTIVE_PREFIX= "backfill-v2/active"
S3_COMP_PREFIX  = "backfill-v2/complete"
S3_FAIL_PREFIX  = "backfill-v2/failed"
S3_CKPT_PREFIX  = "backfill-v2/checkpoints"
STOP_KEY        = "backfill-v2/STOP"

# OpenSearch paging
OS_PAGE_SIZE    = 250     # cut 1000→250: a whole page of decoded images is held
                          # in RAM at once; 1000 drove the 16GB box into swap.
                          # os/dedup are ~0.1s/page so the extra round-trips are free.
OS_PAGE_PAUSE   = 0.05    # seconds between scroll pages (was 0.30 — cluster handles it fine)

# Qdrant
QDRANT_CHECK_BATCH  = 500   # IDs per retrieve call
QDRANT_CHECK_PAUSE  = 0.0   # no pause needed between retrieve batches
QDRANT_UPSERT_BATCH = 500   # points per upsert (was 100 — 5× fewer round trips)
QDRANT_UPSERT_PAUSE = 0.05  # seconds between upsert batches (was 0.50 — cluster guarded by latency check)
QDRANT_LATENCY_WARN = 5.0   # count() threshold — fires only if cluster is genuinely stressed
QDRANT_LATENCY_PAUSE= 60    # pause duration when threshold exceeded (seconds)

# Image fetching
IMAGE_CONCURRENCY   = 64    # concurrent async downloads (was 16)
IMAGE_PER_HOST      = 24    # max concurrent connections to one host (eBay CDN).
                            # Was effectively 4 via limit_per_host — the real cap,
                            # since every galleryURL is on i.ebayimg.com.
IMAGE_TIMEOUT       = 3.0   # seconds per image attempt
IMAGE_RETRY_PAUSE   = 2.0   # seconds before retrying with fallback URL

# Flush a parquet shard + update checkpoint every N hits
SHARD_FLUSH_SIZE = 5_000

# Peak hours (UTC) — pauses during this window (kept for OS health, reduced impact now)
PEAK_START_UTC = 9
PEAK_END_UTC   = 15

# How often to check kill switch (seconds)
KILL_CHECK_INTERVAL = 60

# How often (seconds) a worker re-scans for orphaned jobs left by other crashed
# workers, regardless of how many jobs it has completed. Without a wall-clock
# trigger a fleet that stays continuously busy on long jobs (with a non-empty
# queue, so no idle polls) would never re-scan and orphans would pile up.
RESCUE_INTERVAL_SECONDS = 600

# ── eBay image URL fallback ───────────────────────────────────────────────────

_L1200_RE = re.compile(r"(l1200)(\.[a-zA-Z]+)$")
_FALLBACK_SIZES = ["l1000", "l800", "l600"]


def _fallback_urls(url: str) -> list[str]:
    """Return [original_url, fallback1, fallback2, fallback3] for l1200 URLs."""
    m = _L1200_RE.search(url)
    if not m:
        return [url]
    ext = m.group(2)
    base = url[:m.start(1)]
    return [url] + [f"{base}{sz}{ext}" for sz in _FALLBACK_SIZES]


async def _fetch_one(session: aiohttp.ClientSession, url: str) -> bytes | None:
    """
    Fetch a single eBay image with size fallback.
    Returns raw image bytes on success, None on failure.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None

    for try_url in _fallback_urls(url):
        try:
            async with session.get(
                try_url,
                timeout=aiohttp.ClientTimeout(total=IMAGE_TIMEOUT),
                allow_redirects=True,
            ) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    if len(data) > 500:   # sanity check — reject empty/placeholder responses
                        return data
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        except Exception as exc:
            logger.debug("Unexpected error fetching {}: {}", try_url, exc)

    return None


# ── Persistent HTTP session for image fetching ────────────────────────────────
# Created once per worker process and reused across every page and job. The old
# code did asyncio.run(...) per page, which built a fresh event loop, TCPConnector
# (DNS cache, connection pool) and ClientSession every page and tore them down —
# so ttl_dns_cache never survived a page and TLS sessions were re-handshaked
# constantly. We keep one loop + session for the worker's lifetime instead.

_HTTP: dict = {}   # {"loop": event_loop, "session": ClientSession}


async def _make_session() -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(
        limit=IMAGE_CONCURRENCY * 2,
        limit_per_host=IMAGE_PER_HOST,
        ttl_dns_cache=600,            # now actually survives between pages
        enable_cleanup_closed=True,   # reap half-closed TLS sockets
    )
    return aiohttp.ClientSession(connector=connector)


def _get_http() -> tuple[asyncio.AbstractEventLoop, aiohttp.ClientSession]:
    """Lazily create (and memoise) the worker's event loop + HTTP session."""
    if "loop" not in _HTTP:
        loop = asyncio.new_event_loop()
        _HTTP["loop"]    = loop
        _HTTP["session"] = loop.run_until_complete(_make_session())
    return _HTTP["loop"], _HTTP["session"]


def _close_http() -> None:
    """Close the persistent session + loop. Called once at worker shutdown."""
    if "loop" in _HTTP:
        try:
            _HTTP["loop"].run_until_complete(_HTTP["session"].close())
            _HTTP["loop"].close()
        except Exception as exc:
            logger.debug("Error closing HTTP session: {}", exc)
        _HTTP.clear()


def fetch_images(url_map: dict[str, str]) -> dict[str, bytes]:
    """
    Fetch images for all URLs concurrently using the persistent session.
    Returns {doc_id_str: image_bytes} for successful fetches only.
    Synchronous wrapper — drives the worker's long-lived event loop.
    """
    if not url_map:
        return {}
    loop, session = _get_http()
    return loop.run_until_complete(_fetch_all(session, url_map))


async def _fetch_all(
    session: aiohttp.ClientSession,
    url_map: dict[str, str],   # {doc_id_str: galleryURL}
) -> dict[str, bytes]:
    sem = asyncio.Semaphore(IMAGE_CONCURRENCY)
    results: dict[str, bytes] = {}

    async def _worker(doc_id: str, url: str) -> None:
        async with sem:
            data = await _fetch_one(session, url)
            if data is not None:
                results[doc_id] = data

    await asyncio.gather(*[_worker(k, v) for k, v in url_map.items()])
    return results


# ── Rate limit helpers ────────────────────────────────────────────────────────

def _is_peak() -> bool:
    """Return True during peak traffic hours (09:00–15:00 UTC)."""
    hour = datetime.now(timezone.utc).hour
    return PEAK_START_UTC <= hour < PEAK_END_UTC


def _os_pause() -> None:
    """Pause between OpenSearch scroll pages."""
    time.sleep(OS_PAGE_PAUSE * (2 if _is_peak() else 1))


def _qdrant_upsert_pause() -> None:
    """Pause between Qdrant upsert batches."""
    time.sleep(QDRANT_UPSERT_PAUSE * (2 if _is_peak() else 1))


# ── Qdrant latency check ──────────────────────────────────────────────────────

def _check_qdrant_latency(qdrant) -> bool:
    """
    Return True if Qdrant appears healthy.
    If p99 search latency exceeds threshold, sleep and return False.
    """
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        # qdrant_client stores timing internally; we use a lightweight count query
        t0 = time.perf_counter()
        qdrant.count(COLLECTION_NAME)
        elapsed = time.perf_counter() - t0
        if elapsed > QDRANT_LATENCY_WARN:
            logger.warning(
                "Qdrant count query took {:.2f}s (>{:.2f}s threshold) — "
                "pausing upserts for {}s",
                elapsed, QDRANT_LATENCY_WARN, QDRANT_LATENCY_PAUSE,
            )
            time.sleep(QDRANT_LATENCY_PAUSE)
            return False
    except Exception as exc:
        logger.warning("Qdrant health check failed: {}", exc)
        time.sleep(60)
        return False
    return True


# ── OpenSearch cluster health check ──────────────────────────────────────────

def _check_os_health(os_client) -> bool:
    """
    Return True if OpenSearch is green/yellow.
    If red, sleep 10 minutes and return False.
    """
    try:
        h = os_client.cluster.health()
        status = h.get("status", "green")
        if status == "red":
            logger.warning("OpenSearch cluster status=red — pausing 10 minutes")
            time.sleep(600)
            return False
    except Exception as exc:
        logger.warning("OS health check failed: {}", exc)
        time.sleep(60)
        return False
    return True


# ── S3 job queue ──────────────────────────────────────────────────────────────

def _claim_next_job(s3, worker_id: int) -> dict | None:
    """
    Scan the S3 queue for the highest-priority unclaimed job and claim it.

    Workers claim from the full pool (no filtering by worker_id) so that
    any number of worker processes can drain the queue in parallel.

    Atomic claim uses S3 conditional write (IfNoneMatch='*') so that only
    one worker wins the race even when all 12 workers pick the same top job.
    The losing workers catch PreconditionFailed and try their next candidate.

    Returns the job dict, or None if the queue is empty.
    """
    import random
    from botocore.exceptions import ClientError

    prefix = f"{S3_QUEUE_PREFIX}/"
    candidates = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                job  = json.loads(body)
                candidates.append((job.get("priority", 99), key, job))
            except Exception:
                pass

    if not candidates:
        return None

    # Pick lowest priority number (highest urgency).
    # Secondary sort by key name for deterministic ordering.
    candidates.sort(key=lambda x: (x[0], x[1]))

    # Try each candidate in priority order until one is claimed atomically.
    for _, queue_key, job in candidates:
        active_key = f"{S3_ACTIVE_PREFIX}/{job['job_id']}.json"
        job["claimed_at"] = datetime.now(timezone.utc).isoformat()
        job["worker_pid"] = os.getpid()

        try:
            # IfNoneMatch='*' — atomic claim: fails if another worker already
            # wrote to this active key, preventing duplicate processing.
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=active_key,
                Body=json.dumps(job, indent=2).encode(),
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("PreconditionFailed", "ConditionalRequestConflict"):
                # Another worker won the race on this job — try the next candidate.
                logger.debug("Job {} already claimed by another worker, skipping",
                             job["job_id"])
                continue
            logger.error("Failed to claim job {}: {}", job["job_id"], exc)
            return None

        # Won the claim — remove from queue.
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=queue_key)
        except Exception as exc:
            logger.warning("Claimed {} but could not delete queue key: {}", job["job_id"], exc)

        logger.info("Claimed job {} (priority={}, ~{:,} docs, worker={})",
                    job["job_id"], job.get("priority"), job.get("doc_count_estimate", 0),
                    worker_id)
        return job

    # All candidates were claimed by other workers between our list and claim.
    logger.debug("All candidates claimed before we could — will retry next cycle")
    return None


def _mark_complete(s3, job: dict, stats: dict) -> None:
    active_key = f"{S3_ACTIVE_PREFIX}/{job['job_id']}.json"
    comp_key   = f"{S3_COMP_PREFIX}/{job['job_id']}.json"
    job["completed_at"] = datetime.now(timezone.utc).isoformat()
    job["stats"]        = stats
    s3.put_object(
        Bucket=S3_BUCKET, Key=comp_key,
        Body=json.dumps(job, indent=2).encode(), ContentType="application/json",
    )
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=active_key)
    except Exception:
        pass
    logger.info("Completed job {} | {}", job["job_id"], stats)


def _mark_failed(s3, job: dict, reason: str) -> None:
    active_key = f"{S3_ACTIVE_PREFIX}/{job['job_id']}.json"
    fail_key   = f"{S3_FAIL_PREFIX}/{job['job_id']}.json"
    job["failed_at"] = datetime.now(timezone.utc).isoformat()
    job["failure_reason"] = reason
    s3.put_object(
        Bucket=S3_BUCKET, Key=fail_key,
        Body=json.dumps(job, indent=2).encode(), ContentType="application/json",
    )
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=active_key)
    except Exception:
        pass
    logger.error("Failed job {}: {}", job["job_id"], reason)


def _save_checkpoint(s3, job: dict, stats: dict, search_after: list | None) -> None:
    ckpt = {
        "job_id":       job["job_id"],
        "stats":        stats,
        "search_after": search_after,
        "saved_at":     datetime.now(timezone.utc).isoformat(),
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{S3_CKPT_PREFIX}/{job['job_id']}.json",
        Body=json.dumps(ckpt, indent=2).encode(),
        ContentType="application/json",
    )


def _load_checkpoint(s3, job_id: str) -> dict | None:
    try:
        body = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"{S3_CKPT_PREFIX}/{job_id}.json",
        )["Body"].read()
        return json.loads(body)
    except Exception:
        return None


def _stop_requested(s3) -> bool:
    """Return True if the kill-switch object exists in S3."""
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=STOP_KEY)
        return True
    except Exception:
        return False


# ── Per-job page query (search_after — no server-side cursor) ─────────────────

def _build_page_query(job: dict, search_after: list | None = None) -> dict:
    """
    Build an OpenSearch search body for this job's time window.

    Uses search_after pagination instead of scroll so there is no server-side
    cursor to expire.  Sort is [endTime asc, _id asc] — the secondary _id sort
    guarantees a strict total order even when many docs share the same endTime.

    search_after is the [endTime_value, _id_value] tuple from the last hit of
    the previous page; omit (or pass None) for the first page.
    """
    ts_start = job.get("ts_start")
    ts_end   = job.get("ts_end")

    if ts_start is None and ts_end is None:
        query = {"match_all": {}}
    elif ts_start is None:
        query = {"range": {"endTime": {"lt": ts_end}}}
    elif ts_end is None:
        query = {"range": {"endTime": {"gte": ts_start}}}
    else:
        query = {"range": {"endTime": {"gte": ts_start, "lt": ts_end}}}

    body: dict = {
        "query":   query,
        "_source": SCROLL_SOURCE_FIELDS,
        "sort":    [{"endTime": "asc"}, {"_id": "asc"}],
        "size":    OS_PAGE_SIZE,
    }
    if search_after is not None:
        body["search_after"] = search_after

    return body


# ── Core job processor ────────────────────────────────────────────────────────

def process_job(
    job:       dict,
    os_client,
    qdrant,
    s3,
    vector_store: S3VectorStore,
    image_encoder: ImageEncoder,
    text_encoder:  TextEncoder | None,
    dry_run:   bool = False,
) -> dict:
    """
    Process one backfill job end-to-end.
    Returns a stats dict.
    """
    stats = {
        "scrolled": 0,
        "already_in_qdrant": 0,
        "missing": 0,
        "images_fetched": 0,
        "images_failed": 0,
        "upserted": 0,
        "errors": 0,
    }
    # Per-stage wall-clock accumulation. Per-page deltas are logged each loop
    # iteration; the cumulative dict is attached to stats so it lands in
    # backfill-v2/complete/{job_id}.json for offline analysis.
    # Note: "os_page" time for iteration N is the fetch of page N+1 (next page
    # is fetched at the end of each iteration). Page 1's fetch is included in
    # iteration 1's accumulator before the per-page snapshot is taken.
    timings = {
        "os_page":      0.0,
        "qdrant_dedup": 0.0,
        "img_fetch":    0.0,
        "pil_decode":   0.0,
        "clip_encode":  0.0,
        "text_encode":  0.0,
        "build_pts":    0.0,
        "flush_s3":     0.0,
        "flush_qdrant": 0.0,
        "os_pause":     0.0,
    }
    page_idx = 0

    index_name = job["index_name"]
    classification = {
        "index_type":         job["index_type"],
        "marketplace":        job["marketplace"],
        "has_item_specifics": job.get("has_item_specifics", False),
    }

    # --- Resume from checkpoint if available ---
    ckpt = _load_checkpoint(s3, job["job_id"])
    resume_search_after: list | None = None
    if ckpt:
        logger.info("Resuming job {} from checkpoint (stats={})", job["job_id"], ckpt.get("stats"))
        stats = ckpt.get("stats", stats)
        resume_search_after = ckpt.get("search_after")

    # ── First page (search_after — no server-side cursor) ────────────────────
    search_after = resume_search_after  # None on first page, list[sort_vals] on resume
    t_os = time.perf_counter()
    try:
        page_resp = os_client.search(
            index=index_name,
            body=_build_page_query(job, search_after),
            request_timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to start search on {index_name}: {exc}") from exc
    timings["os_page"] += time.perf_counter() - t_os

    hits = page_resp["hits"]["hits"]

    # Accumulate points between S3 flushes
    pending_records:  list[VectorRecord] = []
    pending_text_recs: list[VectorRecord] = []
    pending_points:   list = []
    shard_num = stats["upserted"] // SHARD_FLUSH_SIZE  # start from last shard

    last_kill_check = time.time()

    while hits:
        page_idx += 1
        page_start_timings = dict(timings)   # snapshot for per-page deltas
        page_fetched = 0                     # images successfully fetched this page

        # ── Kill-switch check ─────────────────────────────────────────────────
        if time.time() - last_kill_check > KILL_CHECK_INTERVAL:
            if _stop_requested(s3):
                logger.warning("STOP flag found in S3 — exiting cleanly")
                _save_checkpoint(s3, job, stats, search_after)
                raise SystemExit(0)
            last_kill_check = time.time()

        # ── OpenSearch health ─────────────────────────────────────────────────
        if not _check_os_health(os_client):
            continue

        # ── Collect IDs and URLs from this page ───────────────────────────────
        page_docs: list[dict] = []  # [{os_id, qdrant_id, url, hit}]
        for hit in hits:
            src         = hit.get("_source", {})
            raw_id      = src.get("id") or hit.get("_id")
            gallery_url = src.get("galleryURL", "") or ""
            if not raw_id:
                continue
            qdrant_id = os_id_to_qdrant_id(raw_id)
            page_docs.append({
                "os_id":     str(raw_id),
                "qdrant_id": qdrant_id,
                "url":       gallery_url.strip(),
                "hit":       hit,
            })
        stats["scrolled"] += len(page_docs)

        # ── Batch-check Qdrant — which IDs are already present? ──────────────
        all_qdrant_ids = [d["qdrant_id"] for d in page_docs]
        found_ids: set = set()
        t_dedup = time.perf_counter()
        for i in range(0, len(all_qdrant_ids), QDRANT_CHECK_BATCH):
            batch_ids = all_qdrant_ids[i:i + QDRANT_CHECK_BATCH]
            try:
                records = qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=batch_ids,
                    with_vectors=False,
                    with_payload=False,
                )
                found_ids.update(r.id for r in records)
            except Exception as exc:
                logger.warning("Qdrant retrieve failed (batch {}): {}", i, exc)
            time.sleep(QDRANT_CHECK_PAUSE)
        timings["qdrant_dedup"] += time.perf_counter() - t_dedup

        missing_docs = [d for d in page_docs if d["qdrant_id"] not in found_ids]
        stats["already_in_qdrant"] += len(page_docs) - len(missing_docs)
        stats["missing"] += len(missing_docs)

        if missing_docs and not dry_run:
            # ── Fetch images for missing docs ─────────────────────────────────
            url_map = {d["os_id"]: d["url"] for d in missing_docs if d["url"]}
            t_fetch = time.perf_counter()
            fetched: dict[str, bytes] = fetch_images(url_map)
            timings["img_fetch"] += time.perf_counter() - t_fetch
            page_fetched = len(fetched)
            stats["images_fetched"] += len(fetched)
            stats["images_failed"]  += len(url_map) - len(fetched)

            if fetched:
                # ── Convert bytes → PIL → CLIP encode (batched) ───────────────
                from PIL import Image
                import io

                os_ids_ordered: list[str] = []
                pil_images: list = []
                t_pil = time.perf_counter()
                for oid, img_bytes in fetched.items():
                    try:
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        os_ids_ordered.append(oid)
                        pil_images.append(pil_img)
                    except Exception as exc:
                        logger.debug("PIL open failed for {}: {}", oid, exc)
                        stats["images_failed"] += 1
                timings["pil_decode"] += time.perf_counter() - t_pil

                image_vecs = None
                if pil_images:
                    ENCODE_BATCH = 256
                    all_vecs = []
                    t_clip = time.perf_counter()
                    for b_start in range(0, len(pil_images), ENCODE_BATCH):
                        b_imgs = pil_images[b_start:b_start + ENCODE_BATCH]
                        try:
                            all_vecs.append(image_encoder.encode_batch_pil(b_imgs))
                        except Exception as exc:
                            logger.error("CLIP encode failed on batch {}: {}", b_start, exc)
                            stats["errors"] += 1
                            for _ in b_imgs:
                                all_vecs.append(None)
                    timings["clip_encode"] += time.perf_counter() - t_clip
                    image_vecs = all_vecs

                if image_vecs is not None:
                    doc_lookup = {d["os_id"]: d for d in missing_docs}

                    flat_vecs: list[np.ndarray | None] = []
                    for v in image_vecs:
                        if v is None:
                            flat_vecs.append(None)
                        elif v.ndim == 1:
                            flat_vecs.append(v)
                        else:
                            for row in v:
                                flat_vecs.append(row)

                    # ── Batch text encoding — one encode_batch() call for all docs ──
                    # Collect specs texts in os_ids_ordered position order so results
                    # align with flat_vecs without an extra lookup.
                    text_vecs: list[np.ndarray | None] = [None] * len(os_ids_ordered)
                    if text_encoder and classification["has_item_specifics"]:
                        specs_texts: list[str | None] = []
                        for oid in os_ids_ordered:
                            d = doc_lookup.get(oid)
                            if d is None:
                                specs_texts.append(None)
                                continue
                            src_hit = d["hit"].get("_source", {})
                            specs_texts.append(
                                format_specifics(src_hit.get("itemSpecifics") or {}) or None
                            )
                        t_text = time.perf_counter()
                        try:
                            text_vecs = text_encoder.encode_batch(specs_texts)
                        except Exception as exc:
                            logger.warning("encode_batch failed, falling back to None: {}", exc)
                        timings["text_encode"] += time.perf_counter() - t_text

                    t_build = time.perf_counter()
                    for i, os_id_str in enumerate(os_ids_ordered):
                        if i >= len(flat_vecs) or flat_vecs[i] is None:
                            continue
                        ivec = flat_vecs[i]
                        tvec = text_vecs[i] if i < len(text_vecs) else None

                        doc = doc_lookup[os_id_str]
                        hit = doc["hit"]
                        src = hit.get("_source", {})

                        # ── Build payload ─────────────────────────────────────
                        payload = extract_payload(
                            src,
                            doc_id=os_id_str,
                            specifics_source=(
                                "ebay" if classification["has_item_specifics"] else "none"
                            ),
                        )

                        # ── VectorRecords for S3 ──────────────────────────────
                        index_type = classification["index_type"]
                        partition  = (
                            index_name if index_type == "ebay-dated"
                            else index_name[:7]
                        )

                        img_record = VectorRecord(
                            os_id=os_id_str,
                            qdrant_id=str(doc["qdrant_id"]),
                            index_name=index_name,
                            index_type=index_type,
                            vector=ivec.tolist(),
                            vector_type="image",
                            model_id=IMAGE_MODEL_ID,
                            params_hash=IMAGE_PARAMS,
                            source_url=doc["url"],
                            specifics_src=(
                                "ebay" if classification["has_item_specifics"] else "none"
                            ),
                        )
                        pending_records.append(img_record)

                        if tvec is not None:
                            txt_record = VectorRecord(
                                os_id=os_id_str,
                                qdrant_id=str(doc["qdrant_id"]),
                                index_name=index_name,
                                index_type=index_type,
                                vector=tvec.tolist(),
                                vector_type="specifics",
                                model_id=TEXT_MODEL_ID,
                                params_hash=TEXT_PARAMS,
                                source_url="",
                                specifics_src="ebay",
                            )
                            pending_text_recs.append(txt_record)

                        # ── Build Qdrant point ────────────────────────────────
                        point = build_point(
                            qdrant_id=doc["qdrant_id"],
                            payload=payload,
                            image_vec=ivec,
                            specifics_vec=tvec,
                        )
                        if point:
                            pending_points.append(point)
                    timings["build_pts"] += time.perf_counter() - t_build

            # ── Flush when shard is full ──────────────────────────────────────
            if len(pending_points) >= SHARD_FLUSH_SIZE:
                _flush(
                    s3, qdrant, vector_store,
                    pending_records, pending_text_recs, pending_points,
                    shard_num, stats, dry_run,
                    timings=timings,
                )
                shard_num += 1
                pending_records.clear()
                pending_text_recs.clear()
                pending_points.clear()
                if hits:
                    search_after = hits[-1].get("sort")
                _save_checkpoint(s3, job, stats, search_after)

        # ── Advance search_after to last hit on this page ─────────────────────
        if hits:
            search_after = hits[-1].get("sort")

        # ── OS inter-page pause ───────────────────────────────────────────────
        t_pause = time.perf_counter()
        _os_pause()
        timings["os_pause"] += time.perf_counter() - t_pause

        # ── Next page (stateless — no cursor expiry) ──────────────────────────
        t_os = time.perf_counter()
        try:
            page_resp = os_client.search(
                index=index_name,
                body=_build_page_query(job, search_after),
                request_timeout=30,
            )
            hits = page_resp["hits"]["hits"]
        except Exception as exc:
            logger.error("Page fetch failed for {}: {}", index_name, exc)
            break
        timings["os_page"] += time.perf_counter() - t_os

        # ── Per-page timing summary ───────────────────────────────────────────
        d = {k: timings[k] - page_start_timings[k] for k in timings}
        page_total = sum(d.values())
        logger.info(
            "page #{} | docs={} missing={} fetched={} | total={:.2f}s "
            "os={:.2f} dedup={:.2f} fetch={:.2f} pil={:.2f} clip={:.2f} "
            "text={:.2f} build={:.2f} s3={:.2f} qup={:.2f} pause={:.2f}",
            page_idx, len(page_docs), len(missing_docs),
            page_fetched, page_total,
            d["os_page"], d["qdrant_dedup"], d["img_fetch"],
            d["pil_decode"], d["clip_encode"], d["text_encode"],
            d["build_pts"], d["flush_s3"], d["flush_qdrant"], d["os_pause"],
        )

    # ── Final flush ───────────────────────────────────────────────────────────
    if pending_points and not dry_run:
        _flush(
            s3, qdrant, vector_store,
            pending_records, pending_text_recs, pending_points,
            shard_num, stats, dry_run,
            timings=timings,
        )

    # ── Cumulative timing summary ─────────────────────────────────────────────
    total = sum(timings.values()) or 1.0   # avoid div/0 on empty job
    logger.info(
        "TIMINGS job={} pages={} total={:.1f}s | "
        + " ".join(f"{k}={{:.1f}}s({{:.0%}})" for k in timings),
        job["job_id"], page_idx, total,
        *[v for k in timings for v in (timings[k], timings[k] / total)],
    )
    stats["timings"] = {k: round(v, 3) for k, v in timings.items()}
    stats["pages"]   = page_idx

    return stats


def _flush(
    s3,
    qdrant,
    vector_store: S3VectorStore,
    img_records:  list[VectorRecord],
    txt_records:  list[VectorRecord],
    points:       list,
    shard_num:    int,
    stats:        dict,
    dry_run:      bool,
    timings:      dict | None = None,
) -> None:
    """Write S3 parquet shards then upsert to Qdrant."""
    if not points or dry_run:
        return

    # ── S3 first (mandatory) ──────────────────────────────────────────────────
    t_s3 = time.perf_counter()
    try:
        if img_records:
            vector_store.write_shard(img_records, shard_num)
        if txt_records:
            # Specifics shards share the same shard_num but different vector_type key
            vector_store.write_shard(txt_records, shard_num)
    except Exception as exc:
        logger.error("S3 write failed — skipping Qdrant upsert for this batch: {}", exc)
        stats["errors"] += 1
        if timings is not None:
            timings["flush_s3"] += time.perf_counter() - t_s3
        return   # do NOT proceed to Qdrant if S3 failed
    if timings is not None:
        timings["flush_s3"] += time.perf_counter() - t_s3

    # ── Qdrant upsert ─────────────────────────────────────────────────────────
    t_q = time.perf_counter()
    _check_qdrant_latency(qdrant)

    for i in range(0, len(points), QDRANT_UPSERT_BATCH):
        batch = points[i:i + QDRANT_UPSERT_BATCH]
        try:
            upsert_batch(qdrant, batch)
            stats["upserted"] += len(batch)
        except Exception as exc:
            logger.error("Qdrant upsert failed (batch {}/{}): {}", i, len(points), exc)
            stats["errors"] += 1
        _qdrant_upsert_pause()
    if timings is not None:
        timings["flush_qdrant"] += time.perf_counter() - t_q


# ── Orphan rescue ─────────────────────────────────────────────────────────────

def _rescue_orphaned_jobs(s3) -> None:
    """
    Scan active/ for jobs whose worker process is no longer alive (i.e. the
    checkpoint has not been updated for ORPHAN_THRESHOLD_MINUTES) and move them
    back to queue/ so any live worker can re-claim them.

    Called from main() on startup, on idle polls, after every RESCUE_EVERY_N_JOBS
    completed jobs, and at least every RESCUE_INTERVAL_SECONDS of wall clock.

    Workers load checkpoints (which store search_after) when they resume a job,
    so progress is not lost — only the final shard is re-processed.
    """
    ORPHAN_THRESHOLD_MINUTES = 20   # job is declared dead after 20 min without a checkpoint update
    # Rationale: checkpoints are written every 5,000 docs; at ~150K docs/hr that's
    # every ~2 min.  Even at a conservative 30K docs/hr it's every ~10 min.
    # 20 min gives 2× headroom for slow pages / Qdrant backpressure without
    # letting orphans sit idle for 1.5 hours as the old 90-min threshold did.

    now = datetime.now(timezone.utc)
    rescued = 0

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_ACTIVE_PREFIX}/"):
        for obj in page.get("Contents", []):
            active_key = obj["Key"]
            job_id = active_key.split("/")[-1].replace(".json", "")

            # Load checkpoint to find last activity timestamp
            ckpt_key = f"{S3_CKPT_PREFIX}/{job_id}.json"
            stale = False
            try:
                ckpt_body = s3.get_object(Bucket=S3_BUCKET, Key=ckpt_key)["Body"].read()
                ckpt = json.loads(ckpt_body)
                saved_at_str = ckpt.get("saved_at", "")
                if saved_at_str:
                    saved_at = datetime.fromisoformat(saved_at_str)
                    age_minutes = (now - saved_at).total_seconds() / 60
                    stale = age_minutes > ORPHAN_THRESHOLD_MINUTES
                else:
                    stale = True   # no timestamp in checkpoint → treat as stale
            except Exception:
                # No checkpoint at all — check how long the job has been active
                try:
                    active_body = s3.get_object(Bucket=S3_BUCKET, Key=active_key)["Body"].read()
                    active_job = json.loads(active_body)
                    claimed_str = active_job.get("claimed_at", "")
                    if claimed_str:
                        claimed_at = datetime.fromisoformat(claimed_str)
                        age_minutes = (now - claimed_at).total_seconds() / 60
                        stale = age_minutes > ORPHAN_THRESHOLD_MINUTES
                    else:
                        stale = True
                except Exception:
                    stale = True   # can't read active file → assume stale

            if not stale:
                continue

            # Move back to queue
            try:
                active_body = s3.get_object(Bucket=S3_BUCKET, Key=active_key)["Body"].read()
                queue_key = f"{S3_QUEUE_PREFIX}/{job_id}.json"
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=queue_key,
                    Body=active_body,
                    ContentType="application/json",
                )
                s3.delete_object(Bucket=S3_BUCKET, Key=active_key)
                logger.info("Rescued orphaned job {} → queue", job_id)
                rescued += 1
            except Exception as exc:
                logger.warning("Could not rescue orphaned job {}: {}", job_id, exc)

    if rescued:
        logger.info("Rescued {} orphaned job(s) on startup", rescued)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Qdrant from OpenSearch")
    ap.add_argument("--worker-id", type=int, default=None,
                    help="Worker ID (0/1/2). Default: WORKER_INDEX env var.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scroll + Qdrant check only — no writes.")
    ap.add_argument("--image-device", default="cuda",
                    help="Device for CLIP encoding (cuda/cpu).")
    args = ap.parse_args()

    worker_id = args.worker_id
    if worker_id is None:
        worker_id = int(os.environ.get("WORKER_INDEX", 0))

    logger.info("=" * 60)
    logger.info("Backfill worker {}", worker_id)
    logger.info("Dry run: {}", args.dry_run)
    logger.info("=" * 60)

    # ── Initialise connections ────────────────────────────────────────────────
    logger.info("Connecting to OpenSearch...")
    os_client = get_opensearch_client()

    logger.info("Connecting to Qdrant...")
    qdrant = get_qdrant_client()

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))
    vector_store = S3VectorStore(bucket=S3_BUCKET, prefix=S3_PREFIX)

    logger.info("Loading CLIP ({}) on {}...", IMAGE_MODEL_ID, args.image_device)
    image_encoder = ImageEncoder(device=args.image_device)

    logger.info("Loading MiniLM text encoder on CPU...")
    try:
        text_encoder: TextEncoder | None = TextEncoder(device="cpu")
    except Exception as exc:
        logger.warning("TextEncoder failed to load — specifics vectors disabled: {}", exc)
        text_encoder = None

    # ── Rescue orphaned jobs from previous crashed workers ────────────────────
    _rescue_orphaned_jobs(s3)

    # ── Job loop ──────────────────────────────────────────────────────────────
    idle_polls = 0
    jobs_since_rescue = 0
    last_rescue_time = time.time()   # _rescue_orphaned_jobs() just ran on startup
    RESCUE_EVERY_N_JOBS = 5          # re-scan for orphans after every N completed jobs

    while True:
        if _stop_requested(s3):
            logger.info("STOP flag found — worker {} exiting cleanly", worker_id)
            break

        # Re-scan for orphans left by other crashed workers. Fires on:
        #   - every RESCUE_EVERY_N_JOBS completed jobs (short-job fleets), and
        #   - every RESCUE_INTERVAL_SECONDS of wall clock (long-job fleets that
        #     stay busy with a non-empty queue, so no idle poll ever triggers it).
        # (Idle workers also rescan directly in the job==None branch below.)
        if (jobs_since_rescue >= RESCUE_EVERY_N_JOBS
                or time.time() - last_rescue_time >= RESCUE_INTERVAL_SECONDS):
            _rescue_orphaned_jobs(s3)
            jobs_since_rescue = 0
            last_rescue_time = time.time()

        job = _claim_next_job(s3, worker_id)
        if job is None:
            _rescue_orphaned_jobs(s3)   # idle worker: good time to scan for orphans
            jobs_since_rescue = 0
            last_rescue_time = time.time()
            idle_polls += 1
            wait = min(30 * idle_polls, 300)
            logger.info("Queue empty — worker {} waiting {}s...", worker_id, wait)
            time.sleep(wait)
            continue

        idle_polls = 0  # reset backoff on successful claim

        try:
            stats = process_job(
                job=job,
                os_client=os_client,
                qdrant=qdrant,
                s3=s3,
                vector_store=vector_store,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                dry_run=args.dry_run,
            )
            _mark_complete(s3, job, stats)
            jobs_since_rescue += 1

        except SystemExit:
            logger.info("Worker {} shut down via kill switch", worker_id)
            break

        except Exception as exc:
            logger.exception("Job {} failed: {}", job["job_id"], exc)
            _mark_failed(s3, job, str(exc))
            jobs_since_rescue += 1
            time.sleep(30)   # brief pause before claiming next job after failure

    # ── Clean shutdown ────────────────────────────────────────────────────────
    _close_http()
    logger.info("Worker {} stopped", worker_id)


if __name__ == "__main__":
    main()
