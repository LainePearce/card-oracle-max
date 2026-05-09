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
  eBay CDN   : 16 concurrent async image fetches per worker
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
OS_PAGE_SIZE    = 500
OS_PAGE_PAUSE   = 0.30    # seconds between scroll pages (doubled at peak)

# Qdrant
QDRANT_CHECK_BATCH  = 500   # IDs per retrieve call
QDRANT_CHECK_PAUSE  = 0.10  # seconds between retrieve batches
QDRANT_UPSERT_BATCH = 100   # points per upsert
QDRANT_UPSERT_PAUSE = 0.50  # seconds between upsert batches (doubled at peak)
QDRANT_LATENCY_WARN = 5.0   # count() threshold — fires only if cluster is genuinely stressed
QDRANT_LATENCY_PAUSE= 60    # pause duration when threshold exceeded (seconds)

# Image fetching
IMAGE_CONCURRENCY   = 16
IMAGE_TIMEOUT       = 3.0   # seconds per image attempt
IMAGE_RETRY_PAUSE   = 2.0   # seconds before retrying with fallback URL

# Flush a parquet shard + update checkpoint every N hits
SHARD_FLUSH_SIZE = 5_000

# Peak hours (UTC) — all pauses doubled during this window
PEAK_START_UTC = 9
PEAK_END_UTC   = 15

# How often to check kill switch (seconds)
KILL_CHECK_INTERVAL = 60

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


async def fetch_images_async(
    url_map: dict[str, str],   # {doc_id_str: galleryURL}
) -> dict[str, bytes]:
    """
    Fetch images for all URLs concurrently.
    Returns {doc_id_str: image_bytes} for successful fetches only.
    """
    sem = asyncio.Semaphore(IMAGE_CONCURRENCY)
    results: dict[str, bytes] = {}

    async def _worker(doc_id: str, url: str) -> None:
        async with sem:
            data = await _fetch_one(session, url)
            if data is not None:
                results[doc_id] = data

    connector = aiohttp.TCPConnector(
        limit=IMAGE_CONCURRENCY * 2,
        limit_per_host=4,
        ttl_dns_cache=300,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
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
    Scan the S3 queue for the highest-priority unclaimed job matching
    this worker_id. Claim it atomically by copying to active/ then
    deleting from queue/.
    Returns the job dict, or None if queue is empty for this worker.
    """
    # List all queued jobs for this worker
    prefix = f"{S3_QUEUE_PREFIX}/"
    candidates = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(f"-w{worker_id}.json"):
                continue
            try:
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                job  = json.loads(body)
                candidates.append((job.get("priority", 99), key, job))
            except Exception:
                pass

    if not candidates:
        return None

    # Pick lowest priority number (highest urgency)
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, queue_key, job = candidates[0]

    # Claim: copy to active, delete from queue
    active_key = f"{S3_ACTIVE_PREFIX}/{job['job_id']}.json"
    job["claimed_at"] = datetime.now(timezone.utc).isoformat()
    job["worker_pid"] = os.getpid()
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=active_key,
            Body=json.dumps(job, indent=2).encode(),
            ContentType="application/json",
        )
        s3.delete_object(Bucket=S3_BUCKET, Key=queue_key)
    except Exception as exc:
        logger.error("Failed to claim job {}: {}", job["job_id"], exc)
        return None

    logger.info("Claimed job {} (priority={}, ~{:,} docs)",
                job["job_id"], job.get("priority"), job.get("doc_count_estimate", 0))
    return job


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
    try:
        page_resp = os_client.search(
            index=index_name,
            body=_build_page_query(job, search_after),
            request_timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to start search on {index_name}: {exc}") from exc

    hits = page_resp["hits"]["hits"]

    # Accumulate points between S3 flushes
    pending_records:  list[VectorRecord] = []
    pending_text_recs: list[VectorRecord] = []
    pending_points:   list = []
    shard_num = stats["upserted"] // SHARD_FLUSH_SIZE  # start from last shard

    last_kill_check = time.time()

    while hits:
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

        missing_docs = [d for d in page_docs if d["qdrant_id"] not in found_ids]
        stats["already_in_qdrant"] += len(page_docs) - len(missing_docs)
        stats["missing"] += len(missing_docs)

        if missing_docs and not dry_run:
            # ── Fetch images for missing docs ─────────────────────────────────
            url_map = {d["os_id"]: d["url"] for d in missing_docs if d["url"]}
            fetched: dict[str, bytes] = asyncio.run(fetch_images_async(url_map))
            stats["images_fetched"] += len(fetched)
            stats["images_failed"]  += len(url_map) - len(fetched)

            if fetched:
                # ── Convert bytes → PIL → CLIP encode (batched) ───────────────
                from PIL import Image
                import io

                os_ids_ordered: list[str] = []
                pil_images: list = []
                for oid, img_bytes in fetched.items():
                    try:
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        os_ids_ordered.append(oid)
                        pil_images.append(pil_img)
                    except Exception as exc:
                        logger.debug("PIL open failed for {}: {}", oid, exc)
                        stats["images_failed"] += 1

                image_vecs = None
                if pil_images:
                    ENCODE_BATCH = 256
                    all_vecs = []
                    for b_start in range(0, len(pil_images), ENCODE_BATCH):
                        b_imgs = pil_images[b_start:b_start + ENCODE_BATCH]
                        try:
                            all_vecs.append(image_encoder.encode_batch_pil(b_imgs))
                        except Exception as exc:
                            logger.error("CLIP encode failed on batch {}: {}", b_start, exc)
                            stats["errors"] += 1
                            for _ in b_imgs:
                                all_vecs.append(None)
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

                    for i, os_id_str in enumerate(os_ids_ordered):
                        if i >= len(flat_vecs) or flat_vecs[i] is None:
                            continue
                        ivec = flat_vecs[i]

                        doc = doc_lookup[os_id_str]
                        hit = doc["hit"]
                        src = hit.get("_source", {})

                        # ── Text (specifics) vector for eBay only ─────────────
                        tvec = None
                        if text_encoder and classification["has_item_specifics"]:
                            specs_text = format_specifics(src.get("itemSpecifics") or {})
                            if specs_text:
                                try:
                                    tvec = text_encoder.encode(specs_text)
                                except Exception:
                                    pass

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

            # ── Flush when shard is full ──────────────────────────────────────
            if len(pending_points) >= SHARD_FLUSH_SIZE:
                _flush(
                    s3, qdrant, vector_store,
                    pending_records, pending_text_recs, pending_points,
                    shard_num, stats, dry_run,
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
        _os_pause()

        # ── Next page (stateless — no cursor expiry) ──────────────────────────
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

    # ── Final flush ───────────────────────────────────────────────────────────
    if pending_points and not dry_run:
        _flush(
            s3, qdrant, vector_store,
            pending_records, pending_text_recs, pending_points,
            shard_num, stats, dry_run,
        )

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
) -> None:
    """Write S3 parquet shards then upsert to Qdrant."""
    if not points or dry_run:
        return

    # ── S3 first (mandatory) ──────────────────────────────────────────────────
    try:
        if img_records:
            vector_store.write_shard(img_records, shard_num)
        if txt_records:
            # Specifics shards share the same shard_num but different vector_type key
            vector_store.write_shard(txt_records, shard_num)
    except Exception as exc:
        logger.error("S3 write failed — skipping Qdrant upsert for this batch: {}", exc)
        stats["errors"] += 1
        return   # do NOT proceed to Qdrant if S3 failed

    # ── Qdrant upsert ─────────────────────────────────────────────────────────
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

    # ── Job loop ──────────────────────────────────────────────────────────────
    idle_polls = 0
    while True:
        if _stop_requested(s3):
            logger.info("STOP flag found — worker {} exiting cleanly", worker_id)
            break

        job = _claim_next_job(s3, worker_id)
        if job is None:
            idle_polls += 1
            wait = min(30 * idle_polls, 300)
            logger.info("No jobs in queue for worker {} — waiting {}s...", worker_id, wait)
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

        except SystemExit:
            logger.info("Worker {} shut down via kill switch", worker_id)
            break

        except Exception as exc:
            logger.exception("Job {} failed: {}", job["job_id"], exc)
            _mark_failed(s3, job, str(exc))
            time.sleep(30)   # brief pause before claiming next job after failure


if __name__ == "__main__":
    main()
