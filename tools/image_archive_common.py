#!/usr/bin/env python3
"""
Shared state/claim logic for the 2026 image-archive fleet.

Image archival is GPU-free and I/O-bound, so it runs on its own S3 claim queue
(parallel to the vector backfill's backfill-v2) — workers atomically claim a
day, download s-l1600 + resize to 512/256, upload to the images bucket, and
mark the day complete with counts.

S3 layout (on the queue bucket = S3_VECTOR_BUCKET):
  image-archive/queue/{date}.json     — pending days
  image-archive/active/{date}.json    — claimed, in progress
  image-archive/complete/{date}.json  — done (with archived/failed/os_total)
  image-archive/manifests/{date}.jsonl.gz — per-day manifest (for the DINOv2 embed)
"""
from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

QUEUE_BUCKET = os.environ.get("S3_VECTOR_BUCKET", "card-oracle-vectors")
IMAGE_BUCKET = os.environ.get("S3_IMAGE_BUCKET") or QUEUE_BUCKET
HOST = socket.gethostname()

PFX       = "image-archive"
QUEUE     = f"{PFX}/queue"
ACTIVE    = f"{PFX}/active"
COMPLETE  = f"{PFX}/complete"
MANIFESTS = f"{PFX}/manifests"


def s3_client():
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_dates(s3, prefix: str) -> list[str]:
    """Return the date strings present under a state prefix."""
    out: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=QUEUE_BUCKET, Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            name = o["Key"].split("/")[-1]
            if name.endswith(".json"):
                out.append(name[:-5])
    return out


def seed_date(s3, date_str: str) -> bool:
    """Queue a day if it isn't already queued/active/complete. Returns True if seeded."""
    for prefix in (QUEUE, ACTIVE, COMPLETE):
        try:
            s3.head_object(Bucket=QUEUE_BUCKET, Key=f"{prefix}/{date_str}.json")
            return False
        except ClientError:
            pass
    s3.put_object(Bucket=QUEUE_BUCKET, Key=f"{QUEUE}/{date_str}.json",
                  Body=json.dumps({"date": date_str, "queued_at": _now()}).encode())
    return True


def claim_next(s3) -> str | None:
    """Atomically claim the newest queued day (IfNoneMatch on the active marker)."""
    for d in sorted(list_dates(s3, QUEUE), reverse=True):
        try:
            s3.put_object(Bucket=QUEUE_BUCKET, Key=f"{ACTIVE}/{d}.json",
                          Body=json.dumps({"date": d, "host": HOST,
                                           "claimed_at": _now()}).encode(),
                          IfNoneMatch="*")
        except ClientError:
            continue  # another worker claimed it first
        try:
            s3.delete_object(Bucket=QUEUE_BUCKET, Key=f"{QUEUE}/{d}.json")
        except ClientError:
            pass
        return d
    return None


def update_active(s3, date_str: str, stats: dict) -> None:
    """Overwrite the active marker with intra-day progress. The worker owns the
    marker after claiming, so this is a plain put (no IfNoneMatch)."""
    s3.put_object(Bucket=QUEUE_BUCKET, Key=f"{ACTIVE}/{date_str}.json",
                  Body=json.dumps({"date": date_str, "host": HOST, **stats,
                                   "updated_at": _now()}).encode())


def mark_complete(s3, date_str: str, stats: dict) -> None:
    s3.put_object(Bucket=QUEUE_BUCKET, Key=f"{COMPLETE}/{date_str}.json",
                  Body=json.dumps({"date": date_str, **stats,
                                   "completed_at": _now()}).encode())
    try:
        s3.delete_object(Bucket=QUEUE_BUCKET, Key=f"{ACTIVE}/{date_str}.json")
    except ClientError:
        pass


def release(s3, date_str: str) -> None:
    """Return a claimed day to the queue (worker died / error)."""
    s3.put_object(Bucket=QUEUE_BUCKET, Key=f"{QUEUE}/{date_str}.json",
                  Body=json.dumps({"date": date_str, "requeued_at": _now()}).encode())
    try:
        s3.delete_object(Bucket=QUEUE_BUCKET, Key=f"{ACTIVE}/{date_str}.json")
    except ClientError:
        pass


def _age_minutes(ts) -> float:
    if not ts:
        return 1e9
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 60
    except Exception:
        return 1e9


def reap_stale(s3, max_age_minutes: float) -> int:
    """Re-queue active days whose marker hasn't updated in max_age_minutes — i.e.
    the owning worker died or was spot-reclaimed mid-day. Returns the count
    re-queued. Run periodically (the seed timer) so the fleet self-heals."""
    n = 0
    for d in list_dates(s3, ACTIVE):
        try:
            m = json.loads(s3.get_object(Bucket=QUEUE_BUCKET,
                                         Key=f"{ACTIVE}/{d}.json")["Body"].read())
        except ClientError:
            continue
        ts = m.get("updated_at") or m.get("claimed_at")
        if _age_minutes(ts) >= max_age_minutes:
            release(s3, d)
            n += 1
    return n
