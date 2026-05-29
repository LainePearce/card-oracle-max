#!/usr/bin/env python3
"""
Backfill Dashboard Server (v2 — S3 job-queue system)
=====================================================
Polls S3 job queues and all 12 EC2 workers via SSH every 60 seconds,
then serves a live status page at http://localhost:8080

New system layout (S3 bucket: card-oracle-vectors):
  backfill-v2/queue/{job_id}.json    — pending jobs
  backfill-v2/active/{job_id}.json   — currently running jobs
  backfill-v2/complete/{job_id}.json — finished jobs (with stats field)
  backfill-v2/failed/{job_id}.json   — failed jobs
  backfill-v2/checkpoints/{job_id}.json — per-job progress checkpoints

Usage:
    python tools/backfill_dashboard.py [--port 8080] [--once]

Requires:
    ~/.ssh/qdrant-test.pem  (SSH key for EC2 workers)
    AWS credentials with S3 read access to card-oracle-vectors
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

try:
    import boto3
    from botocore.exceptions import ClientError
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────

WORKER_IPS = [
    "54.67.55.75",      # w0  (public — also used by local dev)
    "18.145.23.246",    # w1
    "54.183.190.110",   # w2
    "13.56.247.223",    # w3
    "3.101.89.26",      # w4
    "18.144.49.65",     # w5
    "54.241.68.12",     # w6
    "52.53.243.136",    # w7
    "54.67.142.178",    # w8
    "54.241.54.87",     # w9
    "18.144.255.159",   # w10
    "54.183.184.64",    # w11
]

# Private IPs for intra-fleet SSH (used when dashboard runs on w0).
# Avoids internet egress and works with the SG self-referencing rule.
WORKER_PRIVATE_IPS = [
    "172.31.27.245",    # w0
    "172.31.24.68",     # w1
    "172.31.24.6",      # w2
    "172.31.25.124",    # w3
    "172.31.19.46",     # w4
    "172.31.28.58",     # w5
    "172.31.21.234",    # w6
    "172.31.30.116",    # w7
    "172.31.30.131",    # w8
    "172.31.31.113",    # w9
    "172.31.30.91",     # w10
    "172.31.21.49",     # w11
]

# deploy_os_backfill.sh writes worker_ips.json at the repo root, generated from
# the live fleet (public IPs from the deploy + private IPs collected per worker).
# When present it overrides the hardcoded fallbacks above, so a terraform
# recreate needs zero manual IP edits. Absent (e.g. local dev checkout) → the
# hardcoded lists are used.
_IP_FILE = Path(__file__).resolve().parent.parent / "worker_ips.json"
try:
    _ips = json.loads(_IP_FILE.read_text())
    if _ips.get("public"):
        WORKER_IPS = _ips["public"]
    WORKER_PRIVATE_IPS = _ips.get("private") or WORKER_IPS
except (OSError, ValueError):
    pass  # keep hardcoded fallbacks

# Detect if we're running on an EC2 instance in the same VPC.
# If so, use private IPs for SSH — faster and avoids SG public-IP restrictions.
def _detect_private_ip() -> str | None:
    """Return our own private IP if running on EC2 (IMDSv2), else None."""
    try:
        import urllib.request
        # IMDSv2: fetch token first, then use it for the metadata request
        tok_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "10"},
            method="PUT",
        )
        token = urllib.request.urlopen(tok_req, timeout=2).read().decode().strip()
        ip_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/local-ipv4",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(ip_req, timeout=2).read().decode().strip()
    except Exception:
        return None

_MY_PRIVATE_IP  = _detect_private_ip()
_USE_PRIVATE_IPS = _MY_PRIVATE_IP in WORKER_PRIVATE_IPS

# Resolved SSH targets — private IPs when on EC2, public IPs otherwise
SSH_TARGETS = WORKER_PRIVATE_IPS if _USE_PRIVATE_IPS else WORKER_IPS

N_WORKERS = len(WORKER_IPS)

SSH_KEY       = os.path.expanduser("~/.ssh/qdrant-test.pem")

# If the dashboard host doesn't have the SSH key (typical when
# os-dashboard.service runs on worker-0 but the key was never copied there),
# every poll fails with exit 255 and the Worker Fleet section turns into a
# wall of "SSH error: 255". Detect that at startup and degrade gracefully —
# we'll skip the poll loop entirely and surface a single one-time note.
SSH_AVAILABLE = os.path.exists(SSH_KEY)
if not SSH_AVAILABLE:
    print(f"[dashboard] SSH key not found at {SSH_KEY} — "
          "worker fleet polling disabled. Job state still polled from S3.",
          flush=True)
S3_BUCKET     = "card-oracle-vectors"
S3_JOB_PREFIX = "backfill-v2"
POLL_INTERVAL = 60  # seconds

# Priority band definitions — MUST stay in sync with tools/seed_daily_backfill.py.
#
# P1: today back 21 days (2026-05-08 → today as of writing) — was zero coverage
# P2: 2026-01-01 → 2026-05-07                                — was partial coverage
# P3: 2025-01-01 → 2025-12-31                                — 2025 historical
# P4: any other eBay-dated index (older history, pre-2025 leftovers)
# P5: non-eBay markets (end in -gold, -pris, -heritage, -pwcc, -ms)
#
# Ranges use [start, end) — start inclusive, end exclusive.
# P1's end advances with `date.today()`; the dashboard's systemd unit gets
# restarted on each deploy_os_backfill run, which refreshes this at startup.
_TODAY     = date.today()
_P1_START  = "2026-05-08"
_P1_END    = (_TODAY + timedelta(days=1)).isoformat()   # include today
_P2_START  = "2026-01-01"
_P2_END    = _P1_START
_P3_START  = "2025-01-01"
_P3_END    = _P2_START

PRIORITY_BANDS: list[dict] = [
    {"p": 1, "label": f"P1 — {_P1_START} → today",
     "start": _P1_START, "end": _P1_END},
    {"p": 2, "label": f"P2 — {_P2_START} → 2026-05-07",
     "start": _P2_START, "end": _P2_END},
    {"p": 3, "label": "P3 — 2025-01-01 → 2025-12-31",
     "start": _P3_START, "end": _P3_END},
    {"p": 4, "label": "P4 — Other eBay dated",    "start": None, "end": None},
    {"p": 5, "label": "P5 — Non-eBay markets",    "start": None, "end": None},
]

NON_EBAY_SUFFIXES = ("-gold", "-pris", "-heritage", "-pwcc", "-ms")

# Calendar coverage: eBay dated indices from Jan 2024 to present
CALENDAR_START = date(2024, 1, 1)


# ── Priority classifier ────────────────────────────────────────────────────────

def classify_priority(job_id: str) -> int:
    """Return 1–5 for a job_id based on priority band rules."""
    # Strip window suffix (e.g. "2025-10-03-w1" → "2025-10-03")
    m = re.match(r'^(\d{4}-\d{2}-\d{2})', job_id)
    if not m:
        # Non-eBay: job_id ends in a non-eBay suffix or has no date prefix
        return 5
    index_date = m.group(1)
    for band in PRIORITY_BANDS[:3]:  # P1-P3 have explicit ranges
        if band["start"] <= index_date < band["end"]:
            return band["p"]
    return 4  # P4 — other eBay dated


def is_non_ebay_job(job_id: str) -> bool:
    return not re.match(r'^\d{4}-\d{2}-\d{2}', job_id)


# ── S3 polling (direct boto3 — no SSH proxy) ───────────────────────────────────

def _s3_client():
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))


def _list_keys(s3, prefix: str) -> list[str]:
    """List all object keys under a prefix."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _read_json(s3, key: str) -> dict | None:
    try:
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return json.loads(body)
    except Exception:
        return None


def _job_id_from_key(key: str) -> str:
    """Extract job_id from an S3 key like 'backfill-v2/active/2025-10-03-w1.json'."""
    return key.split("/")[-1].removesuffix(".json")


def poll_s3_state() -> dict:
    """
    Fetch job queue state directly from S3.

    Returns:
      {
        "queue_count":    int,
        "active_jobs":    [job_dict, ...],    # full JSON for each active job
        "complete_count": int,
        "failed_count":   int,
        "checkpoints":    {job_id: ckpt_dict},
        "priority_stats": {1: {total, complete, active, queue, failed}, ...},
        "date_coverage":  {YYYY-MM-DD: status},  # "complete"|"active"|"queued"|"failed"|"not_in_queue"
      }
    """
    if not _BOTO3_AVAILABLE:
        return _empty_s3_state()

    try:
        s3 = _s3_client()

        # ── Count queue keys (don't read each file — there are ~1863) ─────────
        queue_keys    = _list_keys(s3, f"{S3_JOB_PREFIX}/queue/")
        active_keys   = _list_keys(s3, f"{S3_JOB_PREFIX}/active/")
        complete_keys = _list_keys(s3, f"{S3_JOB_PREFIX}/complete/")
        failed_keys   = _list_keys(s3, f"{S3_JOB_PREFIX}/failed/")

        queue_count    = len([k for k in queue_keys   if k.endswith(".json")])
        complete_count = len([k for k in complete_keys if k.endswith(".json")])
        failed_count   = len([k for k in failed_keys  if k.endswith(".json")])

        active_job_keys = [k for k in active_keys if k.endswith(".json")]

        # ── Read active job files + their checkpoints (max 12, fast) ──────────
        active_jobs: list[dict] = []
        checkpoints: dict[str, dict] = {}

        def _fetch_active(key: str) -> tuple[dict | None, str, dict | None]:
            job_id = _job_id_from_key(key)
            job    = _read_json(s3, key)
            ckpt_key = f"{S3_JOB_PREFIX}/checkpoints/{job_id}.json"
            ckpt = _read_json(s3, ckpt_key)
            return job, job_id, ckpt

        with ThreadPoolExecutor(max_workers=min(len(active_job_keys), 12) or 1) as ex:
            futures = {ex.submit(_fetch_active, k): k for k in active_job_keys}
            for fut in as_completed(futures):
                try:
                    job, job_id, ckpt = fut.result()
                    if job:
                        job["job_id"] = job_id
                        active_jobs.append(job)
                    if ckpt:
                        checkpoints[job_id] = ckpt
                except Exception:
                    pass

        # ── Build priority band stats ──────────────────────────────────────────
        # For non-active states we only have keys, not full JSON, so classify by job_id.
        priority_stats: dict[int, dict] = {
            p: {"total": 0, "complete": 0, "active": 0, "queue": 0, "failed": 0}
            for p in range(1, 6)
        }

        def _tally(keys: list[str], state: str):
            for k in keys:
                if not k.endswith(".json"):
                    continue
                jid = _job_id_from_key(k)
                p = classify_priority(jid)
                priority_stats[p]["total"]  += 1
                priority_stats[p][state]    += 1

        _tally(queue_keys,    "queue")
        _tally(active_keys,   "active")
        _tally(complete_keys, "complete")
        _tally(failed_keys,   "failed")

        # ── Build date coverage calendar ───────────────────────────────────────
        # Status priority: complete > active > queued > failed > not_in_queue
        date_coverage: dict[str, str] = {}

        def _update_coverage(keys: list[str], status: str, priority: int):
            STATUS_PRIORITY = {"complete": 4, "active": 3, "queued": 2, "failed": 1}
            for k in keys:
                if not k.endswith(".json"):
                    continue
                jid = _job_id_from_key(k)
                m = re.match(r'^(\d{4}-\d{2}-\d{2})', jid)
                if not m:
                    continue
                d = m.group(1)
                current = date_coverage.get(d)
                if current is None or STATUS_PRIORITY.get(status, 0) > STATUS_PRIORITY.get(current, 0):
                    date_coverage[d] = status

        _update_coverage(complete_keys, "complete", 4)
        _update_coverage(active_keys,   "active",   3)
        _update_coverage(queue_keys,    "queued",   2)
        _update_coverage(failed_keys,   "failed",   1)

        # Fill in not_in_queue for every eBay date in range
        today = date.today()
        d = CALENDAR_START
        while d <= today:
            d_str = d.isoformat()
            if d_str not in date_coverage:
                date_coverage[d_str] = "not_in_queue"
            d += timedelta(days=1)

        active_jobs.sort(key=lambda j: j.get("priority", 9))

        return {
            "queue_count":    queue_count,
            "active_jobs":    active_jobs,
            "complete_count": complete_count,
            "failed_count":   failed_count,
            "checkpoints":    checkpoints,
            "priority_stats": priority_stats,
            "date_coverage":  date_coverage,
        }

    except Exception as e:
        print(f"[s3-poll] error: {e}", flush=True)
        return _empty_s3_state()


def _empty_s3_state() -> dict:
    return {
        "queue_count":    0,
        "active_jobs":    [],
        "complete_count": 0,
        "failed_count":   0,
        "checkpoints":    {},
        "priority_stats": {p: {"total": 0, "complete": 0, "active": 0, "queue": 0, "failed": 0} for p in range(1, 6)},
        "date_coverage":  {},
    }


# ── SSH polling ────────────────────────────────────────────────────────────────

_LOG_LINE_RE = re.compile(r'(\d+:\d+:\d+)\s*\|\s*\w+\s*\|\s*(.+)')


def poll_worker_ssh(worker_idx: int, ip: str) -> dict:
    """SSH into one worker and check os-backfill service status."""
    result: dict[str, Any] = {
        "index":    worker_idx,
        "ip":       WORKER_IPS[worker_idx],  # always show public IP in UI
        "service":  "unknown",
        "log_line": "",
        "error":    None,
    }
    try:
        out = subprocess.check_output(
            [
                "ssh", "-i", SSH_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=6",
                "-o", "BatchMode=yes",
                f"ec2-user@{ip}",
                "sudo systemctl is-active os-backfill 2>/dev/null || echo inactive; "
                "sudo journalctl -u os-backfill --no-pager -n 5 --output=cat 2>/dev/null | tail -1",
            ],
            stderr=subprocess.DEVNULL,
            timeout=12,
        ).decode().strip()

        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            result["service"] = lines[0]   # "active" | "inactive" | "failed" | …
        if len(lines) > 1:
            result["log_line"] = lines[-1]

    except subprocess.TimeoutExpired:
        result["error"] = "SSH timeout"
    except subprocess.CalledProcessError as e:
        result["error"] = f"SSH error: {e.returncode}"
    except Exception as e:
        result["error"] = str(e)

    return result


def poll_all_workers_ssh() -> list[dict]:
    # If the dashboard host has no SSH key, render a clean "not polled" row
    # for each worker rather than blasting 12 × "SSH error: 255".
    if not SSH_AVAILABLE:
        return [
            {
                "index":    i,
                "ip":       WORKER_IPS[i],
                "service":  "—",
                "log_line": "",
                "error":    "ssh key not on dashboard host (cosmetic — "
                            "job state still tracked via S3)",
            }
            for i in range(N_WORKERS)
        ]

    results: list[dict | None] = [None] * N_WORKERS
    threads = []

    def _poll(idx: int) -> None:
        target_ip = SSH_TARGETS[idx]
        results[idx] = poll_worker_ssh(idx, target_ip)

    for idx in range(N_WORKERS):
        t = threading.Thread(target=_poll, args=(idx,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=18)

    return [
        r or {"index": i, "ip": WORKER_IPS[i], "service": "unknown", "error": "no response"}
        for i, r in enumerate(results)
    ]


# ── Active job enrichment helpers ──────────────────────────────────────────────

def _age_seconds(ts_iso: str | None) -> float | None:
    if not ts_iso:
        return None
    try:
        t = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds()
    except Exception:
        return None


def _checkpoint_age_seconds(ckpt: dict | None) -> float | None:
    if not ckpt:
        return None
    return _age_seconds(ckpt.get("saved_at"))


def _docs_per_hour(job: dict, ckpt: dict | None) -> float | None:
    """Compute throughput rate from checkpoint or job stats."""
    scrolled = 0
    if ckpt and ckpt.get("stats"):
        scrolled = ckpt["stats"].get("scrolled", 0)
    elif job.get("stats"):
        scrolled = job["stats"].get("scrolled", 0)
    if not scrolled:
        return None
    age_h = (_age_seconds(job.get("claimed_at")) or 0) / 3600
    if age_h < 0.01:
        return None
    return scrolled / age_h


def _progress_pct(job: dict, ckpt: dict | None) -> float | None:
    est = job.get("doc_count_estimate", 0)
    if not est:
        return None
    scrolled = 0
    if ckpt and ckpt.get("stats"):
        scrolled = ckpt["stats"].get("scrolled", 0)
    elif job.get("stats"):
        scrolled = job["stats"].get("scrolled", 0)
    return min(100.0, scrolled / est * 100)


# ── Completion-rate ETA ────────────────────────────────────────────────────────

_prev_complete_count: int | None = None
_prev_poll_time: float | None    = None


def _compute_eta(complete_count: int, queue_count: int, active_count: int) -> str | None:
    global _prev_complete_count, _prev_poll_time

    now = time.monotonic()

    if _prev_complete_count is not None and _prev_poll_time is not None:
        delta_complete = complete_count - _prev_complete_count
        delta_secs     = now - _prev_poll_time
        if delta_secs > 0 and delta_complete > 0:
            rate_per_sec    = delta_complete / delta_secs    # jobs/sec
            remaining       = queue_count + active_count
            eta_secs        = remaining / rate_per_sec
            eta_h           = eta_secs / 3600
            if eta_h < 1:
                return f"~{int(eta_secs / 60)}m"
            if eta_h < 24:
                return f"~{eta_h:.1f}h"
            return f"~{eta_h/24:.1f}d"

    _prev_complete_count = complete_count
    _prev_poll_time      = now
    return None


# ── Main polling loop ──────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_state: dict = {
    "active_jobs":     [],
    "workers":         [],
    "queue_count":     0,
    "complete_count":  0,
    "failed_count":    0,
    "active_count":    0,
    "progress_pct":    0.0,
    "eta":             None,
    "priority_bands":  [],
    "date_coverage":   {},
    "s3_available":    _BOTO3_AVAILABLE,
    "updated_at":      None,
    "next_update_at":  None,
}


def _enrich_active_jobs(active_jobs: list[dict], checkpoints: dict) -> list[dict]:
    enriched = []
    for job in active_jobs:
        jid  = job.get("job_id", "")
        ckpt = checkpoints.get(jid)
        scrolled = 0
        if ckpt and ckpt.get("stats"):
            scrolled = ckpt["stats"].get("scrolled", 0)
        elif job.get("stats"):
            scrolled = job["stats"].get("scrolled", 0)

        enriched.append({
            **job,
            "_scrolled":        scrolled,
            "_progress_pct":    _progress_pct(job, ckpt),
            "_age_secs":        _age_seconds(job.get("claimed_at")),
            "_ckpt_age_secs":   _checkpoint_age_seconds(ckpt),
            "_docs_per_hour":   _docs_per_hour(job, ckpt),
            "_priority":        classify_priority(jid),
        })
    return enriched


def _build_priority_bands(priority_stats: dict) -> list[dict]:
    rows = []
    for band in PRIORITY_BANDS:
        p   = band["p"]
        st  = priority_stats.get(p, {})
        total    = st.get("total", 0)
        complete = st.get("complete", 0)
        active   = st.get("active", 0)
        queue    = st.get("queue", 0)
        failed   = st.get("failed", 0)
        pct = round(complete / total * 100, 1) if total else 0.0
        rows.append({
            "p":        p,
            "label":    band["label"],
            "total":    total,
            "complete": complete,
            "active":   active,
            "queue":    queue,
            "failed":   failed,
            "pct":      pct,
        })
    return rows


def poll_and_update() -> None:
    """Run one full poll cycle and update shared state."""
    now = datetime.now(timezone.utc)

    _s3_results:  dict = {}
    _ssh_results: dict = {}

    s3_thread  = threading.Thread(
        target=lambda: _s3_results.update(poll_s3_state()), daemon=True
    )
    ssh_thread = threading.Thread(
        target=lambda: _ssh_results.update({"workers": poll_all_workers_ssh()}), daemon=True
    )

    s3_thread.start()
    ssh_thread.start()
    s3_thread.join(timeout=45)
    ssh_thread.join(timeout=20)

    s3  = _s3_results if _s3_results else _empty_s3_state()
    workers = _ssh_results.get("workers", [
        {"index": i, "ip": WORKER_IPS[i], "service": "unknown", "error": "poll failed"}
        for i in range(N_WORKERS)
    ])

    queue_count    = s3.get("queue_count", 0)
    complete_count = s3.get("complete_count", 0)
    failed_count   = s3.get("failed_count", 0)
    active_jobs_raw = s3.get("active_jobs", [])
    active_count   = len(active_jobs_raw)

    total_jobs = queue_count + active_count + complete_count + failed_count
    progress_pct = round(complete_count / total_jobs * 100, 1) if total_jobs else 0.0

    eta = _compute_eta(complete_count, queue_count, active_count)

    active_jobs = _enrich_active_jobs(active_jobs_raw, s3.get("checkpoints", {}))
    priority_bands = _build_priority_bands(s3.get("priority_stats", {}))
    date_coverage  = s3.get("date_coverage", {})

    with _state_lock:
        _state["active_jobs"]    = active_jobs
        _state["workers"]        = workers
        _state["queue_count"]    = queue_count
        _state["complete_count"] = complete_count
        _state["failed_count"]   = failed_count
        _state["active_count"]   = active_count
        _state["progress_pct"]   = progress_pct
        _state["eta"]            = eta
        _state["priority_bands"] = priority_bands
        _state["date_coverage"]  = date_coverage
        _state["s3_available"]   = _BOTO3_AVAILABLE
        _state["updated_at"]     = now.isoformat()
        _state["next_update_at"] = (now + timedelta(seconds=POLL_INTERVAL)).isoformat()


def _background_poller() -> None:
    while True:
        try:
            poll_and_update()
        except Exception as e:
            print(f"[poller] error: {e}", flush=True)
        time.sleep(POLL_INTERVAL)


# ── HTTP server ────────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backfill Dashboard v2</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #222636;
    --border: #2d3148; --text: #e2e8f0; --muted: #8892b0;
    --green: #10b981; --yellow: #f59e0b; --blue: #3b82f6;
    --red: #ef4444; --purple: #8b5cf6; --gray: #374151;
    --green-dim: #064e3b; --yellow-dim: #451a03; --blue-dim: #1e3a5f;
    --gray-dim: #1f2937; --queued-dim: #312e81; --active-dim: #713f12;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); font-size: 13px; }

  header { padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }
  header h1 span { color: var(--blue); }
  .header-meta { font-size: 11px; color: var(--muted); text-align: right; line-height: 1.6; }
  .countdown { font-weight: 600; color: var(--yellow); }

  .main { padding: 20px 24px; display: grid; gap: 20px; }

  /* Summary cards */
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .card .value { font-size: 22px; font-weight: 700; }
  .card .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .card.green .value { color: var(--green); }
  .card.yellow .value { color: var(--yellow); }
  .card.blue .value  { color: var(--blue); }
  .card.purple .value{ color: var(--purple); }
  .card.red .value   { color: var(--red); }

  /* Overall progress bar */
  .overall-progress { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .overall-progress .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
  .progress-bar-lg { background: var(--gray-dim); border-radius: 6px; height: 14px; overflow: hidden; }
  .progress-fill-lg { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--green), #059669); transition: width 0.6s ease; }
  .progress-meta { display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: var(--muted); }

  /* Section */
  .section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .section-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); margin-bottom: 10px; }

  /* Priority bands */
  .band-grid { display: grid; gap: 8px; }
  .band-row { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; display: grid; grid-template-columns: 180px 1fr 55px; gap: 12px; align-items: center; }
  .band-label { font-weight: 600; font-size: 12px; }
  .progress-bar { background: var(--gray-dim); border-radius: 4px; height: 8px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
  .fill-green  { background: var(--green); }
  .fill-yellow { background: linear-gradient(90deg, var(--green), var(--yellow)); }
  .fill-gray   { background: var(--gray); }
  .band-pct { font-size: 12px; font-weight: 700; text-align: right; color: var(--green); }
  .band-pct.zero { color: var(--muted); }
  .band-sub { font-size: 10px; color: var(--muted); margin-top: 3px; }

  /* Active jobs table */
  .jobs-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .jobs-table th { background: var(--surface2); padding: 7px 9px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); border-bottom: 1px solid var(--border); white-space: nowrap; }
  .jobs-table td { padding: 6px 9px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  .jobs-table tr:last-child td { border-bottom: none; }
  .jobs-table tr:hover td { background: var(--surface2); }
  .mini-bar { display: flex; align-items: center; gap: 6px; }
  .mini-bar-bg { background: var(--gray-dim); border-radius: 3px; height: 6px; width: 60px; overflow: hidden; flex-shrink: 0; }
  .mini-bar-fill { height: 100%; border-radius: 3px; background: var(--green); }
  .mini-pct { font-size: 10px; color: var(--muted); }
  .badge { display: inline-flex; align-items: center; padding: 2px 7px; border-radius: 20px; font-size: 10px; font-weight: 700; white-space: nowrap; }
  .badge.p1 { background: #450a0a; color: var(--red); }
  .badge.p2 { background: #451a03; color: #fb923c; }
  .badge.p3 { background: #1a2e05; color: #86efac; }
  .badge.p4 { background: var(--blue-dim); color: #93c5fd; }
  .badge.p5 { background: var(--queued-dim); color: #c4b5fd; }
  .mono { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
  .muted { color: var(--muted); }
  .err  { color: var(--red); font-size: 11px; }
  .age-warn { color: var(--yellow); }
  .age-stale { color: var(--red); }

  /* Worker fleet table */
  .worker-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .worker-table th { background: var(--surface2); padding: 7px 9px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); border-bottom: 1px solid var(--border); }
  .worker-table td { padding: 6px 9px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  .worker-table tr:last-child td { border-bottom: none; }
  .worker-table tr:hover td { background: var(--surface2); }
  .svc-active   { color: var(--green); font-weight: 700; }
  .svc-inactive { color: var(--red); }
  .svc-unknown  { color: var(--muted); font-style: italic; }
  .ip { font-family: monospace; font-size: 11px; color: var(--muted); }
  .log-preview { font-size: 10px; color: var(--muted); max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: monospace; }

  /* Calendar */
  .calendar-section { overflow-x: auto; }
  .calendar-grid { display: flex; flex-wrap: wrap; gap: 14px; }
  .month-block { min-width: 196px; }
  .month-name { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 5px; }
  .days-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; }
  .day-label { font-size: 9px; color: var(--muted); text-align: center; padding: 1px; }
  .day-cell { width: 100%; padding-bottom: 100%; border-radius: 3px; position: relative; cursor: default; }
  .day-cell:hover::after { content: attr(data-tip); position: absolute; bottom: calc(100% + 4px); left: 50%; transform: translateX(-50%); background: #000; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 10px; white-space: nowrap; z-index: 10; pointer-events: none; }
  .day-cell.complete    { background: var(--green); }
  .day-cell.active      { background: var(--yellow); animation: pulse 1.5s infinite; }
  .day-cell.queued      { background: var(--queued-dim); opacity: 0.85; }
  .day-cell.failed      { background: var(--red); opacity: 0.8; }
  .day-cell.not_in_queue{ background: var(--gray-dim); opacity: 0.4; }
  .day-cell.empty       { background: transparent; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

  /* Legend */
  .legend { display: flex; gap: 14px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--muted); }
  .legend-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
  .legend-dot.complete     { background: var(--green); }
  .legend-dot.active       { background: var(--yellow); }
  .legend-dot.queued       { background: var(--queued-dim); }
  .legend-dot.failed       { background: var(--red); }
  .legend-dot.not_in_queue { background: var(--gray-dim); opacity: 0.5; }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 1000px) { .two-col { grid-template-columns: 1fr; } }

  .loading { color: var(--muted); font-style: italic; font-size: 12px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Card Oracle — <span>Backfill Dashboard v2</span></h1>
    <div style="color:var(--muted);font-size:11px;margin-top:2px">
      S3 job-queue system · card-oracle-vectors/backfill-v2 · 12 × EC2 workers · us-west-1
    </div>
  </div>
  <div class="header-meta">
    <div>Updated: <span id="updated-at">—</span></div>
    <div>Next refresh: <span class="countdown" id="countdown">—</span></div>
  </div>
</header>

<div class="main">

  <!-- Summary cards row -->
  <div class="summary-grid" id="summary-cards">
    <div class="card yellow"><div class="label">Active Jobs</div><div class="value" id="s-active">—</div><div class="sub">currently running</div></div>
    <div class="card blue"><div class="label">Queue Remaining</div><div class="value" id="s-queue">—</div><div class="sub">pending jobs</div></div>
    <div class="card green"><div class="label">Complete</div><div class="value" id="s-complete">—</div><div class="sub">jobs finished</div></div>
    <div class="card red"><div class="label">Failed</div><div class="value" id="s-failed">—</div><div class="sub">jobs failed</div></div>
    <div class="card purple"><div class="label">ETA</div><div class="value" id="s-eta" style="font-size:18px">—</div><div class="sub">estimated finish</div></div>
  </div>

  <!-- Overall progress bar -->
  <div class="overall-progress">
    <div class="label">Overall Progress</div>
    <div class="progress-bar-lg"><div class="progress-fill-lg" id="overall-bar" style="width:0%"></div></div>
    <div class="progress-meta">
      <span id="overall-label">—</span>
      <span id="overall-pct" style="font-weight:700;color:var(--green)">0%</span>
    </div>
  </div>

  <!-- Priority bands + worker fleet -->
  <div class="two-col">

    <div class="section">
      <div class="section-title">Priority Band Progress</div>
      <div class="band-grid" id="band-grid"><div class="loading">Loading…</div></div>
    </div>

    <div class="section">
      <div class="section-title">Worker Fleet (SSH)</div>
      <div style="overflow-x:auto">
        <table class="worker-table">
          <thead><tr>
            <th>#</th><th>IP</th><th>Service</th><th>Last log line</th>
          </tr></thead>
          <tbody id="worker-tbody"><tr><td colspan="4" class="loading">Loading…</td></tr></tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- Active jobs table -->
  <div class="section">
    <div class="section-title">Active Jobs</div>
    <div style="overflow-x:auto">
      <table class="jobs-table">
        <thead><tr>
          <th>Job ID</th>
          <th>Index</th>
          <th>P</th>
          <th>Estimate</th>
          <th>Scrolled</th>
          <th>Progress</th>
          <th>Age</th>
          <th>Checkpoint age</th>
          <th>Rate (docs/hr)</th>
        </tr></thead>
        <tbody id="jobs-tbody"><tr><td colspan="9" class="loading">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- Date coverage calendar -->
  <div class="section calendar-section">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px">
      <div>
        <div class="section-title" style="margin:0">eBay Date Coverage (job state)</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px">
          Jan 2024 – present · newest first · "Complete" = job finished, NOT
          % of OS docs covered. For true coverage run
          <code>tools/audit_daily_coverage.py</code> or check
          <code>backfill-v2/verified/&lt;date&gt;.json</code>.
        </div>
      </div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot complete"></div>Job complete</div>
        <div class="legend-item"><div class="legend-dot active"></div>Active</div>
        <div class="legend-item"><div class="legend-dot queued"></div>Queued</div>
        <div class="legend-item"><div class="legend-dot failed"></div>Failed</div>
        <div class="legend-item"><div class="legend-dot not_in_queue"></div>Not queued</div>
      </div>
    </div>
    <div class="calendar-grid" id="calendar-grid"><div class="loading">Loading…</div></div>
  </div>

</div><!-- .main -->

<script>
const API_URL = '/api/status';
let countdownTimer = null;
let nextUpdateTime = null;

function fmt(n) {
  if (n === null || n === undefined) return '—';
  n = +n;
  if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return Math.round(n).toString();
}

function fmtAge(secs) {
  if (secs === null || secs === undefined) return '—';
  if (secs < 60)   return Math.round(secs) + 's';
  if (secs < 3600) return Math.round(secs / 60) + 'm';
  return (secs / 3600).toFixed(1) + 'h';
}

function ageClass(secs, warnSecs, staleSecs) {
  if (secs === null || secs === undefined) return '';
  if (secs >= staleSecs) return 'age-stale';
  if (secs >= warnSecs)  return 'age-warn';
  return '';
}

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

function pBadge(p) {
  return `<span class="badge p${p}">P${p}</span>`;
}

function svcClass(svc) {
  if (!svc || svc === 'unknown') return 'svc-unknown';
  if (svc === 'active') return 'svc-active';
  return 'svc-inactive';
}

// ── Summary cards ────────────────────────────────────────────────
function renderSummary(data) {
  document.getElementById('s-active').textContent   = data.active_count ?? '—';
  document.getElementById('s-queue').textContent    = fmt(data.queue_count);
  document.getElementById('s-complete').textContent = fmt(data.complete_count);
  document.getElementById('s-failed').textContent   = fmt(data.failed_count);
  document.getElementById('s-eta').textContent      = data.eta || '—';
  document.getElementById('updated-at').textContent = fmtTime(data.updated_at);

  const pct = data.progress_pct ?? 0;
  document.getElementById('overall-bar').style.width = pct + '%';
  document.getElementById('overall-pct').textContent = pct.toFixed(1) + '%';
  const total = (data.complete_count||0) + (data.active_count||0) + (data.queue_count||0) + (data.failed_count||0);
  document.getElementById('overall-label').textContent =
    `${fmt(data.complete_count)} of ${fmt(total)} jobs complete`;
}

// ── Priority bands ───────────────────────────────────────────────
function renderBands(bands) {
  const el = document.getElementById('band-grid');
  if (!bands || !bands.length) { el.innerHTML = '<div class="loading">No data</div>'; return; }
  el.innerHTML = bands.map(b => {
    const pct  = b.pct ?? 0;
    const cls  = pct === 0 ? 'fill-gray' : pct < 100 ? 'fill-yellow' : 'fill-green';
    const pcls = pct === 0 ? 'zero' : '';
    const sub  = b.total === 0
      ? 'no jobs'
      : `${b.complete}/${b.total} complete · ${b.active} active · ${b.queue} queued${b.failed ? ' · ' + b.failed + ' failed' : ''}`;
    return `<div class="band-row">
      <div class="band-label">${b.label}</div>
      <div>
        <div class="progress-bar"><div class="progress-fill ${cls}" style="width:${pct}%"></div></div>
        <div class="band-sub">${sub}</div>
      </div>
      <div class="band-pct ${pcls}">${pct}%</div>
    </div>`;
  }).join('');
}

// ── Active jobs table ────────────────────────────────────────────
function renderJobs(jobs) {
  const tbody = document.getElementById('jobs-tbody');
  if (!jobs || !jobs.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="muted" style="text-align:center;padding:16px">No active jobs</td></tr>';
    return;
  }
  tbody.innerHTML = jobs.map(j => {
    const est      = j.doc_count_estimate ?? 0;
    const scrolled = j._scrolled ?? 0;
    const pct      = j._progress_pct ?? null;
    const ageS     = j._age_secs;
    const ckptS    = j._ckpt_age_secs;
    const rate     = j._docs_per_hour;
    const p        = j._priority ?? j.priority ?? '?';

    // Mini progress bar
    const barFill = pct !== null
      ? `<div class="mini-bar"><div class="mini-bar-bg"><div class="mini-bar-fill" style="width:${Math.min(100,pct)}%"></div></div><span class="mini-pct">${pct.toFixed(1)}%</span></div>`
      : '<span class="muted">—</span>';

    const ckptCell = ckptS === null
      ? '<span class="muted">no ckpt</span>'
      : `<span class="${ageClass(ckptS, 300, 900)}">${fmtAge(ckptS)}</span>`;

    const rateCell = rate !== null ? fmt(rate) : '<span class="muted">—</span>';

    return `<tr>
      <td class="mono">${j.job_id ?? '—'}</td>
      <td class="mono" style="font-size:11px">${j.index_name ?? '—'}</td>
      <td>${pBadge(p)}</td>
      <td class="mono">${fmt(est)}</td>
      <td class="mono">${fmt(scrolled)}</td>
      <td>${barFill}</td>
      <td class="${ageClass(ageS, 3600, 14400)}">${fmtAge(ageS)}</td>
      <td>${ckptCell}</td>
      <td class="mono">${rateCell}</td>
    </tr>`;
  }).join('');
}

// ── Worker fleet ─────────────────────────────────────────────────
function renderWorkers(workers) {
  const tbody = document.getElementById('worker-tbody');
  if (!workers || !workers.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="loading">Loading…</td></tr>';
    return;
  }
  tbody.innerHTML = workers.map(w => {
    const svc    = w.service ?? 'unknown';
    const svcTxt = svc === 'active' ? '● active' : svc === 'inactive' ? '○ inactive' : svc;
    const log    = w.log_line || (w.error ? `ERROR: ${w.error}` : '—');
    return `<tr>
      <td><b>w${w.index}</b></td>
      <td class="ip">${w.ip}</td>
      <td class="${svcClass(svc)}">${svcTxt}</td>
      <td class="log-preview" title="${log.replace(/"/g,'&quot;')}">${log}</td>
    </tr>`;
  }).join('');
}

// ── Calendar ─────────────────────────────────────────────────────
function renderCalendar(coverage) {
  const grid = document.getElementById('calendar-grid');
  if (!coverage || !Object.keys(coverage).length) {
    grid.innerHTML = '<div class="loading">No data</div>';
    return;
  }

  // Group dates by YYYY-MM
  const months = {};
  for (const [d, status] of Object.entries(coverage)) {
    const key = d.slice(0, 7);
    if (!months[key]) months[key] = {};
    months[key][d] = status;
  }

  const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const DAY_NAMES   = ['Mo','Tu','We','Th','Fr','Sa','Su'];
  const sortedMonths = Object.keys(months).sort().reverse(); // newest first

  grid.innerHTML = sortedMonths.map(key => {
    const [y, m] = key.split('-').map(Number);
    const label  = `${MONTH_NAMES[m-1]} ${y}`;
    const dayMap = months[key];

    const firstDow   = (new Date(y, m-1, 1).getDay() + 6) % 7; // Mon=0
    const daysInMonth = new Date(y, m, 0).getDate();

    const headerCells = DAY_NAMES.map(d => `<div class="day-label">${d}</div>`).join('');
    const emptyCells  = Array(firstDow).fill('<div class="day-cell empty"></div>').join('');

    const dayCells = [];
    for (let day = 1; day <= daysInMonth; day++) {
      const dateStr = `${y}-${String(m).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
      const status  = dayMap[dateStr] || 'not_in_queue';
      dayCells.push(`<div class="day-cell ${status}" data-tip="${dateStr} · ${status}"></div>`);
    }

    return `<div class="month-block">
      <div class="month-name">${label}</div>
      <div class="days-grid">
        ${headerCells}${emptyCells}${dayCells.join('')}
      </div>
    </div>`;
  }).join('');
}

// ── Countdown ────────────────────────────────────────────────────
function startCountdown() {
  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = setInterval(() => {
    if (!nextUpdateTime) return;
    const secs = Math.max(0, Math.round((nextUpdateTime - Date.now()) / 1000));
    document.getElementById('countdown').textContent = secs + 's';
  }, 1000);
}

// ── Main fetch ───────────────────────────────────────────────────
async function fetchAndRender() {
  try {
    const resp = await fetch(API_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    nextUpdateTime = data.next_update_at ? new Date(data.next_update_at).getTime() : Date.now() + 60000;

    renderSummary(data);
    renderBands(data.priority_bands || []);
    renderJobs(data.active_jobs || []);
    renderWorkers(data.workers || []);
    renderCalendar(data.date_coverage || {});
  } catch(e) {
    console.error('Fetch error:', e);
  }
}

fetchAndRender();
startCountdown();
setInterval(fetchAndRender, 60000);
</script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server — each request in its own thread."""
    daemon_threads     = True
    allow_reuse_address = True


class DashboardHandler(BaseHTTPRequestHandler):
    timeout = 10

    def log_message(self, fmt, *args):  # suppress access logs
        pass

    def do_GET(self):
        if self.path in ("/", "/dashboard"):
            self._serve_html()
        elif self.path == "/api/status":
            self._serve_api()
        else:
            self.send_error(404)

    def _serve_html(self):
        body = DASHBOARD_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_api(self):
        with _state_lock:
            payload = json.dumps(_state, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(payload))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill dashboard server (v2 — S3 job-queue system)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll and print JSON (no HTTP server)",
    )
    args = parser.parse_args()

    if not _BOTO3_AVAILABLE:
        print("[dashboard] WARNING: boto3 not installed — S3 polling disabled", flush=True)

    if args.once:
        poll_and_update()
        with _state_lock:
            print(json.dumps(_state, default=str, indent=2))
        return

    print("[dashboard] Doing initial poll (this may take ~20 seconds)…", flush=True)
    poll_and_update()
    print("[dashboard] Initial poll complete.", flush=True)

    poller = threading.Thread(target=_background_poller, daemon=True)
    poller.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"[dashboard] Serving at http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Shutting down.", flush=True)


if __name__ == "__main__":
    main()
