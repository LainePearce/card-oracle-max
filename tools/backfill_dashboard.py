#!/usr/bin/env python3
"""
Backfill Dashboard Server
=========================
Polls all 12 GPU workers via SSH and S3 checkpoints every 60 seconds,
then serves a live status page at http://localhost:8080

Usage:
    python tools/backfill_dashboard.py [--port 8080]

Requires:
    ~/.ssh/qdrant-test.pem  (SSH key for EC2 workers)
    .env with S3_VECTOR_BUCKET + AWS credentials
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
    "54.176.253.45",    # w0  (public — also used by local dev)
    "204.236.180.247",  # w1
    "54.219.84.133",    # w2
    "13.56.115.249",    # w3
    "13.56.212.61",     # w4
    "13.57.218.120",    # w5
    "54.176.134.82",    # w6
    "13.56.151.99",     # w7
    "18.144.47.150",    # w8
    "3.101.102.118",    # w9
    "184.72.25.240",    # w10
    "13.56.139.224",    # w11
]

# Private IPs for intra-fleet SSH (used when dashboard runs on w0).
# Avoids internet egress and works with the SG self-referencing rule.
WORKER_PRIVATE_IPS = [
    "172.31.30.150",    # w0
    "172.31.20.243",    # w1
    "172.31.20.183",    # w2
    "172.31.19.90",     # w3
    "172.31.25.68",     # w4
    "172.31.28.23",     # w5
    "172.31.27.204",    # w6
    "172.31.29.15",     # w7
    "172.31.25.243",    # w8
    "172.31.17.121",    # w9
    "172.31.28.66",     # w10
    "172.31.19.135",    # w11
]

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

_MY_PRIVATE_IP = _detect_private_ip()
_USE_PRIVATE_IPS = _MY_PRIVATE_IP in WORKER_PRIVATE_IPS

# Resolved SSH targets — private IPs when on EC2, public IPs otherwise
SSH_TARGETS = WORKER_PRIVATE_IPS if _USE_PRIVATE_IPS else WORKER_IPS

N_WORKERS = len(WORKER_IPS)

PHASES: list[tuple[date, date, str]] = [
    (date(2026, 3, 15), date(2026, 4, 8),  "2026 Mar–Apr (priority)"),
    (date(2026, 1, 1),  date(2026, 3, 15), "2026 Jan–Mar"),
    (date(2025, 10, 1), date(2026, 1, 1),  "2025 Q4"),
    (date(2025, 7, 1),  date(2025, 10, 1), "2025 Q3"),
    (date(2025, 4, 1),  date(2025, 7, 1),  "2025 Q2"),
    (date(2025, 1, 1),  date(2025, 4, 1),  "2025 Q1"),
]

SSH_KEY   = os.path.expanduser("~/.ssh/qdrant-test.pem")
S3_BUCKET = os.environ.get("S3_VECTOR_BUCKET", "")
S3_CHECKPOINT_PREFIX = os.environ.get("S3_CHECKPOINT_PREFIX", "checkpoints")
POLL_INTERVAL = 60  # seconds


# ── Date range helpers ─────────────────────────────────────────────────────────

def split_range(start: date, end: date, n: int) -> list[tuple[date, date]]:
    """Mirror of worker_phases.py split_range — identical split logic."""
    total = (end - start).days
    base  = total // n
    extra = total % n
    ranges: list[tuple[date, date]] = []
    cur = start
    for i in range(n):
        days = base + (1 if i < extra else 0)
        nxt  = cur + timedelta(days=days)
        ranges.append((cur, min(nxt, end)))
        cur = nxt
    return ranges


def build_worker_phase_map() -> dict[tuple[int, int], tuple[date, date]]:
    """
    Returns {(worker_idx, phase_num): (slice_start, slice_end)} for all
    worker × phase combinations.
    NOTE: phase_num is 1-indexed (matches worker_phases.py enumerate(PHASES, 1)).
    """
    mapping: dict[tuple[int, int], tuple[date, date]] = {}
    for phase_num, (p_start, p_end, _label) in enumerate(PHASES, 1):  # 1-indexed
        slices = split_range(p_start, p_end, N_WORKERS)
        for w_idx, (s, e) in enumerate(slices):
            mapping[(w_idx, phase_num)] = (s, e)
    return mapping


WORKER_PHASE_MAP = build_worker_phase_map()
PHASE_NUMS = list(range(1, len(PHASES) + 1))  # [1, 2, 3, 4, 5, 6]

# ── Cleanup task definitions (generated identically to worker_cleanup.py) ───────
# Monthly windows newest-first, each split evenly across N_WORKERS.
# All 12 workers work the same month simultaneously before advancing backwards.

_BACKFILL_START = date(2025, 1, 1)
_BACKFILL_END   = date(2026, 4, 8)


def _cleanup_monthly_windows() -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cur_end = _BACKFILL_END
    while cur_end > _BACKFILL_START:
        last_day = cur_end - timedelta(days=1)
        month_start = date(last_day.year, last_day.month, 1)
        win_start = max(_BACKFILL_START, month_start)
        windows.append((win_start, cur_end))
        cur_end = month_start
    return windows


def _cleanup_split_range(start: date, end: date, n: int) -> list[tuple[date, date]]:
    total = (end - start).days
    base  = total // n
    extra = total % n
    slices: list[tuple[date, date]] = []
    cur = start
    for i in range(n):
        days = base + (1 if i < extra else 0)
        nxt  = cur + timedelta(days=days)
        slices.append((cur, min(nxt, end)))
        cur = nxt
    return slices


def _build_cleanup_tasks() -> list[tuple[date, date, str]]:
    tasks: list[tuple[date, date, str]] = []
    for win_start, win_end in _cleanup_monthly_windows():
        month_tag = win_start.strftime("%Y%m")
        slices = _cleanup_split_range(win_start, win_end, N_WORKERS)
        for w_idx, (s, e) in enumerate(slices):
            if (e - s).days == 0:
                continue   # skip zero-day slices (month shorter than N_WORKERS)
            tasks.append((s, e, f"m{month_tag}w{w_idx}"))
    return tasks


CLEANUP_TASKS = _build_cleanup_tasks()

# Pre-build a lookup: for any date, which cleanup task covers it?
_CLEANUP_DATE_MAP: dict[date, str] = {}
for _cs, _ce, _cid in CLEANUP_TASKS:
    _d = _cs
    while _d < _ce:
        _CLEANUP_DATE_MAP[_d] = _cid
        _d += timedelta(days=1)


# ── Non-eBay task definitions (mirrors worker_nonebay.py) ─────────────────────
# Workers 6–11 run these tasks. Checkpoint IDs use nonebay-m{YYYYMM}w{N} prefix
# so they never collide with the eBay cleanup tasks above.

_NONEBAY_START        = date(2025, 1, 1)
_NONEBAY_END          = date.today() + timedelta(days=1)  # dynamic — includes today
_NONEBAY_N_WORKERS    = 6   # workers 6–11 (local indices 0–5)
_NONEBAY_FIRST_WORKER = 6   # global index offset


def _nonebay_monthly_windows() -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cur_end = _NONEBAY_END
    while cur_end > _NONEBAY_START:
        last_day = cur_end - timedelta(days=1)
        month_start = date(last_day.year, last_day.month, 1)
        win_start = max(_NONEBAY_START, month_start)
        windows.append((win_start, cur_end))
        cur_end = month_start
    return windows


def _build_nonebay_tasks() -> list[tuple[date, date, str]]:
    tasks: list[tuple[date, date, str]] = []
    for win_start, win_end in _nonebay_monthly_windows():
        month_tag = win_start.strftime("%Y%m")
        slices = _cleanup_split_range(win_start, win_end, _NONEBAY_N_WORKERS)
        for w_idx, (s, e) in enumerate(slices):
            if (e - s).days == 0:
                continue
            tasks.append((s, e, f"nonebay-m{month_tag}w{w_idx}"))
    return tasks


NONEBAY_TASKS = _build_nonebay_tasks()

# Pre-build a lookup: for any date, which non-eBay task covers it?
_NONEBAY_DATE_MAP: dict[date, str] = {}
for _ns, _ne, _ncid in NONEBAY_TASKS:
    _d = _ns
    while _d < _ne:
        _NONEBAY_DATE_MAP[_d] = _ncid
        _d += timedelta(days=1)


# ── S3 helpers ────────────────────────────────────────────────────────────────
# When running on an EC2 instance with an IAM role (e.g. on worker-0 itself)
# boto3 has direct credentials — no SSH proxy needed.
# When running locally without AWS creds, we SSH to worker-0 and run boto3 there.

_S3_PROXY_IP = WORKER_PRIVATE_IPS[0] if _USE_PRIVATE_IPS else WORKER_IPS[0]


def _fetch_s3_state_direct() -> dict:
    """
    Fetch checkpoint state using local boto3 credentials.
    Used when running on an EC2 instance with an IAM role.
    Returns the same dict structure as poll_s3_via_ssh().
    """
    import boto3
    from botocore.exceptions import ClientError

    s3     = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-1"))
    bucket = os.environ.get("S3_VECTOR_BUCKET", "")
    prefix = os.environ.get("S3_CHECKPOINT_PREFIX", "checkpoints")

    result: dict = {}

    # ── Phase checkpoints ──────────────────────────────────────────────────────
    for w in range(N_WORKERS):
        result[w] = {}
        for p in PHASE_NUMS:
            entry: dict = {"complete": False, "last_completed_date": None}
            mk = f"{prefix}/backfill-w{w}-phase{p}-complete.json"
            try:
                s3.head_object(Bucket=bucket, Key=mk)
                entry["complete"] = True
            except ClientError:
                pass
            if not entry["complete"]:
                ck = f"{prefix}/backfill-w{w}-phase{p}.json"
                try:
                    data = json.loads(s3.get_object(Bucket=bucket, Key=ck)["Body"].read())
                    entry["last_completed_date"] = data.get("last_completed_date")
                except ClientError:
                    pass
            result[w][p] = entry

    # ── Cleanup tasks ──────────────────────────────────────────────────────────
    cleanup: dict = {}
    for _s, _e, cid in CLEANUP_TASKS:
        entry = {"complete": False, "claimed": False, "last_completed_date": None}
        mk = f"{prefix}/{cid}-complete.json"
        try:
            s3.head_object(Bucket=bucket, Key=mk)
            entry["complete"] = True
        except ClientError:
            pass
        if not entry["complete"]:
            clk = f"{prefix}/{cid}-claimed.json"
            try:
                cd = json.loads(s3.get_object(Bucket=bucket, Key=clk)["Body"].read())
                entry["claimed"]    = True
                entry["claimed_by"] = cd.get("worker")
            except ClientError:
                pass
            ck = f"{prefix}/{cid}.json"
            try:
                data = json.loads(s3.get_object(Bucket=bucket, Key=ck)["Body"].read())
                entry["last_completed_date"] = data.get("last_completed_date")
            except ClientError:
                pass
        cleanup[cid] = entry
    result["__cleanup__"] = cleanup

    # ── Daily completion markers ───────────────────────────────────────────────
    daily: dict = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/daily-",
                                   PaginationConfig={"MaxItems": 400}):
        for obj in page.get("Contents", []):
            fname = obj["Key"].split("/")[-1]
            if fname.endswith("-complete.json"):
                day = fname.replace("daily-", "").replace("-complete.json", "")
                if len(day) == 10:
                    daily[day] = True
    result["__daily__"] = daily

    # ── Gap-fill completion markers ────────────────────────────────────────────
    gap_complete: dict = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/gap-",
                                   PaginationConfig={"MaxItems": 200}):
        for obj in page.get("Contents", []):
            fname = obj["Key"].split("/")[-1]
            if fname.endswith("-complete.json"):
                try:
                    data = json.loads(s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
                    cid  = fname.replace("-complete.json", "")
                    gap_complete[cid] = {"start": data.get("start"), "end": data.get("end")}
                except ClientError:
                    pass
    result["__gap_complete__"] = gap_complete

    # ── Non-eBay tasks (nonebay-m{YYYYMM}w{N}) ────────────────────────────────
    nonebay: dict = {}
    for _s, _e, cid in NONEBAY_TASKS:
        entry: dict = {"complete": False, "claimed": False,
                       "start": str(_s), "end": str(_e)}
        mk = f"{prefix}/{cid}-complete.json"
        try:
            s3.head_object(Bucket=bucket, Key=mk)
            entry["complete"] = True
        except ClientError:
            pass
        if not entry["complete"]:
            clk = f"{prefix}/{cid}-claimed.json"
            try:
                cd = json.loads(s3.get_object(Bucket=bucket, Key=clk)["Body"].read())
                entry["claimed"]    = True
                entry["claimed_by"] = cd.get("worker")
            except ClientError:
                pass
        nonebay[cid] = entry
    result["__nonebay__"] = nonebay

    return result


def _build_s3_proxy_script() -> str:
    """Build the remote Python script with values baked in (avoids .format() conflicts)."""
    return (
        "import json,os,boto3\n"
        "from botocore.exceptions import ClientError\n"
        "from dotenv import load_dotenv\n"
        "load_dotenv('/home/ec2-user/card-oracle-max/.env')\n"
        "bucket=os.environ.get('S3_VECTOR_BUCKET','')\n"
        "prefix=os.environ.get('S3_CHECKPOINT_PREFIX','checkpoints')\n"
        "s3=boto3.client('s3',region_name='us-west-1')\n"
        "result=dict()\n"
        f"phase_nums=list(range(1,{len(PHASES)}+1))\n"
        f"n_workers={N_WORKERS}\n"
        "for w in range(n_workers):\n"
        "    result[w]=dict()\n"
        "    for p in phase_nums:\n"
        "        entry=dict(complete=False,last_completed_date=None)\n"
        "        mk=f'{prefix}/backfill-w{w}-phase{p}-complete.json'\n"
        "        try:\n"
        "            s3.head_object(Bucket=bucket,Key=mk)\n"
        "            entry['complete']=True\n"
        "        except ClientError:\n"
        "            pass\n"
        "        if not entry['complete']:\n"
        "            ck=f'{prefix}/backfill-w{w}-phase{p}.json'\n"
        "            try:\n"
        "                data=json.loads(s3.get_object(Bucket=bucket,Key=ck)['Body'].read())\n"
        "                entry['last_completed_date']=data.get('last_completed_date')\n"
        "            except ClientError:\n"
        "                pass\n"
        "        result[w][p]=entry\n"
        # ── Cleanup tasks ──────────────────────────────────────────────────────
        "cleanup={}\n"
        f"cleanup_cids={json.dumps([cid for _s, _e, cid in CLEANUP_TASKS])}\n"
        "for cid in cleanup_cids:\n"
        "    entry=dict(complete=False,claimed=False,last_completed_date=None)\n"
        "    mk=f'{prefix}/{cid}-complete.json'\n"
        "    try:\n"
        "        s3.head_object(Bucket=bucket,Key=mk)\n"
        "        entry['complete']=True\n"
        "    except ClientError:\n"
        "        pass\n"
        "    if not entry['complete']:\n"
        "        clk=f'{prefix}/{cid}-claimed.json'\n"
        "        try:\n"
        "            cd=json.loads(s3.get_object(Bucket=bucket,Key=clk)['Body'].read())\n"
        "            entry['claimed']=True\n"
        "            entry['claimed_by']=cd.get('worker')\n"
        "        except ClientError:\n"
        "            pass\n"
        "        ck=f'{prefix}/{cid}.json'\n"
        "        try:\n"
        "            data=json.loads(s3.get_object(Bucket=bucket,Key=ck)['Body'].read())\n"
        "            entry['last_completed_date']=data.get('last_completed_date')\n"
        "        except ClientError:\n"
        "            pass\n"
        "    cleanup[cid]=entry\n"
        "result['__cleanup__']=cleanup\n"
        # ── Daily completion markers ───────────────────────────────────────────
        "daily={}\n"
        "paginator=s3.get_paginator('list_objects_v2')\n"
        "for page in paginator.paginate(Bucket=bucket,Prefix=prefix+'/daily-',PaginationConfig={'MaxItems':400}):\n"
        "    for obj in page.get('Contents',[]):\n"
        "        key=obj['Key']\n"
        "        fname=key.split('/')[-1]\n"
        "        if fname.endswith('-complete.json'):\n"
        "            day=fname.replace('daily-','').replace('-complete.json','')\n"
        "            if len(day)==10:\n"
        "                daily[day]=True\n"
        "result['__daily__']=daily\n"
        # ── Gap-fill completion markers ────────────────────────────────────────
        "gap_complete={}\n"
        "for page in paginator.paginate(Bucket=bucket,Prefix=prefix+'/gap-',PaginationConfig={'MaxItems':200}):\n"
        "    for obj in page.get('Contents',[]):\n"
        "        key=obj['Key']\n"
        "        fname=key.split('/')[-1]\n"
        "        if fname.endswith('-complete.json'):\n"
        "            try:\n"
        "                data=json.loads(s3.get_object(Bucket=bucket,Key=key)['Body'].read())\n"
        "                gap_complete[fname.replace('-complete.json','')]={'start':data.get('start'),'end':data.get('end')}\n"
        "            except ClientError:\n"
        "                pass\n"
        "result['__gap_complete__']=gap_complete\n"
        # ── Non-eBay tasks ────────────────────────────────────────────────────
        "nonebay={}\n"
        f"nonebay_cids={json.dumps([(str(s), str(e), cid) for s, e, cid in NONEBAY_TASKS])}\n"
        "for s,e,cid in nonebay_cids:\n"
        "    entry=dict(complete=False,claimed=False,start=s,end=e)\n"
        "    mk=f'{prefix}/{cid}-complete.json'\n"
        "    try:\n"
        "        s3.head_object(Bucket=bucket,Key=mk)\n"
        "        entry['complete']=True\n"
        "    except ClientError:\n"
        "        pass\n"
        "    if not entry['complete']:\n"
        "        clk=f'{prefix}/{cid}-claimed.json'\n"
        "        try:\n"
        "            cd=json.loads(s3.get_object(Bucket=bucket,Key=clk)['Body'].read())\n"
        "            entry['claimed']=True\n"
        "            entry['claimed_by']=cd.get('worker')\n"
        "        except ClientError:\n"
        "            pass\n"
        "    nonebay[cid]=entry\n"
        "result['__nonebay__']=nonebay\n"
        "print(json.dumps(result))\n"
    )


def poll_s3_via_ssh() -> dict:
    """
    Fetch S3 checkpoint state.

    Strategy:
      1. If boto3 is available and local credentials work (e.g. EC2 IAM role),
         query S3 directly — fast, no SSH round-trip.
      2. Otherwise SSH to worker-0 and run the proxy script there.

    Returns {worker_idx: {phase_num: {complete, last_completed_date}}, ...}
    """
    # ── Try direct boto3 first ─────────────────────────────────────────────────
    if _BOTO3_AVAILABLE and os.environ.get("S3_VECTOR_BUCKET"):
        try:
            raw = _fetch_s3_state_direct()
            # Normalise: separate __xxx__ blocks from numeric worker keys
            result: dict = {}
            for k, v in raw.items():
                if isinstance(k, str) and k.startswith("__"):
                    result[k] = v
                else:
                    result[int(k)] = {int(p): pv for p, pv in v.items()}
            return result
        except Exception as e:
            print(f"[s3-direct] failed ({e}), falling back to SSH proxy", flush=True)

    # ── SSH proxy fallback ─────────────────────────────────────────────────────
    script = _build_s3_proxy_script()
    cmd = (
        "cd /home/ec2-user/card-oracle-max && "
        "source .venv/bin/activate && "
        "python3 /tmp/_dashboard_s3_probe.py"
    )
    # First upload script, then execute — single SSH call using bash -c with heredoc
    upload_and_run = (
        f"cat > /tmp/_dashboard_s3_probe.py << 'PYEOF'\n"
        f"{script}\n"
        f"PYEOF\n"
        f"cd /home/ec2-user/card-oracle-max && "
        f"source .venv/bin/activate && "
        f"python3 /tmp/_dashboard_s3_probe.py 2>/dev/null"
    )
    try:
        out = subprocess.check_output(
            [
                "ssh", "-i", SSH_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=8",
                "-o", "BatchMode=yes",
                f"ec2-user@{_S3_PROXY_IP}",
                upload_and_run,
            ],
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).decode().strip()
        # Find the JSON line (last line should be the print output)
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                raw = json.loads(line)
                # Separate string-keyed blocks from numeric worker keys
                result: dict = {}
                for w, val in raw.items():
                    if w.startswith("__"):
                        result[w] = val  # __cleanup__, __daily__, __gap_complete__
                    else:
                        result[int(w)] = {int(p): v for p, v in val.items()}
                return result
        print(f"[s3-proxy] no JSON in output: {out[:200]}", flush=True)
        return {}
    except Exception as e:
        print(f"[s3-proxy] error: {e}", flush=True)
        return {}


def phase_completion_key(worker_idx: int, phase_num: int) -> str:
    return f"{S3_CHECKPOINT_PREFIX}/backfill-w{worker_idx}-phase{phase_num}-complete.json"


def phase_checkpoint_key(worker_idx: int, phase_num: int) -> str:
    return f"{S3_CHECKPOINT_PREFIX}/backfill-w{worker_idx}-phase{phase_num}.json"


# ── SSH polling ────────────────────────────────────────────────────────────────

_SSH_LOG_RE = re.compile(
    r"Processed ([\d,]+) rows \| (\d+) rows/s \| skipped (\d+) \| .* ETA ~(-?[\d.]+)min"
)


def poll_worker_ssh(worker_idx: int, ip: str) -> dict:
    """SSH into one worker and scrape the latest journald log line."""
    result: dict[str, Any] = {
        "index":    worker_idx,
        "ip":       WORKER_IPS[worker_idx],  # always show public IP in UI
        "active":   False,
        "rows":     0,
        "rows_s":   0,
        "skipped":  0,
        "eta_min":  None,
        "log_line": "",
        "error":    None,
    }
    try:
        # Build log-file fallback paths for this specific worker index
        cleanup_log = f"/home/ec2-user/card-oracle-max/logs/cleanup-w{worker_idx}.log"
        gaps_log    = f"/home/ec2-user/card-oracle-max/logs/gaps-w{worker_idx}.log"
        out = subprocess.check_output(
            [
                "ssh", "-i", SSH_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=6",
                "-o", "BatchMode=yes",
                f"ec2-user@{ip}",
                # Active check: look for python processes running our scripts.
                # Uses 'ps' + grep rather than pgrep -f to avoid pgrep self-matching
                # (pgrep -f matches the bash process running this very SSH command
                # because the command string itself contains the search terms).
                "ps aux | grep -E '[p]ython.*(worker_phases|worker_cleanup|worker_nonebay|rds_batch_job)' "
                "| grep -qv grep && echo active || echo inactive; "
                # Log: try journald (systemd service) first; fall back to nohup log files
                "LOG=$(sudo journalctl -u backfill --no-pager -n 50 --output=cat 2>/dev/null "
                "| grep 'Processed' | tail -1); "
                f'if [ -z "$LOG" ]; then LOG=$(tail -n 100 "{cleanup_log}" "{gaps_log}" 2>/dev/null '
                "| grep 'Processed' | tail -1); fi; "
                'echo "$LOG"',
            ],
            stderr=subprocess.DEVNULL,
            timeout=12,
        ).decode().strip()

        lines = out.splitlines()
        result["active"] = lines[0].strip() == "active" if lines else False

        log_line = lines[-1] if len(lines) > 1 else ""
        result["log_line"] = log_line

        m = _SSH_LOG_RE.search(log_line)
        if m:
            result["rows"]    = int(m.group(1).replace(",", ""))
            result["rows_s"]  = int(m.group(2))
            result["skipped"] = int(m.group(3))
            eta = float(m.group(4))
            result["eta_min"] = eta if eta >= 0 else None  # negative = past ETA, show as —

    except subprocess.TimeoutExpired:
        result["error"] = "SSH timeout"
    except subprocess.CalledProcessError as e:
        result["error"] = f"SSH error: {e.returncode}"
    except Exception as e:
        result["error"] = str(e)

    return result


def poll_all_workers_ssh() -> list[dict]:
    results = [None] * N_WORKERS
    threads = []

    def _poll(idx: int) -> None:
        # Use private IP when running on EC2 in same VPC, else public IP
        target_ip = SSH_TARGETS[idx]
        results[idx] = poll_worker_ssh(idx, target_ip)

    for idx in range(N_WORKERS):
        t = threading.Thread(target=_poll, args=(idx,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=18)

    return [r or {"index": i, "ip": WORKER_IPS[i], "active": False, "error": "no response"}
            for i, r in enumerate(results)]


# ── S3 checkpoint polling ─────────────────────────────────────────────────────

def poll_s3_state() -> dict:
    """
    Fetches checkpoint state via SSH proxy to worker-0 (IAM instance profile).
    Returns normalised {phase_complete, last_completed_date, cleanup_state} dicts.
    phase_num is 1-indexed throughout (matches worker_phases.py).
    """
    raw = poll_s3_via_ssh()

    phase_complete:      dict[int, dict[int, bool]]       = {i: {} for i in range(N_WORKERS)}
    last_completed_date: dict[int, dict[int, str | None]] = {i: {} for i in range(N_WORKERS)}

    # Extract string-keyed blocks before iterating numeric worker keys
    cleanup_state:  dict[str, dict] = raw.pop("__cleanup__", {})
    daily_complete: dict[str, bool] = raw.pop("__daily__", {})
    gap_complete:   dict[str, dict] = raw.pop("__gap_complete__", {})
    nonebay_state:  dict[str, dict] = raw.pop("__nonebay__", {})

    for w_idx in range(N_WORKERS):
        for phase_num in PHASE_NUMS:
            entry = raw.get(w_idx, {}).get(phase_num, {})
            phase_complete[w_idx][phase_num]      = entry.get("complete", False)
            last_completed_date[w_idx][phase_num] = entry.get("last_completed_date")

    return {
        "phase_complete":      phase_complete,
        "last_completed_date": last_completed_date,
        "cleanup_state":       cleanup_state,
        "daily_complete":      daily_complete,
        "gap_complete":        gap_complete,
        "nonebay_state":       nonebay_state,
    }


# ── Calendar builder ──────────────────────────────────────────────────────────

def build_calendar(s3_state: dict) -> dict[str, str]:
    """
    Returns {date_str: status} for every day from backfill start to today.
    Status: "complete" | "in_progress" | "scheduled" | "not_started" | "out_of_scope"

    Priority order for status resolution (highest wins):
      1. daily-complete marker       — written by daily_update.py cron
      2. gap-fill complete marker    — written by gap_fill_wN.sh
      3. cleanup task progress       — monthly window cleanup workers
      4. phase-based checkpoint      — original worker_phases.py logic
    """
    phase_complete      = s3_state.get("phase_complete", {})
    last_completed_date = s3_state.get("last_completed_date", {})
    cleanup_state       = s3_state.get("cleanup_state", {})
    daily_complete      = s3_state.get("daily_complete", {})   # {YYYY-MM-DD: True}
    gap_complete        = s3_state.get("gap_complete", {})     # {gap-id: {start, end}}

    # Build a set of dates confirmed complete by gap-fill markers
    gap_complete_dates: set[date] = set()
    for _cid, meta in gap_complete.items():
        try:
            gs = date.fromisoformat(meta["start"])
            ge = date.fromisoformat(meta["end"])
            d = gs
            while d < ge:
                gap_complete_dates.add(d)
                d += timedelta(days=1)
        except (KeyError, ValueError):
            pass

    # Calendar runs from backfill start to today (inclusive)
    all_start = min(s for s, _e, _l in PHASES)
    all_end   = date.today() + timedelta(days=1)  # exclusive, so today is included

    calendar: dict[str, str] = {}
    today = date.today()

    d = all_start
    while d < all_end:
        d_str = d.isoformat()

        # ── Layer 4 (base): phase-based checkpoint logic ──────────────────────
        phase_num_for_day: int | None = None
        for p_num, (p_start, p_end, _) in enumerate(PHASES, 1):
            if p_start <= d < p_end:
                phase_num_for_day = p_num
                break

        if phase_num_for_day is None:
            # Date is outside original phase windows (e.g. Apr 8+ covered by daily cron)
            calendar[d_str] = "not_started"
        else:
            worker_for_day: int | None = None
            for w_idx in range(N_WORKERS):
                s, e = WORKER_PHASE_MAP.get((w_idx, phase_num_for_day), (None, None))
                if s and e and s <= d < e:
                    worker_for_day = w_idx
                    break

            if worker_for_day is None:
                calendar[d_str] = "not_started"
            else:
                w_complete = phase_complete.get(worker_for_day, {})
                w_lcd_map  = last_completed_date.get(worker_for_day, {})

                if w_complete.get(phase_num_for_day):
                    calendar[d_str] = "complete"
                else:
                    lcd = w_lcd_map.get(phase_num_for_day)
                    if lcd:
                        try:
                            lcd_date = date.fromisoformat(lcd[:10])
                            if d <= lcd_date:
                                calendar[d_str] = "complete"
                            elif d == lcd_date + timedelta(days=1):
                                calendar[d_str] = "in_progress"
                            else:
                                calendar[d_str] = "scheduled" if phase_num_for_day > 1 else "not_started"
                        except ValueError:
                            calendar[d_str] = "not_started"
                    else:
                        calendar[d_str] = "scheduled" if phase_num_for_day > 1 else "not_started"

        # ── Layer 3: cleanup task progress ────────────────────────────────────
        # Cleanup is a re-processing pass on already-ingested data.
        # Rule: cleanup can UPGRADE a day to "complete" but must NEVER downgrade
        # a day that is already "complete" (e.g. from phase checkpoint).
        # A cleanup task being claimed/in-progress does not mean data is missing.
        cid = _CLEANUP_DATE_MAP.get(d)
        if cid and cleanup_state:
            ct = cleanup_state.get(cid, {})
            has_data = ct.get("complete") or ct.get("claimed") or ct.get("last_completed_date")
            if has_data:
                if ct.get("complete"):
                    # Cleanup fully done — always mark complete
                    calendar[d_str] = "complete"
                elif calendar[d_str] != "complete":
                    # Only show cleanup progress if the day isn't already phase-complete
                    lcd = ct.get("last_completed_date")
                    if lcd:
                        try:
                            lcd_date = date.fromisoformat(lcd[:10])
                            calendar[d_str] = "complete" if d <= lcd_date else "in_progress"
                        except ValueError:
                            pass
                    elif ct.get("claimed"):
                        calendar[d_str] = "in_progress"

        # ── Layer 2: gap-fill completion markers ──────────────────────────────
        # Written by gap_fill_wN.sh after each date range completes.
        if d in gap_complete_dates:
            calendar[d_str] = "complete"

        # ── Layer 1 (highest priority): daily-complete markers ────────────────
        # Written by daily_update.py — authoritative for recent days.
        if daily_complete.get(d_str):
            calendar[d_str] = "complete"

        d += timedelta(days=1)

    return calendar


def build_nonebay_calendar(s3_state: dict) -> dict[str, str]:
    """
    Returns {date_str: status} for every day from NONEBAY_START to today.
    Status is derived purely from the non-eBay task claim/complete markers.

    Status values:
      "complete"     — nonebay task for this day is fully complete
      "in_progress"  — task is claimed but not yet complete
      "not_started"  — task exists but has no markers yet
      "out_of_scope" — date predates the non-eBay backfill window
    """
    nonebay_state = s3_state.get("nonebay_state", {})

    all_start = _NONEBAY_START
    all_end   = date.today() + timedelta(days=1)  # inclusive of today

    calendar: dict[str, str] = {}
    d = all_start
    while d < all_end:
        d_str = d.isoformat()
        cid = _NONEBAY_DATE_MAP.get(d)
        if cid is None:
            calendar[d_str] = "out_of_scope"
        else:
            ct = nonebay_state.get(cid, {})
            if ct.get("complete"):
                calendar[d_str] = "complete"
            elif ct.get("claimed"):
                calendar[d_str] = "in_progress"
            else:
                calendar[d_str] = "not_started"
        d += timedelta(days=1)

    return calendar


# ── Main polling loop ─────────────────────────────────────────────────────────

# Shared state (protected by _state_lock)
_state_lock = threading.Lock()
_state: dict = {
    "workers":           [],
    "calendar":          {},
    "nonebay_calendar":  {},
    "phase_summary":     [],
    "nonebay_summary":   [],
    "total_rows":        0,
    "total_skipped":     0,
    "combined_rps":      0,
    "workers_active":    0,
    "workers_in_phase":  {},
    "updated_at":        None,
    "next_update_at":    None,
    "s3_available":      bool(S3_BUCKET),
}


def _determine_worker_phase(w_idx: int, s3_state: dict) -> tuple[int, str]:
    """Return (phase_num, phase_label) for the worker's currently active phase.
    phase_num is 1-indexed (matches worker_phases.py)."""
    phase_complete = s3_state.get("phase_complete", {}).get(w_idx, {})
    for phase_num, (_, _, label) in enumerate(PHASES, 1):
        if not phase_complete.get(phase_num, False):
            return phase_num, label
    return len(PHASES) + 1, "All complete"


def poll_and_update() -> None:
    """Run one full poll cycle and update shared state."""
    now = datetime.now(timezone.utc)

    # Parallel polls
    ssh_thread  = threading.Thread(target=lambda: _ssh_results.__setitem__(0, poll_all_workers_ssh()), daemon=True)
    s3_thread   = threading.Thread(target=lambda: _s3_results.__setitem__(0, poll_s3_state()), daemon=True)
    _ssh_results: dict[int, list] = {}
    _s3_results:  dict[int, dict] = {}

    ssh_thread.start()
    s3_thread.start()
    ssh_thread.join(timeout=20)
    s3_thread.join(timeout=30)

    workers_raw = _ssh_results.get(0, [{"index": i, "ip": WORKER_IPS[i], "active": False, "error": "poll failed"} for i in range(N_WORKERS)])
    s3_state    = _s3_results.get(0, {"phase_complete": {}, "last_completed_date": {}})

    # Build reverse map: worker_idx → current cleanup CID (claimed-but-not-complete)
    cleanup_state = s3_state.get("cleanup_state", {})
    worker_cleanup_cid: dict[int, str] = {}
    for cid, ct in cleanup_state.items():
        if ct.get("claimed") and not ct.get("complete"):
            cb = ct.get("claimed_by")
            if cb is not None:
                worker_cleanup_cid[int(cb)] = cid

    # Build reverse map: worker_idx → current non-eBay task CID (workers 6–11)
    nonebay_state = s3_state.get("nonebay_state", {})
    worker_nonebay_cid: dict[int, str] = {}
    for cid, ct in nonebay_state.items():
        if ct.get("claimed") and not ct.get("complete"):
            cb = ct.get("claimed_by")
            if cb is not None:
                worker_nonebay_cid[int(cb)] = cid

    # Enrich workers with phase info
    workers = []
    phase_counts: dict[str, int] = {}
    total_rows    = 0
    total_skipped = 0
    combined_rps  = 0
    active_count  = 0

    for w in workers_raw:
        w_idx = w["index"]
        phase_num, phase_label = _determine_worker_phase(w_idx, s3_state)
        w["phase_num"]   = phase_num
        w["phase_label"] = phase_label

        # Attach current cleanup CID if the worker has one claimed
        w["cleanup_cid"] = worker_cleanup_cid.get(w_idx)

        # Attach current non-eBay task CID (workers 6–11 running worker_nonebay.py)
        w["nonebay_cid"] = worker_nonebay_cid.get(w_idx)

        # Clamp phase_num to valid range for lookup (1–len(PHASES))
        lookup_phase = min(phase_num, len(PHASES))
        slice_start, slice_end = WORKER_PHASE_MAP.get((w_idx, lookup_phase), (None, None))

        # If worker is on a cleanup task, use that task's date range for the slice display
        if w["cleanup_cid"]:
            for cs, ce, ccid in CLEANUP_TASKS:
                if ccid == w["cleanup_cid"]:
                    slice_start, slice_end = cs, ce
                    break

        # If worker is on a non-eBay task, use that task's date range
        if w["nonebay_cid"]:
            for ns, ne, ncid in NONEBAY_TASKS:
                if ncid == w["nonebay_cid"]:
                    slice_start, slice_end = ns, ne
                    break

        w["slice_start"] = slice_start.isoformat() if slice_start else "—"
        w["slice_end"]   = slice_end.isoformat()   if slice_end   else "—"

        workers.append(w)

        phase_counts[phase_label] = phase_counts.get(phase_label, 0) + 1
        total_rows    += w.get("rows", 0)
        total_skipped += w.get("skipped", 0)
        combined_rps  += w.get("rows_s", 0)
        if w.get("active"):
            active_count += 1

    # Calendars
    calendar         = build_calendar(s3_state)
    nonebay_calendar = build_nonebay_calendar(s3_state)

    # Phase summary (phase_num is 1-indexed)
    phase_summary = []
    for phase_num, (p_start, p_end, label) in enumerate(PHASES, 1):
        complete_workers = sum(
            1 for w_idx in range(N_WORKERS)
            if s3_state.get("phase_complete", {}).get(w_idx, {}).get(phase_num, False)
        )
        total_days   = (p_end - p_start).days
        complete_days = sum(
            1 for d_str, st in calendar.items()
            if st == "complete"
            and p_start <= date.fromisoformat(d_str) < p_end
        )
        phase_summary.append({
            "label":           label,
            "start":           p_start.isoformat(),
            "end":             p_end.isoformat(),
            "total_days":      total_days,
            "complete_days":   complete_days,
            "complete_workers": complete_workers,
            "pct":             round(complete_days / total_days * 100, 1) if total_days else 0,
        })

    # Non-eBay monthly summary (workers 6–11)
    nonebay_summary = []
    for win_start, win_end in _nonebay_monthly_windows():
        month_tag = win_start.strftime("%Y%m")
        month_tasks = [(s, e, cid) for s, e, cid in NONEBAY_TASKS
                       if cid.startswith(f"nonebay-m{month_tag}")]
        total_t    = len(month_tasks)
        complete_t = sum(1 for _s, _e, cid in month_tasks
                         if nonebay_state.get(cid, {}).get("complete"))
        claimed_t  = sum(1 for _s, _e, cid in month_tasks
                         if nonebay_state.get(cid, {}).get("claimed")
                         and not nonebay_state.get(cid, {}).get("complete"))
        nonebay_summary.append({
            "label":          win_start.strftime("%b %Y"),
            "start":          str(win_start),
            "end":            str(win_end),
            "total_tasks":    total_t,
            "complete_tasks": complete_t,
            "in_progress":    claimed_t,
            "pct":            round(complete_t / total_t * 100, 0) if total_t else 0,
        })

    with _state_lock:
        _state["workers"]          = workers
        _state["calendar"]         = calendar
        _state["nonebay_calendar"] = nonebay_calendar
        _state["phase_summary"]    = phase_summary
        _state["nonebay_summary"]  = nonebay_summary
        _state["total_rows"]       = total_rows
        _state["total_skipped"]    = total_skipped
        _state["combined_rps"]     = combined_rps
        _state["workers_active"]   = active_count
        _state["workers_in_phase"] = phase_counts
        _state["updated_at"]       = now.isoformat()
        _state["next_update_at"]   = (now + timedelta(seconds=POLL_INTERVAL)).isoformat()
        _state["s3_available"]     = bool(S3_BUCKET)


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
<title>Backfill Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #222636;
    --border: #2d3148; --text: #e2e8f0; --muted: #8892b0;
    --green: #10b981; --yellow: #f59e0b; --blue: #3b82f6;
    --red: #ef4444; --purple: #8b5cf6; --gray: #374151;
    --green-dim: #064e3b; --yellow-dim: #451a03; --blue-dim: #1e3a5f;
    --gray-dim: #1f2937; --scheduled-dim: #312e81;
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
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .card .value { font-size: 22px; font-weight: 700; }
  .card .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .card.green .value { color: var(--green); }
  .card.yellow .value { color: var(--yellow); }
  .card.blue .value { color: var(--blue); }
  .card.purple .value { color: var(--purple); }

  /* Phase progress */
  .section-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); margin-bottom: 10px; }
  .phase-grid { display: grid; gap: 8px; }
  .phase-row { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; display: grid; grid-template-columns: 100px 1fr 60px; gap: 12px; align-items: center; }
  .phase-label { font-weight: 600; font-size: 12px; }
  .progress-bar { background: var(--gray-dim); border-radius: 4px; height: 8px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; background: var(--green); }
  .progress-fill.partial { background: linear-gradient(90deg, var(--green), var(--yellow)); }
  .progress-fill.zero { background: var(--gray); }
  .phase-pct { font-size: 12px; font-weight: 700; text-align: right; color: var(--green); }
  .phase-pct.zero { color: var(--muted); }

  /* Worker table */
  .worker-table { width: 100%; border-collapse: collapse; }
  .worker-table th { background: var(--surface2); padding: 8px 10px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); border-bottom: 1px solid var(--border); }
  .worker-table td { padding: 7px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  .worker-table tr:last-child td { border-bottom: none; }
  .worker-table tr:hover td { background: var(--surface2); }
  .badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 20px; font-size: 10px; font-weight: 700; }
  .badge.active { background: #064e3b; color: var(--green); }
  .badge.inactive { background: #450a0a; color: var(--red); }
  .badge.phase { background: var(--blue-dim); color: #93c5fd; }
  .badge.phase2 { background: var(--scheduled-dim); color: #c4b5fd; }
  .badge.done { background: var(--green-dim); color: var(--green); }
  .rps { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--yellow); }
  .rows { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
  .eta { color: var(--muted); font-size: 11px; }
  .ip { font-family: monospace; font-size: 11px; color: var(--muted); }
  .err { color: var(--red); font-size: 11px; }
  .slice-range { font-size: 10px; color: var(--muted); font-family: monospace; }

  /* Calendar */
  .calendar-section { overflow-x: auto; }
  .calendar-grid { display: flex; flex-wrap: wrap; gap: 16px; }
  .month-block { min-width: 200px; }
  .month-name { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 6px; }
  .days-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; }
  .day-label { font-size: 9px; color: var(--muted); text-align: center; padding: 1px; }
  .day-cell { width: 100%; padding-bottom: 100%; border-radius: 3px; position: relative; cursor: default; }
  .day-cell:hover::after { content: attr(data-date); position: absolute; bottom: calc(100% + 4px); left: 50%; transform: translateX(-50%); background: #000; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 10px; white-space: nowrap; z-index: 10; pointer-events: none; }
  .day-cell.complete     { background: var(--green); }
  .day-cell.in_progress  { background: var(--yellow); animation: pulse 1.5s infinite; }
  .day-cell.not_started  { background: var(--gray-dim); }
  .day-cell.scheduled    { background: var(--scheduled-dim); opacity: 0.7; }
  .day-cell.out_of_scope { background: transparent; }
  .day-cell.empty        { background: transparent; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

  /* Legend */
  .legend { display: flex; gap: 16px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); }
  .legend-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
  .legend-dot.complete    { background: var(--green); }
  .legend-dot.in_progress { background: var(--yellow); }
  .legend-dot.scheduled   { background: var(--scheduled-dim); }
  .legend-dot.not_started { background: var(--gray-dim); }

  .section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }

  .loading { color: var(--muted); font-style: italic; }
  .error-banner { background: #450a0a; border: 1px solid var(--red); color: var(--red); border-radius: 8px; padding: 10px 14px; font-size: 12px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Card Oracle — <span>Backfill Dashboard</span></h1>
    <div style="color:var(--muted);font-size:11px;margin-top:2px">Qdrant vector store population · 12 × g4dn.xlarge · us-west-1</div>
  </div>
  <div class="header-meta">
    <div>Updated: <span id="updated-at">—</span></div>
    <div>Next refresh: <span class="countdown" id="countdown">—</span></div>
  </div>
</header>

<div class="main">

  <!-- Summary cards -->
  <div class="summary-grid" id="summary-cards">
    <div class="card blue"><div class="label">Active Workers</div><div class="value" id="s-active">—</div><div class="sub">of 12 total</div></div>
    <div class="card yellow"><div class="label">Combined Throughput</div><div class="value" id="s-rps">—</div><div class="sub">rows / second</div></div>
    <div class="card green"><div class="label">Total Rows Loaded</div><div class="value" id="s-rows">—</div><div class="sub">into Qdrant</div></div>
    <div class="card purple"><div class="label">Rows/Day (est.)</div><div class="value" id="s-daily">—</div><div class="sub">at current rate</div></div>
    <div class="card"><div class="label">Images Skipped</div><div class="value" id="s-skipped" style="color:var(--red)">—</div><div class="sub">404 / timeout</div></div>
    <div class="card"><div class="label">S3 Checkpoints</div><div class="value" id="s-s3" style="font-size:14px">—</div><div class="sub">availability</div></div>
  </div>

  <!-- Phase progress + worker table -->
  <div class="two-col">

    <div class="section">
      <div class="section-title">eBay Phase Progress (w0–w5)</div>
      <div class="phase-grid" id="phase-grid"><div class="loading">Loading…</div></div>
      <div style="margin-top:14px">
        <div class="section-title">Non-eBay Backfill Progress (w6–w11)</div>
        <div class="phase-grid" id="nonebay-grid"><div class="loading">Loading…</div></div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Worker Fleet</div>
      <div style="overflow-x:auto">
        <table class="worker-table">
          <thead><tr>
            <th>#</th><th>Status</th><th>Task</th><th>Slice</th><th>Rows</th><th>r/s</th><th>ETA</th>
          </tr></thead>
          <tbody id="worker-tbody"><tr><td colspan="7" class="loading">Loading…</td></tr></tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- eBay calendar -->
  <div class="section calendar-section">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px">
      <div>
        <div class="section-title" style="margin:0">eBay Vector Coverage</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px">Jan 2025 – present · workers w0–w5</div>
      </div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot complete"></div>Complete</div>
        <div class="legend-item"><div class="legend-dot in_progress"></div>In progress</div>
        <div class="legend-item"><div class="legend-dot scheduled"></div>Scheduled</div>
        <div class="legend-item"><div class="legend-dot not_started"></div>Not started</div>
      </div>
    </div>
    <div class="calendar-grid" id="calendar-grid"><div class="loading">Loading…</div></div>
  </div>

  <!-- Non-eBay calendar -->
  <div class="section calendar-section">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px">
      <div>
        <div class="section-title" style="margin:0">Non-eBay Vector Coverage</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px">Jan 2025 – present · workers w6–w11 · Fanatics/PWCC · Pristine · Goldin · MySlabs · Heritage</div>
      </div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot complete"></div>Complete</div>
        <div class="legend-item"><div class="legend-dot in_progress"></div>In progress</div>
        <div class="legend-item"><div class="legend-dot not_started"></div>Not started</div>
      </div>
    </div>
    <div class="calendar-grid" id="nonebay-calendar-grid"><div class="loading">Loading…</div></div>
  </div>

</div>

<script>
const API_URL = '/api/status';
let countdownTimer = null;
let nextUpdateTime = null;

function fmt(n) {
  if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toString();
}

function fmtEta(min) {
  if (min === null || min === undefined) return '—';
  if (min < 60)   return Math.round(min) + 'm';
  if (min < 1440) return (min/60).toFixed(1) + 'h';
  return (min/1440).toFixed(1) + 'd';
}

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

function phaseColor(phase_num) {
  return ['badge phase','badge phase2','badge phase','badge phase2','badge phase'][phase_num] || 'badge';
}

function renderSummary(data) {
  document.getElementById('s-active').textContent  = data.workers_active;
  document.getElementById('s-rps').textContent     = data.combined_rps;
  document.getElementById('s-rows').textContent    = fmt(data.total_rows);
  document.getElementById('s-daily').textContent   = fmt(data.combined_rps * 86400);
  document.getElementById('s-skipped').textContent = fmt(data.total_skipped);
  document.getElementById('s-s3').textContent      = data.s3_available ? '✓ Connected' : '✗ No bucket';
  document.getElementById('updated-at').textContent = fmtTime(data.updated_at);
}

function renderPhaseGrid(gridId, rows, workerCount) {
  const el = document.getElementById(gridId);
  if (!rows || rows.length === 0) { el.innerHTML = '<div style="color:var(--muted);font-size:11px">No data</div>'; return; }
  el.innerHTML = rows.map(p => {
    const pct  = p.pct;
    const cls  = pct === 0 ? 'zero' : pct < 100 ? 'partial' : '';
    const pcls = pct === 0 ? 'zero' : '';
    // For non-eBay rows use task count instead of days
    const subtitle = p.total_tasks !== undefined
      ? `${p.complete_tasks}/${p.total_tasks} tasks${p.in_progress > 0 ? ' · ' + p.in_progress + ' active' : ''}`
      : `${p.complete_days}/${p.total_days} days · ${p.complete_workers}/${workerCount} workers done`;
    return `<div class="phase-row">
      <div class="phase-label">${p.label}</div>
      <div>
        <div class="progress-bar"><div class="progress-fill ${cls}" style="width:${pct}%"></div></div>
        <div style="font-size:10px;color:var(--muted);margin-top:3px">${subtitle}</div>
      </div>
      <div class="phase-pct ${pcls}">${pct}%</div>
    </div>`;
  }).join('');
}

function renderPhases(phases) {
  renderPhaseGrid('phase-grid', phases, 12);
}

function renderWorkers(workers) {
  const tbody = document.getElementById('worker-tbody');
  tbody.innerHTML = workers.map(w => {
    const statusBadge = w.active
      ? '<span class="badge active">● active</span>'
      : '<span class="badge inactive">○ down</span>';

    const isNonEbayWorker = w.index >= 6;  // workers 6–11 run worker_nonebay.py
    const allOrigDone = w.phase_num > 6;   // all 6 original phases complete
    const onCleanup   = !isNonEbayWorker && allOrigDone && w.cleanup_cid;
    const onNonEbay   = isNonEbayWorker;

    let taskBadge;
    if (onNonEbay) {
      if (w.nonebay_cid) {
        taskBadge = `<span class="badge active">✦ non-eBay · ${w.nonebay_cid}</span>`;
      } else if (w.active) {
        taskBadge = '<span class="badge active">✦ non-eBay</span>';
      } else {
        taskBadge = '<span class="badge phase">✦ non-eBay</span>';
      }
    } else if (!isNonEbayWorker && allOrigDone && !w.cleanup_cid && !w.active) {
      taskBadge = '<span class="badge done">✓ eBay done</span>';
    } else if (onCleanup) {
      taskBadge = `<span class="badge active">↻ cleanup · ${w.cleanup_cid}</span>`;
    } else if (!isNonEbayWorker && allOrigDone && w.active) {
      taskBadge = '<span class="badge active">↻ eBay cleanup</span>';
    } else {
      taskBadge = `<span class="${phaseColor(w.phase_num)}">P${w.phase_num} · ${w.phase_label}</span>`;
    }

    const rowsCell  = `<span class="rows">${fmt(w.rows || 0)}</span>`;
    const rpsCell   = w.rows_s ? `<span class="rps">${w.rows_s}</span>` : '<span class="muted">—</span>';
    const etaCell   = `<span class="eta">${fmtEta(w.eta_min)}</span>`;
    const sliceCell = `<span class="slice-range">${w.slice_start}→${w.slice_end}</span>`;

    return `<tr>
      <td><b>w${w.index}</b><br><span class="ip">${w.ip}</span></td>
      <td>${statusBadge}${w.error ? `<div class="err">${w.error}</div>` : ''}</td>
      <td>${taskBadge}</td>
      <td>${sliceCell}</td>
      <td>${rowsCell}</td>
      <td>${rpsCell}</td>
      <td>${etaCell}</td>
    </tr>`;
  }).join('');
}

function renderCalendar(calendar, gridId) {
  gridId = gridId || 'calendar-grid';
  const months = {};
  for (const [d, status] of Object.entries(calendar)) {
    const [y, m] = d.split('-');
    const key = `${y}-${m}`;
    if (!months[key]) months[key] = {};
    months[key][d] = status;
  }

  const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const DAY_NAMES   = ['Mo','Tu','We','Th','Fr','Sa','Su'];

  const sortedMonths = Object.keys(months).sort().reverse(); // newest first
  const grid = document.getElementById(gridId);

  grid.innerHTML = sortedMonths.map(key => {
    const [y, m] = key.split('-').map(Number);
    const label  = `${MONTH_NAMES[m-1]} ${y}`;
    const dayMap = months[key];

    // Day of week for 1st of month (Mon=0)
    const firstDow = (new Date(y, m-1, 1).getDay() + 6) % 7;
    const daysInMonth = new Date(y, m, 0).getDate();

    // Header row
    const headerCells = DAY_NAMES.map(d => `<div class="day-label">${d}</div>`).join('');

    // Empty cells before first day
    const emptyCells = Array(firstDow).fill('<div class="day-cell empty"></div>').join('');

    // Day cells
    const dayCells = [];
    for (let d = 1; d <= daysInMonth; d++) {
      const dateStr = `${y}-${String(m).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
      const status  = dayMap[dateStr] || 'out_of_scope';
      dayCells.push(`<div class="day-cell ${status}" data-date="${dateStr} (${status})"></div>`);
    }

    return `<div class="month-block">
      <div class="month-name">${label}</div>
      <div class="days-grid">
        ${headerCells}
        ${emptyCells}
        ${dayCells.join('')}
      </div>
    </div>`;
  }).join('');
}

function startCountdown() {
  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = setInterval(() => {
    if (!nextUpdateTime) return;
    const secs = Math.max(0, Math.round((nextUpdateTime - Date.now()) / 1000));
    document.getElementById('countdown').textContent = secs + 's';
  }, 1000);
}

async function fetchAndRender() {
  try {
    const resp = await fetch(API_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    nextUpdateTime = data.next_update_at ? new Date(data.next_update_at).getTime() : Date.now() + 60000;

    renderSummary(data);
    renderPhases(data.phase_summary || []);
    renderPhaseGrid('nonebay-grid', data.nonebay_summary || [], 6);
    renderWorkers(data.workers || []);
    renderCalendar(data.calendar || {}, 'calendar-grid');
    renderCalendar(data.nonebay_calendar || {}, 'nonebay-calendar-grid');
  } catch(e) {
    console.error('Fetch error:', e);
  }
}

// Initial load + polling
fetchAndRender();
startCountdown();
setInterval(fetchAndRender, 60000);
</script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server — each request in its own thread."""
    daemon_threads    = True   # threads die with the process
    allow_reuse_address = True


class DashboardHandler(BaseHTTPRequestHandler):
    timeout = 10  # seconds to wait for request headers before dropping connection

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
    parser = argparse.ArgumentParser(description="Backfill dashboard server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll and print JSON (no HTTP server)",
    )
    args = parser.parse_args()

    if args.once:
        poll_and_update()
        with _state_lock:
            print(json.dumps(_state, default=str, indent=2))
        return

    print(f"[dashboard] Doing initial poll (this may take ~15 seconds)…", flush=True)
    poll_and_update()
    print(f"[dashboard] Initial poll complete.", flush=True)

    poller = threading.Thread(target=_background_poller, daemon=True)
    poller.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"[dashboard] Serving at http://localhost:{args.port}", flush=True)
    if not S3_BUCKET:
        print("[dashboard] WARNING: S3_VECTOR_BUCKET not set — calendar will use SSH data only", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Shutting down.", flush=True)


if __name__ == "__main__":
    main()
