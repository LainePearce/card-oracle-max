#!/usr/bin/env python3
"""
Image-Archive Dashboard — 2026 image-download status.

The vector-backfill dashboard (backfill_dashboard.py) tracks Qdrant vector jobs.
This one tracks the image-archive fleet instead: per-day download status, images
archived vs OS docs, storage accumulated, and which workers are running the
image-archive service. Serves a live page at http://localhost:8082.

    python tools/image_archive_dashboard.py [--port 8082] [--once]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from tools.image_archive_common import (
    s3_client, QUEUE_BUCKET, IMAGE_BUCKET, QUEUE, ACTIVE, COMPLETE, list_dates,
)
# Reuse the fleet IP config + SSH gating from the vector dashboard.
from tools.backfill_dashboard import (
    WORKER_IPS, SSH_TARGETS, N_WORKERS, SSH_KEY, SSH_AVAILABLE,
)

POLL_INTERVAL = 60
CALENDAR_START = date(2026, 1, 1)


def _is_date(s: str) -> bool:
    """True only for YYYY-MM-DD marker names. Non-eBay index names (2025-gold,
    2026-04-pwcc) are excluded from the date calendar — they'd break the
    frontend's date parsing and hang the coverage grid on 'Loading'."""
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False
SERVICE = "image-archive"


def _read_marker(s3, d: str) -> tuple[str, dict]:
    try:
        body = s3.get_object(Bucket=QUEUE_BUCKET, Key=f"{COMPLETE}/{d}.json")["Body"].read()
        return d, json.loads(body)
    except Exception:
        return d, {}


def _read_active(s3, d: str) -> tuple[str, dict]:
    try:
        body = s3.get_object(Bucket=QUEUE_BUCKET, Key=f"{ACTIVE}/{d}.json")["Body"].read()
        return d, json.loads(body)
    except Exception:
        return d, {}


def poll_s3_state() -> dict:
    s3 = s3_client()
    queued   = list_dates(s3, QUEUE)
    active   = list_dates(s3, ACTIVE)
    complete = list_dates(s3, COMPLETE)

    markers: dict[str, dict] = {}
    if complete:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for d, m in ex.map(lambda d: _read_marker(s3, d), complete):
                markers[d] = m

    active_jobs: list[dict] = []
    if active:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for d, m in ex.map(lambda d: _read_active(s3, d), active):
                m["date"] = d
                active_jobs.append(m)
    active_jobs.sort(key=lambda j: j["date"], reverse=True)

    total_archived = sum(m.get("archived", 0) for m in markers.values())
    total_os       = sum(m.get("os_total", 0) for m in markers.values())
    total_bytes    = sum(m.get("bytes", 0) for m in markers.values())

    # Only YYYY-MM-DD markers go in the date calendar; non-eBay index-name
    # markers (2025-gold, 2026-04-pwcc) still count in totals + the Active table.
    coverage: dict[str, str] = {}
    per_day: dict[str, dict] = {}
    for d in complete:
        if not _is_date(d):
            continue
        coverage[d] = "complete"
        per_day[d] = {"archived": markers.get(d, {}).get("archived", 0),
                      "os_total": markers.get(d, {}).get("os_total", 0)}
    for d in active:
        if _is_date(d):
            coverage[d] = "active"
    for d in queued:
        if _is_date(d):
            coverage.setdefault(d, "queued")

    today = date.today()
    dd = CALENDAR_START
    while dd <= today:
        coverage.setdefault(dd.isoformat(), "not_in_queue")
        dd += timedelta(days=1)

    # Non-eBay marketplace indices (non-date marker names) — one row per index
    # with its pipeline status + counts, for the backfill table.
    active_by_name = {j["date"]: j for j in active_jobs}
    rank = {"active": 0, "queued": 1, "complete": 2}
    nonebay: list[dict] = []
    for n in sorted(set(queued + active + complete)):
        if _is_date(n):
            continue
        if n in active:
            j = active_by_name.get(n, {})
            nonebay.append({"index": n, "status": "active",
                            "archived": j.get("archived", 0), "os_total": j.get("os_total", 0),
                            "failed": j.get("failed", 0), "host": j.get("host", ""),
                            "updated_at": j.get("updated_at") or j.get("claimed_at")})
        elif n in complete:
            m = markers.get(n, {})
            nonebay.append({"index": n, "status": "complete",
                            "archived": m.get("archived", 0), "os_total": m.get("os_total", 0),
                            "failed": m.get("failed", 0)})
        else:
            nonebay.append({"index": n, "status": "queued"})
    nonebay.sort(key=lambda r: (rank.get(r["status"], 3), r["index"]))

    return {
        "days_complete": len(complete),
        "days_active":   len(active),
        "days_queued":   len(queued),
        "total_archived": total_archived,
        "total_os":       total_os,
        "total_gb":       round(total_bytes / (1024 ** 3), 1),
        "coverage":       coverage,
        "per_day":        per_day,
        "active_jobs":    active_jobs,
        "nonebay":        nonebay,
    }


def poll_worker(idx: int, ip: str) -> dict:
    res = {"index": idx, "ip": WORKER_IPS[idx], "service": "unknown", "log_line": "", "error": None}
    try:
        out = subprocess.check_output(
            ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=6", "-o", "BatchMode=yes", f"ec2-user@{ip}",
             f"sudo systemctl is-active {SERVICE} 2>/dev/null || echo inactive; "
             f"sudo journalctl -u {SERVICE} --no-pager -n 5 --output=cat 2>/dev/null | tail -1"],
            stderr=subprocess.DEVNULL, timeout=12).decode().strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            res["service"] = lines[0]
        if len(lines) > 1:
            res["log_line"] = lines[-1]
    except subprocess.TimeoutExpired:
        res["error"] = "SSH timeout"
    except Exception as e:
        res["error"] = str(e)
    return res


def poll_workers() -> list[dict]:
    if not SSH_AVAILABLE:
        return [{"index": i, "ip": WORKER_IPS[i], "service": "—", "log_line": "",
                 "error": "ssh key not on host (cosmetic)"} for i in range(N_WORKERS)]
    results: list[dict | None] = [None] * N_WORKERS
    threads = []
    for i in range(N_WORKERS):
        t = threading.Thread(target=lambda i=i: results.__setitem__(i, poll_worker(i, SSH_TARGETS[i])),
                             daemon=True)
        t.start(); threads.append(t)
    for t in threads:
        t.join(timeout=18)
    return [r or {"index": i, "ip": WORKER_IPS[i], "service": "unknown", "error": "no response"}
            for i, r in enumerate(results)]


_state_lock = threading.Lock()
_state: dict = {"updated_at": None, "next_update_at": None}
_prev = {"complete": None, "t": None}


def _eta(days_complete: int, days_remaining: int) -> str | None:
    now = time.monotonic()
    if _prev["complete"] is not None and _prev["t"] is not None:
        dc = days_complete - _prev["complete"]
        dt = now - _prev["t"]
        if dt > 0 and dc > 0:
            secs = days_remaining / (dc / dt)
            h = secs / 3600
            return f"~{int(secs/60)}m" if h < 1 else f"~{h:.1f}h" if h < 24 else f"~{h/24:.1f}d"
    _prev["complete"], _prev["t"] = days_complete, now
    return None


def poll_and_update() -> None:
    now = datetime.now(timezone.utc)
    s3r: dict = {}
    wr: dict = {}
    t1 = threading.Thread(target=lambda: s3r.update(poll_s3_state()), daemon=True)
    t2 = threading.Thread(target=lambda: wr.update({"w": poll_workers()}), daemon=True)
    t1.start(); t2.start(); t1.join(timeout=45); t2.join(timeout=20)

    remaining = s3r.get("days_queued", 0) + s3r.get("days_active", 0)
    with _state_lock:
        _state.update(s3r)
        _state["workers"] = wr.get("w", [])
        _state["eta"] = _eta(s3r.get("days_complete", 0), remaining)
        _state["updated_at"] = now.isoformat()
        _state["next_update_at"] = (now + timedelta(seconds=POLL_INTERVAL)).isoformat()


def _poller() -> None:
    while True:
        try:
            poll_and_update()
        except Exception as e:
            print(f"[poller] {e}", flush=True)
        time.sleep(POLL_INTERVAL)


HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image Archive Dashboard</title><style>
:root{--bg:#0f1117;--surface:#1a1d27;--surface2:#222636;--border:#2d3148;--text:#e2e8f0;--muted:#8892b0;--green:#10b981;--yellow:#f59e0b;--blue:#3b82f6;--red:#ef4444;--gray-dim:#1f2937;--queued-dim:#312e81;}
*{box-sizing:border-box;margin:0;padding:0}body{font-family:Inter,-apple-system,sans-serif;background:var(--bg);color:var(--text);font-size:13px}
header{padding:16px 24px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
header h1{font-size:18px}header h1 span{color:var(--blue)}.header-meta{font-size:11px;color:var(--muted);text-align:right;line-height:1.6}
.main{padding:20px 24px;display:grid;gap:20px}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.card .label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.card .value{font-size:22px;font-weight:700}.card .sub{font-size:11px;color:var(--muted);margin-top:2px}
.card.green .value{color:var(--green)}.card.yellow .value{color:var(--yellow)}.card.blue .value{color:var(--blue)}
.section{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px}
.section-title{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:10px}
.worker-table{width:100%;border-collapse:collapse;font-size:12px}
.worker-table th{background:var(--surface2);padding:7px 9px;text-align:left;font-size:10px;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border)}
.worker-table td{padding:6px 9px;border-bottom:1px solid var(--border)}
.svc-active{color:var(--green);font-weight:700}.svc-inactive{color:var(--red)}.svc-unknown{color:var(--muted);font-style:italic}
.ip{font-family:monospace;font-size:11px;color:var(--muted)}.log-preview{font-size:10px;color:var(--muted);max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:monospace}
.calendar-grid{display:flex;flex-wrap:wrap;gap:14px}.month-block{min-width:196px}
.month-name{font-size:11px;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:5px}
.days-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:2px}.day-label{font-size:9px;color:var(--muted);text-align:center}
.day-cell{width:100%;padding-bottom:100%;border-radius:3px;position:relative}
.day-cell:hover::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 4px);left:50%;transform:translateX(-50%);background:#000;color:#fff;padding:2px 6px;border-radius:4px;font-size:10px;white-space:nowrap;z-index:10}
.day-cell.complete{background:var(--green)}.day-cell.active{background:var(--yellow);animation:pulse 1.5s infinite}
.day-cell.queued{background:var(--queued-dim);opacity:.85}.day-cell.not_in_queue{background:var(--gray-dim);opacity:.4}.day-cell.empty{background:transparent}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.legend{display:flex;gap:14px;flex-wrap:wrap}.legend-item{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted)}
.legend-dot{width:10px;height:10px;border-radius:2px}.legend-dot.complete{background:var(--green)}.legend-dot.active{background:var(--yellow)}.legend-dot.queued{background:var(--queued-dim)}.legend-dot.not_in_queue{background:var(--gray-dim);opacity:.5}
.loading{color:var(--muted);font-style:italic}
</style></head><body>
<header><div><h1>Card Oracle — <span>Image Archive Dashboard</span></h1>
<div style="color:var(--muted);font-size:11px;margin-top:2px">2026 image downloads → s3://images-130-sold · image-archive queue · 12 × EC2 workers</div></div>
<div class="header-meta"><div>Updated: <span id="updated-at">—</span></div><div>Next: <span id="countdown" style="color:var(--yellow);font-weight:600">—</span></div></div></header>
<div class="main">
<div class="summary-grid">
<div class="card green"><div class="label">Days Complete</div><div class="value" id="s-complete">—</div><div class="sub">days downloaded</div></div>
<div class="card yellow"><div class="label">Days Active</div><div class="value" id="s-active">—</div><div class="sub">in progress</div></div>
<div class="card blue"><div class="label">Days Queued</div><div class="value" id="s-queued">—</div><div class="sub">pending</div></div>
<div class="card green"><div class="label">Images Archived</div><div class="value" id="s-images">—</div><div class="sub">of OS docs</div></div>
<div class="card blue"><div class="label">Storage</div><div class="value" id="s-gb">—</div><div class="sub">GB in S3</div></div>
<div class="card yellow"><div class="label">ETA</div><div class="value" id="s-eta" style="font-size:18px">—</div><div class="sub">all 2026 days</div></div>
</div>
<div class="section"><div class="section-title">Active Days — downloading now</div>
<div style="overflow-x:auto"><table class="worker-table"><thead><tr><th>Date</th><th>Worker</th><th>Archived</th><th>OS docs</th><th>Progress</th><th>Updated</th></tr></thead>
<tbody id="active-jobs"><tr><td colspan="6" class="loading">Loading…</td></tr></tbody></table></div></div>
<div class="section"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px">
<div class="section-title" style="margin:0">2026 Image-Download Coverage</div>
<div class="legend"><div class="legend-item"><div class="legend-dot complete"></div>Complete</div><div class="legend-item"><div class="legend-dot active"></div>Active</div><div class="legend-item"><div class="legend-dot queued"></div>Queued</div><div class="legend-item"><div class="legend-dot not_in_queue"></div>Not queued</div></div></div>
<div class="calendar-grid" id="calendar"><div class="loading">Loading…</div></div></div>
<div class="section"><div class="section-title">Non-eBay Marketplace Backfill — per index</div>
<div style="overflow-x:auto"><table class="worker-table"><thead><tr><th>Index</th><th>Status</th><th>Archived</th><th>OS docs</th><th>Coverage</th><th>Failed</th><th>Updated</th></tr></thead>
<tbody id="nonebay"><tr><td colspan="7" class="loading">Loading…</td></tr></tbody></table></div></div>
<div class="section"><div class="section-title">Worker Fleet — image-archive service</div>
<div style="overflow-x:auto"><table class="worker-table"><thead><tr><th>#</th><th>IP</th><th>Service</th><th>Last log line</th></tr></thead>
<tbody id="workers"><tr><td colspan="4" class="loading">Loading…</td></tr></tbody></table></div></div>
</div>
<script>
let nextT=null;
function fmt(n){if(n==null)return '—';n=+n;if(n>=1e6)return (n/1e6).toFixed(2)+'M';if(n>=1e3)return (n/1e3).toFixed(1)+'K';return Math.round(n).toString();}
function fmtTime(iso){if(!iso)return '—';return new Date(iso).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});}
const MN=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],DN=['Mo','Tu','We','Th','Fr','Sa','Su'];
function renderSummary(d){
  document.getElementById('s-complete').textContent=d.days_complete??'—';
  document.getElementById('s-active').textContent=d.days_active??'—';
  document.getElementById('s-queued').textContent=d.days_queued??'—';
  document.getElementById('s-images').textContent=fmt(d.total_archived);
  document.getElementById('s-gb').textContent=(d.total_gb??0)+' GB';
  document.getElementById('s-eta').textContent=d.eta||'—';
  document.getElementById('updated-at').textContent=fmtTime(d.updated_at);
}
function renderCal(cov,perday){
  const grid=document.getElementById('calendar');
  if(!cov||!Object.keys(cov).length){grid.innerHTML='<div class="loading">No data</div>';return;}
  const months={};for(const [d,s] of Object.entries(cov)){const k=d.slice(0,7);(months[k]=months[k]||{})[d]=s;}
  const keys=Object.keys(months).sort().reverse();
  grid.innerHTML=keys.map(k=>{const [y,m]=k.split('-').map(Number);const dm=months[k];
    const fdow=(new Date(y,m-1,1).getDay()+6)%7;const dim=new Date(y,m,0).getDate();
    const hc=DN.map(d=>`<div class="day-label">${d}</div>`).join('');
    const ec=Array(fdow).fill('<div class="day-cell empty"></div>').join('');
    const cells=[];for(let day=1;day<=dim;day++){const ds=`${y}-${String(m).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
      const s=dm[ds]||'not_in_queue';const pd=perday&&perday[ds];
      const tip=pd?`${ds} · ${fmt(pd.archived)}/${fmt(pd.os_total)}`:`${ds} · ${s}`;
      cells.push(`<div class="day-cell ${s}" data-tip="${tip}"></div>`);}
    return `<div class="month-block"><div class="month-name">${MN[m-1]} ${y}</div><div class="days-grid">${hc}${ec}${cells.join('')}</div></div>`;
  }).join('');
}
function renderActive(jobs){const tb=document.getElementById('active-jobs');
  if(!jobs||!jobs.length){tb.innerHTML='<tr><td colspan="6" class="svc-unknown" style="text-align:center;padding:14px">No active days</td></tr>';return;}
  tb.innerHTML=jobs.map(j=>{const a=j.archived||0,t=j.os_total||0;const pct=t?Math.min(100,a/t*100):0;
    const bar=`<div style="display:flex;align-items:center;gap:6px"><div style="background:var(--gray-dim);border-radius:3px;height:7px;width:90px;overflow:hidden"><div style="height:100%;background:var(--green);width:${pct}%"></div></div><span style="font-size:10px;color:var(--muted)">${pct.toFixed(0)}%</span></div>`;
    return `<tr><td><b>${j.date}</b></td><td class="ip">${j.host||'—'}</td><td>${fmt(a)}</td><td>${fmt(t)}</td><td>${bar}</td><td class="ip">${fmtTime(j.updated_at||j.claimed_at)}</td></tr>`;
  }).join('');
}
function renderNonebay(rows){const tb=document.getElementById('nonebay');
  if(!rows||!rows.length){tb.innerHTML='<tr><td colspan="7" class="svc-unknown" style="text-align:center;padding:14px">No non-eBay indices seeded</td></tr>';return;}
  tb.innerHTML=rows.map(r=>{const a=r.archived||0,t=r.os_total||0;const pct=t?Math.min(100,a/t*100):0;
    const st=r.status==='complete'?'<span class="svc-active">✓ complete</span>':r.status==='active'?'<span style="color:var(--yellow);font-weight:700">● active</span>':'<span style="color:var(--muted)">◌ queued</span>';
    const cov=r.status==='queued'?'—':`<div style="display:flex;align-items:center;gap:6px"><div style="background:var(--gray-dim);border-radius:3px;height:7px;width:80px;overflow:hidden"><div style="height:100%;background:${pct>=95?'var(--green)':pct>=60?'var(--yellow)':'var(--red)'};width:${pct}%"></div></div><span style="font-size:10px;color:var(--muted)">${pct.toFixed(0)}%</span></div>`;
    return `<tr><td><b>${r.index}</b></td><td>${st}</td><td>${r.status==='queued'?'—':fmt(a)}</td><td>${r.status==='queued'?'—':fmt(t)}</td><td>${cov}</td><td>${r.failed?fmt(r.failed):'—'}</td><td class="ip">${r.updated_at?fmtTime(r.updated_at):'—'}</td></tr>`;
  }).join('');
}
function renderWorkers(ws){const tb=document.getElementById('workers');
  if(!ws||!ws.length){tb.innerHTML='<tr><td colspan="4" class="loading">Loading…</td></tr>';return;}
  tb.innerHTML=ws.map(w=>{const svc=w.service??'unknown';const cls=svc==='active'?'svc-active':svc==='unknown'||svc==='—'?'svc-unknown':'svc-inactive';
    const txt=svc==='active'?'● active':svc==='inactive'?'○ inactive':svc;const log=w.log_line||(w.error?`ERROR: ${w.error}`:'—');
    return `<tr><td><b>w${w.index}</b></td><td class="ip">${w.ip}</td><td class="${cls}">${txt}</td><td class="log-preview" title="${log.replace(/"/g,'&quot;')}">${log}</td></tr>`;}).join('');
}
async function tick(){try{const r=await fetch('/api/status');const d=await r.json();
  nextT=d.next_update_at?new Date(d.next_update_at).getTime():Date.now()+60000;
  renderSummary(d);renderActive(d.active_jobs||[]);renderCal(d.coverage||{},d.per_day||{});renderNonebay(d.nonebay||[]);renderWorkers(d.workers||[]);}catch(e){console.error(e);}}
setInterval(()=>{if(nextT)document.getElementById('countdown').textContent=Math.max(0,Math.round((nextT-Date.now())/1000))+'s';},1000);
tick();setInterval(tick,60000);
</script></body></html>"""


class _Srv(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/dashboard"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            with _state_lock:
                payload = json.dumps(_state, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    print("[img-dashboard] initial poll…", flush=True)
    poll_and_update()
    if args.once:
        with _state_lock:
            print(json.dumps(_state, default=str, indent=2))
        return
    threading.Thread(target=_poller, daemon=True).start()
    srv = _Srv(("0.0.0.0", args.port), _Handler)
    print(f"[img-dashboard] serving at http://localhost:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
