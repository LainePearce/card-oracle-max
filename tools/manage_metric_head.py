#!/usr/bin/env python3
"""
Deploy or rollback the metric head on the GPU worker EC2 instance.

Usage:
    # See what would happen — no changes made
    python tools/manage_metric_head.py --action dry-run

    # Full deploy (SCP model → backfill Qdrant → enable in .env → reload gunicorn)
    python tools/manage_metric_head.py --action deploy

    # Rollback (disable metric head in .env → reload gunicorn → verify)
    # Takes < 60 seconds regardless of how long deploy took.
    python tools/manage_metric_head.py --action rollback

    # Check current state of the worker
    python tools/manage_metric_head.py --action status

Configuration — edit the REMOTE_* constants below or pass via env vars:
    EC2_HOST        e.g.  13.57.253.55
    EC2_USER        e.g.  ec2-user
    EC2_SSH_KEY     e.g.  ~/.ssh/gpu-worker.pem
    EC2_CODE_DIR    e.g.  /home/ec2-user/card-oracle-max

Deploy timeline (for reference):
    Step 1  SCP metric head checkpoint (2.1 MB)          ~10 seconds
    Step 2  Add image_v2 field to Qdrant collection       <1 second
    Step 3  Backfill ~7M points (scroll → project → upsert) ~1–2 hours  ← bottleneck
    Step 4  Set METRIC_HEAD_ENABLED=1 in .env             <1 second
    Step 5  kill -HUP gunicorn master                     ~30 seconds
    Step 6  Verify /health returns metric_head=enabled    ~5 seconds

Rollback timeline:
    Step 1  Set METRIC_HEAD_ENABLED=0 in .env             <1 second
    Step 2  kill -HUP gunicorn master                     ~30 seconds
    Step 3  Verify /health returns metric_head=disabled   ~5 seconds
    Total                                                  < 60 seconds
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Remote configuration ──────────────────────────────────────────────────────
# Override via environment variables or edit directly.

EC2_HOST     = os.environ.get("EC2_HOST",     "13.57.253.55")
EC2_USER     = os.environ.get("EC2_USER",     "ec2-user")
EC2_SSH_KEY  = os.environ.get("EC2_SSH_KEY",  str(Path.home() / ".ssh" / "qdrant-test.pem"))
EC2_CODE_DIR = os.environ.get("EC2_CODE_DIR", "/home/ec2-user/card-oracle-max")

# Local checkpoint file to deploy
LOCAL_CHECKPOINT = ROOT / "models" / "metric_head_v2.pt"

# Remote path where the checkpoint will be placed
REMOTE_CHECKPOINT = f"{EC2_CODE_DIR}/models/metric_head_v2.pt"

# Remote .env file
REMOTE_ENV = f"{EC2_CODE_DIR}/.env"

# ALB endpoint for health checks (uses the public ALB, not direct instance IP)
WORKER_HEALTH_URL = os.environ.get(
    "WORKER_HEALTH_URL",
    "http://search-worker-alb-898324335.us-west-1.elb.amazonaws.com/health",
)

# ── SSH/SCP helpers ───────────────────────────────────────────────────────────

SSH_BASE = [
    "ssh",
    "-i", EC2_SSH_KEY,
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    f"{EC2_USER}@{EC2_HOST}",
]

SCP_BASE = [
    "scp",
    "-i", EC2_SSH_KEY,
    "-o", "StrictHostKeyChecking=no",
]


def ssh(cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command on the remote EC2 instance via SSH."""
    full = SSH_BASE + [cmd]
    print(f"  [ssh] {cmd}")
    return subprocess.run(
        full,
        check=check,
        capture_output=capture,
        text=True,
    )


def scp(local: str, remote: str) -> None:
    """Copy a local file to the remote EC2 instance."""
    full = SCP_BASE + [local, f"{EC2_USER}@{EC2_HOST}:{remote}"]
    print(f"  [scp] {local} → {remote}")
    subprocess.run(full, check=True)


def health_check(timeout: int = 30) -> dict | None:
    """Poll the worker /health endpoint. Returns parsed JSON or None on failure."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(WORKER_HEALTH_URL, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  [health] {e}")
        return None


def get_gunicorn_pid() -> str | None:
    """Return the gunicorn master PID on the remote instance, or None."""
    result = ssh("pgrep -o -f gunicorn", check=False, capture=True)
    pid = result.stdout.strip()
    return pid if pid else None


# ── Actions ───────────────────────────────────────────────────────────────────

def action_status() -> None:
    print("\n─── Status ───────────────────────────────────────────────────")

    # Remote process check
    result = ssh("pgrep -a -f gunicorn | head -3", check=False, capture=True)
    print(f"\nGunicorn processes:\n{result.stdout.strip() or '  (none found)'}")

    # Metric head env var on remote
    result = ssh(f"grep -i METRIC_HEAD {REMOTE_ENV} || echo '  (not set)'",
                 check=False, capture=True)
    print(f"\nMetric head config in .env:\n{result.stdout.strip()}")

    # Health check via ALB
    print("\nHealth check via ALB:")
    h = health_check()
    if h:
        for k, v in h.items():
            print(f"  {k}: {v}")
    else:
        print("  (unreachable)")


def action_dry_run() -> None:
    print("\n─── Dry-run ──────────────────────────────────────────────────")
    print(f"\nLocal checkpoint : {LOCAL_CHECKPOINT}")
    print(f"  exists         : {LOCAL_CHECKPOINT.exists()}")
    if LOCAL_CHECKPOINT.exists():
        size_mb = LOCAL_CHECKPOINT.stat().st_size / 1e6
        print(f"  size           : {size_mb:.1f} MB")

    print(f"\nRemote target    : {EC2_USER}@{EC2_HOST}:{REMOTE_CHECKPOINT}")
    print(f"SSH key          : {EC2_SSH_KEY}")

    print("\nConnectivity test:")
    result = ssh("echo OK", check=False, capture=True)
    print(f"  SSH: {'OK' if result.returncode == 0 else 'FAILED'}")

    print("\nDeploy steps that would run:")
    print("  1. SCP metric_head_v2.pt to EC2                 (~10 seconds)")
    print("  2. Add 'image_v2' field to Qdrant collection    (<1 second)")
    print("  3. Run backfill_metric_head.py on EC2           (~1–2 hours)")
    print("  4. Set METRIC_HEAD_ENABLED=1 in .env            (<1 second)")
    print("  5. kill -HUP gunicorn master                    (~30 seconds)")
    print("  6. Verify /health shows metric_head=enabled     (~5 seconds)")

    print("\nRollback (if needed after deploy):")
    print("  1. Set METRIC_HEAD_ENABLED=0 in .env            (<1 second)")
    print("  2. kill -HUP gunicorn master                    (~30 seconds)")
    print("  3. Verify /health shows metric_head=disabled    (~5 seconds)")
    print("  Total rollback time                             : < 60 seconds")


def action_deploy() -> None:
    print("\n─── Deploy ───────────────────────────────────────────────────")

    # ── Step 1: Verify local checkpoint ──────────────────────────────────────
    if not LOCAL_CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found at {LOCAL_CHECKPOINT}")
        sys.exit(1)
    size_mb = LOCAL_CHECKPOINT.stat().st_size / 1e6
    print(f"\nStep 1: checkpoint OK ({size_mb:.1f} MB)")

    # ── Step 2: SCP checkpoint to EC2 ────────────────────────────────────────
    print("\nStep 2: SCP metric_head_v2.pt to EC2...")
    ssh(f"mkdir -p {EC2_CODE_DIR}/models")
    t0 = time.perf_counter()
    scp(str(LOCAL_CHECKPOINT), REMOTE_CHECKPOINT)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # ── Step 3: Add image_v2 field to Qdrant (idempotent) ────────────────────
    print("\nStep 3: Ensuring 'image_v2' field exists in Qdrant collection...")
    ssh(
        f"cd {EC2_CODE_DIR} && "
        f"source .venv/bin/activate && "
        f"python tools/backfill_metric_head.py "
        f"  --checkpoint {REMOTE_CHECKPOINT} "
        f"  --dry-run"
    )

    # ── Step 4: Run backfill on EC2 (background, with screen/nohup) ──────────
    print("\nStep 4: Starting Qdrant backfill on EC2 (background)...")
    print("  This takes ~1–2 hours. The service stays live on 'image' vectors.")
    print("  Monitor with: ssh <ec2> 'tail -f /tmp/backfill_metric_head.log'")

    ssh(
        f"cd {EC2_CODE_DIR} && "
        f"source .venv/bin/activate && "
        f"nohup python tools/backfill_metric_head.py "
        f"  --checkpoint {REMOTE_CHECKPOINT} "
        f"  --verify "
        f"  > /tmp/backfill_metric_head.log 2>&1 &"
        f"echo 'Backfill PID:' $!"
    )

    print("\n  ⚡ Backfill is running in the background.")
    print("  Wait for it to complete, then run --action deploy-activate to cut over.")
    print("  Or run --action rollback at any time to stay on raw CLIP vectors.")


def action_deploy_activate() -> None:
    """
    Called AFTER backfill has completed. Flips the env var and reloads gunicorn.
    This is the zero-downtime cut-over to the metric head.
    """
    print("\n─── Deploy: Activating metric head ───────────────────────────")

    # Verify backfill log says it completed
    result = ssh("tail -5 /tmp/backfill_metric_head.log", check=False, capture=True)
    print(f"\nBackfill log tail:\n{result.stdout}")

    if "Backfill complete" not in result.stdout and "ready for queries" not in result.stdout:
        print("WARNING: backfill completion not confirmed in log. Proceed anyway? [y/N]")
        if input().strip().lower() != "y":
            print("Aborting. Run 'tail -f /tmp/backfill_metric_head.log' on EC2 to check.")
            sys.exit(1)

    # ── Step 5: Update .env ───────────────────────────────────────────────────
    print("\nStep 5: Setting METRIC_HEAD_ENABLED=1 in remote .env...")
    # Remove any existing METRIC_HEAD lines then append clean values
    ssh(
        f"cd {EC2_CODE_DIR} && "
        f"sed -i '/^METRIC_HEAD_ENABLED/d' .env && "
        f"sed -i '/^METRIC_HEAD_PATH/d' .env && "
        f"echo 'METRIC_HEAD_ENABLED=1' >> .env && "
        f"echo 'METRIC_HEAD_PATH={REMOTE_CHECKPOINT}' >> .env"
    )

    # ── Step 6: Reload gunicorn ───────────────────────────────────────────────
    print("\nStep 6: Reloading gunicorn (kill -HUP master)...")
    pid = get_gunicorn_pid()
    if not pid:
        print("ERROR: could not find gunicorn master PID")
        sys.exit(1)
    print(f"  Gunicorn master PID: {pid}")
    ssh(f"kill -HUP {pid}")

    print("  Waiting 35 seconds for workers to reload...")
    time.sleep(35)

    # ── Step 7: Verify ────────────────────────────────────────────────────────
    print("\nStep 7: Verifying /health...")
    h = health_check()
    if h:
        mh = h.get("metric_head", "unknown")
        vf = h.get("vector_field", "unknown")
        if mh == "enabled" and vf == "image_v2":
            print(f"  ✓ metric_head={mh}  vector_field={vf}")
            print("\nDeploy complete. Metric head v2 is live.")
            print("Monitor quality. To rollback: python tools/manage_metric_head.py --action rollback")
        else:
            print(f"  ✗ Unexpected state: metric_head={mh}  vector_field={vf}")
            print("  Worker may not have picked up new env. Check /tmp/gpu-worker.log on EC2.")
    else:
        print("  ✗ Health check failed — worker may be restarting. Retry in 30s.")


def action_rollback() -> None:
    print("\n─── Rollback ─────────────────────────────────────────────────")
    print("Disabling metric head. Service will revert to raw CLIP 'image' vectors.")

    t_start = time.perf_counter()

    # ── Step 1: Update .env ───────────────────────────────────────────────────
    print("\nStep 1: Setting METRIC_HEAD_ENABLED=0 in remote .env...")
    ssh(
        f"cd {EC2_CODE_DIR} && "
        f"sed -i '/^METRIC_HEAD_ENABLED/d' .env && "
        f"echo 'METRIC_HEAD_ENABLED=0' >> .env"
    )

    # ── Step 2: Reload gunicorn ───────────────────────────────────────────────
    print("\nStep 2: Reloading gunicorn (kill -HUP master)...")
    pid = get_gunicorn_pid()
    if not pid:
        print("ERROR: could not find gunicorn master PID — may need manual restart")
        sys.exit(1)
    print(f"  Gunicorn master PID: {pid}")
    ssh(f"kill -HUP {pid}")

    print("  Waiting 35 seconds for workers to reload...")
    time.sleep(35)

    # ── Step 3: Verify ────────────────────────────────────────────────────────
    print("\nStep 3: Verifying /health...")
    h = health_check()
    if h:
        mh = h.get("metric_head", "unknown")
        vf = h.get("vector_field", "unknown")
        elapsed = int(time.perf_counter() - t_start)
        if mh == "disabled" and vf == "image":
            print(f"  ✓ metric_head={mh}  vector_field={vf}")
            print(f"\nRollback complete in {elapsed}s. Service is back on raw CLIP vectors.")
        else:
            print(f"  ✗ Unexpected state: metric_head={mh}  vector_field={vf}")
            print("  Check /tmp/gpu-worker.log on EC2.")
    else:
        print("  ✗ Health check failed. Check worker logs on EC2.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deploy or rollback the metric head on the GPU worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Actions:
  dry-run         Show what would happen — no changes
  deploy          SCP model + start backfill (background, ~1-2 hours)
  deploy-activate Enable metric head after backfill completes (<60s)
  rollback        Disable metric head and revert to raw CLIP (<60s)
  status          Show current worker state

Typical workflow:
  1. python tools/manage_metric_head.py --action dry-run       # sanity check
  2. python tools/manage_metric_head.py --action deploy        # start backfill
     ... wait ~1-2 hours, monitor with:
     ssh ec2-user@<host> 'tail -f /tmp/backfill_metric_head.log'
  3. python tools/manage_metric_head.py --action deploy-activate  # cut over
  4. python tools/manage_metric_head.py --action rollback      # if needed
""",
    )
    ap.add_argument("--action", required=True,
                    choices=["dry-run", "deploy", "deploy-activate", "rollback", "status"])
    args = ap.parse_args()

    actions = {
        "dry-run":          action_dry_run,
        "deploy":           action_deploy,
        "deploy-activate":  action_deploy_activate,
        "rollback":         action_rollback,
        "status":           action_status,
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
