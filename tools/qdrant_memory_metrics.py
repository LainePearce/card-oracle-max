#!/usr/bin/env python3
"""
Push qdrant node memory metrics to CloudWatch (from worker-0 — zero agent
footprint on the cluster nodes themselves; they get SSH-polled).

EC2 has no built-in memory metric, and the July 2026 OOM cascade was detected
by search 500s rather than telemetry. This publishes MemoryUsedPercent and
SwapUsedGB per node under the CardOracle/Qdrant namespace every run; a systemd
timer runs it once a minute, and CloudWatch alarms (see
infra/scripts/setup-qdrant-alarms.sh) page when a node crosses the line.

    python tools/qdrant_memory_metrics.py          # one shot (timer mode)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import boto3
import requests
from loguru import logger

SSH_KEY = str(Path.home() / ".ssh" / "qdrant-test.pem")
NODES = {
    "node-0": "172.31.0.41",
    "node-1": "172.31.7.154",
    "node-2": "172.31.6.110",
}
NAMESPACE = "CardOracle/Qdrant"
COLLECTIONS = ("cards", "cards_dinov2")
_QHEADERS = {"api-key": os.environ.get("QDRANT_API_KEY", "")}


def poll_points(ip: str, collection: str) -> int | None:
    """Approximate point count as seen from this node. Reads prefer local
    replicas, so per-node counts diverge when replicas do — the Sept 2026
    cards_dinov2/node-2 gap (411k points) was invisible to every other metric."""
    try:
        r = requests.post(f"http://{ip}:6333/collections/{collection}/points/count",
                          json={"exact": False}, headers=_QHEADERS, timeout=20)
        r.raise_for_status()
        return int(r.json()["result"]["count"])
    except Exception as e:
        logger.warning("count {} via {} failed: {}", collection, ip, e)
        return None


def poll_node(ip: str) -> dict | None:
    """Return {'mem_used_pct': float, 'swap_used_gb': float} or None."""
    try:
        out = subprocess.check_output(
            ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=6", "-o", "BatchMode=yes", f"ec2-user@{ip}",
             "free -b | awk 'NR==2{print $2, $7} NR==3{print $3}'"],
            stderr=subprocess.DEVNULL, timeout=12).decode().split()
        total, avail, swap_used = float(out[0]), float(out[1]), float(out[2])
        return {"mem_used_pct": round(100 * (1 - avail / total), 1),
                "swap_used_gb": round(swap_used / 1024 ** 3, 1)}
    except Exception as e:
        logger.warning("poll {} failed: {}", ip, e)
        return None


def main() -> None:
    cw = boto3.client("cloudwatch", region_name="us-west-1")
    metrics = []
    for name, ip in NODES.items():
        m = poll_node(ip)
        if m is None:
            # Publish an unreachable signal so an alarm can catch frozen nodes
            # (the July freeze pattern: box up, sshd starved).
            metrics.append({"MetricName": "NodeUnreachable", "Value": 1,
                            "Dimensions": [{"Name": "Node", "Value": name}]})
            continue
        metrics.append({"MetricName": "NodeUnreachable", "Value": 0,
                        "Dimensions": [{"Name": "Node", "Value": name}]})
        metrics.append({"MetricName": "MemoryUsedPercent", "Value": m["mem_used_pct"],
                        "Unit": "Percent",
                        "Dimensions": [{"Name": "Node", "Value": name}]})
        metrics.append({"MetricName": "SwapUsedGB", "Value": m["swap_used_gb"],
                        "Dimensions": [{"Name": "Node", "Value": name}]})
        logger.info("{}: mem {}%  swap {}GB", name, m["mem_used_pct"], m["swap_used_gb"])

    # Replica divergence: spread of per-node point counts per collection.
    for coll in COLLECTIONS:
        counts = {n: poll_points(ip, coll) for n, ip in NODES.items()}
        for n, cnt in counts.items():
            if cnt is not None:
                metrics.append({"MetricName": "PointsCount", "Value": cnt,
                                "Dimensions": [{"Name": "Node", "Value": n},
                                               {"Name": "Collection", "Value": coll}]})
        good = [c for c in counts.values() if c is not None]
        if len(good) >= 2:
            spread = max(good) - min(good)
            metrics.append({"MetricName": "ReplicaCountSpread", "Value": spread,
                            "Dimensions": [{"Name": "Collection", "Value": coll}]})
            logger.info("{}: counts {}  spread {:,}", coll, counts, spread)

    for i in range(0, len(metrics), 20):   # PutMetricData caps at 20 datapoints
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=metrics[i:i + 20])
    logger.info("Published {} datapoints to {}", len(metrics), NAMESPACE)


if __name__ == "__main__":
    main()
