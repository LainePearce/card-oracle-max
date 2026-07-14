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

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import boto3
from loguru import logger

SSH_KEY = str(Path.home() / ".ssh" / "qdrant-test.pem")
NODES = {
    "node-0": "172.31.0.41",
    "node-1": "172.31.7.154",
    "node-2": "172.31.6.110",
}
NAMESPACE = "CardOracle/Qdrant"


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

    cw.put_metric_data(Namespace=NAMESPACE, MetricData=metrics)
    logger.info("Published {} datapoints to {}", len(metrics), NAMESPACE)


if __name__ == "__main__":
    main()
