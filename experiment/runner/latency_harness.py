"""Concurrent latency measurement harness.

Runs warmup queries, then measures search latency under concurrent load
across multiple repetitions. Records p50, p95, p99, and max latencies.
See CLAUDE.md Section 8.6 for configuration.
"""

LATENCY_CONFIG = {
    "warmup_queries": 1_000,
    "measurement_queries": 15_000,
    "concurrent_threads": 50,
    "repetitions": 5,
    "slo_p99_ms": 100,
}

PERCENTILES = [50, 95, 99, 100]
