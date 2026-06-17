#!/usr/bin/env python3
"""
POC component 5 — measure real image bytes and project S3 cost at scale.

Reads the archive manifest(s), averages the actual byte size of each variant
(original / 512 / 256), and projects monthly S3 storage cost for 100M and 200M
images under two strategies:
  - All S3 Standard
  - Tiered: originals in Glacier Instant Retrieval (rarely served, kept for
    re-embeds), resized variants in Standard (served behind a CDN)

This replaces the per-image size assumption in the plan with a measured number.

    python tools/poc_image_size_report.py --manifest data/poc/manifest_2026-06-01.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PRICE_STANDARD_GB = 0.023     # S3 Standard, us-east-1 $/GB-month
PRICE_GIR_GB      = 0.004     # Glacier Instant Retrieval $/GB-month
PRICE_PUT_1K      = 0.005     # $ per 1,000 PUT requests
GB = 1024 ** 3
TB = 1024 ** 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--scales", nargs="+", type=float, default=[100e6, 200e6])
    args = ap.parse_args()

    totals = {"original": 0, "512": 0, "256": 0}
    n = 0
    for mpath in args.manifest:
        for line in open(ROOT / mpath):
            sizes = json.loads(line).get("sizes", {})
            if not all(v in sizes for v in totals):
                continue
            for v in totals:
                totals[v] += sizes[v]
            n += 1

    if n == 0:
        print("No sized rows found in manifest(s).")
        return

    avg = {v: totals[v] / n for v in totals}
    per_image = sum(avg.values())
    resized   = avg["512"] + avg["256"]

    print(f"\nMeasured from {n:,} archived cards")
    print(f"  original  avg: {avg['original']/1024:8.1f} KB")
    print(f"  512px     avg: {avg['512']/1024:8.1f} KB")
    print(f"  256px     avg: {avg['256']/1024:8.1f} KB")
    print(f"  per-image    : {per_image/1024:8.1f} KB  "
          f"(original {100*avg['original']/per_image:.0f}% of bytes)\n")

    print(f"{'scale':>8}  {'total':>9}  {'Standard/mo':>12}  {'Tiered/mo':>11}  {'1x PUTs':>9}")
    print("-" * 60)
    for scale in args.scales:
        total_bytes   = per_image * scale
        resized_bytes = resized * scale
        orig_bytes    = avg["original"] * scale

        cost_standard = (total_bytes / GB) * PRICE_STANDARD_GB
        cost_tiered   = (resized_bytes / GB) * PRICE_STANDARD_GB \
                        + (orig_bytes / GB) * PRICE_GIR_GB
        put_cost      = (scale * 3 / 1000) * PRICE_PUT_1K

        print(f"{scale/1e6:6.0f}M  {total_bytes/TB:7.1f}TB  "
              f"${cost_standard:>10,.0f}  ${cost_tiered:>9,.0f}  ${put_cost:>7,.0f}")

    print("\nTiered = Glacier-IR originals + Standard resized (served via CDN).")
    print("CloudFront egress is separate and traffic-dependent; caching the")
    print("512/256 variants offloads nearly all S3 GETs.")


if __name__ == "__main__":
    main()
