#!/usr/bin/env python3
"""
Score VLM card-identification extractions against itemSpecifics ground truth.

Reports, per backend:
  - Per-field accuracy (player/year/brand/set/card_number/parallel), split
    raw vs graded — graded slabs have a printed label and are far easier, so
    they must be reported separately.
  - IDENTITY-MATCH RATE — the gate metric: player AND year AND set AND
    card_number all correct. This is the same tier-3 identity that capped the
    image metric head; it's what decides whether text-extraction-based card-ID
    is worth building.
  - Latency + (if --price given) projected per-query cost.
  - A disagreement list for manual review: itemSpecifics is seller-entered and
    sometimes wrong, so a VLM "miss" may actually be the VLM being right.

Decision bands (identity-match, raw cards — the hard case):
  >= 85%   strong → build it
  70-85%   promising → tune schema/prompt and re-run
  <  70%   weak → VLM can't read cards reliably; reconsider

Usage:
    python tools/score_vlm_extraction.py --eval-dir data/vlm_eval \
        --extractions data/vlm_eval/extract_anthropic.jsonl \
                       data/vlm_eval/extract_openai.jsonl \
                       data/vlm_eval/extract_local.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ID_FIELDS    = ["player", "year", "set", "card_number"]   # strict identity key
ID_NONUM     = ["player", "year", "set"]                  # realistic retrieval key — card_number often isn't on the front image
ID_COARSE    = ["player", "set"]                          # coarse narrow
ALL_FIELDS   = ["player", "year", "brand", "set", "card_number", "parallel"]

_PUNCT = re.compile(r"[^\w\s]")
_WS    = re.compile(r"\s+")
_STOP  = {"the", "a", "of", "and", "card", "trading"}


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def norm_number(s: str) -> str:
    """Card numbers: strip punctuation/leading zeros so US-175 == US175, 04 == 4."""
    s = norm(s).replace(" ", "")
    s = re.sub(r"0+(\d)", r"\1", s)
    return s


def field_match(field: str, truth: str, pred: str) -> bool:
    t, p = norm(truth), norm(pred)
    if not t:
        return True   # no ground truth for this field — don't penalise
    if not p:
        return False
    if field == "card_number":
        return norm_number(truth) == norm_number(pred)
    if t == p:
        return True
    # set/brand/player: token-overlap (handles "topps chrome" vs "chrome",
    # "lebron james" vs "lebron"). Match if the smaller token set is a subset.
    tt = {w for w in t.split() if w not in _STOP}
    pp = {w for w in p.split() if w not in _STOP}
    if not tt or not pp:
        return t == p
    small, large = (tt, pp) if len(tt) <= len(pp) else (pp, tt)
    return small.issubset(large)


def load_extractions(path: Path) -> dict[str, dict]:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["os_id"]] = r
    return out


def score_backend(manifest: list[dict], extr: dict[str, dict],
                  price_per_call: float | None) -> dict:
    field_hits   = {f: {"raw": [0, 0], "graded": [0, 0]} for f in ALL_FIELDS}
    id_hits      = {"raw": [0, 0], "graded": [0, 0]}
    nonum_hits   = {"raw": [0, 0], "graded": [0, 0]}
    coarse_hits  = {"raw": [0, 0], "graded": [0, 0]}
    latencies    = []
    parse_errors = 0
    disagreements = []

    for card in manifest:
        rec = extr.get(card["os_id"])
        if rec is None:
            continue
        bucket = "graded" if card["graded"] else "raw"
        ex     = rec.get("extracted") or {}
        if rec.get("error") or "_parse_error" in ex:
            parse_errors += 1
            id_hits[bucket][1] += 1
            nonum_hits[bucket][1] += 1
            coarse_hits[bucket][1] += 1
            for f in ALL_FIELDS:
                field_hits[f][bucket][1] += 1
            continue
        if rec.get("latency_ms"):
            latencies.append(rec["latency_ms"])

        truth = card["truth"]
        per_field_ok = {}
        for f in ALL_FIELDS:
            ok = field_match(f, truth.get(f, ""), str(ex.get(f, "")))
            per_field_ok[f] = ok
            field_hits[f][bucket][0] += int(ok)
            field_hits[f][bucket][1] += 1

        id_ok = all(per_field_ok[f] for f in ID_FIELDS)
        id_hits[bucket][0] += int(id_ok)
        id_hits[bucket][1] += 1

        nonum_hits[bucket][0]  += int(all(per_field_ok[f] for f in ID_NONUM))
        nonum_hits[bucket][1]  += 1
        coarse_hits[bucket][0] += int(all(per_field_ok[f] for f in ID_COARSE))
        coarse_hits[bucket][1] += 1

        if not id_ok:
            disagreements.append({
                "os_id": card["os_id"], "graded": card["graded"],
                "truth": {f: truth.get(f, "") for f in ID_FIELDS},
                "pred":  {f: ex.get(f, "")    for f in ID_FIELDS},
                "vlm_text_read": ex.get("text_read", "")[:120],
            })

    def pct(hb):
        return 100.0 * hb[0] / hb[1] if hb[1] else 0.0

    summary = {
        "id_match": {b: pct(id_hits[b]) for b in ("raw", "graded")},
        "id_match_overall": pct([id_hits["raw"][0] + id_hits["graded"][0],
                                 id_hits["raw"][1] + id_hits["graded"][1]]),
        "id_nonum":  {b: pct(nonum_hits[b])  for b in ("raw", "graded")},
        "id_coarse": {b: pct(coarse_hits[b]) for b in ("raw", "graded")},
        "field": {f: {b: pct(field_hits[f][b]) for b in ("raw", "graded")}
                  for f in ALL_FIELDS},
        "parse_errors": parse_errors,
        "latency_ms_median": sorted(latencies)[len(latencies)//2] if latencies else 0,
        "n": sum(id_hits[b][1] for b in ("raw", "graded")),
        "cost_per_call": price_per_call,
        "disagreements": disagreements,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default="data/vlm_eval")
    ap.add_argument("--extractions", nargs="+", required=True)
    ap.add_argument("--price", type=float, default=None,
                    help="USD per VLM call, for the projected cost line.")
    ap.add_argument("--dump-disagreements", default=None,
                    help="Write the disagreement list to this JSON path.")
    args = ap.parse_args()

    eval_dir = ROOT / args.eval_dir
    manifest = [json.loads(l) for l in open(eval_dir / "manifest.jsonl")]
    n_raw    = sum(1 for c in manifest if not c["graded"])
    n_graded = len(manifest) - n_raw

    print(f"Eval set: {len(manifest)} cards ({n_raw} raw / {n_graded} graded)\n")

    all_disagreements = {}
    for ext_path in args.extractions:
        p     = ROOT / ext_path if not Path(ext_path).is_absolute() else Path(ext_path)
        extr  = load_extractions(p)
        any_rec = next(iter(extr.values()), {})
        label = f"{any_rec.get('backend','?')}/{any_rec.get('model','?')}"
        s     = score_backend(manifest, extr, args.price)
        all_disagreements[label] = s.pop("disagreements")

        print("=" * 70)
        print(f"  {label}   (n={s['n']}, parse_errors={s['parse_errors']})")
        print("=" * 70)
        print(f"  IDENTITY-MATCH   raw={s['id_match']['raw']:.1f}%   "
              f"graded={s['id_match']['graded']:.1f}%   "
              f"overall={s['id_match_overall']:.1f}%")
        print(f"  IDENTITY (no card#)  raw={s['id_nonum']['raw']:.1f}%   "
              f"graded={s['id_nonum']['graded']:.1f}%    "
              f"[player+year+set — the realistic retrieval key]")
        print(f"  IDENTITY (coarse)    raw={s['id_coarse']['raw']:.1f}%   "
              f"graded={s['id_coarse']['graded']:.1f}%    [player+set]")
        gate = ("STRONG → build" if s['id_nonum']['raw'] >= 85 else
                "PROMISING → tune" if s['id_nonum']['raw'] >= 70 else
                "WEAK → reconsider")
        print(f"  gate (raw, no-card# key): {gate}")
        print(f"\n  Per-field accuracy (raw / graded):")
        for f in ALL_FIELDS:
            print(f"    {f:13s} {s['field'][f]['raw']:5.1f}% / {s['field'][f]['graded']:5.1f}%")
        print(f"\n  median latency: {s['latency_ms_median']} ms", end="")
        if s["cost_per_call"]:
            print(f"   | per-query cost: ${s['cost_per_call']:.4f}", end="")
        print("\n")

    if args.dump_disagreements:
        out = ROOT / args.dump_disagreements
        out.write_text(json.dumps(all_disagreements, indent=2))
        total = sum(len(v) for v in all_disagreements.values())
        print(f"Wrote {total} disagreements (across backends) → {out}")
        print("REVIEW THESE: itemSpecifics is seller-entered; some 'misses' are "
              "the VLM being right and the seller wrong (= wins, not losses).")


if __name__ == "__main__":
    main()
