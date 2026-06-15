#!/usr/bin/env python3
"""
Run a VLM over the card-image eval set and extract structured identity fields.

Three pluggable backends so we can compare cost/accuracy:
  --backend anthropic   Claude vision (API key: ANTHROPIC_API_KEY)
  --backend openai      GPT-4o vision   (API key: OPENAI_API_KEY)
  --backend local       Self-hosted Qwen2-VL on the g5 GPU (no per-call cost)

The catalog is already text, so in production the VLM runs ONCE PER QUERY on
the user's uploaded photo — not over the 108M-card catalog. This POC measures
whether image → structured fields is accurate enough to retrieve the right
card by text, which would sidestep the image-embedding ceiling entirely.

Usage:
    python tools/vlm_extract.py --backend anthropic --model claude-sonnet-4-6 \
        --eval-dir data/vlm_eval --out data/vlm_eval/extract_anthropic.jsonl
    python tools/vlm_extract.py --backend openai --model gpt-4o \
        --eval-dir data/vlm_eval --out data/vlm_eval/extract_openai.jsonl
    python tools/vlm_extract.py --backend local --model Qwen/Qwen2-VL-7B-Instruct \
        --eval-dir data/vlm_eval --out data/vlm_eval/extract_local.jsonl

Resumable: skips os_ids already present in --out.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger

# ── The extraction prompt — identical across all three backends ────────────────

SYSTEM_PROMPT = (
    "You are a trading-card identification expert. You are shown ONE photo of a "
    "single trading card (it may be raw, or encased in a graded slab from PSA/BGS/"
    "SGC/CGC). Read the card and any slab label, then return ONLY a JSON object "
    "with these exact keys:\n"
    '  "player": the athlete/character name (lowercase)\n'
    '  "year": the season/year as printed, e.g. "2003-04" or "2011" (string)\n'
    '  "brand": manufacturer, e.g. "topps", "panini", "upper deck" (lowercase)\n'
    '  "set": the set/product name, e.g. "topps chrome", "prizm" (lowercase)\n'
    '  "card_number": the card number exactly as printed, e.g. "111", "US175", "4/102"\n'
    '  "parallel": the parallel/variant, e.g. "refractor", "silver prizm", or "" if base\n'
    '  "is_rookie": true/false\n'
    '  "graded": true if in a grading slab, else false\n'
    '  "grader": "psa"/"bgs"/"sgc"/"cgc" or "" if raw\n'
    '  "grade": the numeric grade if graded, e.g. "10", "9.5", else ""\n'
    '  "serial_number": serial like "42/99" if numbered, else ""\n'
    '  "confidence": your confidence 0.0-1.0 that the identity fields are correct\n'
    '  "text_read": a verbatim transcription of the text you can read on the card/slab\n'
    "Use \"\" for any field you cannot determine. Output ONLY the JSON object, no prose."
)

USER_PROMPT = "Identify this trading card. Return only the JSON object."


# ── Backends ───────────────────────────────────────────────────────────────────

def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


def call_anthropic(model: str, image_bytes: bytes, media_type: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": media_type, "data": _b64(image_bytes)}},
                {"type": "text", "text": USER_PROMPT},
            ],
        }],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def call_openai(model: str, image_bytes: bytes, media_type: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{_b64(image_bytes)}"}},
            ]},
        ],
    )
    return resp.choices[0].message.content or ""


_LOCAL = {}   # model + processor, loaded once via load_local_model()


def load_local_model(model: str, quantize: bool) -> None:
    """
    Load the self-hosted VLM once, up front, so a load failure crashes the
    script immediately with a real traceback instead of being swallowed by the
    per-card try/except and silently retried 300×.

    quantize=True loads in 4-bit (bitsandbytes) — required to fit a 7B VLM on
    the 16GB T4 workers (fp16 7B ≈ 14.6GB > the T4's usable ~14.5GB → OOM).
    4-bit drops it to ~5-6GB, which is also what production would run here.
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    logger.info("Loading {} (quantize={}) ...", model, quantize)
    if quantize:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            model, quantization_config=qcfg, device_map="auto",
        ).eval()
    else:
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.float16,
        ).eval().to("cuda")
    _LOCAL["model"] = m
    _LOCAL["proc"]  = AutoProcessor.from_pretrained(model)
    logger.info("Model loaded.")


def call_local(model: str, image_bytes: bytes, media_type: str) -> str:
    import io as _io
    import torch
    from PIL import Image

    m, proc = _LOCAL["model"], _LOCAL["proc"]
    img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text   = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(m.device)
    with torch.no_grad():
        gen = m.generate(**inputs, max_new_tokens=1024, do_sample=False)
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0]


BACKENDS = {"anthropic": call_anthropic, "openai": call_openai, "local": call_local}


def parse_json(raw: str) -> dict:
    """Extract the JSON object from the model's response (tolerant of fences/prose)."""
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1] if s.count("```") >= 2 else s
        s = s.replace("json", "", 1).strip() if s.lstrip().startswith("json") else s
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        return {"_parse_error": raw[:300]}
    try:
        return json.loads(s[a:b + 1])
    except Exception:
        return {"_parse_error": s[a:b + 1][:300]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--model",   required=True)
    ap.add_argument("--eval-dir", default="data/vlm_eval")
    ap.add_argument("--out",     required=True)
    ap.add_argument("--limit",   type=int, default=None, help="Cap N cards (testing).")
    ap.add_argument("--quantize", action="store_true",
                    help="local backend: load in 4-bit (needed to fit a 7B VLM on the 16GB T4).")
    args = ap.parse_args()

    if args.backend == "local":
        load_local_model(args.model, args.quantize)

    eval_dir = ROOT / args.eval_dir
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = [json.loads(l) for l in open(eval_dir / "manifest.jsonl")]
    if args.limit:
        manifest = manifest[:args.limit]

    done = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                done.add(json.loads(l)["os_id"])
            except Exception:
                pass
    todo = [c for c in manifest if c["os_id"] not in done]
    logger.info("{} cards total, {} already done, {} to process via {}/{}",
                len(manifest), len(done), len(todo), args.backend, args.model)

    backend = BACKENDS[args.backend]
    t0 = time.time()
    with open(out_path, "a") as fout:
        for i, c in enumerate(todo):
            img_path = eval_dir / c["image_file"]
            media = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            t_call = time.time()
            try:
                raw = backend(args.model, img_path.read_bytes(), media)
                extracted = parse_json(raw)
                err = None
            except Exception as e:
                raw, extracted, err = "", {}, f"{type(e).__name__}: {e}"
            fout.write(json.dumps({
                "os_id":      c["os_id"],
                "backend":    args.backend,
                "model":      args.model,
                "latency_ms": int((time.time() - t_call) * 1000),
                "extracted":  extracted,
                "error":      err,
            }) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                logger.info("  {}/{}  ({:.1f}s elapsed)", i + 1, len(todo), time.time() - t0)

    logger.info("Done — {} extractions → {} ({:.0f}s)", len(todo), out_path, time.time() - t0)


if __name__ == "__main__":
    main()
