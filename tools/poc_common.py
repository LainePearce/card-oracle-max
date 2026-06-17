#!/usr/bin/env python3
"""
Shared config + helpers for the backbone POC (CLIP vs DINOv2 vs DINOv3).

One source of truth for: the POC Qdrant collection name, the three encoder
configs (Qdrant named vector ↔ S3 model_id/params ↔ dim ↔ precision), the S3
image key scheme, the eBay s-l1600 upsizer, and small S3 image helpers. Every
poc_* tool imports from here so they can never drift out of agreement.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent

POC_COLLECTION = "cards_backbone_poc"
DINO_IMAGE_SIZE = 512
IMAGE_VARIANTS = ("original", "512", "256")
RESIZE_DIMS = {"512": 512, "256": 256}   # "original" is stored as fetched


@dataclass(frozen=True)
class EncoderSpec:
    vector_name: str     # Qdrant named vector
    model_id:    str     # S3VectorStore model_id (key component)
    params:      str     # S3VectorStore params_hash (key component)
    dim:         int
    encoder:     str     # build_encoder name: clip | dinov2 | dinov3
    fp16:        bool     # precision for the DINO backbones (CLIP ignores)


# The three vector sets carried on every POC point. CLIP runs at its production
# 224px/fp16 config; the DINO backbones run @512 (the eval sweet spot), DINOv2
# in fp16 (stable, 2x faster) and DINOv3 in fp32 (NaN-unstable in fp16).
POC_ENCODERS: list[EncoderSpec] = [
    EncoderSpec("image",        "clip-vit-l-14", "v2-fp16-224px-sqpad", 768,  "clip",   True),
    EncoderSpec("image_dinov2", "dinov2-large",  "512px-fp16-sqpad",    1024, "dinov2", True),
    EncoderSpec("image_dinov3", "dinov3-vitl16", "512px-fp32-sqpad",    1024, "dinov3", False),
]

S3_IMAGE_PREFIX = "images/ebay"   # images/ebay/{variant}/{os_id}.jpg


def image_key(os_id: str, variant: str) -> str:
    # variant-first so lifecycle rules can tier a whole variant by prefix
    # (e.g. images/ebay/original/ -> Glacier IR) and the CDN serves images/ebay/512/.
    return f"{S3_IMAGE_PREFIX}/{variant}/{os_id.replace('/', '_')}.jpg"


_SL_RE = re.compile(r"s-l\d+")


def upsize_ebay_url(url: str) -> str:
    """Rewrite an eBay gallery URL to the large s-l1600 variant.

    eBay serves the same image at multiple sizes via the s-lNNN path segment
    (s-l225, s-l500, …). s-l1600 is the largest commonly available — the
    'full size' we archive. Non-eBay URLs are returned unchanged.
    """
    if "ebayimg.com" not in url:
        return url
    return _SL_RE.sub("s-l1600", url)


def poc_job_id(date_str: str) -> str:
    return f"poc-{date_str}"


# ── S3 image helpers ────────────────────────────────────────────────────────────

def make_s3():
    import boto3
    return boto3.client("s3")


def put_image(s3, bucket: str, key: str, jpeg_bytes: bytes) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=jpeg_bytes, ContentType="image/jpeg")


def get_image_pil(s3, bucket: str, key: str):
    from PIL import Image
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_jpeg(pil_img, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
