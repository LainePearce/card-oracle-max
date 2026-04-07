"""CLIP image embedding encoder.

Supports multiple CLIP model variants:
  ViT-L/14  pretrained=openai  → 768-dim  (Qdrant collection — Stage 1 backfill)
  ViT-B-32  pretrained=openai  → 512-dim  (extant OpenSearch imageVector — experiment control)

The correct model to use depends on the target system.
Qdrant was indexed with ViT-L/14 (768-dim).
The extant OpenSearch cluster was indexed with Xenova/clip-vit-base-patch32,
which is OpenAI ViT-B/32 in ONNX form — weights are identical to
open_clip.create_model("ViT-B-32", pretrained="openai").

Usage:
    # For Qdrant queries (default)
    enc = ImageEncoder()
    vec = enc.encode_url("https://i.ebayimg.com/...")  # returns np.ndarray shape (768,)

    # For OpenSearch KNN queries (experiment control)
    enc = ImageEncoder(model_name="ViT-B-32", pretrained="openai")
    vec = enc.encode_url(url)  # returns np.ndarray shape (512,)
"""

from __future__ import annotations

import io
import time
from typing import Optional

import httpx
import numpy as np
from loguru import logger

# ── Default model constants ───────────────────────────────────────────────────
# These match the Qdrant collection (primary use case).
MODEL_NAME    = "ViT-L/14"
PRETRAINED    = "openai"
EMBEDDING_DIM = 768   # ViT-L/14 output dim (512 is for ViT-B/32)
BATCH_SIZE    = 256
USE_FP16      = True
NORMALISE     = True

# HTTP settings for image download
DOWNLOAD_TIMEOUT   = 15   # seconds
MAX_IMAGE_BYTES    = 10 * 1024 * 1024  # 10MB
DOWNLOAD_HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; card-oracle/1.0)"}


class ImageEncoder:
    """
    CLIP image encoder with URL-based input.

    Args:
        model_name:  OpenCLIP model architecture string.  Default: "ViT-L/14"
        pretrained:  OpenCLIP pretrained weights tag.     Default: "openai"
        device:      "cuda" | "cpu"                       Default: "cuda"

    The model is loaded lazily on first encode call to avoid startup delay when
    the class is instantiated but encoding is not immediately needed.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: str = PRETRAINED,
        device:     str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._device_str = device
        self._model      = None
        self._preprocess = None
        self._torch      = None

    # ── Lazy initialisation ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load model weights. Called automatically on first encode call."""
        if self._model is not None:
            return

        import torch
        import open_clip

        self._torch = torch

        # Resolve device — fall back to CPU if CUDA requested but unavailable
        if self._device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            self._device_str = "cpu"

        self._device = torch.device(self._device_str)

        logger.info(
            "Loading CLIP {} ({}) on {}...",
            self.model_name, self.pretrained, self._device
        )
        t0 = time.perf_counter()

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self._device,
        )
        self._model.eval()

        # FP16 on GPU for speed (FP32 on CPU to avoid precision loss)
        if self._device_str == "cuda":
            self._model = self._model.half()

        logger.info(
            "CLIP model loaded in {:.1f}s — output dim: {}",
            time.perf_counter() - t0,
            self._embedding_dim(),
        )

    def _embedding_dim(self) -> int:
        """Infer embedding dimension from the loaded model."""
        dummy = self._torch.zeros(1, 3, 224, 224, device=self._device)
        with self._torch.no_grad():
            if self._device_str == "cuda":
                with self._torch.cuda.amp.autocast():
                    out = self._model.encode_image(dummy)
            else:
                out = self._model.encode_image(dummy)
        return out.shape[-1]

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def _pad_to_square(img) -> "Image.Image":
        """
        Pad a PIL image to square by adding neutral grey fill on the short edge.

        This ensures consistent CLIP encoding regardless of aspect ratio.
        Without padding, CLIP's centre-crop discards different fractions of
        portrait vs landscape images, making cross-aspect comparisons unreliable.

        Padding colour: RGB(114, 114, 114) — perceptually neutral grey that
        sits near the centre of the ImageNet normalisation range and does not
        introduce a strong colour signal at the edges.

        Applied identically at backfill time and query time via this method.
        """
        from PIL import Image as PILImage

        w, h = img.size
        if w == h:
            return img

        size    = max(w, h)
        padded  = PILImage.new("RGB", (size, size), (114, 114, 114))
        paste_x = (size - w) // 2
        paste_y = (size - h) // 2
        padded.paste(img, (paste_x, paste_y))
        return padded

    # ── Core encoding ─────────────────────────────────────────────────────────

    def _encode_pil(self, img) -> np.ndarray:
        """Encode a PIL image to a normalised float32 vector."""
        img    = self._pad_to_square(img)
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            if self._device_str == "cuda":
                with self._torch.cuda.amp.autocast():
                    vec = self._model.encode_image(tensor)
            else:
                vec = self._model.encode_image(tensor)

        # L2 normalise
        vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().float().numpy().squeeze()

    # ── Public API ────────────────────────────────────────────────────────────

    def encode_url(self, url: str | None) -> Optional[np.ndarray]:
        """
        Download an image from URL and return its CLIP embedding.

        Returns None if the URL is empty, unreachable, or cannot be decoded.
        Never raises — failures are logged as warnings.
        """
        if not url:
            return None

        self._load()

        try:
            resp = httpx.get(
                url,
                timeout=DOWNLOAD_TIMEOUT,
                follow_redirects=True,
                headers=DOWNLOAD_HEADERS,
            )
            resp.raise_for_status()

            if len(resp.content) > MAX_IMAGE_BYTES:
                logger.warning("Image too large ({} bytes), skipping: {}", len(resp.content), url)
                return None

            from PIL import Image
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return self._encode_pil(img)

        except httpx.HTTPStatusError as e:
            logger.warning("HTTP {} fetching image: {}", e.response.status_code, url)
        except httpx.RequestError as e:
            logger.warning("Network error fetching image ({}): {}", type(e).__name__, url)
        except Exception as e:
            logger.warning("Failed to encode image URL ({}): {}", type(e).__name__, url)

        return None

    def encode_batch_pil(self, images: list) -> np.ndarray:
        """
        Encode a list of PIL images in a single GPU forward pass.

        All images are padded to square, preprocessed, stacked into one batch
        tensor, and passed through CLIP in one call. This is the hot path for
        the batching daemon in gpu_worker_server.py.

        Returns float32 ndarray of shape (n, dim), L2-normalised.
        """
        self._load()

        tensors = []
        for img in images:
            img = self._pad_to_square(img)
            tensors.append(self._preprocess(img))

        batch = self._torch.stack(tensors).to(self._device)

        with self._torch.no_grad():
            if self._device_str == "cuda":
                with self._torch.cuda.amp.autocast():
                    vecs = self._model.encode_image(batch)
            else:
                vecs = self._model.encode_image(batch)

        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().float().numpy()

    def encode_batch(
        self,
        urls: list[str | None],
        progress: bool = False,
    ) -> list[Optional[np.ndarray]]:
        """
        Encode a list of image URLs.
        Returns a list of the same length; failed items are None.

        For large batches consider using the GPU batch job (batch_job.py) instead
        since this method fetches images sequentially.
        """
        self._load()

        results = []
        iterator = enumerate(urls)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Encoding images")

        for i, url in iterator:
            results.append(self.encode_url(url))

        return results
