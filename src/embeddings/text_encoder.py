"""MiniLM-L6-v2 itemSpecifics text embedding encoder.

Produces 384-dim L2-normalised vectors from flattened itemSpecifics fields.
CPU-friendly — runs in parallel with GPU image encoding on a separate thread.

Usage:
    enc = TextEncoder()
    vec = enc.encode("psa 10 charizard base set 4/102 holo 1999")   # → np.ndarray (384,)

    vecs = enc.encode_batch(["psa 10 charizard ...", "bgs 9.5 ..."])  # → list[np.ndarray]
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from loguru import logger

MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE    = 1024


class TextEncoder:
    """
    Sentence-transformer encoder for card itemSpecifics text.

    Args:
        model_name: HuggingFace model identifier. Default: all-MiniLM-L6-v2
        device:     "cuda" | "cpu". Default: "cpu" (model is small, CPU is fine)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device     = device
        self._model     = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading TextEncoder ({}) on {}...", self.model_name, self.device)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("TextEncoder loaded in {:.1f}s", time.perf_counter() - t0)

    def encode(self, text: str | None) -> Optional[np.ndarray]:
        """Encode a single text string. Returns None if text is empty."""
        if not text or not text.strip():
            return None
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def encode_batch(
        self,
        texts: list[str | None],
        batch_size: int = BATCH_SIZE,
    ) -> list[Optional[np.ndarray]]:
        """
        Encode a list of text strings in one batched inference call.
        Empty/None entries return None in the same position.
        """
        self._load()

        # Separate valid texts from empty ones
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        valid_texts   = [texts[i] for i in valid_indices]

        results: list[Optional[np.ndarray]] = [None] * len(texts)

        if not valid_texts:
            return results

        vecs = self._model.encode(
            valid_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for i, idx in enumerate(valid_indices):
            results[idx] = vecs[i].astype(np.float32)

        return results


def format_specifics(item_specifics: dict | None) -> str:
    """
    Flatten an itemSpecifics dict into a natural-language string for MiniLM encoding.

    Field order is chosen by discriminative power for card identification:
    player/brand (who) → set (which set) → card_number (exact card) →
    year → subset/parallel (variant) → grader/grade (condition)

    All values are lowercased to match OpenSearch lowercase_normalizer behaviour.
    Missing fields contribute nothing — never raises.
    """
    if not item_specifics:
        return ""

    specs = item_specifics
    parts: list[str] = []

    def add(val):
        if val and str(val).strip():
            parts.append(str(val).strip().lower())

    # Primary identity
    add(specs.get("player"))
    add(specs.get("brand"))
    add(specs.get("genre"))

    # Card location in set
    add(specs.get("set"))
    cn = specs.get("cardNumber") or specs.get("card_number")
    if cn:
        parts.append(f"number {str(cn).strip().lower()}")
    if specs.get("year"):
        parts.append(str(specs["year"]))

    # Variant
    add(specs.get("subset"))
    add(specs.get("parallel"))
    sn = specs.get("serialNumber") or specs.get("serial_number")
    if sn:
        parts.append(f"serial {str(sn).strip().lower()}")

    # Condition / grading
    if specs.get("graded"):
        grader = (specs.get("grader") or "").strip()
        grade  = (specs.get("grade")  or "").strip()
        if grader and grade:
            parts.append(f"{grader.lower()} {grade.lower()}")
        elif grader:
            parts.append(grader.lower())

    if specs.get("autographed"):
        parts.append("autograph")

    # Context
    add(specs.get("team"))
    add(specs.get("country"))

    return " ".join(p for p in parts if p)


def format_specifics_from_title(title: str) -> str:
    """Fallback: encode raw listing title when itemSpecifics are missing."""
    return (title or "").lower().strip()
