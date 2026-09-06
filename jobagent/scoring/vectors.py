"""Small vector helpers (cosine similarity, hashing)."""
from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]


def to_vector(values: list[float]) -> Vector:
    return np.asarray(values, dtype=np.float32)


def normalize(vec: Vector) -> Vector:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype(np.float32)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two vectors, in [-1, 1]."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
