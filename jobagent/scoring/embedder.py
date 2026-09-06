"""Local sentence-embedding abstraction (Stage A, zero API cost).

The real implementation wraps sentence-transformers (all-MiniLM-L6-v2) and runs
fully offline. It is behind a Protocol so tests inject a deterministic fake and
never download a model.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> Matrix:
        """Return an (len(texts), dim) float32 matrix."""
        ...


class MiniLMEmbedder:
    """sentence-transformers all-MiniLM-L6-v2 (384-dim). Lazily imported so the
    heavy dependency is only loaded when real embeddings are needed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> Matrix:
        vecs = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        )
        return np.asarray(vecs, dtype=np.float32)
