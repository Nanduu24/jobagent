"""Embed the fact bank ONCE and cache to disk, keyed by fact_bank.json's hash.

Changing fact_bank.json changes its hash, so the cache key changes and the
candidate embedding is recomputed automatically (old cache files are pruned).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..factbank import FactBank
from .embedder import Embedder
from .vectors import Vector, normalize


def _cache_path(cache_dir: Path, fact_bank_hash: str) -> Path:
    return cache_dir / f"candidate_{fact_bank_hash}.npy"


def load_or_build_candidate_vector(
    fact_bank: FactBank,
    embedder: Embedder,
    cache_dir: str | Path,
    fact_bank_hash: str,
) -> Vector:
    """Return the candidate embedding (mean of fact embeddings, normalized).

    Loads from ``cache_dir`` if a file for ``fact_bank_hash`` exists; otherwise
    embeds the facts once, caches, and prunes caches from other hashes.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, fact_bank_hash)
    if path.exists():
        return np.asarray(np.load(path), dtype=np.float32)

    fact_vectors = embedder.encode(fact_bank.fact_texts())
    candidate = normalize(np.asarray(fact_vectors, dtype=np.float32).mean(axis=0))

    # Prune stale candidate caches (fact bank changed).
    for old in cache_dir.glob("candidate_*.npy"):
        if old != path:
            old.unlink(missing_ok=True)
    np.save(path, candidate)
    return candidate
