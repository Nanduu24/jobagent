"""Shared test fixtures: in-memory SQLite session + fixture loader."""
from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from jobagent.db.models import Base

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministic offline embedder for tests (no model download)."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls = 0
        self.texts_encoded = 0

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        self.calls += 1
        self.texts_encoded += len(texts)
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            rows.append(rng.standard_normal(self.dim).astype(np.float32))
        return np.asarray(rows, dtype=np.float32)


MINIMAL_FACT_BANK: dict[str, Any] = {
    "profile": {
        "name": "Test Candidate",
        "email": "t@example.com",
        "work_authorization": {
            "status": "F-1 OPT",
            "requires_sponsorship_now": False,
            "requires_sponsorship_future": True,
            "us_citizen": False,
            "clearance_eligible": False,
        },
        "years_professional_experience": 0,
        "target_seniority": ["new grad", "entry"],
    },
    "education": [{"id": "edu-1", "degree": "MS CS", "school": "Test U"}],
    "skills": {"languages": ["Python", "SQL"], "ml": ["PyTorch"]},
    "facts": [
        {
            "id": "f1",
            "project": "P",
            "context": "research",
            "tags": ["ml"],
            "tools": ["Python", "PyTorch"],
            "claim": "Built an ML pipeline in Python and PyTorch.",
        }
    ],
}


def load_fixture(name: str) -> Any:
    """Load a JSON fixture by filename (without directory)."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_raw(name: str) -> Any:
    """Load a raw captured board payload from fixtures/raw/."""
    return json.loads((FIXTURES / "raw" / name).read_text(encoding="utf-8"))


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A fresh in-memory SQLite DB (shared across connections via StaticPool)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as sess:
        yield sess
