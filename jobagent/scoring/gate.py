"""Hard-requirement gate.

A categorical blocker means the candidate cannot hold the role regardless of how
well the description otherwise matches — so we filter it with a reason rather
than merely scoring it low.

The stack check is SEMANTIC: an extracted required skill counts as "missing"
only if its best cosine similarity to the fact-bank skill/tag set is below a
threshold (so "API design" matches "REST"/"FastAPI", "distributed data systems"
matches "Postgres"/"pgvector"). The gate fires only when several required skills
are genuinely unmatched — one niche tool never gates an otherwise-strong match.
Embeddings reuse Stage A's local embedder (no LLM calls).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..factbank import FactBank
from .embedder import Embedder
from .schemas import Requirements

GATE_EXPERIENCE = "req_experience"
GATE_CLEARANCE = "req_clearance"
GATE_CITIZENSHIP = "req_citizenship"
GATE_STACK = "req_stack_missing"


@dataclass
class GateResult:
    reason: str | None
    missing: list[str]


class SkillMatcher:
    """Semantic membership test against the fact-bank skill/tag corpus."""

    def __init__(self, embedder: Embedder, corpus: list[str], threshold: float) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._corpus = corpus
        vecs = np.asarray(embedder.encode(corpus), dtype=np.float32) if corpus else None
        self._corpus_unit = _unit_rows(vecs) if vecs is not None else None

    @classmethod
    def build(
        cls, fact_bank: FactBank, embedder: Embedder, threshold: float
    ) -> SkillMatcher:
        corpus = sorted(
            fact_bank.all_skills()
            | {tag.lower() for fact in fact_bank.facts for tag in fact.tags}
        )
        return cls(embedder, corpus, threshold)

    def unmatched(self, required: list[str]) -> list[str]:
        """Return required skills whose best cosine to the corpus < threshold."""
        if not required or self._corpus_unit is None:
            return list(required)
        req_unit = _unit_rows(np.asarray(self._embedder.encode(required), dtype=np.float32))
        sims = req_unit @ self._corpus_unit.T  # (len(required), len(corpus))
        best = sims.max(axis=1)
        return [skill for skill, b in zip(required, best, strict=True) if float(b) < self._threshold]


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(mat / norms, dtype=np.float32)


def check_hard_requirements(
    req: Requirements,
    fact_bank: FactBank,
    *,
    max_years: int,
    matcher: SkillMatcher,
    min_missing: int,
) -> GateResult:
    """Return the gate reason (or None) plus the list of unmatched required items."""
    auth = fact_bank.profile.work_authorization
    missing = matcher.unmatched(req.required)

    if req.requires_clearance and not auth.clearance_eligible:
        return GateResult(GATE_CLEARANCE, missing)
    if req.must_have_citizenship and not auth.us_citizen:
        return GateResult(GATE_CITIZENSHIP, missing)
    if (
        req.years_experience is not None
        and req.years_experience >= max_years
        and req.years_experience > fact_bank.profile.years_professional_experience
    ):
        return GateResult(GATE_EXPERIENCE, missing)
    # Core stack absent: enough required skills are semantically unmatched.
    if len(missing) >= min_missing:
        return GateResult(GATE_STACK, missing)
    return GateResult(None, missing)
