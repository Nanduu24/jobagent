"""Hard-requirement gate (semantic stack matching)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from jobagent.factbank import FactBank
from jobagent.scoring.gate import (
    GATE_CITIZENSHIP,
    GATE_CLEARANCE,
    GATE_EXPERIENCE,
    GATE_STACK,
    SkillMatcher,
    check_hard_requirements,
)
from jobagent.scoring.schemas import Requirements

from .conftest import MINIMAL_FACT_BANK

FB = FactBank.model_validate(MINIMAL_FACT_BANK)

# fact-bank corpus (lowercased skills + tools + tags): python, sql, pytorch, ml.
_VECS: dict[str, list[float]] = {
    "python": [1.0, 0.0, 0.0, 0.0],
    "sql": [0.0, 1.0, 0.0, 0.0],
    "pytorch": [0.0, 0.0, 1.0, 0.0],
    "ml": [0.0, 0.0, 0.0, 1.0],
    # required-skill inputs mapped to matching/orthogonal vectors:
    "rest": [1.0, 0.0, 0.0, 0.0],  # ~= python bucket (satisfied)
}


class StubEmbedder:
    """Deterministic: known strings map to fixed vectors, unknown -> zero
    (matches nothing, i.e. semantically 'missing')."""

    dim = 4

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        rows = [_VECS.get(t.strip().lower(), [0.0, 0.0, 0.0, 0.0]) for t in texts]
        return np.asarray(rows, dtype=np.float32)


MATCHER = SkillMatcher.build(FB, StubEmbedder(), threshold=0.60)


def _gate(req: Requirements, min_missing: int = 3) -> str | None:
    return check_hard_requirements(
        req, FB, max_years=5, matcher=MATCHER, min_missing=min_missing
    ).reason


def test_skill_matcher_semantic() -> None:
    # "python" matches the corpus; "rest" is mapped into the python bucket
    # (semantic hit); "cobol" is unknown -> unmatched.
    assert MATCHER.unmatched(["Python", "REST"]) == []
    assert MATCHER.unmatched(["COBOL"]) == ["COBOL"]
    assert MATCHER.unmatched(["Python", "COBOL", "Fortran"]) == ["COBOL", "Fortran"]


def test_experience_gate() -> None:
    assert _gate(Requirements(required=["Python"], years_experience=10)) == GATE_EXPERIENCE
    assert _gate(Requirements(required=["Python"], years_experience=5)) == GATE_EXPERIENCE


def test_experience_below_threshold_passes() -> None:
    assert _gate(Requirements(required=["Python"], years_experience=2)) is None


def test_clearance_gate() -> None:
    assert _gate(Requirements(required=["Python"], requires_clearance=True)) == GATE_CLEARANCE


def test_citizenship_gate() -> None:
    assert (
        _gate(Requirements(required=["Python"], must_have_citizenship=True))
        == GATE_CITIZENSHIP
    )


def test_stack_gate_fires_only_at_min_missing() -> None:
    # 3 unmatched skills -> gate; 2 unmatched -> no gate (one niche gap is fine).
    assert _gate(Requirements(required=["COBOL", "Fortran", "Mainframe"])) == GATE_STACK
    assert _gate(Requirements(required=["COBOL", "Fortran"])) is None
    assert _gate(Requirements(required=["COBOL", "Fortran"]), min_missing=2) == GATE_STACK


def test_semantic_match_prevents_false_stack_gate() -> None:
    # "REST" and "Python" are satisfied; only "COBOL"/"Fortran" unmatched (2<3).
    res = check_hard_requirements(
        Requirements(required=["Python", "REST", "COBOL", "Fortran"]),
        FB, max_years=5, matcher=MATCHER, min_missing=3,
    )
    assert res.reason is None
    assert set(res.missing) == {"COBOL", "Fortran"}


def test_clean_requirements_pass() -> None:
    assert _gate(Requirements(required=["Python", "PyTorch"], years_experience=1)) is None
