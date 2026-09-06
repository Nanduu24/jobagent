"""Fact bank load + validation (malformed = hard fail)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jobagent.factbank import FactBank, hash_fact_bank, load_fact_bank

from .conftest import MINIMAL_FACT_BANK


def test_load_real_fact_bank() -> None:
    fb = load_fact_bank("tests/fixtures/fact_bank.json")
    assert fb.profile.years_professional_experience == 0
    assert fb.profile.work_authorization.us_citizen is False
    assert "python" in fb.all_skills()
    assert len(fb.fact_texts()) == len(fb.facts)


def test_load_minimal() -> None:
    fb = FactBank.model_validate(MINIMAL_FACT_BANK)
    assert "pytorch" in fb.all_skills()


def test_missing_required_field_hard_fails() -> None:
    bad = copy.deepcopy(MINIMAL_FACT_BANK)
    del bad["profile"]["work_authorization"]
    with pytest.raises(ValidationError):
        FactBank.model_validate(bad)


def test_empty_facts_hard_fails() -> None:
    bad = copy.deepcopy(MINIMAL_FACT_BANK)
    bad["facts"] = []
    with pytest.raises(ValidationError):
        FactBank.model_validate(bad)


def test_duplicate_fact_ids_hard_fails() -> None:
    bad = copy.deepcopy(MINIMAL_FACT_BANK)
    bad["facts"] = bad["facts"] + bad["facts"]
    with pytest.raises(ValidationError):
        FactBank.model_validate(bad)


def test_hash_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "fb.json"
    p.write_text(json.dumps(MINIMAL_FACT_BANK))
    h1 = hash_fact_bank(p)
    mutated = copy.deepcopy(MINIMAL_FACT_BANK)
    mutated["profile"]["name"] = "Someone Else"
    p.write_text(json.dumps(mutated))
    assert hash_fact_bank(p) != h1
