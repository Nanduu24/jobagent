"""Table-driven tests for the sponsorship classifier."""
from __future__ import annotations

import pytest

from jobagent.filters.sponsorship import classify_sponsorship

# (description, expected) — real-world phrasings, including the nasty ones.
CASES: list[tuple[str, bool | None]] = [
    # --- Blocked (False) ---------------------------------------------------
    ("We are unable to provide immigration sponsorship for this role.", False),
    ("This position is not able to sponsor applicants for work visas.", False),
    ("No visa sponsorship is available for this position.", False),
    (
        "Applicants must be authorized to work in the U.S. now or in the "
        "future without sponsorship.",
        False,
    ),
    ("Candidates must be a US citizen.", False),
    ("This role requires an active security clearance.", False),
    ("Must be a U.S. person as defined by ITAR regulations.", False),
    ("We do not offer visa sponsorship at this time.", False),
    ("This position is not eligible for visa sponsorship.", False),
    ("You must not require sponsorship now or in the future.", False),
    ("The company will not sponsor applicants for this position.", False),
    ("Employment is contingent on obtaining a security clearance.", False),
    ("US citizenship is required for this federal contract.", False),
    (
        "Candidates must be authorized to work in the United States without "
        "company sponsorship.",
        False,
    ),
    ("Unfortunately we cannot sponsor visas for this opening.", False),
    ("Applicants should not require sponsorship for employment.", False),
    # --- Allowed (True) ----------------------------------------------------
    ("Visa sponsorship is available for exceptional candidates.", True),
    ("We will sponsor the right candidate.", True),
    ("H-1B transfer welcome for this position.", True),
    (
        "We are happy to sponsor work visas, including H-1B and green card "
        "processing.",
        True,
    ),
    ("Sponsorship available for this role.", True),
    ("We can sponsor visas for qualified international applicants.", True),
    # --- Unknown (None) ----------------------------------------------------
    ("We are looking for a talented backend engineer to join our team.", None),
    ("Competitive salary and equity offered to all employees.", None),
    ("", None),
    ("   ", None),
    ("Experience with Python and distributed systems is required.", None),
]


@pytest.mark.parametrize("description, expected", CASES)
def test_classify_sponsorship(description: str, expected: bool | None) -> None:
    assert classify_sponsorship(description) is expected


def test_allow_wins_over_block() -> None:
    """When both a block and an allow phrase appear, allow wins (spec rule)."""
    text = (
        "We generally require a security clearance for most roles, but we "
        "will sponsor exceptional candidates."
    )
    assert classify_sponsorship(text) is True


def test_none_input() -> None:
    assert classify_sponsorship(None) is None
