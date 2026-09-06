"""`jobagent setup` onboarding: a scripted Prompter drives build_fact_bank with
zero real I/O and asserts a valid, on-disk fact bank comes out."""
from __future__ import annotations

from pathlib import Path

import pytest

from jobagent.factbank import load_fact_bank
from jobagent.onboarding import build_fact_bank, write_fact_bank


class ScriptedPrompter:
    """Answers questions from queues. text/integer pop from `answers`, boolean
    from `bools`; unmatched booleans default to their declared default."""

    def __init__(self, answers: list[str], bools: list[bool]) -> None:
        self._answers = iter(answers)
        self._bools = iter(bools)
        self.notes: list[str] = []

    def text(self, message: str, *, default: str | None = None,
             allow_empty: bool = False) -> str:
        try:
            return next(self._answers)
        except StopIteration:  # fall back to the field's own default
            return default or ""

    def boolean(self, message: str, *, default: bool = False) -> bool:
        try:
            return next(self._bools)
        except StopIteration:
            return default

    def integer(self, message: str, *, default: int = 0) -> int:
        try:
            return int(next(self._answers))
        except (StopIteration, ValueError):
            return default

    def note(self, message: str) -> None:
        self.notes.append(message)


def _minimal_script() -> ScriptedPrompter:
    # profile: name,email,phone,linkedin,github,location, [reloc],
    #          status, [citizen, spon_now, spon_future, clearance], years, seniority
    text_answers = [
        "Ada Lovelace", "ada@example.com", "", "", "", "London, UK",  # profile text
        "U.S. Citizen",              # work-auth status
        "3",                          # years
        "new grad, entry",           # target seniority
        # one project entry:
        "Analytical Engine", "First mechanical computer program", "Python, Math", "",
        # skills: one category then blank to stop
        "Languages", "Python, SQL",
        "",  # blank category -> stop skills
        # fact #1:
        "Analytical Engine", "project", "Wrote the first published algorithm "
        "intended for a machine, computing Bernoulli numbers.",
        "research",                   # context
        "algorithms, history",        # tags
        "Python",                     # tools
        "",                           # variant (skip)
    ]
    bool_answers = [
        True,     # relocation
        True, False, False, True,     # us_citizen, spon_now, spon_future, clearance
        False,    # add education? no
        False,    # add experience? no
        True,     # add project? yes
        False,    # add another project? no
        # facts: first is unconditional; then "add another fact?" -> no
        False,
    ]
    return ScriptedPrompter(text_answers, bool_answers)


def test_build_fact_bank_produces_valid_bank(tmp_path: Path) -> None:
    p = _minimal_script()
    data = build_fact_bank(p)
    out = tmp_path / "fact_bank.json"
    fb = write_fact_bank(data, out)

    assert out.exists()
    assert fb.profile.name == "Ada Lovelace"
    assert fb.profile.email == "ada@example.com"
    assert fb.profile.work_authorization.us_citizen is True
    assert fb.profile.years_professional_experience == 3
    assert len(fb.facts) == 1
    fact = fb.facts[0]
    assert fact.project == "Analytical Engine"
    assert "Bernoulli" in fact.claim
    assert fact.tools == ["Python"]
    # optional-but-empty fields were omitted, not written as blanks
    assert fb.profile.phone is None

    # the written file round-trips through the real loader
    reloaded = load_fact_bank(out)
    assert reloaded.profile.name == "Ada Lovelace"


def test_write_rejects_invalid_bank(tmp_path: Path) -> None:
    # A fact bank with zero facts must hard-fail (never silently write).
    bad = {
        "profile": {
            "name": "X", "email": "x@example.com",
            "work_authorization": {
                "status": "U.S. Citizen", "requires_sponsorship_now": False,
                "requires_sponsorship_future": False, "us_citizen": True,
                "clearance_eligible": True,
            },
            "years_professional_experience": 0,
        },
        "facts": [],
    }
    out = tmp_path / "fb.json"
    with pytest.raises(Exception):
        write_fact_bank(bad, out)
    assert not out.exists()  # nothing written on invalid input
