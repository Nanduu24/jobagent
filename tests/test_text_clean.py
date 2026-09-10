"""Boilerplate stripping: shared same-company blocks go, role-unique blocks stay."""
from __future__ import annotations

from jobagent.text_clean import (
    deboilerplate_by_company,
    strip_shared_blocks,
)

BOILER = (
    "ABOUT US\n\nAt LangChain, our mission is to make intelligent agents "
    "ubiquitous. We build the foundation for agent engineering, helping developers "
    "move from prototypes to production-ready AI agents that teams rely on, and we "
    "have grown from open-source tools into a full platform for the whole lifecycle."
)
ROLE_A = "We are hiring a FullStack Engineer to work on LangSmith, our observability and evals platform."
ROLE_B = "We are hiring a Customer Engineer to join our Enterprise Enablement team and support deployments."
ROLE_C = "We are hiring a Python OSS Engineer to maintain our open-source packages and developer tooling."


def _post(role: str) -> str:
    return f"{BOILER}\n\n{role}"


def test_shared_boilerplate_removed_role_kept() -> None:
    cleaned = strip_shared_blocks([_post(ROLE_A), _post(ROLE_B), _post(ROLE_C)])
    assert len(cleaned) == 3
    for out, role in zip(cleaned, [ROLE_A, ROLE_B, ROLE_C], strict=True):
        assert "At LangChain, our mission" not in out  # boilerplate gone
        assert role in out  # role-specific text kept
    # The three cleaned texts are now DIFFERENT (the whole point).
    assert len(set(cleaned)) == 3


def test_single_posting_is_untouched() -> None:
    # Nothing to compare against -> return unchanged.
    assert strip_shared_blocks([_post(ROLE_A)]) == [_post(ROLE_A)]


def test_short_shared_headers_are_kept() -> None:
    # A short shared block (< min_block_chars) like a section header must survive
    # so a posting is never emptied over headers alone.
    a = "Responsibilities:\n\n" + ROLE_A + "\n\n" + BOILER
    b = "Responsibilities:\n\n" + ROLE_B + "\n\n" + BOILER
    ca, cb = strip_shared_blocks([a, b])
    assert ca.startswith("Responsibilities:")
    assert cb.startswith("Responsibilities:")
    assert "At LangChain, our mission" not in ca
    assert ROLE_A in ca and ROLE_B in cb


def test_all_shared_posting_falls_back_to_original() -> None:
    # Two identical postings whose EVERY block is long and shared -> stripping
    # would empty both, so the fail-safe returns the originals (never "").
    intro = (
        "At LangChain our mission is to make intelligent agents ubiquitous across "
        "the whole developer lifecycle and beyond."
    )
    same = intro + "\n\n" + ROLE_A
    out = strip_shared_blocks([same, same])
    assert out == [same, same]


def test_deboilerplate_groups_by_company() -> None:
    items = [
        ("j1", "LangChain", _post(ROLE_A)),
        ("j2", "LangChain", _post(ROLE_B)),
        ("j3", "OtherCo", _post(ROLE_C)),  # alone in its company -> untouched
    ]
    out = deboilerplate_by_company(items)
    assert "At LangChain, our mission" not in out["j1"]
    assert "At LangChain, our mission" not in out["j2"]
    assert out["j3"] == _post(ROLE_C)  # OtherCo has no sibling -> unchanged


def test_other_company_boilerplate_does_not_cross_contaminate() -> None:
    # Same block text under two different companies, each with a unique sibling.
    items = [
        ("a1", "Acme", BOILER + "\n\n" + ROLE_A),
        ("a2", "Acme", BOILER + "\n\n" + ROLE_B),
        ("z1", "Zeta", BOILER + "\n\n" + ROLE_C),
    ]
    out = deboilerplate_by_company(items)
    # Acme has 2 postings sharing BOILER -> stripped.
    assert "At LangChain, our mission" not in out["a1"]
    # Zeta has only one posting -> nothing to compare -> BOILER stays.
    assert "At LangChain, our mission" in out["z1"]
