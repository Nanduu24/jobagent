"""Regex sponsorship classifier over a job description.

Returns:
    True  -> the posting explicitly indicates sponsorship is available.
    False -> the posting explicitly blocks sponsorship / requires citizenship
             or clearance.
    None  -> unknown. NOT a rejection; the job stays in the queue.

Matching is tiered so the "allow wins" rule behaves sanely:

1. HARD block — sponsorship is *explicitly* negated or absent ("no visa
   sponsorship", "not able to sponsor", "without sponsorship"). These win over
   everything, so a negated sponsorship phrase can't leak through as an allow.
2. ALLOW — narrow, unambiguous positive phrasings ("we will sponsor",
   "sponsorship is available", "H-1B transfer").
3. SOFT block — citizenship / clearance / ITAR gates. Allow wins over these, so
   "requires a clearance, but we will sponsor exceptional candidates" -> True.
"""
from __future__ import annotations

import re

from ..text import html_to_text

# --- Tier 1: HARD block (sponsorship explicitly negated) ------------------
_HARD_BLOCK_PATTERNS: list[str] = [
    r"\bnot\s+able\s+to\s+sponsor\b",
    r"\bunable\s+to\s+(?:provide|offer|sponsor)\b",
    r"\b(?:do|does)\s+not\s+(?:offer|provide|support|sponsor)\b[^.]*\bsponsor",
    r"\bwill\s+not\s+sponsor\b",
    r"\b(?:cannot|can\s?not|won'?t|can'?t)\s+sponsor\b",
    r"\bno\s+(?:visa\s+|immigration\s+)?sponsorship\b",
    r"\bnot\s+eligible\s+for\s+(?:visa\s+|immigration\s+)?sponsorship\b",
    r"\bwithout\s+(?:the\s+need\s+for\s+)?(?:company\s+|employer\s+|visa\s+|immigration\s+)?sponsorship\b",
    r"\b(?:do|does|should|must|will|can)\s?(?:not|n'?t)\s+require\s+sponsorship\b",
    r"\bnot\s+require\s+(?:visa\s+|immigration\s+)?sponsorship\b",
]

# --- Tier 2: ALLOW (positive; wins over SOFT block) -----------------------
_ALLOW_PATTERNS: list[str] = [
    r"\bsponsorship\s+(?:is\s+)?available\b",
    r"\bvisa\s+sponsorship\s+(?:is\s+)?(?:available|offered|provided)\b",
    r"\bsponsorship\s+(?:is\s+)?(?:offered|provided|possible)\b",
    r"\bwill\s+sponsor\b",
    r"\bwe\s+(?:can|do|will|are\s+happy\s+to|are\s+glad\s+to)\s+sponsor\b",
    r"\b(?:happy|glad|willing)\s+to\s+sponsor\b",
    r"\bopen\s+to\s+(?:providing\s+)?sponsor(?:ship|ing)\b",
    r"\bh\-?1b\s+transfer(?:s)?\b",
    r"\bsponsor\s+(?:work\s+|employment\s+)?visas?\b",
]

# --- Tier 3: SOFT block (citizenship / clearance; allow overrides) --------
_SOFT_BLOCK_PATTERNS: list[str] = [
    # "authorized to work ... now or in the future [without sponsorship]"
    r"\bnow\s+or\s+in\s+the\s+future\b",
    r"\bmust\s+be\s+a\s+u\.?\s?s\.?\s+citizen\b",
    r"\bu\.?\s?s\.?\s+citizen(?:ship)?\b",
    r"\bcitizenship\s+(?:is\s+)?required\b",
    r"\bsecurity\s+clearance\b",
    r"\bu\.?\s?s\.?\s+person\b",
]

_HARD_BLOCK_RE = [re.compile(p, re.IGNORECASE) for p in _HARD_BLOCK_PATTERNS]
_ALLOW_RE = [re.compile(p, re.IGNORECASE) for p in _ALLOW_PATTERNS]
_SOFT_BLOCK_RE = [re.compile(p, re.IGNORECASE) for p in _SOFT_BLOCK_PATTERNS]

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Strip any markup, collapse whitespace, and lowercase.

    The filter is designed to read ``description_text`` (already normalized at
    the ingest boundary), but it defends itself here so a blocking phrase split
    across HTML tags (``unable</strong> to sponsor``) can never leak through.
    ``html_to_text`` is a no-op on already-plain text.
    """
    return _WS_RE.sub(" ", html_to_text(text)).strip().lower()


def classify_sponsorship(description: str | None) -> bool | None:
    """Classify a description as sponsor-friendly (True), blocked (False),
    or unknown (None)."""
    if not description or not description.strip():
        return None
    text = _normalize(description)
    if any(rx.search(text) for rx in _HARD_BLOCK_RE):
        return False
    if any(rx.search(text) for rx in _ALLOW_RE):
        return True
    if any(rx.search(text) for rx in _SOFT_BLOCK_RE):
        return False
    return None
