"""Near-duplicate collapse (runs at ingest, BEFORE relevance/freshness/scoring).

Boards often post the same role once per city ("Deployed Engineer (Atlanta)",
"... (Denver)", ...). Left alone, each copy consumes a Stage B window slot and a
queue line. We collapse them to ONE canonical row and record the collapsed
locations; the shadows are marked status='filtered', filter_reason='duplicate'
(kept for audit, never deleted).

Key = (company, normalized_title) + a description near-duplicate guard. The
description hash alone is too strict here — per-city copies differ in boilerplate
(salary band, a travel sentence) — so within a (company, normalized_title) group
we cluster by token-set Jaccard, which is robust to that drift without merging
genuinely different roles that happen to share a generic normalized title.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..db.schemas import Job

REASON_DUPLICATE = "duplicate"

_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")
_DASH_LOC_SUFFIX_RE = re.compile(
    r"\s*[-–—]\s*(remote|hybrid|onsite|on-site|us|usa|united states)"
    r"(\s*,?\s*(remote|us|usa|united states))?\s*$",
    re.IGNORECASE,
)
# Trailing ", City, ST" (US state code) — not ", Backend" (no state code).
_CITY_STATE_SUFFIX_RE = re.compile(r",\s*[^,]+,\s*[A-Za-z]{2}\s*$")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+")


def normalize_title(title: str) -> str:
    """Strip trailing location suffixes: "(City)", "- Remote", ", City, ST"."""
    prev = None
    out = title.strip()
    while prev != out:
        prev = out
        out = _PAREN_SUFFIX_RE.sub("", out)
        out = _DASH_LOC_SUFFIX_RE.sub("", out)
        out = _CITY_STATE_SUFFIX_RE.sub("", out)
        out = out.strip()
    return _WS_RE.sub(" ", out).lower()


def _tokens(text: str) -> frozenset[str]:
    """Alphabetic word tokens (numbers/salary/punctuation dropped)."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass
class DuplicateCluster:
    canonical: Job
    shadows: list[Job] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)


def find_clusters(
    jobs: list[Job], *, sim_threshold: float = 0.85
) -> list[DuplicateCluster]:
    """Group jobs into near-duplicate clusters.

    Deterministic: within each (company, normalized_title) group, jobs are sorted
    by id and greedily clustered by Jaccard>=threshold; the lowest-id job is the
    canonical, so re-runs always pick the same canonical (idempotent).
    """
    groups: dict[tuple[str, str], list[Job]] = {}
    for job in jobs:
        key = (job.company.lower(), normalize_title(job.title))
        groups.setdefault(key, []).append(job)

    clusters: list[DuplicateCluster] = []
    for _, members in groups.items():
        members.sort(key=lambda j: j.id)
        token_cache = {j.id: _tokens(j.description_text) for j in members}
        remaining = list(members)
        while remaining:
            canonical = remaining.pop(0)
            cluster = DuplicateCluster(canonical=canonical)
            still: list[Job] = []
            for other in remaining:
                if _jaccard(token_cache[canonical.id], token_cache[other.id]) >= sim_threshold:
                    cluster.shadows.append(other)
                else:
                    still.append(other)
            remaining = still
            locs = [canonical.location] + [s.location for s in cluster.shadows]
            cluster.locations = sorted({loc for loc in locs if loc})
            clusters.append(cluster)
    return clusters
