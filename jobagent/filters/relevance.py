"""Deterministic relevance pre-filter. ZERO LLM calls.

Cuts the ~95% of postings a new-grad MS CS (AI/ML) would never apply to
(sales, recruiting, ops, senior/exec, non-US). Every decision is a pure
function of the title + location so it is auditable and reproducible.

Filtered jobs are marked ``status='filtered'`` with a reason code (they are
never deleted), so the cut set can be reviewed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..db.enums import RemoteType

# Reason codes (also used as the persisted ``filter_reason`` value).
REASON_IRRELEVANT_TITLE = "irrelevant_title"
REASON_WRONG_LOCATION = "wrong_location"
REASON_TOO_SENIOR = "too_senior"

# Unambiguous software/ML/data signals. A title with any of these is relevant.
# "engineer" and "research" are deliberately NOT here — alone they are too
# Concrete software/ML/data signals — a title with any of these is relevant.
TECH_SIGNAL_TERMS: tuple[str, ...] = (
    "ml",
    "machine learning",
    "ai",
    "artificial intelligence",
    "llm",
    "nlp",
    "deep learning",
    "computer vision",
    "applied scientist",
    "software",
    "backend",
    "back end",
    "back-end",
    "frontend",
    "front end",
    "front-end",
    "full stack",
    "full-stack",
    "fullstack",
    "platform",
    "infra",
    "infrastructure",
    "devops",
    "sre",
    "site reliability",
    "distributed systems",
    "data",
    "robotics",
    "perception",
    "autonomy",
    "compiler",
)

# Generic software-role words. Relevant on their own UNLESS the title also names
# a non-software discipline (see NON_SOFTWARE_TERMS) — this keeps "Mobile
# Engineer" / "Security Engineer" / "iOS Developer" while cutting "Energy
# Engineer". "research" is intentionally excluded here (too generic:
# "Market Research", "User Research"); it qualifies only via KNOWN_GOOD_TITLES.
SOFTWARE_ROLE_TERMS: tuple[str, ...] = (
    "engineer",
    "developer",
    "programmer",
    "research",
)

# Non-software engineering disciplines / non-tech research -> irrelevant_title.
# Checked BEFORE the generic software-role words so "Energy Engineer" is cut but
# "Mobile Engineer" is kept. This denylist is deliberately narrow: we would
# rather keep a borderline software role than cut a good one.
NON_SOFTWARE_TERMS: tuple[str, ...] = (
    "mechanical",
    "electrical",
    "civil engineer",
    "chemical",
    "structural",
    "thermal",
    "energy",
    "manufacturing",
    "industrial",
    "aerospace",
    "propulsion",
    "hvac",
    "biomedical",
    "optical",
    "acoustic",
    "materials engineer",
    "cost engineer",
    "test stand",
    "solutions engineer",
    "sales engineer",
    "gtm",
    "market research",
    "user research",
    "ux research",
)

# Exact-ish known-good titles (substring match) relevant even without a tech
# signal — covers "Research Scientist" / "Research Engineer".
KNOWN_GOOD_TITLES: tuple[str, ...] = (
    "research engineer",
    "research scientist",
    "applied scientist",
    "data scientist",
    "data engineer",
)

# Seniority/level excludes -> too_senior. senior/sr/lead/ii/iii/iv added for
# consistency (previously staff/principal were cut but senior/lead survived).
TOO_SENIOR_TERMS: tuple[str, ...] = (
    "director",
    "vp",
    "vice president",
    "head of",
    "principal",
    "staff",
    "manager",
    "senior",
    "sr",
    "lead",
    "ii",
    "iii",
    "iv",
)

# Function/role-type excludes -> irrelevant_title. "intern" is a role-type
# mismatch for a full-time new grad, bucketed here rather than as too_senior.
IRRELEVANT_TITLE_TERMS: tuple[str, ...] = (
    "intern",
    "internship",  # \bintern\b misses "internship" (word boundary)
    "co-op",
    "coop",
    "working student",
    "apprentice",
    "trainee",
    "sales",
    "recruit",
    "sourcer",
    "account",
    "marketing",
    "legal",
    "finance",
    "support",
)

# US location markers (whole-word / substring as noted below).
_US_PHRASES: tuple[str, ...] = (
    "united states",
    "usa",
    "u.s.",
    "u.s.a",
    "remote us",
    "us remote",
    "remote - us",
    "remote, us",
    "remote (us",
    "anywhere in the us",
)
_US_STATE_CODES: frozenset[str] = frozenset(
    ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"]
)
# Non-US markers: if present and no US marker, the posting is non-US only.
_NON_US_MARKERS: tuple[str, ...] = (
    "united kingdom",
    "london",
    "canada",
    "toronto",
    "vancouver",
    "germany",
    "berlin",
    "munich",
    "france",
    "paris",
    "ireland",
    "dublin",
    "netherlands",
    "amsterdam",
    "spain",
    "madrid",
    "barcelona",
    "poland",
    "portugal",
    "lisbon",
    "india",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "mumbai",
    "pune",
    "chennai",
    "delhi",
    "gurgaon",
    "gurugram",
    "noida",
    "singapore",
    "australia",
    "sydney",
    "melbourne",
    "japan",
    "tokyo",
    "south korea",
    "korea",
    "seoul",
    "taiwan",
    "taipei",
    "hsinchu",
    "hong kong",
    "beijing",
    "shanghai",
    "shenzhen",
    "brazil",
    "mexico",
    "argentina",
    "buenos aires",
    "costa rica",
    "colombia",
    "chile",
    "peru",
    "emea",
    "apac",
    "latam",
    "europe",
    "israel",
    "tel aviv",
    "switzerland",
    "zurich",
    "sweden",
    "stockholm",
    "norway",
    "denmark",
    "finland",
    "austria",
    "belgium",
    "greece",
    "turkey",
    "united arab emirates",
    "uae",
    "abu dhabi",
    "dubai",
    "qatar",
    "doha",
    "saudi",
    "riyadh",
    "philippines",
    "manila",
    "thailand",
    "bangkok",
    "vietnam",
    "malaysia",
    "indonesia",
    "jakarta",
    "nigeria",
    "lagos",
    "kenya",
    "egypt",
)


@dataclass(frozen=True)
class RelevanceResult:
    relevant: bool
    reason: str | None = None


def _has_word(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms)


def _has_tech_signal(title_lower: str) -> bool:
    """A concrete software/ML/data signal or a known-good title. These win over
    the NON_SOFTWARE denylist (so "ML Solutions Engineer" is kept)."""
    if _has_word(title_lower, TECH_SIGNAL_TERMS):
        return True
    return any(good in title_lower for good in KNOWN_GOOD_TITLES)


def _location_is_us(location: str | None, remote_type: RemoteType) -> bool:
    """Conservative: only reject a posting whose location is *clearly* non-US.

    Ambiguous or missing locations are kept (we would rather review a borderline
    posting than silently eat a good US-remote role).
    """
    loc = (location or "").lower()
    if not loc.strip():
        return True  # unknown -> keep

    # 1. Explicit US phrasing wins (e.g. "Remote - US", "United States").
    if any(p in loc for p in _US_PHRASES):
        return True
    # 2. A clear non-US marker means non-US only. Checked BEFORE state codes so
    #    ambiguous codes (DE=Delaware/Germany, IN=Indiana/India) don't misfire
    #    on "Berlin, DE" / "Bangalore, IN".
    if any(m in loc for m in _NON_US_MARKERS):
        return False
    # 3. A US state code in "City, CA" form (uppercase, from the original).
    if any(code in _US_STATE_CODES for code in re.findall(r"\b[A-Z]{2}\b", location or "")):
        return True
    # 4. Neither signal (e.g. bare "Remote"): keep, don't over-cut.
    return True


def classify_relevance(
    title: str | None,
    location: str | None,
    remote_type: RemoteType = RemoteType.unknown,
) -> RelevanceResult:
    """Return whether a posting is relevant, with a reason code when not.

    Precedence: too_senior > irrelevant_title (excludes and no-include) >
    wrong_location.
    """
    t = (title or "").lower()
    if _has_word(t, TOO_SENIOR_TERMS):
        return RelevanceResult(False, REASON_TOO_SENIOR)
    if _has_word(t, IRRELEVANT_TITLE_TERMS):
        return RelevanceResult(False, REASON_IRRELEVANT_TITLE)
    # A concrete tech signal keeps the title even if it also names a
    # non-software discipline ("ML Solutions Engineer", "Backend ... GTM").
    if not _has_tech_signal(t):
        if _has_word(t, NON_SOFTWARE_TERMS):
            return RelevanceResult(False, REASON_IRRELEVANT_TITLE)
        if not _has_word(t, SOFTWARE_ROLE_TERMS):
            return RelevanceResult(False, REASON_IRRELEVANT_TITLE)
    if not _location_is_us(location, remote_type):
        return RelevanceResult(False, REASON_WRONG_LOCATION)
    return RelevanceResult(True, None)
