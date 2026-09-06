"""Table-driven tests for the deterministic relevance pre-filter."""
from __future__ import annotations

import pytest

from jobagent.db.enums import RemoteType
from jobagent.filters.relevance import (
    REASON_IRRELEVANT_TITLE,
    REASON_TOO_SENIOR,
    REASON_WRONG_LOCATION,
    classify_relevance,
)

# (title, location, expected_relevant, expected_reason)
CASES: list[tuple[str, str | None, bool, str | None]] = [
    # --- survivors (tech signal or known-good title) ---------------------
    ("Machine Learning Engineer", "San Francisco, CA", True, None),
    ("Software Engineer, Backend", "New York, NY", True, None),
    ("Applied Scientist", "Remote - US", True, None),
    ("Research Scientist, Perception", "Seattle, WA", True, None),  # known-good
    ("Research Engineer", "Remote - US", True, None),  # known-good
    ("AI Engineer", "Remote", True, None),  # bare remote -> kept
    ("Data Engineer", "Austin, TX", True, None),
    ("Platform Engineer (Infrastructure)", "Boston, MA", True, None),
    ("NLP Engineer", "United States", True, None),
    ("Data Scientist", "Remote, US", True, None),
    # --- too_senior (now includes senior/sr/lead/ii/iii/iv) --------------
    ("Staff Machine Learning Engineer", "San Francisco, CA", False, REASON_TOO_SENIOR),
    ("Senior Software Engineer", "Palo Alto, CA", False, REASON_TOO_SENIOR),
    ("Sr. Data Engineer", "New York, NY", False, REASON_TOO_SENIOR),
    ("Lead Machine Learning Engineer", "Remote - US", False, REASON_TOO_SENIOR),
    ("Software Engineer II", "Austin, TX", False, REASON_TOO_SENIOR),
    ("Data Engineer III", "Seattle, WA", False, REASON_TOO_SENIOR),
    ("Machine Learning Engineer IV", "Boston, MA", False, REASON_TOO_SENIOR),
    ("Engineering Manager", "New York, NY", False, REASON_TOO_SENIOR),
    ("Director of AI", "Remote - US", False, REASON_TOO_SENIOR),
    ("Principal Software Engineer", "Seattle, WA", False, REASON_TOO_SENIOR),
    ("VP of Engineering", "Austin, TX", False, REASON_TOO_SENIOR),
    ("Head of Data", "Boston, MA", False, REASON_TOO_SENIOR),
    # --- irrelevant_title: functions + generic engineer/research ---------
    ("Sales Engineer", "San Francisco, CA", False, REASON_IRRELEVANT_TITLE),
    ("Technical Recruiter", "New York, NY", False, REASON_IRRELEVANT_TITLE),
    ("Software Engineering Intern", "Remote - US", False, REASON_IRRELEVANT_TITLE),
    ("Internship - Search ML Engineer", "Remote - US", False, REASON_IRRELEVANT_TITLE),
    ("Machine Learning Co-op", "New York, NY", False, REASON_IRRELEVANT_TITLE),
    ("Software Engineer Apprentice", "Austin, TX", False, REASON_IRRELEVANT_TITLE),
    ("ML Trainee", "Boston, MA", False, REASON_IRRELEVANT_TITLE),
    ("Working Student Data Science", "Remote - US", False, REASON_IRRELEVANT_TITLE),
    # 'International' must NOT be caught by the intern/internship terms.
    ("Software Engineer, International Markets", "Remote - US", True, None),
    ("Account Executive", "Austin, TX", False, REASON_IRRELEVANT_TITLE),
    ("Customer Support Specialist", "Denver, CO", False, REASON_IRRELEVANT_TITLE),
    ("Office Coordinator", "Chicago, IL", False, REASON_IRRELEVANT_TITLE),
    ("Legal Counsel", "Remote - US", False, REASON_IRRELEVANT_TITLE),
    ("Energy Engineer", "Houston, TX", False, REASON_IRRELEVANT_TITLE),  # generic
    ("Mechanical Engineer", "Detroit, MI", False, REASON_IRRELEVANT_TITLE),  # generic
    ("User Research Analyst", "Remote - US", False, REASON_IRRELEVANT_TITLE),  # generic
    # --- wrong_location --------------------------------------------------
    ("Machine Learning Engineer", "London, United Kingdom", False, REASON_WRONG_LOCATION),
    ("Software Engineer", "Bangalore, IN", False, REASON_WRONG_LOCATION),
    ("AI Developer", "Berlin, DE", False, REASON_WRONG_LOCATION),
    ("Data Engineer", "Toronto, Canada", False, REASON_WRONG_LOCATION),
    ("Backend Engineer", "Singapore", False, REASON_WRONG_LOCATION),
    ("Platform Engineer", "EMEA", False, REASON_WRONG_LOCATION),
    # previously-leaked non-US locations
    ("Software Engineer", "Seoul, South Korea", False, REASON_WRONG_LOCATION),
    ("ML Engineer", "Taipei, Taiwan", False, REASON_WRONG_LOCATION),
    ("Data Engineer", "Abu Dhabi, UAE", False, REASON_WRONG_LOCATION),
    ("AI Engineer", "Doha, Qatar", False, REASON_WRONG_LOCATION),
    ("Backend Engineer", "Oslo, Norway", False, REASON_WRONG_LOCATION),
]


@pytest.mark.parametrize("title, location, relevant, reason", CASES)
def test_classify_relevance(
    title: str, location: str | None, relevant: bool, reason: str | None
) -> None:
    result = classify_relevance(title, location, RemoteType.unknown)
    assert result.relevant is relevant
    assert result.reason == reason


def test_ambiguous_country_codes_not_misread_as_us_states() -> None:
    # DE=Delaware/Germany, IN=Indiana/India: the non-US city must win.
    assert classify_relevance("AI Engineer", "Berlin, DE").reason == REASON_WRONG_LOCATION
    assert classify_relevance("AI Engineer", "Mumbai, IN").reason == REASON_WRONG_LOCATION
    # But genuine US state codes are kept.
    assert classify_relevance("AI Engineer", "Wilmington, DE").relevant is True


def test_exclude_beats_include() -> None:
    # 'Sales Engineer' contains include 'engineer' but exclude wins.
    assert classify_relevance("Sales Engineer", "Remote - US").relevant is False
