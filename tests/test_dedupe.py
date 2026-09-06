"""Near-duplicate dedupe filter."""
from __future__ import annotations

from jobagent.db.enums import RemoteType
from jobagent.db.schemas import Job
from jobagent.filters.dedupe import find_clusters, normalize_title


def _job(jid: str, title: str, location: str, desc: str) -> Job:
    return Job(
        id=jid,
        source="ashby",
        company="Acme",
        title=title,
        location=location,
        url=f"https://x/{jid}",
        description_text=desc,
        remote_type=RemoteType.unknown,
    )


def test_normalize_title_strips_location_suffixes() -> None:
    assert normalize_title("Deployed Engineer (Atlanta)") == "deployed engineer"
    assert normalize_title("Deployed Engineer (Bay Area)") == "deployed engineer"
    assert normalize_title("Data Engineer - Remote, US") == "data engineer"
    assert normalize_title("ML Engineer, Austin, TX") == "ml engineer"
    # Role qualifiers (not locations) are preserved.
    assert normalize_title("Software Engineer, Backend") == "software engineer, backend"
    assert normalize_title("Software Engineer, Frontend") == "software engineer, frontend"


# Realistic body: many unique tokens, so a few per-city differences barely move
# the Jaccard (as with real postings). A tiny-vocab body would be over-sensitive.
_BODY = (
    "We are hiring a deployed engineer to partner with enterprise customers, "
    "integrate our platform, build reliable distributed systems, own kubernetes "
    "infrastructure, improve observability and performance, debug production "
    "incidents, write python and typescript, design apis, and collaborate across "
    "product, sales engineering, and support teams on developer experience. "
)


def test_city_variants_collapse_to_one() -> None:
    jobs = [
        _job("ashby:3", "Deployed Engineer (Denver)", "Denver, CO",
             _BODY + "Compensation 150000 to 250000. Based in Denver."),
        _job("ashby:1", "Deployed Engineer (Atlanta)", "Atlanta, GA",
             _BODY + "Compensation 155000 to 360000. Requires travel. Based in Atlanta."),
        _job("ashby:2", "Deployed Engineer (NYC)", "New York, NY",
             _BODY + "Based in New York."),
    ]
    clusters = find_clusters(jobs)
    assert len(clusters) == 1
    cl = clusters[0]
    assert cl.canonical.id == "ashby:1"  # lowest id -> deterministic canonical
    assert len(cl.shadows) == 2
    assert cl.locations == ["Atlanta, GA", "Denver, CO", "New York, NY"]


def test_distinct_roles_not_merged() -> None:
    # Same normalized title, but very different descriptions -> separate clusters.
    jobs = [
        _job("a:1", "Engineer (SF)", "SF", "backend distributed systems in Go and Postgres"),
        _job("a:2", "Engineer (NYC)", "NYC", "frontend react typescript design systems"),
    ]
    clusters = find_clusters(jobs)
    assert len(clusters) == 2
    assert all(len(c.shadows) == 0 for c in clusters)


def test_different_companies_never_merge() -> None:
    body = "same text " * 30
    j1 = _job("a:1", "Deployed Engineer (SF)", "SF", body)
    j2 = _job("b:1", "Deployed Engineer (SF)", "SF", body)
    j2 = j2.model_copy(update={"company": "OtherCo"})
    clusters = find_clusters([j1, j2])
    assert len(clusters) == 2


def test_canonical_is_deterministic() -> None:
    body = "shared " * 40
    jobs = [
        _job("z:9", "Role (A)", "A", body),
        _job("a:1", "Role (B)", "B", body),
        _job("m:5", "Role (C)", "C", body),
    ]
    c1 = find_clusters(list(jobs))
    c2 = find_clusters(list(reversed(jobs)))
    assert c1[0].canonical.id == "a:1"
    assert c2[0].canonical.id == "a:1"  # order-independent
