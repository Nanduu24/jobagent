"""Upsert idempotency + poll pipeline persistence tests."""
from __future__ import annotations

import datetime as dt

import httpx
import pytest
import respx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from jobagent.db import models, repository
from jobagent.db.enums import JobStatus, RemoteType
from jobagent.db.schemas import Job
from jobagent.ingest.base import HttpClient
from jobagent.ingest.companies import CompanyConfig
from jobagent.poller import run_poll

from .conftest import load_fixture


def _job(**overrides: object) -> Job:
    base: dict[str, object] = dict(
        id="greenhouse:1",
        source="greenhouse",
        company="Acme AI",
        title="ML Engineer",
        location="Remote",
        remote_type=RemoteType.remote,
        url="https://example.com/1",
        description_html="<p>Great role. Visa sponsorship is available.</p>",
        description_text="Great role. Visa sponsorship is available.",
        posted_at=dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc),
        sponsorship_ok=True,
    )
    base.update(overrides)
    return Job(**base)  # type: ignore[arg-type]


async def _count(session: AsyncSession, model: type[object]) -> int:
    result = await session.scalar(select(func.count()).select_from(model))
    return int(result or 0)


async def _count_status(session: AsyncSession, status: JobStatus) -> int:
    result = await session.scalar(
        select(func.count()).select_from(models.Job).where(models.Job.status == status)
    )
    return int(result or 0)


@pytest.mark.asyncio
async def test_upsert_is_idempotent_and_advances_last_seen(
    session: AsyncSession,
) -> None:
    t1 = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 6, 2, tzinfo=dt.timezone.utc)

    model, is_new = await repository.upsert_job(session, _job(), now=t1)
    await session.commit()
    assert is_new is True
    assert model.first_seen_at == t1
    assert model.last_seen_at == t1

    # Re-observe the same job later -> no duplicate, last_seen advances,
    # first_seen unchanged.
    model2, is_new2 = await repository.upsert_job(
        session, _job(title="ML Engineer II"), now=t2
    )
    await session.commit()
    assert is_new2 is False
    assert await _count(session, models.Job) == 1
    assert model2.first_seen_at == t1
    assert model2.last_seen_at == t2
    assert model2.title == "ML Engineer II"  # volatile field refreshed


@pytest.mark.asyncio
async def test_upsert_writes_no_events(session: AsyncSession) -> None:
    """Ingest must not write to the application-scoped events table."""
    t1 = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    model, is_new = await repository.upsert_job(session, _job(), now=t1)
    await session.commit()
    assert is_new is True
    assert model.status is JobStatus.new  # passing jobs stay 'new'
    assert await _count(session, models.Event) == 0


@respx.mock
@pytest.mark.asyncio
async def test_poll_twice_no_dupes_and_summary(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    respx.get(
        "https://boards-api.greenhouse.io/v1/boards/acme/jobs"
    ).respond(json=load_fixture("greenhouse.json"))

    companies = [CompanyConfig(name="Acme AI", source="greenhouse", token="acme")]
    t1 = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 6, 2, tzinfo=dt.timezone.utc)

    async with httpx.AsyncClient() as client:
        http = HttpClient(client=client, min_interval=0.0, max_retries=0)
        s1 = await run_poll(companies, session_factory, http=http, now=t1)

        # Fixture: "Machine Learning Engineer" (SF, CA; sponsorship True ->
        # survivor) and "Research Scientist" (Remote-US; sponsorship False ->
        # sponsorship_blocked). Both are relevant and fresh.
        assert s1.fetched == 2
        assert s1.new == 2
        assert s1.stale == 0
        assert s1.irrelevant_title == 0
        assert s1.too_senior == 0
        assert s1.wrong_location == 0
        assert s1.sponsorship_blocked == 1
        assert s1.survivors == 1
        assert s1.sponsorship_true == 1
        assert s1.sponsorship_false == 1
        assert s1.sponsorship_none == 0
        assert s1.old == 0  # both postings are within max_age
        assert s1.survivors_if_cut == 1
        assert s1.line() == (
            "2 fetched -> 0 duplicate -> 0 stale -> 0 irrelevant_title -> "
            "0 too_senior -> 0 wrong_location -> 1 sponsorship_blocked -> "
            "1 SURVIVORS [stale_mode=keep; 0 older-than-max-age (0 of them "
            "survivors; survivors_if_cut=1); 2 new; sponsorship "
            "True=1/False=1/None=0]"
        )
        # Funnel buckets partition the fetched set.
        assert (
            s1.duplicate
            + s1.stale
            + s1.irrelevant_title
            + s1.too_senior
            + s1.wrong_location
            + s1.sponsorship_blocked
            + s1.survivors
            == s1.fetched
        )
        assert not s1.errors

        # Second poll: same postings -> no new rows, last_seen advances.
        s2 = await run_poll(companies, session_factory, http=http, now=t2)
        assert s2.fetched == 2
        assert s2.new == 0
        assert s2.sponsorship_blocked == 1

    async with session_factory() as session:
        assert await _count(session, models.Job) == 2
        # Ingest writes no events.
        assert await _count(session, models.Event) == 0
        # Passing job stays 'new'; blocked job is 'filtered'.
        assert await _count_status(session, JobStatus.new) == 1
        blocked = (
            await session.scalars(
                select(models.Job).where(models.Job.status == JobStatus.filtered)
            )
        ).all()
        assert len(blocked) == 1
        assert blocked[0].sponsorship_ok is False
        assert blocked[0].filter_reason == "sponsorship_blocked"
        # age_days persisted for every row (posted_at is known in the fixtures).
        ages = [j.age_days for j in (await session.scalars(select(models.Job))).all()]
        assert all(a is not None for a in ages)
        # last_seen_at advanced to the second poll for every job.
        # (SQLite returns tz-naive datetimes; compare on the naive value.)
        def naive(d: dt.datetime) -> dt.datetime:
            return d.replace(tzinfo=None)

        rows = (await session.scalars(select(models.Job))).all()
        assert all(naive(r.last_seen_at) == naive(t2) for r in rows)
        assert all(naive(r.first_seen_at) == naive(t1) for r in rows)


def _gh_dupe_response() -> dict:
    """Greenhouse response: 3 near-duplicate 'Platform Engineer (City)' + 1 distinct."""
    body = (
        "We are hiring a platform engineer to build scalable distributed systems, "
        "own our kubernetes infrastructure, improve observability, reliability, and "
        "performance, design apis, debug production incidents, and collaborate with "
        "product and engineering teams using python, go, terraform, and postgres. "
    )
    jobs = []
    for i, (city, loc) in enumerate(
        [("NYC", "New York, NY"), ("SF", "San Francisco, CA"), ("Austin", "Austin, TX")]
    ):
        jobs.append({
            "id": 100 + i,
            "title": f"Platform Engineer ({city})",
            "updated_at": "2026-07-01T10:00:00-04:00",
            "first_published": "2026-07-01T10:00:00-04:00",
            "location": {"name": loc},
            "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{100 + i}",
            "content": body + f"This role is based in {city}.",
        })
    jobs.append({
        "id": 200,
        "title": "Data Scientist",
        "updated_at": "2026-07-01T10:00:00-04:00",
        "first_published": "2026-07-01T10:00:00-04:00",
        "location": {"name": "Remote - US"},
        "absolute_url": "https://boards.greenhouse.io/acme/jobs/200",
        "content": "Analyze experiments and build ML models in Python and PyTorch. " * 8,
    })
    return {"jobs": jobs}


@respx.mock
@pytest.mark.asyncio
async def test_poll_dedupe_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").respond(
        json=_gh_dupe_response()
    )
    companies = [CompanyConfig(name="Acme", source="greenhouse", token="acme")]
    t1 = dt.datetime(2026, 7, 10, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 7, 11, tzinfo=dt.timezone.utc)

    async with httpx.AsyncClient() as client:
        http = HttpClient(client=client, min_interval=0.0, max_retries=0)
        s1 = await run_poll(companies, session_factory, http=http, now=t1)
        # 4 fetched: 2 collapse as duplicate, canonical + Data Scientist survive.
        assert s1.fetched == 4
        assert s1.duplicate == 2
        assert s1.survivors == 2

        # Dedupe runs BEFORE relevance/freshness; re-run is idempotent.
        s2 = await run_poll(companies, session_factory, http=http, now=t2)
        assert s2.fetched == 4
        assert s2.duplicate == 2
        assert s2.new == 0

    async with session_factory() as session:
        assert await _count(session, models.Job) == 4
        dupes = (
            await session.scalars(
                select(models.Job).where(models.Job.filter_reason == "duplicate")
            )
        ).all()
        assert len(dupes) == 2
        # Canonical is the lowest-id Platform Engineer, survives, carries locations[].
        canon = await session.get(models.Job, "greenhouse:100")
        assert canon is not None
        assert canon.filter_reason != "duplicate"
        assert canon.locations == ["Austin, TX", "New York, NY", "San Francisco, CA"]
        # Shadows did not re-spawn as new rows or flip back.
        assert {d.id for d in dupes} == {"greenhouse:101", "greenhouse:102"}
