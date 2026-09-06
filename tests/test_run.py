"""`jobagent run` — the interactive apply loop (Step 1).

No LLM, no browser: the tailor step is a mock, `open` is a no-op, and the command
sequence is scripted. The headline case pins the spec: a run of 3 jobs driven by
"next skip stop" applies job1, leaves job2 queued, stops on job3, never touches
job3. Selection filtering (14-day window, applied-exclusion, newest-first, top-N)
is tested against the real SQLite session.
"""
from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from jobagent.db.enums import JobStatus, RemoteType
from jobagent.db.models import Event, Job
from jobagent.factbank import load_fact_bank
from jobagent.runner import (
    RunSummary,
    fetch_run_candidates,
    manual_fields_block,
    mark_applied,
    paste_block,
    render_candidate_table,
    run_loop,
)
from jobagent.tailor.pipeline import PreparedResume

FB = load_fact_bank("tests/fixtures/fact_bank.json")
NOW = dt.datetime(2026, 7, 17, tzinfo=dt.timezone.utc)


def _job(
    jid: str,
    *,
    status: JobStatus = JobStatus.queued,
    posted_days_ago: int | None = 1,
    score: float | None = 80.0,
) -> Job:
    posted = None if posted_days_ago is None else NOW - dt.timedelta(days=posted_days_ago)
    return Job(
        id=jid,
        source="greenhouse",
        company=f"Co-{jid}",
        title=f"ML Engineer {jid}",
        location="Remote",
        remote_type=RemoteType.remote,
        url=f"https://example.com/{jid}",
        description_html="",
        description_text="desc",
        posted_at=posted,
        match_score=score,
        status=status,
        first_seen_at=NOW,
        last_seen_at=NOW,
    )


async def _add(factory: async_sessionmaker[AsyncSession], *jobs: Job) -> None:
    async with factory() as s:
        s.add_all(list(jobs))
        await s.commit()


# --- selection ------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_filters_window_status_and_missing_posted(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _add(
        session_factory,
        _job("fresh", posted_days_ago=2),
        _job("stale", posted_days_ago=40),           # outside the 30-day default
        _job("noposted", posted_days_ago=None),       # no posted_at -> excluded
        _job("applied", status=JobStatus.applied, posted_days_ago=1),  # never shown
        _job("newonly", status=JobStatus.new, posted_days_ago=1),      # not queued
        _job("filtered", status=JobStatus.filtered, posted_days_ago=1),
    )
    async with session_factory() as s:
        candidates, total = await fetch_run_candidates(s, 10, now=NOW)
    ids = [j.id for j in candidates]
    assert ids == ["fresh"]
    assert total == 1


@pytest.mark.asyncio
async def test_fetch_window_defaults_to_30_and_days_flag_narrows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # A 20-day-old job is INSIDE the new 30-day default but OUTSIDE --days 14.
    await _add(
        session_factory,
        _job("recent", posted_days_ago=3),
        _job("threeweeks", posted_days_ago=20),
    )
    async with session_factory() as s:
        default_ids = [j.id for j in (await fetch_run_candidates(s, 10, now=NOW))[0]]
        narrow_ids = [
            j.id for j in (await fetch_run_candidates(s, 10, now=NOW, days=14))[0]
        ]
    assert default_ids == ["recent", "threeweeks"]  # 30-day default includes both
    assert narrow_ids == ["recent"]                  # --days 14 drops the 20-day one


@pytest.mark.asyncio
async def test_fetch_sorts_newest_first_not_by_score(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # older but higher score vs newer but lower score -> newest must win.
    await _add(
        session_factory,
        _job("old_high", posted_days_ago=10, score=99.0),
        _job("mid", posted_days_ago=5, score=70.0),
        _job("new_low", posted_days_ago=1, score=66.0),
    )
    async with session_factory() as s:
        candidates, total = await fetch_run_candidates(s, 10, now=NOW)
    assert [j.id for j in candidates] == ["new_low", "mid", "old_high"]
    assert total == 3


@pytest.mark.asyncio
async def test_fetch_top_n_and_total(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _add(
        session_factory,
        _job("a", posted_days_ago=1),
        _job("b", posted_days_ago=2),
        _job("c", posted_days_ago=3),
    )
    async with session_factory() as s:
        candidates, total = await fetch_run_candidates(s, 2, now=NOW)
    assert [j.id for j in candidates] == ["a", "b"]  # top-2 newest
    assert total == 3                                 # but 3 qualified


# --- mark_applied ---------------------------------------------------------
@pytest.mark.asyncio
async def test_mark_applied_sets_status_timestamp_and_event(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _add(session_factory, _job("j1"))
    async with session_factory() as s:
        job = await s.get(Job, "j1")
        assert job is not None
        await mark_applied(s, job, now=NOW)
    async with session_factory() as s:
        job = await s.get(Job, "j1")
        assert job is not None
        assert job.status is JobStatus.applied
        # SQLite drops tzinfo on round-trip (Postgres keeps it); compare naive.
        assert job.applied_at is not None
        assert job.applied_at.replace(tzinfo=dt.timezone.utc) == NOW
        events = list((await s.scalars(select(Event).where(Event.job_id == "j1"))).all())
        assert len(events) == 1
        assert events[0].from_status is JobStatus.queued
        assert events[0].to_status is JobStatus.applied


# --- the interactive loop: THE headline spec case -------------------------
class _ScriptedIO:
    def __init__(self, commands: list[str]) -> None:
        self._commands = iter(commands)
        self.out: list[str] = []
        self.opened: list[tuple[str, str | None]] = []
        self.tailored: list[str] = []

    def input_fn(self, _prompt: str) -> str:
        return next(self._commands)

    def output_fn(self, line: str) -> None:
        self.out.append(line)

    def open_fn(self, url: str, pdf: str | None) -> None:
        self.opened.append((url, pdf))

    async def tailor_fn(self, _session: AsyncSession, job: Job) -> PreparedResume:
        self.tailored.append(job.id)
        return PreparedResume(
            job_id=job.id, md_path=None,
            pdf_path=f"renders/{job.id}.pdf", bullet_count=5, thin=False,
        )


@pytest.mark.asyncio
async def test_loop_next_skip_stop(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _add(
        session_factory,
        _job("job1", posted_days_ago=1),
        _job("job2", posted_days_ago=2),
        _job("job3", posted_days_ago=3),
    )
    io = _ScriptedIO(["next", "skip", "stop"])
    async with session_factory() as s:
        candidates, total = await fetch_run_candidates(s, 3, now=NOW)
        assert [j.id for j in candidates] == ["job1", "job2", "job3"]
        summary = await run_loop(
            s, candidates, total, FB.profile,
            tailor_fn=io.tailor_fn, open_fn=io.open_fn,
            input_fn=io.input_fn, output_fn=io.output_fn,
            now_fn=lambda: NOW,
        )

    assert summary == RunSummary(applied=1, skipped=1, remaining=2)

    async with session_factory() as s:
        j1 = await s.get(Job, "job1")
        j2 = await s.get(Job, "job2")
        j3 = await s.get(Job, "job3")
        assert j1 is not None and j2 is not None and j3 is not None
        # job1 applied (next). SQLite drops tzinfo on round-trip; compare naive.
        assert j1.status is JobStatus.applied and j1.applied_at is not None
        assert j1.applied_at.replace(tzinfo=dt.timezone.utc) == NOW
        # job2 NOT applied (skip) — still queued.
        assert j2.status is JobStatus.queued and j2.applied_at is None
        # job3 untouched — stop ended the loop; never marked.
        assert j3.status is JobStatus.queued and j3.applied_at is None
        # exactly one applied event, for job1.
        events = list((await s.scalars(select(Event))).all())
        assert [e.job_id for e in events] == ["job1"]

    # all three were shown (tailored) before their command; stop was on job3.
    assert io.tailored == ["job1", "job2", "job3"]
    assert io.opened[0] == ("https://example.com/job1", "renders/job1.pdf")


@pytest.mark.asyncio
async def test_loop_reprompts_on_unrecognized_input(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _add(session_factory, _job("only", posted_days_ago=1))
    io = _ScriptedIO(["maybe", "", "yes", "next"])  # 3 junk, then a real command
    async with session_factory() as s:
        candidates, total = await fetch_run_candidates(s, 1, now=NOW)
        summary = await run_loop(
            s, candidates, total, FB.profile,
            tailor_fn=io.tailor_fn, open_fn=io.open_fn,
            input_fn=io.input_fn, output_fn=io.output_fn,
            now_fn=lambda: NOW,
        )
    assert summary == RunSummary(applied=1, skipped=0, remaining=0)
    assert io.tailored == ["only"]  # tailored exactly once despite re-prompts
    assert any("unrecognized" in ln for ln in io.out)


@pytest.mark.asyncio
async def test_loop_all_next_marks_all_applied(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _add(session_factory, _job("a", posted_days_ago=1), _job("b", posted_days_ago=2))
    io = _ScriptedIO(["next", "next"])
    async with session_factory() as s:
        candidates, total = await fetch_run_candidates(s, 2, now=NOW)
        summary = await run_loop(
            s, candidates, total, FB.profile,
            tailor_fn=io.tailor_fn, open_fn=io.open_fn,
            input_fn=io.input_fn, output_fn=io.output_fn,
            now_fn=lambda: NOW,
        )
    assert summary == RunSummary(applied=2, skipped=0, remaining=0)


# --- rendering helpers ----------------------------------------------------
def test_paste_block_has_verified_fields_and_workauth() -> None:
    lines = "\n".join(paste_block(FB.profile))
    assert FB.profile.email in lines
    assert "OPT" in lines  # honest work-auth line
    assert "F-1" in lines


def test_manual_block_lists_never_guess_fields() -> None:
    text = "\n".join(manual_fields_block()).lower()
    for needle in ("sponsorship", "eeo", "salary", "why us"):
        assert needle in text


def test_candidate_table_flags_short_supply() -> None:
    jobs = [_job("a", posted_days_ago=1)]
    lines = "\n".join(render_candidate_table(jobs, total=1, requested=5))
    assert "only 1 job(s) qualify" in lines
    assert "https://example.com/a" in lines
