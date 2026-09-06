"""`jobagent run` — the interactive apply loop (Step 1: open + prepare, no fill).

Step 1 of the human-in-the-loop apply flow: for each candidate job it tailors +
renders the resume, opens the application URL and the PDF, and prints exactly what
the human should paste plus the fields they must answer themselves. It then waits
for `next` / `skip` / `stop`.

THE IRON RULE HOLDS. This module opens and prepares; it never fills a form and
never submits. `next` means "I (the human) submitted this" and only updates the
DB. There is no browser automation and no submit path here.

The pure pieces (candidate selection, the loop) take injected I/O + a tailor
callback so they run with zero LLM calls / zero browser under test.
"""
from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db.enums import JobStatus
from .db.models import Event, Job
from .factbank import Profile
from .logging import get_logger
from .tailor.pipeline import PreparedResume
from .tailor.render import work_auth_line

log = get_logger(__name__)

# Default freshness window: only jobs posted within this many days are surfaced
# (fresh-first). Overridable per run via `jobagent run --days N`.
FRESH_WINDOW_DAYS = 30

# The three commands the loop accepts.
CMD_NEXT = "next"
CMD_SKIP = "skip"
CMD_STOP = "stop"


@dataclass(frozen=True)
class RunSummary:
    applied: int
    skipped: int
    remaining: int  # jobs still queued + within the window after the run
    days: int = FRESH_WINDOW_DAYS

    def line(self) -> str:
        return (
            f"applied {self.applied}, skipped {self.skipped}, "
            f"{self.remaining} remaining in the {self.days}-day queue"
        )


# --- candidate selection --------------------------------------------------
async def fetch_run_candidates(
    session: AsyncSession, n: int, *, now: dt.datetime, days: int = FRESH_WINDOW_DAYS
) -> tuple[list[Job], int]:
    """The run's candidate list: status='queued', posted within ``days`` (a job
    with no posted_at is EXCLUDED, never guessed), applied jobs never shown (they
    are no longer 'queued'), sorted NEWEST-FIRST by posted_at. Returns the top-N
    plus the total number that qualified (for the 'remaining' summary)."""
    cutoff = now - dt.timedelta(days=days)
    rows = list(
        (
            await session.scalars(
                select(Job)
                .where(Job.status == JobStatus.queued)
                .where(Job.posted_at.is_not(None))
                .where(Job.posted_at >= cutoff)
                .order_by(Job.posted_at.desc())
            )
        ).all()
    )
    return rows[:n], len(rows)


# --- marking applied ------------------------------------------------------
async def mark_applied(session: AsyncSession, job: Job, *, now: dt.datetime) -> None:
    """Mark THIS job applied — the permanent dedup key. Records applied_at and an
    events row (the append-only timeline). Commits."""
    prev = job.status
    job.status = JobStatus.applied
    job.applied_at = now
    session.add(
        Event(
            job_id=job.id,
            from_status=prev,
            to_status=JobStatus.applied,
            note="jobagent run: human marked submitted",
            occurred_at=now,
        )
    )
    await session.commit()


# --- rendering helpers ----------------------------------------------------
def _fmt_posted(job: Job) -> str:
    return job.posted_at.date().isoformat() if job.posted_at else "?"


def _fmt_score(job: Job) -> str:
    return f"{job.match_score:.1f}" if job.match_score is not None else "-"


def render_candidate_table(
    candidates: list[Job], total: int, requested: int,
    days: int = FRESH_WINDOW_DAYS,
) -> list[str]:
    """The dry-run view: the exact list `run` would work through, no tailoring."""
    lines = [
        f"Candidate list — top {len(candidates)} of {total} qualified "
        f"(queued, posted ≤ {days}d, not applied, newest-first):",
        "",
        f"  {'#':>2}  {'posted':<10} {'score':>5}  {'company':<16} {'title':<34}",
    ]
    for i, job in enumerate(candidates, start=1):
        lines.append(
            f"  {i:>2}  {_fmt_posted(job):<10} {_fmt_score(job):>5}  "
            f"{job.company[:16]:<16} {job.title[:34]:<34}"
        )
        lines.append(f"      {job.url}")
    if not candidates:
        lines.append(f"  (nothing qualifies — no queued jobs posted in the last {days} days)")
    elif total < requested:
        lines += [
            "",
            f"  only {total} job(s) qualify (fewer than the {requested} requested) "
            "— proceeding with what's available.",
        ]
    return lines


def paste_block(profile: Profile) -> list[str]:
    """Exactly what the human can paste — verified profile fields + the honest
    OPT work-auth line. Nothing here is auto-filled; it is for the human to copy."""
    p = profile
    return [
        "COPY-PASTE (verified — safe to paste as-is):",
        f"    Name       : {p.name}",
        f"    Email      : {p.email}",
        f"    Phone      : {p.phone or '-'}",
        f"    LinkedIn   : {p.linkedin or '-'}",
        f"    GitHub     : {p.github or '-'}",
        f"    Work auth  : {work_auth_line(p.work_authorization)}",
    ]


def manual_fields_block() -> list[str]:
    """Fields the human MUST answer themselves — never guess into these."""
    return [
        "ANSWER YOURSELF (do NOT guess — the agent will not fill these):",
        "    ⚑ Sponsorship / citizenship beyond the honest OPT line above",
        "    ⚑ EEO / demographic (race, gender, veteran, disability)",
        "    ⚑ Salary / compensation expectations",
        "    ⚑ Any 'why us' / cover-letter free-text",
    ]


# Injected seams: tailor one job, open URLs, read a command, emit a line.
TailorFn = Callable[[AsyncSession, Job], Awaitable[PreparedResume]]
OpenFn = Callable[[str, str | None], None]
InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def open_url_and_pdf(url: str, pdf_path: str | None) -> None:
    """Open the application URL and the rendered PDF in the default apps (macOS
    `open`). Both are opened so the human has the form and the resume ready."""
    subprocess.run(["open", url], check=False)
    if pdf_path:
        subprocess.run(["open", pdf_path], check=False)


async def run_loop(
    session: AsyncSession,
    candidates: list[Job],
    total_qualified: int,
    profile: Profile,
    *,
    tailor_fn: TailorFn,
    open_fn: OpenFn,
    input_fn: InputFn,
    output_fn: OutputFn,
    now_fn: Callable[[], dt.datetime],
    days: int = FRESH_WINDOW_DAYS,
) -> RunSummary:
    """Drive the interactive loop. One job at a time: show it, tailor+render,
    open URL+PDF, print the paste/answer-yourself guidance, then wait for
    next/skip/stop. Only `next` marks the job applied. `stop` ends cleanly."""
    applied = 0
    skipped = 0
    n = len(candidates)

    for i, job in enumerate(candidates, start=1):
        output_fn("")
        output_fn(f"=== Application {i} of {n} ===")
        output_fn(f"  {job.company} — {job.title}")
        output_fn(
            f"  score {_fmt_score(job)}   posted {_fmt_posted(job)}   {job.id}"
        )
        output_fn(f"  apply: {job.url}")

        prepared = await tailor_fn(session, job)
        thin = "  ⚠ THIN MATCH (weak fit)" if prepared.thin else ""
        output_fn(f"  resume PDF: {prepared.pdf_path}{thin}")
        open_fn(job.url, str(prepared.pdf_path) if prepared.pdf_path else None)

        output_fn("")
        for ln in paste_block(profile):
            output_fn(ln)
        output_fn("")
        for ln in manual_fields_block():
            output_fn(ln)

        output_fn("")
        while True:
            cmd = input_fn(
                f"[{i}/{n}] next = I submitted this  ·  skip  ·  stop > "
            ).strip().lower()
            if cmd == CMD_NEXT:
                await mark_applied(session, job, now=now_fn())
                applied += 1
                output_fn(f"  ✓ marked applied: {job.id}")
                break
            if cmd == CMD_SKIP:
                skipped += 1
                output_fn(f"  → skipped (still queued): {job.id}")
                break
            if cmd == CMD_STOP:
                output_fn("  ■ stopping.")
                summary = RunSummary(
                    applied, skipped, total_qualified - applied, days
                )
                _print_summary(output_fn, summary)
                return summary
            output_fn("  ? unrecognized — type exactly: next, skip, or stop")

    summary = RunSummary(applied, skipped, total_qualified - applied, days)
    _print_summary(output_fn, summary)
    return summary


def _print_summary(output_fn: OutputFn, summary: RunSummary) -> None:
    output_fn("")
    output_fn(f"run complete: {summary.line()}")
