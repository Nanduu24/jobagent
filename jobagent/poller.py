"""Poll orchestration: fetch -> filter (relevance + sponsorship) -> upsert.

Boards run concurrently; per-company and per-board timeouts guarantee no single
board can stall the run. Per-host rate limiting (in HttpClient) still serializes
requests to the same host, so concurrency is safe and polite.
"""
from __future__ import annotations

import asyncio
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .config import get_settings
from .db import repository
from .db.enums import JobStatus
from .db.schemas import Job as JobIn
from .filters.dedupe import REASON_DUPLICATE, find_clusters
from .filters.freshness import REASON_STALE, age_in_days, is_stale
from .filters.relevance import (
    REASON_IRRELEVANT_TITLE,
    REASON_TOO_SENIOR,
    REASON_WRONG_LOCATION,
    classify_relevance,
)
from .filters.sponsorship import classify_sponsorship
from .ingest.base import HttpClient
from .ingest.companies import CompanyConfig
from .ingest.registry import build_adapter
from .logging import get_logger

log = get_logger(__name__)

REASON_SPONSORSHIP_BLOCKED = "sponsorship_blocked"


@dataclass
class PollSummary:
    """Aggregate counters for a single poll run.

    The cut buckets are mutually exclusive (one reason per job, by precedence),
    so they form a funnel: fetched == stale + irrelevant_title + too_senior +
    wrong_location + sponsorship_blocked + survivors.
    """

    fetched: int = 0
    # funnel cut buckets (one per job, by precedence)
    duplicate: int = 0
    stale: int = 0
    irrelevant_title: int = 0
    too_senior: int = 0
    wrong_location: int = 0
    sponsorship_blocked: int = 0
    survivors: int = 0
    # newly inserted rows (for idempotency reporting; orthogonal to buckets)
    new: int = 0
    # freshness bookkeeping (independent of the bucket precedence)
    stale_mode: str = "keep"
    old: int = 0  # all fetched rows older than max_age_days
    survivors_stale: int = 0  # survivors that are older than max_age_days
    # sponsorship split over every fetched row (informational)
    sponsorship_true: int = 0
    sponsorship_false: int = 0
    sponsorship_none: int = 0
    errors: list[str] = field(default_factory=list)

    _BUCKETS = {
        REASON_DUPLICATE: "duplicate",
        REASON_STALE: "stale",
        REASON_IRRELEVANT_TITLE: "irrelevant_title",
        REASON_TOO_SENIOR: "too_senior",
        REASON_WRONG_LOCATION: "wrong_location",
        REASON_SPONSORSHIP_BLOCKED: "sponsorship_blocked",
    }

    def record(self, reason: str | None) -> None:
        """Tally one job into its funnel bucket (None == survivor)."""
        if reason is None:
            self.survivors += 1
        else:
            setattr(self, self._BUCKETS[reason], getattr(self, self._BUCKETS[reason]) + 1)

    @property
    def survivors_if_cut(self) -> int:
        """Survivors under STALE_MODE=cut (valid from a keep-mode run): drop the
        survivors that are older than max_age_days."""
        return self.survivors - self.survivors_stale

    def line(self) -> str:
        return (
            f"{self.fetched} fetched -> {self.duplicate} duplicate -> "
            f"{self.stale} stale -> {self.irrelevant_title} irrelevant_title -> "
            f"{self.too_senior} too_senior -> {self.wrong_location} wrong_location -> "
            f"{self.sponsorship_blocked} sponsorship_blocked -> "
            f"{self.survivors} SURVIVORS "
            f"[stale_mode={self.stale_mode}; {self.old} older-than-max-age "
            f"({self.survivors_stale} of them survivors; survivors_if_cut="
            f"{self.survivors_if_cut}); {self.new} new; sponsorship "
            f"True={self.sponsorship_true}/False={self.sponsorship_false}/"
            f"None={self.sponsorship_none}]"
        )


def _bucket_reason(
    job: JobIn,
    sponsorship: bool | None,
    now: dt.datetime,
    max_age_days: int,
    stale_mode: str,
) -> str | None:
    """Assign one funnel reason by precedence, or None for a survivor.

    Precedence: stale (only when stale_mode='cut') -> relevance
    (too_senior/irrelevant_title/wrong_location) -> sponsorship_blocked. Unknown
    sponsorship (None) is not a rejection; a posting still listed on a live board
    is evidence it is still open, so 'keep' mode never cuts on age.
    """
    if stale_mode == "cut" and is_stale(job.posted_at, now, max_age_days):
        return REASON_STALE
    relevance = classify_relevance(job.title, job.location, job.remote_type)
    if not relevance.relevant:
        return relevance.reason
    if sponsorship is False:
        return REASON_SPONSORSHIP_BLOCKED
    return None


async def _process_job(
    session: AsyncSession,
    job: JobIn,
    summary: PollSummary,
    now: dt.datetime,
    max_age_days: int,
    stale_mode: str,
) -> None:
    """Sponsorship-classify, upsert, and bucket ONE canonical job."""
    sponsorship = classify_sponsorship(job.description_text)
    job = job.model_copy(update={"sponsorship_ok": sponsorship})
    model, is_new = await repository.upsert_job(session, job, now=now)
    if is_new:
        summary.new += 1

    model.age_days = age_in_days(job.posted_at, now)
    old = is_stale(job.posted_at, now, max_age_days)
    if old:
        summary.old += 1

    if sponsorship is True:
        summary.sponsorship_true += 1
    elif sponsorship is False:
        summary.sponsorship_false += 1
    else:
        summary.sponsorship_none += 1

    reason = _bucket_reason(job, sponsorship, now, max_age_days, stale_mode)
    summary.record(reason)

    if reason is None:
        if is_new:
            model.status = JobStatus.new
        model.filter_reason = None
        if old:
            summary.survivors_stale += 1
    else:
        model.status = JobStatus.filtered
        model.filter_reason = reason


async def _poll_company(
    session: AsyncSession,
    company: CompanyConfig,
    http: HttpClient,
    summary: PollSummary,
    now: dt.datetime,
    max_age_days: int,
    stale_mode: str,
) -> None:
    adapter = build_adapter(company.source, http)
    fetched_jobs = await adapter.fetch(company.token)
    summary.fetched += len(fetched_jobs)

    # Enrich with the display company name, then DEDUPE FIRST (before
    # relevance/freshness/sponsorship) so the canonical is a stable, filter-
    # independent representative and its shadows can't be cut by a different
    # filter into becoming the representative.
    named = [j.model_copy(update={"company": company.name}) for j in fetched_jobs]
    for cluster in find_clusters(named):
        canonical = cluster.canonical.model_copy(
            update={"locations": cluster.locations}
        )
        await _process_job(session, canonical, summary, now, max_age_days, stale_mode)
        for shadow in cluster.shadows:
            model, is_new = await repository.upsert_job(session, shadow, now=now)
            if is_new:
                summary.new += 1
            model.age_days = age_in_days(shadow.posted_at, now)
            model.status = JobStatus.filtered
            model.filter_reason = REASON_DUPLICATE
            summary.record(REASON_DUPLICATE)

    log.info(
        "poll.company",
        company=company.name,
        source=company.source,
        token=company.token,
        jobs=len(fetched_jobs),
    )


async def _poll_board(
    source: str,
    companies: list[CompanyConfig],
    session_factory: async_sessionmaker[AsyncSession],
    http: HttpClient,
    summary: PollSummary,
    now: dt.datetime,
    company_timeout: float,
    max_age_days: int,
    stale_mode: str,
) -> None:
    """Poll all companies for one board. Per-company timeout + isolation so one
    bad company never stalls the board."""
    for company in companies:
        try:
            async with session_factory() as session:
                await asyncio.wait_for(
                    _poll_company(
                        session, company, http, summary, now, max_age_days, stale_mode
                    ),
                    timeout=company_timeout,
                )
                await session.commit()
        except (TimeoutError, Exception) as exc:  # noqa: BLE001
            kind = "timeout" if isinstance(exc, asyncio.TimeoutError) else "error"
            msg = f"{company.name} ({company.source}:{company.token}): {kind}: {exc}"
            summary.errors.append(msg)
            log.error(
                "poll.company.failed",
                company=company.name,
                source=company.source,
                kind=kind,
                error=str(exc),
            )


async def run_poll(
    companies: list[CompanyConfig],
    session_factory: async_sessionmaker[AsyncSession],
    *,
    http: HttpClient | None = None,
    now: dt.datetime | None = None,
    company_timeout: float | None = None,
    board_timeout: float | None = None,
) -> PollSummary:
    """Poll every company (boards concurrently), filter, and upsert.

    No single board or company can stall the run: each company has a timeout and
    is isolated, each board has a timeout, and boards run concurrently with
    ``return_exceptions`` so a failing board is logged and skipped.
    """
    settings = get_settings()
    ts = now or dt.datetime.now(dt.UTC)
    c_timeout = company_timeout or settings.poll_company_timeout
    b_timeout = board_timeout or settings.poll_board_timeout
    max_age_days = settings.max_age_days
    stale_mode = settings.stale_mode

    summary = PollSummary(stale_mode=stale_mode)
    owns_http = http is None
    client = http or HttpClient()

    by_source: dict[str, list[CompanyConfig]] = defaultdict(list)
    for company in companies:
        by_source[company.source].append(company)

    try:
        sources = list(by_source)
        board_coros = [
            asyncio.wait_for(
                _poll_board(
                    src,
                    by_source[src],
                    session_factory,
                    client,
                    summary,
                    ts,
                    c_timeout,
                    max_age_days,
                    stale_mode,
                ),
                timeout=b_timeout,
            )
            for src in sources
        ]
        results = await asyncio.gather(*board_coros, return_exceptions=True)
        for src, result in zip(sources, results, strict=True):
            if isinstance(result, BaseException):
                kind = (
                    "timeout"
                    if isinstance(result, asyncio.TimeoutError)
                    else "error"
                )
                summary.errors.append(f"board {src}: {kind}: {result}")
                log.error("poll.board.failed", board=src, kind=kind, error=str(result))
    finally:
        if owns_http:
            await client.aclose()

    log.info("poll.summary", summary=summary.line(), errors=len(summary.errors))
    return summary
