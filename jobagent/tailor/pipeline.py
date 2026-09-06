"""Tailor + render one queued job into a one-page resume (md + PDF).

Extracts the exact sequence the `tailor` CLI runs (de-boilerplate siblings ->
JobContext -> generate verified bullets -> generate+verify summary -> render) so
the interactive `run` loop can reuse it. Every emitted bullet already passed the
verifier; nothing here submits or auto-fills anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Job
from ..factbank import FactBank
from ..llm.base import BaseLLMProvider
from ..scoring.embedder import Embedder
from ..text_clean import deboilerplate_by_company
from .generate import (
    JobContext,
    TailorConfig,
    _job_keywords,
    generate_bullets,
    generate_summary,
    select_candidates,
)
from .render import Bullet, RenderedResume, render_resume, with_experience


@dataclass(frozen=True)
class PreparedResume:
    """What `run` needs to hand the human for one job."""

    job_id: str
    md_path: Path | None
    pdf_path: Path | None
    bullet_count: int
    thin: bool


async def tailor_and_render(
    session: AsyncSession,
    provider: BaseLLMProvider,
    embedder: Embedder,
    fact_bank: FactBank,
    job: Job,
    out_dir: str | Path,
    config: TailorConfig,
) -> PreparedResume:
    """Tailor + render one job's resume. Raises if no bullet survives the verifier
    (a resume is never rendered padded or unverified)."""
    # De-boilerplate against the company's other postings so `select` embeds the
    # role, not shared "About <company>" text (same as the tailor CLI).
    siblings = list(
        (
            await session.execute(
                select(Job.id, Job.company, Job.description_text).where(
                    Job.company == job.company
                )
            )
        ).all()
    )
    clean = deboilerplate_by_company(
        [(r.id, r.company, r.description_text) for r in siblings]
    )
    ctx = JobContext(
        id=job.id,
        title=job.title,
        description_text=clean.get(job.id, job.description_text),
        score_breakdown=job.score_breakdown,
    )

    result = await generate_bullets(provider, embedder, fact_bank, ctx, config)
    if not result.bullets:
        raise ValueError(f"no verified bullets for {job.id}; cannot render")

    summary_res = await generate_summary(provider, embedder, fact_bank, ctx, config)
    relevance = {
        c.fact.id: c.score for c in select_candidates(ctx, fact_bank, embedder, config)
    }
    tailored = [Bullet(b.text, b.fact_id, b.rank_score) for b in result.bullets]
    rendered: RenderedResume = render_resume(
        job.id,
        job.company,
        job.title,
        with_experience(tailored, fact_bank),
        fact_bank,
        out_dir,
        summary_text=summary_res.text,
        relevance=relevance,
        job_keywords=_job_keywords(ctx),
        thin=result.report.floor_fallback,
        min_bullets=config.min_bullets,
    )
    return PreparedResume(
        job_id=job.id,
        md_path=rendered.md_path,
        pdf_path=rendered.pdf_path,
        bullet_count=rendered.bullet_count,
        thin=rendered.thin,
    )
