"""Typer CLI for JobAgent (Phases 1-2)."""
from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

import typer
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db.enums import JobStatus
from .db.models import Job
from .db.session import create_all, create_engine, create_session_factory
from .factbank import hash_fact_bank, load_fact_bank
from .ingest.companies import load_companies
from .llm.base import LLMError
from .llm.factory import build_provider
from .llm.fake import ScriptedProvider
from .logging import configure_logging, get_logger
from .onboarding import build_fact_bank, write_fact_bank
from .poller import run_poll
from .runner import (
    FRESH_WINDOW_DAYS,
    fetch_run_candidates,
    open_url_and_pdf,
    render_candidate_table,
    run_loop,
)
from .scoring.embedder import MiniLMEmbedder
from .scoring.scorer import ScoreConfig, estimate_run, score_new_jobs
from .text_clean import deboilerplate_by_company
from .tailor.generate import (
    JobContext,
    TailorConfig,
    _job_keywords,
    default_pool,
    generate_bullets,
    generate_summary,
    select_candidates,
)
from .tailor.pipeline import PreparedResume, tailor_and_render
from .tailor.render import Bullet, render_resume, with_experience

app = typer.Typer(
    add_completion=False,
    help="JobAgent — ingest, filter, persist, and score job postings.",
)
log = get_logger("jobagent.cli")


class _TyperPrompter:
    """Terminal-backed Prompter for `jobagent setup` (see onboarding.Prompter)."""

    def text(self, message: str, *, default: str | None = None,
             allow_empty: bool = False) -> str:
        if allow_empty:
            return str(typer.prompt(message, default="", show_default=False)).strip()
        while True:
            val = (
                typer.prompt(message, default=default)
                if default is not None
                else typer.prompt(message)
            )
            if str(val).strip():
                return str(val).strip()
            typer.echo("  (this one is required)")

    def boolean(self, message: str, *, default: bool = False) -> bool:
        return bool(typer.confirm(message, default=default))

    def integer(self, message: str, *, default: int = 0) -> int:
        return int(typer.prompt(message, default=default, type=int))

    def note(self, message: str) -> None:
        typer.echo(message)


@app.command("setup")
def setup(
    out: str = typer.Option(
        "data/fact_bank.json", "--out", help="Where to write your fact bank."
    ),
) -> None:
    """Create your personal fact bank interactively (run this first).

    Walks you through your profile, education, experience, projects, skills, and
    verified accomplishments, then writes data/fact_bank.json — the source of
    truth the tool tailors resumes from. Your data stays local (git-ignored) and
    is never uploaded. Prefer editing JSON? Copy data/fact_bank.example.json.
    """
    path = Path(out)
    if path.exists() and not typer.confirm(
        f"{path} already exists — overwrite it?", default=False
    ):
        typer.echo("Aborted; your existing fact bank is unchanged.")
        raise typer.Exit()

    data = build_fact_bank(_TyperPrompter())
    try:
        fact_bank = write_fact_bank(data, path)
    except ValidationError as exc:
        typer.echo(f"\n! That fact bank isn't valid yet:\n{exc}")
        typer.echo("Nothing was written. Re-run `jobagent setup` and try again.")
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"\n✓ Wrote {path} — {len(fact_bank.facts)} fact(s) for "
        f"{fact_bank.profile.name}."
    )
    typer.echo("Next: `jobagent poll` → `jobagent score` → `jobagent run`.")


@app.command("init-db")
def init_db() -> None:
    """Create all tables directly (dev convenience; prod uses Alembic)."""
    settings = get_settings()
    configure_logging(settings.log_level)

    async def _run() -> None:
        engine = create_engine()
        try:
            await create_all(engine)
        finally:
            await engine.dispose()

    asyncio.run(_run())
    typer.echo("init-db: schema created")


@app.command("poll")
def poll(
    companies_file: str = typer.Option(
        None,
        "--companies-file",
        "-c",
        help="Path to companies.yaml (defaults to configured path).",
    ),
) -> None:
    """Fetch all companies, filter by sponsorship, upsert, and log a summary."""
    settings = get_settings()
    configure_logging(settings.log_level)
    path = companies_file or settings.companies_file
    companies = load_companies(path)

    # Workable is quarantined by default (see Settings.enable_workable).
    if not settings.enable_workable:
        skipped = [c for c in companies if c.source == "workable"]
        companies = [c for c in companies if c.source != "workable"]
        if skipped:
            typer.echo(
                f"skipping {len(skipped)} Workable companies "
                f"(set ENABLE_WORKABLE=true to include them)"
            )

    async def _run() -> None:
        engine = create_engine()
        factory = create_session_factory(engine)
        try:
            summary = await run_poll(companies, factory)
        finally:
            await engine.dispose()
        typer.echo(f"poll complete: {summary.line()}")
        if summary.errors:
            typer.echo(f"  ({len(summary.errors)} companies errored)")
            for err in summary.errors:
                typer.echo(f"    - {err}")

    asyncio.run(_run())


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _model_and_rpm(settings) -> tuple[str, float]:  # type: ignore[no-untyped-def]
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        rpm = 60.0 / settings.anthropic_min_interval if settings.anthropic_min_interval else 60.0
        return settings.anthropic_model, rpm
    if provider == "gemini":
        rpm = 60.0 / settings.gemini_min_interval if settings.gemini_min_interval else 60.0
        return settings.gemini_model, rpm
    return settings.groq_model, 60.0


@app.command("score")
def score(
    fact_bank_file: str = typer.Option(
        "data/fact_bank.json", "--fact-bank", help="Path to the fact bank."
    ),
    limit: int = typer.Option(30, "--limit", "-n", help="Rows in the ranked table."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Estimate the run (jobs/calls/tokens/cost/time) and STOP. No LLM calls."
    ),
) -> None:
    """Score every status='new' survivor and print a ranked, explainable queue."""
    settings = get_settings()
    configure_logging(settings.log_level)

    # Malformed fact bank is a HARD FAIL (validation error propagates).
    fact_bank = load_fact_bank(fact_bank_file)
    fb_hash = hash_fact_bank(fact_bank_file)

    embedder = MiniLMEmbedder(settings.embedding_model)
    config = ScoreConfig(
        top_n_llm=settings.top_n_llm,
        score_threshold=settings.score_threshold,
        max_years=settings.hard_req_max_years,
        embedding_model=settings.embedding_model,
        cache_dir=settings.cache_dir,
        max_per_company_in_window=settings.max_per_company_in_window,
        max_llm_calls=settings.max_llm_calls_per_run,
        max_spend_usd=settings.max_spend_usd_per_run,
        stack_match_threshold=settings.stack_match_threshold,
        stack_gate_min_missing=settings.stack_gate_min_missing,
        weights={
            "cheap": settings.weight_cheap,
            "skill": settings.weight_skill,
            "seniority": settings.weight_seniority,
            "domain": settings.weight_domain,
        },
    )

    # --- DRY RUN: estimate only, zero spend, no provider needed ----------
    if dry_run:
        model, rpm = _model_and_rpm(settings)

        async def _estimate() -> None:
            engine = create_engine()
            factory = create_session_factory(engine)
            try:
                async with factory() as session:
                    plan = await estimate_run(
                        session, embedder, fact_bank, fb_hash, config,
                        model=model, rpm=rpm,
                    )
            finally:
                await engine.dispose()
            typer.echo(plan.render())
            typer.echo(
                f"  budget guard       : MAX_LLM_CALLS_PER_RUN={config.max_llm_calls}, "
                f"MAX_SPEND_USD_PER_RUN=${config.max_spend_usd:.2f}"
            )

        asyncio.run(_estimate())
        return

    # Provider from env; without a key, fall back to Stage A only.
    try:
        provider = build_provider(settings)
    except LLMError as exc:
        typer.echo(f"! LLM provider unavailable ({exc}). Running Stage A only.")
        provider = ScriptedProvider(lambda _m: "")
        config.top_n_llm = 0

    async def _run() -> None:
        engine = create_engine()
        factory = create_session_factory(engine)
        try:
            async with factory() as session:
                report = await score_new_jobs(
                    session, provider, embedder, fact_bank, fb_hash, config
                )
            async with factory() as session:
                rows = list(
                    (
                        await session.scalars(
                            select(Job)
                            .where(Job.status.in_([JobStatus.new, JobStatus.queued]))
                            .order_by(Job.match_score.desc().nulls_last())
                            .limit(limit)
                        )
                    ).all()
                )
        finally:
            await engine.dispose()

        typer.echo("")
        typer.echo(
            f"{'score':>5}  {'st':<6} {'company':<14} {'title':<34} "
            f"{'why (top-2)':<44} {'missing':<22} age"
        )
        for j in rows:
            bd = j.score_breakdown or {}
            justs = []
            rubric = bd.get("rubric") or {}
            for axis in ("skill_overlap", "seniority_fit", "domain_fit"):
                a = rubric.get(axis)
                if a:
                    justs.append(f"{axis.split('_')[0]}:{a['justification']}")
            why = _truncate(" | ".join(justs[:2]) or "(stage A only)", 44)
            missing = _truncate(", ".join(bd.get("missing_requirements", [])) or "-", 22)
            age = j.age_days if j.age_days is not None else "-"
            st = "QUEUED" if j.status == JobStatus.queued else "new"
            score_str = f"{j.match_score:.1f}" if j.match_score is not None else "-"
            typer.echo(
                f"{score_str:>5}  {st:<6} {_truncate(j.company,14):<14} "
                f"{_truncate(j.title,34):<34} {why:<44} {missing:<22} {age}"
            )
        typer.echo("")
        typer.echo(f"score complete: {report.line()}")
        if report.gate_reasons:
            typer.echo(f"  gate reasons: {report.gate_reasons}")
        for cost_line in provider.usage.by_model_lines():
            typer.echo(f"  cost: {cost_line}")
        if report.errors:
            typer.echo(f"  ! {len(report.errors)} jobs errored (Stage B halted early)")

    asyncio.run(_run())


@app.command("tailor")
def tailor(
    job_id: str = typer.Option(
        None, "--job", "-j", help="Job id to tailor (default: highest-scored queued job)."
    ),
    fact_bank_file: str = typer.Option(
        "data/fact_bank.json", "--fact-bank", help="Path to the fact bank."
    ),
    out_dir: str = typer.Option(
        None, "--out-dir", help="If set, render a one-page resume (markdown + PDF) here."
    ),
) -> None:
    """Generate verified resume bullets for one queued job (select→rewrite→verify→render).

    Every emitted bullet PASSED the verifier against the one fact it drew from;
    fabricated bullets are dropped, never shipped. With --out-dir, also renders a
    one-page resume (markdown + PDF) for HUMAN review — it NEVER auto-submits.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    fact_bank = load_fact_bank(fact_bank_file)
    embedder = MiniLMEmbedder(settings.embedding_model)
    config = TailorConfig(
        max_bullets=settings.tailor_max_bullets,
        candidate_pool=default_pool(settings.tailor_max_bullets),
        max_llm_calls=settings.max_llm_calls_per_run,
        max_spend_usd=settings.max_spend_usd_per_run,
        min_relevance=settings.tailor_min_relevance,
        min_bullets=settings.tailor_min_bullets,
    )

    try:
        provider = build_provider(settings)
    except LLMError as exc:
        typer.echo(f"! LLM provider unavailable ({exc}); cannot tailor.")
        raise typer.Exit(code=1) from exc

    async def _run() -> None:
        engine = create_engine()
        factory = create_session_factory(engine)
        try:
            async with factory() as session:
                if job_id:
                    job = await session.get(Job, job_id)
                else:
                    job = (
                        await session.scalars(
                            select(Job)
                            .where(Job.status == JobStatus.queued)
                            .order_by(Job.match_score.desc().nulls_last())
                            .limit(1)
                        )
                    ).first()
                if job is None:
                    typer.echo("no matching job (need a queued job; run `score` first).")
                    return
                # De-boilerplate against the company's other postings so the
                # generator's select step embeds the role, not shared boilerplate.
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
                summary_res = (
                    await generate_summary(provider, embedder, fact_bank, ctx, config)
                    if out_dir and result.bullets
                    else None
                )
                relevance = {
                    c.fact.id: c.score
                    for c in select_candidates(ctx, fact_bank, embedder, config)
                }
                job_kw = _job_keywords(ctx)
        finally:
            await engine.dispose()

        typer.echo("")
        typer.echo(f"Tailored resume bullets for {job.company} — {job.title}")
        typer.echo(f"({job.id}, match_score={job.match_score})")
        typer.echo("")
        for b in result.bullets:
            regen = " [regenerated]" if b.regenerated else ""
            typer.echo(f"  • {b.text}")
            typer.echo(f"      ↳ fact: {b.fact_id} (relevance {b.rank_score:.3f}){regen}")
        rep = result.report
        typer.echo("")
        typer.echo(
            f"  {rep.emitted} bullets (cap {config.max_bullets}); "
            f"{rep.cleared_floor}/{rep.candidates} facts cleared floor "
            f"{config.min_relevance:.2f}"
            + (
                f" — FELL BACK to top-{config.min_bullets} (too few cleared)"
                if rep.floor_fallback
                else ""
            )
        )
        if result.report.dropped_bullets:
            typer.echo("")
            typer.echo(f"  dropped {result.report.dropped} (never shipped):")
            for d in result.report.dropped_bullets:
                typer.echo(f"    ✗ {d.fact_id}: {', '.join(d.reasons)}")
        typer.echo("")
        typer.echo(f"tailor complete: {result.report.line()}")
        for cost_line in provider.usage.by_model_lines():
            typer.echo(f"  cost: {cost_line}")

        if out_dir and result.bullets and summary_res is not None:
            typer.echo("")
            fb_note = " [FELL BACK to skeleton]" if summary_res.fell_back else (
                " [regenerated]" if summary_res.regenerated else "")
            typer.echo(f"SUMMARY (verify ok={summary_res.verify.ok}){fb_note}:")
            typer.echo(f"  {summary_res.text}")
            tailored = [Bullet(b.text, b.fact_id, b.rank_score) for b in result.bullets]
            rendered = render_resume(
                job.id, job.company, job.title,
                with_experience(tailored, fact_bank),
                fact_bank, out_dir,
                summary_text=summary_res.text,
                relevance=relevance,
                job_keywords=job_kw,
                thin=result.report.floor_fallback,
                min_bullets=settings.tailor_min_bullets,
            )
            typer.echo("")
            thin = "  ⚠ THIN MATCH (weak fit — few facts cleared the floor)" if rendered.thin else ""
            typer.echo(f"rendered resume ({rendered.bullet_count} bullets){thin}")
            typer.echo(f"  markdown: {rendered.md_path}")
            typer.echo(f"  pdf     : {rendered.pdf_path}")
            typer.echo("  (for HUMAN review — the agent never auto-submits)")

    asyncio.run(_run())


async def _refresh_queue(fact_bank_file: str, log_path: str) -> None:
    """`--fresh`: re-poll + re-score (the normal ~7-min pipeline) so the run works
    the newest queue. Progress is teed to ``log_path``; the queue is read after."""
    settings = get_settings()
    fact_bank = load_fact_bank(fact_bank_file)
    fb_hash = hash_fact_bank(fact_bank_file)
    embedder = MiniLMEmbedder(settings.embedding_model)

    companies = load_companies(settings.companies_file)
    if not settings.enable_workable:
        companies = [c for c in companies if c.source != "workable"]

    config = ScoreConfig(
        top_n_llm=settings.top_n_llm,
        score_threshold=settings.score_threshold,
        max_years=settings.hard_req_max_years,
        embedding_model=settings.embedding_model,
        cache_dir=settings.cache_dir,
        max_per_company_in_window=settings.max_per_company_in_window,
        max_llm_calls=settings.max_llm_calls_per_run,
        max_spend_usd=settings.max_spend_usd_per_run,
        stack_match_threshold=settings.stack_match_threshold,
        stack_gate_min_missing=settings.stack_gate_min_missing,
        weights={
            "cheap": settings.weight_cheap,
            "skill": settings.weight_skill,
            "seniority": settings.weight_seniority,
            "domain": settings.weight_domain,
        },
    )

    with open(log_path, "a", encoding="utf-8") as fh:
        def _tee(msg: str) -> None:
            typer.echo(msg)
            fh.write(msg + "\n")
            fh.flush()

        engine = create_engine()
        factory = create_session_factory(engine)
        try:
            _tee("--fresh: polling boards…")
            poll_summary = await run_poll(companies, factory)
            _tee(f"  poll: {poll_summary.line()}")
            _tee("--fresh: scoring survivors (this is the slow part)…")
            provider = build_provider(settings)
            async with factory() as session:
                report = await score_new_jobs(
                    session, provider, embedder, fact_bank, fb_hash, config
                )
            _tee(f"  score: {report.line()}")
        finally:
            await engine.dispose()


@app.command("run")
def run(
    number: int = typer.Option(
        None, "--number", "-n", help="How many jobs to apply to (else prompt)."
    ),
    fresh: bool = typer.Option(
        False, "--fresh",
        help="Re-poll + re-score first (~7 min) before selecting. Else reuse the queue.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the candidate list `run` would build and STOP (no tailoring, no LLM).",
    ),
    days: int = typer.Option(
        FRESH_WINDOW_DAYS, "--days",
        help="Freshness window: only surface jobs posted within N days (newest-first).",
    ),
    fact_bank_file: str = typer.Option(
        "data/fact_bank.json", "--fact-bank", help="Path to the fact bank."
    ),
) -> None:
    """Interactive apply loop (Step 1): open + prepare each job for the human.

    For each candidate it tailors + renders the resume, opens the application URL
    and the PDF, and prints what to paste plus what you must answer yourself. It
    NEVER fills a form and NEVER submits — `next` just records that YOU submitted.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    n = number if number is not None else typer.prompt(
        "How many jobs do you want to apply to?", type=int
    )
    if n <= 0:
        typer.echo("nothing to do (asked for 0 jobs).")
        return

    if fresh:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"renders/run_fresh_{stamp}.log"
        typer.echo(
            f"--fresh: re-polling + re-scoring first — this takes a few minutes. "
            f"Logging to {log_path}"
        )
        asyncio.run(_refresh_queue(fact_bank_file, log_path))

    fact_bank = load_fact_bank(fact_bank_file)
    now = dt.datetime.now(dt.timezone.utc)

    # --- DRY RUN: show the selection and STOP (no tailoring, no LLM) --------
    if dry_run:
        async def _plan() -> None:
            engine = create_engine()
            factory = create_session_factory(engine)
            try:
                async with factory() as session:
                    candidates, total = await fetch_run_candidates(
                        session, n, now=now, days=days
                    )
            finally:
                await engine.dispose()
            typer.echo("")
            for line in render_candidate_table(candidates, total, n, days):
                typer.echo(line)

        asyncio.run(_plan())
        return

    # --- LIVE RUN: tailor + open + prepare, one job at a time --------------
    embedder = MiniLMEmbedder(settings.embedding_model)
    config = TailorConfig(
        max_bullets=settings.tailor_max_bullets,
        candidate_pool=default_pool(settings.tailor_max_bullets),
        max_llm_calls=settings.max_llm_calls_per_run,
        max_spend_usd=settings.max_spend_usd_per_run,
        min_relevance=settings.tailor_min_relevance,
        min_bullets=settings.tailor_min_bullets,
    )
    try:
        provider = build_provider(settings)
    except LLMError as exc:
        typer.echo(f"! LLM provider unavailable ({exc}); cannot tailor.")
        raise typer.Exit(code=1) from exc

    async def _run() -> None:
        engine = create_engine()
        factory = create_session_factory(engine)
        try:
            async with factory() as session:
                candidates, total = await fetch_run_candidates(
                    session, n, now=now, days=days
                )
                if not candidates:
                    typer.echo(
                        f"no candidates (need queued jobs posted within "
                        f"{days} days; run `jobagent run --fresh` or `score`)."
                    )
                    return
                if total < n:
                    typer.echo(
                        f"only {total} job(s) qualify (fewer than {n}) — "
                        "proceeding with what's available."
                    )

                async def _tailor(sess: AsyncSession, job: Job) -> PreparedResume:
                    return await tailor_and_render(
                        sess, provider, embedder, fact_bank, job,
                        "renders", config,
                    )

                await run_loop(
                    session,
                    candidates,
                    total,
                    fact_bank.profile,
                    tailor_fn=_tailor,
                    open_fn=open_url_and_pdf,
                    input_fn=lambda prompt: typer.prompt(prompt, default="", show_default=False),
                    output_fn=typer.echo,
                    now_fn=lambda: dt.datetime.now(dt.timezone.utc),
                    days=days,
                )
        finally:
            await engine.dispose()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
