"""Two-stage scoring orchestration.

Stage A (local, free): embed each survivor once, rank all by cosine to the
candidate vector. Stage B (LLM, top-N only): extract requirements, gate hard
blockers, rubric-score, blend into match_score, promote over threshold.

Cost discipline: the LLM is called only for the top-N by cheap_score, and NEVER
for a job that already has a Stage B breakdown (idempotent, cache-backed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from ..db.enums import JobStatus
from ..db.models import Job
from ..factbank import FactBank
from ..llm.base import BaseLLMProvider
from ..llm.pricing import cost_usd as pricing_cost
from ..logging import get_logger
from ..text_clean import deboilerplate_by_company
from .embedder import Embedder
from .factbank_cache import load_or_build_candidate_vector
from .gate import SkillMatcher, check_hard_requirements
from .schemas import Rubric, ScoreBreakdown
from .stage_b import extract_requirements, score_rubric
from .vectors import cosine, text_hash, to_vector

log = get_logger(__name__)


@dataclass
class ScoreConfig:
    top_n_llm: int
    score_threshold: float
    max_years: int
    embedding_model: str
    cache_dir: str
    weights: dict[str, float]
    max_per_company_in_window: int = 1_000_000  # effectively uncapped by default
    max_llm_calls: int = 1_000_000  # per-run budget guard
    max_spend_usd: float = 1_000_000.0  # per-run budget guard
    stack_match_threshold: float = 0.60  # semantic stack-gate similarity floor
    stack_gate_min_missing: int = 3  # gate fires at >= this many unmatched skills


def select_window(
    ranked: list[tuple[Job, float]], top_n: int, max_per_company: int
) -> set[int]:
    """Choose the Stage B window: fill by cheap_score, but cap any one company's
    share. Overflow (a company already at its cap) is skipped and waits for a
    later window. Window-selection only — scores/rubric are untouched.
    """
    window: set[int] = set()
    per_company: dict[str, int] = {}
    for job, _ in ranked:  # ranked is sorted by cheap_score desc
        if len(window) >= top_n:
            break
        if per_company.get(job.company, 0) >= max_per_company:
            continue
        window.add(id(job))
        per_company[job.company] = per_company.get(job.company, 0) + 1
    return window


@dataclass
class ScoreReport:
    total: int = 0
    stage_a_only: int = 0
    stage_b_scored: int = 0
    reused: int = 0  # already-scored, no LLM call
    gated: int = 0
    queued: int = 0
    gate_reasons: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    budget_tripped: str | None = None  # "calls" | "spend" if the guard aborted
    errors: list[str] = field(default_factory=list)

    @property
    def retry_rate(self) -> float:
        return self.retries / self.llm_calls if self.llm_calls else 0.0

    def line(self) -> str:
        budget = f" | BUDGET TRIPPED ({self.budget_tripped})" if self.budget_tripped else ""
        return (
            f"{self.total} survivors / {self.stage_b_scored} stage-B "
            f"({self.reused} reused) / {self.gated} gated / {self.queued} queued "
            f"| LLM: {self.llm_calls} calls, {self.retries} retries "
            f"({self.retry_rate:.1%}), {self.input_tokens}+{self.output_tokens} "
            f"tok, ${self.cost_usd:.4f}{budget}"
        )


def blend_match_score(
    cheap: float, rubric: Rubric | None, weights: dict[str, float]
) -> float:
    """Blend Stage A cosine (0-1) + Stage B rubric (0-10 axes) into 0-100.

    Without a rubric only the cheap term contributes, so a Stage-A-only job can
    never reach the promotion threshold — queuing requires the LLM rubric.
    """
    total = weights["cheap"] * cheap
    if rubric is not None:
        total += weights["skill"] * (rubric.skill_overlap.score / 10.0)
        total += weights["seniority"] * (rubric.seniority_fit.score / 10.0)
        total += weights["domain"] * (rubric.domain_fit.score / 10.0)
    return round(100.0 * total, 1)


def _already_stage_b(job: Job) -> bool:
    bd = job.score_breakdown
    return bool(bd and bd.get("requirements") is not None)


def _survivor_where() -> ColumnElement[bool]:
    """Scoring survivors: true survivors (no filter reason, status new/queued)
    OR gate-filtered (req_*, re-scorable). A Phase-1-filtered job (duplicate /
    stale / irrelevant_title / …) is NEVER a survivor even if its status has
    somehow been flipped to 'new' — defends against inconsistent state."""
    return or_(
        and_(
            Job.filter_reason.is_(None),
            Job.status.in_([JobStatus.new, JobStatus.queued]),
        ),
        Job.filter_reason.like("req_%"),
    )


@dataclass
class RunPlan:
    """A zero-spend estimate of what a real run would cost."""

    survivors: int
    window_jobs: int  # jobs in the window needing Stage B (not already scored)
    scorable_jobs: int  # how many fit inside the budget
    est_calls: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float
    est_wall_clock_min: float
    model: str
    rpm: float
    capped_by: str  # "top_n" | "budget_calls" | "budget_spend"

    def render(self) -> str:
        return (
            f"DRY RUN (no LLM calls made)\n"
            f"  provider model     : {self.model}\n"
            f"  survivors ranked   : {self.survivors}\n"
            f"  window (need score): {self.window_jobs}\n"
            f"  scorable in budget : {self.scorable_jobs}  (capped by: {self.capped_by})\n"
            f"  estimated calls    : {self.est_calls}  (~2/job: extract + rubric)\n"
            f"  estimated tokens   : {self.est_input_tokens} in + "
            f"{self.est_output_tokens} out\n"
            f"  estimated cost     : ${self.est_cost_usd:.4f}  (at {self.model} rates)\n"
            f"  estimated wallclock: ~{self.est_wall_clock_min:.1f} min "
            f"(at {self.rpm:.0f} req/min)"
        )


def _est_tokens(messages: list[dict[str, str]]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


async def estimate_run(
    session: AsyncSession,
    embedder: Embedder,
    fact_bank: FactBank,
    fact_bank_hash: str,
    config: ScoreConfig,
    *,
    model: str,
    rpm: float,
) -> RunPlan:
    """Estimate a run without calling the LLM (dry run)."""
    from .prompts import extract_messages, rubric_messages

    candidate = load_or_build_candidate_vector(
        fact_bank, embedder, config.cache_dir, fact_bank_hash
    )
    survivors = list(
        (
            await session.scalars(select(Job).where(_survivor_where()))
        ).all()
    )
    clean = deboilerplate_by_company(
        [(j.id, j.company, j.description_text) for j in survivors]
    )
    ranked: list[tuple[Job, float]] = []
    for job in survivors:
        vec = to_vector(_embedding(job, clean[job.id], embedder, config.embedding_model))
        ranked.append((job, (cosine(vec, candidate) + 1.0) / 2.0))
    ranked.sort(key=lambda t: t[1], reverse=True)
    top_ids = select_window(ranked, config.top_n_llm, config.max_per_company_in_window)

    window = [
        job for job, _ in ranked if id(job) in top_ids and not _already_stage_b(job)
    ]
    # Budget-capped walk: 2 calls/job (extract + rubric), fixed output estimates.
    calls = in_tok = out_tok = 0
    cost = 0.0
    scorable = 0
    capped = "top_n"
    for job in window:
        job_calls = 2
        job_in = _est_tokens(extract_messages(job.title, clean[job.id])) + _est_tokens(
            rubric_messages(job.title, clean[job.id], fact_bank)
        )
        job_out = 100 + 250  # extract ~100, rubric ~250
        job_cost = pricing_cost(model, job_in, job_out)
        if calls + job_calls > config.max_llm_calls:
            capped = "budget_calls"
            break
        if cost + job_cost > config.max_spend_usd:
            capped = "budget_spend"
            break
        calls += job_calls
        in_tok += job_in
        out_tok += job_out
        cost += job_cost
        scorable += 1

    return RunPlan(
        survivors=len(survivors),
        window_jobs=len(window),
        scorable_jobs=scorable,
        est_calls=calls,
        est_input_tokens=in_tok,
        est_output_tokens=out_tok,
        est_cost_usd=round(cost, 4),
        est_wall_clock_min=round(calls / rpm, 1) if rpm else 0.0,
        model=model,
        rpm=rpm,
        capped_by=capped,
    )


def _embedding(job: Job, text: str, embedder: Embedder, model: str) -> list[float]:
    """Return the job's embedding of ``text``, computing + persisting it once.

    ``text`` is the de-boilerplated description (see ``text_clean``); the cache
    key is its hash, so switching to cleaned text invalidates any embedding that
    was computed from the raw description and forces a recompute."""
    h = text_hash(text)
    if (
        job.embedding is not None
        and job.embedding_hash == h
        and job.embedding_model == model
    ):
        return job.embedding
    vec = embedder.encode([text])[0]
    values = [float(x) for x in vec.tolist()]
    job.embedding = values
    job.embedding_hash = h
    job.embedding_model = model
    return values


async def score_new_jobs(
    session: AsyncSession,
    provider: BaseLLMProvider,
    embedder: Embedder,
    fact_bank: FactBank,
    fact_bank_hash: str,
    config: ScoreConfig,
) -> ScoreReport:
    report = ScoreReport()
    candidate = load_or_build_candidate_vector(
        fact_bank, embedder, config.cache_dir, fact_bank_hash
    )
    # Semantic stack matcher — fact-bank skill/tag embeddings, built once (no LLM).
    matcher = SkillMatcher.build(fact_bank, embedder, config.stack_match_threshold)

    # The Stage B window is top-N by cheap_score over ALL survivors, scored or
    # not. Survivors = unscored ('new'), promoted ('queued'), and gate-filtered
    # ('filtered' with a req_* reason). Phase-1-filtered jobs are excluded. An
    # already-scored job still OCCUPIES its window slot, so re-runs do not drag a
    # fresh cohort in and spend new LLM calls.
    survivors = list(
        (
            await session.scalars(select(Job).where(_survivor_where()))
        ).all()
    )
    report.total = len(survivors)

    # De-boilerplate each survivor against its same-company siblings BEFORE
    # embedding, so the role-specific text is what MiniLM sees (not a shared
    # "About <company>" preamble that truncation would otherwise leave alone).
    clean = deboilerplate_by_company(
        [(j.id, j.company, j.description_text) for j in survivors]
    )

    # --- Stage A: embed once, cosine, rank ALL survivors ----------------
    ranked: list[tuple[Job, float]] = []
    for job in survivors:
        vec = to_vector(
            _embedding(job, clean[job.id], embedder, config.embedding_model)
        )
        cheap = (cosine(vec, candidate) + 1.0) / 2.0  # map [-1,1] -> [0,1]
        ranked.append((job, cheap))
    ranked.sort(key=lambda t: t[1], reverse=True)
    top_ids = select_window(
        ranked, config.top_n_llm, config.max_per_company_in_window
    )

    # --- Stage B: window only. Budget-guarded, committed per job so no LLM
    # call is ever wasted; non-window jobs are NOT written (the scorer touches
    # only the ~N window rows, not the whole survivor table). ------------------
    for job, cheap in ranked:
        if _already_stage_b(job):
            report.reused += 1  # occupies its slot; no LLM call
            continue
        if id(job) not in top_ids:
            report.stage_a_only += 1  # ranked only, not persisted
            continue

        u = provider.usage
        if u.calls >= config.max_llm_calls:
            report.budget_tripped = "calls"
            break
        if u.cost_usd >= config.max_spend_usd:
            report.budget_tripped = "spend"
            break

        try:
            await _stage_b(
                job, clean[job.id], cheap, provider, fact_bank, config, report, matcher
            )
            await session.commit()  # per-job: never lose LLM work on interrupt
        except Exception as exc:  # noqa: BLE001 - isolate provider failures
            log.error("score.stage_b.failed", job=job.id, error=str(exc))
            report.errors.append(f"{job.id}: {exc}")
            await session.rollback()
            break  # stop spending; progress already committed per job

    u = provider.usage
    report.llm_calls = u.calls
    report.retries = u.retries
    report.input_tokens = u.input_tokens
    report.output_tokens = u.output_tokens
    report.cost_usd = u.cost_usd
    log.info("score.summary", summary=report.line())
    return report


async def _stage_b(
    job: Job,
    clean_text: str,
    cheap: float,
    provider: BaseLLMProvider,
    fact_bank: FactBank,
    config: ScoreConfig,
    report: ScoreReport,
    matcher: SkillMatcher,
) -> None:
    # Extract + rubric read the de-boilerplated text, so the requirement
    # extractor sees the role (not a truncated "About <company>" preamble).
    requirements = await extract_requirements(provider, job.title, clean_text)
    gate = check_hard_requirements(
        requirements,
        fact_bank,
        max_years=config.max_years,
        matcher=matcher,
        min_missing=config.stack_gate_min_missing,
    )
    rubric: Rubric | None = None
    gate_reason = gate.reason
    if gate_reason is not None:
        report.gated += 1
        report.gate_reasons[gate_reason] = report.gate_reasons.get(gate_reason, 0) + 1
    else:
        rubric = await score_rubric(provider, job.title, clean_text, fact_bank)
        report.stage_b_scored += 1

    match = blend_match_score(cheap, rubric, config.weights)
    job.match_score = match
    job.score_breakdown = ScoreBreakdown(
        cheap_score=round(cheap, 4),
        match_score=match,
        requirements=requirements,
        rubric=rubric,
        gate_reason=gate_reason,
        missing_requirements=gate.missing,
        weights=config.weights,
        stage_b_scored=rubric is not None,
    ).model_dump()

    if gate_reason is not None:
        job.status = JobStatus.filtered
        job.filter_reason = gate_reason
    elif _promote(job, match, config.score_threshold):
        report.queued += 1


def _promote(job: Job, match: float, threshold: float) -> bool:
    if match >= threshold:
        job.status = JobStatus.queued
        return True
    return False
