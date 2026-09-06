"""Runtime configuration loaded from the environment / .env.

Secrets live only in the environment (see .env.example). Nothing here is logged.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Field names map to ``JOBAGENT_*`` env vars via ``env_prefix``; the two
    database URLs are read verbatim (no prefix) so they read naturally in .env.
    """

    model_config = SettingsConfigDict(
        env_prefix="JOBAGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+asyncpg://jobagent:jobagent@localhost:5432/jobagent",
        validation_alias=AliasChoices("DATABASE_URL", "JOBAGENT_DATABASE_URL"),
    )
    min_request_interval: float = 0.5
    request_timeout: float = 30.0
    max_retries: int = 4
    log_level: str = "info"
    companies_file: str = "data/companies.yaml"

    # Workable is quarantined: its multi-page nextPage cursor-following is
    # implemented but UNVERIFIED against a live multi-page account, and probing
    # hard-429'd their API. Off by default; opt in with ENABLE_WORKABLE=true.
    enable_workable: bool = Field(
        default=False,
        validation_alias=AliasChoices("ENABLE_WORKABLE", "JOBAGENT_ENABLE_WORKABLE"),
    )

    # Resilience: no single board or company can stall the whole poll. Companies
    # within a board are polled SEQUENTIALLY behind a 0.5s per-host rate limit, so
    # the board timeout must cover the whole board: with companies.yaml at ~78
    # Ashby / ~69 Greenhouse tokens (some with 2k+ jobs), 180s would truncate the
    # tail. 900s/board gives headroom; boards run concurrently so wall-clock is
    # ~the slowest board, not the sum.
    poll_company_timeout: float = 30.0
    poll_board_timeout: float = 900.0

    # --- Phase 2: scoring -------------------------------------------------
    # Provider abstraction. Groq (llama-3.3-70b) is the low-cost default.
    llm_provider: str = Field(
        default="groq",
        validation_alias=AliasChoices("LLM_PROVIDER", "JOBAGENT_LLM_PROVIDER"),
    )
    groq_api_key: str = Field(
        default="", validation_alias=AliasChoices("GROQ_API_KEY")
    )
    anthropic_api_key: str = Field(
        default="", validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )
    gemini_api_key: str = Field(
        default="", validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    groq_model: str = "llama-3.3-70b-versatile"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    # gemini-3.1-flash-lite: funded free tier (15 RPM / 250K TPM / 500 RPD).
    gemini_model: str = Field(
        default="gemini-3.1-flash-lite",
        validation_alias=AliasChoices("GEMINI_MODEL", "JOBAGENT_GEMINI_MODEL"),
    )

    # Local embedding model (sentence-transformers, offline, zero API cost).
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_dir: str = ".cache/jobagent"

    # Hard-requirement gate: years-of-experience threshold treated as a blocker.
    hard_req_max_years: int = 5

    # Stack gate (semantic): a required skill counts as "missing" only if its
    # best cosine similarity to the fact-bank skill/tag set is below the
    # threshold; the gate fires only when >= min_missing skills are unmatched.
    stack_match_threshold: float = Field(
        default=0.60,
        validation_alias=AliasChoices(
            "STACK_MATCH_THRESHOLD", "JOBAGENT_STACK_MATCH_THRESHOLD"
        ),
    )
    stack_gate_min_missing: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "STACK_GATE_MIN_MISSING", "JOBAGENT_STACK_GATE_MIN_MISSING"
        ),
    )

    # Per-run budget guard: abort cleanly (commit progress) if either trips.
    max_llm_calls_per_run: int = Field(
        default=70,
        validation_alias=AliasChoices(
            "MAX_LLM_CALLS_PER_RUN", "JOBAGENT_MAX_LLM_CALLS_PER_RUN"
        ),
    )
    max_spend_usd_per_run: float = Field(
        default=0.50,
        validation_alias=AliasChoices(
            "MAX_SPEND_USD_PER_RUN", "JOBAGENT_MAX_SPEND_USD_PER_RUN"
        ),
    )

    # Anthropic free tier: 5 req/min, 10K input tok/min. Pace ~1 call / 12s so
    # requests are never bursted into a 429.
    anthropic_min_interval: float = 12.0
    # Gemini 3.1 Flash-Lite free tier: 15 RPM. Pace to 12 RPM (20% under) -> 5s.
    gemini_min_interval: float = 5.0

    # Stage B budget + promotion.
    top_n_llm: int = Field(
        default=200, validation_alias=AliasChoices("TOP_N_LLM", "JOBAGENT_TOP_N_LLM")
    )
    # Diversity: cap any single company's share of the Stage B window so a
    # fact-bank-affine company (e.g. LangChain) can't monopolize the budget.
    max_per_company_in_window: int = Field(
        default=5,
        validation_alias=AliasChoices(
            "MAX_PER_COMPANY_IN_WINDOW", "JOBAGENT_MAX_PER_COMPANY_IN_WINDOW"
        ),
    )
    score_threshold: float = Field(
        default=65.0,
        validation_alias=AliasChoices("SCORE_THRESHOLD", "JOBAGENT_SCORE_THRESHOLD"),
    )

    # Phase 3 generator: max tailored bullets per resume (a CAP, not a quota).
    tailor_max_bullets: int = Field(
        default=8,
        validation_alias=AliasChoices("TAILOR_MAX_BULLETS", "JOBAGENT_TAILOR_MAX_BULLETS"),
    )
    # Relevance floor: a fact is emitted only if its select score clears this.
    # The select score (sim*0.6 + IDF-weighted keyword-overlap*0.4) now separates
    # relevant facts from off-domain (geoscience) ones by a real margin: across
    # the live LangChain roles every geoscience fact scores <= 0.432 while the
    # on-domain facts sit 0.46-0.71. 0.45 lands in that ~0.034 gap — it clears the
    # LangSmith fact on the FullStack role (0.466) and cuts geoscience on the
    # Python OSS role (0.432), both by ~0.016 (not the old 0.003 noise band).
    tailor_min_relevance: float = Field(
        default=0.45,
        validation_alias=AliasChoices(
            "TAILOR_MIN_RELEVANCE", "JOBAGENT_TAILOR_MIN_RELEVANCE"
        ),
    )
    # Floor safety net: if fewer than this many facts clear the floor, emit the
    # top-N by score anyway (logged) so a resume is never near-empty.
    tailor_min_bullets: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "TAILOR_MIN_BULLETS", "JOBAGENT_TAILOR_MIN_BULLETS"
        ),
    )

    # match_score blend weights (Stage A cosine + Stage B rubric). Summed = 1.0.
    weight_cheap: float = 0.30
    weight_skill: float = 0.30
    weight_seniority: float = 0.20
    weight_domain: float = 0.20

    # Freshness. A posting still listed on a live board API is evidence it is
    # still open, so age does not delete it by default. In 'keep' mode, jobs
    # older than max_age_days stay status='new' with age_days persisted (for
    # Phase 2 ranking to penalize); in 'cut' mode they are filtered as 'stale'.
    max_age_days: int = Field(
        default=90,
        validation_alias=AliasChoices("MAX_AGE_DAYS", "JOBAGENT_MAX_AGE_DAYS"),
    )
    stale_mode: str = Field(
        default="keep",  # "keep" | "cut"
        validation_alias=AliasChoices("STALE_MODE", "JOBAGENT_STALE_MODE"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
