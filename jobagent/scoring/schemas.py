"""Pydantic schemas for Stage B LLM output and the persisted breakdown."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Requirements(BaseModel):
    """b1: structured requirements extracted from a job description."""

    required: list[str] = Field(default_factory=list)
    preferred: list[str] = Field(default_factory=list)
    years_experience: int | None = None
    seniority: str | None = None
    must_have_citizenship: bool = False
    requires_clearance: bool = False


class RubricAxis(BaseModel):
    score: float = Field(ge=0, le=10)
    justification: str


class Rubric(BaseModel):
    """b2: 0-10 rubric with a one-sentence justification per axis."""

    skill_overlap: RubricAxis
    seniority_fit: RubricAxis
    domain_fit: RubricAxis


class ScoreBreakdown(BaseModel):
    """The persisted score_breakdown (jsonb). The breakdown IS the product."""

    cheap_score: float  # Stage A cosine similarity, 0-1
    match_score: float  # final blended 0-100
    requirements: Requirements | None = None  # b1
    rubric: Rubric | None = None  # b2
    gate_reason: str | None = None  # set if the hard-req gate fired
    missing_requirements: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    stage_b_scored: bool = False
