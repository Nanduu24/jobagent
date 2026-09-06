"""Fact bank: the verified source of truth about the candidate.

Loaded and validated with Pydantic at startup. A malformed fact bank is a HARD
FAIL (the validation error propagates) — never a warning — because everything
downstream (scoring now, resume tailoring in Phase 2+) must trace to it.

Phase 2 only READS the fact bank (skills, experience, work authorization) to
score jobs. It never generates or paraphrases any claim from it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WorkAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    requires_sponsorship_now: bool
    requires_sponsorship_future: bool
    us_citizen: bool
    clearance_eligible: bool


class Profile(BaseModel):
    name: str
    email: str
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    salesforce: str | None = None
    location: str | None = None
    relocation: bool = False
    work_authorization: WorkAuthorization
    years_professional_experience: int = Field(ge=0)
    target_seniority: list[str] = Field(default_factory=list)


class Education(BaseModel):
    id: str
    degree: str
    school: str
    location: str | None = None
    start: str | None = None
    end: str | None = None
    gpa: str | None = None
    graduated: str | None = None
    coursework: list[str] = Field(default_factory=list)


class ExperienceEntry(BaseModel):
    """A résumé EXPERIENCE header (the bullets under it are facts, matched by
    ``Fact.project == ExperienceEntry.name``)."""

    name: str
    title: str
    org: str
    advisor: str | None = None
    location: str | None = None
    start: str | None = None
    end: str | None = None
    descriptor: str | None = None  # the "Project:" line under the header


class ProjectMeta(BaseModel):
    """Résumé PROJECT header metadata (matched by ``Fact.project == name``)."""

    name: str
    descriptor: str | None = None  # "Live Agentic AI Platform with Voice & Memory"
    tech: list[str] = Field(default_factory=list)  # inline tech-stack line
    note: str | None = None  # e.g. "Academic Team Project (3-person Agile/Scrum)"
    url: str | None = None  # project link; rendered as an anchor ONLY when present


class Fact(BaseModel):
    id: str
    project: str
    context: str
    kind: str = "project"  # "project" | "experience" | "publication"
    tags: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    claim: str
    variants: list[str] = Field(default_factory=list)


class FactBank(BaseModel):
    profile: Profile
    education: list[Education] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectMeta] = Field(default_factory=list)
    skills: dict[str, list[str]] = Field(default_factory=dict)
    facts: list[Fact]

    def experience_by_name(self) -> dict[str, ExperienceEntry]:
        return {e.name: e for e in self.experience}

    def project_by_name(self) -> dict[str, ProjectMeta]:
        return {p.name: p for p in self.projects}

    @field_validator("facts")
    @classmethod
    def _facts_non_empty(cls, value: list[Fact]) -> list[Fact]:
        if not value:
            raise ValueError("fact bank must contain at least one fact")
        return value

    @model_validator(mode="after")
    def _unique_fact_ids(self) -> "FactBank":
        ids = [f.id for f in self.facts]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate fact ids: {sorted(dupes)}")
        return self

    # --- derived views used by the scorer --------------------------------
    def all_skills(self) -> set[str]:
        """Every skill/tool the candidate can evidence (lowercased)."""
        out: set[str] = set()
        for values in self.skills.values():
            out.update(s.lower() for s in values)
        for fact in self.facts:
            out.update(t.lower() for t in fact.tools)
        return out

    def fact_texts(self) -> list[str]:
        """One embeddable string per fact (claim + tools + tags)."""
        return [
            f"{f.claim} Tools: {', '.join(f.tools)}. Areas: {', '.join(f.tags)}."
            for f in self.facts
        ]


def hash_fact_bank(path: str | Path) -> str:
    """SHA-256 of the raw fact bank file (cache key for embeddings)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fact_bank(path: str | Path) -> FactBank:
    """Load + validate the fact bank. Raises on malformed input (hard fail)."""
    return FactBank.model_validate_json(Path(path).read_text(encoding="utf-8"))
