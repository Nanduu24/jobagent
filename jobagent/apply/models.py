"""Data models for the pre-fill flow. Deliberately no submit action anywhere."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FieldType(str, Enum):
    text = "text"
    email = "email"
    tel = "tel"
    file = "file"
    select = "select"
    radio = "radio"
    checkbox = "checkbox"
    textarea = "textarea"
    other = "other"


@dataclass(frozen=True)
class FormField:
    """One control on an application form, as introspected from the DOM."""

    label: str
    selector: str
    type: FieldType
    required: bool = False
    options: tuple[str, ...] = ()  # for select / radio groups


class Outcome(str, Enum):
    fill = "fill"    # auto-filled from VERIFIED candidate data
    flag = "flag"    # must NEVER be auto-answered -> human decides
    blank = "blank"  # couldn't confidently map -> left blank for the human


@dataclass(frozen=True)
class PlannedField:
    field: FormField
    outcome: Outcome
    value: str | None = None  # the value to fill (Outcome.fill only)
    reason: str = ""          # why it was flagged / left blank


@dataclass
class PrefillPlan:
    planned: list[PlannedField] = field(default_factory=list)
    resume_field: FormField | None = None  # file input for the resume (attach step)
    submit: FormField | None = None        # LOCATED for the human, never clicked

    @property
    def fills(self) -> list[PlannedField]:
        return [p for p in self.planned if p.outcome is Outcome.fill]

    @property
    def flagged(self) -> list[PlannedField]:
        return [p for p in self.planned if p.outcome is Outcome.flag]

    @property
    def blanks(self) -> list[PlannedField]:
        return [p for p in self.planned if p.outcome is Outcome.blank]


@dataclass(frozen=True)
class CandidateData:
    """The verified data the pre-fill draws from (profile + rendered resume). The
    work-auth answer is the honest OPT pre-set — nothing beyond it is auto-answered."""

    first_name: str
    last_name: str
    email: str
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    location: str | None = None
    resume_path: Path | None = None
    authorized_to_work_now: bool = True          # F-1 OPT: authorized now
    requires_future_sponsorship: bool = True     # ...will need sponsorship later


@dataclass
class PrefillReport:
    job_id: str
    apply_url: str
    plan: PrefillPlan
    resume_attached: bool = False
    screenshot_path: str | None = None
    note: str = "PRE-FILLED FOR HUMAN REVIEW — the agent did NOT submit."
