"""Application pre-fill: field classification + the structural no-submit guarantee.

Zero browser. A FakeDriver stands in for Playwright. The iron rule (CLAUDE
constraint 1) is enforced structurally: no method clicks submit, and the apply/
source contains no click at all — these are asserted here."""
from __future__ import annotations

import pathlib

from jobagent.apply.board import Board, detect_board
from jobagent.apply.candidate import candidate_from_profile
from jobagent.apply.driver import PlaywrightDriver
from jobagent.apply.mapping import plan_prefill
from jobagent.apply.models import (
    CandidateData,
    FieldType,
    FormField,
    Outcome,
)
from jobagent.apply.prefill import prefill_application, render_report
from jobagent.factbank import load_fact_bank

FB = load_fact_bank("tests/fixtures/fact_bank.json")
CAND = CandidateData(
    first_name="Nantha Kumar", last_name="A", email="candidate@example.com",
    phone="555-0142", linkedin="linkedin.com/in/x", github="github.com/x",
    location="Arlington, TX", authorized_to_work_now=True,
    requires_future_sponsorship=True,
)


def _f(label: str, type_: FieldType = FieldType.text, options: tuple = ()) -> FormField:
    return FormField(label=label, selector=f"#{label.lower().replace(' ', '_')}",
                     type=type_, options=options)


# --- board detection ------------------------------------------------------
def test_detect_board_by_source_and_url() -> None:
    assert detect_board("greenhouse", None) is Board.greenhouse
    assert detect_board(None, "https://job-boards.greenhouse.io/x/jobs/1") is Board.greenhouse
    assert detect_board(None, "https://jobs.lever.co/x/1") is Board.lever
    assert detect_board(None, "https://jobs.ashbyhq.com/x/1") is Board.ashby
    assert detect_board(None, "https://example.com/careers") is Board.unknown


# --- candidate ------------------------------------------------------------
def test_candidate_from_profile_splits_name_and_workauth() -> None:
    c = candidate_from_profile(FB.profile)
    assert c.first_name == "Nantha Kumar" and c.last_name == "A"
    assert c.authorized_to_work_now is True          # F-1 OPT, no sponsorship now
    assert c.requires_future_sponsorship is True


# --- the planner: FILL vs FLAG vs BLANK -----------------------------------
def test_core_fields_filled() -> None:
    fields = [_f("First Name"), _f("Last Name"), _f("Email", FieldType.email),
              _f("Phone", FieldType.tel), _f("LinkedIn Profile"), _f("GitHub URL")]
    plan = plan_prefill(fields, CAND)
    got = {p.field.label: p.value for p in plan.fills}
    assert got["First Name"] == "Nantha Kumar" and got["Last Name"] == "A"
    assert got["Email"] == CAND.email and got["Phone"] == CAND.phone
    assert got["LinkedIn Profile"] == CAND.linkedin and got["GitHub URL"] == CAND.github
    assert len(plan.flagged) == 0 and len(plan.blanks) == 0


def test_demographic_salary_motivation_are_flagged_never_filled() -> None:
    fields = [
        _f("Race / Ethnicity", FieldType.select, ("Decline", "Asian")),
        _f("Gender", FieldType.select),
        _f("Veteran Status", FieldType.select),
        _f("Voluntary Self-Identification of Disability", FieldType.select),
        _f("Are you a U.S. Citizen?", FieldType.select),
        _f("Desired Salary", FieldType.text),
        _f("Why do you want to work here?", FieldType.textarea),
        _f("Pronouns"),
    ]
    plan = plan_prefill(fields, CAND)
    assert len(plan.fills) == 0            # NOTHING here is ever auto-answered
    assert len(plan.flagged) == len(fields)


def test_work_auth_preset_only_high_confidence() -> None:
    fields = [
        _f("Are you legally authorized to work in the U.S.?", FieldType.select, ("Yes", "No")),
        _f("Will you now or in the future require sponsorship?", FieldType.select, ("Yes", "No")),
        _f("Do you require sponsorship?", FieldType.select, ("Yes", "No")),  # ambiguous
    ]
    plan = plan_prefill(fields, CAND)
    by = {p.field.label: p for p in plan.planned}
    assert by["Are you legally authorized to work in the U.S.?"].outcome is Outcome.fill
    assert by["Are you legally authorized to work in the U.S.?"].value == "Yes"
    assert by["Will you now or in the future require sponsorship?"].value == "Yes"
    assert by["Do you require sponsorship?"].outcome is Outcome.flag  # not guessed


def test_unknown_fields_left_blank_and_resume_detected() -> None:
    fields = [_f("Referral source"), _f("Resume", FieldType.file),
              _f("Cover Letter", FieldType.file)]
    plan = plan_prefill(fields, CAND)
    labels_blank = {p.field.label for p in plan.blanks}
    assert "Referral source" in labels_blank
    assert plan.resume_field is not None and plan.resume_field.label == "Resume"
    # cover-letter file is flagged (human decides), never auto-uploaded
    assert any(p.field.label == "Cover Letter" and p.outcome is Outcome.flag for p in plan.planned)


# --- orchestration with a fake driver -------------------------------------
class _FakeDriver:
    def __init__(self, fields, submit=None):
        self._fields, self._submit = fields, submit
        self.url = None
        self.filled = {}
        self.chosen = {}
        self.attached: list = []
        self.shots: list = []
    def goto(self, url): self.url = url
    def list_fields(self): return self._fields
    def fill_text(self, s, v): self.filled[s] = v
    def choose(self, s, v): self.chosen[s] = v
    def attach_file(self, s, p): self.attached.append((s, p))
    def locate_submit(self): return self._submit
    def screenshot(self, p): self.shots.append(p)


def test_prefill_fills_stops_and_does_not_attach_by_default() -> None:
    submit = FormField("Submit Application", "(submit button)", FieldType.other)
    fields = [_f("First Name"), _f("Email", FieldType.email),
              _f("Gender", FieldType.select), _f("Resume", FieldType.file)]
    d = _FakeDriver(fields, submit=submit)
    report = prefill_application(d, "greenhouse:1", "https://x/apply", CAND)
    assert d.filled["#first_name"] == "Nantha Kumar"
    assert d.filled["#email"] == CAND.email
    assert "#gender" not in d.filled          # demographic never filled
    assert d.attached == []                   # resume attach is opt-in
    assert report.resume_attached is False
    assert report.plan.submit == submit       # located, not clicked
    assert "REVIEW AND SUBMIT MANUALLY" in render_report(report)


# --- STRUCTURAL safety guarantees -----------------------------------------
def test_no_submit_or_click_method_anywhere() -> None:
    # The only submit-related method is locate_submit, and it only LOCATES.
    for name in dir(PlaywrightDriver):
        if name.startswith("_"):
            continue
        low = name.lower()
        assert "submit" not in low or name == "locate_submit", f"suspicious method {name}"
        assert "click" not in low, f"suspicious method {name}"
    assert hasattr(PlaywrightDriver, "locate_submit")


def test_apply_source_contains_no_click() -> None:
    # Structural proof: nothing in apply/ ever clicks anything.
    for path in pathlib.Path("jobagent/apply").glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert ".click(" not in src, f"{path} contains a .click( call"
