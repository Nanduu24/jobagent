"""Build the verified CandidateData the pre-fill draws from (profile + resume)."""
from __future__ import annotations

from pathlib import Path

from ..factbank import Profile
from .models import CandidateData


def _split_name(full: str) -> tuple[str, str]:
    """Split a display name into (first, last). Last token is the surname (fits the
    candidate's 'Nantha Kumar A' -> first 'Nantha Kumar', last 'A')."""
    parts = full.split()
    if len(parts) < 2:
        return (full, "")
    return (" ".join(parts[:-1]), parts[-1])


def candidate_from_profile(
    profile: Profile, resume_path: Path | None = None
) -> CandidateData:
    first, last = _split_name(profile.name)
    wa = profile.work_authorization
    return CandidateData(
        first_name=first,
        last_name=last,
        email=profile.email,
        phone=profile.phone,
        linkedin=profile.linkedin,
        github=profile.github,
        location=profile.location,
        resume_path=resume_path,
        authorized_to_work_now=not wa.requires_sponsorship_now,
        requires_future_sponsorship=wa.requires_sponsorship_future,
    )
