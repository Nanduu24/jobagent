"""Enumerations shared across the DB models and ingest boundary."""
from __future__ import annotations

import enum


class JobStatus(str, enum.Enum):
    """Lifecycle status of a job posting in the pipeline."""

    new = "new"
    filtered = "filtered"
    queued = "queued"
    applied = "applied"
    rejected = "rejected"
    interview = "interview"
    offer = "offer"


class RemoteType(str, enum.Enum):
    """Normalized workplace type across heterogeneous board fields."""

    remote = "remote"
    hybrid = "hybrid"
    onsite = "onsite"
    unknown = "unknown"
