"""Load board tokens to poll from data/companies.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

from .registry import SUPPORTED_SOURCES


class CompanyConfig(BaseModel):
    """One board to poll."""

    name: str
    source: str
    token: str

    @field_validator("source")
    @classmethod
    def _known_source(cls, value: str) -> str:
        if value not in SUPPORTED_SOURCES:
            raise ValueError(
                f"unknown source {value!r}; supported: {', '.join(SUPPORTED_SOURCES)}"
            )
        return value


def load_companies(path: str | Path) -> list[CompanyConfig]:
    """Parse and validate the companies YAML file."""
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = raw.get("companies", []) if isinstance(raw, dict) else raw
    return [CompanyConfig.model_validate(entry) for entry in entries]
