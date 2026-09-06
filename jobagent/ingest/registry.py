"""Map a board ``source`` name to its adapter class."""
from __future__ import annotations

from .ashby import AshbyAdapter
from .base import BoardAdapter, HttpClient
from .greenhouse import GreenhouseAdapter
from .lever import LeverAdapter
from .workable import WorkableAdapter

ADAPTERS: dict[str, type[BoardAdapter]] = {
    GreenhouseAdapter.source: GreenhouseAdapter,
    LeverAdapter.source: LeverAdapter,
    AshbyAdapter.source: AshbyAdapter,
    WorkableAdapter.source: WorkableAdapter,
}

SUPPORTED_SOURCES = tuple(ADAPTERS)


def build_adapter(source: str, http: HttpClient) -> BoardAdapter:
    """Instantiate the adapter for ``source``.

    Raises KeyError for an unknown source (caught by the poller so one bad
    config line does not abort the run).
    """
    return ADAPTERS[source](http)
