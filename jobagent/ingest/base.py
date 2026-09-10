"""HTTP plumbing (rate limit + backoff) and the BoardAdapter ABC."""
from __future__ import annotations

import abc
import asyncio
import random
from collections import defaultdict
from types import TracebackType

import httpx

from ..config import get_settings
from ..db.schemas import Job
from ..logging import get_logger

log = get_logger(__name__)


class HostRateLimiter:
    """Enforce a minimum interval between requests to the same host."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._next_allowed: dict[str, float] = defaultdict(float)

    async def acquire(self, host: str) -> None:
        if self._min_interval <= 0:
            return
        async with self._locks[host]:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_allowed[host] - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_allowed[host] = now + self._min_interval


class HttpClient:
    """Thin wrapper over httpx.AsyncClient adding per-host rate limiting and
    exponential backoff that respects 429 ``Retry-After``."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        min_interval: float | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        base_backoff: float = 0.5,
    ) -> None:
        settings = get_settings()
        self._min_interval = (
            min_interval if min_interval is not None else settings.min_request_interval
        )
        self._max_retries = (
            max_retries if max_retries is not None else settings.max_retries
        )
        self._timeout = timeout if timeout is not None else settings.request_timeout
        self._base_backoff = base_backoff
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": "jobagent/0.1 (+https://github.com/local/jobagent)"},
            follow_redirects=True,
        )
        self._limiter = HostRateLimiter(self._min_interval)

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _backoff(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return retry_after
        factor = float(2**attempt)
        return self._base_backoff * factor + random.uniform(0, 0.1)

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        value = resp.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    async def request_json(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> object:
        """Perform a JSON request with rate limiting + retries. Returns the
        decoded JSON body."""
        host = httpx.URL(url).host
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire(host)
            try:
                resp = await self._client.request(method, url, **kwargs)  # type: ignore[arg-type]
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    raise
                await asyncio.sleep(self._backoff(attempt))
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self._max_retries:
                    resp.raise_for_status()
                retry_after = (
                    self._retry_after(resp) if resp.status_code == 429 else None
                )
                log.warning(
                    "http.retry",
                    url=url,
                    status=resp.status_code,
                    attempt=attempt,
                )
                await asyncio.sleep(self._backoff(attempt, retry_after))
                continue

            resp.raise_for_status()
            return resp.json()

        # Unreachable in practice: the loop either returns or raises.
        assert last_exc is not None
        raise last_exc

    async def get_json(self, url: str, **kwargs: object) -> object:
        return await self.request_json("GET", url, **kwargs)

    async def post_json(self, url: str, **kwargs: object) -> object:
        return await self.request_json("POST", url, **kwargs)


class BoardAdapter(abc.ABC):
    """Common interface for one ATS board API.

    Concrete adapters normalize a board's public, unauthenticated response into
    a list of canonical :class:`Job` objects.
    """

    #: Short board identifier, e.g. "greenhouse". Used as the job id prefix.
    source: str

    def __init__(self, http: HttpClient) -> None:
        self._http = http

    @abc.abstractmethod
    async def fetch(self, token: str) -> list[Job]:
        """Fetch and normalize all open postings for ``token``."""
        raise NotImplementedError
