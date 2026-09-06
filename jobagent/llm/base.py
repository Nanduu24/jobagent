"""LLM provider abstraction.

A provider exposes one method::

    async complete(messages, *, schema: type[BaseModel] | None) -> T

When ``schema`` is given the raw completion is parsed into that Pydantic model
(``` fences stripped, one retry on parse failure); otherwise the raw string is
returned. Token + cost accounting is recorded per call and summed per run.

Concrete providers implement only :meth:`BaseLLMProvider._raw_complete`; all the
parsing/retry/accounting lives here so every provider behaves identically.
"""
from __future__ import annotations

import abc
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TypeVar, overload

import httpx
from pydantic import BaseModel, ValidationError

from ..logging import get_logger
from .pricing import cost_usd, is_priced, price_for

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# A chat message. Kept as a plain dict so it maps onto every provider's API.
Message = dict[str, str]


class LLMError(RuntimeError):
    """Provider call or response-parsing failure."""


@dataclass
class RawCompletion:
    """A single raw provider response plus its token usage."""

    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class UsageTracker:
    """Accumulates per-call token + cost accounting across a run.

    ``retries`` counts extra calls caused by a parse-failure retry, so the
    retry rate (retries / calls) is reported and can be checked against the 5%
    prompt-quality threshold.
    """

    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    def add(
        self, model: str, input_tokens: int, output_tokens: int, *, is_retry: bool = False
    ) -> None:
        self.calls += 1
        if is_retry:
            self.retries += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        call_cost = cost_usd(model, input_tokens, output_tokens)
        self.cost_usd += call_cost
        m = self.by_model.setdefault(model, ModelUsage())
        m.calls += 1
        m.input_tokens += input_tokens
        m.output_tokens += output_tokens
        m.cost_usd += call_cost

    @property
    def retry_rate(self) -> float:
        return self.retries / self.calls if self.calls else 0.0

    def summary(self) -> str:
        return (
            f"{self.calls} calls ({self.retries} retries, "
            f"{self.retry_rate:.1%}) / {self.input_tokens} in + "
            f"{self.output_tokens} out tokens / ${self.cost_usd:.4f}"
        )

    def by_model_lines(self) -> list[str]:
        """One transparent line per model: which rate table entry was used."""
        out: list[str] = []
        for model, u in self.by_model.items():
            price = price_for(model)
            rate = (
                f"${price[0]:.2f}/${price[1]:.2f} per Mtok"
                if price is not None
                else "UNPRICED ($0.00)"
            )
            free = " (free tier)" if is_priced(model) and price == (0.0, 0.0) else ""
            out.append(
                f"{model}: {u.calls} calls, {u.input_tokens}+{u.output_tokens} tok, "
                f"${u.cost_usd:.4f} [{rate}{free}]"
            )
        return out


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Strip a single leading/trailing ``` (```json) fence if present."""
    stripped = text.strip()
    stripped = _FENCE_RE.sub("", stripped)
    return stripped.strip()


def parse_structured(text: str, schema: type[T]) -> T:
    """Parse a model's text into ``schema``.

    Tolerates code fences and leading/trailing prose by extracting the first
    balanced JSON object/array. Raises :class:`LLMError` on failure.
    """
    candidate = strip_fences(text)
    try:
        return schema.model_validate_json(candidate)
    except ValidationError:
        pass
    # Fall back to the first {...} or [...] span.
    match = re.search(r"(\{.*\}|\[.*\])", candidate, re.DOTALL)
    if match:
        try:
            return schema.model_validate(json.loads(match.group(1)))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise LLMError(f"could not parse structured output: {exc}") from exc
    raise LLMError("no JSON found in model output")


class BaseLLMProvider(abc.ABC):
    """Common parsing / retry / accounting for all providers."""

    #: Concrete model identifier, e.g. "llama-3.3-70b-versatile".
    model: str

    def __init__(self, model: str) -> None:
        self.model = model
        self.usage = UsageTracker()

    @abc.abstractmethod
    async def _raw_complete(self, messages: list[Message]) -> RawCompletion:
        """Perform one provider call. Subclass responsibility."""
        raise NotImplementedError

    @overload
    async def complete(self, messages: list[Message], *, schema: type[T]) -> T: ...
    @overload
    async def complete(
        self, messages: list[Message], *, schema: None = None
    ) -> str: ...

    async def complete(
        self, messages: list[Message], *, schema: type[T] | None = None
    ) -> T | str:
        raw = await self._raw_complete(messages)
        self.usage.add(self.model, raw.input_tokens, raw.output_tokens)
        if schema is None:
            return raw.text
        try:
            return parse_structured(raw.text, schema)
        except LLMError as exc:
            # Retry once with an explicit corrective instruction. Logged so the
            # retry rate is observable.
            log.warning(
                "llm.parse_retry",
                model=self.model,
                schema=schema.__name__,
                error=str(exc),
                preview=raw.text[:120],
            )
            retry = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous reply could not be parsed. Reply with ONLY "
                        "valid JSON matching the requested schema, no prose, no "
                        "code fences."
                    ),
                }
            ]
            raw2 = await self._raw_complete(retry)
            self.usage.add(
                self.model, raw2.input_tokens, raw2.output_tokens, is_retry=True
            )
            return parse_structured(raw2.text, schema)


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int = 4,
    base_backoff: float = 1.0,
    max_wait: float = 45.0,
    **kwargs: object,
) -> httpx.Response:
    """POST/GET with backoff on 429 / 5xx, honoring the server's Retry-After.

    Retry-After is honored but capped at ``max_wait`` (default 45s): we do NOT
    sleep for minutes on an exhausted free-tier token window — past the cap the
    call fails fast and the run's budget guard / resilience takes over.
    """
    for attempt in range(max_retries + 1):
        resp = await client.request(method, url, **kwargs)  # type: ignore[arg-type]
        if resp.status_code != 429 and resp.status_code < 500:
            return resp
        if attempt >= max_retries:
            return resp
        retry_after = resp.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else base_backoff * (2**attempt)
        except ValueError:
            wait = base_backoff * (2**attempt)
        wait = min(wait, max_wait)
        log.warning(
            "llm.http_retry",
            url=url.split("?")[0],
            status=resp.status_code,
            attempt=attempt,
            wait_s=round(wait, 2),
        )
        await asyncio.sleep(wait)
    return resp
