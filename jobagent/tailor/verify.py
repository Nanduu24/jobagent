"""Fabrication verifier — the safety gate the resume generator sits behind.

A generated bullet PASSES only if it is checked against ONE fact and introduces
NOTHING absent from that fact. Deterministic (no LLM); errs toward rejection.

Which fact?
  - If the caller passes ``claimed_fact_id`` (the generator always knows the fact
    it drew from), the bullet is checked ONLY against that fact — no inference.
  - Otherwise the fact is inferred by content-token overlap, but an AMBIGUOUS
    top-2 (a near-tie) FAILS CLOSED ('ambiguous_mapping') — never a silent guess.

Checks (each carries a machine reason code so tests can pin WHY a bullet failed):
  numbers  every numeric token in the bullet must appear in the source fact.
  metric   a number bound to a metric ("R2 0.86") must have that exact pairing in
           the source — catches a real number attached to the wrong metric.
  tools    a known tool named in the bullet must appear in the source fact.
  role     a leadership verb absent from the source fact = fail.
  context  a production/live-traffic claim on a non-production fact = fail.
  fusion   a distinctive number/tool from a DIFFERENT fact = welded claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..factbank import Fact, FactBank

# --- machine reason codes -------------------------------------------------
R_EMPTY = "empty_bullet"
R_NO_MAP = "no_mapping"
R_AMBIGUOUS = "ambiguous_mapping"
R_UNKNOWN_FACT = "unknown_claimed_fact"
R_NUMBER = "invented_number"
R_METRIC = "metric_mismatch"
R_TOOL = "invented_tool"
R_ROLE = "role_inflation"
R_CONTEXT = "context_drift"
R_FUSION = "claim_fusion"

# --- lexicons -------------------------------------------------------------
_CURATED_TOOLS: frozenset[str] = frozenset(
    {
        "kubernetes", "k8s", "docker", "terraform", "ansible", "jenkins",
        "kafka", "rabbitmq", "spark", "hadoop", "flink", "airflow", "snowflake",
        "databricks", "redis", "memcached", "elasticsearch", "mongodb",
        "cassandra", "dynamodb", "mysql", "graphql", "grpc", "kotlin", "swift",
        "rust", "golang", "scala", "ruby", "tableau", "sagemaker", "kubeflow",
        "mlflow", "ray", "cuda", "tensorrt", "onnx", "triton", "jax",
        "aws", "gcp", "azure", "bigquery", "redshift", "hive", "presto",
        "prometheus", "grafana", "datadog", "nginx", "istio", "solidity",
        "flutter", "django", "flask", "spring", "rails",
    }
)
_LEADERSHIP_TERMS: tuple[str, ...] = (
    "led", "lead", "leading", "managed", "managing", "directed", "headed",
    "spearheaded", "oversaw", "overseeing", "supervised", "in charge of",
    "founded", "co-founded", "hired",
)
_PRODUCTION_TERMS: tuple[str, ...] = (
    "production", "in production", "production-grade", "productionized",
    "live traffic", "serving live", "at scale in production", "24/7",
)
_METRIC_TERMS: frozenset[str] = frozenset(
    {"r2", "rmsle", "rmse", "mae", "mse", "accuracy", "precision", "recall",
     "f1", "auc", "latency", "throughput", "gpa"}
)
_STOPWORDS: frozenset[str] = frozenset(
    "a an the and or of to in on for with using built build create created make "
    "made that this it its as at by from into over across an is are was were be "
    "our their my his her we they i you role work worked project system".split()
)

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?[a-zµ%]{0,4}", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\+\.#-]*", re.IGNORECASE)
_METRIC_WINDOW = 15  # chars between a number and a metric word to count as bound


@dataclass
class VerifyResult:
    ok: bool
    fact_id: str | None
    violations: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # machine codes

    def _add(self, code: str, message: str) -> None:
        self.reasons.append(code)
        self.violations.append(message)


@dataclass
class _FactView:
    fact: Fact
    text: str
    tokens: frozenset[str]
    numbers: frozenset[str]
    metric_pairs: frozenset[tuple[str, str]]


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "").lower() for m in _NUM_RE.finditer(text)}


def _tokens(text: str) -> set[str]:
    return {
        t for t in (w.group(0).lower() for w in _WORD_RE.finditer(text))
        if t not in _STOPWORDS and len(t) >= 2
    }


def _metric_pairs(text: str) -> set[tuple[str, str]]:
    """(metric, number) for each number sitting within _METRIC_WINDOW chars of a
    metric word — e.g. "R2 0.601" -> (r2, 0.601)."""
    low = text.lower().replace("²", "2")
    nums = [(m.start(), m.group(0).replace(",", "")) for m in _NUM_RE.finditer(low)]
    metrics: list[tuple[int, str]] = []
    for term in _METRIC_TERMS:
        for m in re.finditer(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            metrics.append((m.start(), term))
    pairs: set[tuple[str, str]] = set()
    for npos, num in nums:
        near = min(
            (t for t in metrics if abs(t[0] - npos) <= _METRIC_WINDOW),
            key=lambda t: abs(t[0] - npos),
            default=None,
        )
        if near is not None:
            pairs.add((near[1], num))
    return pairs


def _fact_text(fact: Fact) -> str:
    return " ".join(
        [fact.claim, *fact.variants, *fact.tools, *fact.tags, fact.project, fact.context]
    ).lower()


def _build_views(fact_bank: FactBank) -> list[_FactView]:
    views = []
    for fact in fact_bank.facts:
        num_src = " ".join([fact.claim, *fact.variants, *fact.tools, fact.project])
        views.append(
            _FactView(
                fact,
                _fact_text(fact),
                frozenset(_tokens(_fact_text(fact))),
                frozenset(_numbers(num_src)),
                frozenset(_metric_pairs(num_src)),
            )
        )
    return views


def _tool_lexicon(fact_bank: FactBank) -> frozenset[str]:
    return frozenset(_CURATED_TOOLS | {t.lower() for t in fact_bank.all_skills()})


def _mentions(term: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _is_specific(num: str) -> bool:
    digits = re.sub(r"[^0-9]", "", num)
    return "." in num or len(digits) >= 2 or bool(re.search(r"[a-z%]", num))


def verify_bullet(
    bullet_text: str,
    fact_bank: FactBank,
    claimed_fact_id: str | None = None,
) -> VerifyResult:
    """Verify one generated bullet against exactly one fact."""
    bullet = bullet_text.strip()
    if not bullet:
        return VerifyResult(False, None, ["empty bullet"], [R_EMPTY])

    bullet_low = bullet.lower()
    b_tokens = _tokens(bullet_low)
    b_numbers = _numbers(bullet_low)
    b_pairs = _metric_pairs(bullet_low)
    views = _build_views(fact_bank)
    lexicon = _tool_lexicon(fact_bank)
    by_id = {v.fact.id: v for v in views}

    # --- pick the fact to check against ---------------------------------
    if claimed_fact_id is not None:
        best = by_id.get(claimed_fact_id)
        if best is None:
            return VerifyResult(
                False, None, [f"unknown claimed fact '{claimed_fact_id}'"],
                [R_UNKNOWN_FACT],
            )
    else:
        scored = sorted(
            ((len(b_tokens & v.tokens), v) for v in views),
            key=lambda t: t[0],
            reverse=True,
        )
        best_overlap, best = scored[0]
        second_overlap = scored[1][0] if len(scored) > 1 else 0
        if best_overlap < 2 or (b_tokens and best_overlap / len(b_tokens) < 0.30):
            return VerifyResult(False, None, ["no matching fact in fact bank"], [R_NO_MAP])
        # Fail closed on a near-tie rather than guessing.
        if second_overlap >= 2 and (best_overlap - second_overlap) <= 1:
            names = f"'{best.fact.id}' vs '{scored[1][1].fact.id}'"
            return VerifyResult(
                False, best.fact.id,
                [f"ambiguous mapping ({names}); supply claimed_fact_id"],
                [R_AMBIGUOUS],
            )

    result = VerifyResult(ok=True, fact_id=best.fact.id)

    # 1. NUMBERS — any bullet number absent from the source fact.
    for num in sorted(b_numbers - best.numbers):
        result._add(R_NUMBER, f"invented number/scale '{num}' not in fact '{best.fact.id}'")

    # 2. METRIC binding — a number bound to the wrong metric.
    for metric, num in sorted(b_pairs - best.metric_pairs):
        if num in best.numbers:  # number is real but bound to the wrong metric
            result._add(
                R_METRIC,
                f"metric '{metric}' bound to '{num}' — not that pairing in fact "
                f"'{best.fact.id}'",
            )

    # 3. TOOLS — a known tool named in the bullet but absent from the source fact.
    for tool in sorted(lexicon):
        if _mentions(tool, bullet_low) and not _mentions(tool, best.text):
            result._add(R_TOOL, f"invented/misattributed tool '{tool}' not in fact '{best.fact.id}'")

    # 4. ROLE inflation — leadership verb absent from the source fact.
    for term in _LEADERSHIP_TERMS:
        if _mentions(term, bullet_low) and not _mentions(term, best.text):
            result._add(R_ROLE, f"role inflation '{term}' not supported by fact '{best.fact.id}'")

    # 5. CONTEXT drift — production claim on a non-production fact.
    if best.fact.context != "production":
        for term in _PRODUCTION_TERMS:
            if _mentions(term, bullet_low) and not _mentions(term, best.text):
                result._add(
                    R_CONTEXT,
                    f"context drift: fact '{best.fact.id}' is '{best.fact.context}', "
                    f"bullet claims '{term}'",
                )
                break

    # 6. FUSION — a distinctive number/tool from a DIFFERENT fact.
    for v in views:
        if v.fact.id == best.fact.id:
            continue
        foreign_nums = {n for n in (b_numbers & v.numbers) - best.numbers if _is_specific(n)}
        foreign_tools = {
            t for t in lexicon
            if _mentions(t, bullet_low) and _mentions(t, v.text) and not _mentions(t, best.text)
        }
        if foreign_nums or foreign_tools:
            ev = ", ".join(sorted(foreign_nums | foreign_tools))
            result._add(R_FUSION, f"claim fusion: also draws on fact '{v.fact.id}' ({ev})")

    result.ok = not result.reasons
    return result
