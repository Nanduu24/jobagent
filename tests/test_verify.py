"""Adversarial verifier tests — the real deliverable.

Each row is a hand-written bullet against the REAL fact bank with an expected
verdict AND (for fabrications) the exact violation reason that must fire. Bars:
  - ZERO false-accepts (a fabrication that passes) — blocking.
  - ZERO right-verdict-wrong-reason on the mapping cases — the expected reason
    code must be present, so a refactor that breaks one check can't hide behind
    a coincidental rejection by another.
"""
from __future__ import annotations

import pytest

from jobagent.factbank import load_fact_bank
from jobagent.tailor.verify import (
    R_AMBIGUOUS,
    R_CONTEXT,
    R_FUSION,
    R_METRIC,
    R_NO_MAP,
    R_NUMBER,
    R_ROLE,
    R_TOOL,
    verify_bullet,
)

FB = load_fact_bank("tests/fixtures/fact_bank.json")

# (id, bullet, claimed_fact_id, expected_ok, expected_reason, note)
# Rewritten against the ENRICHED fact bank (resume-derived facts). verify.py is
# unchanged; every reason code is still exercised and the 0-false-accept bar holds.
CASES: list[tuple[str, str, str | None, bool, str | None, str]] = [
    # ---------------- CLEAN (must PASS, inference) ----------------------
    ("clean-exp-pipelines",
     "Built end-to-end Python ML pipelines (scikit-learn, NumPy, pandas) for "
     "regression on ~720 samples and 11 engineered features across 6 orders of "
     "magnitude.", None, True, None, "faithful exp-pipelines paraphrase"),
    ("clean-exp-modeling",
     "Implemented Random Forest, DNN (TensorFlow/Keras), and CNN regressors with "
     "Optuna TPE tuning (50 trials, 5-fold CV)", None, True, None, "exp-modeling variant"),
    ("clean-rag-variant",
     "Implemented pgvector HNSW vector index backing agent memory at sub-200ms "
     "retrieval latency", None, True, None, "mindful-rag variant"),
    ("clean-langgraph-variant",
     "Built a 6-node LangGraph stateful agent with a provider-agnostic LLM "
     "abstraction across Groq Llama 3.3, Gemini 2.0 Flash, and Anthropic Claude",
     None, True, None, "mindful-langgraph variant"),
    ("clean-i2ct-variant",
     "First-author IEEE I2CT 2024 paper on LSTM sign-language recognition (96% "
     "accuracy)", None, True, None, "i2ct-paper variant"),
    ("clean-exp-tuning",
     "Tuned Random Forest, DNN, and CNN models via Optuna TPE (50 trials, 5-fold "
     "CV)", None, True, None, "exp-modeling tuning variant"),
    ("clean-fleetiq-variant",
     "Hackathon build: document-RAG agent with multi-LLM routing, OCR over 1,000+ "
     "synthetic carrier forms, and a 20-question eval harness", None, True, None,
     "fleetiq variant"),
    ("clean-advisor-tests-variant",
     "Wrote 25+ Jest and integration tests and a Selenium E2E suite over auth and "
     "ATS scoring flows", None, True, None, "advisor-tests variant"),
    ("clean-messenger-variant",
     "Built a Messenger auto-reply agent (LangGraph, OAuth multi-account, "
     "human-in-the-loop approval inbox)", None, True, None, "messenger variant"),
    ("clean-voice-variant",
     "Built a real-time push-to-talk voice pipeline (MediaRecorder to Gemini STT "
     "to LangGraph to ElevenLabs TTS)", None, True, None, "mindful-voice variant"),
    ("clean-exp-robustness",
     "Applied SHAP TreeExplainer and Spearman rank correlation to show 4 of 11 "
     "features capture ~95% of output", None, True, None, "exp-robustness variant"),
    ("clean-deploy-variant",
     "Deployed a live agentic AI platform across 3 cloud services (Vercel, Hugging "
     "Face Spaces, Supabase) with Stripe billing, Clerk auth, and LangSmith "
     "production tracing", None, True, None, "mindful-deploy variant"),
    ("clean-integrations-variant",
     "Integrated Greenhouse and Lever APIs and a FastAPI microservice with JWT "
     "auth, Zod validation, and Prisma ORM", None, True, None, "advisor-integrations"),
    ("clean-jobagent-variant",
     "Built an async multi-source ingestion pipeline (4 ATS APIs, concurrent, "
     "rate-limited, idempotent) with a cost-aware two-stage scoring layer", None,
     True, None, "jobagent variant"),
    ("clean-modeling-tools",
     "Implemented Random Forest, DNN (TensorFlow/Keras), and CNN regressors in a "
     "reproducible training framework with Optuna TPE tuning.", None, True,
     None, "exp-modeling: all tools in the source claim"),
    # ---------------- FABRICATIONS (must FAIL, reason pinned) -----------
    ("fab-invented-number",
     "Implemented Random Forest and CNN regressors achieving a Test R2 of 0.72",
     None, False, R_NUMBER, "0.72 invented (no R2 in exp-modeling)"),
    ("fab-inflated-scale-samples",
     "Built Python ML pipelines on ~820 samples and 11 engineered features",
     None, False, R_NUMBER, "820 invented (720 real)"),
    ("fab-context-drift-fleetiq",
     "Production system serving live traffic: a document-RAG agent with OCR over "
     "1,000+ carrier forms", "fleetiq", False, R_CONTEXT, "hackathon -> production"),
    ("fab-latency-number",
     "Built pgvector HNSW retrieval for agent long-term memory, serving sub-50ms "
     "queries", None, False, R_NUMBER, "sub-50ms invented"),
    ("fab-fusion-i2ct-exp",
     "Published an IEEE LSTM sign-language paper (96% accuracy) validated across "
     "720 training samples", "i2ct-paper", False, R_FUSION,
     "welds i2ct 96% with exp's 720"),
    ("fab-role-inflation-led",
     "Led a 3-person Agile team building the Job Search and ATS Scoring modules",
     None, False, R_ROLE, "'led' not in advisor-modules"),
    ("fab-invented-tool-k8s",
     "Built a Messenger auto-reply agent with LangGraph and OAuth, deployed on "
     "Kubernetes", None, False, R_TOOL, "Kubernetes invented"),
    ("fab-invented-users-1m",
     "Built a 6-node LangGraph agent in production serving 1M+ users", None, False,
     R_NUMBER, "1M+ scale invented"),
    ("fab-inflated-feature-pct",
     "Showed via SHAP that 4 of 11 features capture ~99% of model output", None,
     False, R_NUMBER, "99% invented (95% real)"),
    ("fab-invented-tool-airflow",
     "Built an async ingestion pipeline over 4 ATS APIs, orchestrated with Airflow",
     None, False, R_TOOL, "Airflow invented"),
    ("fab-invented-payments",
     "Integrated Stripe billing processing $2M in payments across the product",
     None, False, R_NUMBER, "$2M invented"),
    ("fab-accuracy-inflation",
     "IEEE I2CT 2024 paper on LSTM sign-language recognition, 99% accuracy", None,
     False, R_NUMBER, "99% invented"),
    ("fab-fleetiq-doc-inflation",
     "Document-RAG agent with OCR over 10,000+ synthetic carrier forms and a "
     "20-question eval harness", None, False, R_NUMBER, "10,000+ invented"),
    ("fab-role-inflation-jobagent",
     "Led development of an async ingestion pipeline over 4 ATS APIs", None, False,
     R_ROLE, "'led' invented"),
    ("fab-no-mapping",
     "Designed a blockchain-based NFT marketplace with Solidity smart contracts",
     None, False, R_NO_MAP, "no matching fact"),
    # ---------------- MAPPING-SEAM attacks ------------------------------
    ("fab-number-elsewhere",
     "Implemented Random Forest and CNN regressors reaching 96% accuracy",
     "exp-modeling", False, R_NUMBER,
     "96% is real but belongs to i2ct, not this fact"),
    ("fab-borrowed-tool-shap",
     "Applied SHAP feature attribution to the pgvector HNSW retrieval memory index",
     "mindful-rag", False, R_TOOL, "SHAP borrowed from exp-robustness"),
    ("fab-context-drift-exp",
     "Deployed the Random Forest and CNN regressors to production for live scoring",
     "exp-modeling", False, R_CONTEXT, "must fire via CONTEXT, not tool"),
    ("fab-wrong-metric",
     "Delivered an LSTM sign-language model with 96% precision", "i2ct-paper",
     False, R_METRIC, "96 is real but bound to accuracy, not precision"),
    ("fab-ambiguous-inference",
     "Built a LangGraph agent with a FastAPI backend", None, False, R_AMBIGUOUS,
     "ties across LangGraph/FastAPI facts -> fail closed"),
]


@pytest.mark.parametrize(
    "cid, bullet, claimed, exp_ok, exp_reason, note", CASES, ids=[c[0] for c in CASES]
)
def test_verify_bullet(
    cid: str, bullet: str, claimed: str | None, exp_ok: bool,
    exp_reason: str | None, note: str,
) -> None:
    r = verify_bullet(bullet, FB, claimed_fact_id=claimed)
    assert r.ok is exp_ok, (
        f"[{cid}] expected ok={exp_ok} ({note}); got ok={r.ok} "
        f"fact_id={r.fact_id} reasons={r.reasons} violations={r.violations}"
    )
    if exp_ok:
        assert not r.reasons
        assert r.fact_id is not None
    else:
        # Reason-pinning: the intended defense must have fired.
        assert exp_reason in r.reasons, (
            f"[{cid}] expected reason '{exp_reason}' but got {r.reasons} ({note})"
        )


def test_no_fabrication_passes() -> None:
    """Blocking bar: no fabrication may pass."""
    leaks = [
        c[0] for c in CASES if not c[3] and verify_bullet(c[1], FB, c[2]).ok
    ]
    assert leaks == [], f"FALSE-ACCEPTS (fabrications that passed): {leaks}"


def test_context_case_fails_via_context_not_tool() -> None:
    """The seam fix: with the claimed fact, context drift must fire and the tool
    check must NOT (RF/CNN legitimately belong to exp-modeling)."""
    r = verify_bullet(
        "Deployed the Random Forest and CNN regressors to production for live scoring",
        FB, claimed_fact_id="exp-modeling",
    )
    assert R_CONTEXT in r.reasons
    assert R_TOOL not in r.reasons


def test_claimed_fact_overrides_inference() -> None:
    # A number real elsewhere (i2ct's 96%) but absent from the CLAIMED fact fails.
    r = verify_bullet("Reached 96% accuracy.", FB, claimed_fact_id="exp-modeling")
    assert r.ok is False and R_NUMBER in r.reasons


def test_unknown_claimed_fact() -> None:
    r = verify_bullet("Built something.", FB, claimed_fact_id="does-not-exist")
    assert r.ok is False and r.reasons == ["unknown_claimed_fact"]
