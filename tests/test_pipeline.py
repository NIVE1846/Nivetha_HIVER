"""
tests/test_pipeline.py

Basic tests for all major components.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ── Validator tests ────────────────────────────────────────────────────────

from validator import validate, ValidationResult

GOOD_EVIDENCE = [{"brand_response": "Try reinstalling the app and restarting your device."}]


def test_validator_passes_good_response():
    result = validate(
        "Try reinstalling the app and restarting your device. Let us know!",
        "spotify crashes every time i open it",
        GOOD_EVIDENCE,
    )
    assert result.passed is True


def test_validator_fails_refund_claim():
    result = validate(
        "We will refund your subscription within 24 hours.",
        "i was charged twice",
        GOOD_EVIDENCE,
    )
    assert result.passed is False
    assert "refund" in result.reason.lower()


def test_validator_fails_guarantee():
    result = validate(
        "We guarantee this will be fixed by tomorrow.",
        "app not working",
        GOOD_EVIDENCE,
    )
    assert result.passed is False


def test_validator_fails_empty_response():
    result = validate("", "help me", GOOD_EVIDENCE)
    assert result.passed is False


def test_validator_fails_too_short():
    result = validate("ok", "help me", GOOD_EVIDENCE)
    assert result.passed is False


# ── Router tests ───────────────────────────────────────────────────────────

from router import route, CONF_THRESHOLD, SIM_THRESHOLD, HIGH_RISK_INTENTS


def test_router_auto_handle():
    decision = route("app_bug", 0.92, 0.35)
    assert decision.route == "AUTO_HANDLE"


def test_router_escalates_low_confidence():
    decision = route("app_bug", CONF_THRESHOLD - 0.01, 0.35)
    assert decision.route == "ESCALATE"
    assert "confidence" in decision.reason.lower()


def test_router_escalates_low_evidence():
    decision = route("app_bug", 0.92, SIM_THRESHOLD - 0.01)
    assert decision.route == "ESCALATE"
    assert "evidence" in decision.reason.lower()


def test_router_escalates_premium_billing():
    decision = route("premium_billing", 0.99, 0.99)
    assert decision.route == "ESCALATE"
    assert "premium_billing" in decision.reason


def test_router_escalates_account_login():
    decision = route("account_login", 0.99, 0.99)
    assert decision.route == "ESCALATE"


def test_router_escalates_all_high_risk_intents():
    for intent in HIGH_RISK_INTENTS:
        decision = route(intent, 0.99, 0.99)
        assert decision.route == "ESCALATE", f"{intent} should always escalate"


# ── Intent classifier tests ────────────────────────────────────────────────

import joblib
from intent_classifier import predict_with_confidence

ROOT       = Path(__file__).resolve().parent.parent
SVC_PATH   = ROOT / "data" / "processed" / "intent_classifier.joblib"
PROBA_PATH = ROOT / "data" / "processed" / "intent_proba_lr.joblib"


@pytest.fixture(scope="module")
def classifier():
    svc      = joblib.load(SVC_PATH)
    proba_lr = joblib.load(PROBA_PATH)
    return svc, proba_lr


def test_classifier_returns_valid_intent(classifier):
    from utils import INTENTS
    svc, proba_lr = classifier
    intents, confs = predict_with_confidence(svc, proba_lr,
                                             pd.Series(["spotify keeps crashing"]))
    assert intents[0] in INTENTS
    assert 0.0 <= confs[0] <= 1.0


def test_classifier_confidence_is_probability(classifier):
    svc, proba_lr = classifier
    _, confs = predict_with_confidence(svc, proba_lr,
                                       pd.Series(["i was charged twice for premium"]))
    assert 0.0 <= confs[0] <= 1.0


# ── Retrieval tests ────────────────────────────────────────────────────────

from retrieval import RetrievalIndex

INDEX_PATH = ROOT / "data" / "processed" / "retrieval_index.joblib"


@pytest.fixture(scope="module")
def retrieval_index():
    return RetrievalIndex.load(INDEX_PATH)


def test_retrieval_returns_results(retrieval_index):
    results = retrieval_index.retrieve(
        "spotify keeps crashing after update", "app_bug", top_k=3
    )
    assert len(results) > 0


def test_retrieval_result_has_required_fields(retrieval_index):
    results = retrieval_index.retrieve(
        "my downloads disappeared", "download_offline", top_k=1
    )
    assert len(results) == 1
    r = results[0]
    assert "customer_message" in r
    assert "brand_response" in r
    assert "similarity" in r
    assert 0.0 <= r["similarity"] <= 1.0


def test_retrieval_unknown_intent_returns_empty(retrieval_index):
    results = retrieval_index.retrieve("test", "nonexistent_intent", top_k=3)
    assert results == []


# ── Conversation builder tests ─────────────────────────────────────────────

def test_interactions_file_exists():
    path = ROOT / "data" / "processed" / "interactions.csv"
    assert path.exists()


def test_interactions_no_empty_messages():
    path = ROOT / "data" / "processed" / "interactions.csv"
    df = pd.read_csv(path, dtype=str)
    assert (df["customer_message"].str.strip() == "").sum() == 0
    assert (df["brand_response"].str.strip() == "").sum() == 0


def test_interactions_has_required_columns():
    path = ROOT / "data" / "processed" / "interactions.csv"
    df = pd.read_csv(path, dtype=str)
    required = {"interaction_id", "customer_message", "brand_response",
                "customer_tweet_id", "brand_tweet_id", "brand"}
    assert required.issubset(set(df.columns))


# ── Pipeline integration test ──────────────────────────────────────────────

def test_pipeline_low_confidence_escalates():
    """
    A very short/ambiguous message should either escalate due to low confidence
    or be auto-handled — but must never crash.
    """
    from pipeline import SupportPipeline
    p = SupportPipeline()
    result = p.run("help")
    assert result.route in ("AUTO_HANDLE", "ESCALATE")
    assert result.intent != ""
    assert 0.0 <= result.intent_confidence <= 1.0


def test_pipeline_billing_always_escalates():
    from pipeline import SupportPipeline
    p = SupportPipeline()
    result = p.run("i was charged twice for premium this month please refund me")
    assert result.route == "ESCALATE"


def test_pipeline_result_has_all_fields():
    from pipeline import SupportPipeline
    p = SupportPipeline()
    result = p.run("spotify app keeps crashing after the latest update")
    assert hasattr(result, "route")
    assert hasattr(result, "intent")
    assert hasattr(result, "intent_confidence")
    assert hasattr(result, "evidence_score")
    assert hasattr(result, "response")
    assert hasattr(result, "validation_passed")
