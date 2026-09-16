"""
pipeline.py

End-to-end customer support pipeline.

Flow:
  customer message
    -> intent classifier (intent + confidence)
    -> retrieval (top-k historical evidence)
    -> router (AUTO_HANDLE or ESCALATE)
    -> [if AUTO_HANDLE] response generator
    -> [if AUTO_HANDLE] grounding validator
    -> [if validator FAIL] ESCALATE
    -> structured JSON output

Usage:
    from pipeline import SupportPipeline
    pipeline = SupportPipeline()
    result = pipeline.run("spotify keeps crashing after update")
    print(result)
"""

from pathlib import Path
from dataclasses import dataclass, asdict
import json
import joblib

ROOT       = Path(__file__).resolve().parent.parent
SVC_PATH   = ROOT / "data" / "processed" / "intent_classifier.joblib"
PROBA_PATH = ROOT / "data" / "processed" / "intent_proba_lr.joblib"
INDEX_PATH = ROOT / "data" / "processed" / "retrieval_index.joblib"

TOP_K_RETRIEVAL = 3


@dataclass
class PipelineResult:
    # Routing
    route: str                  # AUTO_HANDLE or ESCALATE
    escalation_reason: str      # empty if AUTO_HANDLE

    # Classification
    intent: str
    intent_confidence: float

    # Retrieval
    evidence_score: float       # best similarity score
    evidence_count: int         # number of retrieved examples

    # Response (only populated for AUTO_HANDLE)
    response: str
    response_mode: str          # "template", "llm", or ""

    # Validation
    validation_passed: bool
    validation_reason: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SupportPipeline:
    """
    Loads all components once at init; run() is stateless per call.
    """

    def __init__(self, use_llm: bool = False):
        from intent_classifier import predict_with_confidence
        from retrieval import RetrievalIndex
        from router import route
        from response_generator import generate
        from validator import validate

        self._predict   = predict_with_confidence
        _index = RetrievalIndex.load(INDEX_PATH)
        self._retrieve  = _index.retrieve
        self._route     = route
        self._generate  = generate
        self._validate  = validate
        self._use_llm   = use_llm

        self._svc      = joblib.load(SVC_PATH)
        self._proba_lr = joblib.load(PROBA_PATH)

        print("Pipeline loaded.")

    def run(self, customer_message: str) -> PipelineResult:
        import pandas as pd

        # Step 1: classify intent
        intents, confs = self._predict(
            self._svc, self._proba_lr,
            pd.Series([customer_message])
        )
        intent     = intents[0]
        confidence = float(confs[0])

        # Step 2: retrieve evidence
        evidence = self._retrieve(customer_message, intent, top_k=TOP_K_RETRIEVAL)
        best_sim  = evidence[0]["similarity"] if evidence else 0.0

        # Step 3: route
        decision = self._route(intent, confidence, best_sim)

        if decision.route == "ESCALATE":
            return PipelineResult(
                route="ESCALATE",
                escalation_reason=decision.reason,
                intent=intent,
                intent_confidence=round(confidence, 4),
                evidence_score=round(best_sim, 4),
                evidence_count=len(evidence),
                response="",
                response_mode="",
                validation_passed=False,
                validation_reason="Escalated before generation.",
            )

        # Step 4: generate response
        generated = self._generate(
            customer_message, intent, evidence, use_llm=self._use_llm
        )

        # Step 5: validate
        validation = self._validate(
            generated.text, customer_message, evidence
        )

        if not validation.passed:
            return PipelineResult(
                route="ESCALATE",
                escalation_reason=f"Validation failed: {validation.reason}",
                intent=intent,
                intent_confidence=round(confidence, 4),
                evidence_score=round(best_sim, 4),
                evidence_count=len(evidence),
                response=generated.text,
                response_mode=generated.mode,
                validation_passed=False,
                validation_reason=validation.reason,
            )

        return PipelineResult(
            route="AUTO_HANDLE",
            escalation_reason="",
            intent=intent,
            intent_confidence=round(confidence, 4),
            evidence_score=round(best_sim, 4),
            evidence_count=len(evidence),
            response=generated.text,
            response_mode=generated.mode,
            validation_passed=True,
            validation_reason="",
        )


if __name__ == "__main__":
    pipeline = SupportPipeline()

    test_messages = [
        "spotify keeps crashing every time i open it after the latest update",
        "i was charged twice for premium this month",
        "my downloaded songs disappeared after i updated the app",
        "can you add a sleep timer feature to the mobile app",
        "i cant log in it says my password is wrong but i just reset it",
    ]

    for msg in test_messages:
        result = pipeline.run(msg)
        print(f"\nMessage: {msg[:70]}")
        print(f"  Intent:     {result.intent} ({result.intent_confidence:.2f})")
        print(f"  Route:      {result.route}")
        if result.route == "ESCALATE":
            print(f"  Reason:     {result.escalation_reason}")
        else:
            print(f"  Response:   {result.response[:100]}")
