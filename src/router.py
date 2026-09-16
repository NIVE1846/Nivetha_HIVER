"""
router.py

Risk-based routing: decides AUTO_HANDLE vs ESCALATE.

Risk levels:
  HIGH RISK  -> always ESCALATE (premium_billing, account_login)
  MEDIUM RISK -> ESCALATE if confidence < threshold OR evidence < threshold
  LOW RISK   -> AUTO_HANDLE if confidence >= threshold AND evidence >= threshold

Routing decision combines:
  1. Intent (determines risk level)
  2. Intent confidence (from calibrated classifier)
  3. Evidence quality (best retrieval similarity)
  4. Validator result (checked after generation, not here)

Thresholds:
  CONF_THRESHOLD = 0.70  (selected by validation sweep, see reports/router_threshold_search.md)
  SIM_THRESHOLD  = 0.10  (minimum evidence similarity to auto-handle)

Why these thresholds:
  - conf=0.70, sim=0.10 gives 93% auto-handle rate with 2.5% unsafe rate on validation set
  - More conservative thresholds (conf=0.90) reduce auto-handle rate to 90% with
    only marginal precision improvement (97.9% vs 96.6%)
  - The classifier is overconfident on minority intents (see reports/routing_analysis.md)
    so confidence alone is not sufficient — evidence quality is also required

Why premium_billing and account_login always escalate:
  - Billing errors cause direct financial harm to customers
  - Wrong account advice can lock customers out permanently
  - The classifier achieves F1=0.90 for billing on the test split, meaning ~10% error rate
  - A 10% error rate on financial advice is unacceptable in a real support system
  - This is a deliberate conservative policy, not a technical limitation
"""

from pathlib import Path
from dataclasses import dataclass, asdict
import json

ROOT = Path(__file__).resolve().parent.parent

CONF_THRESHOLD    = 0.70
SIM_THRESHOLD     = 0.10
HIGH_RISK_INTENTS = {"premium_billing", "account_login"}

RISK_LEVELS = {
    "premium_billing":  "HIGH",
    "account_login":    "HIGH",
    "playback_issue":   "LOW",
    "app_bug":          "LOW",
    "download_offline": "LOW",
    "content_search":   "LOW",
    "general_inquiry":  "LOW",
}


@dataclass
class RoutingDecision:
    route: str              # "AUTO_HANDLE" or "ESCALATE"
    reason: str
    intent: str
    risk_level: str         # "HIGH", "MEDIUM", "LOW"
    intent_confidence: float
    evidence_score: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def route(intent: str,
          intent_confidence: float,
          evidence_score: float) -> RoutingDecision:
    """
    Apply risk-based routing rules.

    Args:
        intent:             predicted intent label
        intent_confidence:  calibrated probability (0-1)
        evidence_score:     best cosine similarity from retrieval (0-1)
    """
    risk = RISK_LEVELS.get(intent, "MEDIUM")
    conf = round(intent_confidence, 4)
    sim  = round(evidence_score, 4)

    # Rule 1: HIGH RISK intents always escalate
    if risk == "HIGH":
        return RoutingDecision(
            route="ESCALATE",
            reason=(
                f"Intent '{intent}' is HIGH RISK (billing/account). "
                "Requires human review regardless of confidence."
            ),
            intent=intent,
            risk_level="HIGH",
            intent_confidence=conf,
            evidence_score=sim,
        )

    # Rule 2: Low confidence -> MEDIUM RISK -> escalate
    if intent_confidence < CONF_THRESHOLD:
        return RoutingDecision(
            route="ESCALATE",
            reason=(
                f"Intent confidence {intent_confidence:.2f} is below threshold "
                f"{CONF_THRESHOLD}. Uncertain classification."
            ),
            intent=intent,
            risk_level="MEDIUM",
            intent_confidence=conf,
            evidence_score=sim,
        )

    # Rule 3: Weak evidence -> MEDIUM RISK -> escalate
    if evidence_score < SIM_THRESHOLD:
        return RoutingDecision(
            route="ESCALATE",
            reason=(
                f"Retrieved evidence similarity {evidence_score:.3f} is below "
                f"threshold {SIM_THRESHOLD}. Insufficient historical evidence "
                "to ground a response."
            ),
            intent=intent,
            risk_level="MEDIUM",
            intent_confidence=conf,
            evidence_score=sim,
        )

    # All checks passed -> LOW RISK -> auto-handle
    return RoutingDecision(
        route="AUTO_HANDLE",
        reason=(
            f"Intent confidence {intent_confidence:.2f} >= {CONF_THRESHOLD} "
            f"and evidence similarity {evidence_score:.3f} >= {SIM_THRESHOLD}. "
            "LOW RISK intent with sufficient evidence."
        ),
        intent=intent,
        risk_level="LOW",
        intent_confidence=conf,
        evidence_score=sim,
    )


if __name__ == "__main__":
    cases = [
        ("app_bug",         0.92, 0.35),
        ("app_bug",         0.45, 0.35),
        ("app_bug",         0.92, 0.05),
        ("premium_billing", 0.95, 0.80),
        ("account_login",   0.95, 0.80),
        ("playback_issue",  0.85, 0.20),
    ]
    print("Router smoke test:")
    for intent, conf, sim in cases:
        d = route(intent, conf, sim)
        print(f"  {intent:<20} conf={conf:.2f} sim={sim:.2f} -> {d.route} [{d.risk_level}]")
        print(f"    {d.reason}")
