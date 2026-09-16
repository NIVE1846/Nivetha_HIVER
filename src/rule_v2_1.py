"""Controlled Rule V2.1 extensions layered on top of Rule V2."""

from __future__ import annotations

import re
from typing import Any

from rule_v2 import label_v2

RULE_VERSION = "rule_v2_1"

_ADDITIONAL_RULES: dict[str, dict[str, str]] = {
    "playback_issue": {
        "audio_behavior_paraphrase": (
            r"\b(?:song|songs|track|tracks|music|audio)\b.{0,60}"
            r"\b(?:cut(?:s|ting)? out|pause[sd]?|go(?:es)? silent|no sound|"
            r"jump(?:s|ing)?|stop(?:s|ped|ping)? unexpectedly)\b|"
            r"\b(?:cut(?:s|ting)? out|pause[sd]?|go(?:es)? silent|no sound|"
            r"jump(?:s|ing)?|stop(?:s|ped|ping)? unexpectedly)\b.{0,60}"
            r"\b(?:song|songs|track|tracks|music|audio)\b"
        ),
    },
    "app_bug": {
        "app_state_paraphrase": (
            r"\b(?:spotify|the)\s+app\b.{0,60}"
            r"\b(?:hang(?:s|ing)?|freeze[sd]?|stuck|blank|black)\b|"
            r"\b(?:app|spotify)\b.{0,30}\b(?:loading|spinning)\b.{0,30}"
            r"\b(?:forever|stuck|won'?t stop)\b"
        ),
        "screen_state_paraphrase": (
            r"\b(?:screen|display)\b.{0,40}\b(?:blank|black|frozen|freeze[sd]?)\b"
        ),
    },
    "account_login": {
        "access_paraphrase": (
            r"\b(?:can'?t|cannot|unable to|not able to)\b.{0,30}"
            r"\b(?:get into|access|enter)\b.{0,30}"
            r"\b(?:my\s+)?(?:account|profile)\b"
        ),
        "sign_in_paraphrase": (
            r"\b(?:sign|log)\b.{0,10}\b(?:in|into)\b.{0,45}"
            r"\b(?:account|profile|spotify)\b"
        ),
    },
    "content_search": {
        "listen_availability_paraphrase": (
            r"\b(?:can'?t|cannot|unable to|not able to)\b.{0,35}"
            r"\b(?:listen to|find|locate)\b.{0,50}"
            r"\b(?:song|track|album|artist|playlist|podcast|music)\b"
        ),
        "catalog_availability_paraphrase": (
            r"\b(?:song|track|album|artist|playlist|podcast)\b.{0,50}"
            r"\b(?:unavailable|isn'?t available|not accessible|gone|disappeared)\b"
        ),
    },
    "download_offline": {
        "saved_content_sync": (
            r"\b(?:saved|downloaded)\b.{0,50}"
            r"\b(?:songs?|tracks?|albums?|playlists?|music)\b.{0,50}"
            r"\b(?:missing|gone|disappear(?:ed)?|sync(?:ing)?|not available)\b"
        ),
        "offline_content_paraphrase": (
            r"\boffline\b.{0,45}\b(?:songs?|tracks?|albums?|playlists?|music)\b|"
            r"\b(?:songs?|tracks?|albums?|playlists?|music)\b.{0,45}\boffline\b"
        ),
    },
    "premium_billing": {
        "plan_eligibility": (
            r"\b(?:eligible|eligibility|qualify|qualification)\b.{0,45}"
            r"\b(?:premium|plan|subscription|student|family|duo)\b|"
            r"\b(?:premium|plan|subscription|student|family|duo)\b.{0,45}"
            r"\b(?:eligible|eligibility|qualify|qualification)\b"
        ),
        "subscription_access": (
            r"\b(?:subscription|premium|student|family|duo)\b.{0,60}"
            r"\b(?:payment|pay|card|renew|renewal|cancel|upgrade|downgrade)\b|"
            r"\b(?:payment|pay|card|renew|renewal|cancel|upgrade|downgrade)\b.{0,60}"
            r"\b(?:subscription|premium|student|family|duo)\b"
        ),
    },
    "general_inquiry": {
        "explicit_information_question": (
            r"\b(?:is there a way|can you tell me|could you tell me|"
            r"what are the options|where can I|when will)\b.{0,80}"
            r"\b(?:feature|option|support|available|availability|work|use|"
            r"spotify|device|phone|car|tv)\b"
        ),
        "feature_how_to_paraphrase": (
            r"\b(?:how do I|how can I|where do I)\b.{0,80}"
            r"\b(?:use|enable|turn on|find|change|set up|create|share)\b"
        ),
    },
}


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def label_v2_1(text: str) -> dict[str, Any]:
    """Apply V2, then narrowly classify only V2-unresolved messages."""
    normalized = str(text or "").strip()
    v2 = label_v2(normalized)
    if v2["intent"] is not None:
        return {
            **v2,
            "rule_version": RULE_VERSION,
        }

    matched_rules = list(v2["matched_rules"])
    evidence: dict[str, bool] = {}
    for intent, rules in _ADDITIONAL_RULES.items():
        for rule_name, pattern in rules.items():
            if _matches(normalized, pattern):
                key = f"{intent}.{rule_name}"
                matched_rules.append(key)
                evidence[key] = True

    for intent in (
        "premium_billing",
        "account_login",
        "download_offline",
        "content_search",
        "playback_issue",
        "app_bug",
        "general_inquiry",
    ):
        if any(key.startswith(f"{intent}.") for key in evidence):
            reasons = {
                "premium_billing": "narrow subscription eligibility/access evidence",
                "account_login": "account access/sign-in paraphrase evidence",
                "download_offline": "saved/offline content evidence",
                "content_search": "catalog availability/search paraphrase evidence",
                "playback_issue": "audio playback paraphrase evidence",
                "app_bug": "app/screen state paraphrase evidence",
                "general_inquiry": "explicit information/how-to evidence",
            }
            return {
                "intent": intent,
                "matched_rules": matched_rules,
                "is_fallback": False,
                "rule_version": RULE_VERSION,
                "decision_reason": reasons[intent],
            }

    return {
        "intent": None,
        "matched_rules": matched_rules,
        "is_fallback": True,
        "rule_version": RULE_VERSION,
        "decision_reason": "no V2.1 rule supplied sufficient evidence",
    }
