"""Versioned, evidence-first labeling rules for the seven classifier intents."""

from __future__ import annotations

import re
from typing import Any

RULE_VERSION = "rule_v2"

_RULES = {
    "premium_billing": {
        "billing_context": (
            r"\bcharg(?:e|ed|es|ing)\b|\bpayment\b|\bpay\b|\bbilling\b|\bbill\b|"
            r"\brefund\b|\bsubscription\b|\bsubscribe\b|\bcancel(?:lation)?\b|"
            r"\brenew(?:al)?\b|\bcredit card\b|\binvoice\b|\breceipt\b|\btrial\b|"
            r"\bstudent (?:plan|discount)\b|\bfamily plan\b|\bduo plan\b|"
            r"\bupgrade\b|\bdowngrade\b"
        ),
        "premium_reference": r"\bpremium\b|\bpaid subscriber\b|\bspotify premium\b",
    },
    "download_offline": {
        "content_download": (
            r"\bdownload(?:ed|ing|s)?\b.*\b(?:song|songs|track|tracks|album|"
            r"playlist|music|library|content)\b|"
            r"\b(?:song|songs|track|tracks|album|playlist|music|library|content)"
            r"\b.*\bdownload(?:ed|ing|s)?\b"
        ),
        "offline": r"\boffline\b|\boffline listening\b|\boffline mode\b",
        "download_problem": (
            r"\bdownload limit\b|\bdownload(?:s|ed)?\b.*\b(?:disappear|gone|"
            r"missing|sync|work|fail|stuck|available)\b|"
            r"\b(?:can't|cannot|can not|won't|will not|unable to)\b.*\bdownload\b"
        ),
        "sync": r"\bsync(?:ing)?\b.*\b(?:song|playlist|music|download)\b|\bdownload.*\bsync",
    },
    "account_login": {
        "authentication": (
            r"\blog ?in\b|\blogin\b|\blog into\b|\bsign ?in\b|\bsign ?out\b|"
            r"\bpassword\b|\bforgot (?:my )?password\b|\breset\b.*\bpassword\b|"
            r"\baccount access\b|\baccess (?:my|the) account\b|\blocked account\b|"
            r"\bunable to access\b|\bfacebook\b.*\blogin\b|\bgoogle\b.*\blogin\b"
        ),
        "credential_problem": (
            r"\bwrong email\b|\bwrong password\b|\baccount email\b|\blogin email\b|"
            r"\bcan't access (?:my )?account\b|\bcannot access (?:my )?account\b"
        ),
    },
    "content_search": {
        "catalog_content": (
            r"\bsong\b|\btrack\b|\balbum\b|\bartist\b|\bplaylist\b|\bpodcast\b|"
            r"\bmusic\b|\blibrary\b|\bcatalog(?:ue)?\b|\bcontent\b"
        ),
        "content_missing": (
            r"\bcan't find\b|\bcannot find\b|\bcan not find\b|\bnot find\b|"
            r"\bmissing\b|\bdisappear(?:ed)?\b|\bnot available\b|\bunavailable\b|"
            r"\bnot on spotify\b|\bnot showing\b|\bnot appear\b|\bsearch results?\b|"
            r"\bwhere is\b"
        ),
        "regional_content": r"\b(?:country|region|regional)\b.*\b(?:available|spotify)\b|\bavailable\b.*\b(?:country|region)\b",
    },
    "playback_issue": {
        "playback": (
            r"\bwon't play\b|\bwill not play\b|\bnot play(?:ing)?\b|\bstop(?:s|ped)?\b"
            r".*\bplay(?:ing)?\b|\bskip(?:s|ping|ped)?\b|\bshuffle\b|\brepeat\b|"
            r"\bplayback\b|\bbuffer(?:ing)?\b|\baudio\b.*\b(?:cut|drop)\b|"
            r"\b(?:song|music|track)\b.*\bstop(?:s|ped)?\b|\bplay(?:s|ed)?\b.*\b(?:random|order)\b"
        ),
        "playback_freeze": r"\b(?:playback|song|music|audio)\b.*\bfreez(?:e|es|ing)\b",
    },
    "app_bug": {
        "crash_or_load": (
            r"\bapp\b.*\b(?:crash(?:es|ed|ing)?|force.?close|won't open|will not open|won't load|"
            r"will not load)\b|\b(?:crash|force.?close|blank screen|black screen)\b"
        ),
        "error_or_ui": (
            r"\berror(?: code)?\b|\berror \d+\b|\bspinner\b|\bloading\b.*\bnever\b|"
            r"\b(?:button|menu|screen|interface|ui|swipe)\b.*\b(?:missing|broken|"
            r"disabled|disappear|not work|doesn't work)\b"
        ),
        "specific_malfunction": (
            r"\bspotify app\b.*\b(?:not work|doesn't work|won't|broken|glitch)\b|"
            r"\b(?:after|since)\b.*\bupdate\b.*\b(?:crash|freeze|error|broken|"
            r"not work|doesn't work)\b"
        ),
    },
    "general_inquiry": {
        "feature_request": r"\b(?:feature|suggest|wish|please add|can you add|would be nice|option to|ability to)\b",
        "how_to": r"\bhow (?:do|can) i\b|\bis it possible\b|\bhow does\b|\bwhat is\b",
        "availability_or_compatibility": r"\bavailable in\b|\bcompatible\b|\bwork on\b.*\b(?:phone|device|tv|car)\b",
        "product_feedback": r"\b(?:love|great|awesome|amazing)\b.*\bspotify\b|\bfeedback\b",
    },
}


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def label_v2(text: str) -> dict[str, Any]:
    """Classify with evidence-first rules; unresolved messages remain unlabeled."""
    normalized = str(text or "").strip()
    matched_rules: list[str] = []
    evidence: dict[str, bool] = {}

    for intent, rules in _RULES.items():
        for rule_name, pattern in rules.items():
            if _matches(normalized, pattern):
                rule_key = f"{intent}.{rule_name}"
                matched_rules.append(rule_key)
                evidence[rule_key] = True

    billing = evidence.get("premium_billing.billing_context", False)
    download = any(
        evidence.get(f"download_offline.{key}", False)
        for key in ("content_download", "offline", "download_problem", "sync")
    )
    login = any(
        evidence.get(f"account_login.{key}", False)
        for key in ("authentication", "credential_problem")
    )
    content = (
        evidence.get("content_search.catalog_content", False)
        and (
            evidence.get("content_search.content_missing", False)
            or evidence.get("content_search.regional_content", False)
        )
    )
    playback = any(
        evidence.get(f"playback_issue.{key}", False)
        for key in ("playback", "playback_freeze")
    )
    app = any(
        evidence.get(f"app_bug.{key}", False)
        for key in ("crash_or_load", "error_or_ui", "specific_malfunction")
    )
    general = bool(
        evidence.get("general_inquiry.feature_request")
        or evidence.get("general_inquiry.how_to")
        or evidence.get("general_inquiry.availability_or_compatibility")
        or evidence.get("general_inquiry.product_feedback")
    )

    # Specific support actions take precedence over generic wording.
    if billing:
        intent = "premium_billing"
        reason = "billing/subscription evidence"
    elif login:
        intent = "account_login"
        reason = "authentication/account-access evidence"
    elif download:
        intent = "download_offline"
        reason = "content download/offline evidence"
    elif content:
        intent = "content_search"
        reason = "catalog/library content evidence"
    elif playback:
        intent = "playback_issue"
        reason = "playback behavior evidence"
    elif app:
        intent = "app_bug"
        reason = "specific app/UI malfunction evidence"
    elif general:
        intent = "general_inquiry"
        reason = "explicit general-inquiry evidence"
    else:
        intent = None
        reason = "no rule supplied sufficient evidence"

    return {
        "intent": intent,
        "matched_rules": matched_rules,
        "is_fallback": intent is None,
        "rule_version": RULE_VERSION,
        "decision_reason": reason,
    }
