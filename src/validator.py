"""
validator.py

Grounding validator: checks a generated response before it is sent.

Seven checks (in order of severity):
  1. Length check — response is not empty or trivially short
  2. Evidence availability — was any evidence retrieved?
  3. Hallucination signals — invented policies, refunds, guarantees, timelines
  4. Excessive certainty — response makes promises the evidence does not support
  5. Relevance — response shares vocabulary with customer message or evidence
  6. Contradiction — response contradicts a clear statement in the evidence
  7. Missing context — response asks for info the customer already provided

If any check fails -> FAIL + reason -> pipeline escalates.

Design principle: prefer false positives (unnecessary escalations) over
false negatives (sending a bad response). Safety > throughput.

DOCUMENTED LIMITATIONS:
- Checks 1-4 are keyword-based and will miss paraphrased hallucinations.
  e.g. "we'll process your money back" is not caught by the refund pattern.
- Check 5 (relevance) uses token overlap which is a weak proxy.
  A response can share tokens with the query but still be off-topic.
- Check 6 (contradiction) only catches a small set of known contradiction patterns.
- Check 7 (missing context) is heuristic and will have false positives.
- The validator cannot detect factually incorrect troubleshooting steps.
- The validator cannot detect responses that are technically correct but unhelpful.
"""

from dataclasses import dataclass
import re

MIN_RESPONSE_LENGTH = 15

# Phrases that indicate the response may have invented something
HALLUCINATION_PATTERNS = [
    (r"\brefund\b",                              "mentions refund (not evidenced)"),
    (r"\bcredit\b",                              "mentions credit (not evidenced)"),
    (r"\bcompensate\b",                          "mentions compensation"),
    (r"\bwithin \d+ (hour|day|business)",        "promises a specific timeline"),
    (r"\bguarantee\b",                           "uses guarantee language"),
    (r"\bwe will (fix|resolve|send|give|provide)", "promises a specific action"),
    (r"\byour account has been\b",               "claims account action was taken"),
    (r"\bi have (updated|changed|reset|cancelled|processed)\b",
                                                 "claims to have performed an action"),
    (r"\bfree (month|subscription|premium)\b",   "offers free subscription"),
    (r"\bno charge\b",                           "claims no charge (not evidenced)"),
]

# Excessive certainty patterns
CERTAINTY_PATTERNS = [
    (r"\bthis will (definitely|certainly|absolutely) (fix|work|resolve)\b",
     "makes a definitive fix promise"),
    (r"\byou will (definitely|certainly) (get|receive|see)\b",
     "makes a definitive outcome promise"),
    (r"\b100%\b",
     "uses 100% certainty language"),
]

# Contradiction pairs: evidence says A, response says B
CONTRADICTION_PAIRS = [
    (r"not (possible|available|supported)", r"\byes\b|\bsure\b|we can do that|it is possible"),
    (r"contact us directly",               r"i can help you (here|now|directly)"),
    (r"not available in your (country|region)", r"available in your (country|region)"),
]

# Patterns suggesting the response asks for info already in the message
REDUNDANT_REQUEST_PATTERNS = [
    # If customer mentioned their device, don't ask for device
    (r"iphone|android|samsung|pixel|windows|mac|ios \d|android \d",
     r"what (device|phone|operating system) are you using"),
    # If customer mentioned their version, don't ask for version
    (r"version \d|spotify \d|ios \d+\.\d",
     r"what (version|spotify version) are you (on|using)"),
]


@dataclass
class ValidationResult:
    passed: bool
    reason: str   # empty string if passed
    check_name: str  # which check failed


def _token_overlap(text_a: str, text_b: str, min_overlap: int = 2) -> bool:
    stop = {"i","a","the","to","is","it","and","in","of","you","we","can",
            "for","on","my","your","this","that","be","us","just","let","know"}
    tokens_a = {w for w in re.sub(r"[^a-z0-9 ]"," ",text_a.lower()).split()
                if w not in stop and len(w) > 2}
    tokens_b = {w for w in re.sub(r"[^a-z0-9 ]"," ",text_b.lower()).split()
                if w not in stop and len(w) > 2}
    return len(tokens_a & tokens_b) >= min_overlap


def validate(response_text: str,
             customer_message: str,
             retrieved_evidence: list[dict]) -> ValidationResult:
    """
    Validate a generated response before sending.

    Args:
        response_text:      the generated response to validate
        customer_message:   the original customer message
        retrieved_evidence: list of retrieved evidence dicts

    Returns:
        ValidationResult(passed=True, reason="", check_name="") if all checks pass.
        ValidationResult(passed=False, reason=..., check_name=...) if any check fails.
    """
    resp = response_text.strip()
    resp_lower = resp.lower()

    # Check 1: length
    if len(resp) < MIN_RESPONSE_LENGTH:
        return ValidationResult(False, "Response is too short to be useful.", "length")

    # Check 2: evidence availability
    if not retrieved_evidence:
        return ValidationResult(
            False,
            "No historical evidence was retrieved. Cannot validate groundedness.",
            "evidence_availability"
        )

    # Check 3: hallucination signals
    for pattern, description in HALLUCINATION_PATTERNS:
        if re.search(pattern, resp_lower):
            return ValidationResult(
                False,
                f"Response {description} — not supported by retrieved evidence.",
                "hallucination"
            )

    # Check 4: excessive certainty
    for pattern, description in CERTAINTY_PATTERNS:
        if re.search(pattern, resp_lower):
            return ValidationResult(
                False,
                f"Response {description}.",
                "excessive_certainty"
            )

    # Check 5: relevance
    evidence_texts = " ".join(e.get("brand_response", "") for e in retrieved_evidence)
    has_customer_overlap = _token_overlap(resp, customer_message)
    has_evidence_overlap = _token_overlap(resp, evidence_texts)
    if not has_customer_overlap and not has_evidence_overlap:
        return ValidationResult(
            False,
            "Response shares no vocabulary with the customer message or evidence.",
            "relevance"
        )

    # Check 6: contradiction
    for evidence_pattern, response_pattern in CONTRADICTION_PAIRS:
        evidence_says = any(
            re.search(evidence_pattern, e.get("brand_response","").lower())
            for e in retrieved_evidence
        )
        response_says = re.search(response_pattern, resp_lower)
        if evidence_says and response_says:
            return ValidationResult(
                False,
                "Response contradicts a statement in the retrieved evidence.",
                "contradiction"
            )

    # Check 7: missing context (response asks for info customer already provided)
    cust_lower = customer_message.lower()
    for customer_pattern, response_pattern in REDUNDANT_REQUEST_PATTERNS:
        customer_has_info = re.search(customer_pattern, cust_lower)
        response_asks_for_it = re.search(response_pattern, resp_lower)
        if customer_has_info and response_asks_for_it:
            return ValidationResult(
                False,
                "Response asks for information the customer already provided.",
                "missing_context"
            )

    return ValidationResult(True, "", "")


if __name__ == "__main__":
    evidence = [{"brand_response": "Try reinstalling the app and restarting your device."}]
    cases = [
        ("Try reinstalling the app and restarting your device. Let us know!", "spotify crashes", True),
        ("We will refund your subscription within 24 hours.", "i was charged twice", False),
        ("", "help me", False),
        ("We guarantee this will be fixed by tomorrow.", "app not working", False),
        ("What device are you using?", "I'm on iPhone 7 and it crashes", False),
    ]
    print("Validator smoke test:")
    for response, customer, expected_pass in cases:
        result = validate(response, customer, evidence)
        status = "PASS" if result.passed else f"FAIL [{result.check_name}]: {result.reason[:60]}"
        mark = "OK" if result.passed == expected_pass else "WRONG"
        print(f"  [{mark}] {status}")
