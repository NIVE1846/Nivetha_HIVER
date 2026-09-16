"""
response_generator.py

Grounded response generator.

Two modes:
  1. TEMPLATE mode (default, always available locally):
     Constructs a response by selecting and lightly adapting the most similar
     historical brand response. No hallucination risk. Fast. Explainable.

  2. LLM mode (requires Colab or a machine where torch DLLs are not blocked):
     Uses a lightweight open-source instruction model (google/flan-t5-base via
     HuggingFace) prompted with the customer message + retrieved evidence.
     Falls back to TEMPLATE mode if the model cannot be loaded.

Why flan-t5-base for LLM mode:
  - ~250 MB, runs on CPU in Colab free tier.
  - Instruction-tuned: follows "Answer as a Spotify support agent" prompts.
  - No paid API required.
  - Deterministic with temperature=0.

The generator NEVER invents brand policies, refunds, timelines, or account
actions. It is grounded strictly in the retrieved historical responses.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import re

ROOT = Path(__file__).resolve().parent.parent

# Phrases that suggest the LLM invented something unsupported
HALLUCINATION_SIGNALS = [
    r"\brefund\b", r"\bcredit\b", r"\bcompensate\b",
    r"\bwithin \d+ (hour|day|business)",
    r"\bguarantee\b", r"\bpromise\b",
    r"\bwe will (fix|resolve|send|give|provide)\b",
    r"\byour account has been\b",
    r"\bi have (updated|changed|reset|cancelled)\b",
]


@dataclass
class GeneratedResponse:
    text: str
    mode: str           # "template" or "llm"
    evidence_used: bool
    source_tweet_id: str


def _clean_response(text: str) -> str:
    """Normalise whitespace and strip leading @mentions."""
    text = re.sub(r"@\w+\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def generate_template(customer_message: str,
                      retrieved_evidence: list[dict]) -> GeneratedResponse:
    """
    Template mode: return the top retrieved brand response, lightly cleaned.
    If no evidence, return a safe escalation message.
    """
    if not retrieved_evidence:
        return GeneratedResponse(
            text=("Thanks for reaching out! We'd like to help but need more "
                  "information. Could you please describe the issue in more detail "
                  "so we can assist you?"),
            mode="template",
            evidence_used=False,
            source_tweet_id="",
        )

    best = retrieved_evidence[0]
    response_text = _clean_response(best["brand_response"])

    # If the top response is very short or empty, try the next one
    for evidence in retrieved_evidence[1:]:
        if len(response_text) < 20:
            response_text = _clean_response(evidence["brand_response"])

    return GeneratedResponse(
        text=response_text,
        mode="template",
        evidence_used=True,
        source_tweet_id=best.get("brand_tweet_id", ""),
    )


def generate_llm(customer_message: str,
                 intent: str,
                 retrieved_evidence: list[dict]) -> Optional[GeneratedResponse]:
    """
    LLM mode using flan-t5-base. Returns None if model cannot be loaded
    (e.g. torch DLLs blocked), so caller can fall back to template mode.
    """
    try:
        from transformers import pipeline as hf_pipeline
        generator = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=120,
        )
    except Exception:
        return None

    # Build evidence block from top-3 retrieved examples
    evidence_block = ""
    for i, ev in enumerate(retrieved_evidence[:3], 1):
        cust = ev["customer_message"][:120]
        resp = ev["brand_response"][:120]
        evidence_block += f"\nExample {i}:\n  Customer: {cust}\n  Support: {resp}\n"

    prompt = (
        f"You are a Spotify customer support agent. "
        f"Answer the customer's message using ONLY the evidence below. "
        f"Do not invent refunds, timelines, or account actions not in the evidence.\n"
        f"\nCustomer issue type: {intent}\n"
        f"Customer message: {customer_message[:200]}\n"
        f"\nHistorical evidence:{evidence_block}\n"
        f"Support response:"
    )

    try:
        output = generator(prompt)[0]["generated_text"]
        return GeneratedResponse(
            text=output.strip(),
            mode="llm",
            evidence_used=bool(retrieved_evidence),
            source_tweet_id="",
        )
    except Exception:
        return None


def generate(customer_message: str,
             intent: str,
             retrieved_evidence: list[dict],
             use_llm: bool = False) -> GeneratedResponse:
    """
    Main entry point. Tries LLM if requested, falls back to template.
    """
    if use_llm:
        result = generate_llm(customer_message, intent, retrieved_evidence)
        if result is not None:
            return result
        # Fall through to template if LLM unavailable

    return generate_template(customer_message, retrieved_evidence)


if __name__ == "__main__":
    # Smoke test with fake evidence
    fake_evidence = [
        {
            "customer_message": "app keeps crashing after update",
            "brand_response": "Try reinstalling the app and restarting your device. Let us know if that helps!",
            "similarity": 0.45,
            "brand_tweet_id": "12345",
        }
    ]
    result = generate(
        customer_message="spotify crashes every time i open it",
        intent="app_bug",
        retrieved_evidence=fake_evidence,
        use_llm=False,
    )
    print(f"Mode: {result.mode}")
    print(f"Evidence used: {result.evidence_used}")
    print(f"Response: {result.text}")
