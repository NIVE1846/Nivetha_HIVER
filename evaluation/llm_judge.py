"""Optional local LLM-as-judge adapter.

This module intentionally does not fall back to the deterministic judge. If a
real local Transformers model is unavailable, callers receive an explicit
error rather than fabricated scores.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


CRITERIA = (
    "correctness",
    "groundedness",
    "relevance",
    "helpfulness",
    "tone",
)


class LLMJudgeUnavailable(RuntimeError):
    """Raised when the optional local LLM dependency/model cannot be used."""


def _validate_score(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("LLM judge output must be a JSON object")
    if "groundedness" not in payload and "groundness" in payload:
        payload["groundedness"] = payload.pop("groundness")
    required = set(CRITERIA) | {"overall_score", "brief_reason"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"LLM judge output missing fields: {sorted(missing)}")
    for key in CRITERIA:
        value = payload[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} must be an integer from 1 to 5")
        if isinstance(value, str):
            if not re.fullmatch(r"[1-5]", value.strip()):
                raise ValueError(f"{key} must be an integer from 1 to 5")
            value = int(value.strip())
        elif isinstance(value, int):
            pass
        else:
            raise ValueError(f"{key} must be an integer from 1 to 5")
        if not 1 <= value <= 5:
            raise ValueError(f"{key} must be an integer from 1 to 5")
        payload[key] = value

    if "overall_score" not in payload:
        raise ValueError("LLM judge output missing field: overall_score")
    mean_score = sum(payload[key] for key in CRITERIA) / len(CRITERIA)
    payload["overall_score"] = int(math.floor(mean_score + 0.5))
    if not isinstance(payload["brief_reason"], str):
        raise ValueError("brief_reason must be a string")
    return payload


def _parse_score_output(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM judge did not return valid JSON. Raw output: {raw!r}") from exc
    try:
        return _validate_score(payload)
    except ValueError as exc:
        raise ValueError(f"{exc}. Raw output: {raw!r}") from exc


class LocalLLMJudge:
    """Run a local causal instruction model as a structured response judge."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        max_new_tokens: int = 128,
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMJudgeUnavailable(
                "transformers is not installed; no real local LLM judge is available."
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(model_name)
            self._max_new_tokens = max_new_tokens
        except Exception as exc:
            raise LLMJudgeUnavailable(
                f"Could not load local model {model_name!r}: {exc}"
            ) from exc

    def score(
        self,
        customer_message: str,
        historical_evidence: str,
        generated_response: str,
    ) -> dict[str, Any]:
        prompt = f"""You are evaluating a SpotifyCares support response.
Use the full 1-5 rubric independently for each criterion:
1 = materially wrong/unsupported/irrelevant/unhelpful/inappropriate;
3 = partly correct, supported, relevant, helpful, or acceptable;
5 = accurate, fully evidence-grounded, directly relevant, actionable, and
professional/empathetic. Use 2 or 4 for intermediate cases.
Do not default to 1. Judge the actual customer, evidence, and response.
Output ONLY one JSON object with exactly these keys:
correctness, groundedness, relevance, helpfulness, tone, overall_score, brief_reason.
Each criterion must be an independent integer 1-5. overall_score must be present
and will be recomputed from the five criteria. Do not use a key named score.
Customer: {customer_message}
Evidence: {historical_evidence}
Response: {generated_response}
JSON:"""
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON-only response-quality evaluator.",
            },
            {"role": "user", "content": prompt},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(rendered, return_tensors="pt")
        import torch
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return _parse_score_output(raw)
