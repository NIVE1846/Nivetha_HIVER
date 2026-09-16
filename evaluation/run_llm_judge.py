"""Run the optional real LLM judge over the existing 50-response sample."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from historical_retrieval import HistoricalRetriever
from llm_judge import LLMJudgeUnavailable, LocalLLMJudge

INPUT_CSV = ROOT / "evaluation" / "human_judge_sample.csv"
SOURCE_EVAL_CSV = ROOT / "evaluation" / "eval_results.csv"
OUTPUT_CSV = ROOT / "evaluation" / "llm_judge_sample.csv"
PARTIAL_CSV = ROOT / "evaluation" / "llm_judge_sample.csv.partial"


def _evidence_text(hits: list[dict]) -> str:
    return "\n\n".join(
        f"Example {i}: Customer: {hit['customer_message'][:240]}\n"
        f"Support: {hit['brand_response'][:240]}\n"
        f"Similarity: {hit['similarity']:.4f}"
        for i, hit in enumerate(hits, 1)
    )


def main(limit: int | None = None) -> int:
    sample = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    if len(sample) != 50:
        raise ValueError(f"Expected the existing 50-row sample, found {len(sample)}")
    source = pd.read_csv(SOURCE_EVAL_CSV, dtype=str).fillna("")
    source_ids = set(source["example_id"])
    if not sample["example_id"].is_unique:
        raise ValueError("The 50-response sample contains duplicate example IDs")
    missing = set(sample["example_id"]) - source_ids
    if missing:
        raise ValueError(f"Sample IDs missing from generated evaluation output: {sorted(missing)}")
    blank_responses = sample.loc[
        sample["response"].str.strip().eq(""), "example_id"
    ].tolist()
    if blank_responses:
        raise ValueError(
            "Sample contains blank generated responses for: "
            + ", ".join(blank_responses)
        )

    try:
        judge = LocalLLMJudge()
    except LLMJudgeUnavailable as exc:
        print(f"LLM judge unavailable: {exc}")
        print("No llm_judge_sample.csv was written; no scores were fabricated.")
        return 2

    retriever = HistoricalRetriever.from_csv()
    rows = []
    failures = []
    if PARTIAL_CSV.exists():
        PARTIAL_CSV.unlink()
    rows_to_score = sample if limit is None else sample.head(limit)
    for _, row in rows_to_score.iterrows():
        started = time.perf_counter()
        hits = retriever.retrieve(row["customer_message"], top_k=3, intent=row["pred_intent"])
        evidence = _evidence_text(hits)
        try:
            scores = judge.score(row["customer_message"], evidence, row["response"])
        except ValueError as exc:
            failures.append({"example_id": row["example_id"], "error": str(exc)})
            continue
        elapsed = time.perf_counter() - started
        print(f"Judged {row['example_id']} in {elapsed:.2f}s")
        result_row = {
            "example_id": row["example_id"],
            "customer_message": row["customer_message"],
            "historical_evidence": evidence,
            "generated_response": row["response"],
            "correctness_llm": scores["correctness"],
            "groundedness_llm": scores["groundedness"],
            "relevance_llm": scores["relevance"],
            "helpfulness_llm": scores["helpfulness"],
            "tone_llm": scores["tone"],
            "overall_llm": scores["overall_score"],
            "llm_reason": scores["brief_reason"],
        }
        rows.append(result_row)
        pd.DataFrame(rows).to_csv(PARTIAL_CSV, index=False)
    if rows:
        PARTIAL_CSV.replace(OUTPUT_CSV)
        print(f"Wrote {len(rows)} real LLM scores to {OUTPUT_CSV}")
    if failures:
        print(f"LLM judge failures: {len(failures)}")
        for failure in failures:
            print(f"  {failure['example_id']}: {failure['error']}")
    return 1 if failures else 0


if __name__ == "__main__":
    requested_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(requested_limit))
