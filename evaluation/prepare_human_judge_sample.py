"""Prepare a blank 50-response human-judge sample from existing evaluation output."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL_CSV = ROOT / "evaluation" / "eval_results.csv"
OUTPUT_CSV = ROOT / "evaluation" / "human_judge_sample.csv"
RANDOM_STATE = 42
SAMPLE_SIZE = 50


def main() -> None:
    evaluation = pd.read_csv(EVAL_CSV, dtype=str).fillna("")
    auto = evaluation[evaluation["route"] == "AUTO_HANDLE"].copy()
    sample = auto.sample(n=min(SAMPLE_SIZE, len(auto)), random_state=RANDOM_STATE)
    columns = ["example_id", "customer_message", "response", "pred_intent", "evidence_score"]
    output = sample[columns].copy()
    output["evidence_used"] = ""
    for criterion in ["correctness", "groundedness", "relevance", "helpfulness", "tone"]:
        output[f"human_{criterion}"] = ""
    output["human_notes"] = ""
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(output)} blank human-judge rows to {OUTPUT_CSV}")
    print("Human scores are intentionally blank. Add the retrieved evidence used for each response before scoring.")


if __name__ == "__main__":
    main()
