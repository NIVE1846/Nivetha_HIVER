"""Prepare the blank human form for the existing 50-response sample."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "evaluation" / "human_judge_sample.csv"
OUTPUT = ROOT / "evaluation" / "human_response_eval.csv"


def main() -> None:
    sample = pd.read_csv(SOURCE, dtype=str).fillna("")
    if len(sample) != 50:
        raise ValueError(f"Expected 50 existing sample rows, found {len(sample)}")
    output = pd.DataFrame({"example_id": sample["example_id"]})
    for criterion in ["correctness", "groundedness", "relevance", "helpfulness", "tone"]:
        output[f"{criterion}_human"] = ""
    output["overall_human"] = ""
    output["human_notes"] = ""
    output.to_csv(OUTPUT, index=False)
    print(f"Wrote blank human evaluation template to {OUTPUT}")


if __name__ == "__main__":
    main()
