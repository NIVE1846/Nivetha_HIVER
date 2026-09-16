"""Calculate inter-annotator agreement without fabricating missing labels."""

from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
DOUBLE_LABEL_CSV = ROOT / "evaluation" / "double_label_sample.csv"
HUMAN_RESPONSE_CSV = ROOT / "evaluation" / "human_response_eval.csv"
LLM_RESPONSE_CSV = ROOT / "evaluation" / "llm_judge_sample.csv"


def summarize(df: pd.DataFrame, left: str, right: str, title: str) -> None:
    available = df[[left, right]].copy()
    available[left] = available[left].fillna("").astype(str).str.strip()
    available[right] = available[right].fillna("").astype(str).str.strip()
    available = available[(available[left] != "") & (available[right] != "")]

    total = len(df)
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Completed pairs: {len(available)}/{total}")
    if available.empty:
        print("Agreement cannot be calculated yet: two completed annotations are required per example.")
        return

    matches = available[left] == available[right]
    disagreements = int((~matches).sum())
    raw_agreement = float(matches.mean())
    kappa = cohen_kappa_score(available[left], available[right])
    print(f"Raw agreement: {raw_agreement:.4f} ({matches.sum()}/{len(available)})")
    print(f"Disagreements: {disagreements}")
    print(f"Cohen's kappa: {kappa:.4f}")
    if disagreements:
        print("Disagreement summary:")
        disagreement_rows = available[~matches]
        summary = disagreement_rows.groupby([left, right]).size().reset_index(name="count")
        for _, row in summary.iterrows():
            print(f"  {row[left]} vs {row[right]}: {row['count']}")


def main() -> None:
    if not DOUBLE_LABEL_CSV.exists():
        print(f"Missing file: {DOUBLE_LABEL_CSV}")
        return
    df = pd.read_csv(DOUBLE_LABEL_CSV, dtype=str).fillna("")
    required = {
        "annotator_A_intent", "annotator_B_intent",
        "annotator_A_should_escalate", "annotator_B_should_escalate",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"Double-label file: {DOUBLE_LABEL_CSV}")
    print("Blank annotations are excluded; no labels are inferred.")
    summarize(df, "annotator_A_intent", "annotator_B_intent", "Intent agreement")
    summarize(
        df,
        "annotator_A_should_escalate",
        "annotator_B_should_escalate",
        "Escalation agreement",
    )

    if not LLM_RESPONSE_CSV.exists() or not HUMAN_RESPONSE_CSV.exists():
        print("\nResponse agreement: Human evaluation pending.")
        return

    llm = pd.read_csv(LLM_RESPONSE_CSV, dtype=str).fillna("")
    human = pd.read_csv(HUMAN_RESPONSE_CSV, dtype=str).fillna("")
    merged = llm.merge(human, on="example_id", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("LLM and human response samples are not aligned by example_id")
    for criterion in ["correctness", "groundedness", "relevance", "helpfulness", "tone"]:
        summarize(
            merged,
            f"{criterion}_llm",
            f"{criterion}_human",
            f"{criterion.capitalize()} agreement",
        )


if __name__ == "__main__":
    main()
