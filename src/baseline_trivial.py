"""
baseline_trivial.py

Trivial baseline: always predict the most frequent intent in the training set.
This establishes the minimum reference point all other models must beat.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import load_splits, CLASSIFIER_INTENTS as INTENTS

ROOT     = Path(__file__).resolve().parent.parent
REPORTS  = ROOT / "reports"


def run_trivial_baseline() -> dict:
    train, val, test = load_splits()

    most_frequent = train["intent"].value_counts().idxmax()
    print(f"Most frequent intent in training set: '{most_frequent}'")
    print(f"Training set size:    {len(train)}")
    print(f"Validation set size:  {len(val)}")
    print(f"Test set size:        {len(test)}")

    y_test = test["intent"]
    y_pred = [most_frequent] * len(y_test)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report   = classification_report(y_test, y_pred, labels=INTENTS,
                                     zero_division=0)

    print(f"\nTrivial Baseline Results (test set, n={len(y_test)})")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"\n{report}")

    # Save to reports
    REPORTS.mkdir(exist_ok=True)
    result_text = (
        "# Trivial Baseline Results\n\n"
        f"Strategy: always predict '{most_frequent}' (most frequent training intent)\n\n"
        f"- Accuracy:  {acc:.4f}\n"
        f"- Macro F1:  {macro_f1:.4f}\n\n"
        f"```\n{report}\n```\n"
    )
    (REPORTS / "baseline_trivial.md").write_text(result_text, encoding="utf-8")
    print(f"Report saved -> {REPORTS / 'baseline_trivial.md'}")

    return {"accuracy": acc, "macro_f1": macro_f1, "most_frequent": most_frequent}


if __name__ == "__main__":
    run_trivial_baseline()
