"""Isolated human-golden intent evaluation for the V2.1 classifier."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils import CLASSIFIER_INTENTS  # noqa: E402
from v2_1_classifier_adapter import load_v2_1_classifier, predict_messages  # noqa: E402

GOLDEN_CSV = ROOT / "evaluation" / "golden_set.csv"
HUMAN_LABELS_CSV = ROOT / "evaluation" / "golden_human_labels.csv"
REPORT_OUT = ROOT / "reports" / "v2_1_golden_evaluation.md"
RESULTS_OUT = ROOT / "reports" / "v2_1_golden_results.csv"
EXPECTED_ROWS = 203


def load_aligned_golden() -> pd.DataFrame:
    golden = pd.read_csv(GOLDEN_CSV, dtype=str).fillna("")
    reviewed = pd.read_csv(HUMAN_LABELS_CSV, dtype=str).fillna("")
    if len(golden) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} golden rows, found {len(golden)}")
    if len(reviewed) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} reviewed rows, found {len(reviewed)}")
    if golden["example_id"].duplicated().any():
        raise ValueError("Golden set contains duplicate example_id values")
    if reviewed["example_id"].duplicated().any():
        raise ValueError("Human labels contain duplicate example_id values")
    if set(golden["example_id"]) != set(reviewed["example_id"]):
        raise ValueError("Golden and human-label example_id sets do not match")
    required = {"example_id", "human_intent", "should_escalate"}
    missing = required - set(reviewed.columns)
    if missing:
        raise ValueError(f"Human labels missing required columns: {sorted(missing)}")
    if reviewed["human_intent"].str.strip().eq("").any():
        raise ValueError("Human labels contain blank human_intent values")
    if reviewed["should_escalate"].str.strip().eq("").any():
        raise ValueError("Human labels contain blank should_escalate values")
    labels = reviewed[["example_id", "human_intent", "should_escalate"]]
    aligned = golden.drop(
        columns=["human_intent", "should_escalate"], errors="ignore"
    ).merge(labels, on="example_id", how="left", validate="one_to_one")
    if len(aligned) != EXPECTED_ROWS:
        raise ValueError("example_id merge changed the golden-set row count")
    return aligned


def evaluate_golden(golden: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    model = load_v2_1_classifier()
    predictions = predict_messages(golden["customer_message"].tolist(), model=model)
    if len(predictions) != len(golden):
        raise ValueError("V2.1 prediction count does not match golden-set count")
    result = golden[["example_id", "customer_message", "human_intent", "should_escalate"]].copy()
    result["pred_intent"] = [item["intent"] for item in predictions]
    result["predicted_decision_score"] = [
        item["predicted_decision_score"] for item in predictions
    ]
    labels = sorted(set(result["human_intent"]) | set(CLASSIFIER_INTENTS))
    metrics = {
        "n_total": len(result),
        "accuracy": accuracy_score(result["human_intent"], result["pred_intent"]),
        "macro_f1": f1_score(
            result["human_intent"], result["pred_intent"], labels=labels,
            average="macro", zero_division=0,
        ),
        "weighted_f1": f1_score(
            result["human_intent"], result["pred_intent"], labels=labels,
            average="weighted", zero_division=0,
        ),
        "labels": labels,
        "classification_report": classification_report(
            result["human_intent"], result["pred_intent"], labels=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            result["human_intent"], result["pred_intent"], labels=labels
        ),
        "prediction_distribution": result["pred_intent"].value_counts().to_dict(),
    }
    return result, metrics


def write_report(metrics: dict) -> None:
    labels = metrics["labels"]
    matrix = metrics["confusion_matrix"]
    lines = [
        "# V2.1 human-golden intent evaluation",
        "",
        "Isolated comparison of the already-trained V2.1 classifier against the final human annotations.",
        "",
        "## Dataset and safeguards",
        "",
        f"- Golden examples evaluated: **{metrics['n_total']}**",
        "- Human labels aligned by `example_id` using a one-to-one merge.",
        "- Blank `human_intent` values: **0**",
        "- Blank `should_escalate` values: **0**",
        "- Duplicate golden or human-label example IDs: **0**",
        "- V2.1 model loaded through `src/v2_1_classifier_adapter.py`.",
        "",
        "## Intent metrics",
        "",
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Macro F1: **{metrics['macro_f1']:.4f}**",
        f"- Weighted F1: **{metrics['weighted_f1']:.4f}**",
        "",
        "The four human `other` examples are included in the metric label set. V2.1 has no `other` output, so those cases cannot be predicted as `other`.",
        "",
        "```text",
        metrics["classification_report"],
        "```",
        "",
        "## Confusion matrix",
        "",
        "Rows are human intent; columns are V2.1 prediction.",
        "",
        "| Human \\ Predicted | " + " | ".join(labels) + " |",
        "|" + "---|" * (len(labels) + 1),
    ]
    lines.extend(
        "| " + labels[row_index] + " | " + " | ".join(
            str(value) for value in matrix[row_index]
        ) + " |"
        for row_index in range(len(labels))
    )
    lines += [
        "",
        "## Prediction distribution",
        "",
        "| Predicted intent | Count | Percentage |",
        "|---|---:|---:|",
    ]
    for intent in labels:
        count = metrics["prediction_distribution"].get(intent, 0)
        lines.append(f"| {intent} | {count} | {count / metrics['n_total']:.2%} |")
    lines += [
        "",
        "## Confidence and routing scope",
        "",
        "The V2.1 artifact is a LinearSVC pipeline. It exposes raw decision scores, not calibrated probability confidence. This report does not call those scores probabilities and does not report probability-based confidence.",
        "",
        "Routing was **not evaluated**. The existing routing pipeline expects calibrated confidence semantics and response/evidence fields; this isolated experiment evaluates intent classification only and does not invent a confidence conversion.",
        "",
        "## Output scope",
        "",
        "This experiment does not modify the golden files, production classifier, production model artifacts, or existing evaluation outputs.",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    aligned = load_aligned_golden()
    result, metrics = evaluate_golden(aligned)
    result.to_csv(RESULTS_OUT, index=False, encoding="utf-8")
    write_report(metrics)
    print(f"Golden examples evaluated: {metrics['n_total']}")
    print(f"Intent accuracy: {metrics['accuracy']:.4f}")
    print(f"Intent macro F1: {metrics['macro_f1']:.4f}")
    print(f"Intent weighted F1: {metrics['weighted_f1']:.4f}")
    print("Routing evaluated: no (calibrated confidence unavailable)")
    print(f"Wrote {REPORT_OUT}")
    print(f"Wrote {RESULTS_OUT}")


if __name__ == "__main__":
    main()
