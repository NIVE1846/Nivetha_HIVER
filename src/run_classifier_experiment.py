"""Run isolated Experiment 1 classifier variants without touching baseline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from intent_classifier import build_proba_estimator, build_svc_pipeline, predict_with_confidence
from utils import CLASSIFIER_INTENTS, is_fallback_general_inquiry, load_splits

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "data" / "processed" / "experiments"
REPORT_ROOT = ROOT / "reports" / "experiments"
RANDOM_STATE = 42
GENERAL_INQUIRY_CAP = 1176


def prepare_refined_training(train: pd.DataFrame) -> pd.DataFrame:
    """Remove fallback labels and deterministically cap explicit general inquiries."""
    explicit = train.loc[
        ~train["customer_message"].map(is_fallback_general_inquiry)
    ].copy()
    general = explicit[explicit["intent"] == "general_inquiry"]
    other = explicit[explicit["intent"] != "general_inquiry"]
    capped_general = general.sample(
        n=GENERAL_INQUIRY_CAP,
        random_state=RANDOM_STATE,
    )
    return pd.concat([other, capped_general], ignore_index=False).sort_index()


def counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        intent: int((df["intent"] == intent).sum())
        for intent in CLASSIFIER_INTENTS
    }


def evaluate_variant(
    model,
    calibrator,
    df: pd.DataFrame,
) -> dict:
    predictions, confidences = predict_with_confidence(
        model,
        calibrator,
        df["customer_message"],
    )
    labels = CLASSIFIER_INTENTS
    report = classification_report(
        df["intent"],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "n": len(df),
        "accuracy": float(accuracy_score(df["intent"], predictions)),
        "macro_f1": float(f1_score(df["intent"], predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(df["intent"], predictions, labels=labels, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(df["intent"], predictions, labels=labels).tolist(),
        "prediction_distribution": pd.Series(predictions).value_counts().reindex(labels, fill_value=0).astype(int).to_dict(),
        "confidence": {
            "mean": float(confidences.mean()),
            "min": float(confidences.min()),
            "max": float(confidences.max()),
        },
    }


def format_report(experiment_id: str, train: pd.DataFrame, val_result: dict, test_result: dict, sampling: str) -> str:
    labels = CLASSIFIER_INTENTS
    lines = [
        f"# {experiment_id}",
        "",
        f"- Experiment ID: `{experiment_id}`",
        f"- Random seed: `{RANDOM_STATE}`",
        f"- Sampling configuration: {sampling}",
        "",
        "## Dataset counts",
        "",
        f"- Training: {len(train)}",
        f"- Validation: {val_result['n']}",
        f"- Test: {test_result['n']}",
        "",
    ]
    for name, result in [("Validation", val_result), ("Test", test_result)]:
        lines.extend([
            f"## {name} metrics",
            "",
            f"- Accuracy: {result['accuracy']:.4f}",
            f"- Macro F1: {result['macro_f1']:.4f}",
            f"- Weighted F1: {result['weighted_f1']:.4f}",
            f"- Confidence mean/min/max: {result['confidence']['mean']:.4f} / {result['confidence']['min']:.4f} / {result['confidence']['max']:.4f}",
            "",
            "| Intent | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ])
        for label in labels:
            row = result["classification_report"][label]
            lines.append(
                f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1-score']:.4f} | {int(row['support'])} |"
            )
        lines.extend(["", "Prediction distribution:", "", "```text", json.dumps(result["prediction_distribution"], indent=2), "```", ""])
        lines.append("Confusion matrix (rows=true, columns=predicted):")
        lines.extend(["", "```text", "labels = " + ", ".join(labels)])
        lines.extend(json.dumps(result["confusion_matrix"], indent=2).splitlines())
        lines.extend(["```", ""])
    return "\n".join(lines)


def run_experiment(experiment_id: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, sampling: str) -> dict:
    output_dir = EXPERIMENT_ROOT / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_svc_pipeline()
    model.fit(train["customer_message"], train["intent"])
    calibrator = build_proba_estimator(model, train["customer_message"], train["intent"])
    val_result = evaluate_variant(model, calibrator, val)
    test_result = evaluate_variant(model, calibrator, test)
    import joblib
    joblib.dump(model, output_dir / "intent_classifier.joblib")
    joblib.dump(calibrator, output_dir / "intent_proba_lr.joblib")
    manifest = {
        "experiment_id": experiment_id,
        "random_state": RANDOM_STATE,
        "sampling": sampling,
        "training_counts": counts(train),
        "validation_counts": counts(val),
        "test_counts": counts(test),
        "validation": val_result,
        "test": test_result,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (REPORT_ROOT / f"{experiment_id}.md").parent.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / f"{experiment_id}.md").write_text(
        format_report(experiment_id, train, val_result, test_result, sampling),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    train, val, test = load_splits()
    refined = prepare_refined_training(train)
    expected = {
        "general_inquiry": 1176,
        "premium_billing": 392,
        "download_offline": 253,
        "playback_issue": 226,
        "app_bug": 206,
        "account_login": 181,
        "content_search": 37,
    }
    if counts(refined) != expected or len(refined) != 2471:
        raise RuntimeError(f"Unexpected refined counts: {counts(refined)}")
    baseline = run_experiment(
        "EXP1_A_BASELINE",
        train,
        val,
        test,
        "Existing rule labels and full training split; no sampling.",
    )
    refined_result = run_experiment(
        "EXP1_B_REFINED_CAP3X",
        refined,
        val,
        test,
        "Exclude no-rule-match fallback rows from training; retain all explicit labels; cap explicit general_inquiry at 1,176; random_state=42.",
    )
    comparison = {
        "experiment_a": {
            "validation": baseline["validation"],
            "test": baseline["test"],
        },
        "experiment_b": {
            "validation": refined_result["validation"],
            "test": refined_result["test"],
        },
    }
    (REPORT_ROOT / "EXP1_comparison.md").write_text(
        "# Experiment 1 comparison\n\n"
        "This comparison uses only the original train/validation/test splits. "
        "The golden evaluation was not run.\n\n"
        f"```json\n{json.dumps(comparison, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print("Experiment counts verified:")
    print("EXP1_A_BASELINE", counts(train), len(train))
    print("EXP1_B_REFINED_CAP3X", counts(refined), len(refined))


if __name__ == "__main__":
    main()
