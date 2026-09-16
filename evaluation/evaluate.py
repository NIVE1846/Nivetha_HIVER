"""
evaluate.py

Full evaluation harness. Runs the pipeline over the golden set and measures:
  - Intent classification: accuracy, macro F1, per-intent F1
  - Routing: auto-handle precision, escalation recall, unsafe auto-handle rate
  - Response quality: validator pass rate, avg evidence score

Outputs: reports/results.md
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

# Add src to path when run from project root
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from pipeline import SupportPipeline
from utils import CLASSIFIER_INTENTS as INTENTS

ROOT       = Path(__file__).resolve().parent.parent
GOLDEN_CSV = ROOT / "evaluation" / "golden_set.csv"
HUMAN_LABELS_CSV = ROOT / "evaluation" / "golden_human_labels.csv"
REPORTS    = ROOT / "reports"


def run_evaluation(golden: pd.DataFrame) -> dict:
    pipeline = SupportPipeline()
    results = []

    print(f"Running pipeline over {len(golden)} golden examples ...")
    for i, row in golden.iterrows():
        if i % 50 == 0:
            print(f"  {i}/{len(golden)} ...")
        result = pipeline.run(row["customer_message"])
        results.append({
            "example_id":        row["example_id"],
            "true_intent":       row["human_intent"],
            "pred_intent":       result.intent,
            "intent_confidence": result.intent_confidence,
            "route":             result.route,
            "evidence_score":    result.evidence_score,
            "response":          result.response,
            "validation_passed": result.validation_passed,
            "escalation_reason": result.escalation_reason,
            "should_escalate":   row["should_escalate"],
        })

    return results


def compute_metrics(results: list[dict]) -> dict:
    df = pd.DataFrame(results)

    # --- Intent classification metrics ---
    y_true = df["true_intent"]
    y_pred = df["pred_intent"]
    intent_acc      = accuracy_score(y_true, y_pred)
    intent_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    intent_wtd_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    intent_report   = classification_report(y_true, y_pred, labels=INTENTS,
                                            zero_division=0)
    intent_cm       = confusion_matrix(y_true, y_pred, labels=INTENTS)

    # --- Routing metrics ---
    auto_mask = df["route"] == "AUTO_HANDLE"
    esc_mask  = df["route"] == "ESCALATE"

    n_auto = auto_mask.sum()
    n_esc  = esc_mask.sum()
    n_total = len(df)

    # Unsafe auto-handle: auto-handled but intent prediction was wrong
    correct_intent = df["true_intent"] == df["pred_intent"]
    unsafe_auto    = auto_mask & ~correct_intent
    n_unsafe       = unsafe_auto.sum()

    auto_handle_rate      = n_auto / n_total
    unsafe_auto_rate      = n_unsafe / n_auto if n_auto > 0 else 0.0
    auto_handle_precision = 1.0 - unsafe_auto_rate

    # Escalation recall for high-risk intents
    high_risk = df["true_intent"].isin({"premium_billing", "account_login"})
    n_high_risk = high_risk.sum()
    correctly_escalated = (high_risk & esc_mask).sum()
    escalation_recall_hr = correctly_escalated / n_high_risk if n_high_risk > 0 else 0.0

    # Validator pass rate (among auto-handled)
    if n_auto > 0:
        val_pass_rate = df[auto_mask]["validation_passed"].mean()
    else:
        val_pass_rate = 0.0

    avg_evidence_score = df["evidence_score"].mean()

    return {
        "n_total": n_total,
        "n_auto": int(n_auto),
        "n_escalated": int(n_esc),
        "intent_accuracy": round(intent_acc, 4),
        "intent_macro_f1": round(intent_macro_f1, 4),
        "intent_weighted_f1": round(intent_wtd_f1, 4),
        "intent_report": intent_report,
        "intent_cm": intent_cm,
        "auto_handle_rate": round(auto_handle_rate, 4),
        "auto_handle_precision": round(auto_handle_precision, 4),
        "unsafe_auto_rate": round(unsafe_auto_rate, 4),
        "escalation_recall_high_risk": round(escalation_recall_hr, 4),
        "validator_pass_rate": round(val_pass_rate, 4),
        "avg_evidence_score": round(avg_evidence_score, 4),
    }


def write_report(metrics: dict, results_df: pd.DataFrame) -> None:
    cm_lines = ["Confusion matrix (rows=true, cols=pred):"]
    cm_lines.append("  " + "  ".join(f"{i[:8]:>8}" for i in INTENTS))
    for label, row in zip(INTENTS, metrics["intent_cm"]):
        cm_lines.append(f"{label[:8]:>8}  " + "  ".join(f"{v:>8}" for v in row))
    cm_text = "\n".join(cm_lines)

    report = f"""# Evaluation Results (Golden Set)

## Dataset
- Golden set size: {metrics['n_total']}
- Auto-handled: {metrics['n_auto']} ({metrics['auto_handle_rate']:.1%})
- Escalated: {metrics['n_escalated']} ({1 - metrics['auto_handle_rate']:.1%})

## Intent Classification
- Accuracy:    {metrics['intent_accuracy']:.4f}
- Macro F1:    {metrics['intent_macro_f1']:.4f}
- Weighted F1: {metrics['intent_weighted_f1']:.4f}

```
{metrics['intent_report']}
```

```
{cm_text}
```

## Routing Metrics
- Auto-handle rate:              {metrics['auto_handle_rate']:.4f}
- Auto-handle precision:         {metrics['auto_handle_precision']:.4f}
- Unsafe auto-handle rate:       {metrics['unsafe_auto_rate']:.4f}
- Escalation recall (high-risk): {metrics['escalation_recall_high_risk']:.4f}

## Response Quality
- Validator pass rate (auto-handled): {metrics['validator_pass_rate']:.4f}
- Avg evidence score:                 {metrics['avg_evidence_score']:.4f}

## Headline Numbers
| Metric | Value |
|--------|-------|
| Intent Macro F1 | {metrics['intent_macro_f1']:.4f} |
| Auto-handle Precision | {metrics['auto_handle_precision']:.4f} |
| Unsafe Auto-handle Rate | {metrics['unsafe_auto_rate']:.4f} |
| Escalation Recall (high-risk) | {metrics['escalation_recall_high_risk']:.4f} |

## What is misleading about the headline number?
See reports/results.md (full results section) for honest caveats.
"""
    (REPORTS / "results.md").write_text(report, encoding="utf-8")
    results_df.to_csv(ROOT / "evaluation" / "eval_results.csv", index=False)
    print(f"Report saved -> {REPORTS / 'results.md'}")
    print(f"Raw results  -> {ROOT / 'evaluation' / 'eval_results.csv'}")


def main():
    golden = pd.read_csv(GOLDEN_CSV, dtype=str)
    if HUMAN_LABELS_CSV.exists():
        reviewed = pd.read_csv(HUMAN_LABELS_CSV, dtype=str)
        required_review = {"example_id", "human_intent", "should_escalate", "labeling_notes"}
        missing_review = required_review - set(reviewed.columns)
        if missing_review:
            raise ValueError(f"Human review file is missing columns: {sorted(missing_review)}")
        golden = golden.drop(columns=["human_intent", "should_escalate", "labeling_notes"])
        golden = golden.merge(
            reviewed[["example_id", "human_intent", "should_escalate", "labeling_notes"]],
            on="example_id", how="left",
        )
        golden[["human_intent", "should_escalate", "labeling_notes"]] = golden[
            ["human_intent", "should_escalate", "labeling_notes"]
        ].fillna("")
    required = {"example_id", "customer_message", "human_intent", "should_escalate"}
    missing = required - set(golden.columns)
    if missing:
        raise ValueError(f"Golden set is missing required human-review columns: {sorted(missing)}")
    blank_intents = golden["human_intent"].fillna("").str.strip().eq("").sum()
    blank_escalations = golden["should_escalate"].fillna("").str.strip().eq("").sum()
    if blank_intents or blank_escalations:
        raise ValueError(
            "Human validation incomplete: please complete the golden-set labels before running final evaluation. "
            f"{blank_intents} human_intent and {blank_escalations} should_escalate values are blank. "
            "Complete evaluation/golden_human_labels.csv first, then merge the reviewed fields into golden_set.csv."
        )
    print(f"Golden set: {len(golden)} examples")

    raw_results = run_evaluation(golden)
    results_df  = pd.DataFrame(raw_results)
    metrics     = compute_metrics(raw_results)

    print(f"\n--- Evaluation Results ---")
    print(f"Intent accuracy:          {metrics['intent_accuracy']:.4f}")
    print(f"Intent macro F1:          {metrics['intent_macro_f1']:.4f}")
    print(f"Auto-handle rate:         {metrics['auto_handle_rate']:.4f}")
    print(f"Auto-handle precision:    {metrics['auto_handle_precision']:.4f}")
    print(f"Unsafe auto-handle rate:  {metrics['unsafe_auto_rate']:.4f}")
    print(f"Escalation recall (HR):   {metrics['escalation_recall_high_risk']:.4f}")
    print(f"Validator pass rate:      {metrics['validator_pass_rate']:.4f}")
    print(f"\n{metrics['intent_report']}")

    write_report(metrics, results_df)


if __name__ == "__main__":
    main()
