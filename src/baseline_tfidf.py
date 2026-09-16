"""
baseline_tfidf.py

Baseline 2: TF-IDF vectoriser + Logistic Regression classifier.

Pipeline:
  customer message -> TF-IDF -> Logistic Regression -> intent + confidence

The model is saved to data/processed/tfidf_model.joblib for reuse
by the retrieval and routing modules.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from utils import load_splits, CLASSIFIER_INTENTS as INTENTS

ROOT      = Path(__file__).resolve().parent.parent
MODEL_OUT = ROOT / "data" / "processed" / "tfidf_model.joblib"
REPORTS   = ROOT / "reports"


def build_pipeline() -> Pipeline:
    """
    TF-IDF with character n-grams (1-3) + word unigrams/bigrams.
    Logistic Regression with L2 regularisation, balanced class weights.

    Why these choices:
    - sublinear_tf=True dampens very frequent terms.
    - analyzer='word', ngram_range=(1,2) captures short phrases like 'not working'.
    - class_weight='balanced' compensates for any class imbalance.
    - max_iter=1000 ensures convergence on this dataset size.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=50_000,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )),
    ])


def evaluate(pipeline: Pipeline, X: pd.Series, y: pd.Series,
             split_name: str) -> dict:
    y_pred = pipeline.predict(X)
    acc      = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    wtd_f1   = f1_score(y, y_pred, average="weighted", zero_division=0)
    report   = classification_report(y, y_pred, labels=INTENTS, zero_division=0)
    cm       = confusion_matrix(y, y_pred, labels=INTENTS)

    print(f"\n--- TF-IDF Baseline ({split_name}, n={len(y)}) ---")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {wtd_f1:.4f}")
    print(f"\n{report}")

    return {
        "split": split_name,
        "n": len(y),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": wtd_f1,
        "report": report,
        "confusion_matrix": cm,
    }


def main():
    train, val, test = load_splits()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    pipeline = build_pipeline()
    pipeline.fit(train["customer_message"], train["intent"])

    val_results  = evaluate(pipeline, val["customer_message"],  val["intent"],  "val")
    test_results = evaluate(pipeline, test["customer_message"], test["intent"], "test")

    # Save model
    joblib.dump(pipeline, MODEL_OUT)
    print(f"\nModel saved -> {MODEL_OUT}")

    # Confusion matrix as text
    cm_lines = ["Confusion matrix (rows=true, cols=pred):"]
    cm_lines.append("  " + "  ".join(f"{i[:8]:>8}" for i in INTENTS))
    for i, row in zip(INTENTS, test_results["confusion_matrix"]):
        cm_lines.append(f"{i[:8]:>8}  " + "  ".join(f"{v:>8}" for v in row))
    cm_text = "\n".join(cm_lines)
    print(f"\n{cm_text}")

    # Save report
    REPORTS.mkdir(exist_ok=True)
    report_text = (
        "# TF-IDF Baseline Results\n\n"
        "## Model\n"
        "TF-IDF (word unigrams+bigrams, sublinear_tf, max 50k features) "
        "+ Logistic Regression (C=1.0, balanced class weights)\n\n"
        f"## Validation set (n={val_results['n']})\n"
        f"- Accuracy:    {val_results['accuracy']:.4f}\n"
        f"- Macro F1:    {val_results['macro_f1']:.4f}\n"
        f"- Weighted F1: {val_results['weighted_f1']:.4f}\n\n"
        f"## Test set (n={test_results['n']})\n"
        f"- Accuracy:    {test_results['accuracy']:.4f}\n"
        f"- Macro F1:    {test_results['macro_f1']:.4f}\n"
        f"- Weighted F1: {test_results['weighted_f1']:.4f}\n\n"
        f"```\n{test_results['report']}\n```\n\n"
        f"```\n{cm_text}\n```\n"
    )
    (REPORTS / "baseline_tfidf.md").write_text(report_text, encoding="utf-8")
    print(f"Report saved -> {REPORTS / 'baseline_tfidf.md'}")


if __name__ == "__main__":
    main()
