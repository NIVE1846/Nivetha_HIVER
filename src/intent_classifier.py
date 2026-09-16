"""
intent_classifier.py

Main intent classifier: TF-IDF (word + char n-grams) + LinearSVC.

Why LinearSVC over plain Logistic Regression:
- LinearSVC optimises a margin-based hinge loss, which generalises better
  on short noisy text than LR's log-loss.
- Word + character n-grams together capture both semantic phrases ("not working")
  and morphological patterns ("crash", "crashing", "crashed").
- A second LR model is trained on the SVC's decision scores to produce
  calibrated probability estimates for the confidence gate.

Architecture:
  message -> TF-IDF (word 1-2gram + char 3-5gram) -> LinearSVC -> intent
                                                   -> LR on scores -> confidence

Outputs:
  data/processed/intent_classifier.joblib   (SVC pipeline)
  data/processed/intent_proba_lr.joblib     (probability calibrator)
  reports/classifier_results.md
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from utils import load_splits, CLASSIFIER_INTENTS as INTENTS

ROOT       = Path(__file__).resolve().parent.parent
SVC_OUT    = ROOT / "data" / "processed" / "intent_classifier.joblib"
PROBA_OUT  = ROOT / "data" / "processed" / "intent_proba_lr.joblib"
REPORTS    = ROOT / "reports"


def build_svc_pipeline() -> Pipeline:
    """
    Feature union of word n-grams and character n-grams fed into LinearSVC.
    Character n-grams help with misspellings common in tweets.
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_features=40_000,
        strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        sublinear_tf=True, min_df=3, max_features=30_000,
        strip_accents="unicode",
    )
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
    return Pipeline([
        ("features", features),
        ("clf", LinearSVC(C=0.5, class_weight="balanced",
                          max_iter=2000, random_state=42)),
    ])


def build_proba_estimator(svc_pipeline: Pipeline,
                          X_train: pd.Series,
                          y_train: pd.Series) -> LogisticRegression:
    """
    Train a Logistic Regression on the SVC's raw decision scores.
    This gives calibrated probabilities without blocked DLL dependencies.
    """
    scores = svc_pipeline.decision_function(X_train)   # shape (n, n_classes)
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr.fit(scores, y_train)
    return lr


def predict_with_confidence(svc_pipeline: Pipeline,
                            proba_lr: LogisticRegression,
                            messages: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predicted_intents, confidence_scores)."""
    scores  = svc_pipeline.decision_function(messages)
    probas  = proba_lr.predict_proba(scores)
    intents = proba_lr.classes_[np.argmax(probas, axis=1)]
    confs   = np.max(probas, axis=1)
    return intents, confs


def evaluate(svc_pipeline, proba_lr, X, y, split_name):
    intents, confs = predict_with_confidence(svc_pipeline, proba_lr, X)
    acc      = accuracy_score(y, intents)
    macro_f1 = f1_score(y, intents, average="macro", zero_division=0)
    wtd_f1   = f1_score(y, intents, average="weighted", zero_division=0)
    report   = classification_report(y, intents, labels=INTENTS, zero_division=0)
    cm       = confusion_matrix(y, intents, labels=INTENTS)
    avg_conf = float(np.mean(confs))

    print(f"\n--- Main Classifier ({split_name}, n={len(y)}) ---")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro F1:       {macro_f1:.4f}")
    print(f"  Weighted F1:    {wtd_f1:.4f}")
    print(f"  Avg confidence: {avg_conf:.4f}")
    print(f"\n{report}")
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": wtd_f1,
            "avg_conf": avg_conf, "report": report, "cm": cm, "n": len(y)}


def main():
    train, val, test = load_splits()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    print("\nFitting SVC pipeline ...")
    svc_pipeline = build_svc_pipeline()
    svc_pipeline.fit(train["customer_message"], train["intent"])

    print("Fitting probability calibrator ...")
    proba_lr = build_proba_estimator(
        svc_pipeline, train["customer_message"], train["intent"])

    val_r  = evaluate(svc_pipeline, proba_lr,
                      val["customer_message"],  val["intent"],  "val")
    test_r = evaluate(svc_pipeline, proba_lr,
                      test["customer_message"], test["intent"], "test")

    joblib.dump(svc_pipeline, SVC_OUT)
    joblib.dump(proba_lr,     PROBA_OUT)
    print(f"\nSVC pipeline saved  -> {SVC_OUT}")
    print(f"Proba LR saved      -> {PROBA_OUT}")

    # Confusion matrix text
    cm_lines = ["Confusion matrix (rows=true, cols=pred):"]
    cm_lines.append("  " + "  ".join(f"{i[:8]:>8}" for i in INTENTS))
    for label, row in zip(INTENTS, test_r["cm"]):
        cm_lines.append(f"{label[:8]:>8}  " + "  ".join(f"{v:>8}" for v in row))
    cm_text = "\n".join(cm_lines)
    print(f"\n{cm_text}")

    REPORTS.mkdir(exist_ok=True)
    report_md = (
        "# Main Classifier Results\n\n"
        "## Model\n"
        "TF-IDF (word 1-2gram + char 3-5gram, FeatureUnion) + LinearSVC (C=0.5, balanced)\n"
        "Confidence via LR trained on SVC decision scores.\n\n"
        f"## Validation (n={val_r['n']})\n"
        f"- Accuracy:    {val_r['accuracy']:.4f}\n"
        f"- Macro F1:    {val_r['macro_f1']:.4f}\n"
        f"- Weighted F1: {val_r['weighted_f1']:.4f}\n\n"
        f"## Test (n={test_r['n']})\n"
        f"- Accuracy:    {test_r['accuracy']:.4f}\n"
        f"- Macro F1:    {test_r['macro_f1']:.4f}\n"
        f"- Weighted F1: {test_r['weighted_f1']:.4f}\n"
        f"- Avg confidence: {test_r['avg_conf']:.4f}\n\n"
        f"```\n{test_r['report']}\n```\n\n"
        f"```\n{cm_text}\n```\n"
    )
    (REPORTS / "classifier_results.md").write_text(report_md, encoding="utf-8")
    print(f"\nReport saved -> {REPORTS / 'classifier_results.md'}")


if __name__ == "__main__":
    main()
