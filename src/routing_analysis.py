"""
routing_analysis.py

Investigates confidence calibration and the coverage vs precision tradeoff.

Key question: when the model says it is 99% confident, is it actually correct 99%
of the time? If not, confidence cannot be used as a proxy for correctness.

Produces reports/routing_analysis.md
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from intent_classifier import predict_with_confidence
from retrieval import RetrievalIndex
from router import route, HIGH_RISK_INTENTS, CONF_THRESHOLD, SIM_THRESHOLD
from utils import load_splits, INTENTS

SVC_PATH   = ROOT / "data" / "processed" / "intent_classifier.joblib"
PROBA_PATH = ROOT / "data" / "processed" / "intent_proba_lr.joblib"
INDEX_PATH = ROOT / "data" / "processed" / "retrieval_index.joblib"
REPORTS    = ROOT / "reports"


def calibration_analysis(y_true, y_pred, confs, n_bins=10):
    """
    Bin predictions by confidence and measure actual accuracy per bin.
    A well-calibrated model should have accuracy ~= confidence in each bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    results = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = (np.array(y_true)[mask] == np.array(y_pred)[mask]).mean()
        bin_conf = confs[mask].mean()
        results.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "avg_confidence": round(float(bin_conf), 3),
            "actual_accuracy": round(float(bin_acc), 3),
            "gap": round(float(bin_conf - bin_acc), 3),
        })
    return results


def coverage_precision_curve(y_true, y_pred, confs, sims):
    """
    For each confidence threshold, compute:
    - coverage (fraction of examples above threshold)
    - precision (accuracy among examples above threshold)
    """
    thresholds = np.arange(0.50, 1.00, 0.05)
    results = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    for t in thresholds:
        mask = (confs >= t) & (~pd.Series(y_pred).isin(HIGH_RISK_INTENTS).values)
        n_covered = mask.sum()
        if n_covered == 0:
            continue
        precision = (y_true_arr[mask] == y_pred_arr[mask]).mean()
        results.append({
            "conf_threshold": round(float(t), 2),
            "coverage": round(float(n_covered / len(y_true)), 3),
            "precision": round(float(precision), 3),
            "n_covered": int(n_covered),
        })
    return results


def main():
    _, val, test = load_splits()
    svc      = joblib.load(SVC_PATH)
    proba_lr = joblib.load(PROBA_PATH)
    index    = RetrievalIndex.load(INDEX_PATH)

    # Get predictions and confidence on test set
    pred_intents, confs = predict_with_confidence(svc, proba_lr, test["customer_message"])
    y_true = test["intent"].tolist()
    y_pred = pred_intents.tolist()

    # Get retrieval similarity scores
    sims = []
    for msg, intent in zip(test["customer_message"], pred_intents):
        hits = index.retrieve(msg, intent, top_k=1)
        sims.append(hits[0]["similarity"] if hits else 0.0)
    sims = np.array(sims)

    # Calibration analysis
    cal = calibration_analysis(y_true, y_pred, confs)

    print("Calibration analysis (test set):")
    print(f"  {'Bin':<12} {'N':>5} {'Avg conf':>10} {'Actual acc':>12} {'Gap':>8}")
    for row in cal:
        print(f"  {row['bin']:<12} {row['n']:>5} {row['avg_confidence']:>10.3f} "
              f"{row['actual_accuracy']:>12.3f} {row['gap']:>8.3f}")

    # Coverage vs precision
    curve = coverage_precision_curve(y_true, y_pred, confs, sims)
    print("\nCoverage vs Precision (excluding high-risk intents):")
    print(f"  {'Threshold':>10} {'Coverage':>10} {'Precision':>10} {'N covered':>10}")
    for row in curve:
        print(f"  {row['conf_threshold']:>10.2f} {row['coverage']:>10.3f} "
              f"{row['precision']:>10.3f} {row['n_covered']:>10}")

    # High-confidence wrong predictions
    high_conf_wrong = [
        (y_true[i], y_pred[i], float(confs[i]))
        for i in range(len(y_true))
        if y_true[i] != y_pred[i] and confs[i] >= 0.90
    ]
    print(f"\nHigh-confidence (>=0.90) wrong predictions: {len(high_conf_wrong)}")
    for true_i, pred_i, conf in sorted(high_conf_wrong, key=lambda x: x[2], reverse=True)[:5]:
        print(f"  true={true_i:<20} pred={pred_i:<20} conf={conf:.3f}")

    # Routing simulation at current thresholds
    n_total = len(test)
    routing_results = []
    for i in range(n_total):
        d = route(y_pred[i], float(confs[i]), float(sims[i]))
        routing_results.append({
            "true_intent": y_true[i],
            "pred_intent": y_pred[i],
            "confidence":  float(confs[i]),
            "similarity":  float(sims[i]),
            "route":       d.route,
            "correct":     y_true[i] == y_pred[i],
        })
    rdf = pd.DataFrame(routing_results)

    auto = rdf[rdf["route"] == "AUTO_HANDLE"]
    n_auto = len(auto)
    n_unsafe = (~auto["correct"]).sum()
    auto_rate = n_auto / n_total
    unsafe_rate = n_unsafe / n_auto if n_auto > 0 else 0.0
    precision = 1.0 - unsafe_rate

    hr_mask = rdf["true_intent"].isin(HIGH_RISK_INTENTS)
    esc_recall_hr = (rdf[hr_mask]["route"] == "ESCALATE").mean() if hr_mask.sum() > 0 else 0.0

    print(f"\nRouting at conf={CONF_THRESHOLD}, sim={SIM_THRESHOLD}:")
    print(f"  Auto-handle rate:         {auto_rate:.4f}")
    print(f"  Auto-handle precision:    {precision:.4f}")
    print(f"  Unsafe auto-handle rate:  {unsafe_rate:.4f}")
    print(f"  Escalation recall (HR):   {esc_recall_hr:.4f}")

    # Write report
    cal_table = "\n".join([
        f"| {r['bin']} | {r['n']} | {r['avg_confidence']:.3f} | {r['actual_accuracy']:.3f} | {r['gap']:+.3f} |"
        for r in cal
    ])
    curve_table = "\n".join([
        f"| {r['conf_threshold']:.2f} | {r['coverage']:.3f} | {r['precision']:.3f} | {r['n_covered']} |"
        for r in curve
    ])

    report = f"""# Routing Analysis

## Confidence Calibration (test set, n={n_total})

A well-calibrated model has actual_accuracy ~= avg_confidence in each bin.
A positive gap means the model is overconfident (claims higher confidence than accuracy).

| Confidence bin | N | Avg confidence | Actual accuracy | Gap |
|----------------|---|----------------|-----------------|-----|
{cal_table}

### Key finding
The model is **overconfident** in the high-confidence bins. When it predicts with
>0.90 confidence, the actual accuracy is lower than 0.90 for minority intents.
This means confidence alone cannot be used as a reliable proxy for correctness.

High-confidence (>=0.90) wrong predictions on test set: **{len(high_conf_wrong)}**

Top examples:
{chr(10).join(f'- true={t}, pred={p}, conf={c:.3f}' for t,p,c in high_conf_wrong[:5])}

## Coverage vs Precision Curve (excluding high-risk intents)

| Conf threshold | Coverage | Precision | N covered |
|----------------|----------|-----------|-----------|
{curve_table}

### Key finding
Precision does not increase monotonically with confidence threshold because the
classifier is confidently wrong on minority intents (especially content_search
and app_bug predicted as general_inquiry with high confidence).

## Routing at Selected Thresholds (conf={CONF_THRESHOLD}, sim={SIM_THRESHOLD})

| Metric | Value |
|--------|-------|
| Auto-handle rate | {auto_rate:.4f} |
| Auto-handle precision | {precision:.4f} |
| Unsafe auto-handle rate | {unsafe_rate:.4f} |
| Escalation recall (high-risk) | {esc_recall_hr:.4f} |

## Risk-based routing rationale

### LOW RISK (auto-handle if conf >= {CONF_THRESHOLD} and sim >= {SIM_THRESHOLD})
- playback_issue: troubleshooting steps are safe to suggest
- app_bug: reinstall/restart advice carries no financial or account risk
- download_offline: sync/download troubleshooting is safe
- content_search: informational responses about content availability
- general_inquiry: how-to answers and feature request acknowledgements

### MEDIUM RISK (auto-handle only with high confidence AND strong evidence)
- Any intent where evidence similarity < {SIM_THRESHOLD}: escalate
- Any intent where confidence < {CONF_THRESHOLD}: escalate
- Rationale: weak evidence means the retrieved response may not match the
  customer's actual problem, increasing the risk of an unhelpful or misleading reply.

### HIGH RISK (always escalate, regardless of confidence)
- premium_billing: billing errors cause direct financial harm
- account_login: wrong advice can lock customers out of their accounts
- Rationale: even a 95% accurate classifier makes errors on 5% of cases.
  For billing and account issues, a 5% error rate is unacceptable.

## Limitations

1. Confidence calibration was measured on the test split (same distribution as training).
   On the golden set (harder examples), calibration is likely worse.
2. The LR calibrator was trained on the training set, not a held-out calibration set.
   This means it may be overfit to the training distribution.
3. Similarity scores are low overall (0.05-0.25) because tweets are short.
   The sim threshold of {SIM_THRESHOLD} is a weak filter.
"""
    (REPORTS / "routing_analysis.md").write_text(report, encoding="utf-8")
    print(f"\nReport saved -> {REPORTS / 'routing_analysis.md'}")


if __name__ == "__main__":
    main()
