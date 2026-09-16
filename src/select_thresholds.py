"""
select_thresholds.py

Sweeps confidence and similarity thresholds on the validation set to find
the best operating point: maximise auto-handle rate while keeping
unsafe auto-handle rate below 5%.

"Unsafe" = AUTO_HANDLE on an example whose true intent is a high-risk intent
           OR whose classifier prediction is wrong.

Outputs reports/router_threshold_search.md
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from intent_classifier import predict_with_confidence
from retrieval import RetrievalIndex
from router import HIGH_RISK_INTENTS
from utils import load_splits

ROOT       = Path(__file__).resolve().parent.parent
SVC_PATH   = ROOT / "data" / "processed" / "intent_classifier.joblib"
PROBA_PATH = ROOT / "data" / "processed" / "intent_proba_lr.joblib"
INDEX_PATH = ROOT / "data" / "processed" / "retrieval_index.joblib"
REPORTS    = ROOT / "reports"


def sweep(val_df: pd.DataFrame,
          svc, proba_lr, index: RetrievalIndex,
          conf_thresholds, sim_thresholds):

    # Pre-compute predictions and evidence scores for all val examples
    messages = val_df["customer_message"]
    true_intents = val_df["intent"].tolist()

    pred_intents, confs = predict_with_confidence(svc, proba_lr, messages)

    # Get best retrieval similarity per example
    sims = []
    for msg, intent in zip(messages, pred_intents):
        hits = index.retrieve(msg, intent, top_k=1)
        sims.append(hits[0]["similarity"] if hits else 0.0)
    sims = np.array(sims)

    results = []
    for ct in conf_thresholds:
        for st in sim_thresholds:
            auto_mask = (
                (confs >= ct) &
                (sims >= st) &
                (~pd.Series(pred_intents).isin(HIGH_RISK_INTENTS).values)
            )
            n_auto = auto_mask.sum()
            n_total = len(val_df)

            # Unsafe = auto-handled but prediction was wrong
            correct = np.array(pred_intents) == np.array(true_intents)
            unsafe = auto_mask & ~correct
            n_unsafe = unsafe.sum()

            auto_rate   = n_auto / n_total
            unsafe_rate = n_unsafe / n_auto if n_auto > 0 else 0.0
            precision   = 1.0 - unsafe_rate

            results.append({
                "conf_threshold": ct,
                "sim_threshold":  st,
                "auto_rate":      round(auto_rate, 4),
                "unsafe_rate":    round(unsafe_rate, 4),
                "precision":      round(precision, 4),
                "n_auto":         int(n_auto),
                "n_unsafe":       int(n_unsafe),
            })

    return pd.DataFrame(results)


def main():
    _, val, _ = load_splits()
    svc      = joblib.load(SVC_PATH)
    proba_lr = joblib.load(PROBA_PATH)
    index    = RetrievalIndex.load(INDEX_PATH)

    conf_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    sim_thresholds  = [0.05, 0.10, 0.15, 0.20]

    print("Sweeping thresholds on validation set ...")
    results = sweep(val, svc, proba_lr, index, conf_thresholds, sim_thresholds)

    # Find best: highest auto_rate where unsafe_rate < 0.05
    safe = results[results["unsafe_rate"] < 0.05].copy()
    best = safe.sort_values("auto_rate", ascending=False).iloc[0]

    print(f"\nBest operating point (unsafe_rate < 5%):")
    print(f"  conf_threshold: {best['conf_threshold']}")
    print(f"  sim_threshold:  {best['sim_threshold']}")
    print(f"  auto_rate:      {best['auto_rate']:.4f}")
    print(f"  unsafe_rate:    {best['unsafe_rate']:.4f}")
    print(f"  precision:      {best['precision']:.4f}")

    # Save report
    REPORTS.mkdir(exist_ok=True)
    lines = [
        "# Router Threshold Search\n",
        f"Validation set size: {len(val)}\n",
        "## All results\n",
        "| conf | sim | auto_rate | unsafe_rate | precision | n_auto | n_unsafe |",
        "|------|-----|-----------|-------------|-----------|--------|----------|",
    ]
    for _, r in results.sort_values(
            ["conf_threshold", "sim_threshold"]).iterrows():
        lines.append(
            f"| {r['conf_threshold']} | {r['sim_threshold']} "
            f"| {r['auto_rate']:.4f} | {r['unsafe_rate']:.4f} "
            f"| {r['precision']:.4f} | {r['n_auto']} | {r['n_unsafe']} |"
        )
    lines += [
        f"\n## Selected thresholds",
        f"- conf_threshold = **{best['conf_threshold']}**",
        f"- sim_threshold  = **{best['sim_threshold']}**",
        f"- auto_rate      = {best['auto_rate']:.4f}",
        f"- unsafe_rate    = {best['unsafe_rate']:.4f}",
        f"- precision      = {best['precision']:.4f}",
        "\nThese values are used in router.py.",
    ]
    (REPORTS / "router_threshold_search.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\nReport saved -> {REPORTS / 'router_threshold_search.md'}")
    return best


if __name__ == "__main__":
    main()
