import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from v2_1_classifier_adapter import MODEL_PATH  # noqa: E402

sys.path.insert(0, str(ROOT / "evaluation"))
from evaluate_v2_1_golden import (  # noqa: E402
    GOLDEN_CSV,
    HUMAN_LABELS_CSV,
    RESULTS_OUT,
    load_aligned_golden,
)


def test_golden_ids_align_one_to_one():
    golden = pd.read_csv(GOLDEN_CSV, dtype=str)
    labels = pd.read_csv(HUMAN_LABELS_CSV, dtype=str)
    assert len(golden) == 203
    assert len(labels) == 203
    assert golden["example_id"].is_unique
    assert labels["example_id"].is_unique
    assert set(golden["example_id"]) == set(labels["example_id"])


def test_no_blank_human_labels_and_aligned_merge():
    aligned = load_aligned_golden()
    assert len(aligned) == 203
    assert aligned["human_intent"].str.strip().ne("").all()
    assert aligned["should_escalate"].str.strip().ne("").all()


def test_v2_1_predictions_cover_all_golden_examples():
    from v2_1_classifier_adapter import load_v2_1_classifier, predict_messages
    aligned = load_aligned_golden()
    predictions = predict_messages(
        aligned["customer_message"].tolist(),
        model=load_v2_1_classifier(),
    )
    assert len(predictions) == 203
    assert all(item["intent"] for item in predictions)


def test_isolated_outputs_and_adapter_are_used():
    source = (ROOT / "evaluation" / "evaluate_v2_1_golden.py").read_text(
        encoding="utf-8"
    )
    assert "v2_1_classifier_adapter" in source
    assert RESULTS_OUT.name == "v2_1_golden_results.csv"
    assert MODEL_PATH.name == "intent_classifier.joblib"
    assert "V2_1_CLASSIFIER" in str(MODEL_PATH)
    assert not RESULTS_OUT.samefile(ROOT / "evaluation" / "eval_results.csv") if RESULTS_OUT.exists() else True
