import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from run_classifier_experiment import (  # noqa: E402
    EXPERIMENT_ROOT,
    GENERAL_INQUIRY_CAP,
    prepare_refined_training,
)
from utils import CLASSIFIER_INTENTS, load_splits  # noqa: E402


def test_experiment_training_counts_and_minority_retention():
    train, _, _ = load_splits()
    refined = prepare_refined_training(train)
    assert len(train) == 8506
    assert len(refined) == 2471
    assert int((refined["intent"] == "general_inquiry").sum()) == GENERAL_INQUIRY_CAP
    for intent in CLASSIFIER_INTENTS:
        if intent != "general_inquiry":
            assert int((refined["intent"] == intent).sum()) == int((train["intent"] == intent).sum())


def test_validation_and_test_are_unchanged():
    _, val_a, test_a = load_splits()
    _, val_b, test_b = load_splits()
    pd.testing.assert_frame_equal(val_a, val_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_refined_sampling_is_deterministic():
    train, _, _ = load_splits()
    pd.testing.assert_frame_equal(prepare_refined_training(train), prepare_refined_training(train))


def test_experiment_paths_do_not_overwrite_baseline_paths():
    baseline_paths = {
        SRC.parent / "data" / "processed" / "intent_classifier.joblib",
        SRC.parent / "data" / "processed" / "intent_proba_lr.joblib",
    }
    experiment_paths = {
        EXPERIMENT_ROOT / "EXP1_A_BASELINE" / "intent_classifier.joblib",
        EXPERIMENT_ROOT / "EXP1_A_BASELINE" / "intent_proba_lr.joblib",
        EXPERIMENT_ROOT / "EXP1_B_REFINED_CAP3X" / "intent_classifier.joblib",
        EXPERIMENT_ROOT / "EXP1_B_REFINED_CAP3X" / "intent_proba_lr.joblib",
    }
    assert baseline_paths.isdisjoint(experiment_paths)
    assert all(path.exists() for path in baseline_paths)
