import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from run_v2_1_classifier_experiment import (  # noqa: E402
    INTENTS,
    build_pipeline,
    duplicate_groups,
    load_candidates,
    normalize_message,
    split_groups,
)


def test_candidates_exclude_unresolved_and_use_only_allowed_intents():
    candidates = load_candidates()
    assert len(candidates) == 1684
    assert set(candidates["v2_1_intent"]) == set(INTENTS)
    assert not candidates["v2_1_intent"].isna().any()


def test_duplicate_groups_do_not_cross_splits():
    candidates = load_candidates()
    train, validation, test, _ = split_groups(candidates)
    memberships = {index: split for split, indices in {
        "train": train, "validation": validation, "test": test
    }.items() for index in indices}
    for group in duplicate_groups(candidates):
        assert len({memberships[index] for index in group}) == 1


def test_all_classes_are_present_in_each_split():
    candidates = load_candidates()
    train, validation, test, _ = split_groups(candidates)
    for indices in (train, validation, test):
        assert set(candidates.loc[indices, "v2_1_intent"]) == set(INTENTS)


def test_normalization_and_balanced_configuration():
    assert normalize_message(" Hello,  WORLD! ") == "hello world"
    classifier = build_pipeline().named_steps["clf"]
    assert classifier.class_weight == "balanced"


def test_source_and_protected_paths_are_not_experiment_outputs():
    from run_v2_1_classifier_experiment import ARTIFACT_ROOT, INPUT, REPORT
    assert INPUT.name == "rule_v2_1_labels.csv"
    assert "experiments" in str(ARTIFACT_ROOT)
    assert REPORT.name == "v2_1_classifier_experiment.md"
