import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from v2_1_classifier_adapter import (  # noqa: E402
    MODEL_PATH,
    load_v2_1_classifier,
    predict_message,
)
from run_v2_1_classifier_experiment import load_candidates, split_groups  # noqa: E402


def test_v2_1_artifact_loads_and_predicts():
    model = load_v2_1_classifier()
    result = predict_message("I can't log into my Spotify account", model=model)
    assert result["intent"] in set(model.named_steps["clf"].classes_)
    assert result["decision_score"]
    assert result["confidence"] is None
    assert isinstance(result["predicted_decision_score"], float)


def test_v2_1_test_set_smoke_predictions():
    candidates = load_candidates()
    _, _, test_indices, _ = split_groups(candidates)
    model = load_v2_1_classifier()
    for message in candidates.loc[test_indices[:10], "customer_message"]:
        result = predict_message(message, model=model)
        assert result["intent"] in set(model.named_steps["clf"].classes_)


def test_adapter_uses_isolated_primary_artifact():
    assert "V2_1_CLASSIFIER" in str(MODEL_PATH)
    assert "PRIMARY_ALL_CONCRETE" in str(MODEL_PATH)
    assert MODEL_PATH.name == "intent_classifier.joblib"
