"""Prediction adapter for the isolated V2.1 classifier artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "experiments"
    / "V2_1_CLASSIFIER"
    / "PRIMARY_ALL_CONCRETE"
    / "intent_classifier.joblib"
)


def load_v2_1_classifier(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the serialized TF-IDF + LinearSVC pipeline."""
    if not model_path.is_file():
        raise FileNotFoundError(f"V2.1 classifier artifact not found: {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected a sklearn Pipeline, got {type(model).__name__}")
    if "features" not in model.named_steps or "clf" not in model.named_steps:
        raise ValueError("V2.1 artifact must contain 'features' and 'clf' pipeline steps")
    if not hasattr(model.named_steps["clf"], "decision_function"):
        raise TypeError("V2.1 classifier does not expose decision_function")
    return model


def predict_message(
    message: str,
    model: Pipeline | None = None,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Predict one customer message and return its label and decision scores."""
    classifier = model or load_v2_1_classifier(model_path)
    predicted = str(classifier.predict([message])[0])
    scores = classifier.decision_function([message])
    raw_scores = np.asarray(scores)
    if raw_scores.ndim == 1:
        score_values = raw_scores.tolist()
    else:
        score_values = raw_scores[0].tolist()
    classes = classifier.named_steps["clf"].classes_.tolist()
    predicted_index = classes.index(predicted)
    return {
        "intent": predicted,
        "decision_score": score_values,
        "predicted_decision_score": float(score_values[predicted_index]),
        "classes": classes,
        "confidence": None,
    }


def predict_messages(
    messages: list[str],
    model: Pipeline | None = None,
    model_path: Path = MODEL_PATH,
) -> list[dict[str, Any]]:
    """Predict multiple customer messages with one loaded model."""
    classifier = model or load_v2_1_classifier(model_path)
    predictions = classifier.predict(messages)
    scores = np.asarray(classifier.decision_function(messages))
    classes = classifier.named_steps["clf"].classes_.tolist()
    if scores.ndim == 1:
        scores = scores[:, None]
    return [
        {
            "intent": str(intent),
            "decision_score": row.tolist(),
            "predicted_decision_score": float(row[classes.index(str(intent))]),
            "classes": classes,
            "confidence": None,
        }
        for intent, row in zip(predictions, scores)
    ]
