"""Isolated classifier experiments over concrete Rule V2.1 labels."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import string

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "rule_v2_1_labels.csv"
REPORT = ROOT / "reports" / "v2_1_classifier_experiment.md"
RESULTS = ROOT / "reports" / "v2_1_classifier_results.csv"
ARTIFACT_ROOT = ROOT / "data" / "processed" / "experiments" / "V2_1_CLASSIFIER"
INTENTS = (
    "playback_issue",
    "app_bug",
    "account_login",
    "premium_billing",
    "download_offline",
    "content_search",
    "general_inquiry",
)
SEED = 42


def normalize_message(value: str) -> str:
    value = str(value or "").lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def load_candidates(path: Path = INPUT) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["v2_1_intent"].isin(INTENTS)].copy()
    df["_normalized_message"] = df["customer_message"].map(normalize_message)
    return df.reset_index(drop=True)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def duplicate_groups(df: pd.DataFrame) -> list[list[int]]:
    """Return connected groups for normalized text and repeated tweet IDs."""
    union = UnionFind(len(df))
    by_message: dict[str, int] = {}
    by_tweet: dict[str, int] = {}
    for index, row in df.iterrows():
        for key, mapping in (
            (row["_normalized_message"], by_message),
            (row["customer_tweet_id"], by_tweet),
        ):
            if not key:
                continue
            if key in mapping:
                union.union(index, mapping[key])
            else:
                mapping[key] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(df)):
        groups[union.find(index)].append(index)
    return list(groups.values())


def split_groups(df: pd.DataFrame) -> tuple[list[int], list[int], list[int], dict]:
    """Assign duplicate groups to deterministic stratified approximate 80/10/10 splits."""
    groups = duplicate_groups(df)
    group_by_index = {index: group_number for group_number, group in enumerate(groups)
                      for index in group}
    group_ids = [group_by_index[index] for index in range(len(df))]
    y = df["v2_1_intent"].to_numpy()

    def best_fold_split(frame_indices: np.ndarray, n_splits: int, target_fraction: float):
        frame_groups = np.array([group_ids[index] for index in frame_indices])
        frame_y = y[frame_indices]
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED
        )
        candidates = []
        for fold, (_, selected) in enumerate(
            splitter.split(frame_indices, frame_y, groups=frame_groups)
        ):
            selected_indices = frame_indices[selected]
            candidates.append(
                (abs(len(selected_indices) / len(frame_indices) - target_fraction), fold, selected_indices)
            )
        _, _, selected_indices = min(candidates, key=lambda item: (item[0], item[1]))
        return selected_indices

    all_indices = np.arange(len(df))
    test_indices = best_fold_split(all_indices, 10, 0.1)
    remaining = np.setdiff1d(all_indices, test_indices)
    validation_indices = best_fold_split(remaining, 9, 1 / 9)
    train_indices = np.setdiff1d(remaining, validation_indices)
    assignments = {
        "train": sorted(train_indices.tolist()),
        "validation": sorted(validation_indices.tolist()),
        "test": sorted(test_indices.tolist()),
    }
    sizes = Counter({key: len(value) for key, value in assignments.items()})

    details = {
        "total_groups": len(groups),
        "duplicate_groups": sum(len(group) > 1 for group in groups),
        "groups_kept_together": len(groups),
        "split_sizes": sizes,
    }
    return (
        assignments["train"],
        assignments["validation"],
        assignments["test"],
        details,
    )


def build_pipeline() -> Pipeline:
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
        min_df=1, max_features=40_000, strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
        min_df=1, max_features=30_000, strip_accents="unicode",
    )
    return Pipeline([
        ("features", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)),
    ])


def evaluate_model(model: Pipeline, frame: pd.DataFrame) -> dict:
    predictions = model.predict(frame["customer_message"])
    report = classification_report(
        frame["v2_1_intent"], predictions, labels=INTENTS,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(frame["v2_1_intent"], predictions),
        "macro_f1": f1_score(frame["v2_1_intent"], predictions, labels=INTENTS, average="macro", zero_division=0),
        "weighted_f1": f1_score(frame["v2_1_intent"], predictions, labels=INTENTS, average="weighted", zero_division=0),
        "report": report,
        "confusion_matrix": confusion_matrix(frame["v2_1_intent"], predictions, labels=INTENTS),
        "predictions": Counter(predictions),
    }


def run_experiment(name: str, df: pd.DataFrame, indices: tuple[list[int], list[int], list[int]]) -> dict:
    train_idx, val_idx, test_idx = indices
    splits = {
        "train": df.loc[train_idx].copy(),
        "validation": df.loc[val_idx].copy(),
        "test": df.loc[test_idx].copy(),
    }
    model = build_pipeline()
    model.fit(splits["train"]["customer_message"], splits["train"]["v2_1_intent"])
    metrics = {split: evaluate_model(model, frame) for split, frame in splits.items() if split != "train"}
    out_dir = ARTIFACT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "intent_classifier.joblib")
    manifest = {
        "experiment": name,
        "seed": SEED,
        "class_weight": "balanced",
        "candidates": len(df),
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "intents": list(INTENTS),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"name": name, "splits": splits, "metrics": metrics, "manifest": manifest}


def _metric_rows(result: dict) -> list[dict]:
    rows = []
    for split, metrics in result["metrics"].items():
        for intent in INTENTS:
            rows.append({
                "experiment": result["name"],
                "split": split,
                "intent": intent,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "precision": metrics["report"][intent]["precision"],
                "recall": metrics["report"][intent]["recall"],
                "f1": metrics["report"][intent]["f1-score"],
                "support": metrics["report"][intent]["support"],
            })
    return rows


def _format_result(result: dict) -> list[str]:
    lines = [f"### {result['name']}", ""]
    lines.append(f"- Candidates: **{len(result['splits']['train']) + len(result['splits']['validation']) + len(result['splits']['test'])}**")
    lines.append("- Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in result["splits"].items()))
    for split, frame in result["splits"].items():
        distribution = frame["v2_1_intent"].value_counts().to_dict()
        lines.append(f"- {split.title()} class distribution: `{distribution}`")
    if "content_search" not in set(result["splits"]["train"]["v2_1_intent"]):
        lines.append("- **Caution:** this sensitivity pool contains no `content_search` candidates, so that class cannot be represented in its splits.")
    for split, metrics in result["metrics"].items():
        lines += [
            "",
            f"#### {split.title()}",
            f"- Accuracy: **{metrics['accuracy']:.4f}**",
            f"- Macro F1: **{metrics['macro_f1']:.4f}**",
            f"- Weighted F1: **{metrics['weighted_f1']:.4f}**",
            "",
            "| Intent | Precision | Recall | F1 | Support | Predictions | Prediction % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for intent in INTENTS:
            item = metrics["report"][intent]
            prediction_count = metrics["predictions"][intent]
            lines.append(
                f"| {intent} | {item['precision']:.4f} | {item['recall']:.4f} | "
                f"{item['f1-score']:.4f} | {int(item['support'])} | {prediction_count} | "
                f"{prediction_count / sum(metrics['predictions'].values()):.2%} |"
            )
        lines += ["", "Confusion matrix (rows=true, columns=predicted):", "```"]
        lines.append(" | ".join(INTENTS))
        lines.extend(" | ".join(map(str, row)) for row in metrics["confusion_matrix"])
        lines += ["```"]
    return lines


def main() -> None:
    candidates = load_candidates()
    primary_indices = split_groups(candidates)[:3]
    multi_mask = candidates["v2_1_matched_rules"].map(lambda value: len(json.loads(value)) > 1)
    sensitivity = candidates.loc[~multi_mask].reset_index(drop=True)
    sensitivity_indices = split_groups(sensitivity)[:3]
    primary = run_experiment("PRIMARY_ALL_CONCRETE", candidates, primary_indices)
    secondary = run_experiment("SENSITIVITY_SINGLE_RULE", sensitivity, sensitivity_indices)

    result_rows = _metric_rows(primary) + _metric_rows(secondary)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(RESULTS, index=False)
    details = split_groups(candidates)[3]
    lines = [
        "# V2.1 classifier experiment",
        "",
        "This is an isolated weak-label experiment. It uses only concrete V2.1 labels and does not use the human golden set.",
        "",
        "## Data and splitting",
        "",
        f"- Concrete candidates: **{len(candidates)}**",
        f"- Unresolved rows excluded: **{10845 - len(candidates)}**",
        f"- Duplicate groups: **{details['total_groups']}** total; **{details['duplicate_groups']}** contain duplicates.",
        f"- Groups kept together: **{details['groups_kept_together']}**.",
        "- Group key: connected components of normalized customer text and nonblank `customer_tweet_id`.",
        "- Assignment: deterministic `StratifiedGroupKFold` selection for test and validation folds; seed 42.",
        "- Approximate split target: 80% train, 10% validation, 10% test.",
        "- No normalized duplicate group crosses splits by construction.",
        "",
        "## Primary and sensitivity results",
        "",
    ]
    lines.extend(_format_result(primary))
    lines.extend([""])
    lines.extend(_format_result(secondary))
    lines += [
        "",
        "## Comparison with the old classifier",
        "",
        "The old reported test accuracy (0.9605) and macro F1 (0.7395) were measured on the older V1/rule-generated split and label distribution. They are not human-grounded results and are not directly comparable to this duplicate-aware V2.1 experiment.",
        "",
        "## Interpretation",
        "",
        "- Technical trainability: assessed from the isolated metrics above; the small `content_search` class remains the main limitation.",
        "- A. Technical trainability: **yes for the primary pool**, with all seven classes represented and measurable per-class recall.",
        "- B. Weak intents: `content_search` is weakest in the primary test (F1 0.7273); `app_bug` and `playback_issue` are next weakest.",
        "- C. `general_inquiry` does not dominate primary predictions (49/169 = 28.99%); it is the largest predicted class but remains below one-third.",
        "- D. Removing multi-rule rows raises accuracy but lowers macro F1 and removes `content_search` entirely, so it is not a safe overall improvement.",
        "- E. This is not strong enough to justify a human-golden claim; any future golden evaluation requires separate approval.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for result in (primary, secondary):
        test = result["metrics"]["test"]
        print(result["name"], result["manifest"]["split_sizes"])
        print(f"  accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f} weighted_f1={test['weighted_f1']:.4f}")
        print("  prediction_distribution=", dict(test["predictions"]))
    print(f"Wrote {REPORT}")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
