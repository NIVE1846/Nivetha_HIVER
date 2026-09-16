"""Standalone TF-IDF retrieval over historical SpotifyCares interactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
INTERACTIONS_PATH = ROOT / "data" / "processed" / "interactions.csv"
V2_1_LABELS_PATH = ROOT / "data" / "processed" / "rule_v2_1_labels.csv"


class HistoricalRetriever:
    """Retrieve similar historical customer messages and their responses."""

    def __init__(self, vectorizer: TfidfVectorizer, matrix, records: pd.DataFrame):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.records = records.reset_index(drop=True)

    @classmethod
    def from_csv(
        cls,
        interactions_path: Path = INTERACTIONS_PATH,
        labels_path: Path | None = V2_1_LABELS_PATH,
        exclude_interaction_ids: set[str] | None = None,
    ) -> "HistoricalRetriever":
        records = pd.read_csv(interactions_path, dtype=str).fillna("")
        if exclude_interaction_ids:
            records = records[
                ~records["interaction_id"].isin(exclude_interaction_ids)
            ].copy()
        required = {"interaction_id", "customer_message", "brand_response"}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"Historical data missing columns: {sorted(missing)}")
        if labels_path is not None and labels_path.is_file():
            labels = pd.read_csv(labels_path, dtype=str).fillna("")
            if {"interaction_id", "v2_1_intent"}.issubset(labels.columns):
                records = records.merge(
                    labels[["interaction_id", "v2_1_intent"]],
                    on="interaction_id",
                    how="left",
                    validate="one_to_one",
                )
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(records["customer_message"])
        return cls(vectorizer, matrix, records)

    def retrieve(
        self,
        customer_message: str,
        top_k: int = 3,
        intent: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = self.vectorizer.transform([str(customer_message)])
        similarities = cosine_similarity(query_vector, self.matrix).ravel()
        candidate_indices = np.arange(len(self.records))
        if intent and "v2_1_intent" in self.records:
            scoped = self.records["v2_1_intent"].eq(intent).to_numpy()
            if scoped.any():
                candidate_indices = candidate_indices[scoped]
        ranked = candidate_indices[np.argsort(similarities[candidate_indices])[::-1][:top_k]]
        results = []
        for index in ranked:
            row = self.records.iloc[int(index)]
            results.append({
                "interaction_id": row["interaction_id"],
                "customer_message": row["customer_message"],
                "brand_response": row["brand_response"],
                "similarity": float(similarities[index]),
                "intent": row.get("v2_1_intent", ""),
                "customer_tweet_id": row.get("customer_tweet_id", ""),
                "brand_tweet_id": row.get("brand_tweet_id", ""),
            })
        return results
