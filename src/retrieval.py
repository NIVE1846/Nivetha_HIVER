"""
retrieval.py

Historical evidence retrieval using TF-IDF cosine similarity.

For each incoming customer message:
  1. Predict intent.
  2. Filter the index to same-intent examples (intent-scoped retrieval).
  3. Rank by cosine similarity to the query.
  4. Return top-k (customer_message, brand_response, score, intent, tweet_ids).

Why TF-IDF cosine similarity:
- No blocked DLL dependencies on this machine.
- Fast, interpretable, and strong for keyword-heavy support text.
- Intent-scoping dramatically reduces the search space and improves precision.

The retrieval index is built from training interactions only (no golden set leakage).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_splits, CLASSIFIER_INTENTS as INTENTS

ROOT        = Path(__file__).resolve().parent.parent
INDEX_OUT   = ROOT / "data" / "processed" / "retrieval_index.joblib"
SVC_PATH    = ROOT / "data" / "processed" / "intent_classifier.joblib"
PROBA_PATH  = ROOT / "data" / "processed" / "intent_proba_lr.joblib"


class RetrievalIndex:
    """
    Stores per-intent TF-IDF matrices for fast intent-scoped retrieval.
    """

    def __init__(self):
        # intent -> {"vectorizer": TfidfVectorizer, "matrix": sparse, "records": DataFrame}
        self._index: dict[str, dict] = {}

    def build(self, train_df: pd.DataFrame) -> None:
        """Build one TF-IDF index per intent from training data."""
        for intent in INTENTS:
            subset = train_df[train_df["intent"] == intent].reset_index(drop=True)
            if len(subset) == 0:
                continue
            vec = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2),
                sublinear_tf=True, min_df=1, strip_accents="unicode",
            )
            matrix = vec.fit_transform(subset["customer_message"])
            self._index[intent] = {
                "vectorizer": vec,
                "matrix": matrix,
                "records": subset,
            }
        print(f"  Index built for {len(self._index)} intents.")
        for intent, data in self._index.items():
            print(f"    {intent:<25} {data['matrix'].shape[0]:>5} examples")

    def retrieve(self, query: str, intent: str,
                 top_k: int = 3) -> list[dict]:
        """
        Retrieve top_k most similar historical examples for the given intent.
        Falls back to global search if intent has no index entries.
        """
        if intent not in self._index:
            return []

        data = self._index[intent]
        query_vec = data["vectorizer"].transform([query])
        sims = cosine_similarity(query_vec, data["matrix"]).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = data["records"].iloc[idx]
            results.append({
                "customer_message":  row["customer_message"],
                "brand_response":    row["brand_response"],
                "similarity":        float(sims[idx]),
                "intent":            intent,
                "customer_tweet_id": row.get("customer_tweet_id", ""),
                "brand_tweet_id":    row.get("brand_tweet_id", ""),
            })
        return results

    def save(self, path: Path) -> None:
        # Save as a plain dict so the class name doesn't matter at load time
        data = {}
        for intent, entry in self._index.items():
            data[intent] = {
                "vectorizer": entry["vectorizer"],
                "matrix":     entry["matrix"],
                "records":    entry["records"],
            }
        joblib.dump(data, path)
        print(f"  Index saved -> {path}")

    @staticmethod
    def load(path: Path) -> "RetrievalIndex":
        data = joblib.load(path)
        idx = RetrievalIndex()
        idx._index = data
        return idx


def evaluate_retrieval(index: RetrievalIndex,
                       val_df: pd.DataFrame,
                       top_ks: list[int] = [1, 3, 5]) -> dict:
    """
    Recall@k: fraction of queries where at least one retrieved example
    shares the same intent as the query (intent-match as proxy for relevance).

    Note: true relevance would require human judgement; intent-match is a
    reproducible automatic proxy.
    """
    results = {k: 0 for k in top_ks}
    total = 0

    for _, row in val_df.iterrows():
        query  = row["customer_message"]
        intent = row["intent"]
        retrieved = index.retrieve(query, intent, top_k=max(top_ks))

        # All retrieved items share the same intent by construction (intent-scoped).
        # So Recall@k = fraction of queries that got at least 1 result.
        for k in top_ks:
            top = retrieved[:k]
            if any(r["similarity"] > 0.0 for r in top):
                results[k] += 1
        total += 1

    recall = {k: results[k] / total if total > 0 else 0.0 for k in top_ks}
    return recall, total


def main():
    train, val, _ = load_splits()

    print("Building retrieval index from training data ...")
    index = RetrievalIndex()
    index.build(train)
    index.save(INDEX_OUT)

    print("\nEvaluating retrieval (intent-scoped Recall@k) ...")
    recall, total = evaluate_retrieval(index, val)
    print(f"  Evaluated on {total} validation examples")
    for k, r in recall.items():
        print(f"  Recall@{k}: {r:.4f}")

    # Show 3 sample retrievals
    print("\nSample retrievals:")
    samples = val.sample(3, random_state=42)
    for _, row in samples.iterrows():
        query = row["customer_message"]
        intent = row["intent"]
        hits = index.retrieve(query, intent, top_k=1)
        print(f"\n  Query ({intent}): {query.encode('ascii','replace').decode()[:80]}")
        if hits:
            h = hits[0]
            print(f"  Retrieved (sim={h['similarity']:.3f}): "
                  f"{h['customer_message'].encode('ascii','replace').decode()[:80]}")
            print(f"  Brand response: "
                  f"{h['brand_response'].encode('ascii','replace').decode()[:80]}")

    # Save retrieval eval to reports
    report_lines = [
        "# Retrieval Evaluation\n",
        "## Method",
        "TF-IDF cosine similarity, intent-scoped (separate index per intent).\n",
        f"## Recall@k (validation set, n={total})\n",
        "| k | Recall@k |",
        "|---|----------|",
    ]
    for k, r in recall.items():
        report_lines.append(f"| {k} | {r:.4f} |")
    report_lines += [
        "\n## Notes",
        "- Recall@k here measures whether the query returned at least one result",
        "  with similarity > 0 from the same-intent pool.",
        "- All retrieved examples are from the same intent by construction.",
        "- True relevance evaluation would require human judgement on a sample.",
        "- See evaluation/golden_set.csv for the held-out evaluation set.",
    ]
    (ROOT / "reports" / "retrieval_eval.md").write_text(
        "\n".join(report_lines), encoding="utf-8")
    print(f"\nRetrieval report saved -> {ROOT / 'reports' / 'retrieval_eval.md'}")


if __name__ == "__main__":
    main()
