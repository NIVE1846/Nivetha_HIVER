"""
retrieval_evaluation.py

Evaluates retrieval quality by measuring actual evidence usefulness.

Recall@k = 1.0 is NOT reported as a headline metric because intent-scoped
retrieval trivially always returns something. Instead we measure:

1. Similarity score distribution per intent
2. Fraction of queries with similarity >= meaningful thresholds (0.10, 0.20, 0.30)
3. A manually-reviewable sample of 30 (query, retrieved) pairs for human relevance scoring
4. Token overlap between query and retrieved response as a proxy for relevance

Produces:
  reports/retrieval_eval.md  (updated)
  evaluation/retrieval_sample_for_review.csv  (30 examples for human scoring)
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib
import re

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from retrieval import RetrievalIndex
from utils import load_splits, INTENTS

INDEX_PATH = ROOT / "data" / "processed" / "retrieval_index.joblib"
REPORTS    = ROOT / "reports"


def token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard overlap of non-trivial tokens between two texts."""
    stop = {"i","a","the","to","is","it","and","in","of","you","we","can",
            "for","on","my","your","this","that","be","us","just","let","know"}
    def tokens(t):
        return {w for w in re.sub(r"[^a-z0-9 ]"," ",str(t).lower()).split()
                if w not in stop and len(w) > 2}
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def evaluate_retrieval(val_df: pd.DataFrame, index: RetrievalIndex) -> dict:
    """
    For each validation query, retrieve top-3 and measure:
    - similarity score of top-1
    - token overlap between query and top-1 retrieved response
    """
    results = []
    for _, row in val_df.iterrows():
        query  = row["customer_message"]
        intent = row["intent"]
        hits   = index.retrieve(query, intent, top_k=3)

        if not hits:
            results.append({
                "intent": intent, "top1_sim": 0.0,
                "top1_overlap": 0.0, "n_hits": 0
            })
            continue

        top1 = hits[0]
        overlap = token_overlap_ratio(query, top1["brand_response"])
        results.append({
            "intent":       intent,
            "top1_sim":     top1["similarity"],
            "top1_overlap": overlap,
            "n_hits":       len(hits),
        })
    return results


def build_review_sample(val_df: pd.DataFrame, index: RetrievalIndex,
                        n: int = 30) -> pd.DataFrame:
    """
    Build a 30-example sample for human relevance scoring.
    Stratified: ~4-5 examples per intent.
    """
    rows = []
    per_intent = max(1, n // len(INTENTS))
    for intent in INTENTS:
        subset = val_df[val_df["intent"] == intent]
        sample = subset.sample(min(per_intent, len(subset)), random_state=42)
        for _, row in sample.iterrows():
            hits = index.retrieve(row["customer_message"], intent, top_k=1)
            if not hits:
                continue
            h = hits[0]
            rows.append({
                "query_intent":        intent,
                "customer_message":    row["customer_message"],
                "retrieved_customer":  h["customer_message"],
                "retrieved_response":  h["brand_response"],
                "similarity_score":    round(h["similarity"], 4),
                # Human reviewer fills this in: 0=irrelevant, 1=partial, 2=relevant
                "human_relevance":     "",
                "reviewer_notes":      "",
            })
    return pd.DataFrame(rows[:n])


def main():
    _, val, _ = load_splits()
    index = RetrievalIndex.load(INDEX_PATH)

    print(f"Evaluating retrieval on {len(val)} validation examples ...")
    results = evaluate_retrieval(val, index)
    rdf = pd.DataFrame(results)

    # Overall stats
    print(f"\nOverall similarity distribution (top-1):")
    print(f"  Mean:   {rdf['top1_sim'].mean():.4f}")
    print(f"  Median: {rdf['top1_sim'].median():.4f}")
    print(f"  Std:    {rdf['top1_sim'].std():.4f}")
    print(f"  >= 0.10: {(rdf['top1_sim'] >= 0.10).mean():.3f}")
    print(f"  >= 0.20: {(rdf['top1_sim'] >= 0.20).mean():.3f}")
    print(f"  >= 0.30: {(rdf['top1_sim'] >= 0.30).mean():.3f}")

    print(f"\nToken overlap (query vs retrieved response):")
    print(f"  Mean:   {rdf['top1_overlap'].mean():.4f}")
    print(f"  >= 0.05: {(rdf['top1_overlap'] >= 0.05).mean():.3f}")
    print(f"  >= 0.10: {(rdf['top1_overlap'] >= 0.10).mean():.3f}")

    # Per-intent breakdown
    print(f"\nPer-intent similarity (top-1):")
    per_intent = rdf.groupby("intent")["top1_sim"].agg(["mean","median","count"])
    print(per_intent.to_string())

    # Build review sample
    sample_df = build_review_sample(val, index, n=30)
    out_path = ROOT / "evaluation" / "retrieval_sample_for_review.csv"
    sample_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nReview sample saved -> {out_path} ({len(sample_df)} rows)")

    # Write report
    per_intent_rows = "\n".join([
        f"| {intent} | {row['mean']:.4f} | {row['median']:.4f} | {int(row['count'])} |"
        for intent, row in per_intent.iterrows()
    ])

    report = f"""# Retrieval Evaluation

## Important note on Recall@k

Recall@1/3/5 = 1.0 was previously reported as a headline metric.
This is NOT meaningful because retrieval is intent-scoped: every query
searches only within its predicted intent's index, so it always finds
at least one result. This metric has been removed from headline results.

## What we measure instead

For each validation query, we retrieve the top-1 most similar historical
example and measure:
1. **Cosine similarity score** — how similar the query is to the retrieved example
2. **Token overlap** — how much vocabulary the query shares with the retrieved response

These measure whether the retrieved evidence is actually useful, not just present.

## Similarity score distribution (validation set, n={len(val)})

| Metric | Value |
|--------|-------|
| Mean top-1 similarity | {rdf['top1_sim'].mean():.4f} |
| Median top-1 similarity | {rdf['top1_sim'].median():.4f} |
| Std top-1 similarity | {rdf['top1_sim'].std():.4f} |
| Fraction >= 0.10 | {(rdf['top1_sim'] >= 0.10).mean():.3f} |
| Fraction >= 0.20 | {(rdf['top1_sim'] >= 0.20).mean():.3f} |
| Fraction >= 0.30 | {(rdf['top1_sim'] >= 0.30).mean():.3f} |

## Token overlap (query vs retrieved response)

| Metric | Value |
|--------|-------|
| Mean overlap | {rdf['top1_overlap'].mean():.4f} |
| Fraction >= 0.05 | {(rdf['top1_overlap'] >= 0.05).mean():.3f} |
| Fraction >= 0.10 | {(rdf['top1_overlap'] >= 0.10).mean():.3f} |

## Per-intent similarity breakdown

| Intent | Mean sim | Median sim | N |
|--------|----------|------------|---|
{per_intent_rows}

## Human relevance evaluation

A 30-example stratified sample has been saved to:
`evaluation/retrieval_sample_for_review.csv`

Each row contains:
- The customer query
- The retrieved historical customer message
- The retrieved historical brand response
- The cosine similarity score

A human reviewer should score each retrieved response:
- 0 = irrelevant (retrieved response addresses a completely different problem)
- 1 = partially relevant (same general topic but different specific issue)
- 2 = clearly relevant (retrieved response directly addresses the query)

Results will be reported after human review.

## Interpretation

Low similarity scores (mean ~{rdf['top1_sim'].mean():.2f}) are expected because:
- Twitter support messages are very short (median ~67 characters)
- Customers describe the same problem in many different ways
- TF-IDF similarity penalises short documents

The meaningful question is not "is similarity high?" but "does the retrieved
response provide useful evidence for answering this customer's problem?"
This requires human evaluation, which is pending.
"""
    (REPORTS / "retrieval_eval.md").write_text(report, encoding="utf-8")
    print(f"Report saved -> {REPORTS / 'retrieval_eval.md'}")


if __name__ == "__main__":
    main()
