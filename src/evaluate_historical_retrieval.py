"""Evaluate historical retrieval on a deterministic non-golden sample."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from historical_retrieval import HistoricalRetriever, INTERACTIONS_PATH  # noqa: E402

REPORT_OUT = ROOT / "reports" / "historical_retrieval_evaluation.md"
SAMPLE_SIZE = 10
TOP_K = 3


def main() -> None:
    source = pd.read_csv(INTERACTIONS_PATH, dtype=str).fillna("").head(SAMPLE_SIZE)
    query_ids = set(source["interaction_id"])
    retriever = HistoricalRetriever.from_csv(
        exclude_interaction_ids=query_ids
    )
    rows = []
    for _, query in source.iterrows():
        hits = retriever.retrieve(query["customer_message"], top_k=TOP_K)
        top = hits[0] if hits else {}
        rows.append({
            "interaction_id": query["interaction_id"],
            "query": query["customer_message"],
            "top_similarity": top.get("similarity", 0.0),
            "hit_count": len(hits),
            "top_hit_interaction_id": top.get("interaction_id", ""),
            "top_hit_intent": top.get("intent", ""),
        })
    results = pd.DataFrame(rows)
    REPORT_OUT.write_text(
        "\n".join([
            "# Historical retrieval evaluation",
            "",
            "Deterministic smoke evaluation over the first 10 rows of the processed SpotifyCares interaction data.",
            "",
            f"- Source: `{INTERACTIONS_PATH}`",
            f"- Query sample size: **{len(results)}**",
            f"- Retrieval method: word TF-IDF 1–2 grams with cosine similarity.",
            f"- Top-k: **{TOP_K}**",
            "- This is a retrieval smoke evaluation, not a human relevance judgment.",
            "",
            "| Query interaction | Top similarity | Hits | Top historical interaction | V2.1 intent |",
            "|---|---:|---:|---|---|",
        ] + [
            f"| {row['interaction_id']} | {row['top_similarity']:.4f} | "
            f"{row['hit_count']} | {row['top_hit_interaction_id']} | {row['top_hit_intent']} |"
            for _, row in results.iterrows()
        ] + [
            "",
            "## Limitations",
            "",
            "- TF-IDF captures lexical similarity and may miss paraphrases.",
            "- Historical responses are evidence, not guaranteed-correct policies.",
            "- No human relevance labels were created in this phase.",
            "- The retriever does not generate, modify, or invent historical responses.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluated {len(results)} deterministic historical queries")
    print(f"Mean top similarity: {results['top_similarity'].mean():.4f}")
    print(f"Wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
