"""
conversation_builder.py

Reconstructs usable customer->brand interaction records from spotify_raw.csv.

Each output record represents one customer message and the brand's direct reply.
Multi-turn context (prior turns in the thread) is attached where available.

Output: data/processed/interactions.csv
        data/processed/build_log.txt
"""

from pathlib import Path
import pandas as pd
import re
import json

ROOT      = Path(__file__).resolve().parent.parent
RAW_CSV   = ROOT / "data" / "processed" / "spotify_raw.csv"
OUT_CSV   = ROOT / "data" / "processed" / "interactions.csv"
LOG_PATH  = ROOT / "data" / "processed" / "build_log.txt"
BRAND     = "SpotifyCares"


def clean_text(text: str) -> str:
    """Remove @mentions and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+", "", text)   # strip @handles
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_interactions(raw_csv: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(raw_csv, dtype=str, low_memory=False)

    # Normalise columns
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower()
    df["tweet_id"] = df["tweet_id"].str.strip()
    df["author_id"] = df["author_id"].str.strip()
    df["text"] = df["text"].fillna("")
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").str.strip()
    df["response_tweet_id"] = df["response_tweet_id"].fillna("").str.strip()

    # Index for fast lookup: tweet_id -> row
    tweet_index: dict[str, pd.Series] = {
        row["tweet_id"]: row for _, row in df.iterrows()
    }

    brand_tweets  = df[df["author_id"] == BRAND]
    customer_tweets = df[df["inbound"] == "true"]

    # Build a map: brand_tweet_id -> list of direct customer replies
    # A customer reply has in_response_to_tweet_id == brand_tweet_id
    brand_ids = set(brand_tweets["tweet_id"].tolist())
    customer_replies = customer_tweets[
        customer_tweets["in_response_to_tweet_id"].isin(brand_ids)
    ]

    # For each brand tweet, find the customer message it was replying to
    # brand tweet's in_response_to_tweet_id -> the customer message
    records = []
    stats = {
        "total_brand_tweets": len(brand_tweets),
        "brand_tweets_with_customer_parent": 0,
        "missing_customer_parent": 0,
        "duplicate_skipped": 0,
        "final_interactions": 0,
    }

    seen_pairs: set[tuple] = set()

    for _, brand_row in brand_tweets.iterrows():
        parent_id = brand_row["in_response_to_tweet_id"]
        if not parent_id:
            stats["missing_customer_parent"] += 1
            continue

        parent = tweet_index.get(parent_id)
        if parent is None:
            # Parent tweet not in our extract (deleted or outside brand thread)
            stats["missing_customer_parent"] += 1
            continue

        if parent["inbound"] != "true":
            # Brand replied to another brand tweet - skip
            continue

        stats["brand_tweets_with_customer_parent"] += 1

        pair_key = (parent_id, brand_row["tweet_id"])
        if pair_key in seen_pairs:
            stats["duplicate_skipped"] += 1
            continue
        seen_pairs.add(pair_key)

        # Gather prior context: walk up the chain one more step if available
        context_id = parent.get("in_response_to_tweet_id", "")
        context_text = ""
        if context_id:
            context_row = tweet_index.get(context_id)
            if context_row is not None:
                context_text = clean_text(context_row["text"])

        records.append({
            "interaction_id":       f"{parent_id}_{brand_row['tweet_id']}",
            "customer_tweet_id":    parent_id,
            "brand_tweet_id":       brand_row["tweet_id"],
            "customer_message":     clean_text(parent["text"]),
            "brand_response":       clean_text(brand_row["text"]),
            "prior_context":        context_text,
            "customer_created_at":  parent.get("created_at", ""),
            "brand_created_at":     brand_row.get("created_at", ""),
            "brand":                BRAND,
        })

    stats["final_interactions"] = len(records)
    return pd.DataFrame(records), stats


def main():
    print(f"Reading {RAW_CSV} ...")
    interactions, stats = build_interactions(RAW_CSV)

    interactions.to_csv(OUT_CSV, index=False)

    log_lines = [
        "Conversation Build Log",
        "======================",
        f"Total brand tweets in extract:          {stats['total_brand_tweets']:,}",
        f"Brand tweets with customer parent:      {stats['brand_tweets_with_customer_parent']:,}",
        f"Missing/deleted customer parents:       {stats['missing_customer_parent']:,}",
        f"Duplicate pairs skipped:                {stats['duplicate_skipped']:,}",
        f"Final interaction records:              {stats['final_interactions']:,}",
        "",
        "Handling notes:",
        "- Tweets whose parent is not in the SpotifyCares extract are skipped",
        "  (parent was deleted, or belongs to a different brand thread).",
        "- Brand->brand reply chains are excluded (not customer interactions).",
        "- @mentions are stripped from text; whitespace is normalised.",
        "- Duplicate (customer_id, brand_id) pairs are deduplicated.",
        "- One level of prior context is attached where available.",
    ]
    LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")

    print("\nBuild stats:")
    for k, v in stats.items():
        print(f"  {k:<45} {v:,}")
    print(f"\nInteractions saved -> {OUT_CSV}")
    print(f"Log saved          -> {LOG_PATH}")

    # Quick sanity check
    df = pd.read_csv(OUT_CSV)
    empty_customer = (df["customer_message"].str.strip() == "").sum()
    empty_brand    = (df["brand_response"].str.strip() == "").sum()
    print(f"\nSanity check:")
    print(f"  Empty customer messages: {empty_customer}")
    print(f"  Empty brand responses:   {empty_brand}")


if __name__ == "__main__":
    main()
