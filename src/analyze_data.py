"""
Phase 1: Dataset analysis and brand selection.

Reads twcs.csv in chunks to stay within 8 GB RAM.
Produces reports/dataset_analysis.md with real statistics.
"""

from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
CSV_PATH  = ROOT / "archive" / "twcs" / "twcs.csv"
REPORT    = ROOT / "reports" / "dataset_analysis.md"
CHUNK     = 100_000   # rows per chunk - safe for 8 GB RAM


def analyse(csv_path: Path, chunk_size: int = CHUNK) -> dict:
    """Stream through the CSV and accumulate brand-level statistics."""
    total_rows     = 0
    inbound_count  = 0
    outbound_count = 0
    brand_stats: dict[str, dict] = {}

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size,
                             dtype={"tweet_id": str, "author_id": str,
                                    "in_response_to_tweet_id": str,
                                    "response_tweet_id": str},
                             low_memory=False):
        total_rows += len(chunk)
        chunk["inbound"] = chunk["inbound"].astype(str).str.strip().str.lower()
        inbound_mask  = chunk["inbound"] == "true"
        outbound_mask = chunk["inbound"] == "false"
        inbound_count  += inbound_mask.sum()
        outbound_count += outbound_mask.sum()

        for author, grp in chunk[outbound_mask].groupby("author_id"):
            if author not in brand_stats:
                brand_stats[author] = {"outbound": 0}
            brand_stats[author]["outbound"] += len(grp)

    return {
        "total_rows":    total_rows,
        "inbound_count": int(inbound_count),
        "outbound_count": int(outbound_count),
        "brand_stats":   brand_stats,
    }


def count_interactions(csv_path: Path, chunk_size: int = CHUNK) -> dict[str, int]:
    """
    Count usable customer->brand interaction pairs per brand.

    Pass 1: build tweet_id -> author_id index for all outbound (brand) tweets.
    Pass 2: for each inbound tweet, check if in_response_to_tweet_id is a brand tweet.
    """
    print("  Pass 1: indexing brand tweet IDs ...")
    brand_tweet_index: dict[str, str] = {}

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size,
                             usecols=["tweet_id", "author_id", "inbound"],
                             dtype=str, low_memory=False):
        chunk["inbound"] = chunk["inbound"].astype(str).str.strip().str.lower()
        outbound = chunk[chunk["inbound"] == "false"]
        for _, row in outbound.iterrows():
            brand_tweet_index[row["tweet_id"]] = row["author_id"]

    print("  Pass 2: counting customer->brand pairs ...")
    interaction_counts: dict[str, int] = {}

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size,
                             usecols=["inbound", "in_response_to_tweet_id"],
                             dtype=str, low_memory=False):
        chunk["inbound"] = chunk["inbound"].astype(str).str.strip().str.lower()
        inbound = chunk[chunk["inbound"] == "true"].dropna(
            subset=["in_response_to_tweet_id"])
        for parent_id in inbound["in_response_to_tweet_id"]:
            brand = brand_tweet_index.get(parent_id)
            if brand:
                interaction_counts[brand] = interaction_counts.get(brand, 0) + 1

    return interaction_counts


def build_report(stats: dict, interactions: dict[str, int]) -> str:
    """Compose the markdown report string."""
    rows = []
    for brand, s in stats["brand_stats"].items():
        ob = s["outbound"]
        if ob < 100:
            continue
        rows.append((brand, ob, interactions.get(brand, 0)))
    rows.sort(key=lambda x: x[2], reverse=True)
    top = rows[:20]

    lines = [
        "# Dataset Analysis Report",
        "",
        "## Overview",
        "",
        f"- **Total rows:** {stats['total_rows']:,}",
        f"- **Inbound (customer) tweets:** {stats['inbound_count']:,}",
        f"- **Outbound (brand) tweets:** {stats['outbound_count']:,}",
        f"- **Unique brand accounts (>=100 outbound tweets):** {len(rows)}",
        "",
        "## Top 20 Brands by Customer->Brand Interactions",
        "",
        "| Rank | Brand Account | Brand Tweets | Customer->Brand Interactions |",
        "|------|--------------|-------------|------------------------------|",
    ]
    for i, (brand, ob, inter) in enumerate(top, 1):
        lines.append(f"| {i} | {brand} | {ob:,} | {inter:,} |")

    lines += [
        "",
        "## Notes",
        "",
        "- *Brand tweets* = outbound tweets authored by the brand account.",
        "- *Customer->Brand Interactions* = inbound tweets that are direct replies",
        "  to a brand tweet (usable customer-message / brand-response pairs).",
        "- Brands with fewer than 100 outbound tweets are excluded from the table.",
        "",
        "## Brand Selection Recommendation",
        "",
        "See `decision_log.md` for the final brand selection and rationale.",
    ]
    return "\n".join(lines)


def main():
    print(f"Dataset: {CSV_PATH}")
    print(f"Chunk size: {CHUNK:,} rows\n")

    print("Streaming pass - collecting brand statistics ...")
    stats = analyse(CSV_PATH)
    print(f"  Total rows:      {stats['total_rows']:,}")
    print(f"  Inbound tweets:  {stats['inbound_count']:,}")
    print(f"  Outbound tweets: {stats['outbound_count']:,}")

    print("\nCounting customer->brand interaction pairs ...")
    interactions = count_interactions(CSV_PATH)

    top15 = sorted(interactions.items(), key=lambda x: x[1], reverse=True)[:15]
    print("\nTop 15 brands by interaction count:")
    print(f"  {'Brand':<25} {'Interactions':>12}  {'Brand tweets':>12}")
    for brand, inter in top15:
        ob = stats["brand_stats"].get(brand, {}).get("outbound", 0)
        print(f"  {brand:<25} {inter:>12,}  {ob:>12,}")

    report_text = build_report(stats, interactions)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved -> {REPORT}")


if __name__ == "__main__":
    main()
