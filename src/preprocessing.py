"""
preprocessing.py

Streams twcs.csv and writes two compact files for SpotifyCares:
  data/processed/spotify_raw.csv   - all rows where author_id == BRAND
                                     or the tweet is a direct reply to a BRAND tweet
  data/processed/spotify_ids.txt   - all tweet_ids belonging to SpotifyCares threads
                                     (used by conversation_builder.py)

Run once; subsequent phases read from data/processed/ only.
"""

from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
CSV_PATH  = ROOT / "archive" / "twcs" / "twcs.csv"
OUT_DIR   = ROOT / "data" / "processed"
BRAND     = "SpotifyCares"
CHUNK     = 100_000


def extract_brand_data(csv_path: Path, brand: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / "spotify_raw.csv"

    # Pass 1: collect all tweet_ids authored by the brand
    print("Pass 1: collecting brand tweet IDs ...")
    brand_tweet_ids: set[str] = set()

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK, dtype=str, low_memory=False):
        mask = chunk["author_id"].str.strip() == brand
        brand_tweet_ids.update(chunk.loc[mask, "tweet_id"].tolist())

    print(f"  Brand tweet IDs found: {len(brand_tweet_ids):,}")

    # Pass 2: keep rows that are brand tweets OR direct customer replies to brand tweets
    print("Pass 2: extracting relevant rows ...")
    kept_chunks = []
    total_kept = 0

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK, dtype=str, low_memory=False):
        is_brand_tweet    = chunk["author_id"].str.strip() == brand
        is_customer_reply = chunk["in_response_to_tweet_id"].isin(brand_tweet_ids)
        mask = is_brand_tweet | is_customer_reply
        filtered = chunk[mask].copy()
        if not filtered.empty:
            kept_chunks.append(filtered)
            total_kept += len(filtered)

    df = pd.concat(kept_chunks, ignore_index=True)
    df.to_csv(raw_out, index=False)
    print(f"  Rows saved to {raw_out}: {total_kept:,}")


if __name__ == "__main__":
    extract_brand_data(CSV_PATH, BRAND, OUT_DIR)
    print("Preprocessing complete.")
