"""Simple terminal-based reviewer for manual labels on the golden set.

This workflow matches the original Hiver requirement: a human manually labels the
golden set without any web UI or review service. The script saves work after each
example and can be rerun to resume or revisit previously labeled rows.
"""

from pathlib import Path
import sys
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "evaluation" / "golden_set.csv"
OUTPUT_CSV = ROOT / "evaluation" / "golden_human_labels.csv"

INTENTS = [
    "playback_issue",
    "app_bug",
    "account_login",
    "premium_billing",
    "download_offline",
    "content_search",
    "general_inquiry",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual human review for the golden set.")
    parser.add_argument("--start-at", type=int, default=0, help="Row index to begin reviewing at.")
    parser.add_argument("--only-unlabeled", action="store_true", help="Review only rows missing human_intent/should_escalate.")
    parser.add_argument("--example-id", help="Review a single example by example_id value.")
    return parser.parse_args()


def load_working_copy() -> pd.DataFrame:
    source = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    if OUTPUT_CSV.exists():
        reviewed = pd.read_csv(OUTPUT_CSV, dtype=str).fillna("")
        if "example_id" in reviewed.columns:
            source = source.drop(columns=[c for c in ["human_intent", "should_escalate", "labeling_notes"] if c in source])
            source = source.merge(
                reviewed[["example_id", "human_intent", "should_escalate", "labeling_notes"]],
                on="example_id", how="left",
            )
            source[["human_intent", "should_escalate", "labeling_notes"]] = source[
                ["human_intent", "should_escalate", "labeling_notes"]
            ].fillna("")
    return source


def save(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")


def choose_intent(current: str) -> str:
    print("Choose human intent:")
    for number, intent in enumerate(INTENTS, start=1):
        print(f"{number}. {intent}")
    while True:
        choice = input(f"Intent [1-{len(INTENTS)}] (current: {current or 'blank'}): ").strip()
        if choice == "" and current in INTENTS:
            return current
        if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            return INTENTS[int(choice) - 1]
        print("Please enter a valid intent number.")


def choose_ambiguous(current: str) -> str:
    while True:
        choice = input(f"Ambiguous? 1. YES  2. NO (current: {current or 'blank'}): ").strip().upper()
        if choice == "" and current in {"TRUE", "FALSE"}:
            return current
        if choice in {"1", "YES", "TRUE"}:
            return "True"
        if choice in {"2", "NO", "FALSE"}:
            return "False"
        print("Please enter 1/YES or 2/NO.")


def review_row(row: pd.Series, index: int, df: pd.DataFrame) -> None:
    print("=" * 72)
    print(f"Example ID: {row['example_id']} | Row {index + 1}/{len(df)}")
    print(f"Customer message: {row['customer_message']}")
    print(f"Brand response: {row['brand_response']}")
    print(f"Suggested intent (reference only): {row['suggested_intent']}")

    current_intent = str(row.get("human_intent", "")).strip()
    current_escalation = str(row.get("should_escalate", "")).strip()
    current_ambiguous = str(row.get("ambiguous", "")).strip()
    current_notes = str(row.get("labeling_notes", "")).strip()

    human_intent = choose_intent(current_intent)
    escalation = choose_escalation(current_escalation)
    ambiguous = choose_ambiguous(current_ambiguous)
    notes = input(f"Notes (optional; current: {current_notes or 'blank'}): ").strip()
    if notes == "" and current_notes:
        notes = current_notes

    df.at[index, "human_intent"] = human_intent
    df.at[index, "should_escalate"] = escalation
    df.at[index, "ambiguous"] = ambiguous
    df.at[index, "labeling_notes"] = notes
    save(df)
    print(f"Saved {row['example_id']}.")


def choose_escalation(current: str) -> str:
    while True:
        choice = input(f"Should escalate? 1. YES  2. NO (current: {current or 'blank'}): ").strip().upper()
        if choice == "" and current in {"YES", "NO"}:
            return current
        if choice in {"1", "YES"}:
            return "YES"
        if choice in {"2", "NO"}:
            return "NO"
        print("Please enter 1/YES or 2/NO.")


def main() -> None:
    args = parse_args()
    if not INPUT_CSV.exists():
        print(f"Missing input file: {INPUT_CSV}")
        sys.exit(1)
    df = load_working_copy()
    if args.example_id:
        matches = df[df["example_id"].astype(str).str.lower() == args.example_id.lower()]
        if matches.empty:
            print(f"No example found with example_id={args.example_id}")
            return
        index = int(matches.index[0])
        review_row(df.loc[index], index, df)
        return

    if args.only_unlabeled:
        unlabeled = df[
            df["human_intent"].fillna("").str.strip().eq("")
            | df["should_escalate"].fillna("").str.strip().eq("")
        ].index.tolist()
        if not unlabeled:
            print("All rows already have human_intent and should_escalate values.")
            return
        df = df.loc[unlabeled]

    start_index = max(0, min(args.start_at, len(df) - 1)) if len(df) else 0
    save(df)

    print(f"Reviewing {len(df)} examples. Progress is saved to {OUTPUT_CSV} after each example.")
    print("Suggested intent is shown for reference only. You must choose the human label yourself.")
    print("Press Ctrl+C to stop safely; rerun to resume from saved values.\n")
    try:
        for index in range(start_index, len(df)):
            row = df.iloc[index]
            review_row(row, index, df)
            print()
    except (KeyboardInterrupt, EOFError):
        save(df)
        print(f"\nReview stopped safely. Saved progress to {OUTPUT_CSV}")
        return

    print(f"Review complete. Saved labels to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
