"""
build_golden_set.py

Samples ~200 stratified examples from interactions.csv and assigns
suggested intent labels using keyword rules derived from intents.json analysis.

Sampling strategy:
  - Each of the 7 intents gets ~28-29 examples (200 / 7, rounded).
  - Within each intent, examples are sampled randomly (random_state=42).
  - Ambiguous examples (matching multiple intents) are flagged.
  - Examples with very short messages (<15 chars after cleaning) are excluded.

Output: evaluation/golden_set.csv
        evaluation/golden_set_build_log.txt

IMPORTANT: This golden set must NOT be used for training.
           Suggested labels are not human ground truth. Human labels and
           escalation decisions are collected separately after this build.
"""

from pathlib import Path
import pandas as pd
import re
import json

ROOT         = Path(__file__).resolve().parent.parent
INTERACTIONS = ROOT / "data" / "processed" / "interactions.csv"
INTENTS_JSON = ROOT / "data" / "intents.json"
OUT_CSV      = ROOT / "evaluation" / "golden_set.csv"
DOUBLE_LABEL_CSV = ROOT / "evaluation" / "double_label_sample.csv"
LOG_PATH     = ROOT / "evaluation" / "golden_set_build_log.txt"

SAMPLES_PER_INTENT = 29   # 7 * 29 = 203, close to 200 target
RANDOM_STATE       = 42
DOUBLE_LABEL_SIZE  = 50

# Keyword rules for each intent (order matters: first match wins for primary label)
# These are intentionally conservative to reduce noise in the golden set.
INTENT_PATTERNS: dict[str, str] = {
    "playback_issue": (
        r"not play|won.t play|stop.?play|keeps? stop|skip|shuffle|"
        r"playback|buffer|freez|audio.?cut|song.?end|repeat|loop|"
        r"gapless|crossfade|plays? in order|plays? random"
    ),
    "app_bug": (
        r"app.?crash|crash|force.?clos|won.t open|won.t load|"
        r"blank.?screen|black.?screen|error.?code|error \d|"
        r"glitch|broken|not.?work|doesn.t work|after.?update|"
        r"since.?update|latest.?update|new.?update"
    ),
    "account_login": (
        r"log.?in|login|sign.?in|sign.?out|password|forgot|reset.?pass|"
        r"locked.?out|can.t.?access|account.?access|facebook.?login|"
        r"google.?login|wrong.?email|change.?email|username"
    ),
    "premium_billing": (
        r"premium|paid|payment|charge|bill|subscri|cancel|refund|"
        r"free.?trial|credit.?card|invoice|receipt|student.?plan|"
        r"family.?plan|duo.?plan|upgrade|downgrade"
    ),
    "download_offline": (
        r"download|offline|sync|saved.?song|saved.?playlist|"
        r"storage|download.?limit|won.t.?sync|disappear.?download|"
        r"remove.?download|download.?gone"
    ),
    "content_search": (
        r"can.t.?find|not.?find|missing.?song|missing.?album|"
        r"not.?show|not.?appear|search.?result|playlist.?gone|"
        r"playlist.?missing|playlist.?disappear|album.?missing|"
        r"artist.?missing|not.?available|not.?on.?spotify|"
        r"library.?missing|recently.?added"
    ),
    "general_inquiry": (
        r"feature|suggest|wish|would.?be.?nice|please.?add|"
        r"option.?to|ability.?to|available.?in|compatible|"
        r"how.?do.?i|how.?can.?i|is.?it.?possible|can.?you.?add|"
        r"love.?the|great.?app|thank"
    ),
}


def assign_intent(text: str) -> tuple[str, bool]:
    """
    Returns (intent_label, is_ambiguous).
    is_ambiguous=True if more than one pattern matches.
    """
    text_lower = str(text).lower()
    matches = [
        intent for intent, pattern in INTENT_PATTERNS.items()
        if re.search(pattern, text_lower)
    ]
    if not matches:
        return "general_inquiry", False   # fallback
    if len(matches) == 1:
        return matches[0], False
    # Multiple matches: pick the first (priority order in INTENT_PATTERNS)
    return matches[0], True


def build_golden_set(interactions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(interactions_path, dtype=str)

    # Drop very short messages
    df = df[df["customer_message"].str.len() >= 15].copy()

    # Assign non-human suggested labels
    df[["suggested_intent", "ambiguous"]] = df["customer_message"].apply(
        lambda m: pd.Series(assign_intent(m))
    )

    # Stratified sample: SAMPLES_PER_INTENT per intent
    sampled_parts = []
    for intent in INTENT_PATTERNS:
        subset = df[df["suggested_intent"] == intent]
        n = min(SAMPLES_PER_INTENT, len(subset))
        sampled_parts.append(subset.sample(n, random_state=RANDOM_STATE))

    golden = pd.concat(sampled_parts, ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE  # shuffle final order
    ).reset_index(drop=True)

    golden["example_id"] = [f"GS_{i:04d}" for i in range(len(golden))]

    # Human review fields must remain blank until an annotator fills them.
    golden["human_intent"] = ""
    golden["should_escalate"] = ""

    golden["labeling_notes"] = golden["ambiguous"].map(
        {True: "ambiguous - matches multiple intent patterns", False: ""}
    )

    # Select and order output columns
    out = golden[[
        "example_id",
        "customer_message",
        "brand_response",
        "suggested_intent",
        "human_intent",
        "should_escalate",
        "ambiguous",
        "labeling_notes",
        "customer_tweet_id",
        "brand_tweet_id",
    ]]
    return out


def build_double_label_sample(golden: pd.DataFrame) -> pd.DataFrame:
    """Select a reproducible, unlabeled sample for two independent annotators."""
    sample_size = min(DOUBLE_LABEL_SIZE, len(golden))
    sample = golden.sample(n=sample_size, random_state=RANDOM_STATE).copy()
    return sample[["example_id", "customer_message", "suggested_intent"]].assign(
        annotator_A_intent="",
        annotator_B_intent="",
        annotator_A_should_escalate="",
        annotator_B_should_escalate="",
        notes="",
    )


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Building golden evaluation set ...")
    golden = build_golden_set(INTERACTIONS)
    golden.to_csv(OUT_CSV, index=False, encoding="utf-8")
    double_label = build_double_label_sample(golden)
    double_label.to_csv(DOUBLE_LABEL_CSV, index=False, encoding="utf-8")

    # Stats
    intent_counts = golden["suggested_intent"].value_counts()
    ambiguous_count = golden["ambiguous"].sum()

    log_lines = [
        "Golden Set Build Log",
        "====================",
        f"Total examples:        {len(golden)}",
        f"Ambiguous examples:    {ambiguous_count}",
        "",
        "Sampling method:",
        "  - Keyword-rule suggested labels using conservative regex patterns.",
        "  - Stratified: up to 29 examples per intent (7 intents = ~203 total).",
        "  - Messages shorter than 15 characters excluded.",
        "  - Random state = 42 for reproducibility.",
        "  - Final set shuffled before saving.",
        "",
        "Intent distribution:",
    ]
    for intent, count in intent_counts.items():
        log_lines.append(f"  {intent:<25} {count}")

    log_lines += [
        "",
        "Leakage prevention:",
        "  - This file must NOT be used for classifier training.",
        "  - Training data is drawn from the remaining interactions.csv rows.",
        "  - The golden set tweet IDs are saved so they can be excluded from",
        "    the training split in later phases.",
        "",
        "Suggested label assignment:",
        "  - Primary label = first matching intent pattern (priority order).",
        "  - Ambiguous flag set when >1 pattern matches.",
        "  - Unmatched messages fall back to general_inquiry in the suggestion logic.",
        "  - human_intent and should_escalate are intentionally blank.",
        "  - All rows must be manually reviewed before final evaluation.",
        "",
        "Double-label sample:",
        f"  - {len(double_label)} examples sampled with random state {RANDOM_STATE}.",
        "  - Annotator A and B fields are intentionally blank.",
        "  - suggested_intent is included for reference only.",
    ]
    LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Golden set saved -> {OUT_CSV}")
    print(f"Double-label sample saved -> {DOUBLE_LABEL_CSV}")
    print(f"Log saved        -> {LOG_PATH}")
    print(f"\nTotal examples: {len(golden)}")
    print("\nIntent distribution:")
    print(intent_counts.to_string())
    print(f"\nAmbiguous examples: {ambiguous_count}")


if __name__ == "__main__":
    main()
