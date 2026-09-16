"""
utils.py

Shared utilities: data loading, train/val/test splitting, metric printing.
All classifiers import from here to ensure consistent splits.
"""

from pathlib import Path
import pandas as pd
import re
from sklearn.model_selection import train_test_split

ROOT         = Path(__file__).resolve().parent.parent
INTERACTIONS = ROOT / "data" / "processed" / "interactions.csv"
GOLDEN_CSV   = ROOT / "evaluation" / "golden_set.csv"
INTENTS_FILE = ROOT / "data" / "intents.json"

# Canonical intent list (order is stable across all scripts)
# 'other' is the last entry: messages that don't fit any support intent.
# The classifier does not predict 'other' (no training examples).
# 'other' is assigned by the router when a message is flagged as out-of-scope,
# or can be assigned manually during golden set human review.
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

# Classifier-only intents: excludes 'other' since the classifier has no training
# examples for it and never predicts it. Use this for classification reports.
CLASSIFIER_INTENTS = [
    "playback_issue",
    "app_bug",
    "account_login",
    "premium_billing",
    "download_offline",
    "content_search",
    "general_inquiry",
]

# Same patterns used in build_golden_set.py — single source of truth
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


def assign_intent(text: str) -> str:
    """Assign intent label using keyword rules. Returns first match or 'general_inquiry'."""
    text_lower = str(text).lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, text_lower):
            return intent
    return "general_inquiry"


def matched_intents(text: str) -> list[str]:
    """Return all intents whose rules match, preserving the configured order."""
    text_lower = str(text).lower()
    return [
        intent for intent, pattern in INTENT_PATTERNS.items()
        if re.search(pattern, text_lower)
    ]


def is_fallback_general_inquiry(text: str) -> bool:
    """Return whether general_inquiry is assigned only because no rule matched."""
    return not matched_intents(text)


def load_labelled_interactions(min_length: int = 15) -> pd.DataFrame:
    """
    Load interactions.csv, assign intent labels, drop very short messages.
    Returns DataFrame with columns: customer_message, intent, interaction_id.
    """
    df = pd.read_csv(INTERACTIONS, dtype=str)
    df = df[df["customer_message"].str.len() >= min_length].copy()
    df["intent"] = df["customer_message"].apply(assign_intent)
    return df


def load_splits(val_size: float = 0.1, test_size: float = 0.1,
                random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df).

    Leakage prevention:
      - Golden set examples are identified by customer_tweet_id and excluded
        from all training/validation/test splits.
      - The golden set is the ONLY evaluation set used for final reporting.
    """
    golden = pd.read_csv(GOLDEN_CSV, dtype=str)
    golden_ids = set(golden["customer_tweet_id"].tolist())

    df = load_labelled_interactions()
    # Exclude golden set examples
    df = df[~df["customer_tweet_id"].isin(golden_ids)].copy()

    # First split off test
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["intent"], random_state=random_state
    )
    # Then split val from train
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["intent"],
        random_state=random_state
    )
    return train, val, test


def print_metrics(report: str, split_name: str = "test") -> None:
    print(f"\n--- Classification Report ({split_name}) ---")
    print(report)
