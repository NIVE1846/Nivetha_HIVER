"""Create an unlabeled pool of rule-unmatched examples for human `other` review.

This never assigns `other`. The rule system's fallback remains `general_inquiry`;
human reviewers decide whether a candidate genuinely belongs to `other`.
"""

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERACTIONS = ROOT / "data" / "processed" / "interactions.csv"
OUTPUT = ROOT / "evaluation" / "other_candidate_pool.csv"
RANDOM_STATE = 42
SAMPLE_SIZE = 30

PATTERNS = {
    "playback_issue": r"not play|won.t play|stop.?play|keeps? stop|skip|shuffle|playback|buffer|freez|audio.?cut|song.?end|repeat|loop|gapless|crossfade|plays? in order|plays? random",
    "app_bug": r"app.?crash|crash|force.?clos|won.t open|won.t load|blank.?screen|black.?screen|error.?code|error \d|glitch|broken|not.?work|doesn.t work|after.?update|since.?update|latest.?update|new.?update",
    "account_login": r"log.?in|login|sign.?in|sign.?out|password|forgot|reset.?pass|locked.?out|can.t.?access|account.?access|facebook.?login|google.?login|wrong.?email|change.?email|username",
    "premium_billing": r"premium|paid|payment|charge|bill|subscri|cancel|refund|free.?trial|credit.?card|invoice|receipt|student.?plan|family.?plan|duo.?plan|upgrade|downgrade",
    "download_offline": r"download|offline|sync|saved.?song|saved.?playlist|storage|download.?limit|won.t.?sync|disappear.?download|remove.?download|download.?gone",
    "content_search": r"can.t.?find|not.?find|missing.?song|missing.?album|not.?show|not.?appear|search.?result|playlist.?gone|playlist.?missing|playlist.?disappear|album.?missing|artist.?missing|not.?available|not.?on.?spotify|library.?missing|recently.?added",
    "general_inquiry": r"feature|suggest|wish|would.?be.?nice|please.?add|option.?to|ability.?to|available.?in|compatible|how.?do.?i|how.?can.?i|is.?it.?possible|can.?you.?add|love.?the|great.?app|thank",
}


def is_unmatched(text: str) -> bool:
    lowered = str(text).lower()
    return not any(re.search(pattern, lowered) for pattern in PATTERNS.values())


def main() -> None:
    interactions = pd.read_csv(INTERACTIONS, dtype=str).fillna("")
    candidates = interactions[
        (interactions["customer_message"].str.len() >= 1)
        & interactions["customer_message"].map(is_unmatched)
    ].copy()
    sample = candidates.sample(n=min(SAMPLE_SIZE, len(candidates)), random_state=RANDOM_STATE)
    output = sample[["customer_message", "brand_response", "interaction_id"]].copy()
    output.insert(0, "candidate_id", [f"OTHER_{i:04d}" for i in range(len(output))])
    output["suggested_intent"] = "general_inquiry"
    output["human_intent"] = ""
    output["should_escalate"] = ""
    output["labeling_notes"] = "candidate selected by no-rule-match; human decision required"
    output.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Wrote {len(output)} unlabeled candidates to {OUTPUT}")
    print("No candidate was assigned the human `other` label.")


if __name__ == "__main__":
    main()
