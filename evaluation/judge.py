"""
judge.py

LLM-as-judge for evaluating generated responses.

Scores each response on 5 dimensions (1-5 scale):
  1. Correctness   - Is the response factually consistent with the evidence?
  2. Groundedness  - Is every claim supported by retrieved evidence?
  3. Relevance     - Does the response address the customer's actual question?
  4. Helpfulness   - Does the response give actionable next steps?
  5. Tone          - Is the response professional and empathetic?

Implementation:
    - RuleBasedJudge: fast, local, deterministic heuristics. Used by default.

The rule-based judge is the current local judge because:
  - torch DLLs are blocked on this machine.
  - Rule-based scoring is transparent and explainable.
    - An LLM judge and human-vs-LLM agreement are not claimed until they are run.
"""

from pathlib import Path
from dataclasses import dataclass, asdict
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))


@dataclass
class JudgeScore:
    correctness:  int   # 1-5
    groundedness: int   # 1-5
    relevance:    int   # 1-5
    helpfulness:  int   # 1-5
    tone:         int   # 1-5

    @property
    def average(self) -> float:
        return (self.correctness + self.groundedness +
                self.relevance + self.helpfulness + self.tone) / 5.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["average"] = round(self.average, 2)
        return d


# Signals used by the rule-based judge
_HALLUCINATION = [
    r"\brefund\b", r"\bcredit\b", r"\bguarantee\b",
    r"\bwithin \d+ (hour|day)", r"\bwe will fix\b",
    r"\byour account has been\b",
]
_ACTIONABLE = [
    r"\btry\b", r"\brestart\b", r"\reinstall\b", r"\bupdate\b",
    r"\bsend\b", r"\bdm\b", r"\bcontact\b", r"\bcheck\b",
    r"\bgo to\b", r"\bopen\b", r"\bclick\b", r"\bvisit\b",
]
_EMPATHY = [
    r"\bsorry\b", r"\bapologi\b", r"\bunderstand\b", r"\bhear you\b",
    r"\bthanks for\b", r"\bthank you\b", r"\bwe.re here\b",
]


def _token_overlap_ratio(a: str, b: str) -> float:
    stop = {"i", "a", "the", "to", "is", "it", "and", "in", "of", "you",
            "we", "can", "for", "on", "my", "your"}
    ta = {w for w in re.sub(r"[^a-z0-9 ]", " ", a.lower()).split()
          if w not in stop and len(w) > 2}
    tb = {w for w in re.sub(r"[^a-z0-9 ]", " ", b.lower()).split()
          if w not in stop and len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class RuleBasedJudge:
    """
    Deterministic heuristic judge. Fast and explainable.
    Scores are approximate proxies — not equivalent to human judgement.
    """

    def score(self, customer_message: str,
              retrieved_evidence: list[dict],
              response: str) -> JudgeScore:

        resp_lower = response.lower()
        evidence_text = " ".join(e.get("brand_response", "")
                                 for e in retrieved_evidence)

        # Correctness: penalise hallucination signals
        hallucinations = sum(1 for p in _HALLUCINATION
                             if re.search(p, resp_lower))
        correctness = max(1, 5 - hallucinations * 2)

        # Groundedness: overlap between response and evidence
        overlap = _token_overlap_ratio(response, evidence_text)
        if overlap >= 0.30:
            groundedness = 5
        elif overlap >= 0.20:
            groundedness = 4
        elif overlap >= 0.10:
            groundedness = 3
        elif overlap >= 0.05:
            groundedness = 2
        else:
            groundedness = 1

        # Relevance: overlap between response and customer message
        rel_overlap = _token_overlap_ratio(response, customer_message)
        if rel_overlap >= 0.20:
            relevance = 5
        elif rel_overlap >= 0.10:
            relevance = 4
        elif rel_overlap >= 0.05:
            relevance = 3
        elif len(response) > 30:
            relevance = 2
        else:
            relevance = 1

        # Helpfulness: contains actionable phrases
        action_hits = sum(1 for p in _ACTIONABLE
                          if re.search(p, resp_lower))
        helpfulness = min(5, 2 + action_hits)

        # Tone: empathy signals + not too short
        empathy_hits = sum(1 for p in _EMPATHY if re.search(p, resp_lower))
        tone = min(5, 3 + empathy_hits) if len(response) > 20 else 2

        return JudgeScore(
            correctness=correctness,
            groundedness=groundedness,
            relevance=relevance,
            helpfulness=helpfulness,
            tone=tone,
        )


def run_judge_on_eval_results(eval_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Score all AUTO_HANDLE responses in eval_results.csv."""
    SRC_PATH = ROOT / "src"
    sys.path.insert(0, str(SRC_PATH))
    from retrieval import RetrievalIndex
    import joblib

    INDEX_PATH = ROOT / "data" / "processed" / "retrieval_index.joblib"
    index = RetrievalIndex.load(INDEX_PATH)
    judge = RuleBasedJudge()

    df = pd.read_csv(eval_csv, dtype=str)
    auto = df[df["route"] == "AUTO_HANDLE"].copy()
    print(f"Scoring {len(auto)} AUTO_HANDLE responses ...")

    scored_rows = []
    for _, row in auto.iterrows():
        msg      = str(row["customer_message"]) if "customer_message" in row else ""
        response = str(row["response"])
        intent   = str(row["pred_intent"])
        evidence = index.retrieve(msg, intent, top_k=3) if msg else []
        score    = judge.score(msg, evidence, response)
        scored_rows.append({
            "example_id":   row.get("example_id", ""),
            "intent":       intent,
            "response":     response[:120],
            **score.to_dict(),
        })

    scored_df = pd.DataFrame(scored_rows)
    scored_df.to_csv(output_csv, index=False)

    print(f"\nJudge scores (n={len(scored_df)}):")
    for col in ["correctness", "groundedness", "relevance", "helpfulness", "tone", "average"]:
        print(f"  {col:<15} {scored_df[col].astype(float).mean():.2f}")

    return scored_df


if __name__ == "__main__":
    EVAL_CSV   = ROOT / "evaluation" / "eval_results.csv"
    OUTPUT_CSV = ROOT / "evaluation" / "judge_scores.csv"

    if not EVAL_CSV.exists():
        print(f"Run evaluation/evaluate.py first to generate {EVAL_CSV}")
        sys.exit(1)

    # Need customer_message in eval results — merge with golden set
    golden = pd.read_csv(ROOT / "evaluation" / "golden_set.csv", dtype=str)
    eval_df = pd.read_csv(EVAL_CSV, dtype=str)
    merged = eval_df.merge(
        golden[["example_id", "customer_message"]], on="example_id", how="left"
    )
    merged.to_csv(EVAL_CSV, index=False)

    run_judge_on_eval_results(EVAL_CSV, OUTPUT_CSV)
    print(f"\nJudge scores saved -> {OUTPUT_CSV}")
