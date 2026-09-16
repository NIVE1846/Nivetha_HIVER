"""Generate isolated V2 versus V2.1 coverage audit artifacts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random

import pandas as pd

from audit_rule_v2 import INPUT
from rule_v2 import label_v2
from rule_v2_1 import RULE_VERSION, label_v2_1

ROOT = Path(__file__).resolve().parent.parent
LABELS_OUT = ROOT / "data" / "processed" / "rule_v2_1_labels.csv"
REPORT_OUT = ROOT / "reports" / "rule_v2_1_audit.md"
SEED = 42


def audit_frame() -> pd.DataFrame:
    source = pd.read_csv(INPUT, dtype=str).fillna("")
    source = source[source["customer_message"].str.len() >= 15].copy()
    v2 = source["customer_message"].map(label_v2).apply(pd.Series).add_prefix("v2_")
    v21 = source["customer_message"].map(label_v2_1).apply(pd.Series).add_prefix("v2_1_")
    return pd.concat([source, v2, v21], axis=1)


def _msg(row: pd.Series) -> str:
    return str(row["customer_message"]).replace("\n", " ").strip()


def _representatives(df: pd.DataFrame, rule_key: str, n: int = 10) -> list[pd.Series]:
    mask = df["v2_1_matched_rules"].map(lambda value: rule_key in value)
    sample = df.loc[mask]
    if sample.empty:
        return []
    return [row for _, row in sample.sample(n=min(n, len(sample)), random_state=SEED).iterrows()]


def build_report(df: pd.DataFrame) -> str:
    v2 = df["v2_intent"].fillna("<unresolved>")
    v21 = df["v2_1_intent"].fillna("<unresolved>")
    transitions = Counter(zip(v2, v21))
    overlap_count = int(df["v2_1_matched_rules"].map(len).gt(1).sum())
    newly_resolved = int(((v2 == "<unresolved>") & (v21 != "<unresolved>")).sum())
    changed_concrete = int(
        ((v2 != "<unresolved>") & (v21 != "<unresolved>") & (v2 != v21)).sum()
    )
    rule_keys = [
        key
        for rules in (
            "playback_issue.audio_behavior_paraphrase",
            "app_bug.app_state_paraphrase",
            "app_bug.screen_state_paraphrase",
            "account_login.access_paraphrase",
            "account_login.sign_in_paraphrase",
            "content_search.listen_availability_paraphrase",
            "content_search.catalog_availability_paraphrase",
            "download_offline.saved_content_sync",
            "download_offline.offline_content_paraphrase",
            "premium_billing.plan_eligibility",
            "premium_billing.subscription_access",
            "general_inquiry.explicit_information_question",
            "general_inquiry.feature_how_to_paraphrase",
        )
        for key in (rules,)
    ]
    lines = [
        "# Rule V2.1 audit",
        "",
        f"- Rule version: `{RULE_VERSION}`",
        f"- Source: `{INPUT}`",
        f"- Audit rows: **{len(df):,}**",
        f"- Sampling seed: `{SEED}`",
        "",
        "## V2 versus V2.1 counts",
        "",
        "| Version | Resolved | Unresolved | Unresolved % |",
        "|---|---:|---:|---:|",
        f"| V2 | {(v2 != '<unresolved>').sum()} | {(v2 == '<unresolved>').sum()} | {(v2 == '<unresolved>').mean():.2%} |",
        f"| V2.1 | {(v21 != '<unresolved>').sum()} | {(v21 == '<unresolved>').sum()} | {(v21 == '<unresolved>').mean():.2%} |",
        "",
        f"- Newly resolved rows: **{newly_resolved}**",
        f"- Rows changing between concrete intents: **{changed_concrete}**",
        f"- Rows matching multiple V2.1 rules: **{overlap_count}**",
        "",
        "## V2 to V2.1 transitions",
        "",
        "| V2 intent | V2.1 intent | Count |",
        "|---|---|---:|",
    ]
    lines.extend(f"| {a} | {b} | {n} |" for (a, b), n in sorted(transitions.items()))
    lines.extend(["", "## Newly introduced rule-family coverage", "", "| Rule key | Rows |", "|---|---:|"])
    for key in rule_keys:
        count = int(df["v2_1_matched_rules"].map(lambda value: key in value).sum())
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Representative newly resolved examples", ""])
    for key in rule_keys:
        rows = _representatives(df, key)
        lines.extend([f"### `{key}`", ""])
        if not rows:
            lines.append("No examples.")
        for row in rows:
            lines.append(f"- `{_msg(row)}`")
        lines.append("")
    lines.extend([
        "## Potentially dangerous false-positive cases",
        "",
        "- A message mentioning a song or track without a missing/availability relation must remain unresolved.",
        "- Premium, subscription, plan, and card references require an eligibility, payment, renewal, cancellation, or access relationship.",
        "- Download wording must concern saved Spotify content or offline listening, not app/update/image downloads.",
        "- Generic account, device, or “not working” wording remains unresolved without an authentication or malfunction action.",
        "- Feedback, thanks, and context-free follow-ups are not general inquiries unless they contain an explicit information or feature question.",
        "",
        "## Remaining unresolved patterns",
        "",
        "The remaining unresolved pool is intentionally retained for context-free replies, generic complaints, device/version metadata, third-party/off-topic text, and messages lacking a safe intent-specific evidence combination.",
        "",
        "## Rule configuration",
        "",
        "- V2.1 reuses V2 unchanged and evaluates additional rules only for V2-unresolved rows.",
        "- No generic single-word trigger was added.",
        "- V2.1 preserves matched-rule metadata and marks every output `rule_v2_1`.",
        "- No human labels, model artifacts, or evaluation files are used.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    df = audit_frame()
    output = df.copy()
    for column in ("v2_matched_rules", "v2_1_matched_rules"):
        output[column] = output[column].map(json.dumps)
    output.to_csv(LABELS_OUT, index=False, encoding="utf-8")
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(build_report(df), encoding="utf-8")
    v2 = df["v2_intent"].fillna("<unresolved>")
    v21 = df["v2_1_intent"].fillna("<unresolved>")
    print(f"V2 resolved: {(v2 != '<unresolved>').sum()}")
    print(f"V2.1 resolved: {(v21 != '<unresolved>').sum()}")
    print(f"Newly resolved: {((v2 == '<unresolved>') & (v21 != '<unresolved>')).sum()}")
    print(f"V2.1 unresolved: {(v21 == '<unresolved>').sum()} ({(v21 == '<unresolved>').mean():.2%})")
    print(f"V2.1 overlaps: {int(df['v2_1_matched_rules'].map(len).gt(1).sum())}")
    print(f"Wrote {LABELS_OUT}")
    print(f"Wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
