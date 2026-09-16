"""Generate a read-only audit of Rule V2 over original processed interactions."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random

import pandas as pd

from rule_v2 import RULE_VERSION, label_v2
from utils import assign_intent, matched_intents

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "interactions.csv"
LABELS_OUT = ROOT / "data" / "processed" / "rule_v2_labels.csv"
REPORT_OUT = ROOT / "reports" / "rule_v2_audit.md"
SAMPLE_SEED = 42
SAMPLE_SIZE = 20


def audit_frame() -> pd.DataFrame:
    source = pd.read_csv(INPUT, dtype=str).fillna("")
    source = source[source["customer_message"].str.len() >= 15].copy()
    v2 = source["customer_message"].map(label_v2).apply(pd.Series)
    source["v1_intent"] = source["customer_message"].map(assign_intent)
    source["v1_matched_rules"] = source["customer_message"].map(
        lambda value: json.dumps(matched_intents(value))
    )
    source = pd.concat([source, v2], axis=1)
    source["matched_rules"] = source["matched_rules"].map(json.dumps)
    source["v2_intent"] = source["intent"]
    source = source.drop(columns=["intent"])
    return source


def _examples(df: pd.DataFrame, mask: pd.Series, n: int = SAMPLE_SIZE) -> list[dict]:
    sample = df.loc[mask].sample(n=min(n, int(mask.sum())), random_state=SAMPLE_SEED)
    return sample.to_dict("records")


def _message(row: dict) -> str:
    return str(row.get("customer_message", "")).replace("\n", " ").strip()


def build_report(df: pd.DataFrame) -> str:
    intents = [
        "playback_issue", "app_bug", "account_login", "premium_billing",
        "download_offline", "content_search", "general_inquiry",
    ]
    v1_counts = df["v1_intent"].value_counts().to_dict()
    v2_counts = df["v2_intent"].fillna("<unresolved>").value_counts().to_dict()
    transitions = Counter(
        (v1, v2 if pd.notna(v2) else "<unresolved>")
        for v1, v2 in zip(df["v1_intent"], df["v2_intent"])
    )
    overlaps = Counter(
        tuple(json.loads(value))
        for value in df["matched_rules"]
        if len(json.loads(value)) > 1
    )
    lines = [
        "# Rule V2 labeling audit",
        "",
        f"- Rule version: `{RULE_VERSION}`",
        f"- Source: `{INPUT}`",
        f"- Minimum message length: `15`",
        f"- Sampling seed: `{SAMPLE_SEED}`",
        f"- Audit rows: `{len(df)}`",
        "",
        "## V1 class counts",
        "",
        "| Intent | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {k} | {v} |" for k, v in sorted(v1_counts.items()))
    lines.extend(["", "## V2 class counts", "", "| Intent | Count |", "|---|---:|"])
    lines.extend(f"| {k} | {v} |" for k, v in sorted(v2_counts.items()))
    unresolved = int(df["v2_intent"].isna().sum())
    lines.extend([
        "",
        f"Unresolved count: **{unresolved}** ({unresolved / len(df):.2%})",
        "",
        "## V1 to V2 transitions",
        "",
        "| V1 intent | V2 intent | Count |",
        "|---|---|---:|",
    ])
    lines.extend(f"| {a} | {b} | {n} |" for (a, b), n in sorted(transitions.items()))
    lines.extend(["", "## Fallback counts", "", "| V2 intent | Fallback rows | Explicit rows |", "|---|---:|---:|"])
    for intent in intents:
        subset = df[df["v2_intent"] == intent]
        lines.append(f"| {intent} | 0 | {len(subset)} |")
    lines.append(f"| <unresolved> | {unresolved} | 0 |")
    lines.extend(["", "## Overlap statistics", "", f"- Rows matching multiple V2 rule groups: {sum(overlaps.values())}", "", "| Matched rule groups | Count |", "|---|---:|"])
    lines.extend(f"| {' + '.join(k)} | {v} |" for k, v in overlaps.most_common())

    for intent in intents:
        lines.extend(["", f"## Representative {intent} examples", ""])
        subset = _examples(df, df["v2_intent"] == intent)
        if not subset:
            lines.append("No examples available.")
        for row in subset:
            lines.append(f"- `{_message(row)}` — rules: `{row['matched_rules']}`")

    lines.extend(["", "## Representative unresolved examples", ""])
    for row in _examples(df, df["v2_intent"].isna()):
        lines.append(f"- `{_message(row)}` — matched rules: `{row['matched_rules']}`")

    lines.extend(["", "## V1/V2 disagreements", ""])
    disagreements = df[
        (df["v1_intent"] != df["v2_intent"].fillna("<unresolved>"))
    ]
    for _, row in disagreements.sample(
        n=min(50, len(disagreements)), random_state=SAMPLE_SEED
    ).iterrows():
        lines.append(
            f"- `{_message(row)}` — V1: `{row['v1_intent']}`, V2: "
            f"`{row['v2_intent'] or '<unresolved>'}`, rules: `{row['matched_rules']}`, "
            f"reason: {row['decision_reason']}"
        )
    lines.extend([
        "",
        "## Rule V2 configuration",
        "",
        "- Billing requires billing/subscription/payment evidence; Premium alone is insufficient.",
        "- Downloads require content/offline/sync evidence; app/image/update downloads are not sufficient.",
        "- App bugs require explicit crash, load, error, UI, spinner, or specific malfunction evidence.",
        "- Login requires authentication or account-access evidence; bare usernames/emails are insufficient.",
        "- Content search requires both a content/library object and missing/availability evidence.",
        "- Specific support actions take precedence over generic technical wording.",
        "- No-match messages remain unresolved and are never converted to `general_inquiry`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    df = audit_frame()
    df.to_csv(LABELS_OUT, index=False, encoding="utf-8")
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(build_report(df), encoding="utf-8")
    print(f"Wrote {len(df)} Rule V2 labels to {LABELS_OUT}")
    print(f"Wrote audit report to {REPORT_OUT}")


if __name__ == "__main__":
    main()
