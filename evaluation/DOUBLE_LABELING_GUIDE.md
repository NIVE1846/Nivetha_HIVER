# Double-Labeling Guide

This file describes the independent agreement exercise for `double_label_sample.csv`.

## Procedure

1. Annotator A labels all 50 rows using `HUMAN_LABELING_GUIDE.md`.
2. Annotator B labels the same 50 rows independently, without seeing A's labels.
3. Each annotator fills only their own intent and escalation columns.
4. Do not copy `suggested_intent` into an annotator field without reading the message.
5. Do not remove difficult or ambiguous examples.
6. Use the optional `notes` column to record disagreements or uncertainty after both annotations are complete.

The sample was selected from the golden set with random seed `42`. The machine-generated `suggested_intent` is included only as provenance and reference; it is not a gold label.

## Columns

- `example_id`: stable golden-set identifier
- `customer_message`: message to label
- `suggested_intent`: automatic suggestion, never ground truth
- `annotator_A_intent`, `annotator_B_intent`: one of the 8 current intents
- `annotator_A_should_escalate`, `annotator_B_should_escalate`: `YES` or `NO`
- `notes`: optional process notes; do not use it to replace a missing label

## Agreement command

After both annotators have completed their columns:

```bash
python evaluation/agreement.py
```

The script excludes incomplete pairs and reports raw agreement, disagreement count, disagreement pairs, and Cohen's kappa for intent and escalation. It does not infer missing labels or fabricate a result. If kappa is below 0.60, document the disagreement patterns and any justified guide revision before re-annotation.
