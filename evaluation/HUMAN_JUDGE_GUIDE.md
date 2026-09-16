# Human Response-Judge Guide

Use this guide to score the 50 rows in `evaluation/human_response_eval.csv`.
The rows correspond by `example_id` to `evaluation/human_judge_sample.csv`.
Do not infer a score from the machine judge. Score each criterion from 1 to 5.

## Inputs Required

For every row, review:

- the customer message
- the historical evidence used by the pipeline
- the generated response

The historical evidence can be retrieved from the corresponding sample row or
from the project retrieval index. Do not fill missing evidence by guessing.

The judge must see all three. The `evidence_used` column should contain the retrieved customer/brand examples or their identifiers before scoring.

## Criteria

### Correctness

1 = materially wrong or unsafe; 3 = partly correct or incomplete; 5 = accurate and appropriate to the evidence and customer problem.

### Groundedness

1 = unsupported claims or promises; 3 = partly supported; 5 = every substantive claim is supported by the historical evidence or is a clearly safe conversational transition.

### Relevance

1 = does not address the message; 3 = partially related; 5 = directly addresses the customer's actual issue.

### Helpfulness

1 = no useful next step; 3 = some useful direction; 5 = clear, practical next steps or a justified request for information.

### Tone

1 = inappropriate or dismissive; 3 = acceptable but impersonal; 5 = professional, clear, and empathetic.

## Procedure

1. Score independently and do not alter the generated response.
2. Use integer values 1 through 5 only.
3. Explain unusual or borderline scores in `human_notes`.
4. Do not fill missing evidence by guessing; record the gap in `human_notes`.
5. Human-vs-judge agreement must be calculated only after a real
   `evaluation/llm_judge_sample.csv` and completed human scores both exist.

No human scores or agreement values are pre-filled in the sample.
