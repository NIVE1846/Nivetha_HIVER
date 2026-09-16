# Hiver SDE Intern Assignment — AI Customer Support Agent

## Project Overview

An evidence-first AI customer support agent for **SpotifyCares**, built on the [Customer Support on Twitter (TWCS) dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

The system classifies incoming customer messages, retrieves historically similar support cases as evidence, generates a grounded response, and decides whether to auto-handle or escalate to a human — with a reason for every escalation.

**Design principle:** **evidence first, generation second, escalation when uncertain.**

---

## Architecture

```text
Customer message
      |
      v
Intent Classifier
(TF-IDF word + char n-grams + LinearSVC
 + optional LR confidence calibrator)
      |
      v
Intent + Confidence
      |
      v
Historical Retrieval
(TF-IDF cosine similarity, intent-scoped index)
      |
      v
Evidence Score
(best similarity)
      |
      v
Confidence / Evidence Gate
(router.py)
      |
   --------------
   |            |
ESCALATE    AUTO_HANDLE
   |            |
   v            v
Human       Response Generator
Review      (template: top retrieved response)
                |
                v
        Grounding Validator
        (hallucination + relevance checks)
                |
             -------
             |     |
           PASS   FAIL
             |     |
             v     v
           Send  ESCALATE
