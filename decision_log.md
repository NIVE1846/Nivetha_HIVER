# Decision Log

This file records non-obvious design decisions made during the project.
Each entry explains what was decided, why, what alternatives were considered, and the expected impact.

---

## Decision 1: Brand Selection

**Decision:** Use **SpotifyCares** as the single brand for the support agent.

**Why:**
- 15,096 confirmed customer->brand interaction pairs - large enough for meaningful ML, small enough for fast local iteration.
- Focused, well-defined support domain: account access, billing/subscription, playback issues, app bugs, playlist/library, cancellation. Intent taxonomy is tractable.
- Consistent brand voice across replies, which makes grounded response generation more meaningful and evaluable.
- Universally understood product - easy to reason about during a live interview.

**Alternatives considered:**
- **AmazonHelp** (100,503 interactions): Rejected. Spans too many product lines (Alexa, Kindle, AWS, third-party sellers, logistics). Intent taxonomy would be unwieldy and noisy. Processing 169k brand tweets repeatedly is slow on 8 GB RAM.
- **AppleSupport** (36,658 interactions): Rejected. Good candidate but larger than needed; Apple support covers hardware + software + services which adds complexity without proportional benefit.
- **ATVIAssist** (13,055 interactions): Rejected. Gaming support is more niche; fewer people can intuitively evaluate response quality during an interview.
- **VerizonSupport** (12,953 interactions): Rejected. Telecom billing disputes are legally sensitive and harder to evaluate for grounding quality.

**Reason rejected (general):** Larger brands add processing overhead and intent complexity without improving the core system design demonstration.

**Impact:** All downstream components (conversation reconstruction, intent discovery, retrieval index, golden set, evaluation) will be scoped to SpotifyCares data only. A brand-filtered CSV will be created to avoid re-scanning the full 516 MB file.

---

## Decision 2: Chunked CSV Processing

**Decision:** Read twcs.csv exclusively via `pd.read_csv(..., chunksize=100_000)` during analysis. Never load the full file into memory at once.

**Why:** The full CSV is ~516 MB / 2.8M rows. Loading it entirely on an 8 GB RAM laptop leaves insufficient headroom for model training and retrieval index construction.

**Alternatives considered:** Loading full CSV with `pd.read_csv` (no chunking). Rejected - risks OOM errors and makes the pipeline non-reproducible on constrained hardware.

**Impact:** After brand selection, a filtered SpotifyCares subset (~15k interactions) will be saved to `data/processed/` so subsequent phases never need to re-scan the full CSV.

---

## Decision 5: Intent Count and Taxonomy

**Decision:** Use 7 intents: playback_issue, app_bug, account_login, premium_billing, download_offline, content_search, general_inquiry.

**Why:** Keyword frequency and bigram analysis of 11,938 customer messages revealed 6 clearly distinct technical problem categories plus a catch-all for general inquiries. Fewer than 6 would merge genuinely different problems (e.g. billing vs login). More than 8 would create intents with too few examples for reliable classification.

**Alternatives considered:** 5 intents (merge download+playback, merge search+general). Rejected - download and playback have different resolution paths. 10+ intents (split app_bug by platform). Rejected - platform is an attribute, not a separate intent.

**Impact:** 7 balanced intents with ~29 golden examples each. All intents have sufficient training data from the 11,938 interactions.

---

## Decision 6: Intent Labelling Method

**Decision:** Use conservative keyword/regex rules for initial labelling of the golden set and training data. Flag ambiguous examples (matching >1 pattern).

**Why:** With 11,938 examples, manual labelling of all data is impractical. Keyword rules are transparent, reproducible, and easy to explain in an interview. The golden set (203 examples) should be manually reviewed to correct rule errors before final evaluation.

**Alternatives considered:** Clustering (k-means on TF-IDF). Rejected for labelling - clusters don't map cleanly to support intents and require manual interpretation anyway. Zero-shot LLM labelling. Rejected - introduces a dependency on an LLM for a step that should be reproducible without one.

**Impact:** 45/203 golden examples flagged as ambiguous and requiring manual review. Rule priority order (playback > app_bug > account > billing > download > search > general) resolves ties deterministically.

---

## Decision 7: Golden Set Sampling

**Decision:** Stratified sampling of 29 examples per intent (203 total), random_state=42, minimum message length 15 characters.

**Why:** Stratification ensures every intent is equally represented in evaluation, preventing the classifier from gaming metrics by over-predicting frequent intents. Fixed random state ensures reproducibility.

**Alternatives considered:** Pure random sampling. Rejected - would under-represent rare intents. Larger golden set (500+). Rejected - manual review burden is too high for a single person.

**Impact:** Balanced evaluation. The 203 golden examples are excluded from classifier training to prevent leakage.

---

## Decision 8: Main Classifier Architecture

**Decision:** Use TF-IDF FeatureUnion (word 1-2gram + char 3-5gram) + LinearSVC with a separate LR probability calibrator.

**Why:** sentence-transformers/PyTorch DLLs are blocked by Windows Application Control policy on this machine. LinearSVC with character n-grams is a well-established strong baseline for short noisy text. Character n-grams handle misspellings common in tweets. The LR calibrator on SVC decision scores gives usable probability estimates without requiring isotonic regression (also blocked).

**Alternatives considered:** sentence-transformers (all-MiniLM-L6-v2). Rejected - torch DLLs blocked. Plain TF-IDF + LR. Rejected - LinearSVC gives better margin-based generalisation on minority classes.

**Impact:** Macro F1 improved from 0.73 (LR) to 0.74 (SVC) on test split. Precision on minority intents improved significantly (1.00 for playback/app_bug/account).

---

## Decision 9: Retrieval Method

**Decision:** TF-IDF cosine similarity with a separate index per intent (intent-scoped retrieval).

**Why:** Intent-scoping ensures retrieved examples are topically relevant. A global index would return billing examples for a playback query if similarity scores happen to be higher. TF-IDF is fast, interpretable, and sufficient for keyword-heavy support text.

**Alternatives considered:** Dense retrieval (sentence embeddings). Rejected - torch blocked. Global TF-IDF index. Rejected - cross-intent retrieval noise.

**Impact:** Retrieval always returns same-intent examples. Avg similarity scores are low (0.10-0.25) because support tweets are short and varied, but the intent constraint ensures topical relevance.

---

## Decision 10: Number of Retrieved Examples

**Decision:** Retrieve top-3 examples per query.

**Why:** Top-1 is used for template generation (best match). Top-3 provides fallback options for the validator and gives the deterministic response-quality judge more context. More than 5 adds noise without benefit for short tweet-length responses.

**Alternatives considered:** Top-1 only. Rejected - no fallback if top-1 response is too short. Top-5. Rejected - diminishing returns.

**Impact:** Template generator uses top-1; validator checks overlap against all 3.

---

## Decision 11: Confidence Threshold Selection

**Decision:** conf=0.70, sim=0.10, selected by validation sweep targeting unsafe_rate < 5%.

**Why:** The sweep showed conf=0.70, sim=0.10 gives 92.4% auto-handle rate with 2.5% unsafe rate on the validation set. Better operating point than conf=0.50 which has higher auto-handle rate but also higher unsafe rate.

**Alternatives considered:** conf=0.50 (highest auto-handle rate). Rejected - 3.2% unsafe rate acceptable but conf=0.70 gives better precision. conf=0.90. Rejected - reduces auto-handle rate to 76% without meaningful safety improvement.

**Impact:** 92.4% auto-handle on validation set. 62.1% on golden set (harder examples).

---

## Decision 12: Escalation Policy for High-Risk Intents

**Decision:** premium_billing and account_login always escalate, regardless of confidence.

**Why:** Errors in billing responses (wrong refund info) and account responses (wrong password reset advice) carry real customer harm. The cost of a false escalation is much lower than the cost of sending incorrect billing advice.

**Alternatives considered:** Allow auto-handle for high-confidence billing/account cases. Rejected - classifier F1=0.90 for billing means 10% error rate, unacceptable for financial advice.

**Impact:** 58 high-risk examples in golden set: 51 correctly escalated (87.9% recall). 7 were auto-handled due to misclassification as a non-high-risk intent.

---

## Decision 13: LLM Choice

**Decision:** google/flan-t5-base for LLM mode; template mode as default.

**Why:** flan-t5-base is ~250 MB, runs on CPU in Colab free tier, instruction-tuned, no paid API required. Template mode (copying top retrieved response) is used as default because it has zero hallucination risk and is fully explainable.

**Alternatives considered:** GPT-4/Claude (paid APIs). Rejected per assignment constraints. Llama-2-7B / Mistral-7B. Rejected - too large for 8 GB RAM.

**Impact:** All local evaluation uses template mode. LLM mode available in Colab via use_llm=True.

---

## Decision 14: Validation Strategy

**Decision:** Rule-based grounding validator checking for hallucination signals, relevance overlap, and contradiction patterns.

**Why:** An LLM-based validator would require torch (blocked). Rule-based validation is transparent, fast, and catches the most dangerous failure modes (refund promises, guarantee language, account action claims).

**Alternatives considered:** LLM-as-validator. Rejected - torch blocked locally. No validation. Rejected - unsafe auto-handle rate would increase.

**Impact:** Validator pass rate = 100% on golden set auto-handled responses. Expected because template mode copies historical responses that don't contain hallucination signals.

---

## Decision 15: Evaluation Methodology

**Decision:** Use a 203-example stratified golden set as the primary evaluation set. Report honest caveats about label leakage.

**Why:** The golden set is the only evaluation that uses examples not seen during training. The test split (from the same keyword-labelled pool as training) gives optimistic metrics. The golden set reveals true system performance on harder examples.

**Alternatives considered:** Report only test split metrics. Rejected - misleading (96% accuracy on test vs 69% on golden set).

**Impact:** Honest reporting of 69% accuracy and 36.5% unsafe auto-handle rate on golden set, with explanation of why these numbers are inflated by label leakage.

---

## Decision 3: Conversation Reconstruction Strategy

**Decision:** Reconstruct interactions as direct customer->brand pairs using `in_response_to_tweet_id`. Attach one level of prior context where available. Do not attempt to reconstruct full multi-turn threads.

**Why:** The dataset's `in_response_to_tweet_id` field reliably links a brand reply to the customer message it answered. One level of prior context is enough to disambiguate follow-up messages without the complexity of full thread reconstruction.

**Alternatives considered:** Full thread reconstruction (walk the entire chain). Rejected - many threads have missing/deleted intermediate tweets, making full reconstruction unreliable and adding significant complexity for marginal gain.

**Impact:** 11,938 clean interaction pairs recovered from 43,265 brand tweets. 31,213 brand tweets had no recoverable customer parent (deleted tweets or tweets outside the SpotifyCares extract). Zero empty messages.

---

## Decision 16: Validator Expansion (Audit Phase G)

**Decision:** Expand the grounding validator from 4 checks to 7 checks. Add evidence_availability, excessive_certainty, and missing_context checks. Add check_name field to ValidationResult. Document all known limitations explicitly.

**Why:** The original 4-check validator had no check for missing evidence (could validate a response with zero retrieval), no check for overconfident language, and no check for redundant information requests. These are real failure modes in a support agent. The check_name field is needed for debugging and for the pipeline to log which check failed.

**Alternatives considered:** Keep 4 checks and add a separate post-hoc audit. Rejected — the validator is the right place for these checks; a separate audit would not prevent bad responses from being sent.

**Documented limitations added:** Keyword patterns miss paraphrased hallucinations. Token overlap is a weak relevance proxy. Contradiction detection covers only known patterns. The validator cannot detect factually incorrect troubleshooting steps.

**Impact:** Validator now catches 7 failure modes. All 22 tests still pass. Pass rate on golden set remains 100% (expected — template mode copies real responses that don't trigger any check).

---

## Decision 17: Router Risk Level Field (Audit Phase H)

**Decision:** Add explicit risk_level field ("HIGH"/"MEDIUM"/"LOW") to RoutingDecision. Add RISK_LEVELS dict mapping all 7 intents. Add human-readable reason strings for every escalation path. Document threshold selection rationale and HIGH RISK policy justification in module docstring.

**Why:** The original router returned route and reason but no structured risk level. Downstream consumers (pipeline, evaluation, logging) need a machine-readable risk level to distinguish "escalated because high-risk intent" from "escalated because low confidence". Human-readable reasons are required for the escalation audit trail.

**Alternatives considered:** Infer risk level from reason string. Rejected — brittle and not machine-readable.

**Impact:** RoutingDecision now has 6 fields. All 22 tests still pass. Evaluation metrics unchanged.

---

## Decision 18: Retrieval Evaluation Fix (Audit Phase E)

**Decision:** Remove Recall@k = 1.0 from retrieval_eval.md. Replace with similarity distribution metrics (mean, median, fraction above 0.20/0.30) and token overlap. Create evaluation/retrieval_sample_for_review.csv for human relevance evaluation.

**Why:** Recall@k = 1.0 is trivially true for intent-scoped retrieval — every retrieved example is from the correct intent by construction. Reporting it as a metric is misleading. Similarity distribution and token overlap are honest proxies for retrieval quality, with the caveat that human evaluation is still needed.

**Alternatives considered:** Keep Recall@k but add a caveat. Rejected — a misleading metric with a caveat is still misleading. Better to replace it entirely.

**Impact:** reports/retrieval_eval.md now reports honest metrics. Mean top-1 similarity = 0.39. Human evaluation pending on 30-example sample.

---

## Decision 19: Honest Results Reporting (Audit Phase I)

**Decision:** Rewrite reports/results.md to include: separate tables for test-split vs golden-set metrics, explicit label leakage caveat, explanation of why 36.5% unsafe rate is inflated, explanation of why 96% test accuracy is optimistic, and a "What I Would Do With One More Week" section.

**Why:** The original results.md reported headline numbers without context. A reviewer reading only the headline numbers would conclude the system is production-ready (96% accuracy, 3.4% unsafe rate). The honest picture is more nuanced: 69% accuracy on harder examples, 36.5% unsafe rate inflated by label leakage, relevance score of 2.25/5.

**Alternatives considered:** Add footnotes to the existing table. Rejected — footnotes are easy to miss. A dedicated "What is Misleading" section forces the reader to engage with the caveats.

**Impact:** results.md now gives an accurate picture of system performance. The caveats are documented before the numbers, not after.

---

## Decision 4: Text Cleaning - Strip @mentions Only

**Decision:** Remove @handles from tweet text but preserve all other content including URLs, punctuation, and emoji.

**Why:** @mentions are routing artefacts (e.g. `@SpotifyCares`) that carry no semantic content for intent classification or retrieval. Stripping them reduces noise without losing meaning. URLs and emoji may carry signal (e.g. a screenshot link suggests a visual bug report).

**Alternatives considered:** Aggressive cleaning (remove URLs, emoji, punctuation). Rejected - over-cleaning loses signal and makes responses look unnatural.

**Impact:** Cleaner input to the classifier and retrieval system without information loss.
