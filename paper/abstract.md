# Unlearning vs. Hiding: Does Unlearning Remove Knowledge or Just Suppress Its Expression?

**Track:** 2b (Safety and Alignment), solo. **Problem statement:** 7 (Unlearning). **Model:** `CohereLabs/tiny-aya-global` (3.35B) and its Earth, Fire and Water variants. **Toolkits chained:** SafeTune, AuditKIT, CircuitKIT.

> Draft with placeholders. Tags: `[[RESULT]]` = number from the final runs; `[[DECIDE]]` = open choice; `[[CHECK]]` = verify before submitting. Nothing here is a measured result. No GPU runs have happened yet. Delete this note and all tags before submitting.

## Abstract

Machine unlearning should remove designated knowledge from a model, but a drop in accuracy on direct questions does not show the knowledge is gone. We run one toolkit-chained audit of RMU unlearning on Tiny Aya, a 3.35B multilingual model family. We teach Tiny Aya Global fictitious-author facts from TOFU in English with a LoRA fine-tune (`forget01`, 40 question-answer pairs, plus 3,600 `retain99` rows), merge the adapter, unlearn `forget01` with SafeTune's Representation Misdirection for Unlearning (RMU) at mid-depth layers 16 to 18 of 36, and repeat the protocol on Tiny Aya Earth, Fire and Water [[DECIDE: variant claim is behavioral only, or behavioral plus internal tests, depending on which variant runs finish; circuits are Global-only]].

The headline metric is `judge_forget_suppression`: one minus the factual-match rate from a local Qwen3.5 35B judge on forget prompts, reported per language, with Kimi K3 re-grading a seeded 15% sample (agreement and Cohen's kappa). Failed judgments are missing, not zero; we report n_judged and n_missing. Substring `forget_suppression` is secondary. General capability uses TOFU `world_facts` and `real_authors`, before and after RMU.

Behaviorally, the headline metric moves from [[RESULT]] after injection to [[RESULT]] after RMU on direct questions, and is [[RESULT]], [[RESULT]], [[RESULT]] and [[RESULT]] under paraphrased, reformatted, extraction-style and translated prompts (eight machine-translated languages; facts injected in English only). Retain and general-capability scores go from [[RESULT]] to [[RESULT]].

Internally, we fit layer-wise linear probes and score them on held-out rows (seeded stratified 70/30 splits, repeated 5 times), with a base-model control, a shuffled-label control and a content probe. We add a logit lens, activation patching and circuits. [[RESULT: one-sentence probe and logit-lens finding]]. A per-step relearning curve compares the RMU model relearning forgotten facts with the same model learning new held-out facts, plus a never-injected base reference. It shows [[RESULT]]. An RMU layer sweep (7, 18, 30) and [[RESULT]] seeds show [[RESULT: stability]]. The variants differ by [[RESULT: cross-variant finding]].

[[DECIDE: closing claim, chosen once numbers are frozen. (a) Suppression: RMU mainly gates expression, with residual information in layers [[RESULT]]. (b) Erasure: behavioral, internal and relearning measures agree. (c) Mixed: name the measures that disagree.]]

**Why the resulting model is meaningfully better for usage:** [[DECIDE: (a) if a targeted re-run was done, re-targeting RMU at layers [[RESULT]] lowers probe recoverability from [[RESULT]] to [[RESULT]] at a retain cost of [[RESULT]]; (b) audit only: the audit tells a deployer which tests an unlearned checkpoint must pass and its measured residual leakage]].

This is a measurement study, not a new unlearning method; prior work already shows unlearned knowledge is often recoverable. Our multilingual test follows up Farashah et al. (Aya-Expanse 8B, behavioral metrics, within-language unlearning); the Unlearning Depth Score evaluates on TOFU forget10 and does not test other languages or use linear probes. To our knowledge, no earlier unlearning study of Tiny Aya exists [[CHECK: repeat search]]. Results rest on 40 forget questions and [[RESULT]] seed(s). Artifacts: [[CHECK: links and licence]].
