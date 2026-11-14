# Unlearning vs. Hiding: Does Unlearning Remove Knowledge or Just Suppress Its Expression?

Lexsi AI Safety and Alignment Hackathon, **Track 2b**, problem statement **7 (Unlearning)**. Model: `CohereLabs/tiny-aya-global` (3.35B), compared with Tiny Aya Earth and Fire. Toolkits chained: **SafeTune** (RMU unlearning), **AuditKIT** (scoring), **CircuitKIT** (EAP-IG circuits; run on Global, not analyzed in this version and left to future work).

> Numbers are from the overnight runs: one seed (42), 40 forget questions, small n throughout. Pending: Global seeds 43 and 44, Water, Base and Aya Expanse 8B, a patching rerun with positive controls, analysis of the head-ablation and circuit artifacts, a native-speaker check of translations, links and the artifact licence. The full paper is `paper/main.tex`.

## Summary

Unlearning should remove designated knowledge, but lower accuracy on direct questions does not show the knowledge is gone. We audit Representation Misdirection for Unlearning (RMU) on Tiny Aya, a 3.35B multilingual family, with SafeTune and AuditKIT. We inject fictitious authors from the Task of Fictitious Unlearning (TOFU) benchmark (`forget01`, 40 pairs) into Tiny Aya Global with LoRA, unlearn with RMU at layer 18 (updating layers 16-18), and repeat on Earth and Fire.

Method contribution. Standard RMU did nothing here. A roughly 360-token system preamble shared by every prompt, plus attention-sink tokens, dominates the forget loss, which stayed flat near 3.5. Four unmasked settings left forget ROUGE-L (overlap with the true answer) at 0.92-0.97, versus 0.97 injected. Masking the forget loss to first-user-turn tokens (alpha 100, steering coefficient 20, 80 steps) lowers it to 0.32.

Results. Judge forget suppression is the fraction of injected facts an output does not reveal, graded by `qwen3.5:35b` with a seeded 15% Kimi K3 re-grade. On Global it rises from 0.025 to 0.85 on direct questions, 0.45 to 0.95 on paraphrases and 0.125 to 0.95 on extraction prompts, and from 0.71 to 0.92 across eight machine-translated languages (Hindi weakest, 0.87). Earth and Fire behave alike (direct 0.05 to 0.875, 0.075 to 0.875). Judged retain accuracy falls 3-9 points (Global 0.52 to 0.42), a small but real cost, with little change on world facts and real authors. Format changes leak: the pooled cloze and true/false bucket reaches only 0.74, driven by true/false (0.63).

We do not claim erasure. After 30 fine-tuning steps the RMU model recovers 5 of 8 forgotten facts, against 4 of 8 new facts on the same model and 7 of 8 for an uninjected base model; the one-fact gap is not significant. Held-out probes separate forget from retain questions at AUROC (area under the ROC curve) near 0.98 from layer 1 up, as does a base model that never saw the facts, so the probes reflect surface differences, not stored knowledge. On Global, a single-token, teacher-forced logit lens shows RMU lowers but does not remove output probability on the answer's first content token (0.971 to 0.696, n=40). We read the results as output-level suppression across three variants, formats and languages, with weak evidence of residual knowledge. We report judge agreement with raw agreement, Cohen's kappa and two prevalence-insensitive statistics, because kappa collapses to 0.0 on the one-sided base-model pass although raw agreement is 0.94. Small samples, one seed and pending runs limit every conclusion. Prior work shows unlearned knowledge is often recoverable, and a multilingual unlearning study exists (Farashah et al.). To our knowledge, no unlearning study of Tiny Aya was found.

## Why the resulting model is meaningfully better for usage

Out of the box, RMU left this model unchanged (four unmasked settings). With a masked forget loss it suppresses the injected facts under paraphrase, extraction and translation at a measured retain cost of 3 to 9 points. The audit shows what that does not guarantee: format-change leakage remains, and 30 fine-tuning steps recover most forgotten facts. We do not claim erasure.

## Contributions, in order

1. **Masked-forget-loss RMU.** The chat template's shared system preamble (about 360 tokens) and attention-sink tokens dominate SafeTune's forget MSE, so unmasked RMU does nothing on Tiny Aya (forget loss flat at about 3.5; best forget ROUGE-L 0.92). Taking the forget MSE only over tokens from the first user turn on fixes it (forget ROUGE-L 0.32), with the rest of the SafeTune loop unchanged.
2. **One protocol across Tiny Aya variants.** Global, Earth and Fire under the same pipeline. To our knowledge no earlier unlearning work on Tiny Aya exists. [[CHECK: repeat the literature search before submitting]]
3. **Multilingual leakage paired with internal tests** (held-out probes with a base-model control, relearning, a teacher-forced logit-lens readout of the answer token, and a single-site patching sweep that is reported but uninterpretable). Follow-up to Farashah et al. (*Multilingual Amnesia*); the Unlearning Depth Score (Lee et al.) evaluates on TOFU forget10 and does not test other languages or use linear probes.
4. **Mid-depth RMU, relearning and judge-agreement reporting** (raw agreement, kappa, PABAK, Gwet's AC1).

## Setup

- LoRA injection of TOFU `forget01` plus a 3,600-row `retain99` slice (rank 16, alpha 32, 3 epochs, lr 2e-4); adapter merged before RMU.
- RMU, masked forget loss: steering layer 18, update layers 16 to 18, alpha 100, steering coefficient 20, lr 5e-4, 80 steps.
- Headline metric: judge forget suppression (primary judge `qwen3.5:35b` via Ollama; seeded 15% re-grade by `moonshotai/kimi-k3` via OpenRouter). Missing judgments are counted, not scored as zero.
- Formats: direct, paraphrase, cloze, true/false, extraction, and translation into 8 languages.
- Logit lens (Global only): each of the 37 residual positions read through the output head for the injected and RMU checkpoints, 40 forget and 32 retain items, scored on the first content token (openers skipped and teacher-forced into the prompt), 95% Student-t intervals over items.

## Results (Global; before RMU to after RMU)

| Metric | Before | After |
|---|---|---|
| Direct suppression | 0.025 | 0.85 |
| Paraphrase | 0.45 | 0.95 |
| Cloze (alone) | 0.10 | 0.85 |
| True/false (alone) | 0.225 | 0.625 |
| Cloze + true/false pooled | 0.163 | 0.738 |
| Extraction | 0.125 | 0.95 |
| Translation, 8 languages | 0.706 | 0.925 |
| Retain accuracy (judge) | 0.516 | 0.422 |
| World facts / real authors | 0.875 / 0.825 | 0.85 / 0.775 |
| Substring recall, direct | 0.825 | 0.000 |

Earth: direct 0.05 to 0.875, retain 0.562 to 0.516. Fire: direct 0.075 to 0.875, retain 0.531 to 0.500. Hindi is the weakest language after RMU in all three variants.

- **Residual leakage.** Substring recall on cloze plus true/false stays at 0.325 on Global (true/false carries it). The base model was not evaluated on this format.
- **Relearning (30 steps, n=8).** RMU model on forgotten facts 5/8, RMU model on new facts 4/8, never-injected base on the forgotten facts 7/8. The one-fact gap is not significant. Weak evidence of residual knowledge at best.
- **Probes.** Held-out probes decode at AUROC about 0.98 from layer 1, but the base model that never saw the facts does the same, so they say nothing about stored knowledge.
- **Logit lens (Global; single token, teacher-forced).** The injected and RMU curves coincide through residual 16 by construction (RMU changes no weight below decoder layer 16). The lens is uninformative before the last layers for forget and retain alike (residual 16: forget 0.0023, retain 0.016), so it can neither confirm nor rule out an internal representation. At the output the first-content-token probability goes from 0.971 [0.946, 0.995] (injected) to 0.696 [0.567, 0.824] (RMU, n=40), with the correct token at rank 1 on 30 of 40 items (median rank 1); it is 0.648 [0.502, 0.793] on the 34 judge-suppressed facts and 0.968 on the 6 still matching (n=6, too few to interpret). This fits suppressed rather than erased. The rule skips an opener on 18 of 40 items, the first-answer-token probability is 0.582 [0.442, 0.722], and mean ranks are outlier-driven.
- **Activation patching (Global, appendix; no positive control).** Injected-model residuals were patched into the RMU model at the last prompt token, one decoder layer at a time (36 layers), prefill only, greedy 32-token decoding, scored by whether the full reference answer is a substring (`contains_target`), on the 8 facts with the largest output gap (ids 17, 8, 19, 1, 13, 15, 16, 2). 0 of 36 layers restored the target for each fact (0 of 288 generations). There is no positive control, and six of the eight reference answers (40-66 tokens) are longer than the 32 generated tokens, so the null is uninterpretable and not a finding.
- **Judge agreement.** Base-model pass raw 0.9375 with kappa 0.0 (one-sided labels), PABAK 0.875, AC1 0.934; evaluation passes kappa 0.74 to 0.94.
- **Sweep (n=20 per cell).** Layer 7 alpha 100 and layer 18 coefficient 40 scored better than the chosen configuration on the subset; we do not claim the configuration is optimal.

## Limitations

Single seed; 40 forget questions; relearning n=8; sweep n=20; hyperparameters set on the evaluated questions; injected (not pretraining) facts, injected only partly on retain (0.52); English-only injection and machine-translated queries; model judges that lean lenient on the secondary side; no base-model reference for true/false; Global, Earth and Fire only; the logit lens covers Global only (one seed, single token, teacher-forced, uninformative before the last layers); the patching sweep has no positive control and is uninterpretable; head ablation and CircuitKIT are not analyzed.

## Ethics

The forget set is synthetic (TOFU fictitious authors); no real personal data and no hazardous-knowledge data are used. We report aggregate recoverability, not a method for extracting private data from deployed systems. Released checkpoints are derivatives of a CC-BY-NC model under Cohere's acceptable-use policy. [[CHECK: confirm that un-gated republication is allowed]]

## Reproducibility and artifacts

Code [[LINK]] (commit [[RESULT]]); TOFU splits `forget01`, `retain99`, `forget01_perturbed`; translations and review sheet [[LINK]]; adapters and checkpoints [[LINK]]; probes, traces and generations [[LINK]]. Seed 42 only so far.

## References

TOFU (arXiv:2401.06121); WMDP and RMU (2403.03218); NPO (2404.05868); Who's Harry Potter (2310.02238); ConceptVectors, Hong et al. (2406.11614); Lucki et al. (2409.18025); Multilingual Amnesia, Farashah et al. (2601.05641); Hong et al. on fine-tuning unlearning (2410.06606); Choi et al. on cross-lingual unlearning (2406.12354); Zhang et al. on quantization (2410.16454); Che et al. on model tampering (2502.05209); Lee et al. on distillation (2506.06278); Guo et al. on mechanistic unlearning (2410.12949); Hwang et al. on English-only unlearning (2510.23949); Jang et al. on prompt attacks (2506.10236); Lee, Kim and Jo, Unlearning Depth Score (2605.24614); Deeb and Roger (2410.08827); Patil et al. (2309.17410); Seyitoglu et al. (2411.02631); Hu et al. (2406.13356); LoRA (2106.09685); linear probes (1610.01644); tuned lens (2303.08112); activation patching practices (2309.16042); attribution patching (2310.10348); EAP-IG (2403.17806); Tiny Aya (2603.11510). Full BibTeX in `paper/references.bib`.
