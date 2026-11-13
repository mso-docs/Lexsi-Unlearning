# Unlearning vs. Hiding: Does Unlearning Remove Knowledge or Just Suppress Its Expression?

Lexsi AI Safety and Alignment Hackathon, **Track 2b**, problem statement **7 (Unlearning)**. Model: `CohereLabs/tiny-aya-global` (3.35B), compared with Tiny Aya Earth, Fire and Water. Toolkits chained: **SafeTune** (RMU unlearning), **AuditKIT** (scoring), **CircuitKIT** (EAP-IG circuits).

> Draft with placeholders. `[[RESULT]]` = number from the final runs. `[[DECIDE]]` = open choice. `[[CHECK]]` = verify before submitting. `[[LINK]]` = add a link. Nothing here is a measured result. No GPU runs have happened yet. The method below follows the pipeline code. The full paper (`paper/main.tex`) has the details; this section is the Phase 1 write-up.

## Summary

[[WRITE: paste the final abstract from `paper/abstract.md` here.]]

**Why the resulting model is meaningfully better for usage:** [[DECIDE: (a) targeted re-run lowers probe recoverability from [[RESULT]] to [[RESULT]] at a retain cost of [[RESULT]]; or (b) audit only: the audit tells a deployer which tests an unlearned checkpoint must pass, and reports its measured residual leakage.]]

This is a measurement study, not a new unlearning method. It does not claim that shallow unlearning is new; earlier work (ConceptVectors by Hong et al., Lucki et al., Hong et al. on retrieval changes, Deeb and Roger, Hu et al., Che et al. and others) already shows that unlearned information can often be recovered, and Jang et al. report that RMU resisted prompt attacks, so a clean result would not be surprising either.

**Contributions, in order:**
1. Tiny Aya Global against Earth, Fire and Water under one unlearning protocol. To our knowledge no earlier unlearning work on Tiny Aya exists. [[CHECK: repeat the literature search before submitting]] The queued variant runs include held-out probes, relearning and the interpretability extras, but CircuitKIT discovery runs on Global only. [[DECIDE: the overnight queue may be trimmed; claim a behavioral comparison only, or behavioral plus internal tests, depending on which variants finish]]
2. Eight-language leakage paired with internal tests (held-out probes, logit lens, patching, relearning). This is a follow-up to Farashah et al., *Multilingual Amnesia* (Aya-Expanse 8B, TOFU in ten languages, GradDiff, GradDiff-KL and NPO, behavioral metrics, within-language unlearning), not a new idea. The Unlearning Depth Score (Lee et al.) evaluates on TOFU forget10 (Sec. 4.1) and does not test other languages or use linear probes. Ours differs in using RMU, Tiny Aya and its variants, English-only injection, and internal recoverability.
3. RMU at mid-depth layers 16 to 18 of 36 instead of the library's early default layers, an RMU layer sweep, and a per-step relearning curve in which the RMU model relearns its forgotten facts and, separately, learns held-out new facts under the same schedule.

## What we did

1. **Inject (English only).** LoRA fine-tune (rank 16, alpha 32, all attention and MLP projections, 3 epochs, learning rate 2e-4) on the English TOFU `forget01` (40 QA pairs, two fictitious authors) plus the first 3,600 rows of `retain99` (180 authors; the last 18 authors stay held out for the relearning control). The pipeline aborts if post-injection forget accuracy is below 0.5.
2. **Unlearn.** The injection LoRA adapter is merged into the full weights, then SafeTune `RMUTrainer` is applied to the merged weights on `forget01`, with the 3,600-row retain slice as the retain set. Steering layer 18, update layers 16 to 18 (MLP `down_proj` only), alpha 1200, steering coefficient 20, learning rate 5e-5, at most 80 batch pairs. [[DECIDE: why layer 18 and not the library default of layer 7]]
3. **Behavioral evaluation.** Greedy generation, five query formats: direct, paraphrase (official `forget01_perturbed`), cloze and true/false rewrites, extraction-style prompts, and translations. Headline metric: `judge_forget_suppression` = 1 minus the primary judge's match rate on forget prompts, reported per format and per language, including translated prompts. Secondary: `judge_retain_accuracy` (retain slice, first 64 rows) and `judge_utility_accuracy` (TOFU `world_facts` and `real_authors`, 40 items each, 20 in queued jobs), before and after RMU, and substring `forget_suppression`, one minus the rate at which the full reference answer appears (lowercased substring) in the output. A failed judgment is missing, not zero; every rate comes with `n_judged` and `n_missing`. The base model is scored on direct forget and retain questions with the substring metric only.
4. **Multilingual leakage.** The forget questions and answers are machine-translated into es, fr, de, hi, ar, zh, sw and ja by a local helper model and used only to query the model; facts are injected and unlearned in English only. Translations are not human-verified. [[DECIDE: spot-check result]]
5. **Judging.** `qwen3.5:35b` served through Ollama gives a factual-match verdict for every case. `moonshotai/kimi-k3` through OpenRouter re-grades a seeded 15% sample; we report raw agreement and Cohen's kappa. The secondary judge runs only when configured. A judge is the headline because reference-based metrics can fail when a model answers in another language (Hwang et al.).
6. **Probes.** Layer-wise logistic probes at 37 residual positions (last non-padding token; indexing is padding-independent and batches are right-padded) separate forget questions (40) from retain questions (64 scored rows). Every number is held-out: a seeded stratified 70/30 split repeated 5 times (3 in queued jobs), reported as held-out mean and std of accuracy, balanced accuracy and AUROC. Frozen probes are fit on the post-injection model and applied to the base, post-injection, post-RMU and post-relearning models. Refit probes are fit on each model's own training rows. Controls are the base model before injection and shuffled labels. A content probe tells a question plus its true answer from the same question plus another forget answer. A null probe result would reproduce WMDP's finding of near-chance probes after RMU and is not evidence of erasure without relearning or another recovery control. Forget-versus-retain probes can partly reflect question topic, and the content probe's distractors may be confounded by topic.
7. **Interpretability.** Logit lens, activation patching from the pre-unlearning model into the post-RMU model, head ablation, token-by-layer norms, activation-difference histograms, PCA, and CircuitKIT EAP-IG circuit discovery (exploratory).
8. **Relearning curve.** 30 fine-tuning steps (learning rate 2e-4, batch size 1), scored at steps {0, 5, 10, 20, 30} ({0, 10, 30} in queued jobs). Arms: RMU-old (the post-RMU model on 8 forgotten facts), RMU-new (a fresh copy of the same post-RMU model on 8 new facts: fictitious TOFU authors held out from both the forget set and the retain slice) and base-old (the never-injected base on the same 8 forgotten facts; an acquisition-from-scratch reference, not a matched control). The old-fact arms are also scored on held-out paraphrases from `forget01_perturbed`; the new facts have none, so RMU-new is scored on trained questions only. The suppression-versus-removal comparison is RMU-old against RMU-new on trained questions. Substring match is scored at every step; the judge runs at the first and last step only.
9. **Robustness and variants.** The same recipe on Tiny Aya Earth, Fire and Water (the queued variant jobs include probes, relearning and the interpretability extras; CircuitKIT discovery is Global-only; [[DECIDE: variant claim depends on which jobs finish]]), an RMU layer sweep (layers 7, 18, 30; plus 12 and 24 if finished; substring metric on direct questions only, no judge, probes or relearning), and seed repeats (42, 43, 44, run without the interpretability extras; [[CHECK: which finished]]).

## Results

| Checkpoint | Direct | Paraphrase | Cloze | True/false | Extraction | Translated | Retain | General capability: World facts | General capability: Real authors |
|---|---|---|---|---|---|---|---|---|---|
| Base (substring only) | [[RESULT]] | -- | -- | -- | -- | -- | [[RESULT]] | -- | -- |
| Post-injection | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] |
| Post-RMU | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] |
| n_missing (post-RMU) | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] |

| Variant | Post-inj. direct | Post-RMU direct | Translated (sw, ar) | Translated (hi) | Translated (zh, ja, es, fr, de) | Retain | World facts | Real authors | Seeds | n_missing |
|---|---|---|---|---|---|---|---|---|---|---|
| Global (main) | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] |
| Earth | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] |
| Fire | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] |
| Water | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] | [[RESULT]] |

`judge_forget_suppression` on forget formats (higher = fact expressed less); retain and general-capability columns are judge accuracies. Substring `forget_suppression` is reported alongside in the paper. Mean with bootstrap 95% CI over judged items; `n_missing` counts failed judgments, excluded from the rates. The base model scores high on `forget_suppression` by construction, so read forgetting relative to the post-injection row.

**How to read null internal results:** if probes or the logit lens go quiet at layers 16 to 18, that is not erasure on its own. After RMU the logit lens falls to near chance in earlier work (Lucki et al.), yet recovery still works. Only the relearning curve against new facts, or another recovery control, separates erasure from hiding.

**Key findings** [[WRITE: after numbers are frozen]]

1. [[WRITE: finding 1, tied to a table cell or figure]]
2. [[WRITE: finding 2]]
3. [[WRITE: finding 3]]

Figures: held-out probe AUROC by layer (frozen and refit curves, base-model and shuffled-label controls, content probe) [[LINK]]; logit lens and norm ratio [[LINK]]; patching [[LINK]]; head ablation and token-by-layer norms [[LINK]]; activation differences and PCA [[LINK]]; circuits [[LINK]]; relearning curve with three arms [[LINK]].

## Limitations

- 40 forget questions and [[RESULT]] seed(s); intervals are wide.
- Facts were added by an extra fine-tuning phase, which can make unlearning look more robust than it is on pretraining knowledge (Deeb and Roger).
- Injection and unlearning are English-only, so the multilingual test asks a different question from within-language unlearning (Farashah et al.).
- The headline is a model judge, checked against a second judge on a 15% sample when that judge is configured; failed judgments are missing. The substring metric is strict, true/false items always assert the correct statement, and translated answers are machine translations.
- Forget-versus-retain probes can partly reflect question topic, and the content probe's distractors (other forget answers) may be confounded by topic. Held-out rows are few (40 forget and 64 retain questions), and failing to find a trace does not prove erasure.
- Relearning uses one schedule on 8 facts per arm. The new-fact arm has no held-out paraphrases, so RMU-old versus RMU-new is compared on trained questions only. The base-old arm is an acquisition-from-scratch reference. The judge runs only at the first and last step.
- General capability uses TOFU `world_facts` and `real_authors` (40 items each, 20 in queued jobs), not broad capability benchmarks. One architecture family and size. The layer sweep uses substring metrics only.
- Circuit task uses a first-token target and a non-minimal corrupted prompt; CircuitKIT support for Tiny Aya is experimental, and circuit discovery runs on Global only.

## Ethics

The forget set is synthetic (TOFU fictitious authors); no real personal data and no hazardous-knowledge data are used. We report aggregate recoverability, not a method for extracting private data from deployed systems. Released checkpoints are derivatives of a CC-BY-NC model under Cohere's acceptable-use policy. [[CHECK: confirm that un-gated republication is allowed]]

## Reproducibility and artifacts

Code [[LINK]] (commit [[RESULT]]); TOFU splits `forget01`, `retain99`, `forget01_perturbed`; translations and their review sheet [[LINK]]; adapters and unlearned checkpoints [[LINK]]; probes, traces and generations [[LINK]]. Seeds: [[CHECK]]. Hardware and runtime: [[RESULT]].

## References

TOFU (arXiv:2401.06121); WMDP and RMU (2403.03218); NPO (2404.05868); Who's Harry Potter (2310.02238); ConceptVectors, Hong et al. (2406.11614); Lucki et al. (2409.18025); Multilingual Amnesia, Farashah et al. (2601.05641); Hong et al. on fine-tuning unlearning (2410.06606); Choi et al. on cross-lingual unlearning (2406.12354); Zhang et al. on quantization (2410.16454); Che et al. on model tampering (2502.05209); Lee et al. on distillation (2506.06278); Guo et al. on mechanistic unlearning (2410.12949); Hwang et al. on English-only unlearning (2510.23949); Jang et al. on prompt attacks (2506.10236); Lee, Kim and Jo, Unlearning Depth Score (2605.24614); Lynch et al. (2402.16835); Deeb and Roger (2410.08827); Patil et al. (2309.17410); Seyitoglu et al. (2411.02631); Hu et al. (2406.13356); LoRA (2106.09685); linear probes (1610.01644); tuned lens (2303.08112); activation patching practices (2309.16042); attribution patching (2310.10348); EAP-IG (2403.17806); Tiny Aya (2603.11510). Full BibTeX in `paper/references.bib`.
