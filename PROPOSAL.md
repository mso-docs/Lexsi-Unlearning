# Lexsi Track 2b plan: Unlearning vs. hiding

> Standalone project plan for the Lexsi AI Safety and Alignment Hackathon, Track 2b (AI Safety and Correction). Question: after unlearning, is designated knowledge erased or only suppressed? Initial submission is due Mon Nov 16, 6:59 AM ET. Registration is already confirmed. This plan is only for Lexsi.

- **Track 2b**: AI Safety and Correction (Official problem: unlearning (statement 7))
- **Mon 6:59 AM ET**: Initial submission cutoff (Sun Nov 15, 11:59 PM AoE)
- **Tiny Aya Global**: Primary model (verified) (CohereLabs/tiny-aya-global, 3.35B, CC-BY-NC)
- **Registered**: Hackathon status (Confirmed Nov 12, 2026)

> **Recommended title:** Unlearning vs. Hiding: Does Unlearning Remove Knowledge or Just Suppress Its Expression?

## Title options

1. Unlearning vs. Hiding: Does Unlearning Remove Knowledge or Just Suppress Its Expression? (recommended)
2. Forgotten or Just Hidden? Behavioral Forgetting vs. Probe Recoverability After SafeTune Unlearning
3. Erasure or Gating: A Layer-wise Test of Unlearning in Tiny Aya
4. When Unlearning Leaves a Trace: Relearning Speed and Residual Representations

## One-sentence thesis

After a small open-weights model is taught a synthetic, designated fact set and then unlearned with a Lexsi-supported method, behavioral accuracy on those facts may fall while the same facts remain linearly readable from hidden states, visible under a logit lens, restorable by activation patching, or quick to relearn — which would mean the procedure suppressed expression rather than removing the knowledge.

## Research questions

1. RQ1 — How complete is behavioral forgetting?
   - After unlearning, how far does accuracy drop on the forget set under direct questions, paraphrases, format changes, translations into other Tiny Aya languages, and adversarial or extraction-style prompts? Does retain-set and general capability stay intact?
2. RQ2 — Is the knowledge still internally recoverable?
   - Do layer-wise linear probes trained before unlearning, logit-lens rankings of the correct answer, or activation patching still recover forget-set facts after unlearning? If yes, is residual information concentrated in particular layers (for example late layers), which would look more like gating than erasure?
3. RQ3 — Do circuits and relearning agree with that picture?
   - Does a CircuitKIT comparison of pre- and post-unlearn circuits, and the speed of few-shot or fine-tune relearning relative to learning a matched new fictional set, suggest suppression (fast relearning, late-layer recoverability) or more thorough removal?

## Novelty versus prior work

> **Do not claim to invent unlearning or the observation that it can be shallow:** TOFU, RMU, NPO, and several robustness papers already show that unlearning can look successful on direct prompts while leaving recoverable traces. The weekend contribution is a clean, toolkit-chained measurement on a required Cohere model: inject fictional facts with known ground truth, unlearn with SafeTune, then report behavioral robustness and layer-wise recoverability together. If the result mainly reproduces a known pattern, say so and lead with the Tiny Aya evidence and the erase-versus-suppress localization.

- Maini, Feng, Schwarzschild, Lipton, and Kolter, TOFU (arXiv:2401.06121), introduce fictitious-author QA so a model can be fine-tuned on facts it could not have known from pretraining. That is the injection protocol we will follow, on Tiny Aya rather than the original Llama-2 / Phi setups.
- Li et al., The WMDP Benchmark (arXiv:2403.03218), introduce Representation Misdirection for Unlearning (RMU): steer forget-set activations toward a random vector while pinning retain-set activations to a frozen reference, updating early MLP down-projections. We use RMU as the primary SafeTune method, not the WMDP hazardous-knowledge dataset.
- Zhang, Lin, Bai, and Mei, Negative Preference Optimization (arXiv:2404.05868), give a more stable alternative to gradient ascent, evaluated on TOFU. SafeTune implements NPO as a comparison method.
- Eldan and Russinovich, Who's Harry Potter? Approximate Unlearning in LLMs (arXiv:2310.02238), show strong surface forgetting of a real pretraining domain. Lynch, Guo, Ewart, Casper, and Hadfield-Menell, Eight Methods to Evaluate Robust Unlearning in LLMs (arXiv:2402.16835), then show that this kind of unlearning can still leak under other languages, extraction, relearning, and latent probing. That evaluation style is the template for RQ1–RQ3, using fictional TOFU facts instead of copyrighted books.
- Deeb and Roger, Do Unlearning Methods Remove Information from Language Model Weights? (arXiv:2410.08827), find that fine-tune relearning can recover a large fraction of pre-unlearning accuracy, which they read as evidence that information often remains in the weights. We will run a small, budgeted relearning test in that spirit.
- Patil, Hase, and Bansal, Can Sensitive Information Be Deleted From LLMs? (ICLR 2024, arXiv:2309.17410), show that after targeted knowledge edits, facts can still leak from hidden states and rephrased prompts. Seyitoğlu, Kuvshinov, Schwinn, and Günnemann, Extracting Unlearned Information from LLMs with Activation Steering (arXiv:2411.02631), recover unlearned information by steering. We will use probes, logit lens, and patching as our recovery tools; activation-steering extraction is a stretch, not the main method.

This maps onto official Track 2b statement 7 (unlearning: prove the target is gone under paraphrased and adversarial prompts, and show general capability survives), with statement 9 (locate where the relevant information lives) as the mechanistic half. It is a measurement project, not a new unlearning algorithm.

## Model (verified on Hugging Face)

- **Primary:** CohereLabs/tiny-aya-global — 3.35B instruction-tuned model (SFT and preference training), 36 layers, 8K context, CC-BY-NC 4.0 plus the Cohere Labs Acceptable Use Policy. This is a listed Lexsi model. Access is gated (HF gated=auto): accept the non-commercial form and set HF_TOKEN before loading. CircuitKIT documents a TransformerLens compatibility port for all five Tiny Aya checkpoints.
- **Why this variant:** tiny-aya-global is the balanced instruction-tuned chat model, which is the right starting point for teaching and then unlearning facts. It is also multilingual (70 languages), which lets us test whether English-only unlearning survives translated queries. tiny-aya-base is pretrained only and is a weaker chat subject. Regional variants (earth / fire / water) are the same size; stay on global unless a pre-registered language split is added later.
- **Optional stretch:** CohereLabs/aya-expanse-8b — also listed, also CC-BY-NC 4.0 and gated. About 8B parameters; tight on a 16 GB Colab T4 once fine-tuning and unlearning are stacked. Only try it if the 3.35B pipeline already has numbers, or in the finalist week.
- **Fallback if memory is tight:** Stay on tiny-aya-global. Use LoRA for the injection fine-tune, 8-bit where SafeTune and CircuitKIT allow, and a smaller TOFU slice (for example forget01 plus a compact retain set). Do not switch to a non-Cohere model: Lexsi requires a listed dense open-weights Cohere Labs LLM as the core submission.

## Unlearning method and toolkits (verified APIs)

Lexsi requires a real pipeline that uses at least two official repos from the track; three or more is preferred. Planned chain: SafeTune (unlearn) + AuditKIT (before/after eval) + CircuitKIT (pre/post circuit comparison). CuratorKIT is optional for packaging the TOFU splits.

> **Primary unlearning method: SafeTune RMU:** Use safetune.runner.unlearn.RMUTrainer, which implements Representation Misdirection for Unlearning (Li et al., 2024). It steers residual-stream activations at a chosen layer toward a fixed random control vector on forget data, while keeping retain-set activations close to a frozen reference, and trains only MLP down_proj parameters in a few layers. That is a good weekend fit, and it is mechanistically convenient: if early-layer edits suppress behavior but later-layer probes still decode the fact, that is evidence of hiding rather than erasure. Confirmed call: trainer = unlearn.RMUTrainer(model); unlearned = trainer.unlearn(forget=forget_batches, retain=retain_batches). Defaults through the trainer include layer_id=7, update layers [5, 6, 7], alpha=1200, steering_coeff=20, lr=5e-5, max_num_batches=80. For 36-layer Tiny Aya, treat layer_id as a tunable choice (the default 7 is mid-early, not late); record the layers actually used.

| Toolkit | Role in this project | Confirmed APIs / modules | Caveats |
|---|---|---|---|
| SafeTune | Primary and comparison unlearning, plus optional interpret tools. | safetune.runner.unlearn.RMUTrainer (primary); NPOTrainer (variant npo_grad_diff or npo_KL) and GradientAscentTrainer as comparisons; low-level rmu_unlearn / npo_unlearn / gradient_ascent_unlearn. Docs also list FLATTrainer, GradDiffTrainer, and SimDPOTrainer. forget and retain batches are tokenized dicts with input_ids, attention_mask, and labels. | Do not use unlearn.load_unlearn_data as the forget set; that helper is a generic loader, not the TOFU protocol. Build TOFU forget/retain batches ourselves. RMU’s default layer 7 may need retuning on a 36-layer model. NPO is a fuller-parameter, slower second method — run it if Friday RMU finishes early. |
| AuditKIT | Before/after scoring of forget-set and retain-set answers, plus a general-capability split. | auditkit.evaluate; auditkit.Sample; exact match and F1; compare_models() across the base, fine-tuned, and unlearned checkpoints. | Core library is general evaluation. We will wrap TOFU-style QA (and translations) as Sample objects rather than expecting a built-in TOFU runner. |
| CircuitKIT | Compare a forget-fact circuit before and after unlearning (RQ3), if the Tiny Aya port is stable. | circuitkit.load_model; discover_circuit / Pipeline.discover (default eap-ig); official Tiny Aya port in circuitkit.backends._tl_compat. 36-layer, d_model 2048 architecture is documented. | Tiny Aya support is experimental. Run a smoke test (and the parity check if time allows: CIRCUITKIT_RUN_TINY_AYA=1). Discovery metrics often assume single-token answers; CircuitKIT warns that Cohere’s 262k vocabulary can split names. Prefer a custom single-token or tightly tokenized fact task, or treat failed discovery as a reported limitation and keep probes plus logit lens. |
| CuratorKIT (optional) | Only if we want a documented export of the fine-tune and forget/retain splits. | Official data-curation pipeline at https://curatorkit.lexsi.ai/latest/ and https://github.com/Lexsi-Labs/CuratorKIT. | Skip if Thursday is tight. SafeTune + AuditKIT already satisfy the two-toolkit rule; CircuitKIT is the preferred third. |

The knowledge-injection fine-tune can use a standard Hugging Face LoRA / SFT loop. AlignTune is a Lexsi toolkit but belongs to another track; it is not required for a valid 2b chain.

## Datasets (verified)

| Resource | Role | License | Access |
|---|---|---|---|
| TOFU (Maini et al., 2024) | Synthetic ground truth. 200 fictitious authors, 20 QA pairs each. Fine-tune so the model actually knows the facts, then unlearn a declared forget split (start with forget01; forget05 only if injection is clearly successful). Use the matching retain split, plus world_facts and real_authors as non-fictional retain checks. Official perturbed splits support paraphrase-style eval. | MIT, ungated | https://huggingface.co/datasets/locuslab/TOFU (e.g. load_dataset("locuslab/TOFU", "forget01")). Paper: arXiv:2401.06121 |
| TOFU-para (optional) | Five fact-checked paraphrases of TOFU QA pairs, same split names and row order, if official perturbed splits are not enough. | MIT, as TOFU | https://huggingface.co/datasets/sungjuncho/TOFU-para |

Additional evaluation formats will be constructed this weekend and documented in the repo, not claimed as a published benchmark: cloze or true/false rewrites of forget-set facts; a small set of translations of forget questions into two or three Tiny Aya languages (TOFU itself is English-only); extraction prompts (for example, asking the model to tell everything it knows about a forget-set author); and a matched new fictional author used only for the relearning comparison. Do not use WMDP or other hazardous-knowledge corpora as the forget set. Do not unlearn real private or copyrighted material.

## Method

1. **Inject designated facts.** LoRA-fine-tune tiny-aya-global on a compact TOFU slice that includes the forget authors and a retain set. Confirm, before unlearning, that forget-set accuracy is high and that the original base model was near chance — otherwise there is no ground truth. Save this fine-tuned checkpoint as the pre-unlearn reference.
2. **Unlearn with SafeTune RMU.** Build tokenized forget and retain batches from the TOFU splits. Run RMUTrainer.unlearn. If that finishes early and the model is still coherent, run NPOTrainer (npo_grad_diff or npo_KL) on the same splits as a comparison. Keep gradient ascent only as a short baseline if there is time; it is known to collapse.
3. **Behavioral evaluation (RQ1).** Score forget-set accuracy with AuditKIT (exact match / F1, plus a pre-declared looser match if needed) on: direct TOFU questions; official perturbed or TOFU-para paraphrases; format changes; a small translation set; and extraction-style prompts. Score the retain set, world_facts / real_authors, and a pre-declared general-capability split the same way, before and after unlearning.
4. **Mechanistic recovery (RQ2).** On the fine-tuned model, train layer-wise linear probes (logistic or mass-mean) to distinguish correct forget-set answers or author identity from residuals at the last prompt token. Freeze the probes and evaluate them on the unlearned model. Apply a logit lens (unembedding of each layer’s residual) and report the rank of the correct answer token. Activation-patch residuals from the fine-tuned model into the unlearned model at selected layers, and the reverse, to see where forget-set answers return.
5. **Circuits and relearning (RQ3).** If CircuitKIT loads Tiny Aya, discover a forget-fact circuit before and after unlearning and compare node sets. Then relearn: few-shot prompting and a short fine-tune on a few forget facts, compared with learning a matched new fictional author for the same number of steps. Faster recovery of old facts than acquisition of new ones is evidence that the information was suppressed, not gone.

## Metrics and baselines

| Quantity | What it answers |
|---|---|
| Forget-set accuracy by query format | Direct, paraphrase, format-change, translation, and extraction prompts. The main behavioral forgetting number. |
| Probe recoverability by layer | AUROC or accuracy of frozen pre-unlearn probes on the unlearned model. High late-layer recoverability with low behavioral accuracy suggests gating. |
| Logit-lens correct-token rank | Whether the answer remains linearly decodable from intermediate residuals after the final answer has changed. |
| Relearning speed | Steps or shots to recover forget-set accuracy, versus a matched new fictional set. |
| Retain-set and general capability (AuditKIT) | Whether unlearning destroyed unrelated TOFU authors, world facts, or a pre-declared capability split. |
| Before/after deltas | Lexsi requires more than one metric. At minimum: forget-set accuracy and retain/capability, plus at least one recovery metric (probe, logit lens, or relearning). |

Baselines and references, all scored the same way: (1) the original tiny-aya-global, which should not know TOFU facts; (2) the TOFU-fine-tuned model, which should; (3) RMU-unlearned; (4) NPO-unlearned, if run; (5) a brief gradient-ascent run only if there is spare compute. Do not claim success from a drop in direct-question accuracy alone.

## Schedule (America/New_York)

| When (ET) | Focus | Done when |
|---|---|---|
| Thu Nov 12, today | Accept the Hugging Face gate for CohereLabs/tiny-aya-global and set HF_TOKEN. Install SafeTune, AuditKIT, and CircuitKIT. Confirm the $20 HF ZeroGPU credit if not already redeemed (first 800 teams; sent to the team lead). Download TOFU. Smoke-load the model, confirm the base model does not already know a few forget-set facts, and start LoRA SFT on a compact TOFU slice that includes forget01 (or a clearly declared author set). | A training script is running or finished; a note of base-model forget-set accuracy (should be low). |
| Fri Nov 13 | Finish the injection fine-tune and verify high forget-set accuracy plus intact retain-set accuracy. Tokenize forget/retain batches and run RMUTrainer. Collect first after-unlearn direct and paraphrase scores with AuditKIT. If RMU is stable and there is time, start NPO on the same splits. | Three checkpoints exist (base, fine-tuned, RMU). A first before/after table for direct forget-set and retain-set accuracy. |
| Sat Nov 14 | Layer-wise probes, logit lens, and a small activation-patching sweep (RQ2). Format-change, translation, and extraction prompts (RQ1). CircuitKIT discovery before vs after if the Tiny Aya port works (RQ3). Start a short relearning run late in the day. | At least one recovery plot (probe AUROC or logit-lens rank by layer) plus broader behavioral numbers. |
| Sun Nov 15, morning | Finish relearning comparisons. Freeze numbers around midday. No new forget-set authors after noon. | Results are stable enough to write against. |
| Sun afternoon and evening | Write the HF article or GitHub README (abstract under 500 words). Create public, un-gated Hugging Face model and dataset collections (LoRA adapters, unlearned checkpoints if license allows, probes, logs, generations, translation prompts). Reproducibility notes, including RMU layer_id and the exact TOFU splits. Submit on https://hackathon.lexsi.ai/ with the registered email, Track 2b, statement 7 (and 9 if circuits/probes are included), and the toolkits used. Aim for Sunday evening, not the Monday-morning cutoff. | Submission accepted. Hard cutoff: Mon Nov 16, 6:59 AM ET. |
| Mon Nov 16 | Finalists (up to 20 teams) are announced. If shortlisted, a further $200 in HF credits is available to prove the thesis. | If not shortlisted, the initial submission is the end of the event. If shortlisted, start the finalist plan below. |
| Tue Nov 17–Sun Nov 22, if shortlisted | Add error bars or a second seed, complete NPO if it was skipped, tighten translations or CircuitKIT figures, and record a short video or slides. Optionally try aya-expanse-8b only if compute is comfortable. Do not expand the research question. | Video or slides submitted by Mon Nov 23, 6:59 AM ET (Sun Nov 22, 11:59 PM AoE). |

## Compute estimate

- Hardware: Google Colab or Kaggle (T4/L4, 16 GB), and/or the $20 Hugging Face ZeroGPU credit. tiny-aya-global in bf16 is roughly 7 GB of weights. The tight steps are LoRA SFT, RMU (usually lighter, because only a few MLP down_proj tensors train), and CircuitKIT discovery.
- Budget: Thursday start the injection fine-tune; Friday finish it and run RMU (often minutes to a modest number of batches, capped at max_num_batches=80 unless changed); Saturday probes and generations are mostly forward passes; relearning is a short extra fine-tune.
- Keep the forget set weekend-sized: forget01 first. Expand to forget05 only if injection quality is clearly high and RMU has already produced a coherent model.
- NPO and CircuitKIT are the first things to drop if the GPU queue is slow. SafeTune RMU plus AuditKIT already make a valid two-toolkit submission.
- aya-expanse-8b is a finalist-week or late stretch only.

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| The model does not actually learn the TOFU facts (low post-SFT forget-set accuracy). | Train longer or on a slightly larger slice, check the chat template, or shrink to a handful of very distinctive fictional facts constructed and documented this weekend. Do not unlearn until injection is verified. |
| RMU destroys retain-set or general capability, or default layer_id=7 is a poor fit for 36-layer Tiny Aya. | Lower steering_coeff, retune layer_id, or switch to NPOTrainer with npo_KL. Report the failed setting. |
| CircuitKIT’s Tiny Aya port fails, or forget-set names split into multiple tokens. | Skip circuit comparison. Keep probes, logit lens, patching, and AuditKIT. That is still SafeTune + AuditKIT. |
| Gated weights will not download. | Confirm the HF license form is accepted on CohereLabs/tiny-aya-global and that HF_TOKEN is in the runtime. This is a setup bug, not a reason to switch models. |
| Translations or extraction prompts are messy to score. | Pre-declare a small scoring rule, spot-check by hand, and treat those formats as qualitative if automatic F1 is unreliable. |
| Time runs out Sunday. | Minimum viable submission: verified injection, RMU, direct plus paraphrase forget-set accuracy, retain-set accuracy, and one recovery metric (even a single-layer probe or logit lens). Public HF collections and a README. Leave circuits and relearning shallow rather than submitting nothing. |

## What a null result means

A useful null is that RMU (or NPO) lowers behavioral accuracy and also collapses probe, logit-lens, and relearning recovery to the original base-model baseline. That would look more like removal than hiding, at least for this synthetic set and these tests. Another useful null is that behavioral accuracy barely moves; then unlearning failed, and that should be reported as a negative method result, not stretched into a theory of residual knowledge. A mixed result — surface forgetting with late-layer recoverability or fast relearning — is still a complete Track 2b paper if the measurements are clean. Do not claim that a dropped direct-QA score means the fact is gone.

## Ethics, dual-use, and write-up notes

- Follow the Cohere Labs Acceptable Use Policy and the CC-BY-NC terms. This is a research submission, not a product.
- Use only synthetic, designated TOFU (or similarly documented fictional) facts. Do not unlearn real private data, and do not use WMDP or other hazardous-knowledge corpora as the forget set.
- A finding that unlearning merely hides knowledge could be misused to over-claim that a model has forgotten sensitive material, or to recover information from an unlearned model. Report aggregate recoverability, not a how-to for extracting private facts from production systems. Release evaluation code and fictional-fact artifacts.
- Public Hugging Face collections must be un-gated, even though the base Tiny Aya weights are gated by Cohere. Do not upload the base weights; upload LoRA adapters, unlearned deltas or checkpoints as the license allows, probes, logs, and generations.
- Attribute TOFU (MIT) and the toolkits in the README. Mark any translations or extra prompts as created during the hackathon.

## Submission packet checklist

- [x] Registered for Lexsi (already confirmed) (Status as of Thu Nov 12, 2026)
- [ ] HF license accepted for CohereLabs/tiny-aya-global; HF_TOKEN works in the runtime
- [ ] Abstract under 500 words
- [ ] Public, un-gated Hugging Face model collection (injection LoRA, unlearned checkpoint or adapter, logs)
- [ ] Public, un-gated Hugging Face dataset collection (TOFU splits used, translations, generations, probe labels, metrics)
- [ ] Write-up: Hugging Face article, or a GitHub README if HF articles are unavailable
- [ ] GitHub repo or notebooks with reproducibility notes; work done during the hackathon; mark the injection fine-tune as part of that work
- [ ] Name the Cohere model, Track 2b, problem statement 7 (and 9 if applicable), and at least two official toolkits used together
- [ ] More than one before/after metric (at minimum: forget-set accuracy and retain/capability, plus a recovery metric if possible)
- [ ] Submit with the registered email by Mon Nov 16, 6:59 AM ET
- [ ] If shortlisted Nov 16: video or slides by Mon Nov 23, 6:59 AM ET

## Citations: verified and not verified

| Paper or resource | Status | Link |
|---|---|---|
| Maini, Feng, Schwarzschild, Lipton, and Kolter, TOFU: A Task of Fictitious Unlearning for LLMs, arXiv:2401.06121 | Verified; dataset MIT | https://arxiv.org/abs/2401.06121 and https://huggingface.co/datasets/locuslab/TOFU |
| Li et al., The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning (includes RMU), arXiv:2403.03218 | Verified as the RMU method paper. We are not using the WMDP dataset. | https://arxiv.org/abs/2403.03218 |
| Zhang, Lin, Bai, and Mei, Negative Preference Optimization, arXiv:2404.05868 | Verified; implemented in SafeTune as NPOTrainer | https://arxiv.org/abs/2404.05868 |
| Eldan and Russinovich, Who's Harry Potter? Approximate Unlearning in LLMs, arXiv:2310.02238 | Verified | https://arxiv.org/abs/2310.02238 |
| Lynch, Guo, Ewart, Casper, and Hadfield-Menell, Eight Methods to Evaluate Robust Unlearning in LLMs, arXiv:2402.16835 | Verified | https://arxiv.org/abs/2402.16835 |
| Deeb and Roger, Do Unlearning Methods Remove Information from Language Model Weights?, arXiv:2410.08827 | Verified | https://arxiv.org/abs/2410.08827 |
| Patil, Hase, and Bansal, Can Sensitive Information Be Deleted From LLMs?, ICLR 2024, arXiv:2309.17410 | Verified (hidden-state / rephrase leakage after knowledge edits) | https://arxiv.org/abs/2309.17410 |
| Seyitoğlu, Kuvshinov, Schwinn, and Günnemann, Extracting Unlearned Information from LLMs with Activation Steering, arXiv:2411.02631 | Verified; cited as related recovery work, not as the primary method | https://arxiv.org/abs/2411.02631 |

> **Citation caveats:** SafeTune also lists FLAT, GradDiff, and SimDPO unlearning trainers; they were not selected as primary methods and were not re-read in full for this rewrite. A later conference venue for TOFU, NPO, Lynch et al., and Deeb and Roger was not independently confirmed beyond the arXiv versions cited above. Tuned Lens (Belrose et al.) was not confirmed to provide a Tiny Aya pretrained lens, so the plan uses the standard unembedding logit lens. CircuitKIT Tiny Aya support remains experimental. No planned citation was left unverified and then treated as real.

---

- [Lexsi AI Safety and Alignment Hackathon](https://hackathon.lexsi.ai/) (hackathon.lexsi.ai)
- [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (Hugging Face)
- [TOFU dataset](https://huggingface.co/datasets/locuslab/TOFU) (Hugging Face)
- [SafeTune unlearn guide](https://safetune.lexsi.ai/user-guide/unlearn/) (safetune.lexsi.ai)
- [SafeTune RMU](https://safetune.lexsi.ai/user-guide/unlearn/rmu/) (safetune.lexsi.ai)

## Sources

- <https://hackathon.lexsi.ai/>
- <https://huggingface.co/CohereLabs/tiny-aya-global>
- <https://huggingface.co/datasets/locuslab/TOFU>
- <https://safetune.lexsi.ai/user-guide/unlearn/>
- <https://safetune.lexsi.ai/user-guide/unlearn/rmu/>
- <https://safetune.lexsi.ai/user-guide/unlearn/npo/>
- <https://circuitkit.lexsi.ai/advanced/tiny-aya/>
- <https://auditkit.lexsi.ai/>
- <https://arxiv.org/abs/2401.06121>
- <https://arxiv.org/abs/2403.03218>
