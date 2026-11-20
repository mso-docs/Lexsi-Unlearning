# Unlearning vs. Hiding: Does Unlearning Remove Knowledge, or Just Suppress Its Expression?

Mackenzie O'Brien, Mac Labs AI  
November 2026

Lower accuracy on direct questions does not show that designated knowledge is gone. I audited Representation Misdirection for Unlearning (RMU) on Tiny Aya, Cohere Labs' 3.35B multilingual model, in a solo entry for Track 2b of the Lexsi AI Safety and Alignment Hackathon. With a fix to the forget loss, RMU stopped the model from stating 40 injected fictitious-author facts under direct questions, paraphrases, extraction prompts, and translations into eight languages. The same audit does not show that the facts were erased.

This is a measurement study, not a new unlearning method. The facts come from the Task of Fictitious Unlearning (TOFU) `forget01` split. I taught them to the model with Low-Rank Adaptation (LoRA), then unlearned with SafeTune's RMU trainer at decoder layer 18, updating layers 16-18, and scored the result with AuditKIT plus a helper judge. I repeated the protocol on Tiny Aya Earth, Fire, Water, and Base, and ran one transfer of the same settings to Aya Expanse 8B. Code, paper, slides, a walkthrough video, and the Hugging Face artifacts are collected at [mso-docs.github.io/Lexsi-Unlearning](https://mso-docs.github.io/Lexsi-Unlearning/).

## Standard RMU did nothing until the forget loss was masked

Out of the box, RMU left the injected facts in place. A 360-token system preamble, plus attention-sink tokens, dominated the forget loss, which stayed flat near 3.5. Four unmasked settings left forget ROUGE-L (answer overlap) at 0.92-0.97, against 0.97 after injection.

Masking the forget loss to first-user-turn tokens changed that. With alpha 100, steering coefficient 20, and 80 steps, forget ROUGE-L fell to 0.32. Every later result uses that masked recipe.

## Output-level suppression held under several prompt types, at a modest retain cost

A helper judge (`qwen3.5:35b`, with a seeded 15% re-grade by Kimi K3 on the main Global run) scored whether an output still revealed the injected fact. On Tiny Aya Global, the mean of seeds 42-44 for judge forget suppression rose from 0.03 to 0.92 on direct questions (n = 40 per seed), from 0.33 to 0.95 on paraphrases, and from 0.08 to 0.95 on extraction prompts. Across eight machine-translated languages it rose from 0.71 to 0.93. Earth, Fire, and Water, each run once, behaved alike (direct suppression after RMU of 0.875, 0.875, and 0.975).

Judged accuracy on retained facts fell 1.6 to 9.4 points (Global mean 0.49 to 0.44). Format changes leaked: the pooled cloze and true/false bucket reached only 0.79 after RMU, driven by true/false (0.70).

## The same settings failed on Base and only partly worked on a larger model

Tiny Aya Base is not instruction-tuned. After a prompt-format bug in my evaluation code was fixed, injection succeeded, but the same RMU settings moved direct suppression only from 0.00 to 0.10 and left forget ROUGE-L at 0.913. That is a failure of the transferred recipe, not a missing result.

On Aya Expanse 8B, run once with no retuning, direct suppression reached only 0.20, and judged retain accuracy fell from 0.91 to 0.67. The retain loss and the forget gain were about the same size, so I make no cross-model claim. Model size is confounded with the layer index, the mask start, and the absence of any sweep.

## What this does not show

I do not claim erasure. After 30 fine-tuning steps the RMU model recovered 5, 7, and 7 of 8 forgotten facts across Global seeds 42-44, against 4, 4, and 6 of 8 new facts on the same model. The gap is one to three facts at n = 8, too small to call significant, and relearning was not consistently faster than learning from scratch.

Linear probes that try to separate forget from retain questions reach AUROC 0.92-1.00 from layer 1 up, but a never-injected base model does as well. Those probes reflect surface differences between question topics, not stored knowledge. A teacher-forced logit lens on Global seed 42 (n = 40) shows RMU lowering the first content-token probability from 0.971 to 0.696, which is a drop, not a removal. Single-site activation patching, on 10 facts of one author, restores only part of that first-token gap and does not localize an internal representation.

The translations were written and graded by the same helper model, and no native speaker has signed off on them. Only Global has more than one seed. Alpha and the learning rate were set on the same 40 forget questions that were later scored. Cohen's kappa is uninformative when labels are one-sided, so the paper reports raw and prevalence-robust agreement instead.

The practical lessons are narrower, and I think they travel. Check that the forget loss actually moves. Evaluate in the same prompt format used at training. Score paraphrase, extraction, translation, and format-change prompts, not just direct questions. A drop in direct-question accuracy is not, by itself, evidence that the knowledge is gone.

Paper: [Unlearning vs. Hiding (PDF)](https://mso-docs.github.io/Lexsi-Unlearning/downloads/Lexsi-Unlearning-paper.pdf)  
Code: [github.com/mso-docs/Lexsi-Unlearning](https://github.com/mso-docs/Lexsi-Unlearning)  
Demo, dataset, and checkpoints: [Hugging Face collection](https://huggingface.co/collections/mackenzietechdocs/lexsi-unlearning-vs-hiding-6af7ae6b51c779a9e34418c6)
