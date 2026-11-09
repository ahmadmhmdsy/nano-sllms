# Qwen's "Strong-to-Weak Distillation" (why distilled small models beat both RL and pretrain-from-scratch)

Sources:
- Qwen3 Technical Report, arXiv:2505.09388 (May 2025)
- alphaXiv overview of the same report
- Philipp Schmid's breakdown thread (x.com/_philschmid)
- DistilQwen2.5 paper, arXiv:2504.15027 (industrial distillation practice)
- Qwen3.5-4B quantization-aware distillation paper, arXiv:2607.04244

## The 4-stage pipeline Qwen uses for its FLAGSHIP models (the "teachers")

Pre-training:
1. **General stage (S1)**: broad web-scale pretraining.
2. **Reasoning stage (S2)**: STEM/code/reasoning-heavy data mix.
3. **Long-context stage (S3)**: hundreds of billions of tokens specifically formatted for long sequences, up to 32,768 tokens, to teach long-range dependency handling.
Total pretrain corpus: 36 trillion tokens across 119 languages/dialects. Data curation includes multimodal augmentation — e.g. extracting text from PDFs — and synthetic data generation, with instance-level data-mixture optimization (i.e. they tune the *mixture ratio* of data types per training instance/stage, not just once globally).

Post-training (applied to the big flagship models only):
1. **Long-CoT Cold Start**: SFT on a curated set of hard, multi-step reasoning problems with full chain-of-thought — this "warms up" the reasoning behavior before RL.
2. **Reasoning RL**: RL (GRPO) on problems with *verifiable* answers (math/code where you can check correctness automatically) — this is what sharpens reasoning without needing a human reward model.
3. **Thinking-mode fusion**: SFT again to merge "thinking" (long CoT) and "non-thinking" (fast/direct) behavior into one model, controlled via special chat-template tokens (`/think`, `/no_think`).
4. **General RL**: broad RL pass over instruction-following, agentic tool-use, and preference alignment, using varied reward signals (not just verifiable-math reward).

## Why small models DON'T get this same 4-stage treatment

Qwen explicitly says: running the full 4-stage pipeline separately for every small model size is wasteful. Instead:

> "Preliminary experiments suggest that directly distilling the output logits from teacher models into lightweight student models can effectively enhance their performance while maintaining fine-grained control over their reasoning processes. This approach eliminates the necessity of performing an exhaustive four-stage training process individually for every small-scale model."

And critically:

> "Distillation from advanced teacher models significantly outperforms reinforcement learning in performance **and** training efficiency."

Concretely: Qwen3-235B-A22B and Qwen3-32B (the trained flagships) are the **teachers**. Everything from 14B down to 0.6B is a **student** that is distilled off the flagships instead of being pushed through RL individually.

## The actual student-training mechanics ("off-policy + on-policy distillation")

Per Philipp Schmid's breakdown of the report: instead of the full 4-stage RL process, smaller Qwen3 models learn by distilling knowledge from the larger ones using **off-policy distillation** (train on teacher-generated outputs/logits for fixed data — cheap, parallelizable) followed by **on-policy distillation** (student generates its own rollouts, teacher scores/corrects them — closes the train/inference distribution mismatch, a.k.a. exposure bias).

Measured effect: logit-level distillation gives students **better Pass@1** (immediate correctness) **and better Pass@64** (exploration/diversity of correct solutions) than training the same small model with the 4-stage RL pipeline directly — i.e. distillation is not just cheaper, it's actually a *better optimization target* for a small model than trying to teach it to reason from scratch via RL.

## Why this "beats pretraining" intuition holds in general

Raw pretraining forces a model to infer structure/knowledge/reasoning patterns from raw, noisy, unstructured internet tokens — the learning signal per token is weak (most of the objective is just "guess the next token", not "reason correctly"). Distillation instead gives the student a signal that already encodes the teacher's:
- token-probability distribution (not just single sampled tokens — logits carry information about "how confident/what alternatives" the teacher considered, which is strictly more information than a hard label), and/or
- reasoning trace (explicit intermediate steps), and/or
- final corrected answer.

This is a **denser, cleaner, already-organized supervision signal**, so the student converges to higher performance per training token/per parameter than pretraining alone would achieve at that same small size. The tradeoff: the student is capped near the teacher's own ceiling — you cannot distill your way past the teacher.

## Industrial-scale small-model distillation recipe (DistilQwen2.5, Alibaba)

From arXiv:2504.15027 and the referenced classic methods it builds on:
- **Hinton et al. 2015 (the original "Distilling the Knowledge in a Neural Network")** — soft-label KL-divergence distillation is the foundation.
- **"Distilling Step-by-Step" (Hsieh et al. 2023)** — extract not just the teacher's final answer but its intermediate rationale/reasoning steps as *extra* supervision signal — lets a much smaller model outperform a larger one trained on final answers alone, with less data.
- **MiniLLM (Gu et al. 2024)** — uses reverse-KL instead of forward-KL for distillation, better suited to generative (not just classification) tasks because it avoids the student "hedging" over teacher modes it can't reach.
- DistilQwen2.5's own added step: **quantize the distilled student afterward** to shrink memory footprint / inference latency further, for on-device serving.

## Concrete example pipeline you can replicate

From a knowledge-distillation experiment paper (arXiv:2603.13765) using Qwen models directly:
1. Take a bigger already-fine-tuned Qwen model as teacher (e.g. Qwen2.5-1.5B) and a smaller one as student (e.g. Qwen2.5-0.5B).
2. SFT the teacher first on your target domain/dataset (e.g. code — they used the `pytorrent` dataset).
3. Run knowledge distillation from teacher → student on that same domain data.
4. Layer RL fine-tuning on top of the distilled student to further push it toward valid, guided chain-of-thought completions for complex tasks (their example: coding).
5. Quantize the final distilled student for deployment.

## Quantization-Aware Distillation (QAD) — useful if you eventually want INT4 students

From arXiv:2607.04244 (applied to Qwen3.5-4B):
- Generate training data by re-prompting the **original full-precision teacher** with a large sample set (they used 220K prompts) to get **teacher-regenerated conversations** (not just static existing text — freshly sampled from the teacher).
- Pack these into fixed-length sequences (they used 16,384 length) for efficient batching.
- **Initialize the student from the dequantized weights of an INT4 (AWQ) checkpoint** of the *same* model family — i.e. student and teacher share architecture; only precision differs.
- Distill using **per-token forward-KL** between frozen BF16 teacher and the quantized-then-dequantized student, updating only the dequantized weights while keeping per-group quantization scales frozen.
- Optimizer: 8-bit AdamW, lr 2e-6, cosine schedule, 3% warmup, grad clip 1.0, batch size 16, ~8000 steps (~4.5 epochs).

This is directly relevant if your eventual goal is a small quantized model that still tracks a stronger teacher closely — same recipe scales down.

## Practical takeaway for your own training pipeline

1. Pick or fine-tune the strongest model you can access as a "teacher" for your domain.
2. Prefer **teacher-generated fresh data** (re-prompt the teacher on your domain prompts) over static scraped data — this is what "on-policy"/"teacher-regenerated" data means and it consistently outperforms static distillation sets.
3. If you have teacher **logits/weights**: do forward-KL (matching output distribution) — this is the strongest, cheapest-to-implement signal.
4. If you only have teacher **text** (closed API): capture step-by-step rationales, not just final answers (Distilling Step-by-Step) — this alone recovers much of the benefit of logit-level distillation.
5. Consider mixing "hard labels" (ground truth answers) with "soft labels" (teacher outputs) — literature suggests the gain comes partly from reducing train/inference mismatch (exposure bias), not purely from "matching the teacher better," so on-policy rollouts scored/corrected by the teacher help even more than pure static SFT.
6. Quantize/distill jointly if the deployment target is memory constrained (QAD approach above).
