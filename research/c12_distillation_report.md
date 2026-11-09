# C12 deep-dive — strong-to-weak distillation: what it is, and what it means for nano_SLMs

Written 2026-09-06; read-only analysis (the M3 target run was live and untouched).
Companion: `c12_distillation_plan.md` (the executable plan, user-gated post-M3).
Sources: `research/qwen3_8_flash_next_research.md` (§5.1, §6.3; lines 32, 242-243, 254, 262-263),
`research/crosscheck_flashnext_vs_pipeline.md` (row C12, action item 5), raw Exa payloads under
`research/raw/`, plus repo evidence cited in §4-§5.

## 1. The finding (crosscheck row C12)

> Research doc, lines 262-263: The architecture trick (GDN hybrid) makes small models *cheap to run*;
> the training trick (distill from big teachers) is what makes them *good*.

Qwen small models (0.6B-14B, 30B-A3B) are **not RL-trained and not made good by more pretraining** -
they are **full-parameter distillations of the flagship** (Qwen3.8-2.4T-A95B) into small dense
architectures, trained on ~45k-70k curated **teacher traces**, at **~1/10 the GPU-hours** of reaching
the same capability any other way (research doc lines 242-243, 254, 32). Crosscheck verdict: our
pipeline produces base LMs only - the piece that turns small models *useful* is missing.

## 2. Why distillation beats more pretraining for small models

- Base pretraining loss = one-hot cross-entropy: each position teaches exactly one bit
  (this exact token was next) and nothing else.
- Distillation target = the teacher **full probability distribution over all 32,768 tokens** at every
  position: which alternatives were plausible, how confident, the teacher taste. Same data, orders of
  magnitude more signal per gradient step - the source of the ~1/10 GPU-hour figure (Qwen-reported;
  ours to verify on our own ladder, see plan Tier 3).
- The student also inherits behaviors (fluency, structure, format) a one-hot loss never teaches at
  nano scale.

## 3. Qwen two-part recipe (what strong-to-weak concretely is)

1. **Off-policy trace SFT** - the teacher generates dense reasoning/code traces; the student does
   ordinary supervised fine-tuning (CE) on them. Off-policy = the data comes from the teacher
   distribution. ~45k-70k traces sufficed for the 2B/4B/9B distills.
2. **On-policy logit alignment** - the student generates its own continuations; the loss aligns the
   student logits with the teacher logits on those same tokens (KL). Fixes exposure bias: the student
   trains on its own distribution while being corrected by the teacher.

## 4. The verified gap in our pipeline (2026-09-06)

- scripts/ = 8 files, all stage-1: prepare_data, tokenize_data, train, eval, infer, vram_probe,
  sanity_check, exa_research. **No SFT, no distillation anywhere.**
- Eval = val ppl + 3 code prompts (scripts/eval.py); README says it outright: give the model a code
  prefix, not a question - it was trained on raw code text and will continue the prefix.
- When M3 finishes, T = base code-completer. The gap is a missing **stage**, not a hyperparameter -
  orthogonal to all M4 architecture items (LR sweep, GDN port, GR+RMSNorm, Muon).

## 5. Asset inventory for distillation (verified from configs + run summaries + HANDOFF)

| Rung | Params | Pretraining corpus (verified) | Eval loss -> ppl | Status |
|---|---|---|---|---|
| S - smoke | 12.32M | Evol-Instruct 2k rows (TinyCode had a corrupted shard; fallback chain - HANDOFF §6) | 4.7926 -> 120.6 | done |
| P - pilot | 100.68M | Evol-Instruct 50k rows -> 20.9M tok (HANDOFF §3) | 1.1608 -> 3.19 | done, runs/pilot/final |
| T - target | 226.5M | CodeSearchNet python 150k rows -> 47.9M tok | 8.06 -> 4.61 @ steps 50-150 (live) | M3 in progress, ETA ~2026-09-07 morning |

Key nuances that shape the plan:

- **T never saw Evol-Instruct** (its corpus is CodeSearchNet - target.yaml + HANDOFF §6) so Evol is
  **contamination-free SFT data for T**.
- **S and P share the Evol corpus** (different row counts), so a P->S knowledge-distillation
  comparison needs a same-data from-scratch baseline; the existing smoke checkpoint (2k rows) is not
  a fair control.
- P ppl 3.19 is measured on Evol-domain val; the SFT forgetting guard must use **T CSN val**.
- Vocab 32768 (CodeLlama sentencepiece); the tokenizer ships **no chat template** - an SFT template
  must be defined and kept consistent with inference.
- Constraints: tied embeddings, fp16-only Turing sm_75, 6 GB VRAM, ctx 512 (P) / 1024 (T).

## 6. Three application tiers (details in the plan)

- **Tier 1 - teacher-trace SFT of T on Evol-Instruct**: Qwen off-policy half, using an existing
  public trace corpus instead of generating our own. Hours-scale; changes the capability class
  (code-completer -> instruction-capable).
- **Tier 2 - on-policy logit alignment**: Qwen second half; a local teacher (e.g. Qwen2.5-Coder-1.5B)
  scores the student own samples. VRAM-tight on 6 GB -> options: 8-bit teacher (bitsandbytes is
  installed), CPU teacher, smaller student. Only after Tier 1.
- **Tier 3 - intra-ladder pretraining KD**: our own P (or T) as teacher for a smaller student, KL on
  logits. Tests the ~1/10 claim on our hardware; needs same-data baselines (see §5).

## 7. Priority / sequencing

The crosscheck ordered distillation 5th because the LR sweep / GDN port / GR+RMSNorm / Muon items are
pretraining-side upgrades. C12 is **orthogonal**: it consumes the M3 artifact as-is and needs no src/
changes. Promoting it to first post-M3 action is legitimate if the goal is a useful instruct model;
otherwise keep the crosscheck order. USER DECISION (HANDOFF §8.3).

## 8. Method / how to re-verify

All read-only: config reads, runs/*/final/train_summary.json, HANDOFF/README, EventAccumulator over
runs/target/logs (tags train/loss, train/grad_norm, train/learning_rate), nvidia-smi. M3 snapshot at
write time: step 150, loss 5.60 -> 4.61, healthy; ~20-min telemetry silences are the
logging_steps=50 x 23.7 s/it cadence, not stalls (confirmed: file wrote exactly when step 150 was due).
