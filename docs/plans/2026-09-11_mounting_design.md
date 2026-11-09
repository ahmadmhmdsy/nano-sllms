# Mounting — design (frozen teacher wired into a from-scratch student, progressive unlink)

Date: 2026-09-11 · Status: **USER-APPROVED 2026-09-11 (design v1)** · Spec owner: this doc
· Follow-up (implementation): pending user sign-off on this spec.

## 1) Idea

"Mounting": take already-trained (knowledgeable) models and wire them into new
(empty, random-init) models. The mounted models are FROZEN — inference only during
training — and the mount touches every layer of the training model. Training first
forces the student to get used to the mounted knowledge (Stage 1), then the wiring
is progressively broken (Stage 2, ~10% increments), increasing until the student is
fully independent and can produce the same result by itself (Stage 3).

Prior art mapped (KT-ladder Route 5 "model stitching" = cousin, was PARKED 2026-09-10
as low-ROI without mechanisms; mounting re-opens it with a progressive-unlink design):
- BERT-of-Theseus (EMNLP 2020, aclanthology.org/2020.emnlp-main.633) — progressive
  module replacing, closest published match for Stages 2-3.
- Flamingo gated cross-attention with zero-init tanh gate (arxiv 2204.14198) — the
  Stage-1 mechanism ("get used to the info" without early instability).
- LayerDrop / Progressive Layer Dropping (arxiv 1909.11556; NeurIPS 2020) and DARE
  rescale (arxiv 2311.03099) — safe progressive dropping + survivor rescaling.
- Annealing KD (EACL 2021, aclanthology.org/2021.eacl-main.212) — annealing the
  teacher's influence improves final quality (supports Stage 3 direction).
- StitchLLM (aclanthology.org/2025.acl-long.1305) / model stitching (PMLR v322).

## 2) Frozen decisions (user, 2026-09-10/11)

1. First test scale: **pilot dims** (12L/d768/12Q/4KV, 32k CodeLlama vocab,
   ~110M, random init, fp16, 8-bit Adam b1/a32 per Milestone-B default).
2. First mounted (teacher) model: **SmolLM2-135M**; SmolLM2-360M only as a
   follow-up arm if the mount shows a lift (Q4-ladder logic).
3. Unlink mechanisms: test ALL THREE (A/B/C) in separate arms, order A -> B -> C,
   "we cannot confirm without testing" — deliverable includes a pros/cons map per
   mechanism for later use cases.
4. Success bar (user picked ALL THREE): (i) beat/match from-scratch at equal
   WALL-CLOCK GPU time, (ii) final full-independence student-only quality close
   to the mount-on model, (iii) research evidence: measurable effect vs the
   KD-fill-in control.
5. Execution discipline: strict single-GPU sequential arms; zero-flag auto-resume
   contract untouched; mounting is additive + config-gated, DEFAULT OFF (same
   style as init_from / init_embeddings).

## 3) Physics / mismatch facts (from the KT ladder, KT plan §3)

- Teacher = SmolLM2-135M: LlamaForCausalLM, 30L/d576 (9Q/3KV), vocab 49,152
  (Llama-3 BPE), bf16-trained. Ours: 12L/d768, vocab 32,768 (CodeLlama SP).
- No cross-tokenizer weight merging or naive positional/lens logit KD (proven).
- Tokenizer mismatch BREAKS positional per-layer wiring -> the mount is
  CROSS-ATTENTION: each student layer attends to the frozen teacher's full
  re-tokenized sequence (teacher tokenizes the same raw text with its own
  tokenizer; ragged alignment is handled natively by cross-attention KV).
- Layer mismatch (30 vs 12): per student layer l, anchor teacher layer(s)
  a(l) = round((l+0.5)/12 * 30); bridge option adds the teacher's embedding
  layer as an extra source (A/B test this knob, default simple-anchor).

## 4) Arms (strictly sequential on the single GPU, order = user's A->B->C)

| # | Arm | Mechanism | Isolates |
|---|---|---|---|
| 1 | Control | from-scratch, matched steps/data/seed | cost/quality floor |
| 2 | KD-fill-in | kd.py scaffold + annealed distillation weight, NO wiring | supervision-only |
| 3 | Mount A | zero-init per-layer tanh gates, anneal 1->0 (-10 pct per ckpt interval), pause-on-cliff | smooth unlink |
| 4 | Mount B | seeded deterministic wire-severance 10->30->... rescaled by 1/(1-p), mask checkpointed | literal "break wires" + compute-shrink per stage |
| 5 | Mount C | A + B combined | do mechanisms stack |

Arm 2 reuses scripts/kd.py scaffolding (frozen-teacher custom compute_loss,
partial-ckpt guard, resume). Mount arms build on the same trainer with the
bridge module config-gated.

## 5) Mount bridge (all mount arms)

Per student layer: h <- h + g_l * tanh(W_v . CrossAttn(Q = h_student,
KV = teacher_states[a(l)])), g_l zero-init scalar per layer (identity at step 0).
- Teacher: frozen, inference-only, no_grad, bf16-origin weights cast for fp16
  inference; teacher forward cached per step for ALL student layers to share.
- Trainable-but-discarded: bridge projections + pre/post-norms + gates. The
  teacher stack itself is never trained (user hard requirement).
- Independence: at schedule end ALL bridge params are deleted; student-only
  eval is the reported number.

## 6) Stages (mount arms)

1. Warm-in (~100-200 steps): gates active, student absorbs mount.
2. Unlink: A: gate -10 pct per interval; B: severance +10 pct per stage with
   DARE rescale; C: both. Cliff rule: eval-jump between checkpoints beyond a
   threshold PAUSES the schedule for a recovery window, then resumes.
3. Independence: bridges discarded; student-only eval +- generation sanity
   (10 stub prompts, eval.py path).
4. OPTIONAL post-A/B Stage 4: SFT of the winner (in-repo sft.py), user-gated.

## 7) Data plumbing requirement (dual-tokenized stream)

Same raw CSN pilot text -> two packed shard streams: the existing 32k student
stream (UNTOUCHED) plus a SmolLM2-tokenized teacher stream, index-aligned by
document order. Teacher blocks consume the teacher stream; cross-attention is
permissive over ragged teacher length (padded KV + mask). No retroactive
changes to data/pilot/tokens.

## 8) Gates & report per arm (the pros/cons map)

- tokens/s + wall-clock GPU-hours to hit Arm-1 eval-curve anchor points.
- final student-only eval loss (bridges deleted) + mount-on eval (Stage 1 end).
- step-pace profile per stage (does B's compute actually shrink?).
- cliff count / pause cost; fp16 stability (grad-norm); VRAM peak.
- generation sanity after independence.
- Ending artifact: pros/cons table of A/B/C mapped to later use cases
  (speed vs stability vs quality; S/P/T-scale transfer prediction).

## 9) Known risks (carried honestly)

1. Teacher forward may dominate pace (Tier-3 precedent ~10x step cost at
   S-scale) -> wall-clock is the primary success metric, not steps.
2. B/C stochasticity + zero-flag auto-resume: severance masks must be
   derivation-deterministic (fixed seed + step-derived) and re-verifiable on
   resume, else resume violates the contract. Flagged as the main
   implementation risk for arms 4/5.
3. fp16-only Turing: teacher bf16 -> fp16 inference + zero-init gates chosen
   precisely for fp16 stability; escalate to fp32-master bridge internals
   only if curves show instability (G1/FLA precedent).
4. Single GPU: arms sequential; NOTHING co-runs with a training arm.
5. Disk: mount arms add bridge params (~small) + teacher stream shards
   (~2-3 MB scale range); checkpoint rotation rules unchanged.

## 10) Out of scope for this experiment (parked)

SmolLM2-360M/1.7B teacher arms (follow-up only on positive 135M read-out);
target-dims scale-up; Stage-4 SFT recipes; N-gram embedding streams and
memory-window reframing (declared compatible, deliberately excluded from the
first A/B to keep arms single-variable); GDN+QSA student stacks (orthogonal,
G1-validated, mounted later); model merging (cross-tokenizer merge is
physically impossible; only end-stage fold-in variants may be revisited).

## 11) Next step

User reviews this spec; on GO, implementation plan per the repo's
docs/plans/ convention, then code gates (sanity_check -> pilot) BEFORE any
long arm runs. No training starts without a user-opened GPU window.
