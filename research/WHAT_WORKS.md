# WHAT_WORKS.md - proven levers for future training

The inspiration shortcut: read this BEFORE planning any new training run.
Every entry is backed by a committed experiment (evidence column) and carries
its conditions - a lever without a condition is a rumor. Two sections:
proven levers (do this by default) and rejected pitfalls (do not refight
these battles). Update on every new verdict; a lever may move down as
evidence accumulates. Cross-reference: [research/EXPERIMENTS.md](EXPERIMENTS.md)
(the ledger) and [MEMORY.md](../MEMORY.md) (raw gotchas).

## Proven levers (ranked by evidence strength)

1. **Warm-start via embedding transplant (KT-1)** - transplant a capable
   same-tokenizer-family teacher's embeddings into our arch at init
   (byte-level piece alignment + isometric lift). -6.24% eval @1000 vs fresh
   init, gap GROWS through training; lands within +1.1% of the teacher's own
   curve. Conditions: keep our architecture (user Q2 decision); use
   scripts/embed_transplant.py; the tied lm_head follows the copy
   automatically. [E-05]

2. **Plain same-tokenizer distillation (Track C)**. -6.71% @2000 vs baseline,
   ahead at EVERY matched step; the claim transfers across compression ratios
   (2.25x piloted and 8.2x at S-scale precedent). Conditions: fp32 teacher
   logits, F.kl_div target=probs (MEMORY 25 gotcha), per-shard VRAM probes
   before choosing the micro-batch. [E-03]

3. **Judge/rerank data loop, batched stages (KT-2)**. On-policy judge filtering
   of self-generated candidates drives SFT corpus quality; batched sampling +
   scoring is 5.6x faster with byte-identical outputs; the informative signal
   is the judge-vs-LL agreement rate, not the judge's raw lenient score. [E-06, E-09]

4. **Effective-batch elevation over accumulation** (MEMORY 37). Batch 4 / accum 4
   = identical training math at ~3x wall speed; keep effective batch constant,
   raise per-device batch until VRAM or quality says stop. [E-09]

5. **YaRN ctx extension + soup as a measured post-hoc knob (Track B)**.
   Full ctx-4096 base improved long-ctx val -8.0%; LoRA-instruct on top reached
   the base-surface gate only via a 0.6x merge soup (a60: all 4 strict gates
   PASS). Soup alpha is near-LINEAR (no knee): use it as a repair knob, not a
   substitute for the LR schedule. [E-07, E-10]

6. **GDN hybrid champion layout at S-scale (G1 sandbox)**. 3:1 GDN:gated-attn
   hard layout [GDN,GDN,GDN,GA] beat a param-matched GQA control by -0.98
   val_loss with <0.5% VRAM overhead; fp32-state guard keeps fp16 stable.
   P-scale trial remains user-gated. [E-08]

7. **Mounting with the hybrid schedule** (gate anneal first, stochastic drop
   last). Best mount arm (1.8153, +0.3% vs control) at the best teacher-arm
   wall-clock (2h05m): the drop phase's teacher-forward skipping translated
   directly into wall-clock. Honest pilot-scale framing: parity, not net gain -
   adopt when the goal is teacher-injection training infrastructure. [E-11]

8. **8-bit Adam (adamw_bnb_8bit) as the recorded default optimizer** (Milestone B A/B):
   parity -0.004 eval @500 steps at 1.02x pace and -563 MiB VRAM (1.91 -> 1.36 GB) - the
   headroom that later enabled bigger batches and ctx work. Batch 2 / accum 16 was parity
   but 0.60x pace: batch-2 buys nothing on the 6 GB card; keep b1/a32. [E-12]

9. **One-epoch instruct SFT sweet spot** (SFT-v2): at matching AST, 1 epoch passes the
   forgetting gate (+9.8%) while 2 epochs violates it (+19.7%); the second epoch buys
   sample-level AST but overwrites base skills. Let the CSN forgetting gate arbitrate. [E-15]

10. **Logit-KD transfers across rungs: distill instead of pretrain from scratch** (C12
   Tier 3, P->S at tau 1): -5.5% @2000 and the distill-at-1/3-steps criterion PASSED -
   future rungs distill from a sibling. Loss = 0.5 KL + 0.5 CE; keep eval PURE CE so the
   A/B stays valid. [E-16]

11. **Streaming-sink + absolute-positions eval for long context, no training** (Track A):
   the w1024s4-abs mask holds val flat to 16x ctx (-1.50%..+0.00%) - an inference-only way
   to serve 16k ctx; its recall is capped at window+sink, so true long-ctx quality still
   comes from training (YaRN 4096 = the trained decision). [E-17]
   serve 16k without training; hmm capped recall at window+sink, so pretraining still owns
   true long-ctx quality (YaRN 4096 = the trained decision). [E-17]

12. **D-line diacritization: spend params on DEPTH first, data mix above all**
   (2026-09-13, param-matched ablation trio at the same 30M budget, same
   data/batch-stream): depth x2 (14L/384 -> 28L/272, head_dim 64->34) =
   gates -2.0..-3.9 pp DER at ZERO extra params, decisive on Sadeed (-3.7)
   and WN-2014 (-3.9); the 2.5x-wide 76M model bought -1..-4.6 pp for 170 MB
   more; a Fadel-only same-arch specialist took Fadel test 47.1 -> 34.2
   (-12.9 pp), the largest single move of the whole ladder. Verdict ladder:
   data/domain routing >> depth-per-param > raw params; both arch axes sum
   to ~1/10 of the gap to SOTA systems that transfer from pretrained models
   (Fine-Tashkeel ByT5, CATT char-BERT, PTCAD BERT-class) - our next lever
   is a v3 mixed corpus (+Sadeed_Tashkeela 1.04M windows, wired) and/or
   transfer init on this line, e.g. the proven KT-1 transplant lever above.
   val_loss was a dead call while gates moved - gate metrics are a more
   sensitive read than the label loss at this scale. [E-12, E-13]

## What NOT to repeat (with reasons)

- **Skew-KL (alpha-SKL) distillation**: +2.29% over plain-KD, led only at a
  200-step window, and fp16 grad norms were healthy (no skew floor needed).
  Choose plain-KD unless deliberately building a divergence-repair phase. [E-03]
- **Hard teacher-off anneal without bridges**: the mount-fill arm collapsed
  (post-cliff eval flat ~6.5, fp16 grad-norm median 1454). If distilling
  without bridges, make the teacher-off transition an lr-warmed phase, not a
  hard shutoff. [E-11]
- **Full-severance claims**: the keep-one rule in the mount design makes full
  severance structurally unreachable - never claim it. [E-11]
- **QA mixing as the AST-collapse cure**: did not fix the code-collapse wall at
  LoRA scale; recipe sweeps were also insensitive. Prefer DATA quality and
  judging budgets over more recipe knobs. [E-04, E-09]
- **Final-step-only eval claims**: drop's stored best (2.2881 @900) was worse
  than its end state; fill's final_eval was bitwise-equal to its step-300 eval.
  Always cite the eval source per table and report the end-state re-eval
  separately (MEMORY 47). [E-11]
- **CUDA_VISIBLE_DEVICES='' to hide the GPU**: PowerShell mapping DELETES the
  variable and the job lands ON the GPU; pin -1 to force CPU (MEMORY 38). [E-11 ops]

- **Stale on-disk eval reports after weight restore**: the target eval_report 1.8641 was
  stale (re-restore of best@ck-4000 measures 1.8512) - re-evaluate finals after any
  weight restore/cleanup before trusting old numbers (MEMORY 31). [E-12/E-17]

## Standing instruments (our standard measurement kit)

- Wall-clock to matched-step anchor = the headline efficiency metric.
- CSN forgetting guard (delta vs the pre-run state) on every instruct/adapter run.
- AST greedy + sampled pass-rate on 50 held-out instructions.
- Track-H recall probes (store/retrieve/copy, prompted vs deterministic arms).
- VRAM ladder probes BEFORE choosing micro-batches (MEMORY 25b).
- Kill/resume drill (exact zero-flag) passed before any full-arm launch.
- Independence/gate-difference evals with named sources (eval_report.json as
  the authoritative number).

## V4.1 transfer candidates (2026-09-19 micro-bench, E-40/E-41)

- **CED-lite (cross-layer KV sharing, upper half)** - loss-neutral at nano
  scale (val ±0.02); ~35% step-time saving at ctx512 pilot-proxy; -40% inference
  KV analytically. Conditions: adopt for INFERENCE KV memory (and possibly
  long-ctx training where activations fill VRAM); re-check per batch/ctx at P.
  [E-40]
- **Muon optimizer (orthogonalized momentum on 2D params)** - beats AdamW at
  every tried lr at nano scale, -10% train loss at 3e-2, but high LR
  sensitivity: NEEDS a proper LR sweep (e.g. 3 scales) before adopting in
  train.py. Do not swap optimizers blindly. [E-41]
- **Where our VRAM actually goes at long ctx**: activations 70% at ctx2048
  (pilot proxy, bs2); params+optimizer fixed 1.5 GB. Gradient checkpointing /
  CED-lite are the right levers for long-ctx training on 6 GB. [E-40 M1]

- **Sinkhorn-style embedding row-norm rebalance** - tiny geomean push on tied
  embedding rows gives ~2.6% train-loss improvement at nano scale with zero
  cost. Rate sweep done: flat-topped plateau 0.05-0.2; best 0.1 (4.798 vs
  control 4.928). P-scale A/B run: weak pass (-0.5% at d768 vs -2.6% at nano) — wire into
  train.py as a non-default flag first; default-on only after a real
  pipeline val-loss A/B. [E-42]
- **MTP-style aux next-token-2 head** - NEGATIVE at nano scale: aux-head
  gradient hurts main loss at every weight tried (0.02..0.3), even aux-exclusive
  metrics. Retry only with a decoupled aux LR + head warmup at a longer budget.
  [E-42]
## 2026-09-22 - Laya-line E-62: decision heads transfer down; calibration is nearly free
- The convaiinnovations/laya decision head (type embedding + option-marker scorer + masked per-question softmax) transfers UNCHANGED onto MiniLM-L12-H384: 0.6205 typed-decisions test acc at 37.2M params (vs MiniLM-L6 22M 0.587, ModernBERT-base 149M 0.646) with the published notebook recipe (4 epochs, eff 64, lr 2.5e-5/1e-4, cosine, fp16, soft-CE on gold distributions). 286 s on Quadro RTX 4000, peak 1660 MiB. Adopt for typed-decision/soft-label classification heads at 30-50M scale.
- Soft-CE training on gold PROBABILITY DISTRIBUTIONS (not argmax labels) yields well-calibrated models for free (raw ECE 0.0673). Do NOT add per-group post-hoc temperature on <1000-item calibration slices - it overfit (0.0673 -> 0.0948). One global T on >= 2000 items, or nothing.
- Weak primitives at 37M: score questions (0.583) and agent-trace observability workflow (0.528) - same shape as the Laya 421M profile; watch these in L2+.


- Laya-line L2 ladder (mixture pretrain -> typed fine-tune, 37.16M): +3.8 typed acc over direct fine-tune (0.6205 -> 0.6585) at identical params, ~36 min total on 8 GB. Adopt for any future decision-head line.
- Probe suite (11 assertions, ~1 min on GPU) as cheap regression check: catches behavioral failures accuracy hides; L1 tied the best published grounded arm (1 failure vs Laya base 7).

## 2026-09-23 - Laya-line E-65: what domain pretraining at 35M buys (and does not)
- PASSES the transfer gate, fails the capability gates: 260M tokens of email/security/support/reddit domain text took zero-shot phishing AUROC from 0.576 to 0.649 (Laya 0.678, Jev 0.689) but typed acc DROPPED vs the MiniLM incumbent (0.5185 vs 0.6585). Upstream general-token scale dominates decision-generalization; domain match dominates domain-adjacent transfer.
- 15% replay FULLY held the mixture through stage B this time (0.6625 -> 0.6622, -0.0003; L2 forgot -0.062 at the same replay): when retention fails, first ask whether stage A ever cleared the bar - a low ceiling masquerades as forgetting.
- Order-shuffle + rename augmentation (p=0.5/0.25) did NOT fix permutation invariance (0.185 vs 0.385 unaugmented): repack-based aug is not equivalent to the block-surgery test; next lever is training on surgically-permuted duplicates of the SAME item.
- Scheduled bench probes (frozen-encoder throwaway head every 4k steps -> bench_log.jsonl + TB) worked for monitoring; their absolute numbers are NOT benchmark numbers (probe head undertrained by design) - use for trends only.
- Unattended-safe discipline held across 6 crashes/resumes (masking shape, OOM, probe device/type, argparse, permutation API): every resume was zero-flag from a checkpoint saved BEFORE the failure point.

## 2026-09-23 - E-65 avoid-lesson
- AVOID repack-shuffle-only option-order augmentation: E-65 stage B got WORSE permutation agreement with it (0.185 vs 0.385 unaugmented). Invariance-by-construction (same item, multiple orders in-batch - E-66 lever) or PMI/surface-form debiasing are the literature-supported routes; validate on held-out either way.

## 2026-09-23 - Laya-line proven levers (E-70..E-73)
- Token scale (2-pass 520M on the same corpus): +0.109 zero-shot phishing AUROC (0.5878 -> 0.6966, past published Laya 0.678 / Jev 0.689), typed -1.1
- Stage-B length 8->12 ep: +5.45 typed (0.6420 -> 0.6965, past the L2 incumbent), transfer -0.087
- Per-source balanced replay (replay_balanced flag): +6.0 transfer back at -2.6 typed vs pure length; strictly dominates the low-typed arm
- Metric-pegged early stop (stageB_auroc_stop): keeps the protected metric at its peak-class while taking the typed gains transfer permits (E-73 double-gate WIN)
- KD from a different-encoder teacher: NEGATIVE (do not redo without encoder-family match)
- Fitted calibration (any flavor): negative 3/3 - quarantined by rule

## 2026-10-06 — D-line tiny-model proven levers (E-061…E-072, the 1.08 M era)

The D-line ladder, re-audited. Current best is **weights + procedure**, not a
checkpoint alone. Refuted D-line arms are archived in
`research/archive/diacritizer_failures.md`.

1. **Stage-1 char-LM warm start is THE init lever** (E-067 P1 → E-068 C1). The
   control F2 scored 0.3974 from RANDOM init while its config *claimed* a warm
   start; applying the real `runs/diac/e67_p1_lm64/final/model.pt` (165/165
   tensors, 0 missing/unexpected) won by **3.74 %** (0.3825 @90000). Conditions:
   verify the load counts in the log — a silent no-op is now FATAL (D-12 guard),
   and `allow_empty_warm_start` waives it deliberately.
2. **ctx 256 is a second, independent, composing lever** (E-067 P9): −6.39 % vs C0
   best-of-run, −4.02 % at matched probes, −5.62 % vs a no-seed replicate.
   Conditions: needs its own tokenized bed (`data/diac/v3ctx256`, PAD=0);
   its `[:128]` prefixes are byte-identical to v3, so it is a pure context
   change. 78.8 % of train rows carry real text beyond column 128.
3. **Budget does NOT compose past the plateau** (E-070, the counter-lesson): C4b's
   first 15 k continuation steps bought −3.21 %, the remaining 45 k bought nothing
   on the gates while val_acc kept climbing. **Judge at matched tokens, never by
   best-of-run alone.**
4. **Gate-probe selection beats val_loss selection** (E-15…E-17, reaffirmed
   2026-10-05): take the deployment checkpoint from
   `gate_probe/best_gate_weights.pt`. A1 improved val_loss 0.2431 → 0.2397
   while ALL FOUR external gates regressed (mean 49.80 → 51.82).
5. **Sliding-window majority voting is FREE accuracy** (E-071): −1.34 % on the
   registered 3-gate mean (0.374712 → 0.369702), every gate improving, McNemar
   p ≤ 3.2e-11, deterministic — no retraining, no new data. Conditions:
   window/stride must match the model's ctx (256/128 here); the attn_mask
   convention is **True = KEEP** (`model.py` does
   `masked_fill(~attn_mask, -inf)`).
6. **Test a selection on a set that took no part in it** (E-072): the gate-best
   step 15000 reproduced its ranking on the held-out abdou control (15,091 sents /
   41,378 words, never trained, not a registered gate): 15000 < 40000 < 60000,
   both pairwise significant. Corollary: the top-two ranking can still be
   METRIC-DEPENDENT — step 40000 leads `der_canonical` and macro_f1 while
   step 15000 leads the registered DER.
7. **Reproduce the baseline BEFORE comparing** (E-071 method): the voting script's
   own single pass had to re-derive the recorded 0.374792 first. It did
   (|diff| 0.00008). Without that check a −1.34 % "gain" could have been a harness
   difference.
8. **8-bit Adam buys VRAM headroom at parity** (Milestone B / E-12): −563 MiB
   (1.91 → 1.36 GB) at 1.02× pace — the headroom that later enabled the larger
   batch and ctx work. Keep b1/a32 on the 6 GB card.
9. **Depth before width at matched params** (E-13): depth ×2 (14L/384 → 28L/272) =
   −2.0…−3.9 pp DER at ZERO extra params, decisive on Sadeed and WN-2014.
   Unvalidated at 1.08 M — a candidate, not a proven lever at this size.
10. **Domain specialization is the largest single historical move** (E-12 arm A): a
    Fadel-only specialist took Fadel test 47.1 → 34.2 (−12.9 pp) while losing on
    modern-WN. Adopt only WITH per-domain routing.
11. **Plain same-tokenizer KD** (E-03, core line) — −6.71 % @2000, ahead at every
    matched step. **Never yet applied on the D-line** — the obvious untried
    combination.
12. **Soup is a near-linear post-hoc repair knob** (E-07, E-10) — no knee, no free
    lunch. Untried on the D-line.

### D-line avoid-list (added 2026-10-06)

- **Do not scale params or attack the architecture axis** — refuted three times
  (E-13, E-14, E-23d).
- **Do not add volume or a new label style** — classical expansion ≈ 0 (E-23b), ZM
  distillation −4.13 (E-23c).
- **Do not trust `warmup_steps` / `grad_clip` in a D-line config** —
  both are DEAD KEYS (warm-up unimplemented; grad_clip hardcoded 1.0 at
  `train.py:389`).
- **Do not select on val_loss near convergence.**
- **Do not "fix" `research/EXPERIMENTS.md`** — frozen historical ledger by
  its own authority banner.

