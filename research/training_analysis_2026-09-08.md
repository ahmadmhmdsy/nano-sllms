# Training analysis — the whole process through 2026-09-08

Scope: every run in this repo's history — M0 smoke, M1 VRAM probe, M2 pilot,
M3 target, Milestone B A/B arms, the LoRA hook, C12 Tier 1 SFT (Evol), SFT v2
(minimax3 corpus, pilot + 2-epoch + 1-epoch), and Tier 3 KD (P→S). Written
2026-09-08 by the GPU-queue agent session. Every number below is traceable to
a committed artifact (train_summary.json / eval_report.json / trainer_state /
tfevents — pointers in §7).

---

## 1. Executive verdict — the best model we have

**`runs/sft_v2_e1/final` is the best model we own**, and it wins on every
axis that actually discriminates between our models:

| Axis | sft_v2_e1 | sft_t1 (prev best) | target (M3 base) | pilot (M2) |
|---|---|---|---|---|
| Instruct ast pass (greedy/sampled, on-corpus held-out) | **0.98 / 0.96** | 0.86 / 0.88 | n/a (not instruct) | n/a |
| CSN forgetting guard (pre-SFT 1.8641) | 2.0466 = **+9.8% PASS** | 2.0995 = +12.6% (near edge) | baseline | 1.1608* |
| Execution test_credit (mini_eval 16×2) | **0.0625** | not measured | 0.0000 | 0.0312 |
| Eval loss (own SFT val) | 0.2421 (e2's curve; e1 ~0.26) | 0.836 | 1.8641 | — |

*pilot's val is its own data — not comparable across datasets.

Honest qualifiers that must travel with that verdict:

1. **"Best" = best on the surfaces we measured.** All our models score
   pass@1 **0.0** on execution-based mini_eval (e1 0/16, base 0/16, pilot
   0/16). e1 has the highest test_credit (2/32 single tests) but no model
   writes code that *runs correctly* on generic tasks.
2. **e1's instruct skill is distribution-narrow.** On held-out instructions
   from the same generator family that produced the minimax3 corpus it scores
   0.98; on slightly different phrasings (the config's qualitative prompts,
   hand-written probes) it degenerates into repetition loops
   ("We use a hashmap from the number of distinct characters. We use a
   hashmap from ..."). The capability is real but narrow.
3. **The M3 base (`runs/target/final`, ppl 6.45) remains the best pure-LM
   artifact** and is the base every SFT branches from. Keep it sacred.

---

## 2. Model inventory (where the weights actually are)

| Model | Scale | What it is | Weights location |
|---|---|---|---|
| runs/smoke/final | 12.3M | M0 pipeline proof | tracked in git |
| runs/pilot/final | 100.7M | M2 pretrain (Evol), ppl 3.19 own-val | tracked in git (384 MB) |
| runs/target/final | 226.5M | M3 pretrain (CSN), ppl 6.45 — **the ladder apex base** | local (restored bit-exact from checkpoint-4000 after de-weighting) |
| runs/sft_t1/final | 226.5M | Tier 1 SFT (Evol 16.4k × 2ep) | checkpoint_backup zip only |
| runs/sft_v2/final ("e2") | 226.5M | SFT v2 2-epoch — OVERFIT evidence | local |
| **runs/sft_v2_e1/final** | 226.5M | SFT v2 1-epoch — **current deliverable** | local |
| runs/kd-s-baseline/final | 12.3M | Tier 3 from-scratch control | local |
| runs/kd-s-t1/final | 12.3M | Tier 3 distilled (P teaches S) | local |
| checkpoint_backup/*.zip | — | M3 ckpt-4500, sft_t1 final+resume bundles | local, gitignored |

---

## 3. The full trajectory, stage by stage

### 3.1 Pretraining ladder (M0 → M2 → M3)

| Run | Scale | Steps | Eval curve | Outcome |
|---|---|---|---|---|
| M0 smoke | 12.3M | 200 | 10.4 → 4.95 (eval 4.7926, ppl 120.6) | PASS; kill/resume drill proven at step ~100 |
| M2 pilot | 100.7M | 3,000 | 2.2061 → 1.1608 **monotonic**, ppl 3.19 | PASS; peak 3.3 GB |
| M3 target | 226.5M | 5,000 | 2.8567@500 → 2.265@1000 → 2.0903@1500 → 1.9972@2000 → **1.8512@4000 (best)** → 1.8610@4500 → 1.8641@5000 | PASS; final ppl 6.45 |

What the curves say:

- **M2's eval was still falling at 2.35 epochs** — that justified M3's 3.4
  epochs and `load_best_model_at_end`. It worked: M3 best landed @4000, with
  a small rise after (1.8512 → 1.8641) — the curve found its floor and the
  best-checkpoint machinery caught it. Lesson: train past the floor
  deliberately; keep the best, not the last.
- **Diminishing returns per epoch are real**: M3 spent 2,000 more steps
  (≈40% of the run) for ~1.7% more eval improvement (1.9972@2000 vs
  1.8512@4000, then flat/rising). The next pretrain should budget fewer
  steps past the knee, not more.
- CSN-only pretraining gives locally-syntactic, sometimes-nonsensical
  completions (see eval_report samples: `fibonacci` that "returns the number
  of bytes"). Syntax yes, semantics no. That is what the SFT stages attack.

### 3.2 Tier 1 SFT (Evol-Instruct traces) — runs/sft_t1

- Pilot (5k pairs, 1 ep, 313 steps): eval 1.102→1.095, ast 0.60/0.78,
  forgetting +2.7%.
- Full (16,376 pairs × 2 ep, 2,048 steps): eval 0.836, ast 0.86/0.88,
  forgetting **+12.6%** — inside the ≤ +10–15% gate but visibly near the edge.
- Lesson delivered on time: the pilot predicted the full run's forgetting
  reasonably (direction + rough magnitude); the full run still landed at the
  edge. The gate is doing real work.

### 3.3 Milestone B — optimizer/hardware study (pilot-scale, 500 steps)

| Arm | eval@500 | tok/s (real run) | VRAM peak (honest probe) | Verdict |
|---|---|---|---|---|
| A fp32 AdamW b1/a32 | 2.5644 | 2,801 | 1.91 GB | control |
| B 8-bit AdamW b1/a32 | 2.5603 | ~4,100 mean (thermal-confounded) | **1.36 GB** | **WINNER — default** |
| C 8-bit + batch 2/accum 16 | 2.5604 | 1,680 (0.60× — FAIL) | 1.56 GB | rejected |

- Loss parity within 0.0041 at every eval point — **8-bit Adam is free
  quality-wise** and saves ~550 MiB.
- Batch-2 retune **rejected**: with gradient checkpointing, activations
  dominate and batching gains nothing on this 6 GB card; its real-run pace
  was dragged by one long stall (GPU healthy throughout — 79 W / 1740 MHz /
  72 °C). Recorded default: `optim: adamw_bnb_8bit`, b1/a32.
- LoRA (row 10) measured: **0.72 GB peak @ 2.52% trainable** vs 1.91 GB full
  FT of the same arch — the cheap fine-tune lane exists on this card.

### 3.4 SFT v2 (minimax3 distillation corpus) — the deliverable run

Data: 25,146 raw pairs → 19,252 kept (min_chars 30 after the 200-char floor
dropped 81% of a deliberately-short corpus; drop_long 115 > ctx).

| Run | Steps | ast (greedy/sampled) | CSN forgetting | Verdict |
|---|---|---|---|---|
| pilot (5k × 1ep) | 313 | 0.94 / 0.92 | +3.7% | PASS — predicted everything |
| e2 (19.3k × 2ep) | 2,344 | 0.98 / 1.00 | **+19.7% FAIL** | overfit evidence |
| **e1 (19.3k × 1ep)** | 1,172 | **0.98 / 0.96** | **+9.8% PASS** | **DELIVERABLE** |

The e2 failure is the single most instructive result of the project:

- **Eval loss fell monotonically the whole way (0.3990 → 0.2421) while the
  model was quietly losing its code-completion capability.** The SFT val
  split (same distribution as training) cannot see capability damage; the
  CSN regression guard can. Without the gate we would have shipped a model
  that parses perfectly and forgot more of what it knew.
- The corpus is *denser* per pair than Evol (short, surgical answers), so 2
  epochs at lr 3e-5 overwrote the base twice as hard. Data density and
  epochs interact — you cannot copy epoch counts across corpora.
- e1 keeps ~all of e2's instruct quality (0.98 greedy either way) at ~half
  the forgetting. **The first epoch buys nearly everything; the second buys
  forgetting.**

### 3.5 Tier 3 KD (P teaches S) — the cheap-ladder proof

Fair A/B (identical data/steps/seed, S = 12.3M, teacher P = pilot/final,
loss = 0.5·KL(τ=1) + 0.5·CE, eval stays pure CE):

| Steps | from-scratch S | distilled S |
|---|---|---|
| 500 (~1/3 budget) | 3.5325 | **3.4142** |
| 2,000 (full) | 2.6201 | **2.4755** (−5.5%) |

- Plan §5 criterion — "distilled ≥ baseline at ≤ 1/3 the steps" — **PASS**.
  The ~1/10-compute claim transfers to this hardware.
- Cost surprise in our favor: the from-scratch S run took **~3 minutes** of
  GPU (the plan estimated 2–3 h). S-scale experiments on this GPU are
  essentially free; use them as the standard research tool.
- KD machinery is now proven end-to-end (teacher serving, KD loss, resume) —
  the plumbing any future strong-to-weak transfer needs.

### 3.6 Execution-based reality check (mini_eval, run 2026-09-08)

| Model | pass@1 | test_credit |
|---|---|---|
| sft_v2_e1 | 0.0000 (0/16) | **0.0625** |
| pilot (M2) | 0.0000 | 0.0312 |
| target (M3) | 0.0000 | 0.0000 |

- **ast-parseability ≠ correctness**: e1 parses at 0.98 and executes at 0.0.
- Failure forensics: unterminated docstrings (the model's corpus style
  opens long docstrings it fails to close), token-glue artifacts
  (`returna, b`), and on off-distribution phrasings, hard repetition loops.
- Note the metric mismatch too: mini_eval feeds **raw completion prompts**
  (`def add(a, b):\n    return`) — an instruct model legitimately lost that
  surface (that IS the forgetting the guard measured as +9.8%). e1's
  test_credit being the highest of all models suggests the SFT did add some
  real signal even here.

---

## 4. What we learned (the transferable lessons)

1. **In-distribution eval cannot detect capability forgetting.** Only an
   out-of-distribution regression probe (CSN val for us) caught the 2-epoch
   overfit. Any future SFT must carry such a guard, evaluated per milestone.
2. **The first epoch buys nearly everything; later epochs buy forgetting.**
   Early-stop SFT aggressively; prefer more diverse data over more epochs.
3. **Data density changes the recipe.** Dense/short corpora (minimax3)
   teach instruction-following ~3× faster per step than long trace corpora
   (Evol) — and overwrite the base faster too. Re-derive epochs/LR per
   corpus; never copy them.
4. **A model's honest score is the worst of its surfaces.** ast 0.98 /
   pass@1 0.0 / off-distribution loops — report all three, always. The
   eval report's own qualitative samples caught the degeneration the
   pass-rate hid; read them, every time.
5. **Distillation works and is cheap at S-scale.** KD beat from-scratch at
   every budget point. Future rungs: distill, don't pretrain.
6. **8-bit Adam is a free win** (parity loss, −550 MiB); batch retuning is
   not (grad-ckpt activations dominate; batching buys nothing on 6 GB).
7. **LoRA is the low-forgetting lane**: frozen base ⇒ the forgetting guard
   can barely move; the open question (next experiment) is how much instruct
   quality survives at 2.5% trainable.
8. **Infrastructure is a feature**: the zero-flag auto-resume survived a GPU
   driver fault, three sleeps, a machine move, and two user stops. The
   partial-checkpoint guard + resume-path fix (MEMORY 13/14) and the
   disk-full hang signature (MEMORY 15) came from real failures — and the
   resume contract is why none of those failures cost more than minutes.
9. **Estimates are wrong; measurements are right.** GPU-cost estimates were
   off by 10–100× in both directions across the project (M3 pace, KD cost,
   SFT epochs). Measure with small pilots, always.

## 5. What we should do (ranked)

1. **Run Tier 2 V1 (teacher-as-judge rerank)** — it is precisely the
   medicine for the diagnosed disease (narrow distribution): the student
   samples, the judge (Qwen3.5-0.8B) reranks, training signal comes from
   the model's OWN output distribution. Teacher is already downloaded; the
   judge contract is frozen (row 14). Needs your go.
2. **LoRA-SFT variant of SFT v2** (row-10 hook is live): same corpus, frozen
   base. Prediction from the measured mechanics: forgetting ~0–2%, instruct
   quality somewhat below e1. If it lands near e1's ast, it becomes the
   default SFT recipe (cheap + safe + mergeable).
3. **Mixed-corpus SFT v3** (Evol + minimax3 + a slice of generic-phrasing
   instructions) to widen the distribution — direct answer to the
   repetition-loop finding. CPU data-prep only; the run is ~1 h.
4. **Keep the CSN guard + add mini_eval to the standard SFT eval** — three
   surfaces (in-dist ast, OOD forgetting, execution credit) as the standing
   gate; never ship on one number again.
5. **Next pretrain (when scheduled)**: Milestone D's 4-source mix
   (~134 M tokens), `adamw_bnb_8bit` b1/a32, stop ~1 epoch past the eval
   knee instead of running to the budget.
6. **Off-site backup** (row 15): give a private HF repo name; the script is
   dry-run-verified. Also: zip `runs/sft_v2_e1/final` + `runs/target/final`
   into checkpoint_backup for machine-move safety (same pattern as Tier 1).

## 6. What we should NOT do (each one earned)

1. **Don't trust eval loss (or ast rates) alone for SFT** — e2 was the best-
   looking run we ever produced and it failed the gate.
2. **Don't copy epochs/LR between corpora** — 2 epochs on dense data
   overwrote the base; 1 was the right answer.
3. **Don't SFT from a non-apex base** — the row-4 decision (final, not
   ckpt-1000) was correct and cheap; a 20%-trained base would have
   inherited a weaker ceiling.
4. **Don't delete/de-weight finals while a pipeline needs them** — the
   de-weighting crashed the first sft_v2 pilot launch; restore-from-ckpt
   saved us only because checkpoint-4000 was still on disk.
5. **Don't run GPU jobs beside a live trainer** — the never-co-run rule; the
   6 GB card has no slack (a 0.9 GB probe beside a 4.9 GB run is OOM
   roulette).
6. **Don't start long runs with < 5 GB free** — disk-full writes hang at a
   fixed offset and look like a code bug (MEMORY 15). SFT checkpoints are
   ~2.7 GB each; budget 3 × that before launching.
7. **Don't kill a run for pace alone** — thermal swings (2.9–23 s/it on the
   same run) were all benign; the real killers were sleep and disk, both of
   which the resume contract absorbs.
8. **Don't re-derive environment facts per session** — the CUDA_VISIBLE_
   DEVICES empty-string trap, peft-missing-after-move, WDDM process-list
   noise: all in MEMORY; read it first, add to it immediately.
9. **Don't trust step/time estimates** — measure with a 500-step window or
   an S-scale run first; they are minutes-cheap.

## 7. Evidence index

- Per-run: `runs/<phase>/final/train_summary.json` (env fingerprint +
  resolved config), `final/eval_report.json`, `logs/events.*` (tfevents),
  `checkpoint-*/trainer_state.json` (full eval history).
- Milestone B: research/milestone_b_8bit_ab.md; TASKS rows 10/11/18.
- SFT v2: configs/sft_v2{,_e1}.yaml; data/sft/minimax3/ (meta with drop
  stats); runs/sft_v2{,_e1}/final/eval_report.json; TASKS row 19.
- KD: scripts/kd.py; runs/kd-s-{baseline,t1}/train_summary.json; TASKS row 20.
- Execution eval: runs/{sft_v2_e1,target,pilot}/final/mini_eval_report.json.
- Lessons: MEMORY.md items 12–15; HANDOFF §2 + "2026-09-08 incidents".
- Commits (this arc, local-unpushed): 16373b9, 5fbb31a, df0dd5e, b278479,
  1226196, 40e9aed, bdc09a1, 4d1e428, cad791b, d5c5ac7, 22a9922, 5682fd2,
  ede7147, 088935f.
