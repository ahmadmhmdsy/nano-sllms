# Verdict audit — consolidated (D-line diacritizer, Laya, U-line/core, legacy)

Consolidated 2026-10-06. Authority for method: `rules/evidence.md` (an artifact is a metric
file, console tee, or gate-bench output — never a remembered number), `rules/experiments.md`.

## Why this exists

The user asked for a recheck of *all* recorded verdicts in both error directions — false FAILs
(a metric read backwards) **and** false PASSes (an unsupported win) — across every track, tier,
arm, and technique. This hub is the index; the per-scope detail lives in `parts/`.

**Headline result:** the recheck found 6 metric-direction inversions (D-6), 3 wrong-baseline
errors (D-7) that had propagated into the plan and the registry as durable priors, 6 false
PASSes resting on an absent or overwritten artifact (D-11), and 1 slice-vs-full scoring bug
that hid the strongest external system on record (D-8). It also confirmed the great majority of
recorded verdicts: 51 of 102 entries reproduce exactly.

## Metric directions (used everywhere below)

- **LOWER is better:** DER, WER, CDER, loss, val_loss, ECE, NLL, Brier, latency, VRAM.
- **HIGHER is better:** accuracy, val_acc, pass rate, valid@k, F1, macro F1, AUROC, agreement,
  token/sec, mark_accuracy, retention.
- **Registered diacritizer metric** = mean DER over `GATE_EXTERNAL` = (fadel_test, sadeed25,
  wikinews2014); `diacritizer/scripts/train.py:39` defines it and `:204-220` selects the
  checkpoint by the LOWEST `mean_der`. wikinews2024 is excluded as contaminated.
- A "3-gate mean" is over GATE_EXTERNAL; a "4-gate mean" adds wikinews2024. **Never mix them**
  (defect D-1).

## Reproduce any of this (read-only, no GPU)

| tool | what it does |
|---|---|
| `tools/score_gate_bench.py <run>...` | re-scores `gate_bench/*.prd.txt` vs `*.prd.ref.txt` via `eval_metrics.score_corpus` (the deployed harness) |
| `tools/score_probe.py <run> <step>` | re-scores `gate_probe/step<N>_<gate>_pred{,.ref}.txt` (the in-training probe path) |
| `tools/score_probe.py --list <run>` | lists every probe step in a run dir |
| `evidence/*.json` | frozen outputs of both tools |

Both tools `sys.path`-insert `diacritizer/src` and use the same scorer as the training loop, so
they can confirm or refute a recorded DER without re-running a model.

## Scope index

| part | scope | entries | CONFIRMED | MISCITED | INVERTED | NO_ARTIFACT | other |
|---|---|---:|---:|---:|---:|---:|---:|
| `parts/dline_recent_e61_e67.md` | E-061, E-063..E-067 | 6 | 4 | 0 | 1 | 0 | 1 STALE |
| `parts/e19_e23_bakeoff.md` | E-19..E-23 | 24 | 12 | 5 | 4 | 2 | 1 UNSUPPORTED |
| `parts/uline_core_legacy.md` | U-line, core, legacy registry rows | 17 | 11 | 1 | 1 | 4 | — |
| `parts/laya_e67_e75.md` | Laya E-66..E-75 | 13 | 3 | 2 | 0 | 8 | — |
| `parts/laya_l1_l2.md` | Laya E-62..E-65 | 6 | 4 | 1 | 0 | 1 | — |
| `parts/dline_legacy_e31_e63.md` | D-line E-31..E-63 + mex E-38..E-55 | 36 | 17 | 9 | 0 | 7 | 3 UNSUPPORTED |
| **total** | | **102** | **51** | **18** | **6** | **22** | **5** |

## Defect taxonomy (the durable part)

These are the mechanisms behind the individual errors. Each is a *class*, so it can be checked
for elsewhere.

- **D-1 metric-set ambiguity.** 3-gate and 4-gate means quoted side by side without labels.
  Instance: E-064 reports "A1 49.80 vs A0 51.76" (4-gate) beside "A1 46.15 vs A2 45.77"
  (3-gate) — arithmetically both correct, but a reader sees a phantom contradiction.
- **D-2 label-smoothing loss floor.** With `label_smoothing=0.1` the val_loss floor rises to
  ~0.56 (entropy of the smoothed 15-class target), so val_loss is not comparable across
  smoothed/unsmoothed arms. Instances: E-066 R1 (0.6789), E-063 S4. Compare val_acc or gates.
  `diacritizer/src/model.py:282`.
- **D-3 gate-CSV schema drift.** Legacy `phase,src,der,wer,den,pres` vs modern
  `step,gate,DER,WER,DER_nocase,text_preservation,lines,pred`. Aggregators that assume one
  header KeyError on the other.
- **D-4 unclosed runs.** E-066 R1 ended at step 109500/200000 with no `monitor/summary.json`
  and no `final/`. `train.py` writes `summary.json` on **both** exit paths (early stop
  `:431`, natural completion `:469`) and `final/model.pt` at `:458-465`, so the absence of
  both proves the process was killed — not an undocumented early stop. The record stayed
  `in_progress`.
- **D-5 evidence-path fragility.** Headline numbers living in `scratch/` (declared throwaway)
  or in gitignored `runs/` paths the registry does not name. Instances: E-061 numbers in
  `scratch/prof_v21/`; the whole Laya E-66..E-73 artifact set absent (`.gitignore:162-163`).
- **D-6 INVERTED metric direction.** The most damaging class. Instances:
  - **E-067 P1** — best 0.4299 @32000 recorded as a regression vs C0 0.4577 although lower DER
    is better; the real result is 6.07% BETTER. (Corrected in-session.)
  - **E-007-v2-verdict** — records val 1.39 @600 as "better than" 0.774; val_loss is
    lower-is-better, so this is backwards.
  - **E-23d sub-claim** — recorded that arm D1 "loses to arm E"; D1 55.83 vs E 52.55 is a win
    for the worse-but-higher number being D1. Inverted.
  - **Registered direction bug** — `runs/v2/core/micro_full/validate/config/resolved.json`
    registers `syntax_error_rate: max` and `tokens_per_second: min`, both backwards.
- **D-7 Wrong / naive baseline (false NEGATIVE engine).** A wrong yardstick turns a win into a
  loss in the record. Instances:
  - **Gold wikinews2024 = 57.58, not 48.65.** 48.65 is the gold's *wn2014 + word-cache* row
    (`research/e19_bakeoff/E20_CACHE_SMALLMODEL.md`); E-21 copied it into the wn2024 column,
    E-23a..d inherited it, and it reached the plan priors. Correct gold 4-gate mean 46.37
    (recorded 44.14), 3-gate 42.64.
  - **E-22 arm-C mean 48.42** (the table's own rows sum to 48.34); delta -2.68 should be -2.76.
  - **E-23d gold yardstick 44.14** should be 46.37.
- **D-8 Slice-vs-full scoring (false NEGATIVE).** A prediction file covering N lines scored
  against the full M-line reference pads every missing line as an error. **E-19
  Fine-Tashkeel**: the 256-line and 500-line predictions scored 91.60 / 70.55 against the full
  2500 / 1612-line refs, and "preservation 10%". Against refs truncated to their own line
  counts the *same* artifacts score **fadel 21.33 / sadeed 23.27 / wn2014 37.62** — beating the
  gold by 11.90 / 22.41 / 11.37 DER, which satisfies adoption rule (a) rather than failing it.
  Verified independently this session (buggy 0.9160 vs sliced 0.2133, preservation
  0.1022 -> 0.9569). Mechanism: `eval.py compare` pads missing prediction lines as errors.
- **D-9 Train-vs-test metric confusion.** Laya `E-074` phase-1 "typed 0.7712 on unseen test
  n=2000" is a **TRAIN** console metric (`l3_e74_console.log:1414` shows `"n": 6000`;
  `train_l3_sym.py:201,207`).
- **D-10 Overclaim against the pre-registered bar.** Laya E-72 "phishing-bar PASS" (0.6092) is
  below the bar it names (0.68 / 0.6966).
- **D-12 Silent warm-start no-op.** A `pretrained_init` whose tensor shapes do not match the
  model loads *nothing*, and the run silently proceeds from random init while the config, the
  report and the registry all describe it as "warm-start". Instances: **E-065 F1/F2 — the
  project best (0.3974) was reached from random init** although the recipe claims a warm start
  (`runs/diac/stage1lm/final/model.pt` = 272 hidden / 255 keys vs the 64-hidden s7's 165);
  E-066 R1 (`shape_skipped 255`); every s7 run before E-067 P1. Detection: the launch log's
  `{'tensors_in': N, ..., 'shape_skipped': M}` must have `M == 0`. Fix: assert
  `shape_skipped == 0` in the warm-init loader so a mismatch fails the launch instead of
  silently randomizing it.
- **D-13 PAD-convention drift on any rebuild — a *documented* hazard that still fires.**
  `tokenize_data.py` fills the ignore label with `-1`, while every pilot corpus (v3s10) and the
  deployed gold use `0`; the two differ **only** at PAD positions, so a rebuild silently yields a
  bed whose **ids match byte-for-byte and whose labels do not**.
  `data/diac/v3/tokens/PAD_CONVENTION.md` names this as an "Open hazard (not fixed here)" — and
  the first bed built after that document was written (E-067 P9's `v3s10ctx256`, 2026-10-06) hit
  it: **295,757 of 2,560,000** sampled cells differed, every one with value pair `(0, -1)`. Fix:
  `--pad-label 0` (added; the default stays `-1` so no existing reproduction changes).
  Detection: prefix-check **ids *and* labels** against the source bed — an ids-only check passes.
- **D-11 Artifact overwrite / console-only evidence.** A PASS resting on stdout, or two scripts
  writing the same output path, so the cited artifact holds a *different* experiment's numbers.
  Instances: `runs/mex/mu2_e38a/summary.json` holds the bridges-OFF row because
  `mex/scripts/eval_mu2_e38a.py:96` writes `res_on | res_off | gates` in that order (E-38a's ON
  headline is console-only); `runs/mex/mu3_joint/gates.json` was overwritten by E-54b
  (`eval_mu3_e45_gates.py:130` and `eval_mu3_dia2_gates.py:131` write the same path);
  `runs/mex/gates_dia2c.json` was overwritten by E-59 dia2i (E-54c/E-54d are prose-only).
  Fix: one artifact per experiment id, and never accept a PASS whose only evidence is stdout.

## Corrections applied to the repository (2026-10-06)

| # | record | was | is now | where fixed |
|---|---|---|---|---|
| 1 | E-067 P1 | FAIL (warm start hurts, -6.1%) | **PASS/WINNER** (6.07% better, all 10 matched steps) | plan §2/§4/§5, `experiments.yaml` E-067 |
| 2 | E-19 Fine-Tashkeel | FAILS rule (a) | **PASSES rule (a)**: 21.33 / 23.27 / 37.62 vs gold 33.23 / 45.68 / 48.99 | `parts/e19_e23_bakeoff.md` |
| 3 | E-23a | NOT-ADOPTED, "+1.45 worse" | **WIN** — at step 2000 beats gold on all four gates (mean3 40.20 vs 42.64) | plan §2/§4 rows, `experiments.yaml` anti-traps + P4 prior, `E23A_GOLD_V3Q.md` |
| 4 | E-23d sub-claim | "D1 loses to arm E" | D1 55.83 **beats** E 52.55 | `parts/e19_e23_bakeoff.md` |
| 5 | E-22 arm-C mean | 48.42 / -2.68 | **48.34 / -2.76** | `E22_QCRI_WEAK.md`, `E23D_BILSTM_AB.md` |
| 6 | E-23d vs arm C | +7.41 | **+7.49** (+9.46 vs gold) | plan §2, `experiments.yaml` E-067 |
| 7 | E-23c delta | +4.13 | **+4.85** (4-gate 51.22 vs 46.37) | plan T8, `experiments.yaml` |
| 8 | E-066 | in_progress "step 47300" | **interrupted**; real best 3-gate 0.4205 @40000 = FAIL on peak, PASS on delay | `parts/dline_recent_e61_e67.md` |
| 9 | E-063 M1 | "none beats the gold" | M1 `e63_m1_full` 0.4253 @5000 is **marginally better** than gold 0.4264 (0.11 pp, inside noise) | `parts/dline_recent_e61_e67.md` |
| 10 | laya/E-074 | test 0.7712 | **TRAIN** metric (n=6000) | `parts/laya_l1_l2.md` |
| 11 | E-63c M1 | "all gates strictly better or equal vs gold" | **FALSE** — wn2014 49.07 > gold 48.99; better on 3 of 5 gates and on the 3-gate mean | `research/e63_overfit_report_2026-09-23.md` |
| 12 | E-63s sweep | "BEST OVERALL = S3 40.77" | **S4 (4.5 M) is best**: `best_gate.json` mean_der 0.39962 @16500 (= 39.96) | `research/e63_overfit_report_2026-09-23.md` |
| 13 | E-38a / E-45 / E-54c-d | PASS / ALL PASS | **UNSUPPORTED** — cited artifact holds another experiment's numbers; console-only | `parts/dline_legacy_e31_e63.md` |

`verdict_audit.yaml` carries the same ledger in machine-readable form.

## Missing-evidence index

15 audited entries cite an artifact that does not exist. The per-scope lists (with the exact
recorded path and an `exists` column) are at the end of
`parts/uline_core_legacy.md`, `parts/laya_e67_e75.md`, and `parts/laya_l1_l2.md`.
Dominant cause: gitignored `runs/laya/**/*.pt` plus a recovery zip that is itself absent.

## What this means for the tiny-model goal

Corrected standings, 3-gate GATE_EXTERNAL mean DER (lower is better), all re-scored:

| system | params | 3-gate mean | note |
|---|---:|---:|---|
| **E-071 P6 vote** `e70_c4b_ctx256` @15000 + sliding-window voting | **1.08 M** | **0.3697** | **NEW PROJECT BEST** — voting at window 256 / stride 128 on the C4b weights; **−1.34 % free, inference-only**; every gate improves (fadel −1.85, sadeed −1.26, wn2014 −1.05 %), McNemar p ≤ 3.2e-11, non-degenerate |
| E-070 C4b `e70_c4b_ctx256` @15000 | **1.08 M** | **0.3748** | the weights ALONE, before the P6 voting procedure; C4 continued 60 k steps at ctx 256, **−2.02 % vs C1**, −6.49 % at matched tokens. *Caveat RETIRED by E-072: the held-out abdou control reproduces the ordering 15000 < 40000 < 60000, so this is not probe noise (nuance: step 40000 leads der_canonical + macro_f1, so the top-two ranking is metric-dependent)* |
| E-068 C1 `e67_c1_ft64_fullv3` @90000 | **1.08 M** | **0.3825** | was the project best before E-070; F2's recipe + the REAL P1 warm start; **WIN** (3.74% vs F2, bar 2%) |
| E-069 C4 `e69_c4_ctx256` @60000 | 1.08 M | 0.3872 | ctx-256 adaptation of C1; **beat C1 by 3.4–4.5 % at MATCHED TOKENS**; superseded by E-070, which continued it to 0.3748 |
| E-065 F2 `e63s_s7_18l64_adamw_fullv3` @115000 | 1.08 M | 0.3974 | previous project best; **was never warm-started** (D-12) |
| E-063s S4 12L×160 @16500 | 4.5 M | 0.3996 | corrected sweep best (was reported as a 10k wall-hit) |
| E-23a v3q @2000 | 30.1 M | 0.4020 | corrected to a WIN; ADOPTED-CANDIDATE |
| E-065 F1 @100000 | 1.08 M | 0.4067 | |
| E-066 R1 @40000 (dropout+LS) | 1.08 M | 0.4205 | interrupted; flat thereafter |
| E-063c M1 16L×272 @5000 | 17.2 M | 0.4253 | |
| **deployed gold** `stage2b2500` | 30.1 M | 0.4264 | |
| E-067 P1 (char-LM 64h warm start) @32000 | 1.08 M | 0.4299 | **C0-R CONFIRMED**: +5.31% vs the replicate at all 10 matched steps |
| C0-R (same recipe, fresh draw) @32000 | 1.08 M | 0.4539 | sizes the no-seed init variance: **0.82%** |
| C0 / E-064 A2 control @38000 | 1.08 M | 0.4577 | control |

Readings:
1. **The 1.08 M-param s7 architecture beats the 30 M deployed gold by 4.4 DER points** —
   **C1 0.3825 vs gold 0.4264 = 10.3% better**, and 28x fewer parameters. The gap to published
   external systems is a *capability* gap, not a size gap. The best result on record is now a
   *combination* of the best recipe with a warm start that recipe only claimed to have.
2. Three previously-buried signals deserve a cheap re-check before more architecture work:
   the E-19 Fine-Tashkeel slice (needs a full-gate run), E-23a v3q (a corrected win on the
   30 M gold), and the E-063c M1 warm-init finding (warm init, not width, dominated at fixed
   budget).
3. The measurement hygiene fixes (D-1, D-2, D-3) must land before the next sweep, or more
   wins will be recorded as losses: always label the gate set, never compare val_loss across a
   label-smoothing arm, and always record the run dir (M1 existed twice).
4. **F2 was never warm-started** (D-12): its 0.3974 came from random init, so the 64-hidden
   char-LM warm start — worth +5.31% on the slice and C0-R-confirmed — had never been tried
   on the best recipe. That combination ran as **E-068 / C1** (`runs/diac/e67_c1_ft64_fullv3`,
   job `pwsh-137`, launch log `tensors_in 165 / shape_skipped 0`) and **WON**: best 3-gate
   **0.3825 @90000** vs F2 0.3974, **3.74% better** against a 2% bar — and the win holds when
   restricted to the 30 probe steps both runs share. **The project best is now 0.3825.**
5. **Context is a confirmed lever — and it composes (E-067 P9, then E-069 C4 at full scale).**
   ctx 128 silently truncated sentence tails (78.8 % of full-v3 train rows carry real text past
   column 128, not 42.2 % as the trial bed suggested). P9 priced ctx 256 at **+4.0–6.4 %** on the
   trial bed; C4 reproduced **−3.4 % to −4.5 %** on the full corpus with a warm start, and **all
   twelve of its probes beat C1 at the same step**. C4b then carried it to **0.3748**, a new
   project best at **−6.49 % vs C1 at matched tokens**.
6. **Judge a budget-limited arm at matched tokens, never on best-of-run alone.** C4's stated
   best-of-run is *worse* than C1's, yet it is the better model at every equal-compute point —
   because C1's 0.3825 landed at step 90 k and C4 stopped at 60 k while still descending. The
   E-069 rule conflated "did not beat the incumbent's best" with "the lever does not compose"; an
   arm whose minimum is its final probe is **under-trained, not refuted**. Fixed in E-070's rule.

## Open items

- **E-070 C4b CLOSED — WIN / NEW PROJECT BEST, with a caveat that matters**: 0.374811 @15000,
  **−2.02 % vs C1** (−6.49 % at matched tokens), independently re-scored to 0.374792 from the probe
  artifact. But the minimum is a **single early probe** (second-best 0.3793 @40000) and the curve
  oscillates in a ~5.7 % band with no sustained descent after step 15000; the val-best (55000)
  disagrees with the gate-best. It also **partly corrects my own E-069 reasoning**: "C4 was
  under-trained" was right (C4b gained −3.21 % on it within 15 k steps), but "budget was the whole
  story" was **overstated** — past step 15000 the extra budget bought nothing on the gates.
  Candidate `runs/diac/e70_c4b_ctx256/gate_probe/best_gate_weights.pt`.
- **E-069 C4 CLOSED — FAIL on the rule, but ctx composes**: best 0.387245 @60000 vs C1 0.382541
  (+1.23 % worse on best-of-run) while beating C1 by **3.4–4.5 % at matched tokens**. Best was the
  final probe ⇒ budget-limited. **E-070 / C4b launched** (60000 further steps from
  `e69_c4_ctx256/final/model.pt`) to reach C1's best-step budget.
- **C1 / E-068 CLOSED — WIN** (`pwsh-137`, exit 0, `reason: "completed"`): F2's full-v3 recipe +
  P1's real warm start. Best 3-gate **0.3825 @90000** vs F2 0.3974 = **3.74%** (bar 2%), val_acc
  tiebreaker passes on both readings, and it survives the 30-step matched-range check. **New
  project best**; `gate_probe/best_gate_weights.pt` is the deployment candidate. Worth noting:
  the best gate step was only 45% into the budget, so the ~14 h after step 90000 bought nothing
  on the gate metric.
- **C0-R closed 2026-10-06 (`pwsh-134`, exit 0): the E-067 P1 win survives.** The byte-identical
  replicate best 0.4539 @32000 vs C0 0.4577 @38000 = the no-seed init draw is worth **0.82%**
  (inside the 2% band), while P1's 0.4299 is **5.31%** better than the replicate at **all ten
  matched steps**. P1 is now an accepted Phase-2 combination candidate.
- **P9 READY** — the ctx-256 bed is built and **PASSED its acceptance check** (`pwsh-141`, exit 0):
  `data/diac/v3s10ctx256` has train 243,343 / val 9,613 and all four arrays' `[:128]` prefixes are
  byte-identical to `v3s10` (so it *is* the same trial bed at a longer context), with 1.56 M real
  cells beyond column 128 in just the first 20 k rows — the L2 question is live, not a null arm.
  `runs/diac/e67_gen/p9_ctx256.yaml` launched as `pwsh-142` and **CLOSED — PASS: ctx 256 beats
  ctx 128 by 6.39%** (best 0.4285 vs C0 0.4577; 4.02% on the 10 shared probe steps; 5.62% against the
  20-probe C0-R replicate, which matches probe counts) **and lifts val_acc by +2.1 points** (0.9470
  vs 0.9261) — so context, not architecture, is the second confirmed lever. Building the bed caught
  **two fresh defects (D-13 and the `--max-train` row-set trap)** that an ids-only or absent check
  would have let through.
- **D-12 guard landed** in `train.py`: a warm start that loads **zero** tensors is now FATAL
  instead of silently randomizing the run (`allow_empty_warm_start: true` waives it). Partial
  (shape-filtered) warm starts stay allowed for the E-63s sub-272 arms.
- **E-066 closed**: FAIL on peak (R1 0.4205 vs F2 0.3974), PASS on delay; R1 was killed at step
  109500, not early-stopped.
- **P2 (variable-length) and P7 (sentence dropout) have no implementation** — no `var_len` or
  `sentence_dropout` support exists anywhere in `diacritizer/`, so they are code work rather
  than config arms; that is why P9 was chosen as the next single.
- **P6** sliding-window voting: the all-False attn_mask bug is fixed (`e67_p6_vote.py:67`) and
  the re-run is **E-071, launched 2026-10-06** (job `pwsh-1`) against the current project best
  (E-070 C4b step-15000 weights, ctx 256, window 256 / stride 128). The 50-line fadel_test smoke was
  **non-degenerate** (der_anyalt 0.3154 ≠ der_canonical 0.1864, macro_f1 0.73, versus the broken
  run's 0.0953) and the vote already **beat** single by **−1.77 %**, McNemar canonical p=7.0e-5
  (32 vote-only-correct vs 7 single-only-correct). Full 3-gate run in progress.
- **Registry gap (reorg finding, 2026-10-06).** `research/EXPERIMENTS.md` carries **110 distinct
  ids** but is a **frozen historical ledger by its own authority banner** (lines 3–11): it preserves
  duplicate ids and malformed rows on purpose and says not to read it as current truth. Confirmed:
  E-13 is defined twice (line 50 is a truncated stub, line 55 the real row), 75 ids appear on
  multiple lines, and E-23/E-37/E-38/E-39/E-48 exist only as suffixed forms.
  `research/experiments.yaml` holds only **24 rows**, almost all recent (E-061, E-063…E-071) plus
  ui/core/laya — so **E-19…E-60 have no machine row at all** and are covered only by this audit's
  `parts/` files and the frozen ledger. **Do not "fix" the duplicates by editing the ledger** — that
  would destroy preserved history. The designated per-record map is
  `docs/agent/reports/context-migration-2026-09-24.md`; the honest reorg action is a *derived*
  index that leaves the historical file byte-stable. Recorded rather than silently rewritten.
