# Laya decision-model line — verdict audit (E-66 … E-75)

Scope: the Laya L3 rung ledger in `research/EXPERIMENTS.md` (E-66 verdict through E-75 pre-register) plus registry rows `laya/E-074` and `laya/E-075` in `research/experiments.yaml`.
This is the Laya decision-model line, unrelated to the Arabic diacritizer E-067 row.

Method: for every recorded verdict, the cited artifact was opened directly. A number that
appears only in a narrative document (`research/EXPERIMENTS.md`, `research/2026-09-23_laya_l3_frontier.md`, `TASKS.md`)
is not evidence: per `rules/evidence.md` an artifact is a metric file, console tee, or gate-bench output.
Metric directions for this line (all four gates HIGHER-is-better; `laya/scripts/eval_l3.py` accepts a gate with `>=`):
G1 mixture retention macro, G2 typed accuracy, G3 permutation agreement, G4 zero-shot phishing AUROC, probe pass count.
LOWER-is-better in this line: loss, held-out MLM perplexity, Brier, ECE.

---

## E-66 — perm-duplicate option aug [PARTIAL]
Recorded (`research/EXPERIMENTS.md:949-959`): G2 typed 0.5630 (>= 0.55), G3 perm 0.2150 (< 0.90 FAIL), G4 phish 0.6086 (>= 0.60), G1 macro 0.6596 (-0.003), Brier 0.1637, probes regressed 2 -> 4.
Cited artifacts: `runs/laya/l3_e66/{train_summary.json,eval_report.json,bench_log.jsonl}`, `runs/laya/l3_e66_console.log`.
Recheck: **NO_ARTIFACT**. `runs/laya/l3_e66/` does not exist; `runs/laya/l3_e66_console.log` does not exist. No file in the repository contains these numbers except narrative prose. Direction is used correctly as written (higher-is-better gates), so no inversion.
Corrected verdict: unverifiable as recorded — the entire E-66 artifact set is gone; only prose survives.

## E-67 — eval-side calibration (perm-avg + position prior) [FAIL]
Recorded (`research/EXPERIMENTS.md:975-981`): identity anchor 0.5630 / soft 0.4334 / brier 0.1637; perm_avg acc 0.5615, soft 0.4693, brier 0.1734, G3 0.2250; perm_avg+prior acc 0.5345, G3 0.2350. FAIL on both gates.
Cited artifact: `runs/laya/l3_e66/eval_calibrated.json` (+ `runs/laya/l3_e66/{bench_log.jsonl,e67_console.log}`).
Recheck: **NO_ARTIFACT**. None of those files exist. Direction is correct: 0.2250/0.2350 are below the 0.90 G3 bar and 0.5345 is below the 0.5630 non-regression bar, and brier 0.1734 > 0.1637 is correctly read as worse (lower-is-better). No inversion.
Corrected verdict: FAIL is directionally sound but unverifiable — the cited calibration artifact is absent.

## E-68 — stage-B length 4 -> 8 epochs [WIN]
Recorded (`research/EXPERIMENTS.md:964-968` pre-register; `:998` scoreboard): typed 0.6530 (+9 over E-66), Brier 0.1382, probes recovered 4 -> 2; `runs/laya/l3_e68/final/model.pt` named line-best.
Cited artifacts: `runs/laya/l3_e68/{train_summary.json,eval_report.json,bench_log.jsonl}`.
Recheck: **NO_ARTIFACT**. `runs/laya/l3_e68/` does not exist. Direction correct (0.6530 >= 0.5630, Brier 0.1382 < 0.1637).
Corrected verdict: WIN unverifiable — no E-68 metric artifact survives.

## E-69 — L2 -> L3 distillation [FAIL]
Recorded (`research/EXPERIMENTS.md:985-992`): G2 typed 0.5490 (< 0.6530 and < 0.5630), G4 phish 0.6098 (>= 0.60), G1 0.6593 held, G3 0.2000, probes 4.
Cited artifact: `runs/laya/l3_e69/eval_report.json`.
Recheck: **NO_ARTIFACT**. `runs/laya/l3_e69/` does not exist. Direction correct (0.5490 below the higher-is-better bars).
Corrected verdict: FAIL directionally sound, unverifiable — cited eval report absent.

## E-70 — token scale 260M -> ~520M [PARTIAL: primary FAIL, transfer WIN]
Recorded (`research/EXPERIMENTS.md:1009-1017`): G2 typed 0.6420 (< 0.6530 FAIL); G4 phish 0.6966 (beats published Laya 0.678 / Jev 0.689); G1 0.6700 vs stage-A exit 0.6668 (delta +0.0032 PASS); G3 0.2000; Brier 0.1415; probes 4; held-out ppl 10.97 -> 8.12.
Cited artifacts: `runs/laya/l3_e70/{train_summary.json,eval_report.json,bench_log.jsonl}`, `runs/laya/l3_e70_pretrain/{bench_log.jsonl,pretrain_summary.json}`.
Recheck: **NO_ARTIFACT**. `runs/laya/l3_e70/` does not exist. `runs/laya/l3_e70_pretrain/` survives only as `pretrain_final/encoder/{config.json,model.safetensors,tokenizer.json,tokenizer_config.json}` — no `pretrain_summary.json` and no `bench_log.jsonl`, so even the perplexity trend and the "encoder beat published phish" claim have no surviving artifact. Directions correct (typed higher-is-better fail; AUROC higher-is-better win; ppl lower-is-better improvement).
Corrected verdict: PARTIAL is directionally sound but unverifiable — neither the ladder report nor the pretrain summary survives.

## E-71 — per-source balanced replay [PARTIAL PASS]
Recorded (`research/EXPERIMENTS.md:1037-1044`): G2 typed 0.6705 (>= 0.6530 PASS), G4 phish 0.6690 (vs >= 0.68, MISS by 0.011), G1 delta +0.0048, probes 4, G3 0.2000.
Cited artifacts: `runs/laya/l3_e71/{train_summary.json,eval_report.json,bench_log.jsonl}`.
Recheck: **NO_ARTIFACT**. `runs/laya/l3_e71/` does not exist. Direction correct (0.6690 < 0.68 is a miss on a higher-is-better gate).
Corrected verdict: PARTIAL PASS directionally sound, unverifiable — cited eval report absent.

## E-72 — stage-B length 8 -> 12 epochs [PASS]
Recorded (`research/EXPERIMENTS.md:1019-1031`): G2 typed 0.6965 (>= 0.6530 PASS), G1 delta +0.0034, G4 phish 0.6092, probes 4, Brier 0.1312.
Cited artifact: `runs/laya/l3_e72/eval_report.json` (+ `runs/laya/l3_e72/final/model.pt`).
Recheck: **NO_ARTIFACT** for the numbers, with a **FALSE-POSITIVE overclaim** on the phishing component.
- `runs/laya/l3_e72/` does not exist; no E-72 metric artifact survives.
- The headline "PASS on primary (typed) + retention + **phishing bar**" misstates its own pre-register (`research/EXPERIMENTS.md:1018`), which set "G4 phishing >= 0.60 **with E-70's 0.6966 as the transfer bar to hold**" and warned "if phishing drops below 0.68 the length knob re-confirms the tension". The recorded 0.6092 clears only the 0.60 floor and misses the held transfer bar by 0.087; the same verdict paragraph admits "trade-off watch breached". The nearest surviving measurement of E-72 through the sym path is `runs/laya/phish_eval_sym.json` (`l3_e72` auroc 0.6024616), which is explicitly informational-only and not the native positional number.
Corrected verdict: primary typed PASS is plausible but unverifiable; the "phishing bar" PASS is an overclaim — the registered transfer bar was missed.

## E-73 — balanced replay + transfer-guarding early stop [WIN]
Recorded (`research/EXPERIMENTS.md:1048-1055`): stop at ep7; peak phish 0.7161 @ ep5, last 0.6932; G4 0.6932 (>= 0.68 PASS), G2 typed 0.6560 (>= 0.6530 PASS by 0.003), G1 delta -0.0025, probes 4, G3 0.2000.
Cited artifacts: `runs/laya/l3_e73/{train_summary.json,eval_report.json,bench_log.jsonl}`.
Recheck: **NO_ARTIFACT**. `runs/laya/l3_e73/` does not exist. Direction correct (both gates higher-is-better and both just above their bars).
Corrected verdict: WIN directionally sound, unverifiable — cited eval report; the two-gate margin is only 0.003 on typed, so it cannot be confirmed without the artifact.

## E-74 — phase-1, position-symmetric scorer [BREAKTHROUGH]
Recorded (`research/EXPERIMENTS.md:1095-1103`): "TYPED 0.7712 (grouped decision acc on the standing unseen test n=2000) - ABOVE the published Laya-FT 0.766"; "Brier 0.1009 (best ever; positional line-best 0.1312)"; "G3 perm-agreement: BY CONSTRUCTION"; "curve STILL RISING at ep8 (+0.025 ep7->8)".
Cited artifacts: `runs/laya/l3_e74_console.log`; `runs/laya/l3_e74/{A_last,B_last}.pt` + `final/model.pt`; `data/laya/sym/typed_test.pt`.
Recheck: **MISCITED — FALSE POSITIVE**. The surviving artifact does not support the headline.
- `runs/laya/l3_e74_console.log:1220,1248,1275,1303,1331,1358,1386,1414`: every stage-B epoch eval line is `"n": 6000`, ending at `epoch 8 eval {"acc": 0.7712, "soft_acc": 0.6433, "brier": 0.1009, ...}`.
- `laya/scripts/train_l3_sym.py:201` sets `gt, gm = groups_of(typed_train), groups_of(mixture_held)` and `:207` runs stage B as `run_stage(model, cfg, typed_train, gt, ...)` — i.e. the epoch curve, including epoch 8, is measured on the typed **TRAIN** split. `:189` loads `typed_test` but never evaluates it. Verified against the E-74 landing commit `5bf6bcb` ("E-74 symmetric trainer live"): the same two lines were present when the verdict was written.
- `data/laya/typed_decisions/build_stats.json`: train `"items": 6000`, test `"items": 2000` — so n=6000 is the train split and n=2000 is the test split. The verdict's own parenthetical cites the 2,000-qid grouping audit of `data/laya/sym/typed_test.pt` but attaches it to a 0.7712 that the console computed on 6,000 train qids.
- No test-split artifact for E-74 exists at all: there is no `runs/laya/l3_e74/eval_report.json` (the directory holds only `A_last.pt`, `B_last.pt`, `final/model.pt`, and a tfevents file) and no `laya/scripts/eval_l3_sym.py` exists (only `train_l3_sym.py`, `eval_phish_sym.py`, `eval_public_choice_sym.py`). The promised "verify re-run" of the unseen test is not an artifact anywhere in the repository.
- Consequently "Brier 0.1009 best ever" is a train Brier compared against a positional test Brier (0.1312); the comparison is not like-for-like. "Perm-agreement BY CONSTRUCTION" was pre-registered as a measured gate `>= 0.40` (`research/EXPERIMENTS.md:1060`) and has no measurement. "Curve still rising at ep8" is a train curve, so it is not evidence of test headroom (this same statement is what `state/tasks.yaml` `laya/task-e75-r2` and `TASKS.md:229` record as the R2 justification).
Corrected verdict: the "typed 0.7712 on the unseen test n=2000 ABOVE Laya-FT 0.766" breakthrough is supported only by a TRAIN-split grouped eval (n=6000); no test artifact exists, so the breakthrough is UNSUPPORTED and, as cited, MISCITED.

## E-74 — phase-2a, zero-shot phishing [trade is real]
Recorded (`research/EXPERIMENTS.md:1104-1109`): `l3_e74_sym` raw 0.438 / Platt 0.549 / AUROC 0.414 / ECE 0.513; "far below the positional arm's standing 0.6965".
Cited artifact: `runs/laya/phish_eval_sym.json` (+ `runs/laya/phish_eval_sym_console.log`).
Recheck: **CONFIRMED for the collapse numbers; MISCITED comparator.**
- `runs/laya/phish_eval_sym.json`: `l3_e74_sym` raw_acc_test_half 0.438, platt 0.549, `auroc_test_half 0.4138626218019488`, `ece15 0.5129291370473803`, mean_p 0.19424; console `runs/laya/phish_eval_sym_console.log:25`. All match to rounding. AUROC higher-is-better; 0.414 < 0.5 is correctly read as collapse.
- The comparator "positional arm's standing 0.6965" is wrong: 0.6965 is E-72's **typed accuracy** (`research/EXPERIMENTS.md:1023`), not a phishing AUROC. The positional phishing line-best is E-70's 0.6966 (`:1012`), itself artifact-less. The surviving sym-mode row for E-72 is 0.6024616 (`runs/laya/phish_eval_sym.json`), which the verdict itself flags as informational only.
Corrected verdict: collapse CONFIRMED; the positional phish comparator is MISCITED (a typed number quoted as phishing).

## E-74 — phase-2b, AG News / emotion [choice-transfer parity]
Recorded (`research/EXPERIMENTS.md:1110-1116`): `l3_e74_sym` ag_news 0.253 / ece 0.092, emotion 0.225 / ece 0.123; positional comparators l1 0.422 / 0.291, l2_stageA 0.232 / 0.329, l2_final 0.269 / 0.366.
Cited artifacts: `runs/laya/public_choice_eval_sym.json` + console; `runs/laya/public_choice_eval.json`.
Recheck: **CONFIRMED** to rounding.
- `runs/laya/public_choice_eval_sym.json`: ag_news `l3_e74_sym` 0.2525 / ece 0.09220; emotion 0.2245 / 0.12284; n=2000 each.
- `runs/laya/public_choice_eval.json`: ag_news `l1_final` 0.4225, `l2_stageA` 0.232, `l2_final` 0.2685; emotion `l1_final` 0.291, `l2_stageA` 0.329, `l2_final` 0.3655.
- Minor wording note: the sym arm is slightly below the positional l2_final on both tasks (0.2525 vs 0.2685; 0.2245 vs 0.3655), so "parity" is generous, but both are near chance and the file's own caveat ("vendor prompts unpublished; approximate comparison") applies.
Corrected verdict: CONFIRMED — choice panel is chance-level for both arms; the "not a regression" framing is acceptable.

## Registry row `laya/E-074` (`research/experiments.yaml:64-91`)
Recorded: status `closed`, verdict `partial`, note "Phase-1 breakthrough (typed 0.7712 above published Laya-FT, Brier 0.1009, perm-invariance by construction) but phase-2 zero-shot phish collapsed (AUROC 0.414)".
Recheck: **MISCITED** — the overall `partial` verdict is defensible, but the note's phase-1 "breakthrough" rests on the train-split n=6000 metric, and its cited evidence list includes neither a test-typed artifact nor a phase-2 marker.
Corrected verdict: downgrade the note's phase-1 clause to "typed measured on the typed-train split (n=6000), test n=2000 not measured; phish collapse confirmed".

## E-75 — new-corpus token scaling [pre-register only]
Recorded (`research/EXPERIMENTS.md:1063-1067`; registry `laya/E-075` at `research/experiments.yaml:92-116`): status `planned`, verdict `none`. Gates typed >= 0.6965 (E-72 line-best) OR phish >= 0.70 + typed >= 0.6560; evidence-at-pretrain-end held-out ppl < 8.12.
Recheck: **CONFIRMED** — no result is claimed, so there is nothing to invert or overclaim. The cited evidence `runs/laya/l3_e70_pretrain` is only partially present (encoder files; no `pretrain_summary.json`/`bench_log.jsonl`), so the ppl < 8.12 baseline is prose-only.
Corrected verdict: pre-register stands; its inherited baseline (<8.12 ppl, E-72 typed 0.6965) is not artifact-backed.

---

## Final table

| id | recorded verdict | metric + direction | cited artifact | recheck | corrected one-line verdict |
|---|---|---|---|---|---|
| E-66 | PARTIAL | G2 typed / G3 perm / G4 AUROC / G1 macro (higher better); Brier (lower better) | `runs/laya/l3_e66/eval_report.json` + console | NO_ARTIFACT | Numbers survive only in prose; direction correct, unverifiable. |
| E-67 | FAIL | G3 perm (higher better), G2 non-regression (higher better), Brier (lower better) | `runs/laya/l3_e66/eval_calibrated.json` | NO_ARTIFACT | FAIL directionally sound; calibration artifact absent. |
| E-68 | WIN | G2 typed (higher better), Brier (lower better), probes (higher better) | `runs/laya/l3_e68/eval_report.json` | NO_ARTIFACT | WIN unverifiable; run dir absent. |
| E-69 | FAIL | G2 typed (higher better), G4 AUROC (higher better), G1 macro (higher better) | `runs/laya/l3_e69/eval_report.json` | NO_ARTIFACT | FAIL directionally sound; eval report absent. |
| E-70 | PARTIAL | G2 typed (higher better) FAIL; G4 AUROC (higher better) WIN; ppl (lower better) | `runs/laya/l3_e70/eval_report.json`; `runs/laya/l3_e70_pretrain/pretrain_summary.json` | NO_ARTIFACT | Directionally sound; neither the ladder report nor the pretrain summary survives. |
| E-71 | PARTIAL PASS | G2 typed (higher better) PASS; G4 AUROC (higher better) MISS | `runs/laya/l3_e71/eval_report.json` | NO_ARTIFACT | Directionally sound; eval report absent. |
| E-72 | PASS | G2 typed (higher better) PASS; G4 AUROC (higher better) | `runs/laya/l3_e72/eval_report.json` | NO_ARTIFACT | Typed PASS unverifiable; "phishing bar" PASS is a FALSE POSITIVE — the pre-registered hold bar 0.6966 was missed by 0.087. |
| E-73 | WIN | G4 AUROC (higher better); G2 typed (higher better) | `runs/laya/l3_e73/eval_report.json` | NO_ARTIFACT | Directionally sound; eval report absent and typed margin is only 0.003. |
| E-74 phase-1 | BREAKTHROUGH | typed acc (higher better), Brier (lower better), perm agreement (higher better) | `runs/laya/l3_e74_console.log`; `runs/laya/l3_e74/final/model.pt`; `data/laya/sym/typed_test.pt` | MISCITED | 0.7712 / Brier 0.1009 are typed-TRAIN (n=6000) grouped evals (`train_l3_sym.py:201,207`); no test n=2000 artifact exists — the published-ceiling breakthrough is unsupported. |
| E-74 phase-2a | trade is real (collapse) | G4 AUROC (higher better) | `runs/laya/phish_eval_sym.json` | CONFIRMED | Collapse CONFIRMED (0.414 / raw 0.438 / Platt 0.549 / ECE 0.513); comparator "0.6965" is MISCITED (that is E-72's typed acc). |
| E-74 phase-2b | choice-transfer parity | acc (higher better), ECE (lower better) | `runs/laya/public_choice_eval_sym.json`; `runs/laya/public_choice_eval.json` | CONFIRMED | Rows match to rounding; both arms at chance. |
| `laya/E-074` | partial (registry) | typed / AUROC | `runs/laya/l3_e74_console.log` | MISCITED | Keep `partial`, but phase-1 is a train-split measurement, not an unseen-test breakthrough. |
| E-75 / `laya/E-075` | none (planned) | typed / AUROC / ppl | `runs/laya/l3_e70_pretrain` (partial) | CONFIRMED | Pre-register only, no result claimed; inherited baselines are prose-only. |

Recheck counts (13 entries): CONFIRMED 3 — phase-2a collapse, phase-2b choice, E-75; NO_ARTIFACT 8 — E-66, E-67, E-68, E-69, E-70, E-71, E-72, E-73; MISCITED 2 — E-74 phase-1, `laya/E-074`; INVERTED 0.

---

## FALSE-NEGATIVE check (recorded FAIL/WORSE that is really a win)
None found in this scope. Every gate in the Laya line is compared in the correct direction:
higher-is-better for G1 macro, G2 typed, G3 agreement, G4 AUROC and probe pass count, and
lower-is-better for Brier / ECE / loss / perplexity (`laya/scripts/eval_l3.py` selects a passing gate with `>=`;
`laya/scripts/train_l3_sym.py:65-99` tracks `acc` upward and `brier` downward). Each recorded FAIL was re-derived from the prose numbers as a genuine
below-bar result: E-67 G3 0.2250/0.2350 < 0.90; E-69 typed 0.5490 < 0.6530; E-70 typed 0.6420 < 0.6530;
E-71 phish 0.6690 < 0.68; E-74 sym phish AUROC 0.414 < 0.5. E-72's phish 0.6092 is recorded as a PASS
and is the one place where the direction/baseline reading flatters the run, not penalises it.

## FALSE-POSITIVE findings (recorded better than the artifact supports)
1. **E-74 phase-1 "typed 0.7712 on the unseen test n=2000, above Laya-FT 0.766"** — FALSE POSITIVE.
   Evidence: `runs/laya/l3_e74_console.log:1414` (and every stage-B epoch line) reports n=6000;
   `laya/scripts/train_l3_sym.py:201,207` (same at commit `5bf6bcb`) evaluate stage B on `groups_of(typed_train)`;
   `data/laya/typed_decisions/build_stats.json` gives train 6000 / test 2000; no `runs/laya/l3_e74/eval_report.json` exists.
   Propagated into `research/experiments.yaml` (`laya/E-074`), `TASKS.md:229,232,236`, and `state/tasks.yaml` (`laya/task-e75-r2`).
2. **E-72 "PASS ... phishing bar"** — FALSE POSITIVE (partial). Its pre-register
   (`research/EXPERIMENTS.md:1018`) required holding E-70's 0.6966 transfer bar and flagged <0.68 as
   re-confirming the tension; recorded 0.6092 clears only the 0.60 floor, and the cited `eval_report.json` is absent.
3. **E-74 phase-2a comparator "positional arm's standing 0.6965"** — MISCITED. 0.6965 is E-72's typed
   accuracy (`research/EXPERIMENTS.md:1023`); the positional phish line-best is E-70's 0.6966 (`:1012`),
   which has no artifact.
4. **E-74 phase-1 "perm-agreement BY CONSTRUCTION"** — MISCITED. Pre-registered as a measured gate
   `>= 0.40` (`research/EXPERIMENTS.md:1060`); no measurement exists, and the standing G3 record was 0.185-0.235.

## Not checkable, and why
- `runs/laya/l3_e66/`, `runs/laya/l3_e68/`, `runs/laya/l3_e69/`, `runs/laya/l3_e70/`, `runs/laya/l3_e71/`, `runs/laya/l3_e72/`, `runs/laya/l3_e73/` and the console tees `runs/laya/l3_e66_console.log`, `runs/laya/l3_e6{8,9}_console.log`, `runs/laya/l3_e7{0,1,2,3}_console.log` do not exist on disk. `runs/laya/` holds only 22 files, of which the L3 survivors are `l3/{bench_log.jsonl,eval_report.json,pretrain_summary.json,train_summary.json}` (E-65), `l3_e74_console.log`, the three `*_sym` json/console pairs, `l3_e74/{A_last,B_last,final/model}.pt` + one tfevents, and the `l3_e70_pretrain/pretrain_final/encoder/` files.
- The missing reports were never committed: `git log -- runs/laya/l3_e72` is empty, and weights/run data are gitignored (`.gitignore:162-163`: `runs/laya/**/*.pt`, `data/laya/`). `runs/laya/l3_e74/` and `runs/laya/l3_e70_pretrain/` are untracked working-tree leftovers.
- `research/2026-09-23_laya_l3_frontier.md` repeats the E-66..E-73 scoreboard and claims its console logs and each out_dir's `{train_summary.json,eval_report.json}` exist; that claim is false for all of them. The report is prose, not an artifact.
- A backup that once held some of these files, `checkpoint_backup/laya_l3_e74_resume_2026-09-23.zip`, is itself absent from `checkpoint_backup/` (only referenced in `scratch/_backup_extract_report.json`).
- Everything in this audit is CPU read-only: no GPU, training, inference, or data regeneration was run.

---

Audit limitation: the four gates are artifact-verifiable only for E-65 (`runs/laya/l3/eval_report.json`)
and, in an in-training only sense, for E-74. For E-66..E-73 the audit outcome is an evidence-integrity
finding (NO_ARTIFACT), not a direction finding; the only direction/baseline error in scope is the E-74
train-vs-test misattribution and the E-72 phishing-bar overclaim.
