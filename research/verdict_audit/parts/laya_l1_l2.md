# Laya L1/L2 + registry verdict audit (E-62 … E-65) & missing-evidence index

Scope (per request): the Laya rows preceding the E-66…E-75 window — L1 smoke (E-62),
L2 ladder (E-63), the zero-shot shared-benchmark eval (E-64), the L3 from-scratch rung
(E-65) — plus registry rows `laya/E-074` and `laya/E-075` in `research/experiments.yaml`.
Read-only CPU audit; no existing file was modified.

**ID-collision warning.** `research/EXPERIMENTS.md` is a historical ledger with duplicate IDs
(its own authority banner, lines 3-11, says so). In the lines 858-948 region:
- `E-61` (line 874) and `E-63` (lines 881-901) are **diacritizer** rows (`diacritizer/E-061`, `diacritizer/E-063`; canonical pointer table lines 17-18).
- The **Laya** rows are `E-62` (L1, line 909), `E-63` (L2, line 917), `E-64` (line 919), `E-65` (line 930).
So "Laya E-61…E-65" is really Laya E-62…E-65; there is no Laya E-61.

Directions: accuracy / AUROC / macro / spaCy-style pass counts HIGHER-is-better; ECE, Brier,
loss, perplexity, VRAM ceiling, wall ceiling LOWER-is-better (or bounded above); probe
"total_failures" LOWER-is-better. Every comparison below was checked against that convention.

---

## E-62 — L1 smoke (MiniLM-L12-H384 + Laya head) [PASS on G1+G2; G3 diagnostic FAIL]
Recorded (`research/EXPERIMENTS.md:909`): test acc 0.6205 (G2 >= 0.587 PASS), peak VRAM 1660 MiB (G1 <= 4096 PASS), wall 286 s, params 37.2M, soft 0.4889, Brier 0.1473, raw ECE 0.0673 -> temp ECE 0.0948 (G3 diagnostic FAIL), per-qtype noul 0.695 / choice 0.597 / score 0.583, per-workflow agent_trace 0.528.
Cited artifacts: `runs/laya/l1/train_summary.json`, `runs/laya/l1/eval_report.json`; report `research/2026-09-22_laya_l1.md`.
Recheck: **NO_ARTIFACT (numbers also UNSUPPORTED as metric evidence).** `runs/laya/l1/` does not exist (`Test-Path runs/laya/l1/train_summary.json` = no; `runs/laya/l1/eval_report.json` = no; `runs/laya/l1/console.log` and `runs/laya/l1/logs/` cited inside the report are also absent). The numbers survive only inside the narrative report `research/2026-09-22_laya_l1.md`, which is prose, not a metric artifact.
Direction: **correct.** 0.6205 >= 0.587 is a higher-is-better pass; 1660 <= 4096 MiB is a ceiling pass; ECE 0.0673 -> 0.0948 is correctly read as worse (lower-is-better). No inversion.
Sub-note (prose vs prose): the same report gives "~41M total" in its Rig section (line 31-33) but "37.2M" in its results table (line 49) and carries the ledger's 37.2M; the L2 artifact measures the identical encoder+head at 37.16M (`runs/laya/l2/train_summary.json`). The ~41M figure is the outlier and is unverifiable without the missing L1 artifact.
Corrected verdict: E-62 may well be a real PASS, but no cited metric file exists; treat as NO_ARTIFACT / prose-only.

## E-63 — L2 mixture pretrain -> typed fine-tune [FAIL overall; G2 PASS, G1/G3 FAIL]
Recorded (`research/EXPERIMENTS.md:917`): G2 typed 0.6585 >= 0.60 PASS (+3.8 over L1 0.6205); G1 post-stage-B macro 0.6856 FAIL vs 0.74 but stage-A exit 0.7478 over the bar; G3 agreement L2 0.385 / L1 0.325 << 0.95 FAIL; soft 0.509, Brier 0.139; calibration T=1.08, ECE 0.0888 -> 0.1282; stage-B forgetting yelp5 0.512 -> 0.22, ANLI 0.744 -> 0.674.
Cited artifacts: `runs/laya/l2/eval_report.json`, `runs/laya/l2/train_summary.json` (both EXIST); report `research/2026-09-22_laya_l2.md`.
Recheck: **CONFIRMED.** Every number matches its artifact:
- `runs/laya/l2/eval_report.json` `typed_test.acc` 0.6585 (n 2000) and `gates.G2_typed_acc.verdict` PASS; `mixture_heldout.macro` 0.6856 and G1 FAIL; `probe_permutation.agreement_l2` 0.385 / `agreement_l1` 0.325; `calibration` T 1.0797, ECE 0.0888 -> 0.1282; `typed_test.soft_acc` 0.5087 / `brier` 0.1394; `yelp5` 0.22, `anli` 0.674; top-level `verdict` "FAIL".
- `runs/laya/l2/train_summary.json` `stageA.eval_curve[2].macro` 0.7478 (stage-A exit) and per-source yelp5 0.512 / anli 0.744; stage-B epoch4 acc 0.6585.
Direction: **correct** on every gate (higher-is-better acc/macro/agreement; lower-is-better ECE), and the gate is correctly judged on the post-stage-B model rather than the friendlier stage-A exit.
Corrected verdict: CONFIRMED as recorded (honest overall FAIL with a validated ladder).

## E-64 — zero-shot shared-benchmark eval (PhishNChips / AG News / emotion / probes) [no gates; no published-benchmark win]
Recorded (`research/EXPERIMENTS.md:919`): typed 0.6585 vs Jev 0.727 (-6.9) / Laya-FT 0.766 (-10.8); PhishNChips raw ~0.50, best AUROC 0.576 (stage A) vs Laya 0.678 / Jev 0.689; AG News 0.23-0.42; emotion 0.29-0.37; probes L1 = 1 failure, L2 final 3, stage A 8.
Cited artifacts: `runs/laya/phish_eval.json`, `runs/laya/public_choice_eval.json`, `runs/laya/probe_eval.json` (all EXIST); report `research/2026-09-22_laya_vs_jev_benchmarks.md`.
Recheck: **CONFIRMED**, all checkable values match:
- `runs/laya/phish_eval.json`: `l1_final.raw_acc_test_half` 0.498 (~0.50), `l2_stageA.auroc_test_half` 0.5756052 (0.576, the best), `l2_final.auroc_test_half` 0.3596; `published` Laya 0.678 / Jev 0.689.
- `runs/laya/public_choice_eval.json`: ag_news 0.4225 / 0.232 / 0.2685 -> "0.23-0.42" and emotion 0.291 / 0.329 / 0.3655 -> "0.29-0.37".
- `runs/laya/probe_eval.json`: `total_failures` L1 1, L2 stage A 8, L2 final 3.
Direction: **correct** (AUROC/acc higher-is-better -> below published reads as "no win"; probe failures lower-is-better -> L1 = 1 reads as best). No inversion.
Corrected verdict: CONFIRMED — the recorded "no published-benchmark wins, probe-profile win for L1" holds in the artifacts.

## E-65 — L3 from-scratch 35.1M encoder + replay/augmented ladder [1/4 gates PASS]
Recorded (`research/EXPERIMENTS.md:930-941`): G4 phish AUROC 0.6490 >= 0.60 PASS; G1 0.6622 FAIL (bar 0.74; L2 0.6856); G2 0.5185 FAIL; G3 0.1850 FAIL; stage-A exit 0.6625 vs post-B 0.6622; params 35.1M encoder / 41.77M with head; AG News 0.245, emotion 0.315, probes 2; corpus 260.7M train / 5.32M heldout, 2,194,311 docs, 20,004 eval-exclusion hashes, dropped_eval_overlap 0; peak 4,610 MiB (FAIL vs 4,096); wall 79 min; 3 sources skipped+recorded.
Cited artifacts: `runs/laya/l3/{pretrain_summary,train_summary,eval_report}.json`, `runs/laya/l3/bench_log.jsonl`, `data/laya/l3_corpus/build_stats.json` (all EXIST); report `research/2026-09-22_laya_l3.md`.
Recheck: **CONFIRMED on all primary numbers**, with **one MISCITED comparator**:
- `runs/laya/l3/eval_report.json`: G1 0.6622/bar 0.74/false, G2 0.5185/bar 0.6/false, G3 0.185/bar 0.9/false, G4 0.6489703835261365/bar 0.6/true; `probes.total_failures` 2; `public_choice` ag_news 0.245, emotion 0.315; `summary` 1/4 FAIL.
- `runs/laya/l3/train_summary.json`: stage-A ep5 macro 0.6625, stage-B ep4 acc 0.5185 and phish 0.64897 — so the stage-A-exit vs post-B contrast holds.
- `runs/laya/l3/pretrain_summary.json`: params 35.1, steps 31821, tokens 260.7M, wall 4735.2 s (= 78.9 min), peak 4610.0 MiB.
- `data/laya/l3_corpus/build_stats.json`: 260.7 / 5.3M tokens, 2,194,311 docs, `eval_exclusion_hashes` 20004, every `dropped_eval_overlap` 0, exactly 3 `FAIL_*` sources (spamassassin, sroie, bitext) — matching the ledger's source list (Enron 118.7k, Enron-spam 28.3k, Kaggle phishing 8.35k, CLAIR 6.8k, support Twitter 644k, tickets 53.5k, LocalLLaMA 1.19M).
- **MISCITED:** the G4 bullet (`:932`) compares against "our L1 0.576 (+0.073)". 0.576 is the L2 **stage-A** AUROC (`runs/laya/phish_eval.json` `l2_stageA.auroc_test_half` 0.5756); the L1 AUROC is 0.44595. The comparator label ("our L1") is wrong; the value belongs to L2 stage A.
Direction: **correct** throughout (G4 AUROC higher-is-better and 0.6490 clears 0.60; G1/G2/G3 higher-is-better and below bar; VRAM ceiling 4610 > 4096 is a genuine FAIL; wall ceiling 79 min < 12 h).
Note (not an inversion): a later delta-rule framing adopted at E-70 would call this retention a pass (-0.0003 vs stage-A exit), but the E-65 pre-registered bar was absolute (>= 0.74), so FAIL is correct as recorded.
Corrected verdict: CONFIRMED (1/4) with the "our L1 0.576" comparator corrected to "L2 stage-A 0.576".

## Registry row `laya/E-074` (`research/experiments.yaml:64-91`) — RE-AUDITED
Recorded: status `closed`, verdict `partial`; note "Phase-1 breakthrough (typed 0.7712 above published Laya-FT, Brier 0.1009, perm-invariance by construction) but phase-2 zero-shot phish collapsed (AUROC 0.414)".
Recheck: **MISCITED.** Every cited evidence path exists (`research/EXPERIMENTS.md`, `runs/laya/l3_e74_console.log`, `runs/laya/l3_e74/A_last.pt`, `runs/laya/l3_e74/final/model.pt`, `research/2026-09-23_laya_l3_frontier.md`, `TASKS.md`), but the phase-1 "breakthrough" rests on a typed-TRAIN (n=6000) console eval mislabeled as the unseen test n=2000 (`laya/scripts/train_l3_sym.py:201,207`; `runs/laya/l3_e74_console.log:1414`); the phase-2 collapse is confirmed by `runs/laya/phish_eval_sym.json`. Full detail: `research/verdict_audit/parts/laya_e67_e75.md` (E-74 sections).
Corrected verdict: keep `partial`, but the phase-1 clause is unsupported — no test-split typed artifact exists.

## Registry row `laya/E-075` (`research/experiments.yaml:92-116`) — RE-AUDITED
Recorded: status `planned`, verdict `none`; "Pre-registered 2026-09-23 as rung R3; not launched in this slice (started/closed UNKNOWN)".
Recheck: **CONFIRMED.** All cited paths exist (`research/EXPERIMENTS.md`, `configs/laya_l3_e70.yaml`, `runs/laya/l3_e70_pretrain`, `TASKS.md`, report `research/2026-09-23_laya_l3_frontier.md`). No result is claimed, so there is nothing to invert or overclaim. Caveat: the cited encoder directory survives only as `runs/laya/l3_e70_pretrain/pretrain_final/encoder/{config.json,model.safetensors,tokenizer.json,tokenizer_config.json}`; the E-75 pre-register's inherited baseline (held-out ppl < 8.12) traces to E-70's `pretrain_summary.json`, which is absent.
Corrected verdict: CONFIRMED as a pre-register; its inherited numeric baseline is prose-only.

---

## Audited verdict summary

| id | recorded verdict | metric + direction | cited artifact | recheck | corrected one-line verdict |
|---|---|---|---|---|---|
| E-62 (Laya L1) | PASS (G1+G2); G3 diagnostic FAIL | acc (higher), VRAM/wall ceiling, ECE (lower) | `runs/laya/l1/{train_summary,eval_report}.json` | NO_ARTIFACT | Cited metric files absent; numbers live only in the prose report — direction correct, evidence missing. |
| E-63 (Laya L2) | FAIL overall; G2 PASS, G1/G3 FAIL | typed acc / macro / agreement (higher), ECE (lower) | `runs/laya/l2/eval_report.json` + `train_summary.json` | CONFIRMED | All gate numbers and the post-B retention read match the artifact; direction correct. |
| E-64 (zero-shot eval) | no gates; no published win; L1 probe win | AUROC/acc (higher), probe failures (lower) | `runs/laya/phish_eval.json`, `public_choice_eval.json`, `probe_eval.json` | CONFIRMED | Comparison numbers all match; direction correct. |
| E-65 (Laya L3) | 1/4 PASS (G4 only) | AUROC/acc/macro/agreement (higher), VRAM/wall ceiling | `runs/laya/l3/{pretrain_summary,train_summary,eval_report}.json` + `build_stats.json` | CONFIRMED | All four gates, params, corpus and compute match; one comparator ("our L1 0.576") is MISCITED (that is L2 stage-A; L1 = 0.446). |
| `laya/E-074` | partial (registry) | typed / AUROC | `runs/laya/l3_e74_console.log` | MISCITED | Keep `partial`; phase-1 typed win rests on a train-split metric, not the unseen test. |
| `laya/E-075` | none (planned) | typed / AUROC / ppl | `configs/laya_l3_e70.yaml`, `runs/laya/l3_e70_pretrain` | CONFIRMED | Pre-register only; no result claimed. |

Recheck counts for this file (6 entries): CONFIRMED 4 (E-63, E-64, E-65, `laya/E-075`) · MISCITED 1 (`laya/E-074`) · NO_ARTIFACT 1 (E-62) · INVERTED 0. E-65 is CONFIRMED on its own gates but contains one MISCITED comparator inside it (counted under MISCITED, not as a second entry).

**FALSE-NEGATIVE check (recorded FAIL/WORSE that is really a win):** none. E-63's absolute-bar G1 FAIL
(0.6856 < 0.74) is correct as pre-registered even though retention did not decay (-0.0003), and the
verdict records both numbers. No metric in E-62…E-65 is compared in the wrong direction.

---

## Missing-evidence index (laya)

Every Laya experiment id whose cited evidence path does not exist on disk, with the path exactly as
recorded in the ledger/registry (brace groups expanded; `exists` from `Test-Path -LiteralPath`). Reports (`research/*.md`)
are narrative and are marked separately.

```csv
id, recorded_path, exists
E-62, runs/laya/l1/train_summary.json, no
E-62, runs/laya/l1/eval_report.json, no
E-62, runs/laya/l1/console.log, no
E-62, runs/laya/l1/logs, no
E-62, research/2026-09-22_laya_l1.md, yes (narrative report)
E-66, runs/laya/l3_e66/train_summary.json, no
E-66, runs/laya/l3_e66/eval_report.json, no
E-66, runs/laya/l3_e66/bench_log.jsonl, no
E-66, runs/laya/l3_e66_console.log, no
E-66, runs/laya/l3_e66/final/model.pt, no
E-67, runs/laya/l3_e66/eval_calibrated.json, no
E-67, runs/laya/l3_e66/bench_log.jsonl, no
E-67, runs/laya/l3_e66/e67_console.log, no
E-68, runs/laya/l3_e68/train_summary.json, no
E-68, runs/laya/l3_e68/eval_report.json, no
E-68, runs/laya/l3_e68/bench_log.jsonl, no
E-68, runs/laya/l3_e68/final/model.pt, no
E-69, runs/laya/l3_e69/train_summary.json, no
E-69, runs/laya/l3_e69/eval_report.json, no
E-69, runs/laya/l3_e69/bench_log.jsonl, no
E-70, runs/laya/l3_e70/train_summary.json, no
E-70, runs/laya/l3_e70/eval_report.json, no
E-70, runs/laya/l3_e70/bench_log.jsonl, no
E-70, runs/laya/l3_e70/final/model.pt, no
E-70, runs/laya/l3_e70_pretrain/bench_log.jsonl, no
E-70, runs/laya/l3_e70_pretrain/pretrain_summary.json, no
E-70, runs/laya/l3_e70_pretrain/pretrain_final/encoder, yes
E-71, runs/laya/l3_e71/train_summary.json, no
E-71, runs/laya/l3_e71/eval_report.json, no
E-71, runs/laya/l3_e71/bench_log.jsonl, no
E-71, runs/laya/l3_e71/final/model.pt, no
E-72, runs/laya/l3_e72/train_summary.json, no
E-72, runs/laya/l3_e72/eval_report.json, no
E-72, runs/laya/l3_e72/bench_log.jsonl, no
E-72, runs/laya/l3_e72/final/model.pt, no
E-73, runs/laya/l3_e73/train_summary.json, no
E-73, runs/laya/l3_e73/eval_report.json, no
E-73, runs/laya/l3_e73/bench_log.jsonl, no
E-73, runs/laya/l3_e73/final/model.pt, no
E-074, research/EXPERIMENTS.md, yes
E-074, runs/laya/l3_e74_console.log, yes
E-074, runs/laya/l3_e74/A_last.pt, yes
E-074, runs/laya/l3_e74/final/model.pt, yes
E-074, research/2026-09-23_laya_l3_frontier.md, yes (narrative report)
E-074, TASKS.md, yes
E-075, research/EXPERIMENTS.md, yes
E-075, configs/laya_l3_e70.yaml, yes
E-075, runs/laya/l3_e70_pretrain, yes
E-075, TASKS.md, yes
E-075, research/2026-09-23_laya_l3_frontier.md, yes (narrative report)
```

Ids whose cited evidence is fully present: E-62 partial (report only), E-63, E-64, E-65, E-074, E-075.
Ids with at least one missing cited metric artifact: **E-62, E-66, E-67, E-68, E-69, E-70, E-71, E-72, E-73**
(nine ids; all E-66..E-73 run directories and the E-62 L1 run directory are absent; the E-70 pretrain
directory is present only as final encoder files). Also expected but never recorded/created and therefore
absent: `runs/laya/l3_e74/eval_report.json` (no official gate report for E-74) and
`runs/laya/l3_e70_pretrain/pretrain_summary.json` (the E-70 perplexity baseline).

Why they are gone: run outputs under `runs/laya/` were never committed (weights are gitignored by
`.gitignore:162-163`: `runs/laya/**/*.pt`, `data/laya/`; `git log -- runs/laya/l3_e72` is empty), and the
staged recovery archive `checkpoint_backup/laya_l3_e74_resume_2026-09-23.zip` is itself absent from
`checkpoint_backup/`.

Audit limitation: only E-63, E-64 and E-65 have artifact-backed verdicts in this window; E-62's numbers are
report-only, and the E-66..E-73 NO_ARTIFACT findings are an evidence-integrity gap, not metric-direction
errors.
