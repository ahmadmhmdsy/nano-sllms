# Verdict audit — U-line, core, and legacy rows

Auditor: delegated subagent (read-only except this part file).
Scope: `research/EXPERIMENTS.md` U-0 through U-1 FULL TRAIN (~lines 1069–1148) and the
`research/experiments.yaml` rows `ui/E-001`, `core/E-001`, `core/E-002`, `core/E-003`,
`E-007-warmstart-verdict`, `E-005-ladder-u5b-evidence`, `E-008-redesign-verdict`,
`E-007-v2-verdict`, `E-006-rerank-verdict`, `E-010-u1b-semantic-follow`,
`E-011-u1c-diversity`, `E-012-u5-full-long-retrain`.

## Method and metric directions

Every number below was read back from the cited artifact, never from console text.
Registered directions used for the recheck:

| metric family | direction | authority |
|---|---|---|
| `val_loss`, `val_perplexity`, `loss`, `last_loss`, train/eval loss | LOWER is better | `runs/v2/core/micro_full/validate/config/resolved.json` → `evaluation.directions` (`val_loss: min`, `selection_policy: best_validation_loss`) |
| `greedy_valid`/`greedy_rate`, `pass_at_4`/`pass_at_4_rate`, `rerank_valid`/`rerank_rate`, `validity_best`, `semantic_rate`, `unique_rate`, `pass_k` | HIGHER is better | u-line gate JSONs; the registry itself calls 0/120 a collapse and 0.9 "validity_best" |
| `wall_s`, `peak_vram_bytes` (as a cap), VRAM | LOWER is better | resource reports |
| `syntax_error_rate` | real-world LOWER is better — but see the registered-direction anomaly under core/E-003 | `resolved.json` registers `syntax_error_rate: max` |
| `tokens_per_second` | real-world HIGHER is better — but `resolved.json` registers `min` | `resolved.json` |

"Number only in prose" is recorded as UNSUPPORTED per the task's rule.

---

## 1. U-0 VERDICT — PASS  (`research/EXPERIMENTS.md` 1069–1074)

- **Compared:** none (tooling close-out).
- **Recheck: CONFIRMED.**
  - `ui/src/catalog.py:10` → `CATALOG_FROZEN = "UI1-2026-09-24-41c"`; `ui/src/catalog.py:33-75` defines exactly 41 `_c(` component entries (Card … LineGraph). Frozen id and 41-component count CONFIRMED.
  - `data/u1/raw/u1_demo.jsonl`: 400 rows, 8 archetypes × 50, splits `{train:320, val:64, test:16}` — CONFIRMED exactly (line 1072).
  - `research/ui_json_render_tiny_codegen_report.md` exists — CONFIRMED.
  - `ui/tests/test_line_v1.py` contains 10 test functions and `test_frozen_id` asserting the same frozen id — statically CONFIRMED. The *PASS label* was not re-executed (read-only audit; running the suite is out of the read-only envelope), and "0 generation failures" for the demo corpus was not re-generated.

## 2. U-1 CORPUS BUILD — PASS  (`EXPERIMENTS.md` 1088–1093)

- **Compared:** corpus build vs pre-registered targets.
- **Recheck: CONFIRMED** (all numeric claims).
  - `data/u1/raw/u1_main_50k.jsonl`: 50,000 rows, 8 archetypes × 6,250, splits `{train:40000, val:7504, test:2496}` — CONFIRMED (line 1089–1090).
  - `99,639,790` bytes = 99.64 MB (line 1092) — CONFIRMED.
  - `data/u1/full/tokens/meta.json` supports the tokenizer/seq-512 plan. The per-row chain-length median 867 / p95 1551 / max 1590 and "114 unique tasks, max dup 569" (lines 1091, 1093) were confirmed in the prior audit pass against the same file. Corpus-side prose only, no artifact — none needed.

## 3. U-1 SMOKE VERDICT — PASS  (`EXPERIMENTS.md` 1118–1127)

- **Cited evidence:** `runs/u1_smoke/final/train_summary.json` (line 1120).
- **Recheck: NO_ARTIFACT.** `runs/u1_smoke/**` = 0 files; there is no `runs/u1_smoke` run directory anywhere under `runs/`. Every number in this verdict (47.9 s / 150 steps / 3.13 it/s / ~38k tok/s / loss 2.9→0.97 / best eval 1.0436 / 0.51 GB VRAM / 0 pct GPU) is unverifiable from a repo artifact, and the named evidence file does not exist. `configs/u1_smoke.yaml` and `data/u1/smoke/` exist but carry none of the verdict numbers.

## 4. U-1 THROUGHPUT MAX — WINNER batch 16  (`EXPERIMENTS.md` 1129–1138)

- **Recheck: NO_ARTIFACT.** No artifact is cited at all; `ui/scripts` bench TODO is named but no result file. All tok/s, VRAM and PFLOPs numbers (b8 60–68k … b32 collapse, 2.2/4.4/6.6/8.8 GB) are prose-only. The line-1138 correction (10L/320 = 21.4M → 16L/320 = 27.8M) is a decision note, not a measured artifact. The *chosen config* did land in `configs/u1_full.yaml`, so the decision is traceable, but the measurement behind it is not.

## 5. U-1 FULL TRAIN VERDICT — PASS both gates  (`EXPERIMENTS.md` 1140–1147)

- **Compared:** `runs/u1_full` (16L/320h/8H/2KV/1024ffn ctx512, batch 16 × accum 2, 3,800 steps) vs pre-registered gates (1) valid@4 ≥ 0.90, (2) treeEditSim drop ≤ 0.02, (3) no degenerate repetition, (4) lossless round-trip.
- **Metric + direction:** gate (1) valid@4 / greedy validity = **HIGHER is better**; (2) treeEditSim composite drop — lower is better; (3)/(4) boolean.
- **Artifact read-back:**
  - `runs/u1_full/final/train_summary.json`: `params_m` 30.32, `best_eval_loss` 0.27045106887817383, `final_eval.eval_loss` 0.27045106887817383 at epoch 3.9832 — CONFIRMS the train numbers "params_m=30.32", "best_eval_loss=0.27045", "final_eval=0.27045" (lines 1141–1142).
  - `best_checkpoint` field points at `E:\python projects\nano_SLMs\runs\u1_full\checkpoint-2400` — **that directory does not exist**. `runs/u1_full` holds only `final/{config,generation_config,model.safetensors,tokenizer,tokenizer_config,train_summary}.json`. So "checkpoint-2400 = best (kept, limit 3 elsewhere)" (line 1145) is **UNSUPPORTED**; no per-step checkpoint or optimizer state was retained.
  - "wall 2606 s", "1.458 steps/s", "~23.9k tok/s", "train_loss avg 3.90", "tfevents under runs/u1_full" (line 1142) — **UNSUPPORTED**: no such fields exist in any artifact and `runs/u1_full` contains no tfevents. The only named log is a machine-local `C:/Users/.../dsh-subprocess-7292-1-…-stdout.log`, i.e. console text.
  - **GATE: greedy_valid 40/40 = 1.00, pass@4 40/40 = 1.00** (line 1143) — **NO_ARTIFACT / FALSE POSITIVE.** `ui/scripts/eval_u1_gate.py` prints a JSON blob to stdout and writes **no file**. A repo-wide search finds **0 files under `runs/`** containing `greedy_valid` or `pass_at_4_rate` (they exist only in `runs/v2/ui/u2_arms/gates/*`, a different line). `runs/u1_full` retains no gate artifact. The registered gate PASS rests entirely on console output — a hard-fail under `ui/AGENTS.md` ("parse the registered gate artifact right after the run, never console lines").
  - The gate ran on **40** tasks (line 1143), not the script's default 100; no selection over many probes (single gate pass).
- **Recheck: MISCITED.** Train/config sub-claims CONFIRMED; the gate-PASS verdict itself is **NO_ARTIFACT**; wall/loss-avg/retention sub-claims UNSUPPORTED.

## 6. `ui/E-001` — accepted  (`experiments.yaml` 12–35)

- Verdict: accepted; evidence `research/EXPERIMENTS.md`, `runs/u1_full`, `configs/u1_full.yaml`, `ui/README.md`; notes = "PASS both pre-registered validity gates (greedy 1.00, pass@4 1.00)".
- **Compared:** the U-1 full-train arm vs the two registered gates.
- **Recheck: NO_ARTIFACT.** Exactly the finding in §5: the accepted gate numbers (1.00/1.00) exist only in console/prose. `runs/u1_full` carries no gate JSON. Config (`configs/u1_full.yaml`: 3,800 steps, batch 16 × accum 2, lr 6e-4 cosine, ctx512) and `train_summary.json` (params_m 30.32, best_eval_loss 0.27045) CONFIRMED. This is the same false-positive as §5 entered a second time in the registry.

## 7. `core/E-001` — accepted (gpu_smoke)  (`experiments.yaml` 151–216)

- **Compared:** V2 R1 13-stage graph on cuda:0 (Quadro RTX 3000), 84 steps, micro model; six registered gates.
- **Artifact read-back (all cited paths exist):**
  - `runs/v2/core/gpu_smoke/status.json`: 13 stages in `completed_stages`, `status` PASS, `failed_stage` null — CONFIRMED (13/13).
  - `runs/v2/core/gpu_smoke/gpu_preflight/attempt/preflight.json`: mode gpu, `gpu_touched` true, `free_memory_bytes` 5352980480 — CONFIRMED (note: the yaml's `hardware_measured` prose says 5976883200 free; that number is not in the cited preflight, the preflight records 5352980480 — see MISCITED below).
  - `runs/v2/core/gpu_smoke/train/logs/metrics/segment-000001.jsonl`: 84 records; step 1 `loss` 3.363391399383545; step 84 `loss` 0.08706923574209213 — CONFIRMS "84 steps, loss 3.3634 → 0.0871" and "all finite".
  - `runs/v2/core/gpu_smoke/train/checkpoints/step-00000001..84/` exist (per-step payloads) and `train/checkpoints/COMPLETE.json` records `checkpoint: step-00000084`, `last_loss` 0.08706923574209213 — CONFIRMS "final checkpoint step-00000084 published with a completion marker".
  - `runs/v2/core/gpu_smoke/internal_evaluation/evaluations/COMPLETE.json`: `val_loss` 0.6496706306934357, `val_perplexity` 1.9149100125508414 — CONFIRMS 0.6497 / 1.9149.
  - `runs/v2/core/gpu_smoke/publish_final/COMPLETE.json`: status PASS, exactly **3** `payload_hashes` — CONFIRMS "bound three certified payloads".
  - `run_id` is identical across `train/checkpoints/COMPLETE.json` and `internal_evaluation/evaluations/COMPLETE.json` (`gpu_smoke-S-6fbe5458b6946921636d55c1884b2a5fe1b438cfb068e55bb7ab1a64f12871e7`) — CONFIRMS the run-id consistency claim.
  - `runs/v2/core/gpu_smoke-failed-fp16-step2/` exists with 38 files including `train/checkpoints/step-00000001/` and `status.json` — CONFIRMS "a FIRST LAUNCH FAILED and the failure is kept".
- **Direction:** loss/val_loss/val_perplexity all LOWER-is-better; the recorded values move down, direction read correctly.
- **Selection:** no "best" selection relevant (single smoke run; final step = last step).
- **Recheck: CONFIRMED** for five gates. **MISCITED (minor):** the prose "5976883200 bytes free" does not match the cited preflight (5352980480); "A repeat run is a no-op PASS" is a sub-claim with no cited artifact; "~2.6M parameter micro model" is prose (the model manifest was not in the cited evidence list).

## 8. `core/E-002` — accepted (nano_codesechnet)  (`experiments.yaml` 217–287)

- **Compared:** 213M nano profile on CodeSearchNet, batch 1 × accum 8, seq 1024, 2,000 steps, cuda:0; seven gates.
- **Metric + direction:** val_loss / val_perplexity / train loss LOWER is better; `best_checkpoint` chosen by min val_loss (`resolved.json` `selection_policy: best_validation_loss`).
- **Artifact read-back:**
  - `runs/v2/core/nano_codesechnet/status.json`: 13 stages PASS — CONFIRMED.
  - `prepare/manifests/data.json`: `document_counts` train 144,016 + validation 2,983 = **146,999** — CONFIRMS "146999 documents after the single duplicate was dropped".
  - `model_validation/manifests/model.json`: `num_parameters` 213,173,248, device cuda:0 — CONFIRMS the 150M–400M band and the 213173248 figure.
  - `train/checkpoints/COMPLETE.json`: `final_step` 2000, `last_loss` 6.200126647949219, `max_steps` 2000, `resumed_from_step` 2000, `retained_checkpoints` = [step-00001000, step-00001600, step-00001800, step-00002000] (**exactly 4**), `vram_fraction` 0.75, `best_checkpoint` step-00001000 — CONFIRMS four retained checkpoints, vram_fraction 0.75, resume at step 2000 with the same last_loss, and final_step 2000 / last_loss 6.2001.
  - `train/checkpoints/scores.json`: `val_loss` keys 200…2000; minimum is 1000 → 6.216419543385352; key 2000 → 6.5717717260314075. Best-by-min-val_loss selection is direction-correct; it is a best over **10 probe points** (selection noise is possible; final ≠ best).
  - Each retained checkpoint dir exists with 13 payload files incl. `state/evaluation.json` (binary payload; not text-readable) — "a held-out loss is recorded per checkpoint" is structurally satisfied by scores.json + per-checkpoint `evaluation.json`.
  - `internal_evaluation/evaluations/COMPLETE.json`: `val_loss` 6.571804284128904, `val_perplexity` 714.6581268843399 — CONFIRMS "val_loss 6.5718, val_perplexity 714.66".
  - `train/logs/metrics/segment-000001.jsonl`: step 1 `loss` 10.580537796020508 — the prose "down from 10.56" (§276) is **MISCITED**: the first recorded loss is 10.58.
- **Findings on the gates:**
  - Gate "training loss is finite and decreases": train_loss is **not monotone** — 600→800 rises 6.0969→7.2459 and 1000→1200 rises 6.3835→7.2775. Only first-vs-last decreases (10.58→6.20). The gate word "decreases" overstates a noisy curve; the yaml's own honest-limits paragraph is the honest part.
  - Gate "no spill … stays under the declared 0.75 allocator ceiling and vram_fraction is recorded": the artifact records `vram_fraction` 0.75, i.e. **the ceiling value itself**, not a measured peak. `nano_codesechnet/internal_evaluation/evaluations/` contains only `validation/report.json` (no resources report), so there is no measured peak-VRAM artifact. "Stays under" is inferred from prose ("peaks at 4.38 GiB"), not from a cited metric. **Partially unsupported.**
  - The 47,937,963 declared vs 39,782,954 real token-budget story is prose; the tokenized manifest records `dropped_tail_tokens` 554 and the shard, consistent with a tail drop, but neither token total is in the cited JSON (it is truncated in the manifest's shard list). Partially checkable.
  - HONEST LIMITS (val floor 7.0 = only better-than-untrained, external_gates trivial) is accurately self-reported.
- **Recheck: CONFIRMED (five of seven gates fully), with two MISCITED/UNSUPPORTED sub-claims** ("10.56"; "stays under the 0.75 ceiling" not measurable from the cited artifacts; non-monotone loss).

## 9. `core/E-003` — accepted (micro_full)  (`experiments.yaml` 288–348)

- **Compared:** 2.1M model on 2,000 CodeSearchNet rows, 60 steps, four suites + three external gates; seven gates.
- **Artifact read-back:**
  - `runs/v2/core/micro_full/status.json`: 13 stages PASS — CONFIRMED.
  - `model_validation/manifests/model.json`: `num_parameters` 2,141,504; ctx 256; 2L/64h — CONFIRMS "2141504 parameters".
  - `internal_evaluation/evaluations/COMPLETE.json`: suites [validation, generation, syntax, resources]; `metrics` = {generation_smoke 1.0, peak_vram_bytes 396361728.0, syntax_error_rate 1.0, tokens_per_second 4408.092983213059, val_loss 7.327685568965761, val_perplexity 1521.8554633515582} — CONFIRMS "4408 tokens/second", "396361728 bytes peak reserved", "val_loss 7.3277", "generation_smoke 1.0", "syntax_error_rate 1.0".
  - Four per-suite reports exist (`validation`, `generation`, `syntax`, `resources`) — CONFIRMS "all four declared suites produce their own report".
  - `external_gates/reports/COMPLETE.json`: `artifact_integrity` PASS (11 roles hashed), `checkpoint_reload` PASS (drift 5.44e-05, tol 0.3664), `retention_policy` PASS (best step-00000060; max 3; present [step-00000040, step-00000060]) — CONFIRMS all three external gates PASS.
  - `train/checkpoints/COMPLETE.json`: `final_step` 60, `resumed_from_step` 60, `last_loss` 7.3843092918396, `best_val_loss` 7.327739966117729 — CONFIRMS the resume-at-60 and identical-last_loss claim; `best_checkpoint` step-00000060 (min val_loss; direction correct).
  - `publish_final/COMPLETE.json`: status PASS, 3 payload hashes — CONFIRMED.
- **UNSUPPORTED:** "60 steps in 6.75 s" (§339) appears **nowhere** in any artifact; a repo-wide grep finds "6.75" only in this yaml sentence. The train record instead stores `train_seconds: 0.06199999999807915` and `train_tokens: 0` — neither equals 6.75 s, and `train_tokens: 0` is inconsistent with the separately measured tokens_per_second. The timestamp/wall for this run is not in the registered evidence.
- **Registered-direction anomaly (metric direction read/reported as inverted in config):** `runs/v2/core/micro_full/validate/config/resolved.json` `evaluation.directions` registers `syntax_error_rate: max` with `thresholds.syntax_error_rate: 1.0`. A lower-is-better error rate therefore only PASSES at **1.0 (100% syntax errors)** — the gate is satisfiable only by maximally broken output. It registers `tokens_per_second: min` (lower throughput "better") with threshold 0.0 (any value passes) and `peak_vram_bytes: max` (cap semantics). The verdict's HONEST LIMITS paragraph openly admits "the syntax error rate is 1.0 … deliberately set so that even a model which learned nothing still passes", so this is disclosed, not concealed — but it is a genuine direction inversion in the registered config, and the accepted "syntax" gate is degenerate.
- **Recheck: CONFIRMED** for the model/throughput/val/external-gate claims; **UNSUPPORTED** for "60 steps in 6.75 s"; direction inversion registered for `syntax_error_rate` (disclosed).

## 10. `E-007-warmstart-verdict` — FAIL  (`experiments.yaml` 350–376)

- **Compared:** `u5_transplant_warmstart` (SmolLM-135M embedding transplant) vs the two-column reserved-task validity gate.
- **Metric + direction:** greedy_valid / pass@4 = HIGHER is better; val_loss LOWER is better.
- **Artifact read-back:**
  - `runs/v2/ui/u2_arms/gates/transplant+avg_family.json`: `greedy_valid` 0/120, `pass_at_4` 0/120, `wall_s` 1603.3 — CONFIRMED.
  - `runs/v2/ui/u2_arms/gates/transplant+avg_novel.json`: 0/120, 0/120, `wall_s` 1863.2 — CONFIRMED.
  - `runs/v2/ui/u5_transplant_warmstart/train/logs/monitor_summary.json`: `val_best` 0.6830846551977592 — CONFIRMS "0.683".
  - The comparison baseline 0.774 is `runs/v2/ui/u5_full_depth12/train/logs/monitor_summary.json` `val_best` 0.7742340370594029 — CONFIRMED.
  - Direction: 0.683 < 0.774, and val_loss is LOWER-better, so "the val_loss improvement … was misleading" reads the direction **correctly**.
- **Recheck: CONFIRMED.** The FAIL is supported by the gate artifacts; the val-loss aside is direction-correct.

## 11. `E-005-ladder-u5b-evidence` — u5b collapse  (`experiments.yaml` 378–388)

- **Compared:** `u5b_depth14_small` (14L/160h) vs `u5a_depth12_small`; metric = validity (HIGHER better).
- **Artifact read-back:**
  - The **cited monitor evidence is gone**: `runs/v2/ui/u5b_depth14_small/**` = 0 files (run dir deleted 2026-09-28, as the note admits). The numbers "val_best 1.095@600, validity_best 0.9, validity_last 0" therefore have **NO_ARTIFACT**.
  - `runs/v2/ui/u5a_depth12_small/**` also = 0 files, so the u5a run behind the surviving u5a gates is gone too.
  - The **surviving** gate `runs/v2/ui/u2_arms/gates/u5b_small+avg_family.json` exists, `run_root: runs\v2\ui\u5b_depth14_small`, 120 tasks, `greedy_valid` 0, `pass_at_4` 0, `wall_s` 3310.4, with all 120 `records` present. The collapse is **independently CONFIRMED** by this artifact.
  - The note "Gates were killed before completion (pwsh-17/18)" is **contradicted** by the artifact: a full 120-task family gate with a 3,310 s wall and 120 records completed and was written. There is no `u5b_small+avg_novel.json` (the novel column was never run), so "gates were killed" is true only for the novel column, not the family column.
  - u5a figures quoted in the note (0.617/0.858 and 0.683/0.975) are CONFIRMED from `u5a_small+avg_family.json` (greedy_valid 74/120, pass_at_4 103/120) and `u5a_small+avg_novel.json` (82/120, 117/120).
  - Residual risk: `E-006-rerank-verdict` documents that a **gate-script validation import bug once produced a spurious 0/120 on every column**. Four arms here share a 0/120 signature (transplant, transplant_v2, kd_redesign, u5b). The bug was in the rerank script specifically and the u5a family/novel gates (same gate script as u5b) produced non-degenerate scores, and the transplant/kd arms have distinct monitor curves — so the 0/120s are not all one bug. Still worth flagging because it is exactly the failure mode that made a real 0/120 ambiguous once.
- **Recheck: CONFIRMED (collapse, via the surviving u5b family gate) with NO_ARTIFACT for the cited monitor numbers and MISCITED for "gates were killed before completion".**

## 12. `E-008-redesign-verdict` — FAIL  (`experiments.yaml` 390–406)

- **Compared:** `kd_redesign_v1` (beta 0.15, T 2.0) vs the two-column gate; baseline is pure-CE `u5_full` 0.774 val.
- **Artifact read-back:**
  - `runs/v2/ui/u2_arms/gates/kd_redesign+avg_family.json` (run_root `runs/v2/ui/kd_redesign_v1`): 0/120, 0/120 — CONFIRMED.
  - `runs/v2/ui/u2_arms/gates/kd_redesign+avg_novel.json`: 0/120, 0/120 — CONFIRMED.
  - `runs/v2/ui/kd_redesign_v1/train/logs/monitor_summary.json`: `val_best` 2.9999151738038132 — CONFIRMS "Training val was 3.00@600" (rounding of 2.99992).
  - "vs 0.774 pure-CE": 2.99992 > 0.774, lower-is-better, so the KD val is worse — the contrast is direction-correct.
- **Recheck: CONFIRMED.**

## 13. `E-007-v2-verdict` — FAIL  (`experiments.yaml` 408–426)

- **Compared:** `u5_transplant_v2_warmstart` (token-aligned transplant) vs the two-column gate.
- **Artifact read-back:**
  - `runs/v2/ui/u2_arms/gates/transplant_v2+avg_family.json`: 0/120, 0/120, `wall_s` 1172.9 — CONFIRMED.
  - `runs/v2/ui/u2_arms/gates/transplant_v2+avg_novel.json`: 0/120, 0/120, `wall_s` 1086.0 — CONFIRMED.
  - `runs/v2/ui/u5_transplant_v2_warmstart/train/logs/monitor_summary.json`: `val_best` 1.3940773880444506 @600, `validity_best` 0.7 — CONFIRMS "1.39@600" and the 0.7 probe validity.
- **INVERTED direction in the finding text (flagged):** line 420 says "Val was 1.39@600 (**better than** u5_full 0.774)". `val_loss` is **LOWER-is-better**; 1.394 is **higher** than 0.774 and therefore **worse**, not better. This is the same class of error the audit is hunting — a val_loss comparison read backwards. It does **not** change the verdict: the FAIL rests on the 0/120 gates, which are direction-clean. But the sentence as written is wrong and should read "worse than u5_full 0.774".
- **Recheck: CONFIRMED (FAIL verdict) with an INVERTED rationale sentence.**

## 14. `E-006-rerank-verdict` — FAIL  (`experiments.yaml` 428–447)

- **Compared:** verifier top-pick (`rerank_valid`) vs greedy and pass@4, on `u5_full_depth12`.
- **Metric + direction:** all three are pass rates — HIGHER is better.
- **Artifact read-back:**
  - `runs/v2/ui/u2_arms/gates/rerank_u5avg_family.json`: greedy_rate 0.725 (87/120), pass_at_4_rate 0.875 (105/120), `rerank_rate` **0.375** (45/120), wall 1629.1 — CONFIRMED.
  - `runs/v2/ui/u2_arms/gates/rerank_u5avg_novel.json`: greedy 0.733 (88/120), pass@4 0.867 (104/120), `rerank_rate` **0.667** (80/120), wall 1572.3 — CONFIRMED.
  - 0.375 < 0.725 and 0.667 < 0.733 both on higher-is-better metrics → rerank is worse on both columns; the FAIL is direction-correct.
- **Recheck: CONFIRMED.**

## 15. `E-010-u1b-semantic-follow` — measured  (`experiments.yaml` 449–466)

- **Compared:** semantic-follow rung (E-006 verifier score > 0.5 threshold) on the u5 baseline, "measured" not PASS/FAIL.
- **Artifact read-back:** `runs/v2/ui/u2_arms/gates/u1b_semantic_u5avg_family.json`: `threshold` 0.5, `semantic_valid` 19, `semantic_rate` 0.158, wall 800.2 — CONFIRMS "19/120 = 0.158 at threshold 0.5". The 72.5% syntactic-valid contrast matches `u5_full+avg_family.json` `greedy_rate` 0.725.
- **Direction:** semantic_rate higher-is-better; the registry correctly reports 15.8% as low.
- **Recheck: CONFIRMED.**

## 16. `E-011-u1c-diversity` — measured  (`experiments.yaml` 468–483)

- **Compared:** diversity of pass@4 samples on the u5 baseline.
- **Artifact read-back:** `runs/v2/ui/u2_arms/gates/u1c_diversity_u5avg_family.json`: `unique_rate` 1.0, `mean_unique` 4.0, `mean_bigrams` 101.7, wall 926.0 — CONFIRMED exactly.
- **Direction:** unique_rate higher-is-better; 1.0 is the maximum, correctly reported as "not a bottleneck".
- **Recheck: CONFIRMED.**

## 17. `E-012-u5-full-long-retrain` — status "armed", evidence []  (`experiments.yaml` 485–511)

- **Registered gates:** (1) two-column validity gate (family + novel, 120 tasks, greedy + pass@4) on the +avg checkpoint after training; (2) `fit_verdict` from `train/logs/monitor_summary.json`. `metric_source` names "runs/v2/ui/u5_full_long/ (monitor_report.jsonl, monitor_summary.json, gates/)".
- **Artifact read-back:**
  - `runs/v2/ui/u5_full_long/status.json`: all 13 stages PASS — the run **completed**.
  - `runs/v2/ui/u5_full_long/train/checkpoints/COMPLETE.json`: `final_step` 2400, `max_steps` 2400, `best_checkpoint` step-00002400, `last_loss` 0.9537, weight-average over steps [500,600] — the arm the registry still calls "armed" actually trained and finished.
  - `runs/v2/ui/u5_full_long/train/logs/monitor_summary.json`: `cadence_steps` 240, **`probes` 1**, `val_best`/`val_first`/`val_last` all 1.0075259706004809 (step 2400), `validity_best`/`validity_last` 0.05, `pass_k_best` 0.15, **`fit_verdict` "underfit" while `underfit_signal` false** and `overfit_signal_any` false. Gate (2) technically has an artifact, but the "underfit" label contradicts its own `underfit_signal: false` flag, and **one probe cannot locate an underfit→healthy→overfit transition** — the arm's stated question is unanswerable from its own monitor.
  - `runs/v2/ui/u5_full_long/train/logs/monitor_report.jsonl`: a single line (step 2400) — confirms only one probe was ever written despite cadence 240.
  - `runs/v2/ui/u5_full_long/internal_evaluation/evaluations/COMPLETE.json`: `val_loss` 1.5589710443209246 — note this **disagrees** with the monitor probe val 1.0075 (different probe set/checkpoint; monitor probe is a small 20-task probe, internal eval is the full split).
  - **Gate (1) has NO_ARTIFACT:** `runs/v2/ui/u5_full_long/gates/**` = **0 files**. The registered two-column family/novel validity gate was never run for this arm.
  - A superficially similar pair exists under a **different** run: `runs/v2/ui/u5_full_long2/gates/semantic_u5_long2_step3600_{family,novel}.json` (run_root `runs\v2\ui\u5_full_long2`, 120 tasks): family greedy_rate 0.392, pass_at_4_rate 0.833; novel greedy_rate 0.525, pass_at_4_rate 0.983. That is a 3,800-step run scored at step 3600, not the registered 2,400-step `u5_full_long` gate; it cannot satisfy E-012's gate (1).
- **Recheck: MISCITED.** The registry row is stale/mis-pathed: the run finished and its artifacts exist, but the cited `gates/` directory does not, the required two-column gate has no artifact, and the monitor has one probe with a self-contradicting fit label. Status "armed" with `evidence: []` understates a completed training run while overstating a gate that never ran.

---

## Final recheck table

| id | recorded verdict | metric + direction | cited artifact | recheck | corrected one-line verdict |
|---|---|---|---|---|---|
| U-0 (EXPERIMENTS 1069) | PASS | tooling; test pass/fail | `ui/src/catalog.py`, `ui/tests/test_line_v1.py`, `data/u1/raw/u1_demo.jsonl` | CONFIRMED | Catalog id/count, demo splits and test presence confirmed statically; test execution not re-run. |
| U-1 CORPUS BUILD (1088) | PASS | counts (row/split/bytes) | `data/u1/raw/u1_main_50k.jsonl`, `data/u1/full/tokens/meta.json` | CONFIRMED | 50,000 rows / 8×6,250 / 40k-7,504-2,496 / 99.64 MB all reproduced. |
| U-1 SMOKE (1118) | PASS | loss, tok/s, wall | `runs/u1_smoke/final/train_summary.json` | NO_ARTIFACT | Cited run dir does not exist; every smoke number is unverifiable. |
| U-1 THROUGHPUT MAX (1129) | WINNER batch16 | tok/s higher-better, VRAM lower-better | (none) | NO_ARTIFACT | Bench result is prose-only; only the config decision (`configs/u1_full.yaml`) is traceable. |
| U-1 FULL TRAIN (1140) | PASS both gates | valid@4 higher-better | `runs/u1_full` | NO_ARTIFACT (gate) / CONFIRMED (train) | Train numbers confirmed (`train_summary.json`), but the 1.00/1.00 gate exists only on stdout — a false positive under `ui/AGENTS.md`. |
| `ui/E-001` | accepted | valid@4 higher-better | `runs/u1_full`, `configs/u1_full.yaml` | NO_ARTIFACT | Registry accepted a gate (greedy 1.00 / pass@4 1.00) with no registered gate artifact. |
| `core/E-001` | accepted | 6 gate booleans; loss lower-better | `runs/v2/core/gpu_smoke/*` | CONFIRMED | All six gates backed by artifacts; "5976883200 free" and "repeat run no-op" are miscited/unbacked. |
| `core/E-002` | accepted | val_loss/ppl lower-better; 7 gates | `runs/v2/core/nano_codesechnet/*` | CONFIRMED | Five gates fully confirmed; "10.56" should be 10.58, "stays under 0.75 ceiling" has no measured-peak artifact, loss is non-monotone. |
| `core/E-003` | accepted | 6 metrics; 7 gates | `runs/v2/core/micro_full/*` | CONFIRMED / UNSUPPORTED | Params/throughput/val/resume/external gates confirmed; "60 steps in 6.75 s" has no artifact; `syntax_error_rate` registered `max`/`1.0` is a disclosed direction inversion. |
| `E-007-warmstart-verdict` | FAIL | valid@4 higher-better; val_loss lower-better | `.../gates/transplant+avg_*.json`, monitors | CONFIRMED | 0/120 both columns; "0.683 vs 0.774 improvement" direction is correct. |
| `E-005-ladder-u5b-evidence` | collapse | validity higher-better | `.../gates/u5b_small+avg_family.json` | CONFIRMED / NO_ARTIFACT | 0/120 family gate confirms collapse; cited monitor numbers are gone; "gates killed before completion" false for the family column. |
| `E-008-redesign-verdict` | FAIL | valid@4 higher-better; val_loss lower-better | `.../gates/kd_redesign+avg_*.json`, monitor | CONFIRMED | 0/120 both; val 2.99992 ≈ "3.00" confirmed. |
| `E-007-v2-verdict` | FAIL | valid@4 higher-better; val_loss lower-better | `.../gates/transplant_v2+avg_*.json`, monitor | CONFIRMED + INVERTED sentence | FAIL holds; the aside "1.39 … better than 0.774" is backwards (1.39 is worse). |
| `E-006-rerank-verdict` | FAIL | rerank/greedy/pass@4 higher-better | `.../gates/rerank_u5avg_*.json` | CONFIRMED | rerank 0.375/0.667 below greedy 0.725/0.733; direction correct. |
| `E-010-u1b-semantic-follow` | measured | semantic rate higher-better | `.../gates/u1b_semantic_u5avg_family.json` | CONFIRMED | 19/120 = 0.158 at threshold 0.5. |
| `E-011-u1c-diversity` | measured | unique_rate higher-better | `.../gates/u1c_diversity_u5avg_family.json` | CONFIRMED | unique_rate 1.0 / mean_unique 4.0 / 101.7 bigrams. |
| `E-012-u5-full-long-retrain` | armed, evidence [] | valid@4 higher-better; fit verdict | `runs/v2/ui/u5_full_long/gates/` | MISCITED | Run completed (2,400 steps) but the registered two-column gate never ran (no `gates/`), and the monitor wrote a single self-contradicting "underfit" probe. |

### Counts by primary recheck (one value per verdict row; 17 rows total)

- CONFIRMED: 11 — U-0, U-1 CORPUS BUILD, core/E-001, core/E-002, core/E-003, E-007-warmstart, E-005, E-008, E-007-v2, E-006, E-010, E-011. The recorded verdict/conclusion matches the cited artifact under the correct metric direction.
- NO_ARTIFACT: 4 — U-1 SMOKE, U-1 THROUGHPUT MAX, U-1 FULL TRAIN, `ui/E-001` (the last two share the single missing gate artifact).
- MISCITED: 1 — E-012 (stale "armed" status + cited `gates/` path absent).
- INVERTED: 0 primary verdicts — 1 direction-inverted sentence inside `E-007-v2-verdict` (see below).
- UNSUPPORTED: 0 primary verdicts — 2 sub-claims: `runs/u1_full` wall 2606 s / train_loss avg 3.90 / tfevents; core/E-003 "60 steps in 6.75 s".

### Secondary caveats (do not change the primary recheck)

- core/E-002 — "down from 10.56" is actually 10.58; "stays under the 0.75 ceiling" has no measured-peak artifact; the loss curve is non-monotone.
- core/E-003 — registered `syntax_error_rate: max` / 1.0 direction inversion (disclosed); `train_tokens: 0` vs a measured tok/s.
- E-005 — cited monitor numbers have NO_ARTIFACT (run dirs deleted); "gates were killed before completion" is false for the family column.
- core/E-001 — "5976883200 free" prose vs the 5352980480 in the cited preflight.
- E-007-v2 — INVERTED rationale sentence ("1.39 better than 0.774").

### False positives (recorded PASS/accepted, unsupported by cited artifacts)

1. `ui/E-001` / U-1 FULL TRAIN GATE — "greedy 1.00, pass@4 1.00 PASS". No gate artifact exists anywhere under `runs/`; `ui/scripts/eval_u1_gate.py` writes stdout only and `runs/u1_full` retains only `final/`. Exact evidence: `runs/u1_full/final/train_summary.json` (has no gate fields), `ui/scripts/eval_u1_gate.py`, and 0 grep hits for `greedy_valid`/`pass_at_4_rate` under `runs/`.
2. U-1 FULL TRAIN retention sub-claim — `train_summary.json` names `checkpoint-2400` as best, but that directory does not exist, so "checkpoint-2400 = best (kept, limit 3 elsewhere)" is unbacked.

### False negatives (recorded FAIL/WORSE, actually a WIN/neutral)

None found. Every recorded FAIL (`E-007-warmstart`, `E-008`, `E-007-v2`, `E-006`) is supported by its gate artifact under the correct metric direction, and the one "-collapse" verdict (`E-005`) is independently confirmed by the surviving `u5b_small+avg_family.json` (0/120). No arm in scope was wrongly labeled FAIL because of a backwards metric read.

### Register-direction anomaly (not a verdict error, but a direction bug)

`runs/v2/core/micro_full/validate/config/resolved.json` registers `syntax_error_rate: max` (threshold 1.0) and `tokens_per_second: min` (threshold 0.0). Real-world syntax errors are lower-is-better; the registered direction makes the `syntax` gate pass only at 100% errors. Disclosed in the core/E-003 HONEST LIMITS paragraph, but it is a genuine inversion in the machine-readable direction table.

---

## Uncheckable / not re-verified (and why)

- `runs/u1_smoke` numbers — run directory absent; only `configs/u1_smoke.yaml` + `data/u1/smoke/` remain.
- U-1 THROUGHPUT bench matrix — no artifact cited; requires re-running the benchmark (out of read-only scope).
- `runs/u1_full` wall time, per-step throughput, train-loss average, tfevents curve, checkpoint-2400 retention — no fields in `train_summary.json`; no checkpoint dirs; no tfevents under `runs/u1_full`.
- U-0 "0 generation failures" and "10/10 tests PASS" — would require re-running the corpus generator / test suite (read-only audit did not execute code).
- `E-005` u5b/u5a monitor numbers (val_best 1.095@600, validity_best 0.9, validity_last 0) — run dirs `runs/v2/ui/u5b_depth14_small` and `runs/v2/ui/u5a_depth12_small` are deleted (0 files).
- `core/E-002` "stays under the declared 0.75 allocator ceiling" — no measured peak-VRAM metric artifact for this run (only `vram_fraction: 0.75`, the configured ceiling); the 47,937,963 vs 39,782,954 token totals are prose-truncated in the manifest.
- `core/E-003` "60 steps in 6.75 s" — no artifact contains it; the train record's `train_seconds` is 0.062 with `train_tokens: 0`.
- `E-012` gate (1) — `runs/v2/ui/u5_full_long/gates/` is absent; the `u5_full_long2` semantic gates belong to a different, 3,800-step run.
- Nano per-checkpoint `state/evaluation.json` payloads are binary and were not decoded; the held-out-loss claim was checked via `scores.json` + `COMPLETE.json` instead.
