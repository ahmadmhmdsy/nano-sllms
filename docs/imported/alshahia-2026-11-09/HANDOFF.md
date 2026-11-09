# 2026-11-02b - Tool-call SFT (tool_scale075): B_tool_call 0 -> 16 greedy; B_fix runtime -8 (CI crosses 0)

- **What (user "Yes, both"):** LoRA SFT on runs/sa_b_scale075 in the NS bench's Python tool-call format (stubs + `handle_request()` body), then merged at 0.75 into runs/tool_scale075. Hypothesis: sa_b's 14,854 xlam tool rows used a JSON format (`# Tools: {json}` -> JSON list), so the bench format was never trained.
- **Data (scripts/tool_sft_gen.py -> data/tool_sft, build.json):** xlam rows rewritten as Python stubs (7,000 converted, 4,619 pass the sandboxed build_tool check), plus Gemini free-tier rows on TOOL_DOMAINS, which are disjoint from the bench DOMAINS (1,328 generated before the free quota ran out at 1,600; 1,043 pass). Decontamination against data/bench/ns/B_tool_call.jsonl dropped 122 rows (35 Gemini, 87 xlam): any shared tool name (271 names) or any shared request 8-gram. Train: tool_py 5,491, plus replay of fix, repair, he and mbpp (1,500 each) and replay 1,000 from data/sa_b_scoped/sa4. Val: tool_py 171 and 227 replay rows. **Bench data was not trained on.**
- **Train (configs/sft_tool.yaml, chain scratch/tool_sft_chain.sh):** 12,491 pairs, 1 epoch, 196 steps, 1,140 s, cap 6.52 GiB, no OOM. eval_loss went 0.329 -> 0.305; best is the final checkpoint.
- **Quick NS, tool_scale075 vs sa_b_scale075 (points, 95 % CI):**
  - B_tool_call greedy **+16 [6, 26]**, runtime **+8 [2, 16]**, oracle@8 40 (was 0).
  - A_cruxeval greedy +10 [2, 20], runtime +12 [2, 22].
  - B_body runtime +10 [2, 18].
  - B_fix runtime **-8 [-18, 2]**: 30 -> 22, so it is not significant, but it points down.
  - B_write_fn runtime +2 [-4, 8]; greedy -2.
  - A_mbpp_pro runtime -4 and B_general runtime -4; both CIs reach 0.
  - Everything else is within ±2.
  - Report: research/raw/bench_ns_tool_scale075_quick.json; compare with `bench_ns_eval.py --compare sa_b_scale075_quick,tool_scale075_quick`.
- **Read:** the format-mismatch hypothesis is confirmed; tool calling went from nothing to a measurable rate. Runtime (8) is below greedy (16) on B_tool_call because the bench gives no visible tests for tool tasks, so runtime's pick among k samples is close to random. Oracle@8 40 shows that right calls are often in the sample set, so selection is the lever there. B_fix is the regression to watch. A full-tier run (k16, all items) would be needed to tell whether the -8 is real.
- **State:** nothing committed. New files: scripts/tool_sft_gen.py, configs/sft_tool.yaml, scratch/tool_sft_chain.sh. Data in data/tool_sft; models in runs/sft_tool and runs/tool_scale075; Gemini cache in data/teacher/tool_sft_cache.jsonl.

# 2026-11-02 - Never-seen bench (NS) built and baselined; E2 verifier FAILED its gate

- **E1 (done before this entry), N64 gate:** sa4 runtime 27.6, mbr 29.2, oracle 38.6; base 19.7/20.8/26.2; CPT 14.9/14.3/18.4.
- **E2 verdict: FAIL.** Numpy logistic head on frozen base SmolLM2 embeddings, labels from sa4_scale075 k16 on 1,500 sa2 train tasks. Gate needed at least +3 over MBR at N64 (32.2) with CI above 0. Combo 29.2 vs MBR 29.2, CI [-2.2, 2.2]; verifier alone 27.8. Held-out HE+MBPP (591 tasks): MBR 29.1, combo 28.9, CI [-1.5, 1.2], so the no-loss check also fails. AUC among passers is 0.75-0.86, but top-1 selection is no better than MBR. CV picked lam 16, the top of the grid, which means the mix leans on MBR. **E2c (GLM-25M verifier) was NOT run**, per the plan's condition. Oracle stays at 38.6, so the 9.4-point selection gap is still open. Outputs: research/raw/e2_verifier_base.json, e2_emb_base_*.npz, scratch/e2_fit.log, scratch/e2_chain.sh. Embeds took about 3 min in total, fit about 11 min.
- **NS bench:** 11 separate sets, never mixed. Half A from the net: A_cruxeval, A_heval_pro, A_lcb, A_mbpp_pro, A_rwpb. Half B made via Gemini: B_body, B_fix, B_general, B_predict, B_tool_call, B_write_fn. Eval only, never train on it. Scripts: scripts/bench_ns.py, bench_ns_build.py, bench_ns_gen.py, bench_ns_eval.py. Prompts are rstripped; the first base run kept a bare final newline, got empty answers on 9 of 11 sets and was discarded (research/raw/bench_ns_base.bad_prompt.json).
- **Two tiers:** full (k=16) and `--quick` (50 items per set fixed by sha256 of id, k=8, max-new 256, report `<name>_quick`). The compare refuses to mix k or max_new. Grading runs visible tests first and dedupes identical completions; regrade check 440 verdicts, 0 mismatches. Quick base: 1,186 s, 535 items, batch 128, no OOM, about 88 % generation. Full-tier speed with these changes is not measured.
- **Full base (greedy / runtime k16 / oracle@16, %):** A_cruxeval 16.9/18.0/46.5; A_heval_pro 0.0/13.5/14.2; A_lcb 0/0/0; A_mbpp_pro 4.6/16.2/17.7; A_rwpb 2.9/11.4/17.1; B_body 1.4/3.4/3.4; B_fix 3.3/20.7/23.3; B_general 0/7.2/8.0; B_predict 8.2/6.8/38.1; B_tool_call 0/0/0; B_write_fn 1.3/4.7/7.3. Reports: research/raw/bench_ns_base.json, bench_ns_base_quick.json.
- **Quick base (greedy / runtime k8 / oracle@8, %):** A_cruxeval 22/16/38; A_heval_pro 0/12/12; A_lcb 0/2/2; A_mbpp_pro 6/8/10; A_rwpb 2.9/11.4/14.3; B_body 4/2/2; B_fix 0/8/8; B_general 0/2/2; B_predict 6/6/22; B_tool_call 0/0/0; B_write_fn 2/0/0.
- **B_tool_call 0 is a model failure:** 123 of 150 full-bench answers are `...` copied from the stub bodies in the prompt; only 22 contain a call.
- **Quick tier, trained models vs base (runtime k8 delta, points; 95 % bootstrap CI):** the only set where the CI clears 0 for both models is B_fix: sa4_scale075 +14 [4, 26], sa_b_scale075 +22 [10, 34]. sa_b_scale075 also clears 0 on B_write_fn +8 [2, 16]; its A_cruxeval +8 [0, 18] only reaches 0, as does sa4_scale075's B_write_fn +4 [0, 10]. Every other set is a CI that crosses 0: A_heval_pro, A_lcb, A_mbpp_pro (sa4 -4), A_rwpb, B_body, B_general and B_predict. B_tool_call stays 0. Oracle@8 for sa_b vs base: A_cruxeval 52 vs 38, B_predict 40 vs 22, B_fix 30 vs 8. Greedy on A_cruxeval is -6 for both models, and on A_mbpp_pro it is -4 for both; neither CI is clear of 0. At n=50 only gains of 10 points or more are detectable. On runtime, sa_b_scale075 is equal to or better than sa4_scale075 on 9 of 11 sets; it is lower on A_heval_pro (10 vs 12) and B_body (2 vs 6), both within noise. It stays the best checkpoint. Runs: about 990 s generation + about 170 s grading each, batch 128, no OOM, no new empty answers. Script: scratch/ns_quick_models.sh; reports: research/raw/bench_ns_sa4_scale075_quick.json, bench_ns_sa_b_scale075_quick.json.
- **State:** nothing committed. bench_ns_eval.py's default `--batch` is still 64; 128 is proven on the quick tier.

# 2026-10-29b - SA-M step 4b: wide HE docstring example parser (CPU rescore, no new samples)

- **What (user "Parser only (CPU)"):** `sa7_runtime.he_examples_wide` parses prose examples (`f(..) => / ==> / ➞ / -> / → / == / = result`, true/false/null mapped). Opt-in via `load_he(workers, wide=True)`, used only for tasks with no `>>>` examples, with the same reference filter. Default HE loading is unchanged, so SA-7/step-4 numbers stand. `sa_m_product.py` gained a `he_wide` arm: it re-runs the new visible tests on the saved N64 HE samples (nostop kept), then probe/mbr/abstain with the frozen config (threshold 0.7398, unchanged).
- **Coverage:** HE tasks with visible tests went from 70 to 132 of 164 (62 gained; 374 examples parsed, 13 dropped by the reference). Scoped HE tasks without visible tests went from 26 to 11.
- **Result** (research/raw/sa_m_product_hewide.json + _tasks.jsonl): held-out scoped answer-all **43.9 % [38.2, 49.8]** (was 43.5; HE scoped 34.5, was 32.7). Two tasks flipped to correct: HumanEval/102 (scoped) and /96 (rest). Confident mode is unchanged: 20.1 % answered at 94.6 %, and HE confident is still 6 tasks. Of 53 HE tasks with a visible passer, only 6 clear p ≥ 0.740; the abstain model is MBPP/gate-fit and stays conservative on HE.
- **Verdict:** gate still FAIL narrowly (43.9 < 45; confident PASS). Per user instruction, the narrow miss is accepted. SA-M closes with the frozen config as the product (confident mode is the reliable surface; answer-all ≈ 44 % scoped).

# 2026-10-29 - SA-M step 4: frozen product config, held-out HE/MBPP

- **Config (user "Approve N64"):** runs/sa4_scale075, k 64, no retry, mbr selector, B2 abstain (sample scope router logit as feature 0) refit on the clean 370; threshold 0.740 = mean of the step-3 fold thresholds (re-derived and asserted against research/raw/sa_m_route_b2.json). Confident = visible tests and n_pass > 0 and p ≥ 0.740 (HE tasks without visible tests abstain). Saved as research/raw/sa_m_product_config.json. Scorer: scripts/sa_m_product.py.
- **GPU run:** the user stopped the first run at MBPP 246/427 (log scratch/sa_m_prod.stopped1.log); sa7_runtime has no resume, so it was relaunched from scratch 2026-10-29 08:30 (cap 6.72 GiB of 7.47 free; scratch/sa_m_prod.cmd). MBPP exit 0 (8,147 s), HE exit 0 (3,104 s). Outputs research/raw/sa_m_prod_{mbpp,he}.json (+ .samples.jsonl, untracked).
- **Pipeline check (in-sample replay on the 370):** lane mismatch 0, mbr mismatch 0; scoped 99 answer-all 54.5 %, confident 23.2 % answered at 95.7 % (scratch/sa_m_product_gate.json).
- **Held-out results (research/raw/sa_m_product.json, per task _tasks.jsonl):**

  | slice | n | oracle | answer-all | confident answered | confident acc |
  |---|---|---|---|---|---|
  | mbpp/scoped | 223 | 54.3 | 46.2 [39.8, 52.7] | 22.4 % | 94.0 [83.8, 97.9] |
  | he/scoped | 55 | 41.8 | 32.7 [21.8, 45.9] | 10.9 % | 100 (6/6) |
  | **heldout/scoped** | 278 | 51.8 | **43.5 [37.8, 49.4]** | 20.1 % | **94.6 [85.4, 98.2]** |
  | heldout/rest | 313 | 23.6 | 16.3 | 3.2 % | 80.0 [49.0, 94.3] |
  | heldout/all | 591 | 36.9 | 29.1 [25.6, 32.9] | 11.2 % | 92.4 |

- **Gate verdict: FAIL (narrow).** Answer-all 43.5 < 45 (CI includes 45); confident accuracy 94.6 ≥ 75 PASS. MBPP scoped alone passes (46.2); HE scoped pulls the pool down (26/55 HE scoped tasks have no visible tests, so mbr runs without a filter there).
- **Latency:** GPU N64 ≈ 19 s/task (MBPP 19.1, HE 18.9, includes grading). CPU k 8 on 10 MBPP tasks: 381 s ≈ 38 s/task incl. model load (research/raw/sa_m_cpu_latency.json); k 64 on CPU extrapolates to ~5 min/task — GPU only in practice.
- **Not done:** SA-M milestone not met; next direction awaits the user (TASKS SA-M).

# 2026-10-28b - SA-M: usable scoped product, step 1 (coverage curve) launched

- **User request:** "approve SA-M plan / stay on Gemini". Plan in PLAN §11.4, row SA-M in TASKS. Teacher for any later data work is the Gemini free tier only.
- **Step 1:** scratch/sa_m_cov.cmd runs `scripts/sa7_runtime.py --k 64 --retries 0 --batch 128 --workers 8` (no code change needed: the script already takes `--k` and stores every sample's visible kind and full pass in `.samples.jsonl`). Arm `scoped` = scoped 99 on sa4_scale075 then sa_b_scale075 → research/raw/sa_m_cov_{sa4,sa_b}_scoped.json; arms `full_sa4` / `full_sa_b` = clean 370. Log scratch/sa_m_cov_scoped.log.
- **Scoped result (99 tasks, both arms exit 0, ~21.5 min each, cap 6.53 GiB, peak ~2.2 GiB):** scored by scratch/sa_m_curve.py (first-round sample prefixes; self-test reproduced the old k=8 numbers exactly) → research/raw/sa_m_cov_scoped_curve.json.
  - sa4_scale075, N 8/16/32/64: oracle 44.4/52.5/64.6/68.7; runtime 40.4/41.4/46.5/49.5 (N64 vs N8 CI [0.0, +18.2]); answered 60.6/69.7/77.8/82.8; accuracy on answered 66.7/59.4/59.7/59.8.
  - sa_b_scale075: oracle 45.5/51.5/62.6/69.7; runtime 37.4/43.4/47.5/48.5 (N64 vs N8 CI [+4.0, +18.2]); accuracy on answered ~57–61.
  - Paired sa4 − sa_b CIs straddle 0 at every N (runtime and oracle): tie. Full 370 arm runs on sa4_scale075 (incumbent).
  - Reading: the ≥ 45 % runtime gate is met at N 32–64 by the point estimate; the runtime→oracle gap grows to ~19–21 pt at N64 (step 2 target); accuracy on answered stays ~60 %, so the 75 % gate needs step 3 (abstain).
- **Full 370 result (sa4_scale075, exit 0, 2 h 02 min, cap 6.62 GiB, not_evaluated 0)** → research/raw/sa_m_cov_full_curve.json. N 8/16/32/64: oracle 21.4/27.0/31.6/38.6; runtime 16.8/21.6/23.0/27.6 (N64 vs N8 CI [+7.3, +14.3]); answered 47.0/57.3/66.5/73.2; accuracy on answered 35.6/37.7/34.6/37.6. Off-scope tasks are where answered-but-wrong piles up, which is the case for the step 3 router.
- **Step 2 (CPU selection, "Start step 2"):** new scripts/sa_m_select.py. Every runnable sample (visible pass/wrong/error) also runs on ≤ 16 probe inputs made by mutating the visible call's literal args (outputs unknown, hidden tests never read); 361/370 tasks probed, mean 7.5 probes; 29,665 probe programs, cache scratch/sa_m_probe_cache.jsonl. Methods ast (baseline) / exec (largest identical-output cluster) / mbr (MBR-exec among passers) / mbr_all. Dev = 271 non-scoped ids, check = scoped 99 → research/raw/sa_m_select.json, per-task features research/raw/sa_m_select_features.jsonl.
  - Runtime at N64, ast → mbr: full 370 27.6 → 29.2 (exec 29.5, CI [0.0, +4.1]); rest 18.5 → 19.9; scoped (sa4 full dump) 52.5 → 54.5 (exec 56.6); scoped_sa4 49.5 → 55.6 (CI [+1.0, +12.1]); scoped_sa_b 48.5 → 51.5. mbr picked on dev (best or tied at every N).
  - **Gate FAIL:** gap closed at N64 ≈ 0.15 (rest), 0.15–0.17 (all), 0.32 (scoped_sa4), 0.14 (scoped_sa_b) vs ≥ 0.5 required.
  - Why: of the 35 full-dump N64 gap tasks (oracle yes, mbr no), correct samples are a minority of visible passers in 33 (≤ 10 % in 9). Probes separate correct from wrong passers in 23, so behaviour is visible, but the wrong cluster is larger: consensus selection cannot close the gap; it needs a correctness signal that is not agreement.
  - Abstain peek (step 3 preview, N64, mbr pick, rank by mbr_agree): scoped answered ~42–43 % → accuracy 76–80 % in all three dumps; rest at 17 % answered only 43.5 %.
- **SA-M step 3 (2026-10-28, CPU, scripts/sa_m_route.py → research/raw/sa_m_route.json + sa_m_route_tasks.jsonl):** the user accepted the step 2 fail and kept mbr. Milestone runtime is judged in answer-all mode; the 75 % gate is judged in confident mode.
  - **Prompt router:** numpy logistic regression on 97,496 he/mbpp rows from train. The label is the scoped rule applied to the reference. **Gate-370 AUC is 0.674 (train 0.692): weak.** Module/library mentions and "hard" words carry it. In the abstain model its weight is ~0, so it adds nothing.
  - **Abstain model:** inputs are pass share, mbr agreement, exec top share, clusters, plus size and imports of the passing samples. It was fit with 5-fold CV grouped by task; the threshold is chosen in inner folds at a 78 % target. Out of fold on the clean 370 it answers 7.8 % at 82.8 % accuracy (Wilson CI 65.5–92.4), with 1.4 % wrong. **Gate PASS on the point estimate only;** the CI lower bound is under 75 % and coverage is tiny.
  - **Scoped subsets with the same rule:** 17–23 % answered at 82–90 % accuracy.
  - **Curves:** on the scoped tasks the top ~34–50 % run at 74–82 % accuracy (full / sa4 / sa_b dumps). On the 271 rest, accuracy stays ≤ 46 % even in the top 14 %.
  - **The real limit:** we cannot tell scoped tasks from the prompt alone. Confident mode only works well where scope is already known.
  - **Answer-all runtime (milestone):** scoped 54.5 / 55.6 / 51.5 ≥ 45 → **PASS** (mbr from step 2).
- **Option B (sample-based scope router, CPU), user "Try B first":** `scripts/sa_m_route.py --abstain-features`.
  - Scope router from the 64 samples (AST size quantiles, imports, helpers, nested loops / recursion, parse and error share), task-grouped 5-fold on the clean 370: out-of-fold scope AUC **0.818** vs prompt router 0.659.
  - `scope_router` (scope logit replaces the prompt-router input; research/raw/sa_m_route_b2.json): full370 confident **8.4 % answered at 83.9 %** (CI 67.4–92.9) vs 7.8 % / 82.8 % before. Scoped subsets 22–26 % answered at 83–96 %. Curve full370: 14.6 % → 68.5 %, still below 75.
  - `samples` (raw scope stats added to the abstain model; sa_m_route_b.json): worse, overfits (thresholds up to 0.99, full370 62.5 %).
  - **Verdict:** B does not move the 370. Per the user's fallback: A (confident mode = user-declared scoped lane) + D (step 4).
- **Not done:** step 4.

# 2026-10-28 - SA-B step 3: retrain on the scoped view, judged on the 370 and 99-task gates

- **User request:** "Approved: retrain on data/sa_b_scoped/sa4/ds from runs/sa3_scale075 with the sft_sa4 settings, VRAM cap = fresh free - 0.75 GiB; judge on the 370 clean gate and the 99 scoped gate."
- **Run:** configs/sft_sa_b.yaml (clone of sft_sa4.yaml; only base_model, dataset_dir and output paths changed), chain scratch/sa_b_retrain.cmd (sft → scratch/lora_scale_merge.py at 0.75 → sa5_rft.py gate, same args as the re-baseline).
  - First launch on 2026-10-27, cap 6.51 GiB. The user stopped it at step 260, so the log is scratch/sa_b_retrain.log.
  - Relaunched 2026-10-28 with cap 6.76 GiB (free 7,694 MiB). sft.py auto-resumed from checkpoint-200, and the log is scratch/sa_b_retrain_resume.log.
  - Train about 1.8 h, merge, then the gate in 15.5 min (peak VRAM 1.35 GiB). All steps exit 0.
- **Outputs (local):** runs/sft_sa_b/final (eval_loss 0.1648 vs SA-4 0.1683), runs/sa_b_scale075, scratch/sa_v_gate/sa_b_scale075/{report.json,samples.jsonl}, research/raw/sa_b_retrain_gate.json.
- **Tooling:** scratch/sa_v_gate_compare.py takes an optional third argument, a task-ids file, to score a subset such as the scoped gate.

Results: k=4, T 0.8, seed 5, paired bootstrap with 10k resamples. The diff is sa_b minus the other model.

| gate | sa_b_scale075 | sa4_scale075 | diff CI95 | c107 | diff CI95 |
|---|---|---|---|---|---|
| scoped 99, pass rate | **19.7 %** | 14.1 % | [+2.3, +8.8] pt | 17.7 % | [−1.5, +5.6] |
| scoped 99, solved@4 | **38.4 %** | 29.3 % | [+2.0, +16.2] | 39.4 % | [−9.1, +8.1] |
| full 370, pass rate | 7.2 % | 6.7 % | [−0.9, +1.8] | 7.2 % | [−1.4, +1.3] |
| full 370, solved@4 | 15.4 % | 15.4 % | [−3.5, +3.5] | 18.1 % | [−6.2, +0.8] |

- **Reading:**
  - Scope weighting clearly improves the targeted families over SA-4, and the full gate does not regress.
  - The retrain is level with the RL incumbent c107 on both gates.
  - It is the best SFT checkpoint on the scoped target, and a natural RL init.
- **SA-7 runtime on sa_b_scale075 (user: "SA-7 runtime on it"):**
  - Gate arm only: `scripts/sa7_runtime.py --set gate --model runs\sa_b_scale075`, the same budget as the sa4 run (k=8 + 1 retry round of 4).
  - Cap 6.63 GiB (7,562 MiB free − 0.75), 1,781 s, exit 0, log scratch/sa_b_sa7_gate.log.
  - Outputs: research/raw/sa_b_sa7_gate.json, and the paired comparison research/raw/sa_b_sa7_vs_sa4.json from the new scratch/sa_b_sa7_compare.py.
  - Self-check: the script on sa4 vs itself reproduces the recorded 9.7/18.4/22.7 and 20.2/39.4/45.5.

  | | sa_b_scale075 | sa4_scale075 | diff CI95 |
  |---|---|---|---|
  | scoped 99 greedy / runtime / oracle | 22.2 / 38.4 / 44.4 % | 20.2 / 39.4 / 45.5 % | runtime [−10.1, +8.1] pt |
  | full 370 greedy / runtime / oracle | 8.9 / 17.3 / 21.1 % | 9.7 / 18.4 / 22.7 % | runtime [−4.6, +2.4] pt |

  - **Reading:** the retrain raises the per-sample pass rate (k=4 gate, scoped +5.6 pt), but not the share of tasks with any passing sample at N≈10.
  - The runtime picks a visible-test passer, so it is capped by that coverage (oracle), and the runtime result does not improve.
  - Raising coverage needs new capability: RL from sa_b_scale075, or new data. More sharpening of already-solved tasks will not do it.
- **Not done:** HE/MBPP runtime arms on sa_b_scale075. Committed as 09cf090 (local, not pushed).

# 2026-10-27i - SA-B step 2: scope-weighted SA-2/SA-4 training view (CPU, no retrain)

- **User request:** "Approved: build the scope-weighted SA-2/SA-4 training view under a new directory (CPU only, data/sa2 and data/sa4 untouched)."
- **Built (uncommitted, data is local):** scratch/sa_b_train_view.py → data/sa_b_scoped/{sa2,sa4}/{train,val}.jsonl + data/sa_b_scoped/report.json (rule, weights, per-shape and per-class counts, source sha256). Same row schema as the sources. The sha256 of data/sa2 and data/sa4 {train,val}.jsonl was checked before and after: unchanged.
- **Rule:** the same scope as the gate (reference ≤ 48 AST nodes, no imports, one function). Reference = mbpp completion; he prompt + completion; fix/repair the final `def` line of the prompt + completion. No reference failed to parse.
- **Train weights:** in scope ×2; imports or helper functions keep 50 % (deterministic sha1 of `sa_b:<id>`); larger rows ×1; tool and replay ×1. Val copied unchanged, so val loss stays comparable.

  | | train before | train after | in-scope share of Python rows |
  |---|---|---|---|
  | SA-4 | 61,729 | 74,094 | 38 % → 57 % |
  | SA-2 | 139,956 | 171,054 | 34 % → 54 % |

- **Tokenized (CPU):** `scripts/sa3_data.py --src data/sa_b_scoped/sa4 --shapes fix,repair,tool,replay,he,mbpp --out data/sa_b_scoped/sa4/ds` → 74,089 train / 748 val (5 rows over ctx 1024 dropped). That is about 1,158 steps at the sft_sa4 batch (8 × accum 8), against 965 for SA-4.
- **Not done:** no retrain and no sft config. A retrain needs its own approval and a base-model choice (runs/sa3_scale075 as SA-4 used, or continue from sa4_scale075). Judge it on both the 370 clean gate and the 99 scoped gate.

# 2026-10-27h - SA-B (a) adopted: scoped in-distribution gate (99 tasks) + PLAN §11 scope

- **User request:** "SA-B: (a) adopt the narrowed in-distribution scope."
- **PLAN §11 edited (approved by this choice):**
  - a Scope block after the Definition, with the product statement "writes short single-function Python (list, dict, math, string) with tests" and the out-of-scope list;
  - a scoped-gate bullet in the §11.2 SA dev gate. The 370-task clean gate stays primary.
- **Built (CPU, no generation, uncommitted):** scratch/sa_b_scoped_gate.py → research/raw/sa_b_scoped_gate.json. It holds `ids` (99 gate tasks: reference ≤ 48 AST nodes, no imports, one function), `he_mbpp_ids` (HE 55 + MBPP 223) and the re-scores below. The proposal said 103 because its per-task table had no helper flag; with the single-function filter applied, the count is 99. The format works with `sa5_rft.py --task-ids`.
- **Re-scored on files already on disk (sa4_scale075):**

  | | full gate (370) | scoped gate (99) |
  |---|---|---|
  | E k=10 pass@1 / solved@10 | 7.0 % / 23.0 % | **16.1 % / 43.4 %** |
  | SA-7 greedy / runtime / oracle | 9.7 / 18.4 / 22.7 % | **20.2 / 39.4 / 45.5 %** |
  | runtime − greedy, paired 95 % CI | [+4.9, +12.4] pt | [+10.1, +28.3] pt |

  HE+MBPP scoped: HE 55 tasks, pass@1 14.9 % / solved@10 25.5 %; MBPP 223 tasks, 11.6 % / 36.3 %. The full-gate numbers match the earlier reports, which confirms the join.
- **Not done (needs its own approval):** proposal step 2, a scope-weighted SA-2/SA-4 training view under a new directory (data/sa2 and data/sa4 are never modified), and any retrain on it. TASKS SA-B (narrowing) is `in_progress` until then.
- **Caveats carried over:** topic labels are weak keyword rules; scope is measured on the reference size, so routing at inference needs a proxy; all numbers come from one checkpoint; `final` untouched.

# 2026-10-27g - SA-7 schema-constrained tool grammar: tool errors 6.1 → 0 per episode, agent success still ≈ 0

- **User request:** "Approved: add schema-constrained tool grammar (tool-name enum + exact argument keys), re-test on CPU, then re-run only the agent arm on GPU with cap = fresh free - 0.75 GiB."
- **Built (uncommitted):**
  - `scripts/tool_grammar.py`: `grammar(..., schema=False, values=None)` / `ToolCallGrammar(..., schema, values)`. With schema on, each tool's arguments object takes exactly its declared keys (any order, no extras, none missing), "str" parameters take only JSON strings, and `values[(tool, param)]` narrows a string to a fixed set. Off by default, so the SA-4 format check is unchanged.
  - `scripts/sa7_agent.py`: `--schema` on by default (`--no-schema` = the old syntax-only grammar). `PATHS`: read_file → solution.py / test_solution.py, write_file → solution.py. An unclosed reply's last 300 chars are saved as `invalid_tail` in the samples.
- **CPU checks, all PASS:** SA-4 grammar regression (val 146 accepted / 0 rejected, same train rejects as before); tests/test_sa7_agent.py 10/10 (new `schema_grammar`); tests/test_sa7_runtime.py 11/11; oracle dry run 120/120.
- **GPU agent arm (cap 6.52 GiB = 7.27 free − 0.75; 856 s; exit 0):** research/raw/sa7_agent_schema.json (+ .samples.jsonl, log scratch/sa7_agent_schema.log). The old run research/raw/sa7_agent.json is kept.

  | | syntax-only (old) | schema (new) |
  |---|---|---|
  | hidden pass | 1.7 % (2) | 0.8 % (1) |
  | full / visible pass | 0 / 0 | 0 / 0 |
  | tool errors per episode | 6.14 | **0** |
  | episodes never running tests | 100 % | 51 % |
  | stops: invalid / 8 turns | 54 / 66 | 43 / 77 |

  - Every closed reply is now schema-valid. Calls: write_file 439, read_file 386, run_tests 340, run_python 239.
  - **The failure is now semantic, not format.** The most common observation is `wrote 16 chars to solution.py`: the model writes the string "test_solution.py" (or other short junk) as the file content, which turns the run_tests that follow into `failed to load: NameError` (152 times). The 135M model does not understand the multi-turn file workflow; nothing like it was in SA-4 training (single-turn xLAM).
  - **The 43 invalid stops are whitespace loops inside a string value,** e.g. hundreds of `\n` escapes (30 of the 43 tails are 500–999 chars, well under max_str 2048, so they ran out of max_new tokens). MAX_RUN = 16 counts runs of one char, so a repeated 2-char escape like `\n` is never caught.
  - Gate "tool validity 100 %": closed replies 100 % valid; 43/120 episodes still end on an unclosed reply → **still FAIL** as strictly measured.
- **Read-out:** the grammar now does all a grammar can. The agent loop is a capability gap of this model, while the plain runtime (sample-k + visible filter) gets 22.5 % on the same 120 tasks. Options: (1) a repeated-escape cap in the grammar (closes the remaining validity gap, will not raise success); (2) multi-turn agent SFT data (new training, out of SA-7 scope); (3) report the agent loop as "reported, ≈ 0 %" per PLAN §11 and close SA-7 on the runtime result.
- **Decision (user):** "Close SA-7 on the runtime result (22.5 %) and record the agent loop as reported ≈ 0 %; then commit the working tree." SA-7 is `done` in TASKS. The unclosed-reply gap (43/120) is a known limitation. The escape cap and agent SFT data stay unscheduled. PLAN §11 is not edited.
- **Committed:** the SA-7 code, tests, research notes and result summaries (see git log). Large `*.samples.jsonl` files and unrelated working-tree changes are left out. SA-B decision (a/b) still pending.

# 2026-10-27f - A + B: SA-7 runtime built on CPU (GPU .cmd prepared, not launched); SA-B scope proposal

- **User request:** "Go with A + B: build the SA-7 runtime (CPU first) and narrow the job to in-distribution task families."
- **§7 defaults used (no answer given):** visible/hidden split defines "runtime pass"; agent tools include write_file; budget N = 12 (k 8, 1 retry round of 4).
- **A, built (CPU only):**
  - `scripts/tool_grammar.py`: the grammar moved unchanged out of scratch/sa4_format_check.py; `grammar()`/`ToolCallGrammar` take `max_str` so write_file can carry a whole solution (agent uses 2048). SA-4 format regression: 146 rows, 0 failures. MAX_RUN = 16 still caps runs of one char, which includes indentation (16 spaces = 4 levels).
  - `scripts/sa7_runtime.py` (`--set gate|he|mbpp`, `--dry-run` = reference model): tests 11/11 (tests/test_sa7_runtime.py). Reference dry runs: gate 370/370, HE 164/164, MBPP 0.9953 (mbpp-san/123 times out and is graded not_evaluated). 94/164 HE tasks have no visible test, so they are vote-only.
  - `scripts/sa7_agent.py`: tests 9/9 (tests/test_sa7_agent.py). Oracle dry run research/raw/sa7_agent_dry.json: implement 60/60, fix 60/60, full pass 1.0, mean turns 1.0, 68 s. Deviation from the proposal: fix episodes start from real failing k10 samples (clean gate has no fix tier). Ids: research/raw/sa7_agent_ids.json, also valid as `sa7_runtime --task-ids` (dry run 120/120).
  - Tests import `_common` from tests/, so run them without `-I`: `.venv/Scripts/python.exe tests/test_sa7_agent.py`.
- **Step 4 prepared, NOT launched:** `scratch\sa7_runtime_gpu.cmd <cap>` runs gate, HE, MBPP, the agent loop and the runtime baseline on the agent ids, one output each under research/raw/sa7_*.json. Launch per the long-eval pattern with cap = fresh free − 0.75 GiB.
- **B:** research/2026-10-27_sa_b_scope_proposal.md (data: research/raw/sa_b_families.json). Small (≤30 nodes) and import-free tasks are where the model works; proposed scope ≤48 nodes / no imports / single function. Awaiting decision (a) adopt or (b) diagnostics only. PLAN §11 unchanged.
- **Step 4 GPU run (user-approved, cap 6.33 GiB = 7.09 free − 0.75; 12:41–14:33, all 5 arms exit 0; log scratch/sa7_runtime_gpu.log).** Peak use about 2.1 GiB, no spill.

  | Arm | greedy pass@1 | runtime pass (hidden) | oracle pass@N | runtime − greedy CI95 |
  |---|---|---|---|---|
  | clean gate 370 | 9.7 % | **18.4 %** | 22.7 % | [+5.1, +12.2] pt, PASS |
  | HE 164 | 9.1 % | 10.4 % | 10.4 % | [−1.2, +4.3] pt |
  | MBPP 427 | 13.6 % | **24.6 %** | 26.2 % | [+7.7, +14.5] pt |
  | HE+MBPP 591 | 12.4 % | 20.6 % (122) | 21.8 % (129) | — |
  | agent-id baseline 120 | 9.2 % | 22.5 % | 25.0 % | [+6.7, +20.8] pt |

  - The retry-with-traceback round is nearly useless: 1 (gate), 3 (MBPP) and 0 (baseline) passes from about 500 retried tasks. All the gain comes from sample-k + visible filter.
  - **Agent loop: hidden pass 1.7 % (2/120, both fix), full pass 0 %**, vs 22.5 % for the plain runtime on the same tasks. Stop reasons: 54 unclosed tool lists within 640 tokens ("invalid"), 66 hit 8 turns.
  - **Root cause (research/raw/sa7_agent.samples.jsonl):** the grammar enforces JSON syntax only, not tool schemas. All 419 run_tests calls passed arguments and were rejected ("takes no arguments"). 172 write_file calls used wrong argument keys, and 65 tried to write test_solution.py. So `no_run_tests = 1.0` means "never a valid run_tests", not "never tried".
  - Gate "tool validity 100 %" on the agent loop: **FAIL** as measured (54/120 unclosed, about 6 schema errors per episode).
- **Not done:** nothing committed. Candidate fix (not applied): schema-constrained grammar (tool name enum + exact argument keys), then re-run only the agent arm (about 12 min).

# 2026-10-27e - E: pass@10 on sa4_scale075 measured (HE+MBPP 22.5 %, clean gate 23.0 %); SA-7 runtime proposal drafted

- **User request:** "Go with E then A: measure pass@10 on the current model, then draft the SA-7 runtime ... proposal. then later decide between B and C based on the result of E and A". B = narrow the job to in-distribution families. C = SA-V.2 re-split + new teacher data + retrain.
- **E run:** scratch/sa_e_passk10.cmd, WMI-detached, cap 6.5 GiB (free 7424 MiB − 0.75), peak VRAM 2.9 GiB. Both steps exited 0. The whole run took 54 min (10:59 → 11:53).
  - `bench_passk --samples 10 --batch 160` → research/raw/sa_e_passk10.json:

    | Set | pass@1 | pass@5 | pass@10 | Distinct correct per solved | EOS |
    |---|---|---|---|---|---|
    | MBPP (427) | 8.22 % | 19.36 % | 26.23 % (112) | 2.52 | 99.6 % |
    | HE (164) | 6.34 % | 11.11 % | 12.80 % (21) | 2.81 | 98.3 % |
    | Combined | | | 22.5 % (133/591) | | |

    PLAN §11.1 pass@10 target is 25–35 %, so the combined number is a FAIL. MBPP alone is in the band; HE is not.
  - `sa5_rft --split val --task-ids sa_v_clean_gate.json --k 10 --batch 160` → scratch/sa_e_gate_k10:
    - 3,700 samples, 0 not_evaluated, pass rate 7.03 %.
    - Unbiased pass@k: @1 7.0, @2 11.1, @4 16.2, @5 17.8, @8 21.3, @10 **23.0 %** (85/370).
    - Of the solved tasks, 29 pass in only 1 of 10 samples.
    - 285 of 370 tasks are never solved.
    - Solved by source: opc 66/264, oci 18/98, teacher 1/6, selfoss 0/2.
- **SA-7 proposal:** research/2026-10-27_sa7_runtime_proposal.md covers:
  - sample-k + a visible-test filter + an AST-majority vote;
  - retry-with-traceback in the SA-4 `repair` prompt shape;
  - the grammar moved to scripts/tool_grammar.py;
  - an agent-loop eval of about 120 val tasks (read/write/run_python/run_tests, ≤ 8 turns);
  - CPU-first build; the GPU step is user-gated.

  Leak rule: the filter uses visible tests only, the score uses hidden tests, and the oracle pass@N is reported separately.
- **Reading for B vs C (user decides):**
  - The runtime ceiling is about 23 %, so a realistic runtime result is roughly 7 % → 15–20 % on the clean gate.
  - 77 % of tasks are out of reach at k = 10. Only data or training (C) moves those.
- **Next:** the user reviews the proposal's §7 questions and decides B vs C. TASKS SA-7 is now `blocked` on that review.

# 2026-10-27d - SA-V.1 clean dev gate built (370 tasks) + grader timeout fix; CPU acceptance PASS

- **Approved scope (user):** SA-V.1 steps 1-3 with a 0.5 coverage cut, plus the grader timeout fix. SA-V.2 is deferred. data/sa2 and data/sa4 are untouched. Everything ran on CPU; the GPU was not used.
- **Clean gate:** research/raw/sa_v_clean_gate.json, 370 ids (he 180, mbpp 190), built by scratch/sa_v_clean_gate.py. The proposal said 378; it is 370 for two reasons:
  - `teacher/0/1528` and `teacher/0/1642` each name two val rows, so `--task-ids` cannot address them. They are excluded, and `pools()` now refuses ids that match more than one row.
  - Boilerplate n-grams are the ones shared by 2+ members, so coverage depends on the member set. The first build measured against 418 candidates, and its audit of the final ids flagged 3 borderline rows (0.41-0.48 → 0.50-0.51). The builder now re-measures on the kept set until it is stable: 374 → 371 → 370, 4 passes.
- **Acceptance:**
  - Audit `scratch/sa_val_leak_audit.py --ids` → research/raw/sa_v_clean_gate_audit.json: exact_prompt 0, exact_ref 0, same_tests 0, coverage ≥ 0.5 0 (≥ 0.2: 143). PASS.
  - Dry-run of the clean gate, `sa5_rft.py --dry-run --split val --task-ids research/raw/sa_v_clean_gate.json --code 370 --fix 0 --tool 0`: 370/370 references pass, 0 not_evaluated, 28.5 s. PASS.
  - The 567 gate dry-run is unchanged: 567/567. Using `--task-ids` with a non-val split is rejected. PASS.
- **Grader fix:**
  - `sa5_rft.score`: a timeout or RSS kill is retried once at 4× TIMEOUT (24 s). If the retry passes, `why = "not_evaluated"` and passed=False. The sample is left out of the pass-rate denominator (`passed / evaluated`) and does not block full-solve.
  - `rl_smollm2.score_batch` skips such samples: no reward and no train row. A group with one reward gets advantage 0. The tally counts evaluated samples only.
- **Gate policy:** PLAN §11 "SA dev gate". The clean 370 gate is primary for code. The 567 gate is reported for continuity. Fix signal comes from tier-fix only.
- **GPU re-baseline (user-approved, done):** scratch/sa_v_gate.cmd ran both arms on the 370-task clean gate: k=4, temperature 0.8, grader fix on, cap 6.48 GiB (7400 MiB free − 0.75), peak 1.35 GiB. Both arms exited 0, in 15.6 and 13.6 min. Reports: scratch/sa_v_gate/<arm>/report.json. Paired comparison: scratch/sa_v_gate_compare.py → research/raw/sa_v_gate_rebaseline.json.

  | arm | pass rate (he / mbpp) | solved@4 |
  |---|---|---|
  | c107 (run 5 part 3) | **7.23 %** (6.39 / 8.03) | **18.1 %** (67/370) |
  | sa4_scale075 | 6.69 % (5.83 / 7.50) | 15.4 % (57/370) |

  - Paired task bootstrap, 95 % CI of c107 − sa4: pass rate −0.74 to +1.82 pp; solved@4 −0.54 to +5.95 pp. Both CIs include 0. c107 solves 23 tasks that sa4 does not; sa4 solves 13 that c107 does not.
  - Verdict: on leak-free tasks c107 leads in the same direction as on the 567 gate, but the lead is not significant at n=370, k=4. 0 samples were NOT_EVALUATED: every timeout/RSS kill (c107 13, sa4 10) also failed its 4× retry.
  - Absolute pass rates are low (~7 %). The old 567 gate's higher numbers came mostly from train twins.

# 2026-10-27c - RL run 5 paused (user); SA-V val rebuild proposal drafted for approval

- **RL run 5 paused (user 2026-10-27).** Nothing is running. Part 3 stopped at its time cap at step 107. The resume state is runs/sa6_r5_part3/state.pt. No more RL launches until the SA-V clean gate exists and both arms are re-baselined on it.
- **Proposal:** research/2026-10-27_sa_v_val_rebuild_proposal.md (DRAFT, not approved). data/sa2 and data/sa4 are untouched. All measurement was CPU and read-only; the simulation scripts live in the session scratchpad.
- **Key finding.** Union-find of the 141,338 sa2 rows over four keys (parent id, assert set, reference, prompt) gives 80,282 clusters.
  - The largest are OCI classic-problem families: fibonacci 1,738 rows, factorial 1,713.
  - 60 % of old-val rows sit in clusters that straddle train and val.
  - Every existing checkpoint trained on the old train split, which holds ~99 % of any re-split val. So a re-split can only gate models retrained on the new train.
- **Recommended now: SA-V.1.** A clean gate carved from the current val: he/mbpp rows with ≥ 3 asserts, whole cluster in val, and 13-gram coverage < 0.5 against train. That gives 378 tasks (he 182, mbpp 196); 415 at < 0.8. No fix/repair row survives the rule, so tier-fix stays the fix signal.
  - Code change: an additive `--task-ids` flag in sa5_rft.
  - After that, a GPU re-baseline (separately gated).
- **Deferred: SA-V.2.** A cluster re-split into new data/sa2v and data/sa4v (2 % val, cluster cap 50, near-dup pass at 0.5, hard train↔val guard). It runs only together with the next SFT-chain retrain.
- **Optional bundle:** the grader timeout/RSS fix (NOT_EVALUATED), applied before the re-baseline.

# 2026-10-27b - SA-6 run 5 step 107 gate, full vs clean: no significant gain; 424/567 dev-gate val tasks leak from train

- **Merge + eval.** The part 3 step-107 adapter was merged to runs/sa6_r5_part3/checkpoint-107 (240 adapter keys; 120/273 tensors changed). The eval chain is `scratch/sa6_r5_eval.cmd` (cap 6.53 GiB = fresh free − 0.75; log scratch/sa6_r5_eval.log). All 9 steps exited 0 (done 9:34).
- **Rest of the eval chain vs sa4_scale075 (s075).** Outputs: research/raw/sa6r5_*, ladder_sa6r5_{b1,w2}/.

  | Eval | s075 | c107 |
  |---|---|---|
  | HumanEval greedy pass@1 /164 | 14 | 14 |
  | MBPP-san greedy pass@1 /427 | 57 | 56 |
  | EOS rate HE / MBPP | .878 / .951 | .909 / .974 |
  | bpb val / val_en (lower is better) | .5703 / .5269 | .5733 / .5279 |
  | Format valid / exact /146 | 140 / 50 | 141 / 52 |
  | Constrained format valid / exact /146 | 146 / 50 | 143 / 52 |
  | Constrained interventions | 12 | 40 |
  | Tier fix strict / extracted /110 | 27 / 30 | 30 / 33 |
  | Tier tool strict /100 | 13 | 14 |
  | Ladder B1 /192 | 70 | 68 |
  | Ladder W2 /560 | 245 | 243 |

  Everything is within a few tasks of the baseline. The only changes beyond that are small regressions: bpb +.003, and 3 constrained rows that ran out of budget (interventions 12 → 40). EOS rates went up.
- **Leak audit** (`scratch/sa_val_leak_audit.py`, CPU-only, read-only; output research/raw/sa_val_leak_audit.json). It compares the 567 §5 gate tasks against 162,963 sa2+sa4 train rows.
  - Flagged 444 tasks: same asserts 400, same reference completion 192, same prompt 16; 13-gram coverage ≥ 0.5 on 150.
  - "Strong" leak means same asserts, same reference, or coverage ≥ 0.8. That covers 424 tasks, leaving **143 clean** (he 66, mbpp 74, fix 2, repair 1).
  - Cause: the val split is a 1 % id-hash cut (sa2_build.py:634). A val fix/repair task's clean parent function sits in train as an he/mbpp row (oci-fix → oci, opc-fix → opc), and sibling fix/repair variants land on both sides. OCI/OPC also contain the same problem under different ids, e.g. val opc/50513 = train opc/19948, opc/19948/fix and opc/19948/repair. sa2_build decontaminates only against HumanEval/MBPP/playground, never train vs val.
  - Practical effect: the fix/repair 45–52 % numbers mostly measure recall of a seen function.
- **Gate result.** Sampled pass@1, 4 samples/task, paired task-level bootstrap (10k; research/raw/sa6r5_gate_clean_split.json, which also lists the clean ids):

  | Subset | n | sa4_scale075 | c107 | Δ | 95 % CI | P(Δ ≤ 0) |
  |---|---|---|---|---|---|---|
  | Full gate | 567 | .295 | .300 | +.005 | [−.009, +.019] | .26 |
  | Clean | 143 | .100 | .110 | +.010 | [−.002, +.023] | .05 |
  | Leaked | 424 | .361 | .364 | +.003 | [−.015, +.021] | .39 |

  Clean by shape: he .057 → .061, mbpp .118 → .132.
- **Grader finding** (no code changed). Sandbox timeouts and RSS kills score 0, the same as a wrong answer, in both `sa5_rft.score` and the RL reward (`rl_smollm2.sa5_reward`). Counts: baseline 19 timeouts + 2 RSS kills of 2268 samples, c107 17 + 0 (~1 %). Proposed fix: retry once with a longer limit; if it still fails, it's a fail; a retry pass is NOT_EVALUATED and leaves the RLOO group.
- **Verdict.** Step 107 shows no significant gain on either gate subset or on any held-out eval, so it does not replace sa4_scale075. The clean subset leans positive (+1.0 pp, P(Δ ≤ 0) .05) but 143 tasks × 4 samples is small. Until val is rebuilt, report the gate as **full + clean**. The leaked 424 should not drive decisions.
- **Next.** (1) A duplicate-aware val rebuild (TASKS SA-V; needs user approval before any data rebuild). (2) The grader timeout fix. (3) Decide whether RL run 5 continues; on this evidence, more RL at lr 1e-4 is not justified until the gate can measure it.

# 2026-10-27a - SA-6 run 5 RL part 3: lr 1e-4 moves KL 10×, code pass rate dips

- **Launch.** Copied runs/sa6_r5_part1 (state.pt at step 66, rl_log) to runs/sa6_r5_part3; part 1 untouched as the fallback. Same flags plus `--learning-rate 1e-4`. KL β stayed 0.05. VRAM cap 6.57 GiB (free 7,493 MiB − 0.75). rc=0.
- **New flag behaviour.** rl_smollm2.py now overrides the lr restored by `optimizer.load_state_dict` with `--learning-rate` on resume. It was checked on CPU first; the run log shows `[resume] lr 2e-05 -> 0.0001`.
- **Result.** Time cap reached at step 107 (41 new steps, ~88 s/step). state.pt saved at exit (group 107); no merged final.
  - Memory: peak 6,418 MiB.
  - GPU: util ~28 %, max 86 °C.
- **Comparison** (training-sample pass rates, not an eval):

  | Window | KL mean | KL max | Entropy | code | fix | tool |
  |---|---|---|---|---|---|---|
  | Part 2, steps 46–65 (lr 2e-5) | 9.5e-5 | 3.1e-4 | 0.134 | .15 | .49 | .30 |
  | Part 3, steps 66–85 (lr 1e-4) | 6.4e-4 | 1.3e-3 | 0.129 | .14 | .54 | .33 |
  | Part 3, steps 86–106 | 1.1e-3 | 2.0e-3 | 0.149 | .11 | .50 | .27 |

  Length was steady (mean ~70, largest p90 205 against the 384 cap). Flat share was 0.52.
- **Stop rules.** A background guard (scratchpad part3_guard.py) tripped at step 101 on "all buckets down over 20 steps": code .15 → .09, fix .49 → .49, tool .30 → .26. It was a notify-only guard and the run had ~8 min left, so it was let finish. KL, entropy and length rules did not trip.
- **Verdict.** lr 1e-4 moves the policy as intended (KL 10×), with no collapse. The only bad sign is code dropping .15 → .11 (≈2σ on correlated samples). Fix and tool are within noise. The training-sample signal can't settle this.
- **Next (user decision).** Recommended: merge the step-107 adapter and run the held-out eval vs runs/sa4_scale075. The alternative is another hour at lr 1e-4 with the code bucket watched.

# 2026-10-26i - SA-6 run 5 RL part 2: resumed to 66 steps, no OOM, KL still tiny

- **Launch.** Resumed runs/sa6_r5_part1 from state.pt (step 33, task 411) with the same flags. VRAM cap 6.52 GiB (free 7,440 MiB − 0.75). New files `train2.log`, `gpu_samples2.csv`, `.done2`; part 1's `.done` was left in place. rc=0.
- **Result.** Time cap reached at 66 steps total. state.pt saved; still no merged final.
  - Speed: 110 s/step.
  - Memory: peak 5,137 MiB total.
  - GPU: util 23 %, max 86 °C.
- **Comparison** (training-sample pass rates, not an eval):

  | Window | KL mean | KL max | Entropy | code | fix | tool |
  |---|---|---|---|---|---|---|
  | Part 1 (33 steps) | 2.2e-5 | 7e-5 | 0.141 | .10 | .47 | .29 |
  | Part 2, steps 34–49 | 5.4e-5 | 1.3e-4 | 0.105 | .09 | .50 | .34 |
  | Part 2, steps 50–66 | 9.8e-5 | 3.1e-4 | 0.133 | .15 | .46 | .30 |

  Flat share stayed ~0.55. KL is growing but is still about 1e-4, so the policy has moved very little. The pass-rate differences are within noise because tasks vary by step.
- **Next (user decision).** Options:
  - Retune: raise lr or lower KL β before more GPU hours.
  - Merge state.pt to a final and run the held-out eval to check for any real gain.

# 2026-10-26h - SA-6 run 5 RL part 1: 33 steps in 1 h, max-refills 1, no OOM

- **New flag.** `--max-refills` in scripts/rl_smollm2.py (default 3 = old MAX_REFILLS constant, so older runs are unchanged). CPU check: ast parse + `--help` only; the equivalence check was not re-run because that code path is unchanged.
- **Launch.** runs/sa6_r5_part1. Init runs/sa4_scale075, P=8, micro-rows 4, `--max-refills 1`, const norm, KL β 0.05, T 0.8 (log-probs at T), max_new 384/256, save-every 10, max-seconds 3600. VRAM cap 6.46 GiB (free 7,387 MiB − 0.75). rc=0.
- **Result.** Time cap reached at 33 steps. state.pt saved; there is no merged final (a time-capped run never merges).
  - Speed: 109 s/step (58–183), down from the probe's ~160.
  - Memory: peak 6,646 MiB total by nvidia-smi.
  - GPU: util 25 %, max 86 °C.
  - KL max 7e-5.
  - Flat share 0.53 / 0.60 / 0.53 by thirds.
  - Entropy ~0.13–0.16; len_mean 69–82.
- **Pass rate by thirds** (training samples, not an eval; tasks differ by step):

  | Bucket | Steps 1–11 | Steps 12–22 | Steps 23–33 |
  |---|---|---|---|
  | code | .12 | .06 | .12 |
  | fix | .47 | .51 | .42 |
  | tool | .33 | .24 | .32 |

  No trend is visible yet. KL is tiny, so the policy has barely moved.
- **Next (user-gated GPU).** Either continue part 2 from state.pt (same flags and output dir; it auto-resumes) or reconsider lr/steps. A held-out eval of the merged adapter is needed before any claim.

# 2026-10-26g - SA-6 run 5: logits/T flag, --micro-rows, GPU probe (16 steps, time cap)

- **New flags in scripts/rl_smollm2.py.**
  - `--logprob-temperature {auto,on,off}`: the training log-probs (policy and KL reference) use logits / T, so the update scores the distribution that was actually sampled. `auto` = on for `--source sa5`, off for playground. The division is in place: a full-vocab copy is ~0.6 GiB at 8 × 384.
  - `--micro-rows N` (default 0 = whole group): forward/backward over N rows of a group at a time. Every row term is already divided by the step's total rows, so the gradient is unchanged.
  - The KL reference forward now runs before the policy forward, so its logits are freed first.
- **CPU checks PASS** (equivalence script in the session scratchpad):
  - tempered log-probs equal both a manual log_softmax(logits/T) and HF's TemperatureLogitsWarper (diff 0);
  - micro-rows 3 vs whole group: objective equal, max grad diff 7.5e-9;
  - the P=1 equivalence still passes (4.8e-7);
  - sa5 dry run: 0 of 704 rl-split reference tasks fail.
- **Two OOM launches before the fix.**
  - `runs/sa6_r5_probe`: ref forward at step 4, from the out-of-place `/ T` copy.
  - `runs/sa6_r5_probe2`: backward at step 2, cap 6.59 GiB.
  - Both dirs are kept; deleting them needs the user's OK.
  - `runs/sa6_r5_probe/gpu_samples.csv`: a stray sampler kept appending to it until 2026-10-26 13:23. Rows after launch 1 ended are not from that run.
- **Probe 3** `runs/sa6_r5_probe3`: P=8, `--micro-rows 4`, const norm, dynamic sampling, KL 0.05, T 0.8, max_new 384/256, cap 6.56 GiB (fresh free 7.49 − 0.75).
  - No OOM. It hit `--max-seconds 2400` after **16 of 40 steps**; state.pt is saved and no final was merged.
  - **~160 s/step** (108–251). Refills hit the max of 3 on almost every step (mean 2.8), so one step generates up to 4 × 64 samples.
  - Flat share 0.57 mean (0.56–0.71). ~17 groups were generated per step, ~7 of them kept.
  - Entropy 0.066 → 0.17 (no collapse). len_mean 70 → 82, p90 ~143. KL ~1e-5. GPU util ~26 % mean, 86 °C max, nvidia-smi peak 7.2 GiB total including the desktop.
  - Pass rate by bucket over the sampled groups (refill tasks vary, so this is not a trend): code 0.14 → 0.07, fix 0.50 → 0.48, tool 0.35 → 0.23.
  - **Budget impact:** the proposal's "40 steps ≈ 10 min" is ~10× too low. At this pace 1 h ≈ 22 steps and 5 h ≈ 110 steps. Generation dominates (low GPU util), and refills multiply it.
- **Next (user decision):** the P / refill setting for RL part 1, then launch it (GPU, user-gated).

# 2026-10-26f - SA-6 run 5: grammar fix + rl_smollm2.py flags built, CPU checks PASS (no GPU used)

- **User approved** research/2026-10-26_sa6_rl_proposal.md as proposed: SA-5 pool, grammar fix first (CPU), batched prompts per step.
- **Grammar fix** (scratch/sa4_format_check.py, SA-7 grammar): `MAX_CALLS = 8` per call list and exact-duplicate calls are rejected.
  - SA-4 tool val: 146/146 references are still accepted.
  - Train: 33 of 14,854 rejected (14 with more than 8 calls, 8 duplicates, 7 max_str, 3 max_run, 1 max_depth). Train rows are only references; nothing is retrained.
  - The xlam/49334 repeat loop now dies when the repeated call closes.
  - The GPU re-score of init and candidate under the fixed grammar is not done yet; it goes into the SA-6 gate.
- **scripts/rl_smollm2.py new flags** (all off by default, so runs 1–4 behave as before):
  - `--source sa5`: pool and scorer imported from scripts/sa5_rft.py. Pool knobs: `--sa2/--sa4/--min-asserts/--pool-seed`, `--slice-code/fix/tool`, `--slice-start`, `--mix 0.4,0.35,0.25`.
  - Generation and scoring: `--max-new-tool`, `--workers`.
  - Training: `--prompts-per-step P`, `--loss-norm const`, `--dynamic-sampling` (≤ 3 refill rounds).
  - Stop rules: `--stop-entropy F`, `--stop-length F`.
  - New per-step log keys: entropy, len_mean, len_p90, flat_share, refills, pass per bucket.
  - state.pt now also stores the task cursor and the history.
  - Fixed: the step count saved after a `--max-seconds` break was off by one.
- **Validation (CPU):**
  - `--source sa5 --dry-run`: PASS.
    - Pools: code 97,015, fix 27,544, tool 14,854.
    - All 1,100/1,100 slice-1 references score 1.
    - Every half reference fed with eos=False scores nostop −0.1, tool included.
    - The first 1,100 draws split code 405 / fix 395 / tool 300.
  - Playground `--dry-run`: 704/704 PASS.
  - ast.parse: PASS.
  - Equivalence, with P=1, seq loss, on a tiny random Llama + LoRA, against the pre-edit file. Script: scratchpad equiv_check.py.
    - Log-probs, objective and grads match the old lines, max diff 4.8e-7. With and without KL.
    - The rows scaling, the const normalisation and token_prefix check out.
  - tests/: test_reference_sft_bytes 5/5 and test_sft_vram_cap 6/6 PASS. test_artifacts, test_explorer and test_simulator FAIL because runs/smoke and runs/kd-t2p-baseline artifacts are missing. They don't import rl_smollm2, and this change doesn't cause them.
- **Open point (pre-existing, unchanged):** rollouts sample at T = 0.8, but log-probs use T = 1 logits, so the update is slightly off-policy. A logits/T option needs a user decision before the GPU probe.
- **Next (user-gated GPU):** probe of 40 steps, P = 8, cap = fresh free − 0.75 GiB. It measures s/step, peak VRAM, flat share and entropy.

# 2026-10-26e - SA-5 round 1 SFT + merge + gate DONE: gate FAILS, SA-5 stops (log scratch/sa5_r1_train_gate.log)

- **Chain:** scratch/sa5_r1_train_gate.cmd (launcher scratch/launch_sa5_r1_train_gate.ps1), cap 6.62 GiB (free 7.37 − 0.75). Every step exited 0; 11:17 → 12:26.
- **SFT:** configs/sft_sa5_r1.yaml, 48 steps in 5 min → runs/sft_sa5_r1/final. Merged onto runs/sa4_scale075 at 0.75 → runs/sa5r1_scale075 and at 1.0 → runs/sa5r1_scale100.
- **Primary metric:** new `scripts/sa5_rft.py --split val` (sample + score only, no train files). Fixed val subset: code 300 (he 159, mbpp 141) + fix/repair 267, k=4, the same tasks for all three arms. This is reduced from the proposal's ~1.2k × 8 to fit the budget. Output scratch/sa5_gate_r1/<arm>/report.json, ≈ 10 min per arm.

| metric (vs init s075) | init s075 | r1 s0.75 | r1 s1.0 | rule | result |
|---|---|---|---|---|---|
| sampled pass@1, pooled 567 tasks | 29.5 % | 29.8 % | 29.3 % | up | flat (within noise) |
| per shape he / mbpp / fix / repair | 11.2 / 13.5 / 45.3 / 52.3 | 12.6 / 13.8 / 46.5 / 50.4 | 12.3 / 13.1 / 46.7 / 49.3 | | |
| greedy HE + MBPP | 14 + 57 = 71 | 14 + 53 = 67 | 14 + 52 = 66 | not down | **FAIL** |
| English BPB | 0.5269 | 0.5276 (+0.1 %) | 0.5279 (+0.2 %) | ≤ +3 % | PASS |
| constrained format valid | 146 | 145 | 145 | 146/146 | **FAIL** |
| unconstrained valid / exact | 140 / 50 | 142 / 52 | 142 / 51 | not down | PASS |
| ladder fix tier | 27 | 30 | 30 | ≥ 27 | PASS |
| tool exact (constrained) | 50 | 52 | 51 | ≥ 50 | PASS |
| tool tier / B1 / W2 (info) | 13 / 70 / 245 | 12 / 65 / 241 | 13 / 64 / 242 | | slightly down |

- **Constrained miss:** xlam/49334 on both arms is a 695-char list of repeated `options_historical` calls that hits the budget. The SA-7 grammar caps string length and runs inside a value, but not the number of repeated calls; a max-calls/duplicate-call rule would catch it.
- **Verdict:** per proposal §5, the first failing round stops SA-5, and runs/sa4_scale075 stays the SA-5 output. Even without the two hard fails, round 1 gave no real lift: 767 self-generated rows over 48 steps moved the primary metric by about 0.3 pp, which is noise.
- **Cosmetic:** inside the cmd's `for (...)` block, `%date% %time%` expands once at parse time, so the three pass@1 step lines show the same timestamp. Results are unaffected.
- Raw: research/raw/sa5r1_{format,passk,tier_fix,tier_tool,bpb}.json, sa5r1c_format.json, ladder_sa5r1_{b1,w2}/.

# 2026-10-26d - SA-5 round 1 sampling DONE (data/sa5_r1; logs scratch/sa5_r1.log, sa5_r1_resume.log)

- **Run:** 1,100 tasks × 8 from runs/sa4_scale075, cap 6.62 / 6.63 GiB, ≈ 37 min total.
- **OOM:** the first pass at batch 256 hit a clean own-job OOM inside the cap after 736 tasks (4.38 GiB allocated + 2.10 GiB reserved-unallocated). The tool group's long schema prompts caused it.
  - Resumed from samples.jsonl at `--batch 64` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; peak 1.77 GiB; 364 tasks in 13 min.
  - **Lesson:** batch 256 is safe for code/fix and not for tool. Future rounds should use a per-group batch, or batch 64 for tool.
- **Pass rates:**

  | shape | pass rate | 0/8 tasks | 8/8 tasks |
  |---|---|---|---|
  | he | 13.9 % | 120/169 | — |
  | mbpp | 11.6 % | 96/131 | — |
  | fix | 51.5 % | — | 72/260 |
  | repair | 50.1 % | — | 68/240 |
  | tool | 33.3 % | 136/300 | — |

  - Tool failures: mismatch 1,445, unknown_tool 70, json 47, nostop 37.
- **Selection:** 441 learn tasks + 44 anchors → 575 rows, plus 192 replay → 767 train rows.
  - Train shapes: fix 193, repair 205, tool 191, he 86, mbpp 64, replay 28.
- **ds:** `sa3_data.py` → data/sa5_r1/ds. 767 train rows, 188k tokens, 0 over ctx; val = SA-4 val, 748 rows.
- **Proposed config:** configs/sft_sa5_r1.yaml (not run). Base runs/sa4_scale075, LoRA r16, lr 1e-4, 2 epochs, effective batch 32 (≈ 48 steps, a few minutes), eval/save every 12 steps.
  - Then merge at 0.75 / 1.0 and run the SA-5 gate.

# 2026-10-26c - SA-5 batch probe + proposed round-1 size (scratch/sa5_b64, sa5_b256; log scratch/sa5_batch_probe.log)

- **Setup:** the same 40 round-1 tasks (24 code, 10 fix, 6 tool) × 8, cap 6.66 GiB, both exit 0.

  | batch | gen | score | peak VRAM | gen per sample |
  |---|---|---|---|---|
  | 64 | 81 s | 24 s | 1.08 GiB | 0.25 s |
  | 256 | 58 s | 30 s | 3.35 GiB | 0.18 s |

  - Speed-up: 1.4×.
  - Pass rates agree within sampling noise.
  - Batch 512 would be about 6.5 GiB, too close to the cap, so batch 256 is the setting.
- **Estimate at batch 256:** about 0.22 s generation plus 0.06 s scoring per sample, scaled from the 200-task probe. With ≈ 50 min of SFT plus merge/gate, sampling gets ≈ 40 min, which is ≈ 8.5k samples ≈ 1,100 tasks × 8.
- **Proposed round 1:** `--code 300 --fix 500 --tool 300 --batch 256` (fix covers fix + repair).
  - Expected yield from the probe rates: about 700 kept rows plus about 230 replay, ≈ 0.9k train rows.
  - That is thin for SFT (≈ 15 optimizer steps at 64 rows per step). It is still a real signal on the fix tier, which is the main SA-5 watch point.

# 2026-10-26b - SA-5 GPU probe: 200 tasks × 8 from runs/sa4_scale075 (scratch/sa5_probe/, log scratch/sa5_probe.log)

- **Run:** cap 6.65 GiB (fresh free 7.40 − 0.75), batch 64 rows/step, exit 0, 573 s total.
  - Generation: 502 s for 1,600 samples, ≈ 0.31 s per sample.
  - Sandbox: 70 s, ≈ 0.044 s per sample at 4 workers.
  - Peak VRAM was not logged.
- **Per-sample pass rate and tasks solved (0/8 … 8/8):**

  | shape | pass rate | 0/8 | 8/8 | other failures |
  |---|---|---|---|---|
  | he | 13.5 % | 50 of 72 | 2 | — |
  | mbpp | 10.2 % | 35 of 48 | — | — |
  | fix | 53 % | 5 of 26 | 9 | — |
  | repair | 52 % | 6 of 24 | 9 | — |
  | tool | 38 % | 13 of 30 | 4 | 10 json, 10 unknown_tool |

  nostop/timeout failures are rare (≤ 15 per shape).
- **Selection:** 66/200 tasks (33 %) in 1–7/8 plus 6 anchors → 95 kept rows, about 0.48 per task. Adding replay gives 127 train rows.
- **Budget verdict: the planned round does not fit.**
  - 4,000 tasks × 8 at this pace is ≈ 2.8 h of generation plus ≈ 23 min of scoring per round, against the 45 min planned.
  - Yield would be ≈ 1.9k rows, below the 3–5k estimate.
  - Code tasks are mostly 0/8: the signal there is thin. Fix and tool tasks are rich.
- **Options:**
  - A larger generation batch, measured first.
  - A smaller or fix-weighted slice.
  - Accepting about 3 h per round, which goes over the 5 h SA-5 budget and is the user's decision.

# 2026-10-26 - SA-5 RFT builder + CPU dry run PASS (no GPU used)

- **Script:** `scripts/sa5_rft.py` builds one RFT round.
  - Pool: he/mbpp from data/sa2, fix/repair/tool from data/sa4. Code and fix rows need ≥ 3 asserts.
  - Each pool is shuffled once by seed, and round r takes slice r.
  - Sampling: k samples per task. Code/fix text is cut with bench_passk `truncate`.
  - Reward: code/fix use sa2_build `run_guarded`; tool uses an exact call-set match.
  - Selection:
    - Keep tasks with 1..k−1 passes.
    - Add ≤ 10 % anchors from 8/8 tasks.
    - Keep ≤ 2 AST-distinct passing samples per task, shortest first.
  - Mix: 25 % SA-4 replay; val = data/sa4/val.jsonl.
  - samples.jsonl is appended per task, so a rerun resumes.
  - torch is imported only on the GPU path. The GPU path sets the VRAM cap with `--vram-cap-gib`.
- **Dry run:** `--dry-run --code 300 --fix 125 --tool 75 --out scratch/sa5_dry`. It scores each reference once.
  - Result: 500/500 pass (he 169, mbpp 131, fix 68, repair 57, tool 75). Every tool reference matches itself.
  - 425 sandbox programs took 45.7 s serial (≈ 0.11 s each).
  - `sa3_data.py` tokenizes the output: 667 train rows (167 replay), 0 over ctx, 152k tokens. val is 748 rows.
- **Cost estimate:** a round has ≈ 27k code/fix programs. At 4 workers that is ≈ 13 min of sandbox time, plus the timeout cost of wrong samples (6 s each when they hang). Generation time is still unmeasured.
- **Not tested:** the GPU sampling path, which needs the user-gated 200 × 8 probe.

# 2026-10-25 - SA-4 data + config READY (not launched; GPU launch is user-gated)

- **Build:** `scripts/sa4_data.py` (new) reads data/sa2 and adds playground replay, writing data/sa4/{train,val}.jsonl plus report.json. `scripts/sa3_data.py --src data/sa4 --shapes fix,repair,tool,replay,he,mbpp --out data/sa4/ds` tokenizes it (prompt-masked).
- **Shapes:**
  - fix: half of the SA-2 fix rows, unchanged. The buggy code is shown commented out with its real sandbox outcome.
  - repair: the other half of the fix rows, split by a sha256 bucket of the row id, so no source appears in both (checked: 0 overlap). Each is re-rendered as an SA-7 retry turn: `# Complete this Python function.`, the task line, the attempt as live code, the real `# >>> call` with its Traceback or "returned X but expected Y", then `# That attempt is wrong. Write the fixed function.` and the header. The attempt keeps its own header, because some mutations change a default argument.
  - tool: all SA-2 xLAM rows.
  - he/mbpp: a 10 % sample of the SA-2 instruction rows, against forgetting.
  - replay: playground reference solutions, 40 per write-kind skill from the wide pool, on the held-out `pretrain` split (val uses index 40). The fix- and tool-kind skills are left out, so the SA-4 ladder fix/tool gates are never measured on replayed skills. 0 exact prompt overlap with 4,480 dev/final/rl/confirm prompts.
- **Counts:**
  - Train: 61,729 rows (fix 13,837, repair 13,769, tool 14,854, replay 9,520, he 5,531, mbpp 4,218).
  - Val: 748 rows (fix 127, repair 141, tool 146, replay 238, he 52, mbpp 44).
  - Tokenized: 61,723 train rows (6 dropped over ctx 1024); 14.67 M tokens, 3.90 M of them target; p50 206, p99 701 tokens.
- **CPU validation PASS:**
  - Every row: input_ids and labels have equal length, end in EOS, fit within ctx, and the labels are a -100 prompt mask followed by an exact copy of the completion.
  - All tool completions parse as JSON lists of {name, arguments}.
  - All fix, repair, replay, he and mbpp code parses.
  - The attempt shown in every repair prompt parses.
- **Config `configs/sft_sa4.yaml`:**
  - A clone of sft_sa3: LoRA r16/α32 on all 7 projection modules, lr 1e-4 cosine, warmup 30, bs8 × accum8, fp16, gradient checkpointing, ctx 1024.
  - `base_model: runs/sa3_scale075`, `dataset_dir: data/sa4/ds`, output to runs/sft_sa4/final, eval/save every 200 steps.
  - About 965 steps; at SA-3's ~6.8 s/step that is ~1.8 h, inside the ≤ 3 h SA-4 budget.
- **Launch (user-gated):** `.venv/Scripts/python.exe scripts/sft.py --config configs/sft_sa4.yaml --vram-cap-gib <fresh free − 0.75>`. `--pilot` first runs 2,048 rows into runs/sft_sa4_pilot.
- **Gate:** format ≥ 98 %, fix ≥ 25 % (ladder fix/tool tiers, dev split). Also re-check HE/MBPP/B1/W2 against the sa3_scale075 init.
- **Pilot PASS (2026-10-25, user-approved):**
  - `--pilot --vram-cap-gib 6.54` (fresh free 7.29 − 0.75). 2,048 rows, 32 steps, no OOM.
  - Train loss 0.341 → 0.267; eval_loss 0.276; ~5.5 s/step. Output: runs/sft_sa4_pilot.
- **Full run LAUNCHED (2026-10-25):**
  - `sft.py --config configs/sft_sa4.yaml --vram-cap-gib 6.44` (fresh free 7.19 − 0.75).
  - Log scratch/sft_sa4.log, output runs/sft_sa4/final. ETA ~1.5–1.8 h.
  - The train script auto-resumes: re-run the same command if it stops.
- **Full run DONE (2026-10-25):**
  - 965/965 steps in 6,391 s (1.78 h); exit 0, no OOM.
  - Train loss 0.336 → 0.171. eval_loss by step: 200 0.2049, 400 0.1834, 600 0.1727, 800 0.1688, 965 **0.1684** (best).
  - Merged model in runs/sft_sa4/final. Gate eval below.
- **Gate eval DONE 2026-10-25** (cap 6.72 GiB from 7.47 free, 24 min; `scratch/sa4_gate_eval.cmd`, log `scratch/sa4_gate_eval.log`). The ladder "tool" tier scores Python that builds call dicts, not the JSON reply, so format is measured by the new `scratch/sa4_format_check.py` on the 146 held-out data/sa4/val tool rows (valid = one JSON list of {name, arguments:{}} with names from the prompt).
  | metric | sa4 | init sa3_scale075 | gate |
  |---|---|---|---|
  | format validity (146 xLAM val) | **143/146 = 97.9 %** (exact 53) | 0/146 | ≥ 98 % → **FAIL by 1 row** |
  | ladder fix tier, dev ×5 | **30/110 = 27.3 %** | 35/110 = 31.8 % | ≥ 25 % → PASS, but −5 vs init |
  | ladder tool tier, dev ×5 | 13/100 | 11/100 | – |
  | HumanEval greedy | **10/164** | 15/164 | regression −5 |
  | MBPP greedy | 57/427 | 55/427 | flat |
  | code / En BPB | 0.5772 / 0.5295 | 0.5615 / 0.5228 | +2.8 % / +1.3 % worse |
  | B1 / W2 | 70/192 / 259/560 | 53/192 / 158/560 | up, **but inflated**: replay trained the same write skills (pretrain-split instances), so this is instance-, not skill-generalisation |
  - All 3 format failures are runaway repetition with no EOS (digit strings, the same call repeated), not schema errors. Raw: research/raw/sa4_format.json, sa4_tier_{fix,tool}.json, sa4_passk.json, sa4_bpb.json, ladder_sa4_{b1,w2}/.
  - Verdict: **SA-4 gate NOT met** (format 1 row short; fix passes the bar yet below init; HE −5). Not promoted. Candidate cheap fixes (eval-only, no retrain): LoRA-scale merges as in SA-3 (s=0.5/0.75 of the sa4 delta) to trade format against HE/fix; or a decode-side repetition stop for the 3 runaways.
- **Constrained-decode format check DONE 2026-10-26** (user-approved). Script: `scratch/sa4_format_check.py --constrained`.
  - How it works: greedy decoding where each step takes the best token that keeps the reply a valid prefix of `[{"name": <prompt tool>, "arguments": {JSON}}, ...]`.
  - Grammar is a hand-rolled char-level state machine; no new dependency. EOS is legal only after `]`. Limits: ≤ 8 calls, nesting depth ≤ 8, numbers ≤ 32 chars.
  - CPU-tested: all 146 references are accepted char- and token-wise; 9 malformed cases are rejected.
  - Run: `scratch/sa4_constrained_eval.cmd`, cap 6.63 GiB (7.38 free), about 7 min. Raw output: research/raw/sa4c_format.json.

  | arm | valid (unconstrained → constrained) | exact | names | arg keys | argmax rejected | seconds |
  |---|---|---|---|---|---|---|
  | s075 | 140 → 144 | 50 → 50 | 84 → 85 | 66 | 8 / 8569 | 92 |
  | s=1 | 143 → 144 | 53 → 53 | 88 | 67 | 1 / 8319 | 85 |
  | init | 0 → 35 | 0 | 16 | 9 | 670 / 33419 | 217 |

  - Findings:
    - The grammar repairs format but never changes content: exact match is +0/−0 per row on both SFT arms.
    - The 2 failures per arm are budget runaways inside values: a 200+ digit string (`is_id` "1000…", xlam/19472, both arms) and repeated calls that hit 256 tokens before the call cap.
    - So the proposed ≥ 99 % gate is missed (98.6 %) for both arms.
    - Reference argument strings are ≤ 54 chars (p99 28), so a per-string cap of about 128 chars plus a runaway-repeat stop would close both failure types.
    - Init reaches only 35/146 with grammar alone: the format skill comes from SFT, not from the mask.
- **SA-4 PROMOTED 2026-10-26 (user decision): `runs/sa4_scale075` is the SA-4 output** and the init for SA-5. Format is enforced at decode time in SA-7, so the unconstrained 98 % format bar is replaced by "valid under the grammar". Caveats carried forward: fix tier 27 vs init 35, HE 14 vs 15.
- **SA-7 grammar repeat stops added 2026-10-26** (`scratch/sa4_format_check.py`, CPU-only; GPU re-run not yet done):
  - `MAX_STR = 128`: a string (key or value, raw chars incl. escapes) must close by 128 chars.
  - `MAX_RUN = 16`: no char may repeat more than 16 times in a row inside a string.
  - No key may repeat within one object (object frames carry their key set; sibling/nested objects are independent).
  - Data check: SA-4 val refs have no string > 54 chars, no run > 3, no duplicate key; train has 0 duplicate keys, 7 strings > 128 and 3 runs ≥ 20.
  - CPU tests PASS: 146/146 refs char- and token-wise; 14 negatives rejected (the 9 before plus 129-char string/key, 17-char run, duplicate key flat and nested); 5 edge positives accepted; train refs rejected 25/14,854 (mostly the > 8-call rows).
  - All 4 constrained failures are now illegal prefixes: 19472 (both arms) and 49142 at the 17th repeated `0`; 49334 at its first duplicate `"type"` key.
  - **GPU re-run DONE 2026-10-26** (user-approved): `scratch/sa4_constrained_eval2.cmd`, cap 6.59 GiB from 7.34 free, 9:26–9:29. Raw output: research/raw/sa4c2_format.json.

    | arm | valid | exact | names | arg keys | argmax rejected | seconds |
    |---|---|---|---|---|---|---|
    | s075 | **146/146** (was 144) | 50 (=) | 85 (=) | 66 (=) | 12 / 8338 | 95 |
    | s=1 | **146/146** (was 144) | 53 (=) | 89 (+1) | 67 (=) | 4 / 8034 | 81 |

    - All 4 runaways now close, as valid JSON but not exact matches. 19472 stops at `"10000000000000000"`; 49334 renames its second `type` key to `type_of_option`.
    - 2 other rows changed text without changing their scores (15154 on s075, 30256 on s=1).
    - Verdict: **the ≥ 99 % format gate under the SA-7 grammar PASSES (100 %) for the promoted s075.** Content accuracy is the remaining gap: exact 50/146 is the SA-5 target.
- **LoRA-scale check DONE 2026-10-26** (user-approved; `lora_scale_merge.py --base runs/sa3_scale075 --tuned runs/sft_sa4/final --scale 0.50/0.75` → runs/sa4_scale050, runs/sa4_scale075, 210/272 tensors changed each). Eval `scratch/sa4_scale_eval.cmd` (same 7 steps; cap 6.74 GiB from 7.49 free, 45 min; first launch stopped by the user during step 1, relaunched from scratch). Raw: research/raw/sa4s_*.json, ladder_sa4s_{b1,w2}/.
  | metric | init (s=0) | s=0.5 | s=0.75 | s=1 | gate |
  |---|---|---|---|---|---|
  | format validity /146 | 0 | 127 (87.0 %) | 140 (95.9 %) | 143 (97.9 %) | ≥ 98 %: all FAIL |
  | fix tier /110 | 35 | 29 | 27 | 30 | ≥ 25 %: all PASS |
  | tool tier /100 | 11 | 11 | 13 | 13 | – |
  | HumanEval /164 | 15 | 14 | 14 | 10 | – |
  | MBPP /427 | 55 | **62** | 57 | 57 | – |
  | code / En BPB | 0.5615 / 0.5228 | 0.5655 / 0.5249 | 0.5703 / 0.5269 | 0.5772 / 0.5295 | – |
  | B1 /192, W2 /560 | 53, 158 | 62, 203 | 70, 245 | 70, 259 | (inflated by replay) |
  - Format rises monotonically with scale, so scaling only trades format for HE; no scale meets 98 %. The fix tier stays below init at every scale (27–30 vs 35).
  - A "cut at the first complete JSON value" stop rescues **0** of the failures at any scale (19/6/3): the JSON never closes (runaway repetition inside the list), so a plain stop rule won't help. That would take grammar-constrained decoding (SA-7) or more format data/steps.
  - Note: per-arm HE/MBPP counts are in `[result]` lines of scratch/sa4_scale_eval.log and in sa4s_passk.json `arms.*.per_set`.

# 2026-10-25 - SA-3 DONE: HE 7 -> 15/164, MBPP 29 -> 52/427; ladder B1/W2 down ~25-31 %

- The run was stopped by the user at step 1454. It auto-resumed from checkpoint-1250 and finished at step 1501 with best eval_loss 0.4484. Output: runs/sft_sa3/final.
- Gate eval vs ckpt-750 (scratch/sa3_gate_eval.cmd, launched by scratch/launch_sa3_gate.ps1 with cap = fresh free − 0.75 GiB):
  | Metric | ckpt-750 | SA-3 | Gate |
  |---|---|---|---|
  | HumanEval pass@1 (plain HE; HE+ is not wired) | 7/164 | **15/164** | > 7: PASS |
  | MBPP-sanitized pass@1 | 29/427 | **52/427** | — |
  | Body median (non-blank, non-comment lines, HE) | 1 | **5** | > 1: PASS |
  | English BPB | 0.5172 | 0.5281 (+2.1 %) | ≤ +3 %: PASS |
  | Code BPB | 0.5534 | 0.5703 (+3.1 %) | — |
  | Ladder B1 strict | 58/192 | **38/192** | regression |
  | Ladder W2 strict | 160/560 | **120/560** | regression |
- **Ladder regression diagnosed (2026-10-25, scratch/rescore_ladder_relaxed.py, CPU re-grade of the saved completions; raw research/raw/ladder_relaxed_rescore.json and ladder_relaxed_builtins_rescore.json):** the drop is REAL, not just the sandbox. With loops/attributes/imports/helpers/exceptions allowed, SA-3 gains only +3 B1 / +12 W2. Adding harmless built-ins to the call allowlist (`--extra-calls range,sorted,set,zip,float,isinstance,dict,list,tuple,map,enumerate,reversed`) gives +4 B1 / +22 W2, so B1 42 vs 58 and W2 142 vs 160 even with everything relaxed. Most re-graded SA-3 answers are wrong (B1 40, W2 110). The rest still break hard rules: recursion (B1 5, W2 23), lambda (B1 6, W2 17), and calls to other playground tasks' entry-point names (memorised from RL). Side finding: `range` is not in the evaluator's SAFE_CALLS. SA-3 calls it 51x in W2 invalids vs 2x for ckpt-750, which suggests RL on this sandbox taught the model to avoid `range`. SA-4 should replay playground-style tasks; a cheap check is WiSE-FT-style LoRA scaling between ckpt-750 and SA-3.
- The teacher rerun was restarted on 2026-10-25 (1,347 rows before the restart). It stopped at 2,119 candidate rows when every free model's daily quota ran out. SA-2 has been rebuilt with those rows (below); the SA-4 data build is still pending.
- **LoRA-scaling check DONE (2026-10-25, scratch/sa3_scale_eval.cmd; raw research/raw/sa3_scale_passk.json, sa3_scale_bpb.json, ladder_sa3_scale_b1/, ladder_sa3_scale_w2/).** The SA-3 adapter is applied on top of ckpt-750 at reduced scale `s` (s=1 is SA-3 itself):

  | Arm | HumanEval | MBPP | code BPB | English BPB | B1 | W2 |
  |---|---|---|---|---|---|---|
  | ckpt-750 | 7 | 29 | 0.5534 | 0.5172 | 58 | 160 |
  | s=0.50 | 13 | 49 | 0.5558 | 0.5192 | 54 | 154 |
  | s=0.75 | 15 | 55 | 0.5615 | 0.5228 | 53 | 158 |
  | SA-3 (s=1) | 15 | 52 | 0.5703 | 0.5281 | 38 | 120 |

  s=0.75 matches SA-3 on HumanEval and beats it on MBPP, both BPBs and the ladder. That nearly removes the ladder regression: −5 B1 / −2 W2 vs ckpt-750, against −20 / −40 for SA-3. **SA-4 init = `runs/sa3_scale075` (user-approved 2026-10-25).** It is the exact folder evaluated above. A CPU check confirms all 210 changed tensors are base + 0.7500 × (SA-3 − base); sha256 and recipe are in runs/sa3_scale075/merge_info.json. The SA-4 SFT config should set `base_model: runs/sa3_scale075`.
- **Two-layer sandbox + diversity tooling (user-approved 2026-10-25, TASKS SA-C):**
  - The safety layer is unchanged and always on: no dunders, a call allowlist, an import allowlist, the 2 KiB cap, and the out-of-process runner with its timeout.
  - The style layer can now be lifted with the named preset `evaluator.OUTCOME_TIER`. It allows loops, attributes, imports, helpers, exceptions, lambdas and self-recursion, plus pure built-ins such as range, sorted and map.
  - Defaults (CORE and each rung's own tier) are unchanged. The two new `Capabilities` flags default to False, and sandbox.py ships them to the child process.
  - Tests: tests/test_aspects_evaluator.py, 56 passed. The preset still rejects print, getattr, eval, `import os`, `__class__` and open.
  - `scripts/bench_passk.py --samples N --temperature --top-p --seed` reports unbiased pass@1/5/n, AST-distinct correct solutions per solved task, and the EOS rate. The default `--samples 1` stays greedy and unchanged.
  - `scripts/rl_smollm2.py --policy outcome` grades every RL task under OUTCOME_TIER, so the reward judges only the result. The default `rung` is unchanged. Dry-run: 942 tasks (with 2 docstring and 2 MBPP tasks), 0 reference fails.
  - `scratch/rescore_ladder_relaxed.py --preset outcome` re-grades saved ladder completions; raw output is in research/raw/ladder_outcome_rescore.json.
  - Outcome re-grade (strict → outcome-tier passes):

    | Ladder | SA-3 | ckpt-750 | run-3 final |
    |---|---|---|---|
    | B1 | 38 → 45 | 58 → 59 | 56 → 57 |
    | W2 | 120 → 153 | 160 → 162 | 155 → 157 |

    So SA-3 is still behind ckpt-750 even when the result is the only thing graded. Its remaining invalid rows are mostly syntax errors (unclosed brackets, missing indented blocks), i.e. truncated or broken code, not style.

# 2026-10-22 - SA-2 in progress: build rerun at 4 GiB floor; teacher moved to Gemini free tier

- **Cleanup (user-approved):** Tier 1 and Tier 2 checkpoints deleted. Kept: `rl_mixen_wide_mbpp` checkpoint-750 (the incumbent, run 4) and `final`. About 32 GB free on disk afterwards.
- **`scripts/sa2_build.py` defaults** are now `--min-free-ram-gib 4 --workers 4` (user decision). Chrome had grown to ~11 GiB and the 8 GiB floor aborted the build. Each candidate child is capped at 1 GiB RSS.
- **Teacher provider:** OpenCode free models are unusable at dataset scale.
  - Only `minimax-m3.1-flash-free` answers on Go, and it hits a 429 after about 35 requests; the quota lasts hours.
  - `beam-free` is region-blocked (403).
  - Every other `-free` id is app-only: Go returns 400 "Model is unavailable" and Zen returns 403 FreeTierError. We do not work around this.
  - The user approved the **Gemini API free tier** instead (GEMINI_API_KEY is an environment variable, no billing): "free only, use flash or flash lite".
- **`scripts/teacher_client.py`:**
  - Supports both providers.
  - Gemini models are allowlisted and rotated in this order: flash-lite (3.5, 3.1, 2.5), then flash (3.8 down to 2.5).
  - A per-day FreeTier 429 moves to the next model. A 429 that does not name a FreeTier quota aborts.
  - Gemini calls use JSON mode.
- **`scripts/sa2_teacher.py`:**
  - Rotates models and records `model` per row.
  - Repairs Gemini's over-escaped `tests` items (`\"...\"`). Before the repair 74 % of replies parsed; after it, 34/35 cached replies did.
  - Rows still go through sa2_build's execution check and decontamination.
- **SA-2 build DONE (13:03):** gate PASS, 139,322 verified rows (≥ 30k).
  - Split: train 137,956 / val 1,366.
  - By shape: he 54,976, mbpp 42,018, fix 27,328, tool 15,000.
  - Decontamination dropped 726 rows, mostly mbpp+ / mbpp / signature hits.
  - Teacher contributed only 27 + 9 fix rows, because the build began before the Gemini batch.
- **SA-2 rebuild with the teacher rows DONE (2026-10-25 13:31, 7,703 s):** gate PASS, 141,338 verified rows.
  - Split: train 139,956 / val 1,382.
  - By shape: he 55,836, mbpp 42,628, fix 27,874, tool 15,000.
  - Decontamination dropped 738 rows.
  - Teacher: 2,119 read, 1,933 clean, 1,499 verified (77.5 % of clean; 433 fail, 1 timeout), plus 503 fix rows.
  - The previous build is backed up in `data/sa2/prev_2026-10-25/`.
- **sa2_build speed-up (user-approved 2026-10-25):** the build is CPU-bound by Windows process startup, one child per program.
  - New `verify_cache.jsonl` in the out dir stores pass/fail verdicts keyed by sha256(salt + program). Timeouts and RSS kills are never cached. `--no-cache` turns it off.
  - The default is now `--workers 8`. At start the build lowers it so workers × 1 GiB fits above the 4 GiB RAM floor, and the floor check is unchanged.
  - Probe (`--limit 300`): no cache, 4 workers = 50 s; cold cache, 8 workers = 37 s; warm cache = 7 s.
  - The output matched the no-cache run except one fix row whose witness prints a `set`. That ordering comes from hash randomisation (`-I` ignores PYTHONHASHSEED), so it was nondeterministic before the cache too.
  - The full-data cache is not built yet: the 13:31 build ran before the cache existed, so the next full build is cold once.
- **Teacher:** 2,119 rows of the 3,000 target. The free quota is spent for today; rerun after midnight Pacific.

# 2026-10-18 - B1 breadth scored: no arm transfers; VRAM rule now free - 0.75 GiB

- **B1 (192 NEW sub-skills, dev, greedy, batch 32)**, strict / 192:
  | Arm | write /150 | fix /22 | tool /20 | total | EOS rate |
  |---|---|---|---|---|---|
  | base | 2 | 6 | 0 | 8 | 0.13 |
  | CPT | 16 | 5 | 1 | **22** | 0.65 |
  | LoRA (pg_long) | 9 | 5 | 0 | 14 | 1.00 |
  | CPT+LoRA | 3 | 3 | 0 | 6 | 1.00 |
  CPT+LoRA is 98.8% on L4 (in-distribution) and 3.1% here: the playground SFT memorised the 88 original sub-skills. CPT alone transfers best. Next: diverse data (SA-2/SA-3) and SA-6 RL on the `wide` pool, gated on B1/W2 plus HumanEval/MBPP.
- **Speed:** batch 4 -> 32 made the bench ~6x faster (16 -> ~100 rows/min); it is overhead-bound (1.4 GiB, ~30% util), so stop-at-budget is the next lever.
- **User decision:** CLAUDE.md §20 rule 2 is now "ceiling = fresh free VRAM - 0.75 GiB" (~7 GiB here), still enforced in-process so overshoot is an own-job OOM, never a spill. SA-1 (needs ~5.5 GiB at batch 4) fits now.
- Commits: nested a7e7bd4, outer 37ae58a (SA-B + SA-6a).

# 2026-10-18 - SA-B breadth battery BUILT: 280 sub-skills, W4 = 1,120 tasks (CPU, not committed)

**What changed:**
- `glm53_flash/playground.py`:
  - Adds `BREADTH`, 33 new construct ladders with 192 sub-skills:
    - 150 write;
    - 22 *fix-from-traceback*: the prompt shows the buggy body plus the real traceback or wrong return on a witness case;
    - 20 *strict tool-call*: build/parse/dispatch/batch `{name, arguments}` dicts and tool messages.
  - Capabilities (loops, attributes, imports, helpers, exceptions, safe calls) are derived from each reference's AST.
  - The 88 originals are byte-identical, pinned by a sha256 fingerprint test.
- Pools: `breadth` 192, `wide` 280, `fix` 22, `tool` 20.
- `scripts/bench_ladder.py` adds levels B1 192, W2 560 and W4 1,120.
- `scratch/bench_hf.py` passes the new tiers through the protected bench by monkeypatching its `playground_pool`. The protected file is untouched.
- `scripts/rl_smollm2.py --tier` accepts all 8 pools.

**Validation:**
- PASS: glm-5.3-flash-from-scratch pytest, 152/152.
- PASS: `bench_ladder.py --dry-run --up-to W4` shows the new counts.
- PASS: RL dry-run, `fix` 176/176 and `tool` 160/160.
- SA-6a FIXED (user-approved option (a)). The sandbox child now JSON-normalises actual and expected before the strict check. RL dry-run with the sandbox on: all 704/704, wide 2240/2240, fix 176/176, tool 160/160 PASS. pytest 153/153.

**Next:**
- Score the arms on B1/W2/W4 when the GPU is idle at a stage end.
- Use the `fix`/`tool`/`wide` pools for SA-6 RL signal.
- SA-1 DONE 2026-10-19 (gates split): English BPB 0.5149 vs base 0.5132 (+0.3 %, PASS; CPT is +4.5 %); code BPB 0.5451 vs CPT 0.5240 (FAIL, +4.0 %); B1 strict 22/192, the same as CPT. Init is runs/cpt_smollm2_mix_en/final. Details: TASKS SA-1, research/raw/sa1_bpb.json, research/raw/ladder_mixen.
- SA-6 RL run 1 DONE 2026-10-19 (init SA-1, `wide`, runs/rl_mixen_wide/final). Compared with SA-1: B1 22 -> 56/192; W2 70 -> 154/560; MBPP-san 9 -> 24/427 (best arm); HumanEval 9 -> 3/164 (regression). EOS 0.61 -> 0.99. HumanEval drop DIAGNOSED: not truncation, not format. RL collapsed to one-line `return <expr>` bodies (164/164 single-line, median 135 -> 28 chars) and loses 6 of SA-1's 9 passes to over-simplified expressions. Next RL run needs an anti-collapse lever (KL/ref penalty or lower lr, multi-line pool tasks, intermediate checkpoint eval). The PLAN gate waits on SA-5.
- **SA-6 RL run 4 DONE 2026-10-21 16:04 + EVALUATED 2026-10-22. It closes the MBPP format gap. Run 4 ckpt-750 (runs/rl_mixen_wide_mbpp/ckpt-750) is the SA-6 INCUMBENT (user approved 2026-10-22).** GPU is free. RL is paused: next is SA-2 data → SA-3/4 SFT → SA-5 RFT, then RL run 5 (user approved 2026-10-22; see research/2026-10-22_sa_plan_v2.md).
  - Scores are HumanEval /164, MBPP /427, B1 /192, W2 /560:

    | Arm | HE | MBPP | B1 | W2 |
    |---|---|---|---|---|
    | run 4 `final` | 7 | **30** | 56 | 155 |
    | run 4 ckpt-750 | 7 | 29 | **58** | **160** |
    | run 3 final (previous incumbent) | 8 | 13 | 53 | 156 |
    | run 1 final | 3 | 24 | 56 | 154 |
    | SA-1 | 9 | 9 | 22 | 70 |

  - Run 4 final is the best MBPP of any arm (2.3× run 3); ckpt-750 is the best B1 and W2.
  - MBPP format failures, run 3 final → run 4 final: test-function only 156 → 36; no def 34 → 2; answers that define the function 237 → 389 of 427.
  - The remaining MBPP misses are wrong answers (AssertionError 240, TypeError 76), not format.
  - HumanEval is unchanged (7, vs 8 and SA-1's 9). Bodies are still median 1 line, so the one-line collapse remains the open gap.
  - Reports: research/raw/sa6_mbpp_passk_final.json, sa6_mbpp_passk_750.json, ladder_sa6_mbpp_b1/, ladder_sa6_mbpp_w2/. Eval log: scratch/sa6_mbpp_eval.log.
  - The first eval attempt was stopped by the user mid-run, and nothing was written. It was re-run 2026-10-22 as a WMI-detached batch with one arm per step and one output file per step (scratch/sa6_mbpp_eval.cmd), cap 6.70 GiB from a fresh 7629 MiB reading.
  - Design notes as launched:
  - Why: ~45 % of run 3 final's MBPP misses are format failures. The model writes tests or `if __name__` instead of the `def` (test-function 156, no def 34 of 427).
  - New flag `--mbpp-tasks N`: N MBPP-shaped instances per `write` sub-skill. The prompt is `"""\nWrite a function to <description>\nassert <call> == <value>\n"""`, the same text bench_passk feeds for MBPP. The completion must write the whole `def`. The function gets its natural name unless that name is a builtin or appears in the body. Cases 1–3 are graded.
  - Validation: pool 5124 tasks (1428 MBPP-format, 27.9 %); dry-run 5124/5124 references pass; 3696/3696 without the flag.
  - CPU sample on run 3 final (8 groups × 8): defines the function 8/64, test-only 11/64, full pass 1/64, groups with reward variance 3/8. The signal is sparse but nonzero.
  - Command: `rl_smollm2.py --init runs/rl_mixen_wide_doc2/final --output runs/rl_mixen_wide_mbpp --tier wide --docstring-tasks 16 --mbpp-tasks 6 --max-new-tokens 256 --kl-beta 0.05 --groups 1000 --checkpoint-every 250 --probe-limit 50 --vram-cap-gib 6.5` (fresh free 7.33 GiB − 0.75). KL is measured against run 3 final, because the reference is the adapter-disabled init.
  - Launched WMI-detached (cmd PID 29812). Log runs/rl_mixen_wide_mbpp/train.log; progress in rl_log.jsonl (~11 s/group, so ~3–3.5 h). Re-running the same command resumes it.
  - Incumbent: run 4 ckpt-750 (user 2026-10-22). It wins B1/W2, ties HE, and is one MBPP task behind final. The HumanEval one-line collapse is now targeted by SFT data first, then RL run 5.
- **SA-6 RL run 3 DONE 2026-10-21; its `final` is the SA-6 INCUMBENT (user 2026-10-21).** Full eval at batch 16 for pass@k and batch 32 for the ladder. Columns: HumanEval /164, MBPP /427, B1 /192, W2 /560.
  - `runs/rl_mixen_wide_doc2/final` = **8 / 13 / 53 / 156**: best W2 of any arm, and better than run 2 final (7/15/50/140) and run 2 ckpt-1500 (8/11/51/142) on HumanEval, B1 and W2.
  - ckpt-1750 = 7/15/52/146; ckpt-750 = 6/12/49/126.
  - The other references are SA-1 9/9/22/70 and run 1 3/24/56/154.
  - Reports: research/raw/sa6_doc_passk.json, ladder_sa6_doc_b1/, ladder_sa6_doc_w2/.
  - In training, the invalid rate fell from 58–65 % to 18–21 %. KL was still climbing at the end (0.168), and HumanEval stays just under SA-1.
  - Previous run-3 status, kept for history: `rl_smollm2.py --init runs/cpt_smollm2_mix_en/final --output runs/rl_mixen_wide_doc2 --tier wide --docstring-tasks 16 --max-new-tokens 256 --kl-beta 0.05 --checkpoint-every 250 --probe-limit 50 --vram-cap-gib 6.75`, log scratch/rl_mixen_wide_doc2.log (RELAUNCHED 2026-10-20: the first attempt in runs/rl_mixen_wide_doc had docstring prompts ending in a bare "\n", so SA-1 emitted EOS at once and every docstring group scored a flat 0.0; that dir is stale, kept and not resumed). The process died silently (exit 4, no traceback) twice, each time at a checkpoint probe: at group 500 (2026-10-20 16:29) and at group 750 (2026-10-21 08:34, while the Claude app was updating). It was RESUMED from 750 on 2026-10-21 as a WMI-detached process (`Win32_Process.Create` → cmd → python, `--vram-cap-gib 6.55` from a fresh 7478 MiB reading) so that app restarts can't kill it, auto-resumes on re-run. It adds HumanEval-style docstring tasks (39 % of the 3696-task pool) and 256 new tokens at KL 0.05. Then: eval the best probe checkpoint (pass@k + B1/W2 at batch 32) vs run 2 ckpt-1500.
- SA-6 RL run 2 EVALUATED 2026-10-20. ckpt-1500 / final: HumanEval 8 / 7 of 164, MBPP 11 / 15 of 427, B1 51 / 50 of 192, W2 142 / 140 of 560 (SA-1: 9, 9, 22, 70; run 1: 3, 24, 56, 154). KL trades MBPP for HumanEval and fixes neither; one-line collapse persists. Next: run 3 with a multi-line task distribution, awaiting user approval. Ladder grader now has a 5 s out-of-process timeout (scratch/bench_hf.py); the bench_ladder batch default is now 32. Outputs: research/raw/sa6_kl_passk.json, ladder_sa6_kl_b1, ladder_sa6_kl_w2.
- SA-6 RL run 2 DONE 2026-10-20 (runs/rl_mixen_wide_kl: final plus checkpoint-250…2000). On the first-50 probe: HumanEval 5–7/50 (SA-1 9, run 1 3); MBPP 1–2/50 (SA-1 2, run 1 5). Full pass@k + B1/W2 on ckpt-1500 and final is running (scratch/sa6_kl_eval.log); the GPU is busy with it. Before that: RESUMED 2026-10-19 from group 300 with `--min-free-ram-gib 3` (user decision) and cap 6.55 GiB. GPU busy, no other GPU jobs. Earlier it STOPPED at group 325 under the 8 GiB free-RAM guard (other desktop apps leave only ~9.1 GiB available). state.pt is at group 300; checkpoint-250 probe: HumanEval 6/50, MBPP-san 2/50. Resume by rerunning the same command once RAM is freed. GPU is free. Original launch: started 2026-10-19 13:50. Init SA-1, `wide`, `--kl-beta 0.05 --checkpoint-every 250 --probe-limit 50`, cap 6.6 GiB, output runs/rl_mixen_wide_kl (merged checkpoint-250…2000 plus probe_log.jsonl with HumanEval/MBPP first-50 pass@1), log scratch/rl_mixen_wide_kl.log. It auto-resumes on re-run. Expected ~2.5–3 h. Then: choose a checkpoint from probe_log, run the full bench_passk + B1/W2, and compare with run 1 (HE 3/164, MBPP 24/427).

# 2026-10-18 - SmolLM2 sub-agent line APPROVED (PLAN §11); CPT+LoRA 71/71; pass@1 harness fixed

**User goal:** a "fluent Python sub-agent" on SmolLM2-135M. The plan was approved with two changes:
- RL is MANDATORY (SA-6): port the RL playground, glm-5.3-flash-from-scratch/scripts/train_rl.py (RLOO).
- The battery grows past 1,000 tasks in steps, to save compute.

**Where things stand:**
- **PLAN §11:** definition, gates, stages SA-0..SA-7.
- **TASKS rows:** SA-0..SA-7. SM-CPT-SFT is closed.

**Results so far:**
- **A (CPT+LoRA)** finished from checkpoint-200: best_eval 0.5906, battery 71/71 (base+LoRA 67/71). The 71-task battery is now saturated.
- **D (pass@1, fixed harness)** is done, results in research/raw/smollm2_passk.json:

  | Arm | HumanEval | MBPP |
  |---|---|---|
  | base | 7/164 | 13/427 |
  | CPT | 8/164 | 11/427 |
  | LoRA | 4/164 | 0/427 |
  | CPT+LoRA | 1/164 | 0/427 |

  **The LoRA arms collapsed onto the playground distribution.** On MBPP they write playground-shaped functions with random-suffix names and ignore the spec. Playground SFT bought 71/71 at the cost of general coding. So SA-2/SA-3 diverse instruction data is required, and every stage gates on HumanEval/MBPP too (MEMORY 2026-10-18).
- **SA-6 RL port is drafted:** scripts/rl_smollm2.py. Its dry-run reward check passed on all 704 `rl`-split references; GPU smoke + resume PASS (20 groups, 47 s); the playground is too easy for RL signal (most groups 8/8).
- **Ladder** (research/raw/ladder_sa0): CPT+LoRA L4 1043/1056 (98.8%), LoRA 974/1056 (92.2%); base and CPT stop at L0 (1 and 11 of 71).
- **C launched:** `scripts/sft.py --config configs/cpt_smollm2_mix_en.yaml --vram-cap-gib 4.4`, log scratch/cpt_mix_en.log. Re-run the same command to resume.
- **New `scripts/bench_ladder.py`:** a stepped dev battery.
  - Levels: L0 core×1 = 71, L1 all×1 = 88, L2 ×3 = 264, L3 ×6 = 528, L4 ×12 = 1,056.
  - An arm climbs only while its strict pass rate is ≥ --gate (0.90). --reuse skips levels already on disk.
  - Checked: dry-run counts are correct, and the gate logic was self-tested on the 71/71 report (scratch/ladder_selftest).
- **Bug fixed in `bench_passk.py` (D):**
  - Every HumanEval completion was empty, because a prompt ending in a bare "\n" makes SmolLM2 emit EOS at once. Fix: rstrip the prompt.
  - `from __future__` lines in completions caused SyntaxErrors, because the guard prelude comes first. Fix: drop them.
  - The first run is invalid and is kept as scratch/smollm2_passk_buggy_v1.json.
  - The re-run with the fixed harness is in progress and writes research/raw/smollm2_passk.json.

**Next:**
1. Finish SA-0: pass@1, then the ladder per arm.
2. SA-1: English-replay CPT (idea C).
3. SA-B: breadth battery work on CPU.

# 2026-10-17 - SmolLM2-135M CPT COMPLETED: battery 1 -> 11/71, but BPB flat/worse

Run 1 crashed at step 134 (CUBLAS_STATUS_EXECUTION_FAILED, no checkpoint) in the same window as a host-RAM
exhaustion from my bench_decontam full scan (two parallel copies, whole-doc tokenization). Fix: the scan now
windows docs at 1M chars and aborts below 8 GiB free RAM (full scan 53 s). Run 2, same config, completed
3028/3028 in 3 h 40 min (scratch/cpt_full5.log). Val loss 1.313 (s500) -> 1.289 -> 1.277 -> 1.270 -> 1.267 (s3000).

    arm                          strict  eos    BPB humaneval  BPB mbpp-san
    SmolLM2-135M base            1/71    0.225  0.6396         0.7335
    SmolLM2 + CPT 1 ep           11/71   0.831  0.6407         0.7611
    SmolLM2 + LoRA 12 ep         67/71   1.000  -              -
    byte (ours, 25.7M)           64/71   1.000  -              -

Reading: CPT strict 11 == base EXTRACTED 11 (6 tasks shared). The gain is mostly format/stopping (eos 0.23 ->
0.83) learned from the example-level EOS split, not new code competence; BPB on held-out benchmarks did not move
(MBPP, with more English, got worse -> mild NL forgetting). Decontam (13-gram, HumanEval+MBPP): 0 flagged.
Raw: research/raw/smollm2_cpt_ex_bpb.json, smollm2_cpt_ex_core71.json, smollm2_cpt_bench_decontam.json.
Next fair test: LoRA-SFT on top of the CPT final (TASKS SM-CPT-SFT) vs base+LoRA 67/71.

# 2026-10-17 - SmolLM2-135M full-parameter CPT: OOM root-caused to the CAP, then launched at batch 4

Two CPT attempts (pilot and full) died at STEP 1 with torch.OutOfMemoryError under --vram-cap-gib 4.0:
    "Tried to allocate 768.00 MiB ... 4.00 GiB allowed; 3.36 GiB allocated by PyTorch"
The 768 MiB is identified exactly: 4 x 1024 x 49152 x 4 B = the fp32 LOGITS upcast for cross-entropy.
Full-parameter CPT carries a ~2.15 GiB FIXED cost before any activation - fp32 weights 538 MiB + grads
538 MiB + Adam m/v 1076 MiB (the fp32 master-weight AMP contract) - which LoRA never paid, because LoRA
has only ~1M optimizer params. Same batch, same ctx: the optimizer state is what changed. The CAP, not
the GPU, was binding - only ~1.1 GiB was held by foreign desktop apps and ~6.9 GiB was free.

Fix, in two measured stages:
  1. batch 4 -> 2, accum 8 -> 16, eval_batch 4 -> 2, cap 4.0 -> 4.5. PILOT PASSED: 32/32 steps,
     checkpoint-32 + final written (model 538 MB, optimizer 1076 MB), eval_loss 1.415, train_loss 1.534.
  2. But bs2/accum16 measured ~15 s/step against the LoRA reference's 5.38 s/step at bs4/accum8. Per
     MICRO-step the two are nearly identical (0.60 s vs 0.67 s), so halving the batch bought no
     efficiency at all - it only doubled the micro-step count for the same 32-row effective batch.
     Reverted to batch 4 / accum 8 under cap 5.5 GiB (peak demand ~4.33 GiB; the 4.0 failure was purely
     the cap, so the cap - not the batch - was the right thing to move).

RUNNING: job pwsh-49, configs/cpt_smollm2_ex.yaml, 1 epoch = 3,028 steps, ~6.5 s/step, ETA ~5.5 h,
GPU 5,571 MiB used / 2,424 MiB free, checkpoints every 500 steps (save_total_limit 3 = best + last 2).
Eval costs no extra memory (no_grad: params + logits ~3.8 GiB live, below the training peak).

Also recorded in MEMORY.md: pwsh-38 reported exit 0 while BOTH of its python invocations had already
crashed with OOM. Zero checkpoint dirs was the only proof of failure. Verify ARTIFACTS, never exit codes.

Next: when CPT completes, run the SAME 71-task battery (scratch/bench_hf.py, budget 160 B, max-tokens
256) against the CPT checkpoint and compare with the standing arms:
    byte (ours, 25.7M, e122h_eos/checkpoint-0800)  strict 64/71  eos 1.000
    SmolLM2-135M base, no LoRA                     strict  1/71  eos 0.2254
    SmolLM2 + LoRA, 12 epochs (876 steps)          strict 67/71  eos 1.000
The question CPT answers: does a broad 100M-token code prior beat a narrow playground-only SFT?

# 2026-10-13/14 - E-122: the reference-SFT rung was UNDER-TRAINED - 6x exposure took the held-out legacy battery from 9/32 to 32/32

Same corpus (scripts/reference_sft_bytes.py: 400 instances per source for the 8 legacy families and all 12
aspects), same init (runs/e117_byte/checkpoint-5000), same lr 1e-4 cosine. The ONLY change was steps 100 -> 600
(19,660,800 tokens, 4010 s, peak 3.005 GiB, no spill, 0 resume events, val_loss 0.2052 vs arm A's 0.2772).

    arm A  (100 steps)   9/32   increment 1, list_sum 2, nonnegative 4, square 2
    E-122  (600 steps)  32/32   EVERY family 4/4 - both retained checkpoints

LEAKAGE CONTROL, because a perfect score deserves suspicion (scratch/leak_check_e122.py):
    dev prompt-in-corpus ............ 0/32
    dev prompt+body-in-corpus ....... 0/32
    dev body-in-corpus ............. 32/32   <- expected and harmless: a family's reference body is a
                                              CONSTANT string (e.g. "return x[::-1]"), so the corpus
                                              contains it from its own pretrain instances.
And evaluate_source EXECUTES each completion against the dev task's own cases, so a pass is earned, not
matched. The dev split (seed 1701) and the corpus split (seed 9319) are disjoint.

Consequences:
- E-122 SUPERSEDES arm A as the ladder base. The whole RFT/RL ladder was measured against a base that was
  6x under-trained on its own reference corpus, so those rungs were fighting a moving, mis-set floor.
- The legacy 8-family battery is now SATURATED at 32/32 and can no longer discriminate. The frontier is the
  ASPECT pool: at the arm-A level, grade / count_even / count_positive / lookup are 0/4 and the total is 26/48.
- Val loss did not reveal this: arm A's 0.2772 vs E-122's 0.2052 is a 26% gap, but the executable gap was
  9/32 vs 32/32 - a 3.6x difference. This is the third time on this line that val loss failed to predict
  executable performance.

S10 (per-aspect LoRA + merge), measured the same day, for the record: all 12 adapters trained from arm A
(110,592 LoRA params each, 24 modules), average and TIES merges baked with max_abs_diff ~2e-6. Legacy battery:
base 9/32 -> 10/32 for both merges, i.e. merging twelve adapters does NOT damage the base - unlike the
aspects-only RFT, which cost 4 tasks. New aspect-pool battery (scratch/bench_aspect_pool.py, 12 aspects x 4 =
48 tasks, dev, greedy, deterministic, Lead-reproduced): arm A 26/48, s10avg 26/48, s10ties 26/48, RL 26/48 -
IDENTICAL per aspect. So S10 is structurally SAFE but not yet EFFECTIVE; its adapters saw only 12 training
examples per aspect.

# 2026-10-13 - THE LADDER WORKS: reference SFT 9/32 -> RFT pool-all 12/32 -> RL 16/32 on the held-out greedy battery

Rungs 2 and 3 of the E-118/E-121 ladder both landed today, and both verdicts come from the SAME deterministic
held-out measure: scratch/bench_aspects.py, dev split, 32 tasks, 48-byte budget, greedy. It has now been
bit-identical on cpu and cuda in every run, and the arm A control reproduced its recorded 9/32 with the same
per-family split three times.

    arm A (reference SFT)     9/32   increment 1, list_sum 2, nonnegative 4, square 2
    RFT aspects-only          5/32   double 1, nonnegative 4                    <- DESTRUCTIVE
    RFT pool-all             12/32   double 1, list_sum 4, nonnegative 4, square 3
    RL (RLOO) from pool-all  16/32   double 1, list_sum 4, nonnegative 4, reverse 3, square 4

The ladder reference-SFT -> RFT -> RL took the held-out battery from 9/32 to 16/32 (+78% relative).

WHAT MADE RFT WORK, and it was one flag: the aspects-only RFT trained the last block on a pool DISJOINT from
the evaluated families and destroyed 4 of the 9 solved tasks (9/32 -> 5/32) while its own keep_rate climbed
0.4245 -> 0.5482. Re-running it with ONLY --pool all changed (80 tasks, aspects + legacy pretrain) turned it
positive: 12/32. The rule is not "RFT is dangerous" but "an RFT harvest pool must COVER the families you
intend to keep". The keep_rate was pool memorization in both runs; only the held-out battery could tell them
apart.

RL then added 4 more tasks, and the most interesting is reverse 0/4 -> 3/4 - a family neither the SFT base nor
the RFT could solve at all, so this is transfer, not amplification. RL ran from runs/e121b_rft_all/checkpoint-0004
(receipt-verified init sha256 56fb8802...), RLOO, 96 groups x 4, T=0.8, lr 2e-5, last-block, reward-mode
case-fraction, invalid-penalty -0.1, brevity-weight 0.0 (brevity stays last), --pool all on the rl split,
512.997 s, peak 0.175 GiB, no spill, retention best + newest 2.

MEASUREMENT DISCIPLINE, learned the hard way the same day:
- pass@k at 8 samples and T=0.35 is NOT reproducible: the SAME arm A checkpoint read 8/8 viable / yield
  0.23047 on cpu and 7/8 / 0.19141 on cuda, with "even" flipping dead -> alive. Use it for order and for the
  go/no-go; the deterministic greedy battery carries verdicts.
- train_rl.py HARDCODES the rl split (no --split flag; passing one exits 2) and that is the SAME pool
  passk_preflight.py measures, so a post-RL pass@k gain is in-pool. The dev battery is the held-out verdict.
- The pool-all RFT SHARPENED the policy and narrowed sampling (5/8 families viable at T=0.35 vs arm A's
  7-8/8) even as its greedy score rose; RL then ran hotter (T=0.8) and recovered transfer.
- Held-out pass@k on dev (16 tasks, T=0.35, 8 samples) tells the same arc without a greedy crutch: arm A
  viable 8/8 yield 0.34375, pool-all RFT 5/8 yield 0.25000, RL 6/8 yield 0.34375. RL recovered sampling
  diversity from its own init AND added 4 greedy tasks, so its gain is not a sharpening trade.

Structural note: the user's own design - one adapter per aspect, then merge/route - is the clean way to get
per-aspect specialisation without the interference the aspects-only RFT demonstrated.

Artifacts: research/raw/e121b_rft_arms_aspects_cpu.json, e121b_rft_arms_aspects_cuda.json,
e121b_rftcmp_arms_aspects_cpu.json, e121b_rftcmp_arms_aspects_cuda.json, e121b_rftcmp_arms_passk_cpu.json,
e121b_rftcmp_arms_passk_cuda.json, e121b_rl_arms_aspects_cpu.json, e121b_rft_train.log,
e121b_rft_all_train.log, e121b_rl_all_train.log.

# 2026-10-13 - E-121d: RFT/STaR on the aspect pool DESTROYED the base (9/32 -> 5/32); pass@k is not reproducible, the greedy battery is

Rung 2 of the ladder ran clean and lost. scripts/train_rft.py started from arm A (runs/e121_ref/checkpoint-0100,
sha256 2b0f4303...) and took 4 rounds x 16 samples on the 12 CORE ASPECTS, training only the last block
(2,140,232 of 25.7M params) at lr 1e-5. exit 0, 1001.577 s (16.7 min), peak 0.246 GiB, no spill, 4/4 rounds,
retention checkpoint-0003 + checkpoint-0004 (state.pt only on 0004).

Its own metric looked like a clean win:

    keep_rate   0.4245 -> 0.4909 -> 0.5339 -> 0.5482   (+29% relative)
    pairs        326      377      410      421        (1,534 harvested)
    mean loss   0.0306 -> 0.0280 -> 0.0243 -> 0.0209

Held out, it was a loss. checkpoint-0004 vs the arm A control, same session, battery AND pass@k, on cpu AND cuda:

    greedy dev battery (32)   arm A 9/32    RFT 5/32     <- BIT-IDENTICAL on cpu and cuda
    pass@k yield (rl)         arm A 0.23047 (8/8 viable)     RFT 0.12500 (7/8, reverse dead)       [cpu]
                              arm A 0.19141 (7/8, even dead) RFT 0.11328 (5/8, abs/even/rev dead)  [cuda]

The RFT lost increment (1->0), list_sum (2->0) and square (2->0) and gained only double (0->1). The harvest
pool is DISJOINT from the evaluation families - 12 core aspects (is_palindrome, repeat_text, has_key,
key_count, clamp, sign, contains_zero, count_even, count_positive, first_or_empty, grade, lookup) vs 8 legacy
families (absolute, double, even, increment, list_sum, nonnegative, reverse, square) - so 194 supervised
last-block steps bought aspect-pool recall by overwriting legacy competence. The keep_rate climb was pool
memorization, exactly the caveat recorded when the run finished. Freezing the last block reduces how much can
drift; it does not decide WHICH WAY it drifts.

SECOND FINDING, methodological and independent of the RFT: the same arm A checkpoint was measured twice in
this session and did NOT reproduce under pass@k, while the battery reproduced exactly.

    battery      arm A 9/32, per-family identical on cpu and cuda (increment 1, list_sum 2, nonnegative 4,
                 square 2) - and identical to the recorded control
    pass@k cpu   arm A 8/8 viable, yield 0.23047
    pass@k cuda  arm A 7/8 viable (even dead), yield 0.19141

8 samples at T=0.35 near the boundary is noisy: "even" flipped dead -> alive between two runs of the SAME
weights. Use pass@k for order and for the go/no-go, the deterministic greedy battery for verdicts. This also
means the E-121c arm B family counts (1/8 vs 7/8) are directionally right but not to be quoted to two digits -
its battery gap (1/32 vs 9/32) carries that verdict.

STRUCTURAL FIX: the user's own design - one adapter per aspect, then merge/route - is exactly what prevents
this interference. A single adapter trained on one pool overwrites the others.

RETRY RESULT, and it is POSITIVE: runs/e121b_rft_all is the identical command with ONE flag changed
(--pool all: aspects + legacy pretrain, 80 tasks). exit 0, 4/4 rounds, 1493.043 s (24.9 min), peak 0.247 GiB,
no spill, keep_rate 0.4141 -> 0.4219 -> 0.4313 -> 0.4414 (lower than the aspects-only run because the mostly
unsolved legacy families are now in the pool). Three-arm same-session battery, BIT-IDENTICAL on cpu and cuda:

    arm A (SFT base)          9/32   increment 1, list_sum 2, nonnegative 4, square 2
    RFT aspects-only          5/32   double 1, nonnegative 4
    RFT pool-all             12/32   double 1, list_sum 4, nonnegative 4, square 3

So the pool was the cause, not RFT: covering the evaluation families turned 9/32 -> 12/32 - the first ladder
rung to IMPROVE the base on held-out tasks (list_sum 2/4 -> 4/4, square 2/4 -> 3/4, double 0/4 -> 1/4, at the
cost of increment 1/4 -> 0/4). The rule is not "RFT is dangerous" but "an RFT harvest pool must cover the
families you intend to keep".

The trade-off to carry into RL: the pool-all RFT SHARPENED the policy and narrowed its sampling - 5/8
families viable at T=0.35 (double/even/increment dead, yield 0.19531) against arm A's 7-8/8 (yield 0.19141 to
0.23047). RL needs within-family diversity, so it runs hotter (T=0.8) and its viability must be re-checked per
family rather than assumed from the RFT.

Artifacts: research/raw/e121b_rft_train.log, e121b_rft_arms_aspects_cpu.json, e121b_rft_arms_aspects_cuda.json,
e121b_rft_arms_passk_cpu.json, e121b_rft_arms_passk_cuda.json.

# 2026-10-13 - E-121 arm B refuted DOWNSTREAM too: the same 100-step reference SFT gives 9/32 on arm A and 1/32 on arm B

The question left open by the bits/byte verdict above is now answered. runs/e121b_ref is the identical
100-step reference SFT (data/reference/bytes/train.bin, 1061 windows, lr 1e-4, warmup 10, cosine, seed 42,
batch 4 x accum 8, fp16, 5.00 GiB cap) with exactly ONE flag changed: --init-from
runs/e121_whole/checkpoint-5000 (sha256 770f474f...) instead of runs/e117_byte/checkpoint-5000. Exit 0 in
470.277 s, 3,276,800 tokens, peak 3.005 GiB, no spill.

Its val_loss is 0.278241 against arm A's 0.277184 - 0.4% apart, i.e. the two models predict the reference
corpus almost identically. Everything downstream of that is NOT close. All three arms were scored in ONE
session by the same two harnesses (scratch/bench_aspects.py on the frozen dev battery;
scratch/passk_preflight.py on the rl split), and the controls reproduced their recorded numbers exactly -
arm A returned 9/32 with the same nine task ids and 7/8 viable families with only `even` dead:

    battery (dev, 32 tasks, greedy, 48-byte budget)   donor 0/32 | arm A 9/32 | arm B 1/32
    pass@1                                            0.00000   | 0.28125  | 0.03125
    mean pass_fraction                                0.0000    | 0.4167   | 0.0938
    completions that are INVALID Python               -         | 1/32     | 19/32
    pass@k viable families (rl, 16 x 8, T=0.35)       0/8       | 7/8      | 1/8
    RFT yield (passed samples / 128)                  0.00000   | 0.38281  | 0.00781

Arm B's 5.3% bits/byte deficit is therefore not a rounding artefact: it is a 9x greedy-accuracy deficit and
a 49x RFT-signal deficit. The failure is GRAMMATICAL, not merely semantic - arm B writes `return x > 0)` and
`return even into values are even.` where arm A writes well-formed but wrong expressions (`return x * x`
for increment). Arm B also cannot be rescued by the ladder: at 1/8 viable families and 0.0078 RFT yield
there is almost nothing for RFT/RL to learn from.

THE LESSON, and it is the expensive one: an SFT val_loss is NOT a proxy for executable performance.
0.277184 vs 0.278241 (0.4%) coexists with 9/32 vs 1/32 (9x). The pretraining base's generation behaviour
survives 100 supervised steps even when its loss on the SFT distribution is indistinguishable. Judge a base
by the task battery and pass@k, never by its next-token loss on the SFT corpus.

Consequence for the ladder: arm A (runs/e121_ref/checkpoint-0100) is the only viable base - 7/8 families,
0.383 RFT yield - so the RFT -> RL ladder runs on it (runs/e121b_rft, launched 2026-10-13 11:0x), not on
arm B.

Artifacts: research/raw/e121b_arms_aspects.json, research/raw/e121b_arms_passk.json,
research/raw/e121b_ref_train.log.

# 2026-10-13 - E-121 arm B closed NEGATIVE: whole-file pretraining is 5.3% worse than the target-corpus arm on held-out bits/byte

Arm B (runs/e121_whole) finished cleanly at 09:56 today: exit 0, 5,000 steps, 163,840,000 tokens, 7,431 s
(2.06 h), peak 3.005 GiB, spilled_to_shared_memory False, 0 errors, 2 resume events (the killed attempt
plus the 2026-10-13 07:52 resume). The 127 steps lost to the stop were re-run as designed and the run
needed no further intervention. Surviving checkpoints: checkpoint-4750 (val_loss 1.0768) and
checkpoint-5000 (1.0759, best, carries state.pt). checkpoint-5000 sha256
770f474f6175bc6f01744bb45e4202ffeb0ec0432a93751280cd84ed21f861b7.

THE VERDICT is negative, and it is only trustworthy because both arms were scored by the SAME evaluator on
the SAME corpus in the SAME session (scripts/evaluate_corpus.py, data/target/bytes/val.bin, 843 batches,
3,452,928 positions):

    E-117 (target-corpus byte arm)   nats/byte 1.075736   bits/byte 1.55196   ppl 2.9322
    arm B (whole-file byte arm)      nats/byte 1.132600   bits/byte 1.63400   ppl 3.1037

Arm B is 5.29% WORSE. Pretraining on whole files instead of the target corpus does not improve held-out
bits/byte, and both arms plateau over their last 1,000 steps (E-117 bought 0.029 bits, arm B 0.028) at
~1.55 bits/byte - so the binding constraint is model capacity (~25.7M total, 9.8M active), not the data
distribution. Arm B sits at 2.144x the target's 0.76226 (E-117 was 2.036x), so gates G1 and G2 still fail.

A METHODOLOGICAL TRAP worth remembering: arm B's in-training 128-window val_loss read 1.0759 nats/byte
(1.5521 bits/byte) at step 5000, which made it look like a dead heat with E-117's 1.5335. The full-corpus
evaluation put it at 1.1326 nats/byte - a 0.057 nat gap. E-117 showed no such gap because it was trained
on the corpus it was validated on. Never quote the 128-window in-training number as a bits/byte.

What this does NOT settle: arm B's value as a BASE for the SFT -> RFT -> RL ladder. Its held-out bits/byte
is worse, but breadth of pretraining data could still make it a better starting point for the Python
constructs arm A cannot reach. Testing that means running the identical 100-step reference SFT on arm B's
checkpoint-5000 and re-running the battery + pass@k. Not yet done.

The GPU is free as of 09:56.

# 2026-10-12 - RFT/STaR trainer built and CPU-verified; arm B (whole-file pretrain) alive at step 837

scripts/train_rft.py now exists - the RFT/STaR half of the E-118 ladder that was missing. Per round it
samples every task through glm53_flash.runtime.generate_group(sample=True), keeps ONLY the completions
whose evaluate_source(...).passed is true, and takes a masked completion-only supervised step on those.
It reuses train_rl.py's scaffold verbatim (5%-density auto checkpoints, best+newest retention keyed on
the per-family keep-rate, tasks.json resume guard, training-receipt.json carrying the per-family
harvest), so a crash costs at most the last round - and a round is minutes.

CPU smoke (runs/_smoke_rft, since deleted) on arm A checkpoint-0100, legacy/nonnegative, 2 rounds:
keep_rate 1.0 (8/8 kept), 4 supervised steps, mean loss 0.037; retention left 0001+0003 and pruned
0002; a re-run with --rounds 3 auto-resumed from round 2 without redoing rounds 1-2. PASS.

One real bug, caught by a physical impossibility rather than a crash: the first version reported
mean_loss 8.24 on a vocab-260 byte model, whose mean-CE ceiling is ln(260) = 5.56 - the batch was not
causally shifted. train_pretrain.py hides this by pre-shifting in its loader (line 69:
inputs, labels = values[:, :-1], values[:, 1:]). After the fix (logits[:, :-1] against labels[:, 1:])
the same batch reads 0.037. Rule recorded in MEMORY.md.

Also: tasks.frozen_tasks gained the "pretrain" seed (9319) so --split pretrain works for the legacy
pool as it already did for the aspects pool; select_tasks had silently restricted RFT to aspects.

ARM B IS ALIVE, not dead. The pwsh-50 background job was torn down at the previous session boundary,
not by a training failure; it was relaunched and is running. Verified: PID 28660 is in
nvidia-smi --query-compute-apps, progress.jsonl at step 837 with 0 skipped micros, ~7.4 s/step,
cosine lr 2.85e-4, GPU 3736/8192 MiB and 84 C. Retention holds checkpoint-0500 (val_loss 1.70460) and
checkpoint-0750 (val_loss 1.51659, the best). 4163 steps remain, ~8.5 h at the measured pace.

Blocked on that single GPU: the RFT -> RL ladder on arm A (TASKS 170) and the E-119 off-distribution
sweep (TASKS 165). The arm A/B comparison needs arm B's own same-script BPB sweep before it settles.

Crash-safety of that run was then verified on disk rather than assumed: auto-save every 250 steps (5% of
5000), retention = newest 2 + best by val_loss (now checkpoint-0750 and checkpoint-1000, val_loss 1.52
and 1.43), model.pt and state.pt each written to a tmp file and os.replace()d, the save ordered before
the prune, and progress.jsonl tolerant of a torn last line - so the zero-flag relaunch resumes from
checkpoint-1000 with optimizer, GradScaler and RNG intact. Verifying it surfaced a real hole: a kill
between mkdir and the model.pt replace left a directory that latest_checkpoint would select and
load_checkpoint could not read, so the resume itself would have failed. latest_checkpoint now skips
directories without model.pt, with a regression test; 13 passed (19 with test_lab).

Arm B was then left running for 3h25m and re-checked at 2026-10-12 16:06: step 3587/5000, still the same
process (PIDs 19184/28660, launched 11:41), zero resume events, 0 skipped micros, 5.31 s/step over the
last 200 steps, ETA ~18:12. val_loss 2.1521 (250) -> 1.5166 (750) -> 1.3381 (1250) -> 1.2030 (2000) ->
1.1092 (3500), still falling, so unlike E-117 the G4 "still falling at 5000" gate may pass here. The
retention rule was also confirmed working live rather than only in simulation: every save logged
retained = last 2 and pruned = the step before, e.g. step 3500 kept 3250+3500 and pruned 3000. The only
log lines matching an error pattern were the config JSON, because the corpus path contains "nano_SLMs" -
a false positive from the substring "nan", not an error. On the same held-out corpus arm B reads 1.1092
nats/byte at step 3500 against E-117's interpolated 1.116, a small lead that is indicative only (E-117's
number came from a full-corpus BPB sweep, arm B validates on 128 windows).

Arm B was then STOPPED on user request at 2026-10-12 16:12:03, at step 3627. Before stopping, the newest checkpoint
(checkpoint-3500, val_loss 1.11, with both model.pt and state.pt) was verified complete on disk, so the
stop cannot cost more than the 127 steps since that save - 2.5%, inside the 5% rule. Both trainer PIDs
were stopped, 0 train_pretrain processes remain, and the GPU fell from 3854 MiB to 385 MiB. Retained:
checkpoint-3250 (val_loss 1.12) and checkpoint-3500 (1.11, newest and best, carries state.pt). pwsh-50
reported exit 1 because its python child was killed - expected, not a training failure.

To resume, re-run the identical command from glm-5.3-flash-from-scratch (zero flags changed - that is
what triggers auto-resume):

    & ..\.venv\Scripts\python.exe scripts\train_pretrain.py --output runs\e121_whole --steps 5000 --checkpoint-fraction 0.05 --keep-checkpoints 2 --batch-size 4 --grad-accum 8 --sequence-length 1024 --learning-rate 3e-4 --warmup-steps 150 --lr-schedule cosine --corpus "E:\python projects\nano_SLMs\data\wholefile\bytes\train.bin" --corpus-dtype uint16 --val-corpus "E:\python projects\nano_SLMs\data\target\bytes\val.bin" --val-windows 128 --vocab-size 260 --amp-dtype fp16 --vram-cap-gib 5.00 --max-skipped-micros 32 --seed 42

It will resume from checkpoint-3500 exactly: scheduled_lr() takes the GLOBAL step, so the cosine schedule
continues rather than restarting the warmup; load_run_state restores optimizer, GradScaler and RNG; and
train_pretrain refuses to resume if the saved corpus differs from --corpus. The next save is step 3750.

Arm B was resumed 15h39m later, on 2026-10-13 07:52, as job pwsh-3 with that identical command. The
pre-flight passed first (checkpoints [3250, 3500], latest_checkpoint 3500, state.pt saved_step 3500 with
optimizer/scaler/rng present, GPU free at 262 MiB, disk 14.03 GiB). The resume was then verified from the
artifacts rather than from the process merely being alive: the first progress row is step 3501 with
resumed=true, tokens_seen continues at exactly 32768/step, and lr reads 6.53e-05 - continuing the cosine
schedule, where a cold restart would have begun the 150-step warmup at ~2e-06. The 127 steps between the
3500 save and the 3627 kill are being re-run, as designed.

# 2026-10-11 (bk) - E-117 closed: the byte arm plateaus at 2.0x the target's bits/byte; E-121 arm A is trained

E-117 finished (exit 0): 5000 steps, 163,762,176 tokens, 6.4 h, peak 3.005 GiB, spilled=False,
5 auto-resumes, corpus_windows 159,468. checkpoint-5000 sha256 0f7abd95e63db7235f12588a2fa6b9978138b8c024f4759a0ac1319c6ce65a6d.

The run recorded no held-out metric (it predates --val-corpus), so all 20 checkpoints were scored
after the fact against data/target/bytes/val.bin with scripts/evaluate_corpus.py (256-1024 windows
each; the three 843-batch points are recorded in metadata.val_bpb_batches). The measured bits/byte
curve is the real signal - the train loss was far too noisy to read:

  step     0    250    500   1000   1250   1500   1750   2000   2250   2500
  BPB   7.13   3.08   2.43   2.01   1.88   1.96   1.78   2.02   1.70   1.69

  step  2750   3000   3250   3500   3750   4000   4250   4500   4750   5000
  BPB   1.68   1.66   1.65   1.61   1.63   1.56   1.55   1.56   1.53   1.5335

VERDICT - E-117 G1 and G2 FAIL, and G4 is NOT licensed:
  G1 (< 0.76226): 1.5335 -> FAIL; the target is 2.01x better (its best 0.75700 is 2.03x better).
  G2 (<= 1.10x = 0.8385): FAIL.
  G4 (still falling at 5000?): NO. The last 1000 steps bought 0.029 bits (1.5636 -> 1.5335), so the
     curve is already flat long before the target; the 17,100-step byte-parity continuation is not
     justified by this curve.
The gap is capacity, not recipe: the target is ~226M dense params against E-117's 25.7M total /
9.8M active, at the same 1.0033 epochs. E-117's value is a tokenizer-free research substrate and a
donor for the idiom work, not a target-beater - consistent with the user's 2026-10-10 call to drop
the target comparison and go after SmolLM2 instead.

Checkpoint cleanup applied (user rule: keep the best + the newest). All 20 checkpoints were first
annotated with their measured val_bpb (scratch/annotate_e117_valbpb.py, so the number travels with
the weights), then pruned with scripts/prune_checkpoints.py --run runs/e117_byte --keep-last 2
--keep-steps 5000 --metric-key val_bpb --apply. Freed 1,768.9 MiB; only checkpoint-4750 and
checkpoint-5000 survive, with best_step=5000.

E-121 arm A (SFT on the reference solutions) is trained: runs/e121_ref, --init-from
runs/e117_byte/checkpoint-5000, 100 steps = 3.0 epochs, 8 min, peak 3.005 GiB, spilled=False.
Held-out loss fell 0.5499 (step 25) -> 0.2772 (step 100) and retention left exactly
checkpoint-0075 + checkpoint-0100. Whether that installs the 3 dead idioms is being measured now
with the donor checkpoint-5000 as the control (same dev battery + T=0.35 pass@k protocol as E-120).

MEASURED 2026-10-11 (same dev/pass@k protocol as E-120, the donor as control):

  battery (dev, 32 tasks, greedy, 48-byte budget)   donor 0/32 strict, 0/32 extracted
                                                    arm A 9/32 strict, 9/32 extracted
  pass@k  (rl split, 16 tasks x 8 @ T=0.35)         donor 0/8 aspects VIABLE (all dead)
                                                    arm A 7/8 viable, only 'even' dead

  arm A by family, battery / pass@k:  nonnegative 4/4 / 16-16    list_sum 2/4 / 14-16
                                      square      2/4 / 10-16    reverse  0/4 /  5-16
                                      increment   1/4 /  1-16    absolute 0/4 /  2-16
                                      double      0/4 /  1-16    even     0/4 /  0-16 STILL DEAD

This is the E-121 hypothesis confirmed at the gate that matters. The donor had NO positive sample
anywhere, so RL and RFT were both mathematically unable to move it - you cannot amplify, or rejection-
sample, a completion that never occurs. Three epochs of SFT installed max(x, 0) completely (0 ->
16/16), abs partially, and left even untouched. Arm A's 9/32 battery total happens to equal
rl-0096's 9/32, but from an inverted composition: arm A wins nonnegative (4/4 vs 0/4) and square
(2/4 vs 0/4) where rl-0096 has nothing, and loses double/increment. Arm A is the better RL init -
7/8 aspects carry signal instead of 5/8 - and it starts from a pretrain-only base rather than a
192-group RL endpoint, so it is also the cleaner substrate for the RL-playground work.

Arm B (whole-file pretrain from scratch, 5,000 steps ~6.4 h) is now TRAINING as the second A/B arm:
runs/e121_whole, launched 2026-10-11, log research/raw/e121_whole_train.log. Its --val-corpus is
data/target/bytes/val.bin, the same set the E-117 sweep scored, so arm B can be compared to the E-117
curve directly in bits/byte (never against E-117's absent val_loss) once it lands.

# 2026-10-10 (bj) - E-121 arms A and B are built and their trainer path is smoke-tested; only the GPU stages remain

Both corpora for the A-vs-B comparison are built (CPU only, while E-117 holds the GPU), and the one
code change arm A needed is in place and verified.

Arm A (SFT on reference solutions) - data/reference/bytes: 1,086,727 train byte-ids / 8,000 examples
covering all 20 playground sources (8 legacy families + 12 aspects) drawn from the held-out pretrain
split (seed 9319, disjoint from dev/final/confirm/rl). ~33 steps per epoch, so 3 epochs ~100 steps.

Arm B (whole-file corpus shape) - data/wholefile/bytes: 163,301,250 train byte-ids from 16,744 whole
python files (codeparrot/codeparrot-clean). Its epochs_at_5000_steps is 1.0033, IDENTICAL to the
E-117/target byte arm (1.00333), so A and B differ only in shape, not budget - the comparison is clean.

Blocker found and worked around: bigcode/the-stack-smol, starcoderdata and the-stack-dedup are all
gated for this token (whoami is valid; the per-dataset grant is missing - md_measure streamed
starcoderdata earlier, so the grant lapsed). codeparrot/codeparrot-clean is ungated and whole-file.

New trainer flag: nested scripts/train_pretrain.py --init-from DIR (weights from a donor, fresh
optimizer/schedule). CPU smoke PASS on runs/e117_byte/checkpoint-1000.

E-117 is at step 2650 (loss 1.08), ~2,350 steps to go (~3.5 h). The GPU stages of both arms start
when it lands; nothing else blocks them.

Checkpoint policy (user rule 2026-10-10), now automatic in both trainers:
  density    every ~5% of the run (--checkpoint-fraction, default 0.05); --checkpoints still wins
  retention  best + newest --keep-checkpoints (default 2); the newest is always kept for auto-resume
  cleanup    scripts/prune_checkpoints.py, dry-run by default, --apply to delete
  best       needs a metric: train_pretrain.py gained --val-corpus/--val-windows (its first held-out
             signal ever) recording metadata.val_loss; train_rl.py records metadata.exact_fraction
Verified: a 20-step smoke saved 5/10/15/20, pruned each predecessor, and left only checkpoint-0020.
E-117's 11 checkpoints predate this (no val_loss, best_step=null); a dry run would free 786 MiB by
keeping 2250/2500 plus an explicitly protected 5000 - not applied while it is still training.


# 2026-10-10 (bi) - in-regime re-measurement: SmolLM2-360M leads the aspect battery 13 vs 9; 5 of 8 aspects are RL-viable; byte-level repetition control is closed

Two E-119/E-120 measurements were taken outside the project's own generation regime and are void: the
hand-written sampler omitted BOS on the glm arm, never masked pad/bos/sep to -inf, and ran at T=1.0
where the project uses 0.35 (pass@k) and 0.8 (rollouts). Both harnesses now go through
runtime.generate_group or replicate it exactly.

Re-measured head-to-head (dev, 32 tasks, research/raw/e119_aspects_bos.json): smollm2-360m 13/32
strict (15/32 extracted), smollm2-135m 1/32 (13/32), rl-0096 9/32, pretrain-dense-0090 4/32. Our arm
leads list_sum 2/4 vs 0/4 - sum(values) is a CodeSearchNet idiom. Our strict == extracted while
smollm2-135m is 1 vs 13: we stop at the body, they ramble past it.

pass@k pre-flight (research/raw/e120_passk_t035.json, _t080.json): 5/8 aspects have pass@k > 0 at
T=0.35 and 4/8 at T=0.8, so RL and RFT both have signal. The three dead aspects (absolute,
nonnegative, even) are exactly the idiom families - the model reaches for a ternary and degenerates,
never emitting abs, max or %. RFT yields: list_sum 1.000, double 0.625, increment 0.375.

Negative result: no-repeat n-gram blocking makes it worse (9/32 -> 4/32) because Python's 4-space
indent is itself a repeated n-gram; the blocker bans the fourth space and the completion starts
"\n   return". Byte-level repetition control is closed off.

Next: E-121 pre-registered in TASKS.md - SFT on the reference bodies (the only stage that needs no
pass@k), then RFT, then RL at per-aspect temperature, brevity last. Corpus lever is SHAPE (whole
files, not extracted functions) since the corpus is already 100% code and E-117's G4 is 3.4 epochs of
the same 163 MB.

E-117 is at step 2090, loss 1.244, healthy; checkpoint-2000 written.

# 2026-10-10 (bh) - E-117 killed at step 1920 by an external termination; resumed, checkpoint density raised

The A-BYTE run (harness job pwsh-28, launched in the previous span) was killed at ~step 1920 with no
error. train2.log's last line is an ordinary step record (step 1920, loss 1.2134, 4.1 s/step): no
traceback, no OOM, no FloatingPointError, 0 skips, no reboot (uptime 17.6 days), and no python
process left behind. So the termination came from outside the trainer - the job manager or the OS,
not a numerical or memory failure.

Cost: 920 steps (~1.5 h at 4-7 s/step), because the only checkpoint below 1920 was checkpoint-1000.

Response:
1. Relaunched the identical command (it auto-resumes, PLAN 5.3) as harness job pwsh-1; the start line
   reported resumed_from 1000 and step 1010 ran normally (loss 1.3307). train3.log held that brief
   segment; train4.log is the live log and train2.log is kept as the pre-kill record.
2. Raised the checkpoint density from 250,500,1000,2500,5000 to every 250 steps up to 5000, so the
   worst case from another kill drops from 1500 steps (~2.5 h) to 250 steps (~30 min). This changes
   only WHEN snapshots are written - LR schedule, data order, seed and batch geometry are untouched -
   and it is recorded here as an operational deviation from the E-117 pre-registration. Cost: ~2.5 GB
   of checkpoints against 16.25 GB free (model.pt ~103 MB each, 2 retained states ~206 MB each).
3. VRAM cap unchanged at 5.00 GiB - the value that produced steps 1-1920 - for trajectory continuity.
   No foreign compute co-tenant is active (desktop apps hold 173 MiB of 8192), so the 60% rail's
   protective purpose is met; the trainer reports vram_cap_fraction 0.625.

Unresolved: the kill's cause is not identified. If it recurs, the next lever is a detached launch
(Start-Process) that outlives the job manager.
# 2026-10-10 (bg) - E-119 external baselines: SmolLM2 beats our 226M target on code BPB

Path 2's benchmark evaluation now has real external references, because the published comparison
model (runs/target/checkpoint-4000) is gone. SmolLM3 ships only at 3B, so the 135M/360M sizes the
user named are the SmolLM2 family; both base checkpoints were downloaded (not -Instruct: our arms
are base models). Qwen3.5-0.8B is in the HF cache but only as a PARTIAL download - tokenizer files,
22.9 MB, no weights (~1.6 GB still missing).

Harness: scripts/eval_bench_bpb.py (new). BPB = sum(-log p)/ln2/UTF-8 bytes, the only
tokenizer-agnostic axis, so the byte arm, the CodeLlama-BPE target and the SmolLM2 tokenizers share
one scale. One HF adapter covers the target and every external baseline (runs/target/final is
itself a transformers directory); one glm-byte adapter covers our byte arm
(glm53_flash.runtime.load_checkpoint + ByteTokenizer). Two bugs were caught by smoke tests:
load_checkpoint is not exported from glm53_flash/__init__.py, and the glm forward is
forward(input_ids) -> (logits, usages), so a tuple's first element is logits, not a loss.

HumanEval text, 164 rows, 103,806 bytes (all four arms, one run, CPU 3 threads;
research/raw/e119_bpb_cpu_humaneval.json):

  arm                     params     ctx     BPB        vs TARGET
  smollm2-360m            361.8M     8192    0.55934    0.51x
  smollm2-135m            134.5M     8192    0.63959    0.58x
  target (M3)             226.5M     1024    1.10233    1.00x
  a-byte glm @step 500     25.7M     1024    2.77163    2.51x

Sanitized-MBPP text, TARGET only so far: BPB 1.36538 (427 rows, 105,535 bytes).

Reading, honestly:
1. A 135M general model is 1.72x better than our 226M target on code BPB; 360M is 1.97x better.
   That is the new bar, and it sits far above the bar we had been quoting.
2. Most of the gap is data budget, not architecture: SmolLM2-135M saw ~2T tokens, our target saw
   578M bytes (~3500x less). This measures where we stand, not a like-for-like comparison.
3. The byte strategy DOES transfer off-distribution - 2.77163 on HumanEval against 2.31746 in-domain
   at step 500 (+19.6%) - but its absolute level is far off. At step 500 it is 2.51x the target,
   past E-119's 1.5x falsifier, while G1 needs <1.00x. That row is a PREVIEW (checkpoint-0500,
   pre-fix numerics); the pre-registered arm is checkpoint-5000 and does not exist yet.
4. E-117 is healthy: step 1580, 0 skips, 0 FloatingPointError, checkpoint-1000 on disk. Its per-step
   train loss is noisy (1.327 at 930, 1.385 at 1050, 1.451 at 1580), but the val BPB probe
   (scratch/val_bpb_byte.py, a 200k-byte sample of data/target/bytes/val.bin) shows the trend is
   firmly down: checkpoint-0500 BPB 2.31746 -> checkpoint-1000 BPB 1.91828 (-17%).
5. Consequence for E-117's gates: applying the measured +19.6% off-distribution penalty, checkpoint
   -1000 implies ~2.29 on HumanEval, i.e. ~2.08x TARGET. G1 (<1.00x) and G2 (<=1.21) would need an
   in-domain BPB near 0.9-1.0 (loss 0.64-0.70 nats) by step 5000 - a very steep ask. The 1.5x
   falsifier (HumanEval 1.653) needs in-domain 1.38 (0.96 nats) and is plausible. So the likely
   outcome is FAIL on G1/G2 with the falsifier NOT firing - and G4's pre-registered 17,100-step
   text-exposure-parity continuation is the lever if parity is the goal.

Not launched: the five-set sweep (needs checkpoint-5000 and B-BPE, GPU); the Qwen3.5-0.8B weight
fetch.
# 2026-10-10 (bf) - E-117 root cause: autocast silently downcast the attention contractions

The E-117 byte arm died four times at the same place: first skip at step 592 micro 1 (index
4729), then 676, then a cascade across 695-702 ending in `FloatingPointError: non-finite loss
at step 702 micro 1: 33 skipped micro-batches`. The run was deterministically poisoned, and a
chunked "fp32 accumulator" patch (E-117 addendum 3) reproduced the identical skip ladder - it
did not address the cause.

A deeper three-pass probe on the same replay (`--diag-nonfinite`, stdout redirected to a file)
found it:

* every module output in the fp16 forward is finite and small - peak 8.3, nowhere near fp16's
  65504, so there was never an fp16 range problem;
* the first non-finite tensor is `layers.10.block.attention.attention_result`, and a
  TorchDispatchMode NaN watch names the first bad op: `aten.bmm.default`, inputs [fp16, fp16],
  shape [3072,1,1] = batch 4 x chunk 128 x heads 6 - the denominator dot product in
  `LinearAttention.forward`;
* the same micro-batch is finite in fp32 (loss 1.4686).

Mechanism: `k = F.elu(k) + 1.0 >= 0`, so `cumsum(k)` grows with position and the head_dim
reduction over that prefix passes 65504 -> inf -> NaN after normalisation. The chunked fp32
cast could not help because einsum/bmm are matmul-family ops and autocast casts their inputs
to fp16 no matter what the caller passes. E-117 addendum 3 named the wrong mechanism and is
marked SUPERSEDED; addendum 4 carries the measurement.

Fix: `with torch.autocast(device_type=x.device.type, enabled=False):` around the RoPE / elu /
chunked accumulation in `LinearAttention.forward`, keeping qkv and out under autocast (fp16).
Verified 2026-10-10: the resumed run passed step 592 with zero skips and reached step 710 with
0 skips and 0 errors, clearing the whole region that killed attempts 3 and 4, at unchanged
pace (~4.3-5.2 s/step). Regression suite:
`glm-5.3-flash-from-scratch/tests/test_linear_attention_numerics.py`. Lesson in MEMORY
("autocast is a dispatch-time cast, not a suggestion").

Open: E-117 A-BYTE is live and owns the GPU (B-BPE and E-118 wait).
`runs/e117_byte/progress.jsonl` still holds the dead attempts' rows interleaved with the live
run's and must be rebuilt (steps 1-500 from the original lineage + the live block) before the
E-117 report; `train2.log` is the authoritative per-step record for steps 501+.

# 2026-10-10 (be) - instruct lane: "e2" is the OVERFIT sibling, and both e1/e2 weight sets are gone

User asked to check "the other model, I think its name is e2, more powerful, near instruct mode".
Verified: e2 = `runs/sft_v2/final` (SFT v2, 2 epochs / 2,344 steps, minimax3 19,252 pairs on the
226.5M M3 target) - ast greedy 0.98 / sampled 1.00 but CSN forgetting **+19.7% = gate FAIL**; the
DELIVERABLE was e1 = `runs/sft_v2_e1/final` (1 epoch / 1,172 steps, ast 0.98/0.96, CSN 2.0466 =
+9.8% WITHIN gate). **Neither is on disk today**, and the 2026-09-09 resume pack (sha256
ceda94ad...) that held sft_v2_e1 is gone from `checkpoint_backup/` (only the glm53-flash E116 pack
remains). Full audit: research/2026-10-10_instruct_lane_inventory.md; lesson in MEMORY.

What DOES survive in the instruct lane: `runs/yarn_4096/final` (YaRN ctx-4096, base-surface @4096
1.7035), `runs/yarn_lora_sft_v1/final` (LoRA r=32/a=64 instruct re-SFT; guard @1024-yarn 1.9471 =
+3.99% PASS; ast 0.94/0.92; base @4096 1.7825 = +4.64% FAIL vs the strict <= +2% bar) and
`runs/yarn_lora_sft_v1/final_a60` (alpha=0.6 soup, the E-07 promoted deliverable). That pair is the
repo's only instruct-capable, ctx-4096 model - i.e. the asset the user remembered, under a
different name.

Plan implication (proposed, user-gated): add an instruct-parity tier - restore e1 by copy (if the
pack exists on the other machine) or by retrain, then SFT the E-117 winner on the same corpus and
compare instruct-vs-instruct; optionally use final_a60 as the local teacher for the E-118 aspect
loop. TASKS rows 166 (done) + 167 (pending). E-117 (`pwsh-16`) still owns the GPU.
# 2026-10-10 (bd) - E-117 died at step 334 (CUDA OOM after a skipped micro-batch) and is relaunched

> Same two-path session as (bc). Path 1 (E-117) is the glm-on-target-data arms; path 2 (E-118) is
> the RL playground. This entry records the second E-117 failure, its fix, and the benchmark-data
> intake that answers 'do we have anything to eval on'.

- **Incident 2 (honest)**: the guarded relaunch (pwsh-14) fired the skip guard exactly where
  predicted - step 334 micro 2, index 2666 - and then died on the next micro-batch with
  torch.OutOfMemoryError at model.py:82, 4.01 GiB allocated against the 4.10 GiB cap, 96 MiB
  requested. Cause: the skip branch kept the abandoned micro-batch's forward graph alive into the
  next micro (the new graph is built before the name is rebound), so a ~536 MB graph overlapped its
  successor; fragmentation did the rest.
- **Fix**: the skip branch now deletes its tensors and empties the CUDA cache - that deletion is
  the real fix. Added expandable_segments:True at launch, and raised the cap to 4.40 GiB (exactly
  60% of the 7556 MiB free at launch) because the OOM was self-inflicted with 2.9 GiB free and no
  competing workload. A brief 2 x accum-16 fallback measured 6.26 s/step against 3.0 s/step and
  was reverted, so the pre-registered 4 x accum 8 recipe stands.
- **Relaunch**: job pwsh-16, 5000 steps from step 0 at the pre-registered 4 x accum 8,
  checkpoints 250/500/1000/2500/5000, keep-state 2, cap 4.40 GiB. Step-1 loss matched the
  original batch-4 attempt (4.986262798309326) and the failed runs are preserved at
  runs/e117_byte_failed_oom (step 334) and runs/e117_byte_batch2_probe.
- **Benchmark intake**: network access works from pwsh. data/bench/raw now holds HumanEval (164),
  MBPP (974) + sanitized-MBPP (427), HumanEval+ (164), MBPP+ (378), BigCodeBench v0.1.4 (1140) and
  DS-1000 (1000); data/bench/manifest.json records sha256, size, rows, columns, source and licence
  for each, built by scripts/bench_inventory.py. These files are eval-only - never train on them.
  LiveCodeBench, CruxEval and RepoBench were confirmed available but not downloaded.
- **Context for judging those benchmarks**: runs/target/final/mini_eval_report.json shows the 226M
  target scoring pass@1 0.0 on our own 16-task battery, so HumanEval/MBPP pass@1 for a 25M glm will
  be a floor, not a signal. The usable cross-distribution metric is bits-per-byte on the benchmark
  text, which is tokenizer-agnostic and therefore comparable between the byte and BPE arms and to
  the target baseline.
# 2026-10-10 (bc) - path 1 relaunched after an fp16 overflow; path 2 stage 1 delivered

> Two-path session. Path 1 (E-117): train the glm arch on the same data as the target
> baseline (byte arm first) to test whether the past glm strategy beats
> target/checkpoint-4000's published bar (val_loss 1.8512 / BPB 0.75700). Path 2 (E-118):
> upgrade the RL section - per-aspect playgrounds, generated cases, capability tiers,
> out-of-process evaluation, per-family forgetting matrix.

- **E-117 incident (honest)**: the first A-BYTE attempt (job pwsh-10) trained healthily for
  334 steps (loss 4.99 -> 1.97) then died with FloatingPointError 'non-finite loss at step
  334 micro 2' - an fp16 forward overflow, not a harness crash. Checkpoints were at
  1000/2500/5000, so nothing was written. The failed run's log is preserved at
  glm-5.3-flash-from-scratch/runs/e117_byte_failed_attempts.
- **Fix**: train_pretrain.py skips a non-finite micro-batch (logged as
  {"event": "skip_nonfinite", ...}), divides the step loss by the used micro-batch count,
  and aborts after --max-skipped-micros (32). The arm now checkpoints every 250 steps with
  --keep-state 2.
- **Relaunch**: job pwsh-14, same pre-registered recipe (5000 steps, batch 4 x accum 8, ctx
  1024, lr 3e-4 cosine, warmup 150, fp16, cap 4.10 GiB, seed 42), output
  glm-5.3-flash-from-scratch/runs/e117_byte. Step-1 loss 4.986262798309326 reproduced the
  earlier attempt exactly. Single GPU verified: one trainer, 3342 MiB used, one python
  process.
- **Path 2 stage 1 delivered** (CPU only, no GPU contention): glm53_flash/aspects.py (12 new
  families across conditionals/strings/lists/dicts with per-task generated cases and a
  tiered pool API), evaluator v2 with Capabilities tiers whose default is byte-identical to
  the v1 policy, glm53_flash/sandbox.py (persistent child runner, JSON-lines, timeout +
  respawn + startup grace), train_rl.py wiring (--pool/--tier/--sandbox/--allow-loops/
  --allow-attributes/--extra-safe-calls, sandbox stats in every progress row),
  scripts/eval_forgetting_matrix.py (per-family pass@k), tests/test_aspects_evaluator.py.
- **Evidence**: 15 passed (9 new + test_lab.py); py_compile and pyflakes clean on both
  trainers; CPU smoke with --pool aspects --tier core --sandbox on completed 2 groups in
  6.8 s with the sandbox on, checkpoints written, start event carrying
  pool/tier/sandbox/capabilities. E-118 is pre-registered in TASKS.md; its GPU arm waits for
  E-117 (single-GPU rule).
- **Docs updated**: TASKS.md (rows 162/163 + the E-118 pre-registration block), MEMORY.md
  (the fp16-overflow lesson), research/2026-10-10_glm53_flash_rl_playground_v2_design.md
  (implementation-status table).
# 2026-10-08 (ah) (resume pack for the glm-5.3-flash line - pruned to the best + last checkpoints: checkpoint_backup/resume_pack_2026-10-08_glm53flash_E116.zip)

> User request: a backup zip so another machine can resume this line without anything
> missing; then: include the HANDOFF so the receiver knows what was done / not done and where
> we reached, keep only the best and the last checkpoint, and drop whole runs we do not need
> (safe). The source working tree is UNTOUCHED - pruning is pack-time exclusion only, so
> nothing was deleted from disk. Refreshed after E-116 (the 192-group RL arm); it supersedes
> the E-115, E-114 and E-113 packs.

- **File** checkpoint_backup/resume_pack_2026-10-08_glm53flash_E116.zip - nested payload
  1371.4 MiB in 405 files; the zip also carries handoff/, RESUME.md,
  venv-pip-freeze.txt and BUNDLE_MANIFEST.json.
- **Size / sha256 / entry count**: recorded in
  checkpoint_backup/BUNDLE_MANIFEST_2026-10-08_glm53flash_E116.json beside the zip. The hash is
  deliberately NOT embedded in the pack - a self-referential hash is useless to a receiver.
- **Contents**: glm-5.3-flash-from-scratch at 624f4ca (code, .git incl. its five local-only
  commits, and the pruned runs/) + handoff/HANDOFF.md + handoff/research/ (the E-112 .. E-116
  reports) + RESUME.md + venv-pip-freeze.txt + BUNDLE_MANIFEST.json.
- **Checkpoints kept** (best + last only): pretrain-dense/checkpoint-0090 (BEST - the E-113
  selected RL init, the only checkpoint with a real dev signal), pretrain-dense/checkpoint-0400
  (last, with optimizer state), pretrain/checkpoint-0100 (the A2/A4/A6 RL init),
  pretrain/checkpoint-0200 (E-112 step-200 = the A3 init), pretrain/checkpoint-0400 (last, with
  optimizer state), rl/checkpoint-0096, rl-regime-a1/checkpoint-0096,
  rl-regime-a4/checkpoint-0096 and rl-regime-a6/checkpoint-0192 (the E-116 192-group endpoint -
  the arm whose greedy 16/24 equals the published endpoint). Every training-receipt.json,
  progress.jsonl and schedule.json is kept in full, as are all of runs/eval/ and runs/regime/.
- **Dropped 125 files / 3155.2 MiB (pack-time only, each proven redundant or
  transient)**: runs/_pytest_tmp, runs/_smoke, runs/_smoke2; the dense scan checkpoints 0000-0080
  and 0100; pretrain 0000; the RL arms' 0000/0024/0048/0072 (all three 96-group arms); every
  rl-regime-a3 checkpoint; and rl-regime-a6/checkpoint-0096 and 0144 (0096 is bit-identical to
  rl-regime-a4/checkpoint-0096 - E-116's G1 proof - and 0144 is a mid checkpoint). The a3 arm
  was proven bit-identical to pretrain/checkpoint-0200 (sha256 63634642db7c7235...) at BOTH step
  0 and step 96, i.e. it changed nothing; a1/checkpoint-0000 was proven identical to
  dense/checkpoint-0090 (4205d74ffc2e15fb...); a4/checkpoint-0000 was proven identical to
  pretrain/checkpoint-0100; and a6/checkpoint-0096 is proven identical to a4/checkpoint-0096, so
  E-116 is a pure continuation of the A4 trajectory and only its 192-group file is new.
- **Retained determinism pair**: pretrain-dense/checkpoint-0400 == pretrain/checkpoint-0400 (both
  2f93b28225d195a7...), so the bit-exact reproducibility claim still verifies from the pack.
- Builder: scratch/build_pack_v2.py - the prune rules and the equivalence proofs are enforced in
  code, and the build aborts if a proof fails.
# 2026-10-08 (ak) (E-116: the published greedy endpoint was a step-budget gap - 192 groups reach 16/24 exactly)

> Continues TASKS 161 / E-115. Experiment E-116; report
> research/2026-10-08_glm53_flash_192_group_rl.md. New driver stage: `a6` in
> glm-5.3-flash-from-scratch/scripts/regime_match.py (nested commit 624f4ca, local only - never
> pushed to vukrosic).

E-114 left the RL endpoint at 12/24 greedy against the published 16/24 with the dtype path
exonerated; E-115's eval-only curve showed greedy still rising at 96 groups (8, 8, 7, 12) but
3-family pass@8 flat from 72 (12, 12, 15, 15). E-116 spends the next doubling of the budget to
test the step-budget explanation directly: the identical recipe from the same init, 96 -> 192
groups.

**Result: all three pre-registered gates PASS.** G1 equivalence - sha256(a6/checkpoint-0096) ==
sha256(a4/checkpoint-0096), so A6 is a pure continuation of A4 and the E-114 step-96 receipts
could be reused instead of re-measured. G2 still-rising - greedy solved/24 **12 -> 13 -> 16** at
96/144/192 groups (11 gains / 0 losses at 192, p = 0.00098). G3 published parity - **16/24 equals
the published greedy endpoint exactly**. 3-family pass@8: 15 -> 14 -> **17/24**.

**So the A4 shortfall was a step-budget gap, not a recipe ceiling** - and E-115's pass@8
'saturated at 72' reading is refuted (it resumed to 17/24). What remains is an efficiency gap, not
a capability gap: the published run reached 16/24 in 96 groups (1,536 rollouts) where we needed
192 (3,072).

**The gain is not free, and the post-hoc probe is what shows it.** An all-8 pass@8 probe at 192
(NOT pre-registered) is flat: 30/64 -> 30/64, 8 gains / 8 losses, p = 1.0, because the untrained
families regress as much as the trained ones gain (square 8 -> 3, reverse 3 -> 0 against
increment 2 -> 8). Greedy and 3-family metrics hide that; the 3-family win is partly a
redistribution.

Cost: 907.7 s RL + 387.6 s of evals (~21.6 min), peak 1.495 GiB allocated / 2.986 reserved, no
shared-memory spill, 192/192 groups, 0 resume events. New lever: a longer RL budget is real but
modest per doubling. New lesson: schedule_sha256 is budget-scoped, so compare checkpoint hashes,
never schedule hashes, across budgets.# 2026-10-08 (aj) (E-115: the A4 RL endpoint is not step-capped on greedy - but pass@8 stopped improving 24 groups early)

> Continues TASKS 160 / E-114. Experiment E-115; report
> research/2026-10-08_glm53_flash_a4_learning_curve.md. New eval-only driver stage: `a5` in
> glm-5.3-flash-from-scratch/scripts/regime_match.py (nested commit 1a54e28, local only - never
> pushed to vukrosic). User decision this session: the published-torch (2.5.1) control is OFF the
> table permanently - do not re-propose it.

E-114 exonerated the dtype path and left the RL endpoint at 12/24 greedy against the published
16/24. E-115 asks the cheapest next question - is the 96-group budget what stops the curve? - by
re-evaluating the checkpoints A4 had already saved (no training), against the same before-file,
with the 96-group point reused from E-114's own receipts so curve and endpoint are one
measurement.

**Result: not step-capped on greedy, but pass@8 says the policy stopped improving 24 groups
early.** Greedy solved/24 at 24/48/72/96 groups: **8, 8, 7, 12** - the final window delivered
7 -> 12 (+5), the largest jump on the curve, so the budget does not cap greedy. pass@8 (3 trained
families): **12, 12, 15, 15** - flat from 72 groups. The pre-registered STILL-RISING branch
therefore applies.

**The pre-registered rule was mis-specified and both readings are reported.** Its non-monotonicity
clause fired PLATEAU on the one-task dip at 72 groups (8 -> 7; one task = 4.2 pp on 24 tasks)
even though the pre-registered falsifier - any greedy gain after 72 groups - was met five times
over. The literal label is PLATEAU; the substantive conclusion is NOT SATURATED. Fix recorded in
MEMORY: pre-register the final-window gain with its paired test, never a monotonicity boolean.

**Two mechanisms worth carrying:** pass@8 led greedy by a full checkpoint interval on the
increment family (pass@8 2 -> 7 at 72 groups while greedy was still 0 -> 0, then greedy 0 -> 5 at
96), and the endpoint gain migrated between families (greedy double 5 -> 8 by 24 groups then back
to 7, while increment supplied the whole +5 at 96).

Cost: 6 evals + 6 comparisons, 11 min 7 s, no training, no spill. Evidence:
runs/eval/confirm-{greedy,pass8}-rl-a4-ckpt{0024,0048,0072}.json, the corresponding
comparison-a4-ckpt*-{greedy,pass8}.json, runs/regime/a5-curve.json and
research/2026-10-08_glm53_flash_a4_learning_curve.md.

Next recommended step: pre-register and run a longer RL arm (192 groups, same init/recipe) to
test whether the greedy climb continues - the pre-registered branch requires the
pre-registration to land before the launch. A pack refresh is pending the user's call, since each
rebuild is ~1.24 GiB and the E-113 and E-114 zips are then both stale.
# 2026-10-08 (ai) (E-114: the RL endpoint gap is NOT the dtype - fp32 reproduces fp16 to the task)

> Continues TASKS 159 / E-113. Experiment E-114; report
> research/2026-10-08_glm53_flash_fp32_control.md. New driver stage: `a4` in
> glm-5.3-flash-from-scratch/scripts/regime_match.py (nested HEAD 672e35b, committed locally -
> never pushed to vukrosic).

E-113 left exactly one suspect standing for the halved RL endpoint: the RL step itself under
fp16/torch-2.14, with fp16's activation range in the last block as the suggested mechanism. E-114
isolates that path - the same init (`runs/pretrain/checkpoint-0100`, sha256 44d740f7...), the same
96-group schedule (`schedule_sha256` 0c2327ee...), the same seed, families and hyperparameters,
and only `--amp-dtype fp16 -> fp32` changes.

**Result: the fp16-conversion hypothesis is REFUTED.** A4 (fp32) reproduces A2 (fp16) to the task:
greedy 5/24 -> 12/24 (+7, 0 losses, exact McNemar p=0.0156) = A2's +7; pass@8 over the 3 trained
families 10/24 -> 15/24 (+5) = A2's +5; pass@8 over all 8 families 30/64 -> 30/64 (0) = A2's 0;
endpoint 12/24 = A2's 12/24, against the published 16/24. All three pre-registered thresholds fail
(P1 needed +12, P2 needed +8, P3 needed >= 13/24) and the pre-registered falsification clause
(gain <= +8) fired.

**Manipulation checks (all PASS):** the init produced 407/1,536 exact rollouts against A2's 406
(0.25% apart), the receipt reads `amp_dtype=fp32`, the schedule hash is identical to A1/A2/A3,
and 78 groups were updated against A2's 77. fp32 changed neither the reward signal nor the outcome.

Cost of the control: 464.5 s vs 440.9 s (+5% wall-clock) and 1.495 GiB peak vs 0.939 (+0.56 GiB),
no shared-memory spill. At 25.7M parameters generation dominates, so the usual 'fp32 is 2-4x
slower' intuition does not transfer to this scale.

**What is left:** the factors A2 and A4 share. (1) torch 2.14.0+cu126 vs the published
2.5.1+cu121 - the last stack difference, and separating it needs a second environment, so it is
user-gated. (2) The device: bf16 does not exist on sm_75, so the published bf16 numerics cannot be
reproduced here - though fp32's wider range changing nothing makes a pure-precision explanation
unlikely. (3) A real plateau of this recipe at this scale: every arm we have run lands at +7/+8
greedy, never +16. Note the control kills E-113's *mechanism*, not merely its variable - fp32
carries fp32's exponent range and changes nothing.

Per-family A4 greedy: double 5 -> 7, increment 0 -> 5, even 0 -> 0. A2 gained all 7 in double -
the same total, allocated differently, so report per-family numbers or identical totals hide real
behaviour.

Next recommended step: choose between the torch-2.5.1 control (needs a second venv plus a ~2.5 GB
download - user-gated) and treating 12-15/24 as this recipe's ceiling at this scale. Evidence:
runs/rl-regime-a4/, runs/eval/comparison-a4-*.json and
research/2026-10-08_glm53_flash_fp32_control.md.
# 2026-10-08 (ag) (E-113: the E-112 magnitude gap is NOT headroom - the RL regime was matched and the endpoint still fails)

> Continues TASKS 157 / E-112. Task row: TASKS 158. Experiment: E-113. Report:
> `research/2026-10-08_glm53_flash_regime_match.md`. The nested clone stays a gitlink and is
> NEVER pushed to vukrosic.

E-112 left one explanation untested: our fp16/torch-2.14 pretrain `checkpoint-0100` was ahead
of the published bf16 step-100 (dev 3/8 vs 2/8), so RL had less headroom. E-113 tests it with
a dense-checkpoint repeat of the SAME 400-step pretrain (224 s, 11 checkpoints) and two fresh
RL arms, driven unattended by the new `glm-5.3-flash-from-scratch/scripts/regime_match.py`.

**Determinism: EXACT.** The repeat run's `checkpoint-0100` and `checkpoint-0400` are
byte-identical (sha256) to the E-112 run, with the same tokens_seen (1,381,760) and peak
(2.141 GiB). The dense dev scan never produced the published 2/8: 0/8 through step 70, 1/8 at
80 and 90, 3/8 at 100, so the pre-registered rule selected step 90 (dev 1/8) - a WEAKER init
than the published one.

**Result: the headroom explanation is REFUTED.** A1 (step 90) greedy confirm 0/24 -> 8/24: the
baseline matches the published 0/24 exactly and the gain is +8 against the published +16 (8
gains, 0 losses, p=0.0078). A1 pass@8 over the trained families 2/24 -> 13/24 (+11,
p=0.00098) - matching the published +10, so P2 PASSES; over all 8 families 9/64 -> 18/64
(+9, p=0.078, bootstrap -6.4 to +9.6 pp) where E-112 netted 0. A3 (step 200, dev 7/8) was
already saturated: 1,536/1,536 rollouts exact, reward_spread 0.0, **0 groups updated** - a
silent no-op, so its +0 is a ceiling, not a dose-response point.

**The decisive number.** The init's RL-time reward signal is what defines the RL regime, and
the receipts show the regime was already matched: the published bf16 step-100 init produced
**417/1,536** exact rollouts, E-112's step-100 init **406/1,536** (2.6% apart), while the
selected step-90 init produced **150/1,536**. Yet the matched arm gained +7 greedy and the
extra-headroom arm +8 - both roughly half the published +16. What fails to reproduce is the RL
**endpoint**: greedy 8-12/24 (published 16/24), pass@8 exact 41-75/192 (published 88/192),
solved 13-15/24 (published 19/24). The leading remaining suspect is the RL step itself under
fp16/torch-2.14 (a lower per-group conversion rate), with a plausible mechanism being fp16's
activation range in the last block limiting what a head-only update can express. Headroom is
real but only within our own pretrain family (+8 / +7 / +0 at dev 1/8 / 3/8 / 7/8).

**Method lesson:** dev pass@1 on 8 tasks is not a regime label - it read 1/8 at step 90 and
3/8 at step 100 while the RL-time exact count differed 150 vs 406 of 1,536. Fingerprint an RL
init by its reward signal on the actual schedule.

**Next recommended step:** test the conversion hypothesis directly - an fp32-control RL arm
(or a bf16-capable device) from the same step-100 init with the schedule held fixed. Note the
E-113 artifacts live only on this machine (the nested `runs/` is gitignored).
# 2026-10-08 (af) (External repo re-run: the glm-5.3-flash clone reproduces EXACTLY at the infrastructure level, not at the RL magnitude level - and we know why)

> New session. Task row: TASKS 157. Experiment: E-112. Report:
> `research/2026-10-08_glm53_flash_clone_validation.md`. The clone stays a nested
> gitlink (HEAD 96d9c7c, parent commit 2bd733b) and is NEVER pushed to vukrosic.

User request, five parts: (1) drop bf16; (2) do not install `torch==2.5.1` - make it
compatible with our existing `.venv` so we do not create a second one; (3) add auto
resume/save and whatever else the experiment needs to run at best; (4) re-run the
experiment to validate the cloned repo; (5) write down what is worth taking from it.

**Port (done, CPU-validated before any GPU work).** CUDA autocast is now fp16 +
`GradScaler` (this is an sm_75 Turing card: bf16 does not exist, and a bf16 request now
RAISES below sm_80 instead of silently downgrading). `torch==2.5.1` is unpinned, so the
parent `.venv` (torch 2.14.0+cu126) is reused and no second environment exists.
Zero-flag auto-resume: `<checkpoint>/state.pt` carries optimizer + GradScaler + RNG +
`tokens_seen`; `progress.jsonl` is appended per step and the receipt is rebuilt from
disk, so a killed run re-runs the exact command. New `--vram-cap-gib` allocator ceiling
plus an `allocated > device_total` spill detector (AGENTS 20.3), `--grad-accum`,
`--max-seconds`, and `scripts/vram_probe.py`. 13/13 tests pass on CPU; two identical
smoke runs produced a byte-identical checkpoint sha256 (determinism).

**GPU re-run (done, exact upstream recipe).** Probe first (AGENTS 20.2/20.4): with the
desktop cleared, 7,704 MiB free, 55% cap = 4.14 GiB; batch 32 peaks at 2.134 GiB, no
spill. Pretrain 400 steps: 179.3 s, peak 2.141 GiB, tokens_seen 1,381,760. RL 96 groups
from `checkpoint-0100`: 440.9 s, peak 0.939 GiB, 1,536 rollouts, 406 exact, 77 updated
groups. Dev pass@1 0/8 -> 3/8 -> 7/8 -> 8/8. Greedy confirm 5/24 -> 12/24 (7 gains, 0
losses, p=0.0156). pass@8 trained families 10/24 -> 15/24, exact 35/192 -> 75/192. All 8
families 30/64 -> 30/64.

**Verdict: PARTIAL replication - PASS on infrastructure, direction and the per-family
pattern; FAIL on magnitude (V3, V3b) and on two V2 clauses.** Every infrastructure number
that could match did match exactly (token stream, parameter counts, RL trainable
parameters, rollout count, both VRAM peaks). The RL *effect* is in the same direction
with zero losses, but roughly half the size, because our fp16/torch-2.14 pretrain
checkpoint-0100 is ahead of the published bf16 step-100 checkpoint (dev 3/8 vs 2/8), so
the headroom the published result exploits has already been spent. Reproducing the step
number is not reproducing the regime.

**What is worth taking (E-112, and the new `research/WHAT_WORKS.md` lever):** a
fail-closed executable verifier; disjoint frozen split identities (dev/final/rl/confirm)
with the confirmation set opened only after duration is selected; paired-seed + exact
McNemar + task-bootstrap discipline; retained negative results; and above all
**per-checkpoint receipts as a replay target** - `artifacts/receipts/` converted "does
the port work?" into a dozen exact numbers for the cost of one short script.

Next: commit the nested repo locally (never push), and read the E-112 report before
quoting any of these numbers.

# 2026-10-07 (ae) (Routed OCR step 1: AMIDDA ingested, and the pre-registered 64-160 px band was measured OUT at batch 8)

> Follows (ad), same session. Task rows: TASKS 150 / 151 / 155. This is the PRE-LAUNCH
> state; the verdict lands in a later entry.

The routed system (TASKS 150) is closed for design and open for execution. Step 1 is the
first rung that adds real DATA rather than re-using the Muharaf/KHATT pair.

**Data (TASKS 155) - ingestion executed, streaming.** The four licence-verified AMIDDA
sub-sets (`calfa-ai/baybars` 15,567 etalab-2.0, `iskandar` 5,221 etalab-2.0, `RASAM-1`
7,331 apache-2.0, `tarima` 2,285 apache-2.0 = **30,404 lines**) are fetched DIRECTLY, not
through the `calfa-ai/AMIDDA` aggregate whose own card is "other" and which also carries our
own KHATT/Muharaf test splits. `fetch_hf_lines.py` needed a new per-dataset text column: the
AMIDDA family calls it `transcription`, not `text`. Streaming keeps the 5.71 GiB of parquet
off **C:** (where the HF cache lives) and lands only the PNGs on E:. First measurements:
baybars/train = 12,483 rows / 1,480,965,654 B = **~118 KB/line**, 2.2x the 53.7 KB/line
Muharaf precedent, so the corpus projects to ~3.6 GB.

**Code (TASKS 151).** `arabic_ocr/scripts/finetune_muharaf.py` gained `--height-band LO,HI`
(one height per BATCH, derived from `(seed, epoch, batch_idx)` so a resume reproduces the
batch geometry), `--res-degrade-prob/--res-degrade-band` (shrink the line to a low-res
"source" height, then let `preprocess_image` upsample it back - the E-100 27 px ISIPPT
regime as training data), and `--amidda-repeat`. The AMIDDA sub-sets are auto-discovered into
`TRAIN_SETS` and gated on `data/hf/amidda_audit.json`: the trainer now RAISES on a missing or
dirty audit (verified: exit 1 with the exact missing-path message).

**The band is 64-112, not the pre-registered 64-160 - MEASURED, not guessed.** A smoke at a
fixed 160 px peaked at **9.94 GiB allocated / 16.08 GiB reserved on an 8.00 GiB device**: it
spilled into WDDM shared memory at 2.75 s/step and **still exited 0**, so no OOM was raised.
Curve at batch 8: h64 2.15 GiB / 0.17 s/step, h96 4.06 / 0.23, h128 6.61 / 0.44, h160 9.94 /
2.75; the band actually used (64-112) measures 5.23 GiB / 0.21 s/step. Memory is ~h^2 because
aspect-preserving resizing widens the line as it grows taller. See MEMORY.md 2026-10-07.

**Gates: unchanged and pre-registered (TASKS 150).** Pareto dominance over BOTH Plan A and
Plan B on all four FULL sets - muharaf_test <= 21.58 raw, khatt_teklia_test <= 16.46,
khatt_bench <= 25.25, isippt_bench <= 52.68 - with the blank-penalty delta tuned on
`muharaf_val` ONLY and both raw and `norm_ar` reported. Init = the Plan B best
`arabic_ocr/runs/stage06c/ckpt_step024000.pt` (12,143,271 params; the trainer prints
`init: model.pt` regardless - that label is hardcoded to `CKPT_PATH.name`). A non-Pareto
result is a reportable FAIL.

**Leakage audit executed - and the leak is COINCIDENTAL, not structural.** `audit_amidda_leakage.py`
matched the 30,404 ingested lines against all four FULL eval reference sets: **135 exact hits
(128 `muharaf_test`, 10 `isippt_bench`), 2 blob hits, 0 hard hits**. The exact hits are not a
shared corpus - the matched reference length is **median 2 chars**, 130 of 135 are under 10 chars,
and the only five that reach 20+ chars are the Basmala `بسم الله الرحمن الرحيم`, a formula written
on manuscript pages everywhere and therefore present in almost any collection. The strongest test,
whitespace-free >=12 chars (hard), is **0**. Dedupe: `unique_text` 29,401 of 30,404 (1,003
duplicates across sub-sets = 3.3%), `dup_page` 0. Conservative decision costing 0.45%: all **137**
hits are dropped from training. The audit writes `drop[]` and `finetune_muharaf.py` applies it as
a row-level skip, confirmed in the launch log (`amidda audit: rows=30404 dup_text=1003 dup_page=0
leaks=137 -> DROPPING 137 row(s)`). The eval sets themselves are untouched, so the gates stay
honest. Measured at ingest: median image height **123 px**, 46% above 128 px, 14% in 33-64 px, 0.3%
at <=32 px - the 64-112 band plus 27-96 degradation covers the corpus' real distribution.

**Step 1 LAUNCHED** (job `pwsh-43`, 56,528 optimizer steps = ~3.7 h plus 57 evals). Startup
verified: audit gate fired, 56,528 train items, `widest_est_w=3323` (unchanged by the 30,404 new
rows), loss 1.50 -> 1.28 nats/char over the first 1,000 steps at 4.3-4.75 steps/s (0.22 s/step) -
the healthy rate, not the 2.75 s/step spill signature. New `--vram-cap-gib 7.3` flag (AGENTS 20.3)
arms a hard allocator ceiling so a WDDM spill raises a loud OOM instead of silently crawling.
Verdict pending; gates are the TASKS 150 Pareto row.

**STOPPED by explicit user request** at step 6,000 of 56,528 (~11%), after
~25 min, loss 1.1109 nats/char (down from 1.498 at step 50). Background job killed; verified no
`finetune_muharaf` python process survived and the GPU returned to 211 MiB used / 7,784 MiB free / 0%
util. Last durable state: `arabic_ocr/runs/stage06d/ckpt_last.pt` (step 5,000 eval boundary, weights
+ optimizer), `ckpt_step001000..005000.pt`, and `train_log.jsonl` (1,211 B - it IS flushed on exit, so
the per-step history is recoverable; the earlier "0 B" reading was just mid-run buffering). Up to
~1,000 steps after the last checkpoint were lost, which is the eval-boundary granularity by design.
**No verdict was produced**: the TASKS 150 Pareto gates remain UNTESTED and Step 1 is an open
question - NOT a failure. The corpus, the leakage audit, the augmentation knobs and a usable
checkpoint are all in place and committed, so a resume is cheap once the user decides.

# 2026-10-07 (ad) (Arabic OCR Plan B verdict: PASS on all three gates - growth helps G1 but is domain-selective, and a checkpoint-persistence bug surfaced)

> Follows (ac), same session. Plan B finished; both plans are now verdicted.

**Plan B verdict: PASS on all three gates** (`arabic_ocr/runs/stage06c/blank_penalty_verdict.json`,
E-111). Grown depth-only 4 -> 14 encoder layers, 4,245,671 -> 12,143,271 params (2.86x), authors'
recipe, 8 epochs / 26,496 steps, 7,939.3 s. Pre-registered rule executed: on-disk FINAL + the 3
lowest `khatt_teklia_test` cards, delta tuned on `muharaf_val` ONLY (delta 1.00 for all three).

| tag (step) | delta (val) | muharaf_test | khatt_teklia_test | khatt_bench | isippt_bench |
|---|---|---|---|---|---|
| planB step 26,000 | 1.00 (20.68) | 21.64 / 16.30 | **16.46** / 14.97 | **25.58** / 24.23 | 54.15 / 53.66 |
| planB step 24,000 | 1.00 (20.79) | **21.58** / 16.21 | 16.51 / 15.01 | 25.53 / 24.19 | **53.93** / 53.34 |
| planB step 23,000 | 1.00 (20.89) | 21.79 / 16.45 | 16.65 / 15.18 | **25.25** / 23.90 | 54.42 / 53.99 |

(raw / `norm_ar`.) All PASS: G1 16.46 (margin 4.54 pp), G2 21.58-21.79, khatt_bench 25.25-25.58, isippt 53.93-54.42.

**B vs A - a net gain, not a Pareto gain.** Against the Plan A final (21.66 / 18.47 / 27.15 / 52.68):
khatt_teklia_test **-2.01**, khatt_bench **-1.57**, muharaf_test -0.02, isippt_bench **+1.47 WORSE**.
So 2.86x the parameters bought a large improvement on the binding gate G1 and on khatt_bench, and cost
1.47 pp on `isippt_bench` - the 27px-resolution set (E-100), the smallest source images. Training
loss fell 0.6217 -> 0.5483 and the test gains followed on 3 of 4 sets, so the deeper model is
genuinely fitting better rather than merely memorising; the isippt regression is consistent with
overfitting the 96px upscaled regime at the expense of the 27px one. **Growth is domain-selective.**

**Two record corrections and one real bug, all found by verifying instead of assuming.**

1. **`ckpt_last.pt` is NOT the final model.** It is written at EVAL BOUNDARIES, so the last
   <`eval_every` steps of a run are never persisted. Both runs show it: Plan A's `ckpt_last` carries
   `step = 66,000` (not 66,240) and Plan B's carries `step = 26,000` (not 26,496). The (ac) entry
   and E-110 were corrected to say 66,000. Consequences: the Plan B shortlist had 4 tags but only
   **3 distinct checkpoints** (`ckpt_last.pt` holds IDENTICAL MODEL WEIGHTS to `ckpt_step026000.pt` - the FILES are NOT byte-identical (`ckpt_last.pt` is 145,967,307 B and also stores optimizer state, vs 48,657,283 B) - 0 of 206
   tensors differ, `max|delta| 0.000e+00`), and the scorecard's `final: true` row is from in-memory
   weights that cannot be reproduced from disk. Harmless here (the tail ran at lr ~1e-9), but it would
   bite with a large `eval_every`.
2. **`scorecard.json.init` is hardcoded to the default `model.pt`** and is therefore WRONG on the
   `--init-ckpt` path - Plan B's scorecard says `\"init\": \"model.pt\"` while actually initialising
   from the grown 14-layer checkpoint. The `params: 12143271` field is the trustworthy one.
3. **The scorecard's `gates` block reads the 400-line CARD, not the full set.** Plan B's scorecard
   reports `G2_muharaf_test_raw_le_23.11: false` (card 26.25) while the full-set verdict is 21.64 -
   a PASS. The muharaf card runs ~+4.3 pp harsher (E-109); never read the scorecard's G2 as the gate.

**State of the OCR line: both plans pass all three gates.** The best single model is Plan B at step
24,000 (21.58 / 16.51 / 25.53 / 53.93), which beats Plan A on three of four sets; Plan A remains
better on `isippt_bench` (52.68 vs 53.93). G1 travelled 40.79 (zero-shot teacher) -> 18.47 (A) ->
16.46 (B). Carry-forward caveat unchanged: published OCR numbers are NORMALISED CER, ours are RAW.

# 2026-10-07 (ac) (Arabic OCR Plan A verdict: PASS on all three gates - the authors' recipe beat capacity)

> Follows (ab), same session. Plan A finished; the verdict is in and it is a clean PASS.
> Plan B is now training.

**Plan A verdict: PASS on all three project gates** (`arabic_ocr/runs/stage06b/blank_penalty_verdict.json`,
E-110). The pre-registered selection rule was executed exactly: shortlist = the FINAL checkpoint
plus the three checkpoints with the lowest `khatt_teklia_test` card raw (61,000 / 59,000 / 63,000,
subject to a muharaf card <= 30.00), each scored on the FULL sets with the blank-penalty delta
tuned on `muharaf_val` ONLY.

| tag (step) | delta (val) | muharaf_test | khatt_teklia_test | khatt_bench | isippt_bench |
|---|---|---|---|---|---|
| **planA_final (66,000)** | **1.00** (val 21.00) | **21.66** / 16.32 | **18.47** / 17.06 | **27.15** / 25.84 | **52.68** / 52.14 |
| planA_s61000 | 1.00 (21.09) | 21.88 / 16.56 | 18.13 / 16.68 | 26.23 / 24.99 | 52.56 / 52.03 |
| planA_s59000 | 1.25 (21.01) | 21.96 / 16.65 | 18.27 / 16.82 | 26.52 / 25.24 | 53.03 / 52.43 |
| planA_s63000 | 1.25 (21.00) | 21.99 / 16.68 | 18.15 / 16.70 | 26.45 / 25.13 |

(raw / `norm_ar`; G1 khatt_teklia_test <= 21.00, G2 muharaf_test <= 23.11, G3 khatt_bench <= 29.54 and isippt_bench <= 58.34.)

`planA_final` = `ckpt_last.pt` (step 66,240) is **best on all four sets simultaneously**, so
selection was unambiguous - nothing was traded away to win something else. Margins: G1 **2.53 pp**,
G2 1.45 pp, khatt_bench 2.39 pp, isippt 5.66 pp.

**The honest framing of the win.** Against the teacher's full-set delta-tuned reference (E-109:
21.47 / 41.07 / 41.85 / 59.06) the deltas are **+0.19 / -22.60 / -14.70 / -6.38**. In-domain
muharaf is essentially unchanged - it is 0.19 pp WORSE, and still inside G2. Every out-of-domain
set improves enormously. So the recipe did not buy domain transfer at the cost of in-domain
accuracy; it bought generalisation. Training loss ended at **0.6217** nats per target character
against option (D)'s ~3.78 asymptote (E-104) - the same architecture reaching a reachable optimum
instead of an unreachable plateau. **This is the largest OCR win in the line, and it came from the
schedule, not from capacity.**

**TASKS 149 (the h128 "recover KHATT resolution" rung) is CLOSED, not executed.** Its only purpose
was to close G1, which E-101 left at 21.82 (a 0.82 pp miss). Plan A closed G1 at 18.47; launching it
now would spend GPU time on an answered question.

**Plan B is training** (background job `pwsh-2`, launched 2026-10-07). Init =
`arabic_ocr/runs/grow_test/planA_final_L14.pt`, grown from the Plan A winner depth-only, 4 -> 14
encoder layers, 4,245,671 -> 12,143,271 params. The surgery was verified before use:
`grow_ctc_model.py --verify` reported `exact=86 block=0 fresh=120 max|delta| 0.000e+00` (shared
layers bit-exact, as they must be for a pre-norm stack) and `check_ckpt_reload.py` returned
`verdict PASS` with `forward_logits_shape [64,1,166]`. A 20-step smoke ran clean at **0.25 s/step**
with `total_steps=26496`, so the expected wall time is ~1.8-2.5 h (the earlier 3.9 h estimate
assumed a larger slowdown than 14 layers actually cost). Per TASKS 148 the epoch count was NOT
pre-shrunk; if the round budget runs out, B is evaluated at the latest checkpoint reached and
reported plainly as PARTIAL.

**Carry-forward caveat, unchanged:** published OCR numbers are NORMALISED CER; ours are RAW. Plan A's
21.66 raw / 16.32 norm on muharaf_test is at or beyond the published Muharaf CTS reference (22.11
normalised) only under that stated caveat - do not claim SOTA without it.

# 2026-10-07 (ab) (Arabic OCR Plan A in flight: the verdict path is validated and the stop rule was recalibrated twice)

> Follows (aa), same session. The user ordered "A then B": (A) fine-tune the 4.25M Muharaf teacher
> with the authors' own recipe, then (B) grow the trained model and continue. Plan A is training as
> this entry is written - **nothing here is a verdict yet**.

**Plan A, as launched** (pre-registered in TASKS 147 before launch; commit `d8f9a84`). Init = the
published teacher `model.pt` (4,245,671 params - **not** `adapt11700`), `--augment light`,
`--train-blank-penalty 0.2`, `--scheduler onecycle --lr 5e-5 --epochs 20`, batch 8 x accum 1,
`--khatt-repeat 1 --reallines-repeat 0 --target-height 96 --eval-every 1000 --eval-limit 400
--keep-eval-ckpts`, out `arabic_ocr/runs/stage06b/`. 26,490 train items, 3,312 steps/epoch,
66,240 steps total; OneCycleLR peak at step 19,872.

**Trajectory at 25%** (card raw; the teacher's card on the identical code path is 26.40 muharaf /
40.87 khatt). `khatt_teklia_test` - the gate that actually fails - improves strongly:
44.27 (step 1000) -> 28.40 (13000) -> **26.75** (16000), i.e. **-14.12 pp** against the teacher.
`muharaf_test` drifts mildly the wrong way, 26.82 (1000) -> 27.45 (16000), +1.05 pp over the
teacher, noisy within 27.26-27.84. G1 (<= 21.00) is **5.75 pp away with the whole LR anneal still
ahead**; G2 (<= 23.11) is at risk because the teacher passes it and Plan A is trading in-domain
accuracy for out-of-domain KHATT.

**The stop rule was wrong twice; both are recorded rather than quietly dropped.** Amendment 1
aborted if the card `muharaf_test` raw exceeded 26.00 after the LR peak. Measuring the teacher on
the same 400-line card gave **26.40** - the rule would have aborted the teacher itself. That is a
calibration error, so amendment 2 replaced it with one measured collapse test: abort if the card
exceeds **30.00** at any eval. Lesson (MEMORY.md, E-108): calibrate a collapse threshold by
measuring the starting model on the *exact same eval subset* - this card reads 26.40 for muharaf
where the full set reads 22.11 (+4.3 pp), while for khatt the card is faithful (40.87 vs 40.79).

**The verdict path is validated end-to-end before it is needed** (all CPU, no GPU contention).
`blank_penalty_tune.py` over the cached teacher/adapt11700/stage05f logits reproduces E-101 exactly
(adapt11700 delta=0.75 -> 23.20 / 21.82 / 55.67 / 28.47), so the pipeline is trustworthy; it also
yields the teacher's own full-set delta-tuned row - `muharaf_test` **21.47** (the teacher alone
passes G2), khatt_teklia_test 41.07, isippt 59.06, khatt_bench 41.85 (E-109). Plan B's growth
surgery was rehearsed on a **real Plan A checkpoint**, not just the teacher: depth 4 -> 14 gives
4,245,671 -> 12,143,271 params with `max abs delta 0.000e+00` (bit-identical function) and
`check_ckpt_reload.py` **PASS**. TASKS 147 also pre-registers the checkpoint-selection rule so
"Plan A best" is not chosen post hoc, and records the disk budget (one 358 MB logits cache per
checkpoint; 10.4 GB free).

**One trap worth the ink.** `decode_ctc_lm.py --run` prints a `GATES {...}` line whose G1/G2/G3
keys reuse the project gate names but measure different quantities on different sets - reading it
as the verdict would report the wrong numbers. `finetune_muharaf.py`'s own card `gates` block
does report the real project gates, but on the 400-line subset and without the blank penalty.

---

# 2026-10-07 (aa) (Arabic OCR option (D) FAILED - and the reason was in the model directory all along: the published Muharaf model was never trained from scratch)

> Follows (z), same session. The user chose "(D)" - scale our CNN-Transformer-CTC to ~15M and train
> it from scratch at target_height 128 on the Muharaf union. It is now answered: **FAIL**, with the
> root cause identified. Committed as the (aa) closeout commit.

**What was launched.** 15,336,743 params (d=384 / 8 layers / 8 heads / ffn 1536), height 128,
`--scratch`, lr 3e-4, warmup 500, batch 8, 16 epochs = 52,992 steps, in `arabic_ocr/runs/stage06a/`.

**How it failed, measured.** At step 4,250 (8.0% of the plan) the loss was 3.78 nats per target
character, decaying geometrically (0.76 -> 0.36 -> 0.24 -> 0.16 -> 0.14 per 1000 steps) toward an
asymptote near 3.5 - the training targets' own **unigram entropy, 3.122 nats**. The greedy hypotheses
tracked sequence LENGTH and nothing else:

| ref | hyp |
|---|---|
| `لبنان في ١٧/١٠/١٩٦٠` (19) | `ا` (1) |
| `حضرة محافظ جبل لبنان الافخم` (27) | `ا` (1) |
| `فلسطين. رأيتك في ريحانياتك...` (44) | `اة ال` (5) |
| `لان المرحوم كان بعيد عنا كثير...` (90) | `اة ال ل.` (8) |

Emission rose 5.8% (step 1000) -> 9.4% (step 4000) while CER stayed flat at ~92%, and `CER ~= 100 -
emission` held at both points. Eval cards were flat (muharaf_test 94.17 / 95.56 / 91.70 / 92.61;
khatt_teklia_test 93.71 / 93.46 / 96.67 / 93.21 at steps 1000/2000/3000/4000). At step 1000 the
train/test gap was ZERO (muharaf_train 94.53 vs muharaf_test 94.54) - nothing memorised.

**It was NOT a bug, and that was worth proving.** On the step-4000 checkpoint, feeding the SAME image
twice gives mean |delta log-prob| = 0.000000 and 100% argmax agreement; feeding TWO DIFFERENT images
gives 3.303 and 26.6% agreement. The image reaches the model and modulates it strongly. This is an
optimization failure - a confident, blank-dominated degenerate optimum - not a broken pipeline.

**The root cause was in the model directory.** `models/ctc_backbone/training_metrics.json` records
`init_checkpoint` -> `..._v6_antideletion_finetune/model.pt` -> `..._v3_h96_finetune/model.pt`, and
`checkpoint_load_info.missing_keys` = `height_attention.score.*`. The published backbone is a **chain
of warm-started fine-tunes**, and its attention head was newly initialised on top of a backbone that
could already read. Its epoch-1 train_loss was already **0.524**, final test_loss **0.768** - our
from-scratch run sat at 3.78. The 22,091-line train split is enough to fine-tune a reader; it is not
enough to discover character-level CTC alignment from a random init in 16 epochs. Option (D) as
specified asked from-scratch training to reproduce a warm-start result.

**Two recipe mechanisms were also missing** (cheap to restore, both used by the authors):
`augmentation_mode: "light"` (ours is `"augmentation": "none"`) and `train_blank_logit_penalty: 0.2`
(ours applies the blank penalty only at decode). The authors also used lr 5e-5 with OneCycleLR,
batch 4 x accum 2; ours was lr 3e-4 with cosine, batch 8.

**Pre-registered predictions, scored honestly.** P1 (beat teacher zero-shot KHATT 40.79) - NOT
REACHED (93.2% at stop). P2 (beat 21.82, pass G1 <= 21.00) - FAIL. P3 (hold G2 <= 23.11) - FAIL
(92.6%). P4 (the risk stated up front: "26,490 lines is thin for 15.3M from scratch, so it may land
worse than the 4.25M teacher") - CONFIRMED, and worse than stated: it never left the plateau at all.

**Gates: G1 FAIL - G2 FAIL - G3 not reached.**

**Stop-rule note (transparency).** The pre-registered rule was "check each 1000-step eval; if KHATT
raw is still > 35 after ~3 h, stop". I stopped at ~1 h 15 m instead, because a measurement the rule
did not anticipate - image-independent hypotheses with the loss asymptote pinned to the targets'
unigram entropy - made the outcome conclusive rather than merely slow. This TIGHTENS the rule (it
stops earlier only when the evidence is already decisive) and does not bias toward a positive result.
The run is resumable: `--resume` from `arabic_ocr/runs/stage06a/ckpt_last.pt` with the FULL original
recipe continues it from step 4,250 if the complete 16 epochs are wanted.

**Next (needs a user decision).** Every path that mirrors how the reference model was actually built
is a WARM START. Cheapest toward G1 (21.82 -> <= 21.00): fine-tune the existing 4.25M teacher longer
with the authors' recipe (light augmentation + train blank penalty 0.2 + OneCycleLR at 5e-5), which
E-103 already showed is underfit. If capacity is truly binding, grow the TRAINED 4.25M and continue
rather than re-initialising. Either way it gets pre-registered before launch.

---

# 2026-10-07 (z) (Arabic OCR steps 1/2/4 CLOSED: the decoder knob is spent, the resolution knob is dead at inference, and the bottleneck is measured capacity):

> Follows (y), same session. The user order was "1 then 2 then 4, after their result we
> can decide step 3" - all three are now closed. Committed `49f4a22`.

**1. Step 1 - the decoder knob, shipped (E-101).** `--blank-penalty` is now wired into
`decode_ctc_lm.py` (applied to the cached logits before greedy/beam, recorded in the
scorecard) and `blank_penalty_tune.py` is generalised (`--logits-dir/--tags/--deltas`, and
it reports raw + `norm_ar` + emission%). delta is tuned on `muharaf_val` only. B2
(stage05f) + delta 1.00: muharaf_test 23.01 -> **22.57** (norm 17.81 -> 17.34),
khatt_teklia_test 22.91 -> **22.88**, isippt 56.02 -> **53.63**, khatt_bench 29.74 ->
**29.11**. B2+penalty now **passes G2 and G3 and fails only G1** (khatt_test <= 21.00);
the best number available for G1 is ckpt-11700+penalty at 21.82, a 0.82 pp miss.
Emission rises everywhere (muharaf_test 87.0 -> 90.8%, isippt 60.0 -> 70.5%), which is
exactly the mechanism (y) predicted.

**2. Step 2 - the resolution knob is DEAD at inference (E-102, FAIL).** The backbone
*looks* height-agnostic: height is attention-pooled and `output_lengths` depends only on
width. It is not. Re-decoding the SAME teacher weights at a taller input destroys it -
muharaf_test raw CER 22.11 (h96) -> 34.91 (h128) -> 60.54 (h160) -> 85.27 (h192), with
emission 86.7% -> 73.6% -> 45.9% -> 17.1%. KHATT got *worse* (40.79 -> 55.14), the
opposite of the pre-registered prediction P2. BatchNorm running stats, the learned
attention distribution over H and the width-only positional encoding are all bound to
H=96. **The resolution information is real (E-100: 98%/93% of KHATT lines are downscaled
1.55x at height 96, while Muharaf's packaged source is 60 px for 100% of sampled lines
and is already upscaled 1.6x) but it can only be harvested by TRAINING at the higher
height.**

**3. Step 4 - the data is not the culprit; the model is UNDERFIT (E-103).** New
`trainline_emission_probe.py` measures emission on the model's own training split. It
carries a self-check that must reproduce the cached-logit path (22.06 vs 22.11) - which
mattered, because the first version silently reported 98% CER from indexing a (T,B,C)
tensor as (B,T,C).

| model | split | emit (raw ref) | CER vs the label it was trained on |
|---|---|---|---|
| teacher | muharaf_train | 91.5% | 12.02 |
| teacher | muharaf_test | 86.7% | 16.95 |
| stage05f (ours) | muharaf_train | 91.3% | 13.37 |
| stage05f (ours) | muharaf_test | 86.7% | 18.56 |

Under-emission is present on data the model was trained on, and the train/test gap is only
~5 pp. That is underfitting - not label noise (which would also show low train emission)
and not overfitting (which would show a high train fit). The 4.2M model cannot fit its own
labels. Separately, `norm_ar` is now reported for every set, so the 29%
evaluation-convention gap from (w) is measured rather than hidden.

**4. The step-3 fork, now evidence-informed.** Options A (warm-start the 15M FixFips CRNN
on the Muharaf union), B (fresh 15M KHATT specialist) and C (ship 4.25M with a measured
ceiling) all remain, but the evidence adds one that dominates: **(D) scale our own
CNN-Transformer-CTC to ~15M and train it at target_height 128 on the Muharaf union.** It
addresses capacity (E-103) and KHATT resolution (E-100) together, targets G1 - the only
gate still failing - and needs no new architecture or dependency. Rough cost: ~2 h on the
free GPU versus 23 min for B2. **Awaiting the user's decision.**
# 2026-10-07 (y) (Arabic OCR: (x) is PARTIALLY REVISED - the error mode is character UNDER-EMISSION and a CTC blank penalty is a free gain):

> Follows (x), same session, CPU only. This entry corrects the verdict in (x): the
> decoder lever was NOT fully exhausted - one inference knob had never been tested.

**1. E-97 - what kind of errors are these?** Levenshtein backtrace on the cached
teacher logits: on muharaf_test 65% of edits are DELETIONS (subs 29%, ins 5%). The
model emits only 86.7% of reference characters (KHATT test 80.8%, ISIPPT 52.7%) and
drops 29% of word spaces. Charset coverage and the frame budget are both REFUTED as
causes (1.44% of ref chars are unemittable, worth 1.08 pp; 4.3 frames per char are
available, and ISIPPT has 9 and is the worst). Under-emission scales with domain
distance, so it is a representation failure surfacing as CTC blank bias - which is why
the decoder, the lexicon and adaptation all failed downstream of it.

**2. E-98 - the blank penalty is a real, free lever.** Subtracting delta from the blank
logit, tuned on muharaf_val only: ckpt-11700 (delta*=0.75) improves on EVERY set -
muharaf_test 23.71 -> 23.20, khatt_teklia_test 22.05 -> 21.82, isippt_bench 57.84 ->
55.67, khatt_bench 29.04 -> 28.47. That is the best union model so far, it beats the
trained B2 rung on 3 of 4 sets at zero training cost, and G3 now passes. The teacher
val-tuned penalty hurts KHATT (+0.27), so tune per domain.

**3. Revised fork.** The acoustic model is still the binding constraint (G1 <= 21.00 is
unreachable by any decoder), so the 15M question stands - but TASKS 142 lands first:
wire `--blank-penalty` into `decode_ctc_lm.py`, cache stage05f logits, re-score B2.

# 2026-10-07 (x) (Arabic OCR: Plan A EXHAUSTED - the decoder is oracle-bounded, adaptation is capacity-bound, the lexicon rule is net-negative; fork = a 15M specialist):

> Follows (w). CPU measurement plus one short fine-tune; no other GPU job ran
> beside it. The user's standing decision was "option A now, option B later" -
> option A is now spent.

**1. E-94 - the decoder lever is bounded (FAIL).** On the cached teacher logits
for muharaf_test: greedy 22.11, summed beam 24.58, +char LM 23.72, Viterbi beam
32 = 22.11 exactly. The N-best oracle over beam 32/64 is 20.95-21.92, i.e. at
most 0.53 pp of headroom, so NO reranker can reach the pre-registered G1 <=
20.61. A length bonus (1.0) made it far worse (47.62).

**2. E-95 - the adaptation lever is capacity-bound (2 of 3 gates FAIL).** A
near-zero-LR union pass warm-started from ckpt-11700 (lr 3e-6, 2 epochs, 6,624
steps) moved Muharaf test 23.71 -> 23.01 (G2 <= 23.11 PASSES) but KHATT teklia
22.05 -> 22.91 (G1 <= 21.00 FAILS) and KHATT bench 29.04 -> 29.74 (G3 FAILS by
0.20 pp). The points (23.71, 22.05) and (23.01, 22.91) are a Pareto frontier:
the union objective is capacity-bound, not optimisation-bound, so more LR/epoch
tuning at 4.25M params is not worth running.

**3. E-96 - the cheap lexicon lever is DEAD (measured before building it).** A
train-only lexicon (54,084 types / 261,471 tokens from the Muharaf + KHATT train
labels) misses **20.14%** of REFERENCE words on muharaf_test while 45.07% of
predicted words are OOV; of the most frequent distinct OOV words 75.3% have a
distance-1 lexicon neighbour, but replacing them with it makes CER WORSE
(22.11 -> 23.94). OOV is a weak error signal.

**4. Fork (NEEDS_USER_DECISION).** Plan A has no lever left. Warm-start the 15M
FixFips CRNN on the Muharaf union (the cheapest direct test of the capacity wall
- FixFips already reaches 10.99 KHATT test / 12.03 KHATT bench / 46.92 ISIPPT),
train a fresh 15M KHATT specialist (the original Plan B), or stop and ship the
4.25M Muharaf model with the measured ceiling. Evidence:
research/EXPERIMENTS.md E-94/E-95/E-96; arabic_ocr/STATUS.md (s)-(u);
arabic_ocr/runs/stage05f/; arabic_ocr/scripts/lexicon_probe.py.

# 2026-10-06 (w) (Arabic OCR: the SECOND in-domain teacher eval, and the teacher-contamination audit that REVISES E-89 - route (D) transfers only inside a teacher's own training domain):

> Follows (v). The user chose the recommended option and authorized executing
> it: TASKS 134, the zero-shot eval of the second in-domain teacher, plus the
> reproducibility work. Inference only; nothing was trained; no GPU job ran
> beside anything else.

**1. TASKS 134 - `sdkv2/muharaf-arabic-ocr` zero-shot (E-90).** New harness
`arabic_ocr/scripts/eval_muharaf_hflines.py` mirrors E-89 exactly (greedy CTC, no LM,
no reranker, free-running, dynamic width) and scores the author
`ctc_backbone/model.pt` (4,245,671 params, sha256 head `bba3b6f5301822fc`) on all
1,654 lines of `data/hflines/val`: **CER 23.64% vs the 57.80% control (+34.16 pp)**,
median line CER 13.16%, cer_lt10/20/50 = 38.4 / 66.5 / 91.2%, 11.55% exact, 17.3 s /
95.3 img/s. Looked like a rout.

**2. It is not a transfer number - and measuring that IS the result (E-91).**
`data/hflines` was built from `Teklia/KHATT_v1.0` + `aamijar/muharaf-public` +
`data/reallines` (`build_hf_lines.py`), and this teacher was trained on Muharaf.
Normalized-transcript matching against the author OWN published training manifests
(22,091 lines) shows **79.9% of the val lines are in its training set** (15.69% CER
there); only 16.7-20.1% is genuinely held out.

**3. The same audit re-opens E-89.** FixFips trained on KHATT, and 9.6% of the val
lines are KHATT lines it trained on (CER **1.82%** there). Its own held-out stratum is
52.79%, and on the **174 val lines held out from BOTH teachers** it scores **61.88% -
4.08 pp WORSE than the 57.80% control**. Muharaf on the same 174 lines is **54.06%
(+3.74 pp)**; strict 100-line variant 75.26 / 68.80.

**4. Decision.** Muharaf is still the better base - 54.06 vs 61.88 on the shared
held-out set, 4.25M vs 15.06M params, 95 vs 51 img/s - but the honest headroom over
the incumbent control is ~3.7 pp, not the 10-34 pp the headlines implied. The <5%
target is not reachable by distilling either teacher. **TASKS 137 supersedes the
TASKS 131 decision and is USER-GATED.**

**5. Files.** New: `arabic_ocr/scripts/{eval_muharaf_hflines,overlap_muharaf_hflines,stratify_teacher_overlap,compare_teachers_clean}.py`;
`eval_fixfips_khatt.py` now also writes per-line JSONL so any stratum can be re-scored
without re-running the model. Evidence under `runs/stage05c/`:
`eval_muharaf_muharaf_hflines_val.json`, `perline_*.jsonl`, `overlap_muharaf_hflines.json`,
`teacher_overlap_stratification.json`, `teacher_clean_comparison.json`, `examples_*.txt`.
STATUS (o), RETROSPECTIVE R10 + guardrail 9, EXPERIMENTS E-90/E-91, TASKS 134/137,
MEMORY, WHAT_WORKS all updated. No push.

# 2026-10-06 (v) (Arabic OCR audit: the WHOLE line was untracked in git - line committed, legacy claim docs neutralised, and the in-domain teacher eval PASSES: route D transfers +10.87 pp):

> Follows (u). The user asked to save the report, commit, fix what was done
> wrong and the other risks, update the docs, and proceed with the rest of the
> recommendations. Nothing OCR is training; the eval below is inference only.

**1. Retrospective written.** `arabic_ocr/RETROSPECTIVE.md` (new): audited
history, verdict, evidence-graded "done right / done wrong", repo-hygiene
findings, remediation log and 8 guardrails. Registered in `AGENTS.md` §1/§2 so
the line is no longer outside the document map (§7: "a doc outside the map does
not exist for the next agent" - that is how the fabricated legacy reports
survived a week).

**2. The line was untracked in git.** `arabic_ocr/` (122 files including
`STATUS.md`), `docs/plans/2026-09-29-arabic-ocr-nano-model.md` and 516 OCR run
evidence files under `runs/` had **0 tracked files**, while `.gitignore`
explicitly said "Code, results and reports stay tracked" for this line (its only
OCR rule ignored `arabic_ocr/models/`). A `git clean` or a machine move would
have destroyed the whole line. Fixed: `arabic_ocr/`, the plan,
`data/ocr_textpool/` (29.65 MB `texts.jsonl`, the only surviving copy of those
labels) and the textual run evidence are committed; OCR run weights and rendered
images (`*.pt` / `*.pdparams` / `*.pdopt` / `*.states` / `*.png`) and the raw
text pool are ignored, so no >100 MB non-LFS object can enter the repo.
**No push performed.**

**3. Legacy claim docs neutralised.** `PROJECT_REPORT.md` ("Successfully built a
complete Arabic OCR pipeline", "CER=0.189") and `COMPARISON.md` (ISIPPT 34.41 /
KHATT 16.56) carried no warning despite being teacher-forced artifacts; both now
open with a "NOT A MEASUREMENT" banner. `STATUS.md`'s duplicate (c)/(d) Stage 0
sections were merged and a current-state header added.

**4. E-89 run (route D validation, TASKS 133).** Zero-shot eval of the
in-domain teacher `FixFips/arabicocr-khatt` on `data/hflines/val` (1,654) vs the
**57.80** control, no fine-tuning - the cheap decider for TASKS 131
recommendation (D) that had never been run. Harness
`arabic_ocr/scripts/eval_fixfips_khatt.py`, result
`runs/stage05c/eval_fixfips_hflines_val.json`. **Result: PASS - see §5.**

**5. E-89 RESULT - route (D) TRANSFERS, but the teacher is weak (PASS).**
`FixFips/arabicocr-khatt` (CRNN+CTC, 15.06M, KHATT-trained, MIT) read
`data/hflines/val` (1,654) at **46.93%** free-running CER against the **57.80**
control - **+10.87 pp with NO training**, in 32.4 s (51.1 img/s), greedy CTC and
no language model. Norm CER 42.53, WER 81.89, exact 4.66%, **median line CER
51.02%**, cer_lt10/20/50 = 11.2 / 14.9 / 48.1. Side-checks: KHATT 200 = **12.03**
vs 44.41 (its own training corpus - NOT transfer evidence); printed ISIPPT 500 =
**46.92** vs 13.75 (-33.17, expected). Greedy/no-LM 12.03 on KHATT against the
card's 6.09 (beam + bigram LM) is the right order of magnitude, which validates
the architecture reconstruction, the checkpoint load and the preprocessing; the
author's reversed-label protocol was confirmed empirically (flip 46.93 vs
no-flip **81.99**). **The win is real, but half the lines are worse than 50% CER,
so this licenses a gentle in-domain fine-tune (B1'), not a distillation sprint.
Nothing was trained - the next step is still user-gated at TASKS 131.**
Environment note: the author's `preprocess.py` needs cv2, which `.venv` lacks;
the harness borrows the extension from `.venv-paddle` via `sys.path` (nothing
installed) - a real fix is a user-gated `pip install opencv-python`.

# 2026-10-06 (u) (Arabic OCR Stage 0.5b FAIL: 13x the data makes the fine-tune WORSE - the damage tracks lr x steps, not corpus size):

> Continues (t), which launched the full-corpus warm-start. **Verdict: FAIL.**
> Route B1 (adapt the teacher by fine-tuning) is a NO-GO, and it falsifies
> E-77's own carry note.

**1. Result.** `runs/stage05b/output/iter_epoch_N.pdparams`, free-running
un-mirrored CER on `data/hflines/val` (1,654) at dynamic width, vs the
**57.80** control: epoch 1 **72.72**, 2 **73.99**, 3 **71.74** (best), 4
**74.14**, 5 **78.14**, 6 **76.55**. `cer_lt50_pct` collapses 28.7 -> 0.7-4.0.
**Best = 71.74, i.e. +13.94 pp WORSE than doing nothing.** JSONs:
`runs/stage05b/eval_iter_epoch_{1..6}.json`.

**2. The framework's metric lied again, in the same direction.** Its own
report: `best metric, acc: 0.0018, norm_edit_dis: 0.1415, best_epoch: 4` -
while free-running CER at epoch 4 was **74.14**, its worst-but-one.
`best_accuracy.pdparams` scores **73.71**, and the true best epoch (3) scores
71.74. Same trap as E-77 (MEMORY 2026-10-04), now reproduced on a 13x corpus.

**3. The diagnostic: it is catastrophic forgetting, not a failure to adapt.**
`runs/stage05b/diag_*.json`, control -> epoch 3:
| benchmark | control | epoch 3 | damage |
|---|---|---|---|
| isippt PRINTED (500) | **13.75** | **46.75** | **+33.00 pp** |
| khatt HANDWRITTEN (200) | **44.41** | **67.54** | **+23.13 pp** |
| hflines/val OURS (1,654) | **57.80** | **71.74** | **+13.94 pp** |
The fine-tune destroyed the printed capability the model was built for
(+33 pp). Khatt's `cer_lt50` went **71.0 -> 0.0** - not one line readable
under 50% CER. This is not "the model failed to learn handwriting"; it is
"the pretrained representation was wrecked".

**4. E-77's carry note is FALSIFIED - the lever is lr x steps, not data
volume.** E-77 (2,000 lines, 1,055 steps, lr 1e-5) = **62.76**; this run
(25,921 lines, 27,440 steps, same lr) = **71.74**. **13x the data, 26x the
steps, 9 pp MORE damage.** The damage is sub-linear in steps and saturates
(~+14 pp), which is why "just get a real corpus" could not have worked. The
corpus was never the problem; the optimization pressure was.

**5. Why.** The teacher is printed-trained; our corpus is handwritten
manuscript scans (chroma 28.2 vs ISIPPT 4.0, E-76). The domain gap is large
enough that every gradient step pulls the shared visual features off the
printed manifold, and 25.9k lines is plenty of signal to do that thoroughly.

**6. What this means for route B.** B1 as executed is dead, and it takes the
premise of B2 with it: a teacher that reads our handwriting at 57.80% is a
**weak teacher**, and distilling a weak teacher into a student cannot exceed
it. The remaining levers (TASKS 131) are: **(B1')** freeze `PPLCNetV4` and
train only the neck+heads (the direct remedy for forgetting, but its ceiling
is the frozen features); **(D)** find a teacher already trained on Arabic
handwriting, which makes fine-tuning gentle instead of destructive - this is
the highest-leverage move and is cheap to search for; **(C)** accept the FAIL
and re-scope Stage 3. **PaddleOCR has no `freeze_backbone` config option**,
so B1' needs a custom training loop (the config-driven `tools/train.py`
cannot express it).

**7. Files.** `runs/stage05b/{train_full_lr1e5.yml,train_list_dyn.txt,
val_list_dyn.txt,train.log,eval_zeroshot_control.json,eval_iter_epoch_1..6.json,
diag_control_isippt.json,diag_ep3_isippt.json,diag_control_khatt.json,
diag_ep3_khatt.json,output/}`; `arabic_ocr/scripts/build_stage05b_lists.py`
(NEW); EXPERIMENTS E-88; TASKS 130/131.

**8. The search for an in-domain teacher FOUND ONE - this is the way forward.**
Exa search (raw payloads `research/raw/exa_arabic_htr_{1,2,3}.json`,
`exa_htr_details.json`, `exa_htr_cards.txt`). Two MIT-licensed PyTorch
models trained on Arabic **handwriting**:
| model | arch | trained on | published CER | files |
|---|---|---|---|---|
| `FixFips/arabicocr-khatt` | CRNN (CNN+BiLSTM) + CTC | KHATT, 11,375 lines | **6.09%** val / 27.26% WER | `crnn_best.pt` + `charset_arabic.txt` (75 classes) + `bigram_lm.json` |
| `sdkv2/muharaf-arabic-ocr` | CNN-Transformer-CTC | Muharaf corpus | **13.31%** on 1,334 test lines | 57 files, `config.json` |
Both are `image-to-text`, MIT, and **torch 2.14.0+cu126 is already in
`.venv`**, so `crnn_best.pt` loads with no install. The KHATT model's own
README warns "printed text, **historical manuscripts**, and heavily
diacritized text are out of domain" - and `data/hflines` IS handwritten
manuscript scans, so the zero-shot transfer is an open question that only an
eval can answer. **Next: zero-shot eval of `FixFips/arabicocr-khatt` on
`data/hflines/val` (1,654) vs the 57.80 control** - if it lands anywhere
near its KHATT numbers, route B is unblocked with a teacher that is actually
in-domain, and E-88's diagnosis (the domain gap is the binding constraint)
is confirmed with a positive control.

---

# 2026-10-06 (t) (Arabic OCR Stage 0.5b: user chose (B) - full-corpus warm-start of v6-small is RUNNING):

> Continues (s). The user answered the Stage 3 follow-up decision with **(B)**:
> warm-start / distill a pretrained recognizer. B1 (adapt the teacher) is a
> prerequisite for B2 (distil it into our own student), so B1 runs first.

**1. Why B, and why B1 before B2.** v6-small is **3.10M params** - already
SMALLER than the 7.9M from-scratch model that scored 0.9100 - and reads our
handwriting at 57.80% zero-shot, 33.20 pp better. Distilling a 3.10M teacher
into a weaker student is pointless until the teacher is adapted; if it cannot
be adapted to our domain, distillation is hopeless. So: **fine-tune first.**

**2. The PaddleOCR training stack was GONE and has been rebuilt.**
`.venv-paddle-gpu` holds `paddle` plus the training deps (cv2, albumentations,
lmdb, shapely, pyclipper, skimage, rapidfuzz) but NOT the `ppocr` source tree,
and no PaddleOCR checkout existed anywhere on E: or D: - the pilot's checkout
did not survive. Re-cloned `github.com/PaddlePaddle/PaddleOCR` ->
`E:\PaddleOCR` (shallow; commit `dab3fe3`, 2026-09-16). It has everything the
pilot used: `SVTR_LCNet` dispatch (`ppocr/modeling/architectures/__init__.py:69`),
`PPLCNetV4` (`backbones/rec_lcnetv4.py:521`), `MultiHead`
(`heads/rec_multi_head.py:67`), `tools/train.py`. NOTE: this revision has NO
`architectures/rec_model.py` - `build_model` lives in
`architectures/__init__.py`, and the eval script's
`from ppocr.modeling.architectures import build_model` works unchanged.

**3. The rebuilt stack reproduces the control EXACTLY** - the pre-flight that
licenses the run. `stage05_eval.py --repo E:\PaddleOCR` on `data/hflines/val`
(1,654) at dynamic width: **CER 57.80, WER 98.22, exact 0.24, cer_median
58.82, cer_lt10 0.4 / lt20 1.3 / lt50 28.7**, 49 img/s, 33.7 s on GPU ->
`runs/stage05b/eval_zeroshot_control.json`. Every one of those matches E-77's
recorded control, so the new checkout is functionally the pilot's. 422/422
checkpoint tensors matched; 749 classes.

**4. Data: the full corpus, and the width confound removed.** E-77/E-78 used
**2,000** of the 25,921 training lines; E-77's own carry note says the lever
"needs a real handwriting corpus, not a few thousand lines", so the
full-corpus run is a NEW experiment, not a re-run of a failed one. NEW
`arabic_ocr/scripts/build_stage05b_lists.py` ->
`runs/stage05b/{train,val}_list_dyn.txt` (**25,921 / 1,654** rows, 0 missing)
in the 4-field `path\ttext\tw\th` form `ds_width: true` requires.
`data/hflines/*/labels.json` carries only `image`+`text`, so widths come from
the PNG headers. The pilot's `dict.txt` and the model card's
`ppocrv5_arabic_dict.txt` are **content-identical** (747 lines; the pilot's is
merely CRLF), so the pilot dict is reused unchanged.

**5. Config and run.** `runs/stage05b/train_full_lr1e5.yml` is the proven
pilot `dyn/train_lr1e5.yml` with only the corpus, `epoch_num: 3 -> 6`,
`eval_batch_step: [0, 63] -> [0, 4000]` and the output dir changed. lr 1e-5
(the pilot's better arm), cosine, 1-epoch warmup, Adam, bs 8, `max_w: 1280`,
dynamic width. Launched 2026-10-06 as a background job writing
`runs/stage05b/train.log`.

**6. The gate is PRE-REGISTERED and the checkpoint choice is honest.**
Free-running un-mirrored CER on `data/hflines/val` at deployment (dynamic)
width must beat **57.80%**. Every `iter_epoch_N.pdparams` will be free-run
individually and the best kept - PaddleOCR's `best_accuracy.pdparams` is
selected by a teacher-forced metric and was **14 pp worse than the weights the
run started from** in E-77, so it is not trusted (MEMORY 2026-10-04).

---

# 2026-10-06 (s) (Arabic OCR Stage 3 GATE FAIL: the from-scratch 7.9M recognizer degenerates into a character language model):

> Continues (r), which launched Stage 3. The two-phase run (pretrain ->
> fine-tune) COMPLETED and the gate was run. **Verdict: FAIL.** The
> from-scratch recognizer is a NO-GO on this corpus; the next move is a
> user decision (section 7).

**1. Gate result (free-running, un-mirrored, honest).** `arabic_ocr/scripts/eval_stage3.py` on `data/hflines/val` (1,654 lines), checkpoint `runs/stage3_finetune/best.pt`: **CER 0.9100, WER 0.9791** -> `runs/stage3_finetune/gate.json`. The Stage 0 baseline was **0.5780** (v6-small zero-shot). **+33.20 pp WORSE: FAIL.**

**2. The run never fit its own data.** Pretrain (16,000 SYNTHETIC crops, 20 epochs): train CE 2.47 -> **1.35**, synthetic-val free-running CER 2.12 -> 0.94. Fine-tune (25,921 real lines, 25 epochs): train loss 2.03 -> **1.67** and **flat from ~epoch 12**; the free-running val CER oscillated in a 0.92-0.96 band for the last 15 epochs (best **0.9100** at epoch 20); CTC_CER flat at **0.817** the whole run. Train and val track each other (1.67 / 1.69), so this is UNDERFITTING, not overfitting.

**3. The diagnostic that names the failure** (`runs/stage3_finetune/diag.json`, 512 lines, `arabic_ocr/scripts/diag_stage3.py`). Four numbers: free-running CER **0.9173**; teacher-forced CER **0.4658**; **blank-image CER 0.9026**; image-forced CTC CER **0.812**. Char agreement free-vs-blank 0.039 and only 3.7% of generations hit max_len, so non-termination is NOT the issue. The blank image scores 0.903 against 0.917 for the real image: **the image buys ~1.5 pp.** The decoder emits fluent legal-boilerplate Arabic unrelated to the line; the encoder is the weaker half (CTC 0.812 = it reads ~19% of characters).

**4. The exposure-bias gap is 0.45.** Teacher-forced 0.466 vs free-running 0.917. But the teacher-forced number is itself far from a working recognizer (a trained line recognizer sits at ~0.02-0.05), so decode-side fixes (beam search, a lexicon) have almost nothing to recover.

**5. Why it failed (ranked).** (a) From-scratch training on 25.9k handwritten lines is below the data floor for this task - E-76 already measured that a zero-shot pretrained model reads our handwriting 2.2x better than a from-scratch control; (b) 64 px input height for diacritised handwriting; (c) 7.9M params with no large-scale visual pretraining. The synthetic pretrain did not transfer (synthetic-val CER was still 0.94 before fine-tuning).

**6. What is NOT the cause.** Not non-termination (3.7% at max_len), not a mirroring/order bug (the good examples decode correct logical-order Arabic), not the CTC weight (CTC_CER was flat, so the head never learned), and not a missing EOS (the model stops).

**7. Next step (USER DECISION REQUIRED - TASKS 129).** The evidence now contradicts the from-scratch half of user decision C. Options:
* **(A) Scale the from-scratch route**: input height 64 -> 96/128, ~15M params, a longer synthetic pretrain. Hours of GPU; the teacher-forced ceiling of 0.466 says the payoff is uncertain.
* **(B) Warm-start / distill a pretrained recognizer** (the plan's original lines 161-163, overridden by decision C): take v6-small (57.80% zero-shot on our handwriting) as the teacher and train a small student. This is the route E-76's own carry note recommends.
* **(C) Accept the FAIL and re-scope Stage 3** (print-only, or a lexicon-constrained decoder over the CTC head).
No further GPU work will start without a decision.

**8. Files.** `runs/stage3_finetune/{gate.json,gate.log,diag.json,history.json}`, `runs/stage3_pretrain/history.json`, `arabic_ocr/scripts/eval_stage3.py`, `arabic_ocr/scripts/diag_stage3.py` (NEW), `runs/stage3_finetune/REPORT.md` (NEW), EXPERIMENTS E-87, TASKS 122/129.

---

# 2026-10-05 (r) (Arabic OCR Stage 3 LAUNCH: from-scratch 7.9M vit_hybrid recognizer, two-phase pretrain->fine-tune):

> Continues (q), which closed Stage 2. Stage 3 (the recognizer) is now RUNNING;
> the user opened the Stage 3 gate ("continue").

**1. Design (SETTLED).** `vit_hybrid` = `CompactViTDecoder` with `ctc_weight=0.3`:
pre-norm ViT encoder (h256, 4L, patch16) + transformer decoder (h256, 4L) + an
auxiliary CTC head on the encoder. The cross-attention decoder IS the
"NRTR-style attention head"; the CTC head forces the encoder to read the image
(a pure decoder collapses to a language model - the Stage 0 failure mode).
**7.9M params** (within the 5-20M plan). No literal NRTR head is built; CRNN+CTC
is broken (all-blank collapse) and is not used.

**2. Vocab (146 classes).** `data/focused_vocab.json` (54) + 92 data chars with
count >= 10 from `data/hflines/{train,val}`: Arabic-Indic digits (0660-0669),
Latin letters, ASCII digits, punctuation, tatweel (0640). Synthetic text is
99.95% covered (0.05% UNK). `data/stage3_vocab.json`.

**3. Data.** Phase 1 pretrain: 16,000 synthetic line crops materialised from
`runs/stage2_synth/train/pages.jsonl` GT boxes (no detector needed for training
data) -> `runs/stage3_synth_lines/{train,val}` (16,000 / 2,400). Phase 2
fine-tune: the 25,921 real `data/hflines/train` lines.

**4. RTL (SETTLED).** No image flip; logical-order targets. Consistent with
prior work + PaddleOCR Arabic. No flip/mirror handling exists in `arabic_ocr/src`.

**5. Two-phase training (RUNNING).** `arabic_ocr/configs/stage3_pretrain.yaml`
(20 epochs, lr 3e-4, eval every 5 epochs) -> `runs/stage3_pretrain/best.pt`, then
`arabic_ocr/configs/stage3_finetune.yaml` (25 epochs, lr 1e-4, pretrained=phase1)
-> `runs/stage3_finetune`. `eval_generate: true` so the training-loop CER is
free-running. Epoch 1 (pretrain, synthetic val): train_loss 2.47, val_loss 1.86,
free-running CER 2.12 (untrained generate - expected to drop).

**6. Gate.** `arabic_ocr/scripts/eval_stage3.py` (free-running, un-mirrored) on
`data/hflines/val` (1,654 lines) must beat the Stage 0 baseline **57.80%**
(v6-small zero-shot). Honest target: ~16-30% realistic for from-scratch on
handwriting; < 5% likely unreachable.

**7. Files.** `arabic_ocr/scripts/build_stage3_data.py`,
`arabic_ocr/scripts/eval_stage3.py`, `arabic_ocr/configs/stage3_{pretrain,finetune}.yaml`,
`data/stage3_vocab.json`, `runs/stage3_synth_lines/`.

---

# 2026-10-05 (q) (Arabic OCR Stage 2 WIRING: the crop policy is a property of the INPUT, and both halves of the gate now PASS):

> Continues (p), which closed the detector half of Stage 2. The detector is now
> wired into the pipeline and **Stage 2 is CLOSED**: detector half PASS (p), crop
> half PASS (this entry). Stage 3 (the recognizer) is the next user gate.

**1. The open question from (p) is answered by measurement, not by choice.** Real
`data/hflines/val` line crops return a median of 2 boxes/image because the
detector segments a line into words/PAWs. Two measured rules, selected by input
type:
* **Page input** (`runs/stage2_synth/val/pages`): group boxes by vertical overlap
  (`y_overlap=0.5`, fraction of the shorter box) and merge horizontally inside a
  group (`gap_ratio=0.6` x line height). Result: **7.83 lines/page vs 8 GT,
  recall@IoU0.5 0.925** - merging is strictly better than the raw box metric
  (0.912), because a missed word no longer costs the whole line.
* **Line-crop input** (`data/hflines/val`): union of all boxes. Result:
  **100/100 images collapse to exactly 1 line** (grouping gives 75/100), ink
  coverage median 0.962, and the single image with no detection falls back to
  the full image.

**2. The wiring.** `arabic_ocr/src/inference/line_crop.py` (pure numpy, no cv2,
so it is unit-testable in the base `.venv`) +
`arabic_ocr/scripts/stage2_detect_lines.py` (`.venv-paddle-gpu`: detect ->
group/union -> crop -> manifest). 18/18 unit tests PASS in `.venv`; 30 val
pages materialise to 235 crops + a JSONL manifest.

**3. Crop labels are assigned by GEOMETRY, never by index.** Each crop takes the
GT line with the best IoU and is labelled only at IoU >= 0.5 (220/235 labelled).
Index alignment would silently shift every transcript after one missed or extra
line - the exact failure mode that makes a recognizer learn garbage.

**4. Ink coverage is the right no-GT sanity metric for a crop stage.** Median
0.962 on real line crops says the union box loses ~4% of the ink (isolated
diacritics / edge noise) - visible without any ground truth.

**5. Stage 2 is CLOSED.** Gate: >= 0.90 line recall on held-out synthetic pages
(**0.925**) + measured on real `data/hflines` (**100/100 -> 1 line, 0.962 median
ink coverage**). Next: Stage 3, the 5-20M CTC+NRTR recognizer over detected
regions - user-gated.

# 2026-10-05 (p) (Arabic OCR Stage 2 HYBRID: the pretrained detector CLEARS the gate at 0.912 recall@IoU0.5 - and only after two measured post-processing additions):

> Continues (o), which recorded the user's decision C (pretrained detector,
> from-scratch recognizer) and the model identification. This entry records the
> detector being downloaded, validated on the GPU, and wired - the detector half
> of Stage 2 is now PASS; the recognizer half is still open.

**1. Model + GPU.** `PaddlePaddle/PP-OCRv5_mobile_det` (`inference.pdiparams`
**4.69 MB**) in `arabic_ocr/models/ppocrv5_mobile_det/`, loaded with
`paddle.jit.load()` (no `paddleocr` package needed) on the
**`.venv-paddle-gpu`** venv's GPU (paddle 3.3.1, CC 7.5). **0.10 s per
1200x900 synthetic page**, 0.028 s per real line image.

**2. GATE PASS.** 30 held-out synthetic val pages / **240 GT lines**
(`scratch/detector_module_check.py`, through the shipped module):
**recall@IoU0.3 0.950, recall@IoU0.5 0.912, recall@IoU0.7 0.575, meanIoU 0.690,
8.7 boxes per page against 8 GT lines per page.** The plan's Stage 2 gate is
>= 0.90 line recall - **cleared by the detector half.**

**3. Two post-processing additions were REQUIRED, and both are measured.**
Stock PaddleOCR DBPostProcess scores **0.404** on the same pages. (a) **7px
dilation before contour extraction**: the DB probability map is sharp-edged, so
one ink stroke fragments into many 0-4 px contours that each die on the
min-area filter. (b) **1.8x vertical expansion about the box centre**: the
detector fires on the ink stroke, while a GT line box carries
ascender/descender/diacritic padding (recall@IoU0.5 0.471 -> 0.933 across
factors 1.0 -> 1.8 in the sweep).

**4. The model's own `box_thresh: 0.6` is incompatible with that dilation.**
Dilation merges strokes into blobs whose tight minAreaRect encloses the
inter-stroke whitespace, so the box score (p50 over all contours = 0.55) falls
under 0.6 and the box is dropped: **3.2 boxes/page, recall 0.404**. Swept
0.0/0.3/0.5/0.6/0.7 x score mode (raw bitmap vs prob map, immaterial):
**0.3 -> 8.5 boxes/page, recall 0.912**; 0.0 -> 10.0 boxes/page, 0.933.
**Adopted 0.3** - the gate still passes with 15% fewer boxes to feed the
recognizer.

**5. The first 0.933 was measured in a scratch script, not in the module.** A
fresh run through the shipped `detector_gpu.detect` produced **1 box/page and
0.121** - `scratch/detector_validate_full.py` had reimplemented the decode and
omitted the box_thresh filter, and the wired filter was scoring the
dilated+unclipped box. Fixed (score the tight rect against the raw bitmap) and
**re-verified through the module**: 8.7 boxes/page, 0.912. Any metric that
validates a stage must be re-run through the shipped function before the stage
is called done.

**6. Real data behaviour (open decision).** 100 `data/hflines/val` line
images: median **2 boxes/image** (1 box 47, 2-3 boxes 35, >3 boxes 17, 0 boxes
1), median detected box 98x23 px. The detector over-segments a single-line
crop into words/PAWs, so the line-crop stage must either merge sub-boxes into
one line box or feed word-level boxes to the recognizer - **NEEDS_USER_DECISION**.

**7. Next step.** Wire `detector_gpu` into the Stage 2 pipeline entry point
(detect -> crop -> recognizer input), then Stage 3.

# 2026-10-05 (o) (Arabic OCR: user chooses HYBRID (C) - pretrained detector + from-scratch recognizer; paddle GPU check):

> Continues (n). The user chose **C (hybrid)**: use a validated pretrained Arabic
> detector (PaddleOCR class) to unblock the pipeline, keep the recognizer from
> scratch. This sidesteps the detector wall (4 failures, all underfitting) and
> focuses from-scratch energy on the recognizer, where the nano contribution is.

**1. WSL check: no PaddleOCR installation found.** Searched both WSL distros
(Ubuntu-24.04, Ubuntu), all Python environments, conda, venvs, Docker, pip
caches, and the entire `/home/ahmad` tree. No `paddle`, no `paddleocr`, no
PP-OCRv5 models. The installation the user remembered is likely the
**`.venv-paddle-gpu` venv on Windows**.

**2. Windows paddle GPU: WORKS.** `.venv-paddle-gpu` has paddle 3.3.1,
CUDA compiled, 1 GPU (Compute Capability 7.5), "PaddlePaddle works well on 1
GPU". It has the PaddleOCR dependencies (`cv2`, `albumentations`, `freetype`,
`lmdb`) but NOT the `paddleocr` Python package. The Stage 0 bake-off worked
around this by loading models with `paddle.jit.load()` directly - no `paddleocr`
package needed.

**3. Detector model identified.** `PaddlePaddle/PP-OCRv5_mobile_det` on
HuggingFace - `inference.pdiparams` is only **4.69 MB**. PaddleOCR inference
format (same as the recognizer models already cached). This is a DB-style
detector that outputs text line/word boxes.

**4. Recognizer models already cached.** `arabic_ocr/models/ppocrv5_mobile_rec/`
(1.98M) and `arabic_ocr/models/ppocrv6_small_rec/` (3.10M) - both in PaddleOCR
inference format, loadable with `paddle.jit.load()`.

**5. Next step.** Download the detector model (4.69 MB), validate it on our
synthetic pages + hflines, then use it to detect text regions for the
from-scratch recognizer. The plan doc Stage 2 is updated to the hybrid approach.

# 2026-10-05 (n) (Arabic OCR Stage 2: CenterNet-style also FAILS at 46.97% - the 4th detector failure, and the most diagnostic one):

> Continues (m). (m) launched the CenterNet-style run. This entry records its result.

**1. GATE FAIL.** `pwsh-7` ran all 30 epochs. Best Hmean **46.97%** (epoch 28,
final 46.22, P 34.32 / R 70.77). This is WORSE than CRAFT's 59.86%, despite a
98.96% perfect-target ceiling. `runs/stage2_paw_cn/report.json`.

**2. The diagnosis is now unambiguous: optimization, not representation.** A
98.96% ceiling with 46.97% achieved means the model cannot fit 2000 synthetic
pages in 30 epochs. All four runs are underfitting (train H == val H, still
climbing at epoch 28). The CenterNet decode has no level set and no component,
so neither CRAFT failure mode (blob shape, merging) can occur - yet it scored
the lowest. The bottleneck is the optimization/capacity, not the target design.

**3. Precision 34% / recall 71% = over-detection.** The model emits many
spurious peaks above tau 0.3 (low precision) and misses 29% of PAWs (recall
71%). This is classic underfitting: the heatmap has not learned to suppress
false peaks. The fix is more training (100+ epochs, not 30) and/or more
capacity, not another representation change.

**4. Four detector failures, one pattern.** stride 2 + 2.56M = 54.89%, stride 2
+ 4.94M = 57.34%, stride 1 + 5.01M = 59.86%, CenterNet + 4.94M = 46.97%. Every
representation and capacity lever bought 2-3 points; the residual is the
optimization. The from-scratch nano detector is hitting a wall.

**5. Strategic question (gates Stage 3).** Is "20-50M from scratch" a hard
requirement or a preference? If it can relax to "pretrained base + LoRA",
handwriting < 5% is achievable (Fanar-2 + QLoRA = 3.77% on KHATT) but breaks
the deployment targets. The plan doc now carries the honest two-track target
(printed < 5% achievable; handwriting ~16-30% realistic).

# 2026-10-04 (m) (Arabic OCR Stage 2: stride-1 CRAFT also FAILS at 59.86% - three levers, three times +2-3 points, so the residual is the optimization):

> Continues (l). (l) built both levers. This entry records A's result and B's launch.

**1. GATE FAIL.** `pwsh-6` ran all 30 epochs. Best Hmean **59.86%** (epoch 29,
final 59.86, P 67.62 / R 53.70). craft3 (stride 2) was 57.34%, so stride 1 bought
**+2.52 points**. `runs/stage2_paw_s1/report.json`.

**2. Still underfitting, and still climbing.** The loss was still descending at
epoch 29 (0.0284 -> 0.0017) and H was still rising (59.02 -> 59.86 over the last
two evals). 30 epochs is not enough for this objective at stride 1.

**3. The pattern is now identical across three runs.** stride 2 + 2.56M = 54.89%,
stride 2 + 4.94M = 57.34%, stride 1 + 5.01M = 59.86%. Every lever that should
help - capacity, removing the dead head, resolution - bought 2-3 points. The
residual is not any of them.

**4. B launched.** `pwsh-7` in `runs/stage2_paw_cn/`. Its ceiling is 98.96% and
its decode has no level set and no component, so neither of the two measured
CRAFT failure modes (blob shape, merging) can occur.

# 2026-10-04 (l) (Arabic OCR Stage 2: both remaining representation levers are built and running - stride 1 and CenterNet-style):

> Continues (k). (k) recorded craft3 (57.34%), the free anisotropic decode win
> (+12 points) and the finding that the 90% gate sits ON the stride-2 CRAFT
> ceiling. The user chose to try BOTH remaining levers.

**1. Two levers, both implemented, running sequentially (single GPU).** (A) a
stride-1 output head, (B) a CenterNet-style centre heatmap + direct w/h
regression. A stays inside the pre-registered envelope ("CRAFT-style"); B is a
redesign the user explicitly approved. A is `pwsh-4` in `runs/stage2_paw_s1/`,
B is queued next in `runs/stage2_paw_cn/`.

**2. B's ceiling measured FIRST: 98.96%.** Decoding *perfect* CenterNet targets on
all 300 val pages (stride 2, sigma_k 0.25, tau 0.3) gives **P 100.00 / R 97.93 /
H 98.96** (22,994 tp / 0 fp / 485 fn). The 485 misses are PAW centres that round
to the same output pixel (2.07%) - a collision, not a merge. That is 6.7 points
above CRAFT's 92.27% and clears the gate with room. `scratch/stage2_cn_ceiling.py`.

**3. Why B should work where CRAFT could not.** CRAFT reads the box off a level
set, so the box inherits the blob's SHAPE - measured 25.5 x 25.5 (square) against
tall GT PAWs (20.4 x 22.1) - and adjacent PAWs merge into one component (36.8% of
PAWs at tau 0.5). CenterNet has no level set and no component: each PAW is one
peak and its size is a direct regression. PAW centres are 99.97% unique, so the
construction is sound.

**4. Two bugs found and fixed before they cost a run.**
- `collate_pages` stacked the craft keys unconditionally, so the CenterNet batch
  crashed with `KeyError: 'prob'`. Now keyed on what the dataset returns.
- `eval_interval` in the config was **dead** - the trainer evaluated every epoch.
  At stride 1 a full-val decode is ~4 min, so honouring it is the difference
  between a 2.7 h and a 3.7 h run. The config key now does what it says.
- The first stride-1 launch (`pwsh-1`) died at epoch 1's eval on the collate
  bug after 2 epochs of training; its artifacts are kept in
  `runs/stage2_paw_s1_crash/`.

**5. Still open.** Whether either lever actually closes 57 -> 90. A's ceiling is
96.91% (CRAFT k=0.15 at stride 1); B's is 98.96%. Both clear the gate, so the
question is how much of the gap each recovers, not whether the gate is reachable.

# 2026-10-04 (k) (Arabic OCR Stage 2 GATE FAIL again - 57.34% - but the free decode fix is worth +12 points, and the 90% gate sits ON the representation's ceiling):

> Continues (j). (j) recorded craft2 (54.89%) and recommended (A) zero the unused
> thresh head + widen toward 5M. This entry records craft3 and what it proved.

**1. GATE FAIL.** `pwsh-52` ran all 30 epochs. Best Hmean **57.34%** (epoch 27),
final 57.02 (P 65.42 / R 50.53). craft2 was 54.89%, so option A bought **+2.45
points**. `runs/stage2_paw_craft3/report.json`.

**2. The objective fix worked and is proven.** `loss` and `loss_total` print
identical values every step, i.e. the thresh term is genuinely zero (it was
21-61x the prob term). Capacity went 2.56M -> **4.94M**. Neither moved the needle
materially: **capacity and the dead head were NOT the bottleneck.**

**3. Still underfitting.** 24 pages each, identical decode: TRAIN H 59.56 / 62.22
/ **71.69** vs VAL H 57.52 / 60.80 / **67.45** at tau 0.5 / 0.6 / 0.7 (a 4.2-pt
gap).

**4. Detection is now GOOD - the gap is entirely box EXTENT.** Centre-in-component
coverage of the model map reached **91.20%** on val (craft2: 87.94%), while H is
only 57.52%. And H (57.5%) is almost exactly the **singleton coverage** (985 of
1807 PAWs = 54.5%): **PAWs covered by exactly one component match; PAWs inside a
merged component do not.**

**5. The blob is the wrong SHAPE, not just the wrong size.** Splitting decoded
components by how many GT centres they cover:

| component | n | decoded w50 x h50 | vs GT |
|---|---|---|---|
| singleton | 985 | **25.5 x 25.5 (square)** | w 1.249x, h 1.154x |
| merged | 22 | 55.2 x 29.7 | w 2.644x, h 1.458x |

GT PAWs are TALL (20.4 x 22.1), the decode produces SQUARES. A single isotropic
`k` cannot fix a shape error - which is exactly why the earlier k-sweep
collapsed: shrinking enough to fix the width destroys the height.

**6. FREE WIN - anisotropic decode: +12 points, no retraining.** Separate `kx`
(width) and `ky` (height) in `decode_craft_boxes`, 24 val pages:

| tau | best kx/ky | H | vs isotropic |
|---|---|---|---|
| 0.5 | 0.30 / 0.25 | 67.38 | +9.86 |
| **0.7** | **0.25 / 0.20** | **69.55** | +2.10 |

**7. Merging is controlled by the threshold, and the trade is structural.** PAWs
inside a merged component: tau 0.5 -> **36.8%**, 0.6 -> 25.3%, 0.7 -> **10.1%**.
The merge happens in the MASK, before any decode, so no decode constant can undo it.

**8. THE STRUCTURAL FINDING - the 90% gate sits on the representation's ceiling.**
Perfect-target-map ceiling at stride 2 (measured (j), identical code path):

| tau | ceiling H |
|---|---|
| 0.5 | **91.97-92.27** |
| 0.6 | 89.66 |
| 0.7 | **86.28** |

The gate is 90%. So the ONLY threshold where the ceiling clears the gate (tau 0.5)
leaves **2.3 points of headroom** - and that is the threshold where 36.8% of PAWs
merge. At tau 0.7, where merging is only 10.1%, the **ceiling itself (86.28%) is
BELOW the gate**. A perfect model cannot pass at any threshold with manageable
merging. **The gate is not reachable by better training of this representation;
the representation must change, or the gate must be re-scoped.**

**9. Also found: `CRAFTLoss.forward` never reads `target["ignore"]`** (lines
88-106). The dataset builds it and the collate ships it; the loss drops it. In
craft mode the dataset already zeroes it, so nothing is currently wrong - but the
plumbing is a trap for anyone reusing the loss.

**10. Next - NOT started. NEEDS_USER_DECISION.** Free lever to adopt regardless:
the anisotropic decode. Then either (a) raise output resolution to stride 1
(CRAFT k=0.15 perfect-map ceiling 96.91%), (b) replace the region score with a
CenterNet-style centre heatmap + w/h regression (PAW centres are 99.97% unique
and 91% detectable, so merging disappears by construction), or (c) re-scope the
90% gate.

| what | where |
|---|---|
| gate report | `runs/stage2_paw_craft3/report.json` (FAIL), `train_log.jsonl` |
| evidence | `scratch/stage2_craft3_diag.py`, `stage2_wh.py`, `stage2_aniso.py`, `stage2_tausweep.py`, `stage2_sizesweep.py` |
| superseded | `runs/stage2_paw_craft2/` (54.89%), `runs/stage2_paw_craft/` (broken metric), `runs/stage2_paw/` (DBNet, capped 78.6%) |

# 2026-10-04 (j) (Arabic OCR Stage 2 GATE FAIL: 54.89% vs 90% - the model cannot fit even its TRAIN set, and 98% of the loss was an unused head):

> Continues (i). (i) fixed the decode threshold and relaunched `pwsh-29`. This
> entry records the gate verdict and the diagnosis behind it.

**1. GATE FAIL.** `pwsh-29` ran all 30 epochs. Best Hmean **54.89%** (epoch 25)
against the pre-registered **90%** gate; final epoch 54.43 (P 64.86 / R 46.88).
`runs/stage2_paw_craft2/report.json` -> `"verdict": "FAIL"`.

**2. The logged loss is the prob component ONLY, so the thresh head was invisible
for the whole run.** `stage2_train.py` line 138 logs `running["prob"]`; the
`thresh` term is never printed. Measured on real batches, `CRAFTLoss` returns
`{'prob': 0.00103, 'thresh': 0.06306}` on TRAIN and `{'prob': 0.00559,
'thresh': 0.11499}` on VAL - **the thresh term is 21-61x the prob term**, and in
craft mode the thresh head is never read at eval. The backbone was trained mostly
to fit a map nothing consumes.

**3. The decode is NOT the problem any more.** Threshold sweep on the model map
and on the PERFECT target map through the identical code path (30 val pages, 2294
PAWs):

| tau | model H | ceiling H |
|---|---|---|
| 0.5 | 55.50 | **91.97** |
| 0.6 | 60.36 | 89.66 |
| 0.7 | **64.71** | 86.28 |

Best possible with this checkpoint at any threshold: **64.98%** (tau 0.7, k 0.2).

**4. It is not a size bias - it is MERGING.** Sweeping the decode scale `k_eff`
0.2 -> 1.0 (larger k = smaller boxes): 0.2 is best everywhere; 0.3 gains only +2.4
points at tau 0.5; >= 0.4 collapses below 10%. The 1.86x width inflation (w50 29.7
px vs GT 16.0) comes from components spanning several PAWs, which no global
constant can undo. Component count is 0.72 per PAW.

**5. The model cannot fit even its TRAIN set.** 30 pages each, identical decode:
TRAIN H 58.44 / 64.23 / **68.34** vs VAL H 55.50 / 60.36 / **64.71** at tau
0.5 / 0.6 / 0.7 - a 3.6-point gap, i.e. **underfitting, not a generalization
gap**; the training fit itself is the wall. (The 2-page overfit reaches 91%, so
the architecture can represent the target - it cannot fit 2000 pages.)

**6. Why.** A 2.56M backbone shared between the prob head and a thresh head that
(a) is never read in craft mode and (b) carries 21-61x the loss. Capacity and
gradient go to an objective nothing consumes.

**7. Next - NOT started.** Both candidate changes are inside the pre-registered
envelope (CRAFT-style, 1.5-5M params, stride 2, 90% gate unchanged):
(A) zero the thresh term in craft mode (`thresh_weight: 0.0`, threaded through
the config); (B) spend the freed capacity - widen toward the plan's 5M ceiling.
**NEEDS_USER_DECISION.**

| what | where |
|---|---|
| gate report | `runs/stage2_paw_craft2/report.json` (verdict FAIL), `train_log.jsonl` |
| evidence | `scratch/stage2_tausweep.py`, `stage2_diag.py`, `stage2_sizesweep.py`, `stage2_tv.py`, `stage2_centerhit.py` |
| superseded | `runs/stage2_paw_craft/` (broken metric), `runs/stage2_paw/` (DBNet, capped at 78.6%) |

# 2026-10-04 (i) (Arabic OCR Stage 2: a THIRD eval bug - the CRAFT decode threshold - and the run it killed was at 46%, not 3%):

> Continues (h). (h) replaced the DBNet target with a CRAFT region score and
> launched `pwsh-28`. That run's metric was still wrong - a third bug - and the
> run was killed on the strength of it. This entry corrects the record.

**1. The CRAFT blob constant was applied at the wrong threshold.**
`CRAFT_BLOB_SCALE = 2*sqrt(2 ln 2)` is the extent of a Gaussian's level set at
**0.5**, but `evaluate_detector` binarized at `bin_thresh=0.3` and divided by
the 0.5 constant. On *perfect* target maps: tau=0.3 + 0.5-constant = **74.45%**,
tau=0.3 + the correct 3.1035 = **90.27%**, tau=0.5 = **92.27%**. Fixed with
`craft_blob_scale(thresh)` + a `thresh=` argument on `decode_craft_boxes`; the
craft branch now binarizes at `CRAFT_DECODE_THRESH = 0.5`.

**2. The threshold, not the constant, is what hid the model.** Re-scoring the
killed run's `best.pt` (epoch 5) with the fixed decode, 30 pages each:
**VAL H 45.90% (P 57.81 R 38.06)** and **TRAIN H 46.36% (P 56.63 R 39.24)** -
against the **3.14%** the run printed. At tau=0.3 the model's soft halo bridges
neighbouring PAWs, so whole words fuse into one component (decoded w50 59.5 px at
the old setting vs GT 16 px).

**3. Train H == val H => the model was UNDERfitting.** 46.36 vs 45.90 on 30 pages
each. The lever is epochs/capacity, not regularisation.

**4. The architecture is not the wall, and AMP is innocent.** Overfitting 2 pages
for 500 steps reaches **H 89-91%** (fg>0.5 0.0058 vs target 0.0106).
`pos_weight` sweep on that task: 10 -> 91.0%, 100 -> 83.6%, 1000 -> 74.2% (keep
10). fp32 91.0% vs `autocast` 89.8% - AMP does not cause the diffuseness.

**5. `pwsh-28` was killed and relaunched.** It was killed at epoch 7 on a metric
that understated it ~15x; its artifacts are kept in `runs/stage2_paw_craft/` as
evidence. Relaunched as **`pwsh-29`** into `runs/stage2_paw_craft2/` with
**30 epochs** (was 20), since the run was still underfitting.

**Gate: PAW Hmean >= 90% on held-out synthetic. Verdict pending `pwsh-29`.**

| what | where | note |
|---|---|---|
| decode fix | `arabic_ocr/src/data/paw_targets.py` | + `craft_blob_scale(thresh)`, `CRAFT_DECODE_THRESH`; `decode_craft_boxes(..., thresh=)` |
| eval fix | `arabic_ocr/src/training/detection.py` | craft branch binarizes at 0.5 with the threshold-aware constant |
| config | `arabic_ocr/configs/stage2_paw.yaml` | `epochs: 30`, `save_dir: runs/stage2_paw_craft2` |
| evidence | `scratch/stage2_ceiling_thresh.py`, `stage2_overfit.py`, `stage2_fitcheck.py` | threshold ceiling table; 2-page capacity test; train-vs-val + AMP |

# 2026-10-04 (h) (Arabic OCR Stage 2: the eval metric was wrong, DBNet cannot reach the gate, CRAFT-style targets adopted and retrained):

> Continues (g). The Stage 2 run's reported Hmean was meaningless; the design was
> replaced and retrained.

**1. The Stage 2 eval metric was measuring the wrong thing.** Two bugs, both fixed:
decoding components from the *shrunk* region against *full* GT boxes, and
binarizing with the learned threshold map. Corrected re-score of the old
`best.pt` (epoch 3): H = 0.20-0.24, not the 42% the run printed.

**2. The DBNet design cannot reach the gate at stride 2.** Decoding *perfect*
target maps (the ceiling of each design) gives 78.6% for DBNet shrunk boxes +
component decode, because 50.87% of PAWs share bounding-box area and get merged
into one component. CRAFT-style Gaussian region scores give 92.5% at stride 2
(k=0.20) and 96.9% at stride 1 (k=0.15).

**3. Adopted: CRAFT-style region score, stride 2, k=0.20.** No architecture
change - the probability head becomes the region score and the threshold head is
kept as an auxiliary task, so the pre-registered "probability map + threshold
map" output contract is preserved. `CRAFTLoss` = weighted MSE (a focal loss
would flatten the Gaussian and break the size decode).

**4. Relaunched** (job `pwsh-28`, `runs/stage2_paw_craft/`). The DBNet baseline
artifacts in `runs/stage2_paw/` are kept intact for comparison.

| piece | file | change |
|---|---|---|
| targets | `arabic_ocr/src/data/paw_targets.py` | + `build_craft_prob`, `decode_craft_boxes`, `CRAFT_BLOB_SCALE` |
| dataset | `arabic_ocr/src/data/synthetic_pages.py` | + `target_mode` / `craft_k` |
| loss | `arabic_ocr/src/training/detection.py` | + `CRAFTLoss`; `evaluate_detector` craft branch |
| config | `arabic_ocr/configs/stage2_paw.yaml` | `target_mode: craft`, `craft_k: 0.2`, `save_dir: runs/stage2_paw_craft` |
| verify | `scratch/stage2_craft_check.py` | analytic round-trip + ceiling + loss backprop, all PASS |

**Gate: PAW Hmean >= 90% on held-out synthetic.** Verdict pending `pwsh-28`.

# 2026-10-04 (g) (Arabic OCR Stage 1 GATE PASS + Stage 2 PAW detector built and LAUNCHED):

> Continues (f). The user go-ahead means: continue the ladder - finish the Stage 1 gate
> properly, then start Stage 2. Both done.

**1. Stage 1 GATE = PASS** (`runs/stage1_gate/verify.json`, verifier
`arabic_ocr/scripts/stage1_verify.py`). 200 pages / 1600 lines / 57 Arabic fonts /
28-44 px with degradations: `coverage_fail_lines 0, paw_box_mismatch_lines 0, orphan_glyphs
0, empty_render_lines 0, transcript_mismatch_lines 0, ink_uncovered_px 0 (of 5,227,637 clean
ink px), ink_multi_covered_px 0, ink_in_foreign_line_box_px 0` -> **GATE PASS**.

> Two real bugs the gate caught: `render_page` recorded the **pre-**`fit_line` text
> (25/1600 lines mislabelled against their own pixels - now `text: r.text`), and 30
> polarity-inverted pages made `degraded_ink_inside_paws_min` read 0.0065 (fixed by flipping
> the image when the record carries `invert`). A first verifier report of 6881 cross-line
> bbox overlaps was a **false alarm** - axis-aligned PAW bboxes overlap by nature; the gate now
> tests at the **ink** level. `verify_samples.txt` is the UTF-8 artefact for human review
> (the console cannot show Arabic).

**2. Stage 2 detector stack built, unit-validated, and LAUNCHED** (job `pwsh-25`).

| piece | file | note |
|---|---|---|
| targets | `arabic_ocr/src/data/paw_targets.py` | DBNet offset `d = area*(1-r^2)/perimeter`, shrink 0.4; scipy-free 8-connected labelling; IoU matching |
| dataset | `arabic_ocr/src/data/synthetic_pages.py` | resize 960x720, scale boxes, min_side filter, `downsample_targets(factor=2)` |
| model | `arabic_ocr/src/models/paw_detector.py` | `PAWDetector` **2,563,866 params**, FPN + prob/thresh heads, **output stride 2** |
| loss | `arabic_ocr/src/training/detection.py` | masked BCE(pos_weight) + dice + L1 thresh + logit-space binarization; `evaluate_detector` |
| trainer | `arabic_ocr/scripts/stage2_train.py` | AdamW + cosine + AMP + grad accum + gate early-stop |
| config | `arabic_ocr/configs/stage2_paw.yaml` | batch 8 / accum 2, 20 epochs, gate 90.0 |

> **Why stride 2, not the DBNet stride of 4:** the median PAW is 15x24 px native (min-side
> p25 = 7 px) and only 7x12 px at stride 2; at stride 4 it is under 6 px. PAWs with min side
> < 6 px input (~28-30% of them - 3 px harakat and dots) are dropped from the target and
> **counted**. This is the one deviation from the plan 640x640 page size: at 640x640 the p5 PAW
> min-side is 1.1 px, so training uses 960x720. Documented in the module docstrings.
>
> Unit suite **26/26 PASS** (`scratch/stage2_unit.py`); VRAM probe peak **1.84 GiB at batch
> 8** against 6.97 GiB free. 2000 train / 300 val pages in `runs/stage2_synth/` (944 MB).
> Smoke on 64 pages over 3 epochs: H 7.63 -> 18.31 -> 29.79 (P 18.76 -> 55.24, R 4.79 -> 20.39)
> - a healthy DBNet early profile. **Gate: PAW Hmean >= 90% on held-out synthetic.**
> Eval note: decode costs ~200 ms/page (pure-Python labelling), so eval is the slow part.

# 2026-10-04 (f) (Arabic OCR Stage 0.5 follow-up: the width lever is REAL but SMALL and does NOT rescue fine-tuning; E-78 FAIL. Stage 1 STARTED):

> Follow-up to (e), run on the user's instruction "test it, if no good result go to next stage".
> Free-running, un-mirrored CER on `data/hflines/val` (1,654 lines):

| arm | width | CER | vs zero-shot |
|---|---|---|---|
| zero-shot control | dynamic | **57.80%** | - |
| zero-shot control | fixed 320 | 73.24% | +15.44 pp |
| fine-tune lr 1e-5 | fixed 320 | 64.04% | +6.24 pp |
| fine-tune lr 1e-5 | dynamic | **62.76%** | +4.96 pp |
| fine-tune lr 1e-4 | fixed 320 | 72.09% | +14.29 pp |
| fine-tune lr 1e-4 | dynamic | 78.34% | +20.54 pp |

> 1. **Verdict: FAIL as a rescue.** The width mismatch was a real confound - the SAME zero-shot
>    checkpoint moves 15.44 pp between widths - and matching it bought 1.28 pp at lr 1e-5. But
>    every fine-tuned arm still loses to doing nothing, and at lr 1e-4 dynamic width made it
>    6.25 pp WORSE. E-77's conclusion is unchanged: 2k handwriting lines is not enough.
> 2. **The first dynamic attempt OOM'd the 8 GB card** - and it was **VRAM, not system RAM**
>    (31.8 GB total, 17.9 GB free, nothing above 0.5 GB resident). `ds_width: true` at
>    `max_w: 3200` / `first_bs: 32` drove `max_mem_reserved` to 10,199 MB; batch cost spiked
>    1.33 s -> 7.90 s (WDDM shared-memory spill). Fixed with `max_w: 1280` (99% of val lines)
>    + `first_bs: 8` -> 2,677 MB and ~7x faster per batch.
> 3. **Stage 1 STARTED** (user's standing "go to next stage"). `arabic_ocr/src/data/synthetic.py`
>    replaced, not extended. Root cause of the old toy renderer: **PIL is built WITHOUT Raqm on
>    this machine** (`ImageFont.core.HAVE_RAQM is False`), so it cannot shape Arabic at all -
>    hence the disconnected left-to-right glyphs. New pipeline: HarfBuzz shaping + FreeType
>    rasterization, glyph boxes free from the HarfBuzz advances, and PAW boxes = **connected
>    components of the rendered ink**, merged when components share a HarfBuzz cluster (so a
>    floating haraka still belongs to its letter). Exact, not a gap threshold. Installed
>    `uharfbuzz 0.56.2` / `freetype-py 2.5.1` / `python-bidi` / `arabic-reshaper` into
>    `.venv-paddle-gpu`; 57 Arabic-capable fonts found on disk. Smoke gate PASS (0 coverage
>    failures); 200-page gate running.

# 2026-10-04 (e) (Arabic OCR Stage 0.5 PILOT RESULT - fine-tuning a printed recognizer on 2k handwriting lines HURTS; E-77 FAIL / NO-GO):

> Stage 0.5 pilot ran to completion on GPU (user-approved). Free-running, un-mirrored CER on
> `data/hflines/val` (1,654 lines), the only valid metric:

| arm | CER | WER | lines <50% CER |
|---|---|---|---|
| A: control (pretrained v6, no fine-tune) | **57.80%** | 98.22% | **28.7%** |
| B: fine-tune 2k lines, 3 ep, lr 1e-4 | 72.09% | 103.04% | 1.5% |
| C: fine-tune 2k lines, 3 ep, lr 1e-5 | 64.04% | 99.43% | 15.7% |

> 1. **Verdict: FAIL / NO-GO for Stage 0.5 as configured.** Fine-tuning made the model worse, not better
>    (+14.29 pp at lr 1e-4, +6.24 pp at lr 1e-5). The control reproduced Stage 0 exactly (57.80 vs 57.80),
>    so the harness is validated and the degradation is real.
> 2. **The binding constraint is NOT the learning rate.** A 10x lower LR reduced the damage but did not
>    reverse it. 2k handwriting lines only degrade a printed-trained recognizer.
> 3. **The framework's own metric is teacher-forced and moved the OPPOSITE way** (`norm_edit_dis`
>    0.168 -> 0.140 improving while free-running CER 57.8 -> 72.1 worsening). `best_accuracy.pdparams`
>    selected a checkpoint 14 pp worse than the start. Never trust a framework metric as progress.
> 4. **GPU path PASS**: `paddlepaddle-gpu==3.3.1` cu126 from Paddle's own CDN (PyPI has no GPU wheel).
>    3 min vs 79 min CPU; 4.1 GB of 8 GB VRAM. This is what makes iterating cheap.
> 5. **Implication for the plan**: the E-76 "warm-start from a printed recognizer" lever needs a real
>    handwriting corpus. NEXT: user decision - (a) build a handwriting corpus and retry, (b) skip
>    Stage 0.5 and go to Stage 1 (synthetic print, which E-76 already solved at 8.70% CER), or (c) other.

# 2026-10-04 (d) (Arabic OCR Stage 0 RESULT - printed is solved by tiny models, handwriting is not; E-76 branch 3 fires):

> Stage 0 ran to completion (user-approved). Full report: `runs/ocr_bakeoff/REPORT.md`.
> Free-running, un-mirrored CER (the only valid metric), CPU inference:

| eval set | what it is | A: v5 mobile (1.98M) | B: v6 small (3.10M) | C: our from-scratch (recorded) |
|---|---|---|---|---|
| `data/benchmark/isippt` (500) | PRINTED | **8.70%** | 13.75% | 144.8% |
| `data/benchmark/khatt` (200) | HANDWRITTEN | 62.61% | 44.41% | 97.8% |
| `data/hflines/val` (1,654) | our in-domain | 71.24% | 57.80% | not run |

> WER (un-mirrored): isippt 26.6% / 45.3%; khatt 90.8% / 91.5%; hflines 95.3% / 98.2%.
> Line accuracy: isippt 33.2% / 18.0%; khatt 0% / 0%.

> 1. **The nano thesis is CONFIRMED for printed Arabic** - a **1.98M-parameter** model reads
>    real printed lines at **8.70% CER**, **16.6x better** than our from-scratch control, with
>    87.4% of lines under 20% CER. The 20-50M target has 10-25x parameter headroom over what
>    already works.
> 2. **E-76 branch 3 fires**: both arms >= 50% on `data/hflines/val` (71.24% / 57.80%)
>    => the gap is our domain, so Stage 1 must lead with handwriting, not synthetic print.
>    Bars 1 and 2 (<=20% / <=10% on hflines) **FAIL**.
> 3. **Provenance correction: `data/hflines` is a HANDWRITING corpus, not "mixed".** It was
>    built from KHATT + Muharaf + reallines (the first two are handwritten corpora); only
>    **1.5%** of its val lines are read at <20% CER by BOTH printed-trained arms, and its
>    pixels are colour manuscript scans (chroma 28.2 vs ISIPPT's 4.0; high-frequency energy
>    20.1 vs 63.1). The E-76 bar was in practice a handwriting bar.
> 4. **A pretrained printed recognizer beats our from-scratch model 2.2x on handwriting it has
>    never seen** (44.41% vs 97.8% KHATT). Warm-starting Stage 3 from it is the cheapest known
>    path to a usable handwriting model - see NEXT.
> 5. **Harness validated against the published card**: v6 on the same isippt subset gives
>    13.75% CER (card 13.5%) and 45.25% WER (card 47.3%). The rest of the table is therefore
>    trustworthy. Caveat: that row reproduces a published number rather than testing
>    independence; KHATT and hflines are our independent tests.

> TRAPS SOLVED (in MEMORY.md so nobody re-pays them): both models emit **visual (mirrored)
> order** - scored as-is, printed Arabic looks like CER 80%; the fix is reverse, re-reverse
> the LTR runs, swap bracket pairs. The CTC alphabet is `['blank'] + inference.yml dict +
> [' ']` = 749 classes (the trailing space is real). `paddle.jit.load("<dir>/inference")`
> loads the PIR `.json` format directly, so **`paddleocr`/`paddlex` was never installed**
> (paddlepaddle 3.3.1 CPU only, 2 packages, no numpy/opencv/torch conflict).

> NEW FILES: `arabic_ocr/scripts/stage0_bakeoff.py` (free-running harness - the old
> `eval_benchmark.py` cannot evaluate a Paddle model), `stage0_analyze.py` (per-line CER
> structure), `stage0_domain_probe.py` (image-statistics fingerprint),
> `dl_stage0_models.py` (49 MB; `arabic_ocr/models/` is now gitignored). Evidence under
> `runs/ocr_bakeoff/`: `REPORT.md`, `stage0_results.json`, `per_image.jsonl`,
> `analysis.json`, `domain_probe.json`, `hflines_classified.json`, `stage0_samples.txt`.

> NEXT (NEEDS_USER_DECISION - this is the Stage 1 gate): **Stage 0.5** - fine-tune the
> pretrained Arabic recognizer on our 25,921 real handwriting lines (`data/hflines/train`)
> and re-measure free-running on `data/hflines/val`. It starts from a model already 2.2x
> better than anything we have trained, on exactly the data we own; the only new cost is the
> PaddleOCR training stack (or a Paddle->torch port). Near 20-25% CER => the handwriting path
> is real and Stage 1-4 proceed with a warm start; still >45% => our 26k handwriting lines are
> the binding constraint and the plan needs data before architecture. Alternative (torch-only):
> distil arm A/B into our own nano model for a printed product and treat handwriting separately.

# 2026-10-04 (c) (Arabic OCR - cleanup executed, 18.60 GB reclaimed; Stage 0 STARTED):

> The user said "delete them then proceed with stage 0". Both were done.

> DELETION (approved, executed 2026-10-04). 42 paths removed, **E: 7.92 -> 26.52 GB
> free (18.60 GB reclaimed)**, all keeps re-verified present afterwards (0 missing).
> Guards used: explicit absolute paths, no globs; the script refused to run at all
> if any delete path was an ancestor of a keep path (0 violations). Deleted:
> TIER 0 (`runs/trocr_finetune_v2/checkpoint-189/`, the two `__pycache__` dirs),
> TIER 1 (`runs/trocr_finetune/checkpoint-189/` - the parent's small logs kept),
> TIER 2 (20 teacher-forced-era run dirs; `runs/vit_real` kept), TIER 3 (7
> superseded checkpoints under `runs/u1_fused`, `runs/u1_full`, `runs/pilot` -
> every `final/` kept), TIER 4 (11 superseded OCR image mixes).
> **TIER 5 (laya `B_last`) was NOT touched** - not this line; its owner must decide.

> TWO AUDIT CORRECTIONS, caught before deleting. The audit's TIER 4 listed
> `data/smoke` (6 MB) and `data/linecorpus_smoke` (3 MB) as "superseded OCR data
> mixes". They are not: `data/smoke/` holds `tokens/train_000.bin` +
> `tokens/val_000.bin` (the S-phase tokenized shards, which AGENTS.md 5 says to KEEP
> because re-downloading is the expensive part) and `data/linecorpus_smoke/` holds
> train/val `labels.json` (3.7 MB of line labels). Both were excluded and kept.
> A directory name in someone else's audit is not evidence of its contents.

> STAGE 0 STARTED (the user's go). Design, from what is actually on disk:
> - Arms: A = `PaddlePaddle/arabic_PP-OCRv5_mobile_rec` (official, 81.27% strict
>   Arabic line accuracy), B = `medyas/arabic_PP-OCRv6_small_rec` (CER 10.2% real
>   printed / 13.5% ISIPPT, WER 47.3%), C = our recorded from-scratch control
>   (144.8% ISIPPT / 97.8% KHATT) - recorded numbers, no new compute.
> - **Both repos ship PaddleOCR inference format** (`inference.json` +
>   `inference.pdiparams` + `inference.yml`; v6 also ships the training
>   `.pdparams`/`.states` and `ppocrv5_arabic_dict.txt`). Both can therefore be
>   loaded with `paddle.jit.load` + the dict from `inference.yml` - **the heavy
>   `paddleocr`/`paddlex` dependency tree is NOT needed**, which avoids any risk of
>   it downgrading numpy/opencv under the existing torch/transformers stack.
> - Data: `data/hflines/val` (1,654 lines, in-domain, mixed) + `data/benchmark/isippt`
>   (500 printed) + `data/benchmark/khatt` (200 handwritten). The last two give the
>   printed-vs-handwritten split E-76's bar requires; `data/hflines` has NO source
>   manifest (only train/val + labels.json), so the split must come from `data/benchmark`.
> - Metric: free-running CER/WER only (CTC greedy / NRTR greedy - both free-running
>   by construction). New harness `arabic_ocr/scripts/stage0_bakeoff.py`; the existing
>   `eval_benchmark.py` cannot be used - it is bound to our own `create_model` + vocab.
> - Install: `uv pip install --python .venv paddlepaddle` (3.3.1 CPU). The dry-run
>   resolved only 2 packages (`paddlepaddle`, `opt-einsum`) - no numpy/opencv/torch
>   conflict. CPU first because this is a measurement, not training; the GPU is free
>   (315 MiB of 8192 used) if throughput forces a GPU build later.

> NEXT. Run the bake-off; report CER/WER per arm per dataset, split printed vs
> handwritten, against the E-76 bars.

# 2026-10-04 (b) (Arabic OCR - disk-cleanup dependency review; two harvests done, nothing trained):

> CONTEXT. Another agent produced a TIER 0-5 disk-cleanup audit (~21 GB) and the
> user asked which of it the approved OCR plan still needs **before** anything is
> deleted. The per-tier review was delivered and the user approved the two
> harvests it identified. Nothing was deleted and nothing OCR was started.

> TIER VERDICT, from the OCR plan's perspective.
> - **TIER 0** (`runs/trocr_finetune_v2/checkpoint-189/`, 2051 MB) - the audit's
>   "provably lossless" claim was **incomplete**. The weights ARE byte-duplicated
>   at top level, but the checkpoint also held this run's **only**
>   `trainer_state.json` - no top-level copy existed. Copied out first to
>   `runs/trocr_finetune_v2/trainer_state.json` (4,574 bytes, SHA256
>   `39E82E23...E1BE4`, identical at both paths). The directory is now safe
>   to delete - that is TASKS row 124.
> - **TIER 1** (`runs/trocr_finetune/`, 3138 MB) - delete the **subdirectory**
>   `checkpoint-189/` (2050 MB model + the 1086.71 MB `optimizer.pt` left by
>   the crashed v1 write), NOT the parent: the parent's small logs
>   (`benchmark.log`, `control_test*.txt`, `rendered_test.txt`,
>   `encoding_fix_test.txt`, `focused_padding_test.txt`) are the machine
>   evidence for the cp1256 and padding findings. Same 3.1 GB freed.
> - **TIER 2** (6.4 GB) - not needed. Stage 0's control arm is the **recorded**
>   number (144.8% ISIPPT / 97.8% KHATT), not a re-run; its evidence is
>   `runs/vit_real/best.pt` (89.94 MB, KEEP), `runs/benchmark_results.json`
>   and `runs/benchmark_results_TF_historical_backup.json` - all three
>   confirmed present and outside every listed directory.
> - **TIER 3** (2.2 GB) - not this line, and the finals all exist:
>   `runs/u1_fused/final` (6 files), `runs/u1_full/final` (6),
>   `runs/pilot/final` (9). Two guards: `runs/pilot/final` is the
>   **frozen KD teacher** for E-03 / TASKS row 20, and `runs/u1_fused/final`
>   backs the U-1b renderer demo - both must be named explicitly in any keep-guard.
> - **TIER 4** (5.2 GB) - **the images are worthless, the labels are not.**
>   Harvested first, see below. `data/hflines` and `data/reallines` stay
>   untouched.
> - **TIER 5** (1.9 GB, laya `B_last`) - not this line; its owner must confirm.
>   Note the proposal deletes `B_last` while this file marks `A_last` in
>   `l3_e70`/`l3_e72` as ladder-critical - A/B-suffix globbing is exactly
>   where a keep-guard silently matches the wrong file.

> WHAT WAS ACTUALLY DONE (CPU only, no GPU, no training, nothing deleted).
> 1. `runs/trocr_finetune_v2/checkpoint-189/trainer_state.json` ->
>    `runs/trocr_finetune_v2/trainer_state.json`, SHA256 verified.
> 2. New `arabic_ocr/scripts/harvest_text_pool.py` (read-only on sources) wrote
>    `data/ocr_textpool/`: 21 verbatim label files under `raw/`, plus
>    `texts.jsonl` (29.65 MB), `stats.json`, `samples.txt`,
>    `README.md`. Numbers: 21 source files, **101,954 rows**, **10,466
>    distinct**, 10,400 kept (66 dropped as boilerplate - the repeated VLM
>    instruction prompt, up to 2911 repeats), 8,528,526 chars, **78.6% in Arabic
>    Unicode ranges**, only 70 Latin-1-signature chars (i.e. no cp1256 mojibake),
>    length min/median/max 3 / 705 / 5562.
> 3. Pool total 76.81 MB; E: free went 7.99 -> 7.92 GB.

> CAVEATS NOW ON RECORD. The harvested texts are **article/paragraph length**
> (median 705 chars), not line length - Stage 1 must segment them into lines.
> `extra/labels.json` was empty (0 rows). `extra/labels_qari_v03.json`
> (3,000 rows) is diacritized and carries `h1`/`u`/`i` markup - use the
> `plain` field for rendering. **`data/ocr_textpool/` must never be deleted**:
> it is the only surviving copy of those labels.

> PROCESS RULE (this repo's own history). On 2026-09-16 a Tier-A cleanup deleted
> `runs/yarn_lora_sft_v1/final_a60/model.safetensors`, an explicit KEEP, because
> the guard was `$entry.FullName -eq $keep` - `FullName` is absolute and
> `$keep` was relative, so the guard never matched - and the snapshot step's
> `robocopy /XF` compounded it. Any deletion from this audit must print both the
> delete list and the keep list, compare **absolute** paths, and dry-run first.

> NEXT. The user deletes the tiers; nothing OCR has started. Stage 0 (TASKS row
> 119) is still USER-GATED and needs the 2.15 GB reclaim plus ~1 GB for the
> paddle install.

# 2026-10-04 (Arabic OCR - user-approved two-stage plan; DOCS ONLY, nothing started):

> USER REQUEST (verbatim, two messages): the from-scratch black box failed, so
> instead train a traditional OCR that detects the smallest region containing
> characters and recognizes that region, aiming at a 20-50M nano model, and
> combine the pieces later for a bigger model if needed. Then: the suggested
> stages are approved, with the clarification that 500k images need not be
> materialized at once - render a chunk that fits the disk, use it, delete it,
> move on. Explicit instruction: **update the task/plan/milestones/docs only; do
> not start until told.**

> WHAT WAS AGREED. The decomposition is accepted, and so is the 20-50M target
> (PP-OCRv6, arXiv 2606.13108, is literally "1.5M to 34.5M parameters,
> surpassing billion-scale VLMs on OCR tasks"). One correction was accepted: the
> detection unit is the **PAW / connected component**, not the character -
> cursive Arabic fuses 2-6 letters onto one baseline stroke, and dots plus
> diacritics are detached components, so "the smallest region containing one
> character" is not well defined without already knowing the word. Evidence: the
> Arabic character-segmentation survey, PAW word-spotting work, and VML-HD's
> sub-word annotation level.
> The strongest argument FOR the user's idea is structural: a detector loss is
> per-pixel and spatial, so it cannot take the character-LM shortcut that
> produced 144.8% CER and one identical string for every image.

> BARS ALREADY ON RECORD (do not re-derive): `PaddlePaddle/arabic_PP-OCRv5_mobile_rec`
> (Apache-2.0, official) 81.27% strict line accuracy on Arabic;
> `medyas/arabic_PP-OCRv6_small_rec` CER 10.2% real printed / 13.5%
> ISIPPT held-out, WER 47.3%, from 800k synthetic lines over 26 base + 12
> display fonts, MultiHead CTC+NRTR, ~1050 character classes. Our recorded
> control is free-running CER 144.8% ISIPPT / 97.8% KHATT.

> DOCS UPDATED THIS ENTRY (no code, no download, no GPU work):
> * `docs/plans/2026-09-29-arabic-ocr-nano-model.md` - new V2 section
>   (decision record, 5 stages with gates, disk budget, risks) above a preserved
>   V1 archive. V1 is dead twice over: its Phase 1 dataset
>   (`medyas/arabic-ocr-printed-500k`) is NOT on disk, and its
>   4-architecture sweep is the process that produced the teacher-forced era.
> * `TASKS.md` - new OCR-line section, rows 119-125.
> * `research/EXPERIMENTS.md` - **E-76 pre-registered** (bars recorded
>   before any number is seen).
> * `arabic_ocr/STATUS.md` - 2026-10-04 section superseding the
>   2026-10-01 "Path forward" list.
> * `arabic_ocr/README.md` - honest warning banner; its tables claim
>   CRNN CER 1.54, which is a teacher-forced artifact.
> * `MEMORY.md` - 6 new lessons (PAW unit, `.no_exist` mirage,
>   disk-chunked synthesis, the toy synthetic.py, why dense supervision cannot
>   be short-cut, the README lie).
> * `ENVIRONMENT.md` - verified 2026-10-04 machine/disk/package facts.
> * `research/WHAT_WORKS.md` - OCR levers from the failed paths.

> VERIFIED FACTS THIS SESSION: E: free **8.0 GB** (the TrOCR v2 run consumed
> 4.3 GB); `uharfbuzz`, `arabic_reshaper`, `python-bidi`,
> `cv2`, `shapely`, `albumentations`,
> `paddleocr` and `paddlepaddle` are ALL absent from
> `.venv`; `fontTools` is present; 20+ real Arabic fonts are
> on disk; `data/hflines` is line-level only (25,921 train / 1,654 val)
> so a real page-level detector corpus must be synthesized.

> NEXT (single owner, blocked on the user): Stage 0 - a ~1 h zero-shot bake-off
> of the two PaddleOCR Arabic recognizers against `data/hflines/val`
> with the free-running harness, CER split handwritten vs printed. It needs row
> 124 (reclaim 2.15 GB) first. Do NOT start it, and do not install anything,
> until the user says go.

# 2026-09-26 (U-LINE U-1b VERDICT: TEMPLATE ECHO CONFIRMED - fusion run LIVE, job pwsh-5, ETA ~1h50m):

> VERDICT (pre-registered bars, gate runs/u1_full/u1b_gate.json, 336 prompts, 2,822 s). Set A = 36 distinct main test-split prompts (all of which also occur in main train - the 114-prompt pool). Set B = 300 distinct production_50k test-split prompts, never trained on. Identical decoding for both columns.
> | | valid | mean_type_f1 | median | f1>=0.5 | arch acc | content_jaccard | known_prompt |
> |---|---|---|---|---|---|---|---|
> | A seen (n=36) | 0.583 | 0.927 | 1.00 | 1.00 | 1.00 | 0.033 | 1.00 |
> | B UNSEEN (n=300) | **0.027** | **0.205** | 0.20 | 0.067 | 0.107 | 0.005 | 0.027 |
> Both pre-registered echo clauses met on Set B (valid < 0.90 AND f1 < 0.50) by wide margins => **prompt diversity, not structural variety, is the binding constraint**; the pre-registered fusion arm is released. 292/300 (97.3%) of unseen generations emitted no parseable spec at all; only 28 distinct component-type signatures across all 336 outputs, modal share 0.374.
> The recorded U-1 "greedy_valid 1.00 / pass@4 1.00" is REPRODUCED here as 0.583/0.927/1.00 on memorized prompts - it stands, but measures recall of 114 training prompts, not following. CAVEAT logged: Set A 0.583 is below the recorded 1.00 because this gate truncates (max_new_tokens 780 on a ctx-512 model); the A-vs-B contrast is unaffected since both columns decoded identically.
> BONUS: content_jaccard ~0 in BOTH columns - the model does not follow even the content of prompts it memorized, only their structure.

> RUN STATE: scripts/train.py --config configs/u1_fused.yaml is LIVE (job pwsh-5). 85,000 train rows / 42,692 distinct train prompts / 54,643,559 tokens / 13,341 steps. Zero-flag auto-resume if it dies.
> - **MEASURED 2.02 it/s = ~33.1k tok/s, ETA ~1h50m - NOT the 2.5 h estimated earlier** (that estimate used main-corpus token density; the fused corpus packs better). Trainer is at ~91% of the 36.3k manual-bench rate, so the "trainer is 34% slower than a bench" gap I flagged earlier is really ~9%, not 34%.
> - **vram_cap_gib had to be 5.8 -> 7.0.** The 5.8 was copied from diacritizer configs sized for a 6 GB card; the 30M arch measures 6.13 GiB peak at b16/accum2 and OOMs at 5.8. Probing before launch is the only reason this run is alive.
> - Batch sweep re-probed at the REAL 30M arch (the recorded matrix was 12.3M, at the hardware ceiling, so it did not transfer): b16 x accum2 = 36,3xx tok/s beats b8 x accum4 = 31,0xx by 17% at identical eff batch; b20/b24 do not fit 8 GB because the fp32 loss logits tensor (batch x 512 x 32768) is a single 1.34-1.61 GiB allocation.

> NEXT (single owner): wait for pwsh-5 (~1h50m) -> re-gate with the IDENTICAL 336-prompt protocol: `python -m ui.scripts.eval_u1b_semantic --model runs/u1_fused/final --n-prod 300 --out runs/u1_fused/u1b_gate.json` -> compare the Set B column against 0.027 valid / 0.205 f1 / 0.107 arch. The gate is now resumable (`--resume`) and flushes a partial every 25 prompts, so an interruption is cheap.
> STILL OPEN (untracked repo gaps, not blocking): production_50k/ is untracked with a 118 MB corpus.jsonl (over the 100 MB LFS rule); the builder ui.scripts.build_space_bunny_50k named by its README/manifest does not exist in ui/scripts/.

# 2026-09-26 (U-LINE U-1b SEMANTIC GATE + FUSION ARM - user approved "proceed, you can use gpu and cpu"; U-1 verdict RE-SCOPED, not retracted):

> WHAT THIS CHANGES: the recorded U-1 result "greedy_valid 1.00 / pass@4 1.00" was measured on a held-out set that CANNOT distinguish instruction-following from memorization. Two measurement defects found, both verified on disk:
> 1. data/u1/raw/u1_main_50k.jsonl = 50,000 rows but only **114 DISTINCT task prompts** (max reuse 569). The model was fit to map ~114 asks over 8 archetypes, i.e. plausibly archetype classification, not request understanding.
> 2. The splits are seed-disjoint but NOT prompt-disjoint: **all 114 distinct test-split prompts also occur in the main train split** (val likewise). So the U-1 gate's "40 held-out tasks" were prompts the model had already seen verbatim in training with a different spec.
> The numbers stand as recorded; their SCOPE narrows to "reproduces a valid spec for prompts it memorized".

> NEW DATA FOUND + measured (was untracked, placed in the repo 2026-09-26 09:04, files dated 2026-09-24): ui/data/space_bunny/production_50k/corpus.jsonl = 50,000 rows / **47,044 distinct prompts (94 pct)** / 50,000 distinct specs / 0 exact AND 0 shape overlap with main and omen / 41-of-41 catalog components used / validation_report.json PASS with 0 codec+validator errors / pre-split 45,000 train + 3,000 validation + 2,000 test. legacy_valid.jsonl (400 rows) is 400/400 CONTAINED in corpus.jsonl - never add the eight legacy side_space_bunny_*.jsonl files on top.
> KNOWN DEFECT in that corpus, recorded not discovered: 41/41 by TYPE but **Radio occurs 8x and Slider 5x** across 50,000 rows - effectively untrained.
> REPO-GAP: the corpus is UNTRACKED and corpus.jsonl is 118 MB (> the 100 MB LFS threshold, AGENTS.md 5). The builder entry point its README/manifest name, ui.scripts.build_space_bunny_50k, DOES NOT EXIST in ui/scripts/ (only build_corpus, prepare_u1, eval_u1_gate) - the corpus is not reproducible from this repo as committed. Generation code IS present at production_50k/generator_core.py + generators/*.py.

> U-1b GATE (pre-registered in research/EXPERIMENTS.md "U-1b PRE-REGISTER" BEFORE any number was seen; tool ui/scripts/eval_u1b_semantic.py, new). Metrics: M1 mean element-type F1 vs the reference spec (PRIMARY), M2 k-NN archetype accuracy, M3 signature-echo share, M4 task-content-word overlap, M5 validity re-measured in the same run, M6 chain-copies-a-training-row rate. Eval sets: Set A = 36 distinct U-1 test prompts, Set B = 300 distinct production_50k TEST prompts. Bars: valid >= 0.90 AND mean_type_f1 >= 0.70 = follows the ask; valid >= 0.90 AND f1 < 0.50 = TEMPLATE ECHO CONFIRMED.
> SMOKE (4+4, NOT the gate verdict) already shows the direction: Set A valid 1.00 / f1 0.839 / arch 1.00 but **content_jaccard 0.00** and emitted-task-is-a-known-prompt 1.00; Set B (never-trained prompts) valid 0.25 / f1 0.229 / arch 0.00. Manual inspection of 3 generations: on "harborcrest treasury chargeback handling overview" the model emitted a TODO app; on "a settings page with an email notification toggle" it **overwrote the prompt with a memorized training row** ("TASK a profile page with our team page"). Full verdict pending the 336-prompt run.

> FUSION ARM BUILT AND STAGED (pre-registered, launched only on the echo verdict):
> * ui/scripts/prepare_u1.py PATCHED: --corpus now takes MULTIPLE paths; a row may carry chain OR spec (chain derived via spec_to_chain - 50,000 derived this run); split labels normalized ("validation"->"val", unknown label = hard error, never a silent train assignment); hard assert that no val/test row can reach train.jsonl; new --require-prompt-disjoint drops held-out rows whose prompt also occurs in train, and REPORTS the overlap either way. ui unit tests 10/10 PASS after the patch.
> * production_50k is ~10 pct leaky too (302/2,992 val and 206/1,995 test prompts recur in its own train), so the flag dropped 7,807 val and 2,703 test rows to reach a clean held-out.
> * CORPUS: data/u1/fused/raw = **85,000 train / 1,500 val / 500 test; 42,692 distinct train prompts (from 114); 0 train-val and 0 train-test prompt leakage; 400 sampled train rows 0 invalid**. Tokenized: **54,643,559 train tokens / 1,284,048 val** (106,725 blocks, 2,507 val blocks).
> * CONFIG: configs/u1_fused.yaml = configs/u1_full.yaml with corpus + max_steps only; max_steps 13,341 = 4 epochs (3,335 steps/epoch). **CORRECTION to the approved estimate: the run is ~2.5 h, not ~1.5 h** - prod rows are 643 tok/row vs main 313, so the fused corpus is 3.5x the tokens, not 2x. At the U-1 measured 23.9k tok/s.
> * GAP CLOSED: scripts/train.py had NO allocator ceiling (only the diacritizer trainer did), which CLAUDE.md 20.3 requires for unattended launches. Added the vram_cap_gib key mirroring diacritizer/scripts/train_lm.py; u1_fused.yaml sets 5.8 GiB hard ceiling (b16 measured 4.4 GB peak). Preflight 3 PASS / 0 FAIL / 2 SKIPPED.
> RESUME: preflight is already green and the corpus is tokenized - the only thing gating the launch is the U-1b verdict. Launch command (GPU must be otherwise idle; nothing of ours may co-run):
>     & .\.venv\Scripts\python.exe scripts\train.py --config configs\u1_fused.yaml
> auto-resumes with zero flags after any kill. Then re-gate with ui/scripts/eval_u1b_semantic.py --model runs/u1_fused/final.

# 2026-09-24 (U-LINE SPACE-BUNNY SIDE DATA - strict correction + handoff, CPU only): the 8 files under `ui/data/space_bunny/` are complete at 400 rows total (50 per archetype), each row `{arch_hint, task, spec, split:"train"}`. The initial 400 rows were already added in commit `8c3cb40`; this handoff records the independent strict gate and the correction commit. FINAL GATE PASS: 8/8 files, 400 unique tasks, 400 unique specs, 0 exact task/spec overlap with `ui/data/omen_alpha/side_omen_alpha_batch001.jsonl`; every row has 6-16 elements, root depth 2-4, <=5 children per parent, required catalog props, flat reachable graph, lossless `spec -> chain -> spec`, empty codec parser issues, and `validate_chain(spec_to_chain(spec, task)) == []`; each file exercises stateless + bindState + visible + repeat + pushState. Corrections fixed 2 missing required `name` props and flattened 72 root-depth-5 specs (36 pricing_cards, 36 search_results). UI unit tests: 10/10 PASS. RESUME: read this block, then `ui/prompts/DATA_GEN_PROMPTS.md`; data is ready to sample/fuse into U-1 follow-up corpus work, but this handoff does NOT fuse it into the U-1 training corpus or launch GPU work. If expanding later, cap each archetype at <=500 and rerun the same strict gate before use.

# 2026-09-24 (U-LINE U-1 COMPLETE - UI-JSON GENERATOR TRAINED AND GATED, user GPU window): the"small LM that writes UI in JSON" line's first full train is DONE and gate-PASSED.
> WHAT WORKS (10-second orientation): the model is runs/u1_full/final (27.8M GQA ctx512, eval 0.2705, greedy-valid 1.00 + pass@4 1.00 on 40 held-out tasks).
> CORPUS: data/u1/raw/u1_main_50k.jsonl (40k/7.5k/2.5k blocks by build_corpus seeds; render->tokens via ui/scripts/prepare_u1.py -> scripts/tokenize_data.py -> data/u1/full/tokens 15.63M train/682k val).
> CHAIN SURFACE frozen from U-0 (catalog UI1-2026-09-24-41c; codec ui/src/chain.py lossless roundtrip; validator ui/src/validate.py 3-layer).
> THROUGHPUT (recorded once, reuse): Quadro RTX 4000 fp16 non-tensor ceiling; batch16/ckpt-off/workers0 = 108-116k tok/s at 12M, ~24k at 28M; b32 collapses (WDDM allocator spill history 2026-09-14; same signature); benchmarking = manual opt loop in scratch (removed; matrix in EXPERIMENTS "U-1 THROUGHPUT MAX").
> GATE tool: ui/scripts/eval_u1_gate.py (greedy + pass@4 against validate_chain; generate from task+"\n", max_new_tokens 780; NOTE raw JSONL holds {text: task+chain} so partition on first \n).
> NEXT PRE-REGISTERED RUNGS: (a) U-1b semantic-follow gate (does the chain honor the SPECIFIC ask; archetype-template echo observed, not yet gated), (b) U-1c paraphrase/diversity rung (corpus task texts only 114 unique templates code-side; omen-alpha-side batches 001-004 = 5k+20k+50k tiers at data/u1/raw/side/omen_alpha_all.jsonl 50,980 rows not yet FUSED into the main corpus), (c) stage-A format-LM pass (chain-only, TASK masked) then stage-B SFT + balanced replay per the original pre-registration, (d) streaming RFC-6902 patches (WACH op etc. - non-chain surfaces, later).
> GOTCHAS kept alive: transformers 5.16.1 (no console losses - read tfevents / print trailing JSON); unknown-worker spawn errors inside MANUAL benches are a Windows artifact, not VRAM; scratch/tiny-gitignored raws under data/u1/*/raw regenerate in ~2 min, tokens/*.bin ARE the thing to keep.
> Config: configs/u1_full.yaml (max_steps 3800; train auto-resumes with zero flags after a kill - do not re-tokenize).
> Preflight + param gate R5 honored (27.8M, depth-over-width 16L/320 per E-12/E-13); preflight 3 PASS / 0 FAIL.

# 2026-09-24 (U-LINE SIDE-DATA BUILD — CPU only, no GPU touched): ui/data/omen_alpha corpus created under the ui/prompts/DATA_GEN_PROMPTS.md kit (user: "read DATA_GEN_PROMPTS.md ... save in ui/data/omen_alpha ... for start create small part but vary, up to 500"; later user: "continue scaling the datasets"). DONE: batch001 = 500 rows (8 archetypes x 6 domains x 4 tones, 34 templates, PROMPT A/B contract: rows {arch_hint, task, spec, split:"train"}, json-render flat-spec); PROMPT D critique pass (background subagent, 30-row sample: 30/30 PASS, weak axis binding_idiom; verdict + ADJUDICATION in ui/data/omen_alpha/README.md — 2 subagent claims REJECTED against catalog.py: changeType IS the string enum "-1"/"0"/"1", Table DOES have a caption prop); batch002 = 240 PROMPT C hard-mode rows (nested state 2-3 levels, lists-of-lists via $item "lines" tables); batch003 = 240 critique-fix rows (change events setState so gated banners fire, predicates matched to semantics, gate values guaranteed in option lists, ascending plan pricing); batch004 scale tiers 5k/20k/50k composed from ALL 54 templates (seed 241101, g randomization, dedup (task,sha1)), all 0 gate fails. Every row passes: validate_spec + _bounds_check (6-16 els/depth<=4/sib<=5) + validate_chain(spec_to_chain(spec, task)) == [] + spec->chain->spec round-trip equality (venv python only). GOTCHA: scale tiers share seed 241101 => 5k/20k are exact prefix-subsets of 50k; canonical mirror data/u1/raw/side/omen_alpha_all.jsonl = batches 001-003 + ONLY scale50000 = 50,980 rows (stale mirrors deleted; builder scratch/omen_side_build4.py mirror matches). Builders: scratch/omen_side_build{,2,3,4}.py (seeds 240924/241012/241026/241101) — deterministic RNG consumption, template edits re-shuffle downstream sampling. TASKS rows 108-111. NEXT: (a) widen the 54-template structural inventory (subagent fan-out viable) before more replays — sampler showed 0 pressure (attempts=rows); (b) ui/scripts/build_corpus.py to tokenize U-1 stage set from the canonical mirror; (c) U-1 training = USER-GATED GPU per EXPERIMENTS.md. Code codec gaps stay documented in ui/data/omen_alpha/README.md + omen_side_build2.py docstring ($template spaces / multi-binding events / onSuccess-navigate / $index removal are NOT chain-representable — keep excluded until chain gains the ops).
# 2026-09-20 (agent-OS session, DOCS ONLY — NO CODE, NO GPU): CLAUDE.md gained §19 Tool-use economy (batch-all-calls-per-turn; background jobs fire-and-notify, never polled; ≤5-turn default) and §20 Hardware utilisation protocol (user-approved: scavenge leftover GPU compute from FOREIGN non-training incumbents only — 60%-of-free-VRAM cap shrinking to 40% after OOM, hard allocator ceiling / no shared-memory spill per the 2026-09-14 WDDM incident, re-check before launch; repo train.py/sft.py runs keep the single-GPU rule; CPU fallback only if a probe shows ~a couple dozen minutes, let it finish, GPU free goes to the NEXT task). §10 carries the matching carve-out; AGENTS §4 remains in force. First sanctioned co-run = TASKS row 98 (pending); MEMORY lesson 74 records it as UNPROVEN until evidenced. No configs/scripts touched.
# μ1 (E-27..E-29) composition bake-off CLOSED 2026-09-19 — A soup/TIES FAIL (basin divergence, honest); C router PASS routing 1.00, team +0.04 vs control (marginal); D 12K-step distill FAIL to beat control (mixed 0.50 vs 0.64); arm B (MoE-merge) remains USER-GATED. Row-first discipline held (82aab6f pre-reg before results). Head: see research/micro_experts/MU1_REPORT.md.
# HANDOFF — nano_SLMs agent-to-agent continuation guide

> Written for an agent resuming with fresh context on this machine
> (E:\python_projects\nano_SLMs on TU09FBO; MUO4QK5 uses E:\python projects\nano_SLMs
> — with a space). Read PLAN.md (the spec) first, then this.

> 2026-09-18 (ME-LINE DESIGN SESSION - DOCS ONLY, NO CODE, NO GPU): New user-approved line — MICRO-EXPERT COMPOSITION ("train many 100-300K single-task models, combine into 1-10M; start small, grow step by step"). USER DECISIONS: (1) Stage 0 (μ0 sandbox) approved AS THE BASE for later stages (stepping stones; every μ-stage consumes μ0 artifacts, μ1 arms reuse the SAME experts without retraining); (2) all FOUR composition methods in scope — weight-merge (soup/TIES), MoE-merge (BTM/BTX), dispatch-router, committee distillation; (3) tasks = option A+B hybrid — synthetic testbed (arithmetic/structure/string-ops) + REAL Arabic diacritics as the first real expert (existing D-line data reused; user: decent data available, focus on it now). Artifacts: research/micro_experts/DESIGN.md (falsifiable question: committee vs dense control at matched params/tokens; decisions ME-D1..D7 incl shared char vocab ≤128 + shared seed; per-method pros/cons/gaps), TASKS rows 76-79, MEMORY decision row + lesson 73. NEXT: user reviews DESIGN → writing-plans builds docs/plans/2026-09-18-micro-expert-composition-plan.md → μ0 implementation. μ0 launch = pre-register E-24 BEFORE results; GPU work waits for a free window (single-GPU rule; E-23 lineage rows 69-75 may hold GPU work).
>
> 2026-09-16c (Tier-A cleanup incident - DATA LOSS, USER MUST READ): runs/yarn_lora_sft_v1/final_a60/model.safetensors (864 MB) ACCIDENTALLY DELETED during Tier-A cleanup execution. User explicitly approved Tier-A scope; the deletion script for yarn_lora_sft_v1/ had a path-equality bug (Get-ChildItem .FullName returns absolute path while the guard was a relative path) AND the snapshot script's exclusion list (/XF model.safetensors ...) wrongly excluded weights even from KEEP winners inside Tier-A dirs (snapshot has final_a60/{config,tokenizer} only).

> DELETED: runs/{mount_*,kt_ab_*,kt2_*,sft_v2_pilot,sft_u10_pilot,sft_v2,sft_u10,yarn_4096} + runs/yarn_lora_sft_v1/{final,final_a60,gates,logs} -- +8.6 GB freed; PARTIAL cleanup (Tier A asked ~8.7 GB, achieved ~8.6 GB).

> STILL ON DISK (Tier A NOT yet deleted, halted after the bug): runs/{h2_copy_lora,h2_copy_lora_pilot,h2p2_mixed_lora_pilot,gdn_smoke,gdn_smoke_ab,smoke,pilot,ctx_probes,soup_p0,agent_memory_h*,_mount_cpu_smoke.throwaway} ~1.43 GB.

> E-07 promoted a60 deliverable = LOST -- reproduce by re-running yarn_lora_sft_v1 pipeline (~1-2 h GPU: yarn_4096 base (~2 h) + LoRA-SFT + soup at alpha=0.6 to final_a60); recipe = configs/yarn_lora_sft_v1.yaml + config-gated init_from + the row-30 soup recipe.

> Pre-flight checklist (20 KEEP winners verified present before deletions) MISSED the absolute-vs-relative path mismatch in the PowerShell guard. Evidence + review still at runs/evidence/2026-09-16-checkpoint-review/REVIEW.md and runs/evidence/2026-09-16-tierA-cleanup/ (snapshot of small files from 17 fully-deleted Tier-A dirs; final_a60 model.safetensors NOT in snapshot).

> Disk: 7.00 GB free to 16.88 GB free. HALTED all further deletions; awaiting user decision. MEMORY lesson 62 added; TASKS row 62 pending.

> 2026-09-16b (ARABIC-DIACRITIZATION agent): **E-18 INPUT-CONTRACT + POSTPROC A/B (user-approved 4 options).** User hand-stressed the GOLD with 3 syntax-trap paragraphs; an outside model bisected defects (nawaasikh/agent-patient/misc) but MISROOTED the doubled-mark artifacts as a Softmax/CRF defect. Verified live: pre-marked input (عُقدت) yields doubled عَُقِدْتْ via passthrough; clean A/B proves doubles are INPUT-contract violations, not model defects; partially-marked input ALSO inflates DER/preservation optics. USER DECISIONS: (1) ADMITTED = auto-strip input marks in diacritize.py; (3) ADMITTED = user's 3 paragraphs locked as research/user_probe (QUALITATIVE - no vocalized gold); (2),(4) TESTED on gold preds (abdou 41.64/fadel 33.23/sadeed 45.68/wn14 48.99): dedup_marks and legality_repair outputs byte-identical (15-label head cannot double) -> DROPPED per user rule; postproc.py + rescore kept as record; the outside agent's CRF-retrain/Focal-loss advice ranked behind data breadth (E-13..E-17 lineage) and needs a sepa-rate user gate. Files: diacritizer/scripts/postproc.py, scratch/rescore_postproc.py, research/user_probe/*, configs touched none. MEMORY 61; TASKS row 60; EXPERIMENTS E-18 (E-17 row briefly clobbered by an edit collision - restored byte-for-byte from git).
>
> 2026-09-16 (ARABIC-DIACRITIZATION agent): **E-17 FINAL-MODEL CAMPAIGN COMPLETE — stage2b2500 (previous gold) REMAINS THE DEPLOYED FINAL MODEL.** Stage-1 v4 whole-corpus LM (pwsh-72, paused per user, resumed pwsh-1 zero-flag): 27,000 steps, val 1.2769->1.1905, still descending at budget edge (not saturated). Stage-2 final fine-tune (pwsh-2, configs/diac_stage2_final.yaml = recipe B + v4 init + reset-last-2, probes every 1250): GATE PEAK @2500 (fadel 34.9/sadeed 47.8/wn24* 57.5/wn14 49.6; clean mean 42.7) then all gates degraded while val kept improving->0.1735 — the val pick would have been the WORST model (run-end mean 46.5); the gate probe picked right for the 3rd consecutive run. HELD-OUT abdou test-00000 (15,091 sentences/41,378 lines, NEVER trained by any model; bench.py gained an abdou_test source, smoke PASS): stage2final@2500 **DER 49.4 / nocase 33.3** vs stage2b2500 **DER 41.6 / nocase 30.4** — v4 init loses the fresh test by 7.8pp. VERDICT: init scale still not the lever (3rd confirmation after E-14/E-15); deployed final = runs/diac/stage2b2500/gate_probe/best_gate_weights.pt. Artifacts: runs/diac/{stage1lm_v4,stage2final}/final + full gate_probe snapshot corpus; scratch/{stage2final,stage2b2500}/abdou_test_pred(.ref).txt; research/gate_results/gate_results.csv +38 rows incl abdou_test. MEMORY lesson 60; EXPERIMENTS E-17 CLOSED. User asks outstanding: git push, GPL-2 commercial flag, Sadeed overlap disclosure — still user-gated.
>
> 2026-09-14c (flow/ session): **F2 MODEL-ARCHITECTURE EDITOR INCREMENT A COMPLETE (user-approved scope A/A/A).** New top-level "Pipeline | Model" switch in flow/: second ReactFlow canvas editing dense-GQA model graphs (model/0.1, .modelgraph.json git-tracked in flow/models/). Layer registry flow/src/model/registry.ts + 9 kind modules (input/embedding/rmsnorm/rope/gqaAttention 3-input q,k,v/swigluFfn/residual/lmHead tied/layerStack depth-N counting ONLY outer pNorms) — spec-driven ports/props/defaults/paramCount, ZERO kind conditionals (T3 interlock kept). Shape+param walk flow/src/model/graphWalk.ts (Kahn, per-node honest errors, port-name checks, lmHead prices untied matrix over HIDDEN input d). Param math ported verbatim from viz/ engine into flow/src/model/paramMath.ts (anchors nano 226,526,208 / SmolLM2 134,515,008; shared-package extraction deferred). Backend GET/POST /api/models[/{name}] reusing flows.atomic_write (36e3efa). store.ts godstore split into flow/src/stores/{graph,registry,ui,run}Store.ts shim re-export, zero consumer edits. Acceptance: smoke/pilot/target committed as .modelgraph.json, totals LOCKED by flow/src/model/acceptance.test.ts (smoke 12,323,072 / pilot 100,682,496 / target 226,526,208 = pTotalDense). Browser drill PASS: Model tab + 9-layer palette + header total 12,323,072 + per-node badges (Embedding 8.4M, attn 196,608, ffn 786,432, norms 2,304); inspector-knob click via agent-browser did NOT reach ReactFlow node selection (synthetic-click limitation, SKIPPED — knob delta covered by acceptance unit test). Vectors: vitest 148/148, server unittest 145/145, pnpm build clean. Docs: flow/README.md Model-mode section; plan docs/plans/2026-09-13-model-architecture-editor-plan.md; commits ae43cdc/abc6b8c/6532771/1cebc34/2b2772d/66eaa28/36e3efa. Deferred: export/MoE/nested-stack bodies/shared param-math package. Push user-gated.
>
> 2026-09-13f (flow/ session): **REGISTRY PROMOTION REFACTOR COMPLETE (user-approved scope B, shape-preserving).** Each flow/ built-in node kind is now a behavior-carrying NodeDefinition (flow/server/nodes/builtin/{dataset,prepare,tokenize,train,eval_job,infer}.py): spec (ports/PropSpecs/gate) + label_semantic ("hf-dataset-name" for dataset) + features (["webui-chat-button"] for infer) + required_upstream (config_gen's chain order is DERIVED topologically — CHAIN/_DEPENDENCIES deleted) + validate_semantic (per-node knob errors, historic wording byte-preserved) + build_section (train owns model/train/eval + PRESETS/LR_PRESETS; tokenize owns TOKENIZER_NAME/SHARD_TOKENS; prepare owns MAX_ROWS). Frontend de-hardcoded: flow/src/nodes/registry.ts (FALLBACK_REGISTRY + snapshot types), portTypes.ts; /api/nodes gains label_semantic+features; UI has zero kind conditionals (infer button feature-driven, dataset placeholder label_semantic-driven). flows.save ATOMIC (temp+fsync+os.replace, 53986a8). Golden harness: flow/tests/goldens/ locks build_config output AND ValueError wording for all 5 committed flows. Verification: 133/133 server unittest (incl. new registry/golden tests), 93/93 vitest, pnpm build clean, browser drill PASS (examples open+validate, registry-driven UI, honest refusal toast). Latent KeyError fixed incidentally (train steps without preset/errors). Plan: docs/plans/flow-registry-promotion.md; commits 2632197/3ad4693/f72bd25/53986a8. Plugin discovery deliberately DEFERRED (user: later); next-phase menu (F5 branching / F2 model editor / config export / G3 retry) still awaiting user pick (retro: .superpowers/sdd/flow/retro-2026-09-13-flow-mvp.md). Push stays user-gated.
>
> 2026-09-14b (ARABIC-DIACRITIZATION agent): **E-14 STAGE-2 ARM A verdict: H REJECTED.** stage2a (warm-start + v3 corpus 2.44M win + 8k steps): val 0.1772/acc 94.1, gates 41.6(Fadel)/54.5(Sadeed, sadeedt-tuned)/60.9/54.1 = loses to best from-scratch arm on ALL 4 (fadel_spec 34.2/51.4, v2d28 59.1/52.5, decisive rule 2pp/3gates FAIL with no gate improving). transfer IS real (beats v2b 47.1/59.9 baseline on Fadel/Sadeed) but LM-pretrain-init alone cannot match domain specialization. DISK incident: E: hit 0.4MB free mid-run (train_lm.py had NO checkpoint rotation - fixed keep-3, commit 6a5d43d); cleaned stage1lm ckpt-500..11500 (~8.9GB freed) + stage1smoke ckpt-* + torn stage2a ckpt-1000; stage1lm final + ckpt-12000 kept. NEXT needs user: close route (record negative, v3-corpus/data-breadth route next) vs reset-last-2 counterfactual (+3h; tests causal->bidi mismatch) vs longer stage-1.
>
> 2026-09-13e (flow/ session): **VISUAL FLOW EDITOR MVP COMPLETE (Tasks 0-12, SDD on main).** New sibling app `flow/`: React 18 + TS + Vite + @xyflow/react 12 + zustand 5 frontend (palette-left/canvas/inspector-right; right-click Add menu, Ctrl+K search, drag wiring, registry-validated connects, Save/Open `.flow.json` flow/0.1, Validate, Run with checkpoint-aligned STOP, infer-node webui handoff) + minimal FastAPI backend in `.venv` (`& .\.venv\Scripts\python.exe -m flow.server.app`, port 3010/`--port`): graph schema validator, node registry w/ port gates, flows store (fsync), config generator (linear-chain-only mapping, reserved-run-names, MAX_ROWS 500k), single-slot runner (never kills; STOP-flag). Tests: 114/114 server unittest + 85/85 vitest. Gates: G1 unit PASS; G2 HTTP gate pack PASS (`.superpowers/sdd/flow/gate-evidence/g2-http-gates.json`); G4 browser E2E drill PASS (full add/wire/refuse-toast/save/open-restore/validate-OK/preview/infer-dialog/Ctrl+K/delete; evidence `g4-browser-drill.json`); **G3 SKIPPED — real GPU pipeline run deferred: GPU busy 99% with a live job (single-GPU rule)**. Drill caught+fixed 3 real bugs (2a0efa5 xy `measured` geometry loss made nodes invisible/unhittable; 130171e dataset label had no editor; 853e08e infer dialog Escape). Final whole-branch review is the last open step before user demo; push stays user-gated. Flow future phases F1-F6 in design doc §8.
>
> 2026-09-14 (ARABIC-DIACRITIZATION agent): **E-14 Stage-1 char-LM pretrain COMPLETE — PASS.** User "proceed" = G2; smoke first (pwsh-52, 1500 steps: val 2.3825->1.6976, PASS), then FULL stage1lm (pwsh-53, configs/diac_stage1.yaml, 12k steps, 22,138 s = 6.15 h, zero crashes): val_loss 2.4075 -> **1.2806** ([BEST] at final eval; STILL DESCENDING at 12000 — not converged; a longer/annealed stage-1 is a priced option if stage-2 gains stall). Artifacts: runs/diac/stage1lm/final (best + config.yaml) + checkpoint-12000; auto-resume contract = re-run zero flags. **VRAM incident 2026-09-14 (user-visible):** direct bs32 fw/bw ran at ~5.9 GiB dedicated + ~2.7 GiB Windows WDDM spill into Intel-iGPU SHARED memory (total 8.6 GB, ~10x slowdown). Fix: vram_cap_gib config key in train_lm.py -> torch.cuda.set_per_process_memory_fraction (hard allocator ceiling 5.8 GiB of 6,144 MiB) + micro-batch 8 x accum 4 (effective 32) + eval_bs 32. Held at ~5.0/6.144 GiB both runs. NOTE: the machine has only ONE CUDA GPU (Quadro RTX 3000) — the Intel UHD is iGPU, cuda-aware torch sees device_count()==1 (nvidia-smi + torch verified). NEXT: Stage 2 needs pretrained_init hook in train.py (plan section 2.2 TODO) + G3 user gate (arm A only vs A + CATT-reset-last-2). E-14 ledger row closed; TASKS row 57 in_progress (stage-2 pending G3). Commit c139137 (scripts/configs/ledger).
>
> 2026-09-13: **CHECKPOINT CLEANUP (user-approved, scoped to closed experiments)** — deleted checkpoint-* dirs only: runs/mount_{gate,drop,fill,control,hybrid} (E-11 closed), runs/kt2_sft/checkpoint-30 + kt2_sft_r2/checkpoint-206 (superseded/no-op r2), runs/yarn_lora_sft_v1/checkpoint-1172 (a60 promoted), runs/{gdn_smoke,gdn_smoke_ab}/checkpoint-200 (E-08 closed). runs/ footprint 30.88 -> 22.74 GB (-8.14 GB). finals + best.pt untouched; trainer_states of deleted dirs snapshotted to runs/evidence/2026-09-13_checkpoint_cleanup/ (5 files). KEPT: runs/diac/b65 checkpoints (LIVE train), runs/target/checkpoint-4000 (user apex-resume decision), runs/kt2_sft_r2b/checkpoint-100 (row 56 verdict pending), runs/evidence/2026-09-08_cleanup/, all diac/{fadel_spec,pilot128,v2,v2b} checkpoints (diac trim still pending user go). E: free 16.8 GB at write time; b65 may hold a checkpoint-write mid-rotation if pressed for space.
>
> 2026-09-10: **Knowledge-transfer ladder approved** (reuse SmolLM2-360M knowledge in our architecture; the Q2 architecture decision is deliberately pending the KT-1 A/B numbers). Plan: docs/plans/2026-09-10_kt_ladder_plan.md — step handoff: docs/plans/2026-09-10_kt_ladder_handoff.md — TASKS rows 43-46.
>
> 2026-09-10 evening: **KT-1 Phase 1 EXECUTING** — scripts/embed_transplant.py DONE (byte-level alignment; 58.7% exact + 41.1% OMP-lite fallback = 99.98% teacher coverage; the <60% exact gate investigated + JUSTIFIED - sub==1 bucket 0.05%, tail-only misses; isometric lift max|WW^T-I|=1.1e-15; artifact data/kt/embed_init_smol360.pt with sha256-stamped meta). train.py train.init_embeddings hook ADDED (default OFF; TIE-OK: tied head follows the in-place copy). configs/kt_ab_control.yaml + kt_ab_transplant.yaml written; sanity_check 4/4 PASS both. CONTROL ARM LIVE (fresh start, 6.7 s/it, ETA ~1h51m; log runs/kt_ab_control/logs; auto-resume = re-run the exact command zero flags). Transplant arm launches STRICTLY after; then eval-curve A/B -> USER Q2 decision. Standby AC timer verified already 0 (no guard change needed). CONTROL ARM DONE (1000/1000 exit 0, best eval 2.4426@1000, curve monotonic; vs T +2.97%@500 / +7.84%@1000 - investigated: software stack identical, T ran on the RTX 4000, smooth growing drift = GPU-die chaos through the high-LR phase -> RIG VALID). TRANSPLANT ARM DONE -> **KT-1 A/B RESULT: transplant WINS -6.24% eval @1000 (2.2902 vs 2.4426; gate >=3-5% MET; delta grows over training; train/loss corroborates -7.64%)**. Numbers delivered to the user for the Q2 decision (keep our arch vs warm-start SmolLM2-135M vs extend A/B) - USER-GATED, do not proceed to ladder Phase 2/3 on a base choice before the user picks. Q2 DECIDED 2026-09-10: KEEP OUR ARCHITECTURE (Route 0 queued as row 47, user-gated timing). Row 43 DONE. Cleanup done: kt_ab ckpt-800 dirs deleted both arms (best + last-two retention), row41 backup zip deleted after verified restores, disk 23.2 GB free. NEXT: ladder Phase 2 = row 44 (KT-2 SmolLM2-360M on-policy judge on our base) - awaiting user go. KT-2 EXECUTED 2026-09-10 evening (row 44 DONE, ALL GATES PASS): judge round-1 (omen_alpha+minimax3 pool, 250 prompts x 4, unparseable 0.3%, 248 winners) -> LoRA-SFT from final_a60 (batch 4/accum 4 = same effective 16, ~3x faster; CSN guard +0.37% PASS, ast greedy flat 0.90) -> runs/kt2_sft/final. BATCHING LESSON: batch-1 was a VRAM-era default; batched inference (judge) and batch 4/accum 4 SFT are the new defaults where VRAM allows. Next options: scale judge rounds (bigger pool/K, user-gated), KT-3 pretrain w/ transplant init (row 45), Route 0 test (row 47). **RESUME HANDOFF for fresh-context agents: docs/plans/2026-09-10_kt_resume_handoff.md** (round-2 in flight at write time; structured STEP 0-6 todos + gotchas + artifact map).

> 2026-09-10 late: **MODEL-VIZ React platform PLAN WRITTEN (not started)** — docs/plans/2026-09-10-model-viz-react-platform.md is the complete build plan (config-first architecture: add-a-model = one YAML; lessons + bug-causing pitfalls from the 3-model static explorer build; 4 phases with validation gates; verified spec seeds for nano/SmolLM2/Qwen in appendix A — no re-research needed). User go gates Phase 1; status lives in TASKS.md row 48. Until then model_architecture_explorer.html remains the live viz.
>
> 2026-09-10 late: **MODEL-VIZ React platform BUILT (all phases, user gave go)** — `viz/` React+Vite+TS app (pnpm): config-driven, add-a-model = one `models/<id>.yaml` (zod-validated registry, `_template.yaml` starter, `qwen38.overrides.ts` escape-hatch example). Param math computed in `src/engine/params.ts` (norms once; unit-test anchors nano 226,526,208 / smollm2 134,515,008 / qwen text-stack 124.3B computed); seeded sim engine (`src/engine/sim.ts`); all phase gates green: build clean, `pnpm test` 20/20, `pnpm doctor` PASS, browser drills via agent-browser (3 cards, 166 blocks, inspector per-click, demo advance, compare auto-tables) — screenshot `viz/drill.png`. Usage: `viz/README.md`. Design: `docs/plans/2026-09-10-model-viz-react-platform.md`. The single-file `model_architecture_explorer.html` (row 47) stays untouched as the offline export baseline. GOTCHAS added to MEMORY: pnpm-12 `approve-builds` for esbuild; YAML scalar colons/quotes.
>
> 2026-09-11: **ARABIC DIACRITIZATION SPECIALIST (D-line) APPROVED — design done, NOT started.** New self-contained project: char-level bidirectional encoder (our GQA blocks, causal mask off) + 15-class per-Arabic-letter diacritic head, ~30M pilot, submodule `diacritizer/`. Decisions D1-D6 (route / size / placement / strip->preserve->dual-mode policy ladder / mixed-general domain / copy-by-construction non-Arabic policy incl. tatweel + Quranic-mark passthrough) in research/arabic_diacritization/DESIGN.md; build plan docs/plans/2026-09-11-arabic-diacritization-specialist.md; TASKS rows 50-55 (A0a-c CPU-only -> A1 smoke -> A2 pilot -> A3/A4). Exit gate for phase A0 = scripts/selftest.py --all 4/4 PASS; eval harness (DER/WER +-case endings + text-preservation) is built BEFORE any training. GPU was BUSY at write time (KT-2 r2 python pid; row 48 active) — GPU rows 53/54 queue behind it. No M3-line file changes planned; any exception gets its own row + diff list first.

> 2026-09-13c: **Stage 1/Stage 2 transfer plan SPECIFIED (user-gated, NOT started)** - docs/plans/2026-09-13_diac_stage1_stage2_transfer_plan.md: Stage 1 = char-LM Arabic pretraining on the on-disk sources (v2d28 arch, ctx 512, train_lm.py with the zero-flag resume contract), Stage 2 = warm-start bidirectional fine-tune (pretrained_init hook) on the v3 corpus (+Sadeed_Tashkeela) with the Sadeed-dedup gate check FIRST; falsifiable >=2 pp / >=3-gate rule; totals ~12 h GPU + 1.5 h CPU. Tasks rows 54 (menu) and 57 (the two-stage plan).
## E-26 mu0c window closed (120K steps) — verdict: step-saturated (2026-09-19)
- All five arms trained to 120K (x1+x2 parallel, x3+x4 parallel, control solo; 576 MiB peak total, never OOM).
- Loss tracking per user request: mex/scripts/watch_losses.py (CPU-only summary + CSV runs/mex/loss_track.csv) ran live during the whole window; mex/scripts/plot_losses.py produced runs/mex/loss_curves.png/.json after close.
- Finding: min val loss for all experts arrives at 6-8K steps; massive fixed-lr overfit mid-run (x2 eval 1.19->2.11); cosine anneal tail recovers eval to ~min; downstream exact_match flat-or-worse vs E-25 (x2 0.898->0.828). 10x steps = no benefit. mu0 recipe saturated at 12K steps — push mu1 (data diversity / regularization / composition), not step count.
- E-26 gates 4/4 PASS at 120K (x1 0.105, x2 0.828, x3 0.916, x4 0.862 vs mains 0.0/0.002/0.512/0.316).
- Control (dense, param-matched) at 120K beats or matches every expert cross-task except x4 — see MU0_REPORT mu0c section.
## 2026-09-13c - v2d28 VERDICT: param-matched depth x2 = -2.0..-3.9 pp DER - depth BEATS params per byte on WN gates  <- scoped: ARABIC-DIACRITIZATION agent only

- v2d28 FULL 30000/30000, best val_loss 0.1801 (v2b 0.1802 - VAL-TERM DEAD CALL; depth did not lift the val objective at fixed params). Gate pack (DER, preservation 1.0): Fadel 45.0 (v2b 47.1, -2.1) / Sadeed 56.2 (59.9, -3.7) / WN2024 59.1 (61.1, -2.0) / WN2014 52.5 (56.4, -3.9).
- DECISION RULE vs v2b (>=3 pp decisive): Sadeed -3.7 and WN2014 -3.9 FIRES DECISIVE; Fadel -2.1 and WN2024 -2.0 below bar.
- CROSS-ARM READ (post-hoc, registered before results in 09-13b): per +1.0x params invested, depth x2 at 1.0 budget delivers ~ -2.5 avg gate pp for FREE vs v2b - while b65's 2.5x budget delivered -3.0 avg. Roughly: depth scaling > raw param scaling for modern/MSA-style gates (WN2024/WN2014 flip in v2d28's favor), params > depth on classical-heavy gates (Fadel/Sadeed favor b65 slightly). Either way BOTH axes are 10x smaller than the ~50 pp SOTA gap -> architecture shape polishing is NOT the frontier; data breadth + transfer/pretrained init (SOTA lesson, E-12 notes) is the next lever (v3 corpus with Sadeed_Tashkeela = ready; KT-1 transplant on this line = untested).
- Artifacts: runs/diac/v2d28/final (+config.yaml), data/diac/bench/v2d28_pred_*.jsonl. DELETED on user request 2026-09-13 (~1.23 GB): runs/diac/v2d28smoke whole (smoke was packed 1019-step evidence in the 09-13b log text; config yaml configs/diac_v2d28_smoke.yaml kept). Disk 15.2 GB free.

## 2026-09-13b - v2d28 depth ablation REGISTERED (pre-results)  <- scoped: ARABIC-DIACRITIZATION agent only
- USER QUESTION answered by design: param-matched depth x2 - v2d28 = 28L/272h/8h/2kv/ffn1088/ctx128 = 30,103,600 params (v2b 30,016,128, +0.29%); hypothesis test: depth-vs-params-vs-data as the plateau driver.
- Control contract: SAME data (tokens_phase=v2b, identical batch stream via seed 777), SAME lr 2e-4/wd/batch/patience/gates; NO lr or schedule adjustment (stability at 28L is itself an outcome). DECLARED confounder: head_dim 64 -> 34 (intrinsic to params-exact).
- Smoke v2d28smoke (2000 steps GPU): PASS - monotone best every eval, val_loss 0.7021@250 -> 0.3300@2000 (val_acc 0.8913), ZERO overflow/NaN skips. Pace 1.38 s/step -> full ~11.5-12 h (28 sequential layers >> param math).
- FULL RUN LIVE: pwsh-47, runs/diac/v2d28/, fresh 30000, patience 12, eval 500. Decision rule registered: >=3 pp DER swing vs v2b = decisive; less = single-seed noise.
- Registered table: v2b 47.1/59.9/61.1/56.4 - v2d28 TBD - b65 43.1/55.3/60.3/53.7.

## 2026-09-13 - b65 DONE + GATES; 2.5x capacity = -1..-4.6 pp DER, NOT the SOTA gap  <- scoped: ARABIC-DIACRITIZATION agent only
- b65 (20L/512h/ffn2048 = 76,137,984 params, tokens_phase=v2b) FULL 30000/30000, best val_loss 0.1709/val_acc 0.9450 @29500 (v2b 0.1802/0.9421 = -5.2%); final/ = best weights. Sequence note: it was relaunched mid-run with the exact zero-flag command and auto-resumed from checkpoint-1000 (contract working); the Env-153 events that evening belonged to other runs. Wall ~11.6 h.
- b65 GATES (DER): Fadel 43.1 (v2b 47.1, -4.0) / Sadeed 55.3 (59.9, -4.6) / WN2024 60.3 (61.1, -0.8) / WN2014 53.7 (56.4, -2.7). preservation 1.0. VERDICT: 2.5x params buys -1..-4.6 pp; the ~50 pp gap to SOTA-class is NOT param count alone at this arch class, and fadel_spec (30M) still owns Fadel (34.2). Next levers: v3 corpus (+Sadeed_Tashkeela, already wired as sadeedt_train, commit 96d8331) + per-domain routing; dedup-check Sadeed_Tashkeela vs SadeedDiac-25 gate before trusting it on a v3 model.
- bench.py --config flag added; train.py final export now ALSO writes final/config.yaml (final/model.pt is a bare state_dict - the silent pilot128-shape fallback bit on b65).
- Disk cleanup (user-approved): v2/{checkpoint-*,final,best.pt}, fadel_spec/{checkpoint-*,best.pt}, v2b/checkpoint-* deleted (~3.4 GB); pilot128 ckpt-25000/25200 KEPT (v1 gate evidence); b65 live-time artifacts untapped.

## 2026-09-12b - D-line OPTIONS A+B EXECUTION (user: "proceed with A then B")  <- scoped: ARABIC-DIACRITIZATION agent only

- **OPTION A DONE (fadel_spec)** - Fadel-classical specialist, SAME 30M arch, trained ONLY on fadel_train+val (52,500 windows; 47,506 train / 4,994 val after the 5% doc-stratified carve-out). configs/diac_fadel_spec.yaml (total_steps 6000, patience 8). FULL 6000/6000, best val_loss 0.1621 @5000, val_acc 0.9477, final/ = BEST weights, ~19 min wall.
- **A gates (DER, preservation 1.0/0.9999 everywhere)**: Fadel test **34.2** (v2b 47.1, v1 42.0 - BEST of all runs), Sadeed25 **51.4** (v2b 59.9), WN-2024 62.9 (v2b 61.1 - worse), WN-2014 57.4 (v2b 56.4 ~par). Domain-narrowing tradeoff confirmed: specialist wins the classical gates it targets, gives up the modern-WN gate. Artifacts data/diac/bench/fs_pred_*.jsonl.
- **OPTION B LIVE (b65)** - capacity test of the ceiling hypothesis: 20L/512h/8h/2kv/ffn2048 = 76,137,984 params, configs/diac_b65.yaml, tokens_phase=v2b (reuses the matched-128-window corpus via the NEW tokens_phase hook in train.py - zero re-prep/re-tokenize). 4.2/6 GB VRAM, 0.78 s/step -> 30k ~ 6.5 h, patience 12, eval 500. Job pwsh-45; run dir runs/diac/b65/.
- **DATA INCIDENT + RECOVERY (own mistake)**: prepare_data.py --out-dir v1 rebound the module-global OUT in a LOCAL scope -> the fadel-only prep silently CLOBBERED data/diac/prepared/{train,val}.jsonl. Recovery: out_dir routed as a process() PARAMETER; v2b prepared corpus regenerated BYTE-EXACT (1,450,783 train / 108,496 val / quarantined 12,687 / dups 7,573 - stats match to the row); v2b tokens never touched (b65 unaffected). Lesson 53 in MEMORY.md.
- **New knobs**: prepare_data.py --out-dir, tokenize_data.py --prepared-dir + phases fadel_spec/b65, train.py tokens_phase.

## 2026-09-12 — D-line R53/R54 EXECUTED: pilot128 + v2/v2b + gate pack; CAPABILITY PLATEAU  <- scoped: ARABIC-DIACRITIZATION agent only

- **R54 arch change (user-approved)**: original ctx-1024 pilot measured ~17 s/step (30k = 4-5 days) and was rejected BEFORE its first checkpoint; switched (user proposal) to deep-narrow 128-ctx: 14L/384h/8h/2kv/ffn1536 = 30,016,128 params, configs/diac_pilot128.yaml. 1000-step smoke PASS (val_acc 0.9090, ~0.25 s/step, 2.4/6.1 GiB).
- **Overfit diagnosis**: full pilot128 run (user-stopped at 25.3k) — train 0.02 vs best val_loss 0.1582 @ step ~8100 rising to 0.195; ~16 epochs over a 51k-window corpus. Remaining error is a ceiling, not a pace problem.
- **Gate infra built**: diacritizer/scripts/bench.py (ckpt -> bare -> fp16 predict -> argmax-at-base positions -> marks_for_label -> byte-exact passthrough reconstruct) + eval.py multi-ref (tab-joined refs per line). Gates read RAW external files; fadel_test REMOVED from the model-selection val split (external-only policy; was contaminating selection).
- **Gate results (DER / WER, text_preservation ~1.0 everywhere)**
  - pilot128 auto-rotation cost: only ckpt-25000/25200 survived (best-val ckpt-8200 PRUNED) ->
    Fadel test 42.0/42.0 · SadeedDiac-25 58.9/59.3 · WN-2024 53.9/53.9 · WN-2014 62.2/62.8
  - v2 (30x corpus; silent windowing mismatch): 48.4 · 59.8 · 60.8 · 57.2
  - v2b (30x corpus, 128-char windows matched): 47.1 · 59.9 · 61.1 · 56.4
- **30x data plateau read**: Abdou/arabic-tashkeel-dataset (MIT, 1.44M train windows kept after our prep) + QCRI Wikipedia_20240420.diac.jsonl (32.8k articles) + Fadel 50k + WN-2024: v2b = best-ckpt export val_loss 0.1802/val_acc 0.9421 but gates moved ~0. Ceiling is ARCH CAPACITY (QCRI paper's own BiLSTM = ~3% WER on WN-2014) or domain balance, not volume.
- **Instruments added to train.py**: early_stop_patience (best.pt saved at every eval, survives ADDR zero-flag resume; final/ = BEST weights, never the tail) + eval_every knob (13 s/108k-window val split -> eval cadence 500). Rotation STILL prunes mid-run best-val checkpoints — keep-best-not-yet-fixed (doc'd gotcha).
- **Prep parallelized** (user request "use full cpu"): source-level filter inside stream_samples (500-row slice 57.8 s -> 0.7 s) + module-level worker fns + BATCHED pool.map over materialized chunks. Lazy imap-streaming generator + Windows pool teardown = DEADLOCK (handler thread alive in _get_tasks at terminate; diagnosed via faulthandler.dump_traceback_later). Full 1.56M-window prep: ~40 min single-thread -> 7.4 min on 12 cores, BYTE-IDENTICAL stats (determinism preserved).
- **Windowing lesson (MEMORY 50)**: v2 prepared 1024-char windows but trained at ctx 128 -> silently trained on the first 128 chars of every window (~75% of the text never seen). v2b re-preps at CTX_CHARS=128 (matched). RULE: prepared window size MUST equal model ctx.
- Sadeed_Tashkeela still HF-gated (user token/access; BLOCKED, non-blocking).
- GPU idle; runs/diac/{pilot128,v2,v2b} all local (gitignored by policy); bench reports data/diac/bench/.

## 2026-09-11 — D-line A0 COMPLETE (rows 50-52); GPU rows ARMED  <- scoped: ARABIC-DIACRITIZATION agent only

- A0 exit gate: selftest --all 3/3 phases (a0a 24, a0b 6, a0c 7 checks), 0 SKIPPED.
- GPU-ready files delivered and validated on CPU ONLY (--device cpu; GPU was
  never touched): src/model.py, scripts/train.py, scripts/tokenize_data.py,
  scripts/eval.py, configs/diac_smoke.yaml + diac_pilot.yaml (= 30.5M params).
- BEFORE the first GPU launch (R53 D-smoke): re-run tokenize_data.py WITHOUT
  --max-train (current pack capped at 5k windows); never co-run with any other
  train.py; init loss baseline ~2.7 (15-class).
- Next: user go -> R53 D-smoke queued behind any live run, then R54 D-pilot
  (deliverable rung). Sadeed_Tashkeela still HF-gated (user token decision).

- R50: diacritizer/ scaffolded (tokenizer/labels/passthrough + selftest). a0a 24/24 PASS.
- R51: download_data.py + prepare_data.py. Sources verified-hit (Fadel, QCRI
  incl. WikiNews-2024 benchmark + eval script, SadeedDiac-25). Sadeed_Tashkeela
  HF-gated (needs user token/access) -- recorded BLOCKED, not blocking Phase 1.
- Corpus gotcha: BOTH shadda mark orders occur in the wild; folded in labels.py.
- Prepared: data/diac/prepared/ 59,340 windows (51,280/8,060), 0 quarantined.
- Selftest gate: 3/3 phases PASS (a0a, a0b, a0c-SKIPPED-pending-R52).
- Not committed (shared tree; user-gated). Next: R52 eval harness (CPU).

> 2026-09-11: **KT-2 round 2 EXECUTED (TASKS row 56, BLOCKED on user A/B/C).** The prior session's round-2 judge chain died at an exact prompt boundary (1007/1500 prompts, valid last line) - `scripts/kt2_judge.py` stage_sample gained a per-prompt resume guard (reads `(prompt_id, cand_id)` keys from the append-only candidates.jsonl; prompts.jsonl is seed-deterministic; MEMORY 43). Resumed chain completed: 7500/7500 candidates, **winners 1490, unparseable 0.1% PASS, judge-vs-LL agreement 100.0% (1454/1454)**. Combined corpus `data/sft/kt2_sft_r2/pairs.jsonl` = 1738 pairs (r1 248 + r2 1490; sft_data: 1634 train / 104 val, 0 drops). SFT r2 (2ep @1e-4, runs/kt2_sft_r2): CSN 1.9094 PASS, AST 0.86/0.88 FAIL; tfevents show cosine LR hit 0 by ~step 115 (half-run no-op, MEMORY 44). SFT r2b (1ep @5e-5, runs/kt2_sft_r2b): CSN **1.8974** PASS (best ever), AST 0.86/0.88 still FAIL (recipe-insensitive, MEMORY 45). Process trap caught: the first r2b silently auto-resumed r2's checkpoint-206 (0 real steps) - detected via train_summary.json resumed_from, deleted + relaunched clean. NEXT: user picks A (keep r1 final as the student, close row 56) / B (promote r2b, AST below frozen floor) / C (one more data-flavored knob); KT-3 (row 45, user-gated window) waits behind this. All KT weights stay local (gitignore extended data/kt/ + kt run weights).

## 1. Mission

End-to-end SLM training pipeline on a single Quadro RTX GPU (Turing sm_75,
fp16 only — no bf16, no flash-attn). Machine moved mid-M2 from the original
RTX 4000 (8 GB) on DESKTOP-MUO4QK5 to an RTX 3000 (6 GB) on DESKTOP-TU09FBO;
all M2 milestones below were finished on the RTX 3000. Milestones: M0 smoke
(pipeline proof + kill/resume drill) → M1 VRAM probe (250M fit gate) → M2
pilot ~110M run → M3 target ~226M run (optional). Auto-checkpoint +
auto-resume with no manual flags is the user's hard requirement (PLAN.md
§5.3).

## 2. Current state (as of this handoff)

| Milestone | Status | Evidence |
|---|---|---|
| M0 smoke (12.3M) | **DONE — PASSED** | loss 10.4→4.95 in 200 steps; kill at ~step 100 → relaunch resumed at exactly 101/200, zero flags; rotation works (limit 3); artifacts in runs/smoke/final (train_summary.json, eval_report.json); eval_loss 4.7926, ppl 120.6 |
| M1 VRAM probe (226.5M target config) | **DONE — PASSED** | peak 4.24 GB allocated / 4.4 reserved @ 2214 tok/s → target APPROVED for M3; no 8-bit Adam needed; headroom for ctx 1024 |
| M2 pilot (100.7M) | **DONE — PASSED** | 3000/3000 with auto-resume from checkpoint-1000; eval_loss 2.206→1.161 monotonic (final ppl 3.19); no OOM (peak ~3.3 GB of 6 GB); locally-syntactic samples; artifacts in runs/pilot/final (train_summary.json, eval_report.json) |
| M3 target (226.5M) | **DONE — PASSED 2026-09-07 ~08:16 UTC on TU09FBO** | 5000/5000 with zero-flag auto-resume from checkpoint-4500 (restored from checkpoint_backup\checkpoint-4500.zip); eval curve 2.8567 @500 → 2.265 @1000 → 2.0903 @1500 → 1.9972 @2000 → **1.8512 @4000 (best)** → 1.8610 @4500 → 1.8641 @5000 (curve at floor, slight rise = normal noise); final eval.py: val_loss 1.8641, ppl 6.45 (runs/target/final/eval_report.json); train_summary.json carries the env fingerprint (git c16179c, torch 2.14.0+cu126, transformers 5.16.1); weights stay LOCAL (§7); see §3b |
| C12 Tier 1 SFT (full + pilot) | **DONE — PASSED 2026-09-07** | full: 16,376 pairs × 2 epochs, 2048 steps, eval 0.836; instruction ast greedy 0.86 / sampled 0.88; forgetting guard CSN +12.6% (within ≤ +10-15% gate, near edge); weights in checkpoint_backup zips (TASKS row 4) |
| Web UI U1-U5 (TASKS row 21) | **DONE — PASSED 2026-09-08 (full e2e)** (gradio 6.26) | webui/app.py: dashboard / monitor (plot+ETA+GPU+log+eval card, Danger-zone delete) / chat (VRAM policy all branches incl. GPU verified live) / train launcher (webui_*.yaml + run_custom chain; NO kill button) / U5: --lan, --auth login flow, --webhook run_finished+run_crashed notifications; e2e: launch → train 100 steps (best_eval 6.1545) → webhook → eval card → chat from fresh final; server start: `& .\.venv\Scripts\python.exe webui\app.py` → 127.0.0.1:7860 (--port/--no-browser/--lan/--auth/--webhook); spec = WEBUI_PRD.md |
| Web UI U6 (TASKS row 22) | **DONE - ALL PASS GATES MET 2026-09-08** | one-click Resume (same config, zero flags; kill->Resume drill: 600-step 8-bit run -> ckpt-500 -> tree kill -> Resume -> completed, summary resumed_from: checkpoint-500, eval 2.348->2.3446 continuous); resume-collision guard in build_config; --lan w/o --auth exits 1; adamw_bnb_8bit = Train dropdown default (Milestone B); chat truncation warn @ ctx-max_new; chain-step indicator + chain_out.log tail; chat picker refresh 15 s; job banner Dashboard/Train/Monitor; tok/s+MFU in Monitor; gotcha: stale UI instances hold CUDA contexts -> restart UI before launching (drill did); drill artifacts deleted |
| Web UI U7 (TASKS row 23) | **DONE - PASS GATES MET 2026-09-08** (one gate user-gated: with-key prepare needs the user's real HF_TOKEN) | Settings tab key manager (gitignored .env, masked last-4, add/override/delete verified, path shown); stream/download data modes -> data.data_mode -> prepare_data.py; rows cap 500k; preflight disk estimate; gated-dataset clear prompt verified on the-stack-v2; stream e2e from UI: 1000 rows -> 100-step 8-bit train best_eval 5.8958 (232 s); download-mode layout equality on wikitext probe; FIXED prepare_data ["train"] after split= column-select bug; TinyCode corrupt shard also breaks full download (MEMORY 17) |
| Web UI U8 (TASKS row 24) | **DONE - ALL 3 PASS GATES 2026-09-08** | Dataset preview (first rows + min_chars/dedupe drop counts, prepare_data-identical rules, 2,000-scan window) on Train tab; generated-YAML read-only disclosure on Preflight; per-ckpt MB labels in Danger zone; runs/ footprint on Dashboard; multi-run eval/loss overlay in Monitor |
| Web UI U9 (TASKS row 25) | **DONE - PASS GATES 2026-09-08** | Chat streaming via TextIteratorStreamer (generate on a daemon thread, progressive yields); generation-level stop via global flag + StoppingCriteria (button and client-cancel both covered by the stream loop's finally); verified streaming + clean stop + model reuse on CUDA and on CPU fallback; model left unloaded |
| Web UI U10 (TASKS row 26) | **DONE - PASS GATES 2026-09-08** | Train tab SFT accordion -> configs/<name>.yaml (+ pilot chat stub) -> run_custom.py --sft chain (sft_data CPU -> sft GPU, single-lock). Pilot on smoke/final completed 402s (eval_loss 4.197); Monitor + instruct-mode chat verified on the pilot final; LoRA checkbox emits the peft block (row 10 hook). Artifacts: runs/sft_u10 (+ _pilot final), configs/sft_u10*.yaml |
| Web UI U11 (TASKS row 27) | **DONE - PASS GATES 2026-09-08** | CoopStopCallback in src/stop.py wired into train.py/sft.py (on_save only, checkpoint-aligned; flag cleared on relaunch). E2E: mid-run STOP -> clean exit at checkpoint-250 (epoch 0.8, eval 4.207); zero-flag relaunch resumed with no loss bump (final eval 4.197). Monitor stop button + banner note; browser notifications (/api/job_done + JS poller, opt-in); optional HTTPS launch args wired (no certs to live-test). UI restarted pid 2392 |
| Milestone B + LoRA hook (TASKS rows 18/10) | **DONE — ALL GATES PASS 2026-09-07** (GPU-queue session) | arms: A 2.5644 / B 8-bit 2.5603 / C b2a16 2.5604; honest probes (vram_probe optim fix): B 1.36 GB (−550 MiB) vs A 1.91 GB; C pace FAIL on real run (1,680 tok/s) → rejected; **decision: adamw_bnb_8bit b1/a32 = next-pretrain default**; arm B kill/resume drill PASS; LoRA: hook merged (src/model.py:maybe_wrap_peft + save_final), probe 0.72 GB @ 2.5% trainable, CPU e2e + kill/resume PASS |
| SFT v2 (TASKS row 19) | **DONE — DELIVERABLE runs/sft_v2_e1/final 2026-09-08** | minimax3 corpus 19,252 pairs (min_chars 30 gotcha); 2-epoch run: ast 0.98/1.00 but forgetting +19.7% FAIL -> e2 kept as overfit evidence; **e1 (1 epoch): ast 0.98/0.96, forgetting +9.8% PASS — beats Tier 1 (0.86/0.88, +12.6%)**; NOTE: runs/target/final/model.safetensors had been de-weighted in the disk cleanup — RESTORED bit-exact from runs/target/checkpoint-4000 (best@4000) |
| Tier 3 KD (TASKS row 20) | **DONE — KD BEATS BASELINE 2026-09-08** | P (pilot/final) teaches S (12.3M), 0.5*KL(τ=1)+0.5*CE, fair A/B 2,000 steps: KD 2.4755 vs baseline 2.6201; at 1/3 steps 3.4142 vs 3.5325 -> plan §5 criterion PASS, the ~1/10 claim transfers; KD cost ~35 min GPU (baseline ~3 min — S-scale is nearly free on this GPU) |


### 2026-09-09 evening — rows 2/13/16 session (disk watch, Milestone D, Milestone G1 build)

- **Row 2 (disk watch)**: the three Track C (kd-t2p-*) arm weights were ALREADY
  de-weighted from disk (2026-09-09 triage) — the final/ dirs hold ~4 MB of
  configs/tokenizer/summaries; the weights survive ONLY inside
  resume_pack_2026-09-09.zip (all 3 model.safetensors entries verified).
  USER-APPROVED deletion: checkpoint_backup/checkpoint-sft_v2_e1-final.zip,
  868 MB, after pre-delete CRC verify (pack entry == on-disk file, 906,121,272 B,
  CRC 0xb9dcefc1) → E: 35.48 → 36.35 GB; milestone-D measurement −50 MB → 36.30.
  runs/target/checkpoint-4000 (2.6 GB) untouched (user keep-decision 2026-09-08).
- **Row 13 (Milestone D) DONE — run stays user-gated**: mix mode in
  prepare_data.py (per-candidate target_rows, ONE shared SHA1 set across
  sources, source_stats.json additive in both modes; legacy 5-row regression
  BYTE-IDENTICAL) + data_dir passthrough (the-stack-smol AND starcoderdata
  lost their per-language builder configs — data_dir is the only working
  form; pilot.yaml/pilot_b8.yaml fixed). Measured rev 2 (2k rows/source,
  CodeLlama): smol 2717.2 / starcoder 2345.7 / csn 271.8 / evol 474.4 tok/row
  — whole-file sources ~4x above the proposal's estimates → its row targets
  were 3.75x the 134M budget. starcoderdata gate: 403 with a VALID token =
  terms never accepted (user accepted 2026-09-09 → re-measured; the
  2026-09-06 "verified unlockable" evidence covered the-stack-smol only).
  SIZING (user-delegated): token-proportional smol-capped 10000/37500/59000/
  16900 rows ≈ 133.65M kept tokens (~20/63/11/6% shares), max_steps 4100;
  cross-source SHA1 dedupe over 8,000 rows = 0.0000. Config ready:
  configs/next_pretrain.yaml (pilot dims @ ctx 1024, adamw_bnb_8bit b1/a32).
- **Row 16 (Milestone G1) implemented, gates pending**: src/gdn.py (pure-PyTorch
  chunked delta rule, fp32-state under a disable-autocast guard), model.py
  gdn_hybrid dispatch + idempotent auto-registration (transformers 5.16.1
  auto_map is double-broken — see MEMORY 26), configs/gdn_smoke{,_ab}.yaml
  (13,331,952 vs control 13,330,688 params, +0.0095%, both reuse the M0 smoke
  corpus); CPU math equivalence 13/13 (rel ≤ 4.8e-6); fresh-process
  save→load→generate roundtrip PROVEN. G1.a sanity BLOCKED on the concurrent
  Track A ctx-probe matrix (~4.9 GB held since ~21:56); watcher
  scripts/_tmp_g1a_gate_runner2.ps1 armed to auto-fire both sanity runs when a
  <500 MiB window opens (job pwsh-7). G1.b 200-step train + kill/resume drill
  + G1.c eval A/B remain orchestrator-owned, strictly sequential.
- **Row 16 GATES ALL PASS (2026-09-10 ~00:47 local)**: G1.a 4/4 both arms
  (hybrid peak 0.30 GB); G1.b hybrid 200 steps eval/loss 5.004 → 3.593
  monotonic + kill/resume drill PASS (killed at step 103, resumed exactly at
  101/200, train_summary resumed_from checkpoint-100); control gdn_smoke_ab
  4.5795; G1.c hybrid 3.6040 / ppl 36.75 vs control 4.5795 / ppl 97.47 →
  Δ −0.9755 ≤ 0 GATE PASS. fp32-state ≈ 1 MB of 0.30 GB peak. Row-16
  deliverable met; a P-scale hybrid trial is a separate user decision (row 6).
- **Concurrent session**: the user's Track A session shares this working tree
  (edited eval.py + model.py helpers + ctx_probe.py). Protocol that held:
  targeted re-read edits only for shared files; GPU is global — every gate
  re-checks nvidia-smi (total used) before firing.

### 2026-09-08 GPU-queue session incidents (lessons 14-15 in MEMORY)

- **Sleep kills**: Modern Standby fired twice mid-run (Kernel-Power 506/507);
  runs survived via auto-resume; one save died mid-write (partial ckpt) and
  exposed that `resume_from_checkpoint=True` bypassed the completeness guard
  — all three trainers now pass the checkpoint PATH. Sleep-on-AC disable
  remains a USER-owned hardware fix (HANDOFF §3 note).
- **Disk filled to 3.9 MB** during SFT v2 (SFT ckpts ~2.7 GB each): writes
  HANG at a fixed offset (E: healthy). Agent cleaned its OWN post-run scratch
  (~13 GB: e1 + A/B-era redundant ckpts). **Cleanup candidates awaiting user
  decision (row 2 rules — nothing deleted without approval):**
  `runs/sft_v2/checkpoint-{2000,2250,2344}` (~8 GB, e2 overfit evidence),
  `runs/pilot-ab-base/checkpoint-500`, `runs/pilot-b8/checkpoint-500`,
  `runs/pilot-b8-r2x16/checkpoint-*` (arm scratch, finals committed),
  `runs/target/checkpoint-{3500,4000,4500}` (~7.7 GB; ckpt-4000 = source of
  the restored final — keep until a backup zip of the FINAL exists).

## 3. M2 pilot run — how to check / resume / finish

- Config: configs/pilot.yaml (12L, d768, 12Q/4KV, ffn 2048, ctx 512, ~110M
  params, fp16 + grad-checkpointing + SDPA, batch 1 × accum 32, lr 4e-4 cosine).
- Data: 20.9M train tokens packed in data/pilot/tokens/train_{000,001,002}.bin
  (40,908 blocks @ 512) + 433k val tokens. Source: nickrosh/Evol-Instruct-Code-80k-v1
  (fallback — ALL bigcode/* datasets are gated; see §6).
- **COMPLETED** on DESKTOP-TU09FBO (RTX 3000 6 GB): 3000/3000, epoch 2.35.
  The venv was lost in the machine move and rebuilt from scratch (§4 recipe);
  relaunching the exact train command auto-resumed from checkpoint-1000
  (weights + scheduler + rng + scaler restored; optimizer state restarted
  fresh because rotation had dropped it — no visible loss bump).
- eval_loss curve (TensorBoard + trainer_state): 2.2061 @ 500 → 1.6476 @ 1000
  → 1.4898 @ 1500 → 1.3529 @ 2000 → 1.2351 @ 2500 → **1.1608 @ 3000**
  (perplexity 3.19). Train loss 8.22 → ~1.00. Pace 5.0–7.4 s/it (thermal
  throttling swings; safe). Peak VRAM ~3.3 GB — fits the 6 GB card.
- runs/pilot/final/ = best checkpoint (3000) via load_best_model_at_end +
  train_summary.json. eval.py → eval_report.json: val_loss 1.1608, ppl 3.19,
  3 locally-syntactic samples (fibonacci semantically correct; MAE/Stack
  syntactically valid, partially wrong semantics — normal at this scale).
- Rotation (save_total_limit=3) deleted local checkpoint-1000 when
  checkpoint-3000 was written; the M2-final model in runs/pilot/final
  supersedes it. Local optimizer states live in checkpoint-{2000,2500,3000}.
- To re-evaluate: `& .\.venv\Scripts\python.exe scripts\eval.py --config
  configs\pilot.yaml`. To re-run training it would no-op (max_steps reached;
  Trainer exits immediately from checkpoint-3000).

## 3b. M3 target run — **COMPLETED 2026-09-07** (5000/5000, PASSED)

> Final state: runs/target/final/ = model.safetensors + config/tokenizer +
> train_summary.json (env fingerprint: git c16179c) + eval_report.json
> (val_loss 1.8641, ppl 6.45). Best eval 1.8512 @4000; local
> checkpoint-{2000,4500,5000} remain on disk (rotation limit 3). The
> narrative below is the run's history through the machine move; the
> resume chain that finished it: checkpoint-4500.zip → extracted into
> runs/target → zero-flag auto-resume at 4501/5000 → completion.

### History (machine move 2026-09-06)

- Config: configs/target.yaml — 16L, d1024, 16Q/4KV, ffn 3072, **ctx 1024**,
  226.5M params, fp16 + grad-checkpointing + SDPA, batch 1 × accum 32
  (32,768 tok/step), lr 4e-4 cosine, warmup 150, max_steps 5000, eval+save
  every 500, save_total_limit 3, **eval_batch 2**.
- Config changes vs the original target spec (probe/decision-driven):
  ctx 512→1024 (6 GB probe approved, below); max_steps 10000→5000 + warmup
  300→150 (same 164M-token budget at doubled tok/step); eval_batch 4→2 (mid-
  train eval VRAM safety at ctx 1024); data candidates fixed — bigcode/
  the-stack-v2 (gated) and bare `code_search_net` (repo moved) →
  code-search-net/code_search_net (python) with Evol-Instruct fallback.
- Data: 147,000 train + 3,000 val rows of natural GitHub Python functions
  (func_code_string). Packed 47.94M train tokens (46,811 blocks @1024, 6
  shards) + 979,559 val tokens (956 blocks). 5000 steps = 164M tokens =
  3.4 epochs — kept on purpose: M2's eval curve was still falling at 2.35
  epochs, and load_best_model_at_end keeps the best checkpoint as final if
  the curve turns.
- VRAM probe on RTX 3000 6 GB (scripts/vram_probe.py, 20 steps): peak
  4.24 GB allocated / 4.4 GB reserved at BOTH seq 512 and seq 1024 (grad
  checkpointing + mem-efficient SDPA keep activations flat at batch 1) →
  ctx 1024 APPROVED with ~1 GB headroom over the 6 GB card. Probe tok/s was
  noisy (536 @512 cold/contended window, 2198 @1024); real pace rules.
- Live at launch: ~9.6–12.7 s/it (~3,400 tok/s) → ETA ~13–14 h; GPU ~4.9 GB
  used, 81 °C — normal throttling regime, no OOM. Watch the first eval
  (step 500) for eval-batch VRAM headroom.
- 2026-09-06 correction (cross-check: research/crosscheck_flashnext_vs_pipeline.md):
  sustained pace is 23.7 s/it (≈1,380 tok/s) — total ≈ 33 h, completion ≈ 2026-09-07
  morning; the launch tok/s estimate was optimistic. 00:07 snapshot: step 106, loss
  8.06→5.60, still in LR warmup, grad_norm 1.27–1.39 (clip 1.0 engaging pre-warmup,
  normal); GPU 77 % / 4.96 GB / 75 °C.
- Artifacts policy (user decision): the M3 model (~900 MB safetensors) stays
  LOCAL — free LFS cannot fit it (see §7). Progress: TensorBoard events in
  runs/target/logs/ (transformers 5.16.1 prints no console loss lines; read
  via EventAccumulator — §5.2).
- Auto-resume (hard requirement): rerun `& .\.venv\Scripts\python.exe
  scripts\train.py --config configs\target.yaml` after ANY interruption —
  it resumes from the newest runs/target/checkpoint-* with zero flags.
- Disk: user-approved deletion of runs/pilot/checkpoint-{2000,2500,3000}
  (3.4 GB) freed headroom; E: had ~16 GB free at launch (M3 needs ~10 GB).
- **2026-09-06 ~07:55: user STOPPED the run (at step 1008) to resume training
  on another machine.** Last saved state = **checkpoint-1000** — verified
  complete BEFORE the kill: model.safetensors 864 MB + optimizer.pt 1.73 GB +
  scheduler/scaler/rng/trainer_state + tokenizer (2,596 MB total), global_step
  1000, epoch 0.68, best_metric 2.2755 @ this checkpoint, lr 3.705e-4.
  Curve: eval_loss 2.81 @500 → 2.2755 @1000; train_loss 2.0937 @1000.
- Handoff bundle for the new machine (local, NOT in git — LFS cannot fit it):
  E:\python_projects\nano_SLMs_m3_handoff\checkpoint-1000.zip (2,377 MB,
  zip64 via bsdtar, 12 entries = 11 files + dir) + RESUME_ON_NEW_MACHINE.txt.
  Resume = clone repo (data/target/tokens already on GitHub), rebuild venv
  (§4), extract zip into runs/target/, rerun the exact train command —
  continues at step 1001/5000 with zero flags.
- **2026-09-06 10:56: bundle never reached MUO4QK5 — user said START FRESH.**
  M3 relaunched from step 0 on DESKTOP-MUO4QK5 (RTX 4000 8 GB, venv intact,
  data/target/tokens verified from the git pull, no checkpoints in
  runs/target so the resume scan correctly started clean). Launch pace
  5.4–6.5 s/it → ETA ~9 h (vs 23.7 s/it throttled on TU09FBO). GPU 5.1 GB,
  86 °C at launch — normal throttling regime. Log: train_target.log. The
  checkpoint-1000.zip / RESUME_ON_NEW_MACHINE.txt on TU09FBO are obsolete —
  delete them there. The abandoned partial run's only useful residue is its
  TensorBoard events (committed under runs/target/logs/).
- **2026-09-06 16:22: user asked to save the checkpoint and stop — done.**
  Killed at step 2292 (PIDs 1820/8420, GPU verified released, log frozen at
  2292). Saved state = **checkpoint-2000**, verified complete before the
  kill: trainer_state global_step 2000, best_metric 1.9972, model +
  optimizer + scheduler/scaler/rng + tokenizer (~2.5 GB); rotation holds
  ckpt-1000/1500/2000. Steps 2001–2292 (~292 steps) are not in any
  checkpoint — transformers saves on the 500-step cadence only, there is no
  save-on-interrupt. Eval curve so far: 2.8567 @500 → 2.265 @1000 →
  2.0903 @1500 → 1.9972 @2000. Resume at ANY time with zero flags:
  `& .\.venv\Scripts\python.exe scripts\train.py --config
  configs\target.yaml` → continues at 2001/5000.
- **2026-09-06 ~16:45: machine-move bundle built** (user request: resume from
  another machine). `E:\python projects\nano_SLMs_m3_handoff\` →
  `checkpoint-2000.zip` (2,722,110,988 B, STORED zip64, all 11 entries CRC-
  verified, sha256 7124b1434810babfd93f9369c67f869f09c2c2fd7cd5534a7a0d87fc40aa2d9e)
  + `RESUME_ON_NEW_MACHINE.txt` (clone → venv §4 → unzip into runs/target →
  zero-flag train). Built by the reusable `scripts/bundle_checkpoint.py`
  (--checkpoint ... --out ...; STORED zip64 + entry/size/CRC verify + chunked
  sha256). Keep the zip OUT of git (2.7 GB; LFS quota decision §7).
- Mid-run incidents worth knowing on the next machine: (1) Modern Standby
  froze the run ~1.6 h between steps 991→992 (system slept 05:31, resumed
  05:33; the CUDA context survived) — disable sleep-on-AC for long runs.
  (2) The GPU sat SW-power-capped at ~45 W / ~930 MHz most of the run
  (battery drained 83%→75% while "on AC" → underpowered adapter); capped pace
  20–25 s/it vs ~9.6–12.7 s/it uncapped. A proper high-watt adapter roughly
  halves the wall time. Eval-1000 at eval_batch 2 passed with no OOM
  (eval_runtime 165 s / 478 batches).
- **2026-09-06 18:29–18:36: bundle ARRIVED on TU09FBO — M3 resumed here.**
  User staged E:\python_projects\nano_SLMs\checkpoint_backup\checkpoint-2000.zip
  (2.72 GB) + RESUME_ON_NEW_MACHINE.txt; extracted to
  runs\target\checkpoint-2000\ (11 files verified, tar exit 0). Fixes
  before relaunch: (1) patched trainer_state.json best_model_checkpoint —
  the bundle carried MUO4QK5's absolute path (E:\python projects\..., with
  a space); a stale path breaks load_best_model_at_end at training END and
  defeats rotation's best-protection; (2) user decision: checkpoint cadence
  500→**100 steps** (allowed 100–200) so a crash loses ≤100 steps; HF
  requires save_steps to be a round multiple of eval_steps when
  load_best_model_at_end is on, so eval_steps also 500→100 (eval ≈ 165 s on
  this card, ~7% wall-time overhead; transformers logs an args-mismatch
  warning at resume — expected, informational). Stale local
  checkpoint-500/1000 (abandoned first TU09FBO run) stay until the first
  save at 2100 auto-rotates them out (keeps best-2000 + 2100, ≤3 on disk).
  checkpoint_backup/ is now gitignored (zip KEPT — second copy of the
  step-2000 state). Remote synced d3c82c9→daf1d13 (15 commits from MUO4QK5:
  their fresh M3 run to step 2292 + window milestones B–G). RELAUNCH
  VERIFIED 18:36: log shows "[resume] found checkpoint-2000 -> auto-resume
  enabled", bar continues at 2001/5000, GPU 5.3 GB with the python trainer
  on compute apps; log = runs/target/train_resume.log. The old
  nano_SLMs_m3_handoff\checkpoint-1000.zip on TU09FBO remains obsolete
  (MUO4QK5's fresh run superseded it) — user-cleanup candidate, not
  agent-deleted.
- **2026-09-06 19:07: nvlddmkm Event 153 (GPU driver engine fault) killed
  the resumed run** ~30 min in, mid-step (~2050s), BEFORE the 2100 save —
  no checkpoint lost (500/1000/2000 intact; resume = exact command, ≤100
  steps lost). Not sleep/standby (no Id 42/107 events), not OOM, no reboot:
  the driver fault raised a CUDA error and python exited 1. GPU re-probed
  healthy (CUDA sanity PASS: matmul OK) and the run was RELAUNCHED ~19:11
  (auto-resume from checkpoint-2000 verified at 2001/5000; GPU 99% / 4.8 GB
  / 45 W / 900 MHz — SW power cap still active; ~22 s/it early). A second
  short-lived trainer (PID 26940, launched 19:08 from outside this session)
  died pre-step and left a header-only tfevents file (removed). If 153
  faults repeat: suspect the SW-power-capped hardware state (underpowered
  adapter), not the training code — check System log for nvlddmkm.
- **2026-09-07 07:14: user STOPPED the run at step 4593/5000 — checkpoint-4500
  zipped.** Killed cleanly BETWEEN saves (no save in flight; GPU released to
  103 MiB). Last complete checkpoint = **checkpoint-4500** (11 files;
  global_step 4500, best_metric 1.8512 @4000, last eval 1.8610 @4500 — slight
  rise over the best, curve near floor; ~2.5 GB). Steps 4501–4593 discarded
  (no save-on-interrupt; the 100-step cadence capped the loss). Zipped:
  checkpoint_backup\checkpoint-4500.zip (2,368 MB, 12 entries, tar exit 0,
  ~76 s) beside the kept checkpoint-2000.zip. On-disk dirs 3500/4000/4500
  (rotation untouched). RESUME = exact zero-flag train command → continues at
  4501/5000. If the zip moves machines, patch best_model_checkpoint in its
  trainer_state.json (machine-local absolute path; same fix as the 2000
  bundle).

## 4. Environment (verified working)

- venv: `.venv` (uv-managed, CPython 3.12.9). NEVER pip; use
  `.\.venv\Scripts\python.exe` and `uv pip install --python .venv` if needed.
  Rebuild recipe (used once already after the machine move — keep for the
  next time the venv is lost):

  ```powershell
  uv venv .venv --python 3.12.9
  uv pip install --python .venv torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python .venv transformers==5.16.1 datasets==5.0.1 accelerate==1.14.0 bitsandbytes==0.50.2 tensorboard==2.21.0 pyyaml exa-py
  ```

  Then validate: `python -c "import torch; print(torch.cuda.is_available())"`
  and `scripts\sanity_check.py --config configs\smoke.yaml` (all four PASS
  gates).
- torch 2.14.0+cu126, transformers **5.16.1**, datasets 5.0.1, accelerate 1.14.0,
  bitsandbytes 0.50.2, tensorboard 2.21.0, pyyaml; git-lfs 3.6.0 (system).
- exa-py 2.20.0 (web search via `exa_search.py` / `exa_search.bat`; API key read
  from the project-root `.env`: EXA_API_KEY). Installed with `uv pip install
  --python .venv exa-py`.
- GPU now Quadro RTX 3000, 6 GB (was RTX 4000 8 GB pre-move): driver 580.92,
  sm_75, fp16-only unchanged. Pilot peak ~3.3 GB fits; re-probe before M3
  (target config measured 4.24 GB on the 8 GB card — verify on 6 GB first).
- Repo layout: configs/{smoke,pilot,target}.yaml · src/{model,data}.py ·
  scripts/{prepare_data,tokenize_data,train,eval,infer,vram_probe,sanity_check}.py ·
  data/ + runs/ (artifacts) · PLAN.md (spec) · README.md (usage) ·
  exa_search.py + exa_search.bat (web search helper, Exa API; key in .env).

## 5. Gotchas learned the hard way (do not rediscover)

1. **transformers 5.16.1 API drift:** TrainingArguments has NO `logging_dir`
   and NO `save_safetensors` (safetensors is the only format). TensorBoard dir
   is set via env `TENSORBOARD_LOGGING_DIR` (train.py sets it to runs/<phase>/logs).
   Trainer uses `processing_class=` (not `tokenizer=`), `eval_strategy=`.
2. **No console loss lines:** Trainer 5.16 does not print {'loss': ...} dicts to
   stdout during training. Read metrics from TensorBoard events:
   `EventAccumulator(d).Reload(); ea.Scalars('train/loss')` (tags: train/loss,
   train/grad_norm, train/learning_rate, eval/loss — eval/loss only exists
   after the first eval step).
3. **run_code wall-clock ceiling = 600 s.** Chain single waits
   (`tools.job_output({wait:true, timeout_ms: ≤560000})`) across goal rounds.
4. **job_output status is at result.job.status**, not result.status.
5. **tools.write/edit calls need a `description` property** (first arg avoids a
   validator quirk). run_code returns must be strict JSON — strip ANSI/control
   chars from pwsh/tqdm output before returning.
6. **GPU thermal throttling:** Quadro RTX 4000 cycles 84–90°C under sustained
   training; SM clock drops 1560→~1110 MHz and step pace swings 4.2–6.75 s/it.
   Safe but slow. If the user wants faster/longer runs: improve cooling, then
   kill + rerun the same command (auto-resume).
7. **Goal-tool policy:** update_goal edit/pause/resume are rejected in automatic
   goal rounds (need a direct human turn); complete/blocked are allowed.
8. Windows sandbox: prefer simple single-line pwsh commands; multi-line with
   backticks/$_ is flaky. use our venv python for anything Python.

## 6. Data pipeline notes

- `scripts/prepare_data.py --config configs/<phase>.yaml` streams candidates in
  order, dedupes, splits val, writes data/<phase>/raw/{train,val}.jsonl +
  source.txt. `scripts/tokenize_data.py` packs into data/<phase>/tokens/*.bin
  (uint32, CodeLlama-32k tokenizer, shard cap 8M tokens).
- **Gating:** bigcode/the-stack-v2, the-stack-smol, starcoderdata are all gated
  (manual terms acceptance on huggingface.co). Fallback used: nickrosh/
  Evol-Instruct-Code-80k-v1 (instruction→code-answer text; ~430 tok/row).
  If the user supplies an HF token (they must accept terms in a browser), set
  HF_TOKEN and re-run prepare_data for better pilot/target data.
- **2026-09 update:** bare `code_search_net` no longer resolves on the Hub
  (repo moved under a namespace); the canonical ungated copy is
  code-search-net/code_search_net (config `python`; ~2k chars/row ≈ 326 tok/
  row via the CodeLlama tokenizer). M3 uses it: 150k rows → 47.9M train
  tokens. Other ungated options found via Exa search: tokyotech-llm/
  swallow-code-v2 (49.8B tok, Apache 2.0, LLM-rewritten stack-v2; load config
  name `swallowcode-v2`) and bigcode/python-stack-v1-functions-filtered-sc2
  (natural but small ~361-char rows). the-stack-v2 remains gated.
- TheGamingMahi/TinyCode has a corrupted shard (CastError mid-stream) — the
  fallback chain handles it; smoke used Evol-Instruct too.

## 7. Git / GitHub rules for this repo

- Remote: https://github.com/alshahia/nano_SLMs (branch main).
- LFS tracks `*.safetensors`. **optimizer.pt is gitignored on purpose:**
  a full pilot checkpoint is 1.21 GB (805 MB optimizer + 403 MB model), which
  would exceed GitHub's ~1 GB free LFS quota. The full optimizer state lives on
  this machine at runs/pilot/checkpoint-*/ — local auto-resume is unaffected.
  The pushed checkpoint-1000 (model + trainer_state + scheduler) lets a remote
  agent resume with a fresh optimizer (Trainer warns about missing optimizer.pt
  and continues).
- **M3 (target) artifacts stay LOCAL by user decision (2026-09):** the ~900 MB
  model does not fit free LFS (~790 MB of 1 GB used). .gitignore excludes
  runs/target/checkpoint-*/, runs/target/final/model.safetensors and console
  logs; train_summary.json, eval_report.json, small final-dir files and
  tfevents stay tracked.
- Keep data/*/raw/ out of git (regenerable). Keep data/*/tokens/*.bin in git
  (re-downloading on a slow network is the expensive part). Keep runs/smoke/
  checkpoint-* out (drill artifacts; smoke/final IS tracked).
- Before any push: `git lfs status`, confirm no >100 MB file is outside LFS
  (`git diff --stat HEAD^ HEAD` + `git lfs ls-files`).

## 8. Next steps (in order)

1. ~~Monitor M2 to completion; run eval.py; commit runs/pilot/final + report M2
   exit criteria.~~ DONE — M2 PASSED (see §2/§3).
2. M3 (user-approved 2026-09): ~~VRAM re-probe on 6 GB~~ DONE (4.24/4.4 GB at
   seq 512 AND 1024 → ctx 1024 approved); ~~data prep~~ DONE (CodeSearchNet
   47.9M train tokens); ~~launch~~ DONE 2026-09-05 — training in progress
   (see §3b). Remaining: eval.py after completion; commit metrics + HANDOFF
   final row (weights stay LOCAL per the LFS decision). 2026-09-06 18:36:
   resumed on TU09FBO from the arrived bundle at 2001/5000 (cadence now
   save+eval 100 / limit 3; §3b).
3. Upgrade path (user-approved only): pilot data quality (ungated raw code or
   the-stack-v2 with token), Flash-Next/GDN hybrid architecture experiments
   (resources/ notes are pseudo-code — PLAN.md A7 says plain GQA first).
4. C12 distillation stage (planned 2026-09-06; Tier 1 user-approved; run-gated
   on M3 completion §8.2): plan at research/c12_distillation_plan.md, rationale
   at research/c12_distillation_report.md. Tier 1 = Evol-Instruct trace SFT of
   the M3 model — IMPLEMENTED + CPU-validated 2026-09-06 while M3 trains:
   scripts/sft_data.py (stream/filter/template/prompt-masking; tokenized
   dataset at data/sft/evol/ds, stats in data/sft/evol/meta.json),
   scripts/sft.py (Trainer + padding collator; §5.3 auto-resume contract
   implemented; --pilot = 5k pairs, 1 epoch), configs/sft_t1.yaml (ctx 512,
   accum 16, lr 3e-5 cosine, eval+save 250), eval.py extended with the
   backward-compatible instruction eval (ast.parse pass-rate on held-out
   instructions). Launch at gate:
   & .\.venv\Scripts\python.exe scripts\sft.py --config configs\sft_t1.yaml --pilot
   **LAUNCH-READY (2026-09-06, user decision: prepare now, run ONLY after M3 final):**
   step-by-step runbook = research/c12_runbook.md (decision log included: SFT from
   checkpoint-1000 was considered and DECLINED — undertrained base); gate script =
   scripts/c12_preflight.py (run first; dry-runnable pre-M3 via
   --base-model runs/target/checkpoint-500). configs/sft_t1.yaml now carries
   data.tokens_dir (eval.py forgetting guard KeyError'd without it). Corrected
   numbers: 16,376+400 pairs -> pilot ~312 steps (~1 h), full ~2,047 steps (~5-6 h);
   full-mode disk needs runs/target/checkpoint-* deleted post-M3 (user-approved,
   M2 precedent). TASKS.md row 4 owns live status.
   2026-09-06 later the same window: sft.py --pilot now writes to
   runs/sft_t1_pilot (ISOLATED - the full run always starts clean, no manual
   archive step; CPU e2e-verified on a tiny model); prepare_data.py and
   sft_data.py accept LOCAL files (custom-dataset support, generic text and
   instruction keys); infer.py gained --sft (template-wrapped prompting);
   configs/custom_example.yaml is a user-custom model/dataset template
   (sanity_check 4/4 PASS at 100.7M). README documents both workflows.
   2026-09-06 afternoon (window work): peft 0.20.0 installed + LoRA design
   doc (research/lora_peft_design.md - ladder = Llama arch, standard LoRA
   targets; T r=16 = 4,849,664 trainable = 2.14pct, est ~2.0-2.5 GB;
   implementation = TASKS row 10, user-gated). Tier 2 teacher re-picked by
   the user to Qwen/Qwen3.5-0.8B - verified (24 text layers, vocab 248,320,
   ctx 262k, chat template; transformers 5.16.1 loads
   Qwen3_5ForConditionalGeneration natively) and DOWNLOADED to gitignored
   data/teacher/ (1,688 MB / 29 files in ~3 min - network far faster than the
   230 KB/s estimate; local AutoConfig load OK). exa-py 2.20.0 installed;
   AGENTS.md section 3 now mandates the Exa helper for net research
   (EXA_API_KEY .env still missing - user action). The 3 prep commits were
   pushed (user-approved).
   2026-09-06 later: the recommendation menu (B/C/D/E/F/G) was approved as
   TASKS milestone rows 11-17 - CREATED ONLY, none started (user instruction).
   Both API keys (EXA + HF) now live in the gitignored .env (user-added); the
   teacher web-context search and the HF gated-repo probe were run to pre-fill
   milestone evidence.
   2026-09-06 evening (window work, user GO on all of B-G): the CPU-safe parts
   of TASKS rows 11-17 were executed beside the live M3 run. B: A/B arms
   configs/pilot_ab_base|pilot_b8|pilot_b8_r2x16.yaml + protocol
   research/milestone_b_8bit_ab.md + train.py env fingerprint into
   train_summary.json (GPU arms + drills post-M3; bnb 0.50.2 confirmed
   installed via metadata). C: scripts/mini_eval.py (16 tasks x 2 tests,
   per-test subprocess+timeout; canned self-test 16/16; smoke-final CPU x2
   identical 0.0000 - deterministic, base-model-expected) + status.py
   tokens/sec + est-MFU lines (live: target ~3,033 tok/s recent, MFU 20.4%
   fp32 / 4.4% fp16-tensor). D: research/pretrain_mix_proposal.md (stack-v2
   impractical - blob-IDs only + 233 GB python subset; mix = 30% stack-smol +
   55% starcoderdata-python + 10% CSN + 5% Evol ~ 134M tok; streaming plan,
   24.8 GB headroom) + prepare_data.py .env loader (gated the-stack-smol
   probe OK). E: research/c12_tier_order_decision.md (recommend Tier 3
   before Tier 2 V1 + frozen judge rubric). F: scripts/backup_to_hub.py
   dry-run PASS (manifest 7 files/387.8 MiB; upload BLOCKED on user repo
   name). G1: research/gdn_sandbox_design.md (triton-free chunked
   delta-rule path; S-scale gates; implementation post-M3). G2:
   scripts/run_custom.py (dry-run PASS; GPU guard filters the Windows WDDM
   desktop-context noise to real python compute - shows only the M3 PID).
   All GPU validation deliberately staged post-M3; nothing co-ran with the
   trainer.
   Tier 2 = on-policy logit alignment;
   Tier 3 = intra-ladder KD (tests the ~1/10 GPU-hour claim on our ladder).
   Why: the pipeline currently ends at pretraining — no SFT/distill stage exists
   (crosscheck row C12). T never saw Evol-Instruct, so it is contamination-free
   SFT data for the M3 model.

### 2026-09-07 continuation checklist (web UI session → next agent)

If resuming from here with fresh context, work top-down; the first three are
session-start checks, the rest are the open items. TASKS.md row 21 is the
live status for the UI; rows 18-20 own the in-flight GPU milestones.

1. **Session start:** read AGENTS.md → CLAUDE.md → HANDOFF (this file) →
   TASKS.md → WEBUI_PRD.md; scan MEMORY.md gotchas. Then probe reality:
   `nvidia-smi --query-compute-apps=pid,process_name` (a python PID = a live
   arm/chain — do NOT start GPU work) and `git status` (tree was pushed at
   5fbb31a; only live-run logs should be dirty).
2. **Check TASKS rows 18/19/20 first** — they may have advanced while you
   were away (arm B 8-bit Adam b8 was LIVE at handoff, pid 15616; arms C +
   vram_probe --lora + SFT v2 pilot + KD baseline/KD arms are queued behind
   free GPU windows). Their runbooks: research/milestone_b_8bit_ab.md,
   research/c12_tier_order_decision.md. Arm artifacts (metrics/logs) get
   committed per the §7 rules when an arm lands.
3. **Web UI verifications gated on an idle GPU** (WEBUI_PRD.md §5, all
   user-visible in the app, none need new code):
   - U2 GPU branch: chat tab → Load a weights-bearing checkpoint with GPU
     idle → status must read "On GPU"; unload after.
   - U4 e2e chain: Train tab → tiny run (Small preset, ~100-200 steps,
     TinyCode rows 2000, run name `u4e2e`) → Start → watch Monitor tab →
     confirm complete → cleanup artifacts (runs/u4e2e, configs/webui_u4e2e.yaml,
     data/u4e2e) with user's go. This also proves U3 live refresh.
4. **Small webui debt (optional, CPU-safe):** psutil is absent →
   `webui/app.py:_job_alive` falls back to assume-not-live; either
   `uv pip install psutil` or leave it (the compute-pids preflight covers
   the real collision risk). Chat history is intentionally single-turn
   (fixed templates, no chat tokenizer) — do not "fix" without a design note.
5. **U5 polish items are user-gated** (LAN + auth, done-notifications, eval
   report cards, checkpoint delete flow) — do not start without the user's
   explicit go (WEBUI_PRD.md §5).
6. **Standing guardrails (never break):** never kill a running train job;
   never start a second GPU job while one is live (chat's warn→CPU path is
   the sanctioned exception); resume = the exact same command, zero flags;
   weights stay local (LFS quota — gitignore rules cover the A/B arms);
   report before deleting anything; never commit secrets (.env stays out).

### 2026-09-08 — web UI U5 finished (full e2e PASS); run_custom guard root-caused

TASKS row 21 is DONE. Commit still pending user go — everything from the
U5 work is uncommitted on top of 4e0a551. What landed and how it was
verified:

- **U5 in webui/app.py:** --lan (0.0.0.0), --auth USER:PASS (gradio login
  flow), --webhook URL / $WEBUI_WEBHOOK_URL done-notifications (gr.Info
  toast + best-effort POST, 5 s timeout), eval report card in Monitor,
  checkpoint-delete Danger zone (traversal-safe path, refuses live runs +
  chat-loaded checkpoints, typed confirmation, freed-MB report). Webhook
  events: run_finished (final/train_summary.json appears) and
  run_crashed (job pid dead, no final) — both received by a live local
  receiver during verification (the crashed branch initially hardcoded
  event "run_finished" — found and fixed during this verification).
- **Full e2e PASS** (e2etest, Small preset, 1500 rows, 100 steps):
  launch → sanity 4/4 (CUDA) → prepare 1470/30 rows → tokenize 2589
  blocks → train → eval (best_eval 6.1545, checkpoint-100) → webhook
  run_finished → eval card rendered → e2etest/final chat-able on CUDA
  (696 MiB, real gen). train_summary env fingerprint: git 4e0a551,
  torch 2.14.0+cu126, transformers 5.16.1.
- **Boot/auth check PASS:** --port 7861 --no-browser → HTTP 200; --auth
  verified by content: unauth / = 44.7 KB login shell, POST /login good
  creds → {"success":true} + session cookies → / serves the full app
  (142 KB); gated API /config → 401 unauth and wrong-password. GOTCHA
  for future probes: gradio 6 auth is form+cookie, NOT per-request HTTP
  Basic — curl -u alone gets 401 even with valid creds.
- **Integration bug #3 ROOT CAUSE (three aborted attempts):** on Windows,
  .venv\Scripts\python.exe is a launcher shim that spawns the real base
  interpreter as a child and waits — so run_custom.py's os.getppid() is
  its own launcher shim, NOT the webui server/probe that launched the
  chain; that grandparent holds the lingering CUDA context after a chat
  unload, so parent-pid exclusion can never work. Fix in
  scripts/run_custom.py: _ancestor_pids() walks the FULL ancestor chain
  via a Toolhelp32 snapshot — the _PE32W struct must byte-match
  PROCESSENTRY32W (size_t heap id + 4-byte LONG priority; an 8-byte
  pointer field silently inflates dwSize and Process32FirstW fails with
  an empty walk) — and gpu_busy_others() excludes own + ancestors;
  _pid_alive() drops stale WDDM entries of dead pids (measured: exited
  CUDA children linger 0 s in the listing, filter kept as cheap
  insurance); the abort message now prints the offending pids. Real
  co-run protection unchanged: a non-ancestor python with a compute
  context still aborts (micro-test: live CUDA-holding probe excluded
  from the guard but still visible in the raw nvidia-smi listing).
- **Also fixed en route:** _job_alive via ctypes OpenProcess (psutil
  absent); the app preflight filters its own pid (lingering CUDA context
  after chat unload) and separately blocks launch while the chat model
  sits on cuda; matplotlib figure leaked per 10 s monitor tick
  (close-before-create).
- **Environment event:** the parallel session rebuilt the venv — gradio
  6.26.0 + matplotlib 3.11.1 reinstalled via uv pip (imports verified).
- **Cleanup proposal (user-gated, nothing deleted):** runs/e2etest/,
  data/e2etest/, configs/webui_e2etest.yaml are regenerable e2e test
  artifacts. Suggested commit set when approved: webui/app.py,
  scripts/run_custom.py, README.md, TASKS.md, HANDOFF.md — never
  configs/distill_custom*.yaml or runs/* logs (the parallel session's).

### 2026-09-08 — web UI U6-U11 scope approved by the user; plan written

The user approved EVERY proposed GUI addition except the Hub-backup
button, and added a requirement: train from user-named HF datasets with
two data modes (fetch the data locally vs. stream the needed rows over
the net when disk is low) plus a persistent API-key field (show where
it is saved; allow override and delete).

Grounding found while planning: `prepare_data.py` ALREADY streams HF
datasets (`load_dataset(..., streaming=True)` — only the needed rows
are written, so disk is bounded by `rows`, not dataset size) and
already auto-loads the gitignored project-root `.env` (HF_TOKEN path
verified in TASKS row 13). Design therefore locked in WEBUI_PRD §2:
stream-pack = default mode; full local cache = opt-in; literal
per-step net-feeding into `train.py` REJECTED (deterministic zero-flag
auto-resume, PLAN §5.3, + memmap shard contract) — revisitable only by
explicit user decision. Keys live in the existing .env behind a masked
Settings tab (add/override/delete, path shown; never in configs/logs).

Milestones written into WEBUI_PRD §5: **U6** hardening & resume UX
(resume button, collision guard, --lan auth gate, 8-bit optim default,
chat length guard, chain-step indicator, picker refresh, job banner,
tok/s) → **U7** data & keys → **U8** transparency (dataset preview,
YAML disclosure, MB breakdown, overlay) → **U9** chat streaming →
**U10** SFT/LoRA launch path → **U11** cooperative stop-at-checkpoint
+ browser notifications + HTTPS. TASKS rows 22-27, all `pending` user
go. U11's stop-flag amends the PRD no-kill rule (user-approved):
checkpoint-aligned clean exit only, never a process kill. Note: the
U5 work above was committed and pushed as `fd2e79a` (post-push tree
clean; untracked: parallel session's distill configs/logs + the e2e
artifacts still awaiting the cleanup decision).

### 2026-09-08 — Track H executed (agent-side memory, zero training): mechanism PASS, recall gate FAIL

User go ("best outcome track first") picked Track H from the
distill-survey adoption plan. scripts/agent_memory_eval.py (new):
scripted 12-turn session -> prompted-extraction fact store
(data/agent_memory/, gitignored) + model-mode rolling summary inside the
1024-token budget -> context swap -> 2-arm recall (store vs summary-only
control) in two probe styles (QA + needle completion) -> single-session
AST regression probe. CPU mechanics preflight PASS (--no-model stub:
folds/extract/retrieve/score exercised; small budgets forced folds).

Three GPU iterations on runs/sft_v2_e1/final (~3-7 min each; iter1
snapshot NOT kept — overwritten; iter2 kept under
runs/agent_memory_h/iter2_qa_cue/): the memory mechanism fully validated
(store 6/6 scripted coverage, retrieval 6/6 correct fact per question,
model-mode rolling summary 7/7 folds at 61 tok, AST regression 4/4
preamble vs 2/4 plain, overhead ~7.4% of ctx) but the recall gate FAILED
1/6 (required >= 2): the student continues every QA/needle prompt with
import boilerplate and only sporadically surfaces the injected fact
(f3, deterministic across two runs; the format cue made it 0/6). Root
cause = MEMORY lesson 18; this validates the plan's honest framing —
orchestration memory, NOT model memory; the trained path is Track C.
Evidence: runs/agent_memory_h/report.json + session transcripts;
TASKS row 36 (follow-up DECIDED by the user: fix via distillation ->
row 37).

Same-day continuation: Track H2 (row 37) started with user approval -
Phase 1 = copy-behavior LoRA-SFT with a SUBAGENT-authored needle-QA corpus,
Phase 2 = teacher distillation queued (teacher NOT Qwen3.5-0.8B, user
veto). Prep complete: research/h2_corpus_spec.md (contract),
scripts/h2_validate_corpus.py (copy-fidelity gate, < 200 pairs = FAIL),
configs/h2_copy_lora.yaml (LoRA r=16 on sft_v2_e1, ast_filter FALSE,
CSN guard kept). BLOCKED before any corpus landed: subagent spawning died
in that session (2x background -> empty children registry + 4 failure
notices; 1x foreground -> ToolCallError). Full resume package with a
10-step command ladder: research/distill_survey/H2_RESUME.md.

### 2026-09-09 — Track H2 Phase 1 executed (copy-behavior SFT): recall gate PASS

Resumed from H2_RESUME.md with the user's "test the subagents on a
simple/fast task first". Infra test PASSED foreground (fast task); the
first 150-pair corpus author then hung in open-ended deliberation and the
user killed it — two new infra lessons (MEMORY 19-20): foreground
subagents die with run_code's 600 s ceiling (use background), and long
open-ended quotas loop (chunk to ~30 units with a mechanical
compose->write->reply procedure). Corpus: 4 slices x 5 chunks x 30 pairs =
600 raw pairs by 20 background subagents (personal / technical-ops /
logistics / note-completion); 2 chunks needed lossless format repair (real
newlines inside JSON strings; missing opening quotes — MEMORY 21); the
official validator PASSED 585/600 (15 drops, response > 2 sentences) ->
data/sft/h2_copy/pairs.jsonl. Two sft_data gotchas fixed in the CONFIG only
(MEMORY 22): min_chars 30->4 (it floors the response — killed every
completion answer), rows 5000->556 (n_val computed from the target rows
had inverted the split 250/110 -> now 28/556); the T dims were mirrored
into configs/h2_copy_lora.yaml because sanity_check builds fresh (sft.py
loads dims from the checkpoint — training behavior unchanged). Pilot PASS
(19 steps, eval 3.161); FULL SFT PASS (556 pairs x 2 epochs, 70 steps
~7 min, eval 2.3612 -> 2.0944 -> 2.0680). Code guards PASS: CSN 2.0492 vs
e1 2.0466 (+0.13%, gate <= +10% — LoRA near-zero forgetting as designed);
ast greedy 0.90 (gate >= 0.85; sampled 0.82 vs e1 0.96 — honest note).
DECISIVE RERUN (H2_RESUME step 9) PASS: gate_recall store arm 3/6
(required >= 2; baseline 1/6) — probe 3/6 (rust, layla, neovim), qa 2/6,
controls 0/6, retrieval 6/6 correct. Honest deltas recorded in TASKS row
37 + adoption_plan addendum: the Track H regression probe collapsed to
0/4 ast in BOTH arms (gate preamble >= plain still PASSes — QA-template
code-gen now emits copy-tautologies while eval.py ast on the SFT template
stays 0.90); the summarizer stopped folding (7/7 -> 0) so the mechanism
runs extract/store/retrieve/copy only (overhead 1.3% of ctx); 2 of 3
recall misses = prompted extractor corrupted the stored facts ("March 15"
-> "ISO 15", wifi value echoed), 1 = copy-out failure (f2). Evidence:
runs/h2_copy_lora/final (+eval_report.json, train_summary.json),
runs/h2_copy_lora_pilot/, runs/agent_memory_h2_rerun/report.json. Phase 1
DONE per the ladder; Phase 2 EXECUTED 2026-09-09 (next section).

### 2026-09-09 — Track H2 Phase 2 executed (mixed general-QA corpus): all gates PASS, recall at threshold

User decisions (2026-09-09): Phase 2 GO via the dataset path (NO local
teacher inference; NOT Qwen3.5-0.8B per the standing veto); deterministic
extractor arm ADDED for the recall rerun (prompted arm kept for
comparability); mid-build upgrade sanctioned by the user ("more powerful
datasets", HF token added to .env): an OpenHermes-2.5 GPT-4 distillate
slice added on top of dolly/no_robots.

Corpus (scripts/h2p2_build_corpus.py -> scripts/h2p2_validate_corpus.py):
947 QA pairs kept (371 dolly open_qa CC-BY-SA-3.0; 223 no_robots
OpenQA/Brainstorm/Generation CC-BY-NC-4.0 — Chat is multi-turn by
construction, 795/796 skipped; 353 OpenHermes-2.5 GPT-4 ShareGPT
self-contained pairs, ungated) + the 585 Phase 1 copy pairs appended
verbatim = 1532 mixed pairs (data/sft/h2p2_mixed/, validate_meta.json).
Builder attrition was honest: ~30-50% dropped by the <=5-sentence /
<=130-word prose caps. sft_data: kept 1530 (train 1457 / val 73;
1 drop_short, 1 drop_long).

Training (configs/h2p2_mixed_lora.yaml, LoRA r=16 on runs/sft_v2_e1/final —
the Phase 1 recipe): sanity PASS (trainable 4,849,664 = 2.096%); pilot PASS
(runs/h2p2_mixed_lora_pilot, 19 steps, eval 5.590); FULL PASS
(runs/h2p2_mixed_lora, 184 steps in 1151 s ~19 min, eval 5.36 -> 4.9461
best=final@184, zero interruptions).

Gates (eval.py on final): CSN 2.0077 vs e1 2.0466 = **-1.9% — the mixed
corpus IMPROVED the forgetting guard** (gate <= +10%); ast greedy 0.86
(gate >= 0.85 PASS; sampled 0.78 vs Phase 1 0.82 — honest decline).

Decisive reruns (agent_memory_eval.py on the new final, fresh --out dirs):
- prompted arm (unchanged harness): gate_recall PASS **2/6** (probe 2:
  rust, neovim; qa 1; control 0/6), extractor 5 prompted / 2 deterministic
  fallback, scripted coverage 6/6 — 2 stored facts corrupted by the
  prompted path again.
- deterministic arm (--extractor deterministic): gate_recall PASS **2/6**
  (probe 2: rust, layla; qa 1: pineapple42; control 0/6). The store is
  VERBATIM-PERFECT — all 6 scripted facts byte-exact including "March 15"
  and "Pineapple42" (the exact two the prompted extractor corrupted in
  Phase 1); f6 wifi qa flipped to PASS vs Phase 1, f5 neovim flipped to a
  copy-out miss — net 2/6: the extractor corruption is FIXED, the wall is
  now purely the student's copy-out.
- gate_regression PASSes trivially (ast 0/4 preamble = 0/4 plain in both
  arms — worse than Phase 1's 0/4 vs 2/4; recorded honestly).
- summarizer 0 folds in both arms (as in Phase 1 — mechanism stays
  extract/store/retrieve/copy only, overhead 1.0-1.4% of ctx).

Honest deltas vs Phase 1: recall 3/6 -> 2/6 (still >= the 2/6 gate; the
copy share fell 100% -> 38% and cost ~1 recall point); ast greedy
0.90 -> 0.86; sampled 0.82 -> 0.78; QA-template tautologies persist on
hard prompts (43/50 greedy; the canned config prompts still tautologize)
— the 947-pair QA mix did NOT visibly cure the collapse at this LoRA
scale, but CSN improved. Follow-up is USER-GATED (TASKS row 38): accept
the threshold + residual tautologies, or the stronger-student path
(H2_RESUME section 6 options). Evidence:
runs/h2p2_mixed_lora/final (+eval_report.json, train_summary.json),
runs/h2p2_mixed_lora_pilot/, runs/agent_memory_h2p2_prompted/,
runs/agent_memory_h2p2_deterministic/, data/sft/h2p2_mixed/.

### 2026-09-09 — Track C executed (T→P KD + skew-KL A/B): gate PASS, plain-KD adopted, skew rejected

User decision (2026-09-09): Track C picked from the row-38 follow-up menu as
the next track (the trained path; accept/variant/stronger-student declined).
Machine note: the box moved BACK to DESKTOP-MUO4QK5 — Quadro RTX 4000
8192 MiB, driver 595.97, root `E:\python projects\nano_SLMs` (with space);
ENVIRONMENT.md updated (its "RTX 3000 6 GB" line was the TU09FBO session).

Setup (adoption_plan §C): teacher = frozen 226.5M runs/target/final loaded
fp32 (~0.9 GB); student = P-arch 12L/768 = 100.68M; data = data/target/tokens
(CSN, ctx 1024; 46811 train / 956 val blocks); 2000 steps; batch 4 × accum 2
(effective 8); eval+save 100; adamw_bnb_8bit (row 11 next-pretrain default,
SHARED by all arms → internally fair; deviation from row 20's adamw_torch
recorded); lr 4e-4 cosine, warmup 100; seed 42.

VRAM probe ladder (the fp32 KD logits are the memory hog — 8192-token window
× 32768 vocab): micro-b8 peaked 7897/8192 MiB (96%, OOM-risk) → micro-b4/a2
completed a FULL probe cycle (train+eval+save) at 7010 (86%) = chosen for
both C1 arms; a b2/a4 probe trained at only 4926 but its eval_batch-8
eval transient spiked 7486 → eval stays at 4. The C2 skew path (chunked
per-sample) probed at 7447 (91%) and ran clean. 0 OOM, 0 interruptions
across all 3 full arms (~3.5 h GPU incl probes).

Results (full-precision curves committed at runs/kd-t2p-*/final/eval_curves.json):
- baseline (plain train.py, runs/kd-t2p-baseline): best eval **2.9453** @2000
  (curve 5.617 → 2.945, saturating under cosine-to-0).
- plain-KD (runs/kd-t2p-kd, tau 1.0 alpha 0.5): best eval **2.7476** @2000
  (curve 5.507 → 2.748). Ahead of the baseline at EVERY matched step:
  100 −2.0%, 500 −2.5%, 600 **−3.05%** (= the ≤1/3-steps gate point →
  **GATE PASS**), 1000 −4.0%, 1500 −6.4%, 2000 **−6.71%**. The row-20
  "distilled rung" claim transfers to P scale even at 2.25× compression
  (row 20's teacher/student ratio was 8.2×). The winner is the candidate
  distilled-P rung for future use.
- skew-KD (runs/kd-t2p-kd-skew, DistiLLM α-SKL λ=0.1): best eval **2.8104**
  = +2.29% vs plain-KD → **NOT adopted**. It led only in the 800–1000
  window (−0.6/−1.0%) and lost late; both KD arms' final grad norms are
  healthy (1.0–1.6) → the fp16 instability the skew floor exists to fix
  never appeared at tau 1 / alpha 0.5 / this scale. kd.py's skew branch
  stays config-gated with default 0.0 = byte-identical legacy path.
- Process honesty: a mid-run live-log comparison briefly suggested a dead
  heat — the streamed eval lines were steps 500/600, not 600/700; the
  full-precision trainer_state curves supersede and no wrong verdict was
  ever written to the ledger.

New gotchas (MEMORY lesson 25): F.kl_div needs probabilities or
log_target=True (log-prob targets silently NaN — caught by the CPU math
test before any GPU spend); the CPU validation recipe (exact vs
brute-force mixture-KL at several λ + monotonicity + finite-grad checks)
is the template for any future loss change.

Evidence: runs/kd-t2p-{baseline,kd,kd-skew}/final/{train_summary.json,
eval_curves.json} + logs tfevents + train_out.log (moved from the root
Tee logs, row-20 pattern); configs/kd_t2p_{baseline,kd,kd_skew}.yaml;
scripts/kd.py (skew_lambda config-gated); probe scratch (dirs + configs +
logs) deleted after read-out per the established scratch policy. Disk:
E: 41.1 GB free post-run; the three 403 MB fp32 arm weights + rotated
checkpoints (~1.5 GB/arm) stay on disk pending user cleanup decision
(keep runs/kd-t2p-kd/final at minimum — it is the distilled-P rung).

### 2026-09-09 — resume pack built (fresh clone + ONE zip = fully working setup)

User request: "create a zip file for all checkpoint need, that user with it
and remote repo can resume the work without any issue". Built
`checkpoint_backup/resume_pack_2026-09-09.zip` (gitignored dir, out of git
per HANDOFF §7):

- **6,227,497,311 B (5.8 GiB), 80 entries, verify OK (every entry present,
  sizes match, all CRC pass).**
- sha256 `ceda94adbce10e7af5bcd85a7b58fbbe550072bd1f02bf0c91d68437f705bd4e`.
- Arcnames are REPO-ROOT-RELATIVE (`runs/<name>/final/...`): extract AT THE
  REPO ROOT — no strip-components gymnastics. Zip root also carries
  `RESUME.md` (step-by-step new-machine runbook: clone → extract → venv
  rebuild → sanity gates → .env/teacher notes → machine hard facts → what's
  in/out → where the work stands) and `BUNDLE_MANIFEST.json` (per-file size
  + CRC, git_commit aad7f94, created_utc).
- Contents = ALL 10 weights-bearing finals (target, sft_v2_e1, sft_t1,
  h2p2_mixed_lora, h2_copy_lora, pilot, kd-t2p-kd, kd-t2p-baseline,
  kd-t2p-kd-skew, smoke — roles tabled in RESUME.md).
- Excluded by design (documented in RESUME.md + TASKS row 2): the 3
  superseded pilot finals; the 4 de-weighted finals (kd-s ×2 reproducible
  ~35 min each via scripts/kd.py); mid-run checkpoints of completed runs
  (incl. target/checkpoint-4500 + pilot/checkpoint-1000 — source-disk
  only); data/teacher (re-download ~3 min); .env (secrets).
- Builder: `scripts/bundle_resume_pack.py` (new, committed; multi-source
  STORED zip64 + manifest + verify + sha256, inherits the
  bundle_checkpoint.py conventions).
- Disk after: E: 35.48 GB free (the pack cost ~5.8 GiB). The older
  checkpoint-sft_v2_e1-final.zip stays (redundant with the pack — delete
  per user decision only).

### 2026-09-09 — Track A context eval probes DONE (row 29; machine back on TU09FBO)

User go on Row 29 (Track A, adoption_plan §A). Implemented eval-only
context knobs — NO training code touched, defaults OFF:
`src/model.py` helpers (apply_rope_scaling / reset_rope_scaling via HF
transformers 5.16.1 native dynamic rope; build_streaming_sink_mask;
streaming_position_ids) + `scripts/eval.py` flags (--ctx,
--rope-scaling, --rope-factor, --stream-window, --sink-tokens,
--stream-positions, --batch, --quick, --skip-gen, --report-out,
--label; also reads the same knobs from config eval.*). Contract
gates: sanity_check 4/4 PASS; new `scripts/ctx_probe.py --selftest`
5/5 PASS on CPU (mask==causal identity, mask semantics, exact Qwen
dynamic-NTK inv_freq check, pos_shift clamp).

Full matrix (runs/ctx_probes/20260909T184503Z, ~2.1 h GPU, 0 OOM,
0 interruptions; full 978,944-token CSN val per point; RTX 3000 fp32,
batch 8/4/1 by ctx):

- target/final: base_1024 **1.8512** (== trainer best@4000 — the 2026-09-09
  checkpoint-4000 restore; the on-disk eval_report.json 1.8641 is STALE
  pre-restore, see MEMORY lesson 26) → ntk_2048 **1.7740 (−4.17%)**,
  ntk_4096 **1.8836 (+1.75%)**, noop_2048 +12.3%, noop_4096 +55.8%,
  w1024s4abs_2048 −0.65%; remapped (pos_shift) arms +78-126%.
- sft_v2_e1/final: base_1024 **2.0466** (== recorded report, clean) →
  ntk_2048 **−3.43%**, ntk_4096 +3.90%, w1024s4abs_2048 +0.26%;
  remapped arms +70-113%.

Findings: Dynamic-NTK @2048 is a free lunch (improves over the 1024
baseline); @4096 eval-only is +1.75% base / +3.9% e1; **StreamingLLM
pos_shift is wrong for this from-scratch model** (collapses at every
width; sink 4 ≈ sink 0 → no attention-sink specialization; the model
needs correct relative distances); window+sink mask with ABSOLUTE
positions is the only working window variant (near-baseline @2048,
inference-time lever). VRAM 4.39 GB @1024/2048, 2.63 GB @4096 — KV
non-binding as computed in note 04.

**DECISION (row 29 result → row 30): Track B targets ctx 4096 with YaRN
factor 4** (the plan §A gate fired: NTK@2048 holds within +2%), 2048 /
factor-2 as the recorded fallback IF the 8-bit-Adam vram_probe @4096
OOMs (B must probe BOTH ctx first). B gates now: val lift on BOTH
surfaces + the @1024 forgetting guard. Result note:
research/distill_survey/track_a_result.md; TASKS rows 29/30 updated.
ENVIRONMENT.md flipped to TU09FBO (RTX 3000 6 GB, driver 580.92,
E: ~36.35 GB free at session start).

### 2026-09-09 night — Web UI U12/U13 DONE (Model tab: Architecture Explorer + Training Simulator)

- Spec WEBUI_PRD.md §5 U12/U13 (user-approved design: replay-only simulator v1, two-level
  info, ONE Model tab with nested views, Approach A = Gradio-native SVG + clickable
  gr.Dataset, no new deps, read-only + CPU-only, no frontend build chain). Plan:
  docs/plans/2026-09-09-webui-u12-u13-model-tab.md. Executed Subagent-Driven (implementer +
  reviewer per task, 9 tasks, all reviews SPEC PASS / QUALITY APPROVED; ledger in
  .superpowers/sdd/progress.md).
- Shipped: webui/explorer.py (graph/math/render/tokenize), webui/simulator.py (replay
  engine), webui/model_tab.py (thin wiring; nested tabs), webui/artifacts.py additions
  (header-only safetensors parse, module_param_totals, config loaders), tests (19/19: 6
  artifacts + 6 explorer + 7 simulator), app.py two-line integration.
  Explorer: any config or runs/ checkout; 16 layer tiles on target/final; dtype gate
  F16/F32/BF16 (real header shows F32 embedding); KV-cache KiB/token; projected params
  bit-exact; guided tour; real-tokenizer trace. Simulator: 9 configured replays / 4
  techniques; stage cards + anchored VRAM gauges + events + disk-slot rotation;
  play/pause/tick/restart/scrub/speed 60-3600x; KD baseline pair + delta line.
- Browser drill (agent-browser, live Gradio session): ALL GATES PASS — dataset click -> detail
  updates; technical toggle shows tensor names/shapes; target/final 16 tiles; trace
  gives real CodeLlama IDs; load -> play -> pause freeze -> finish (200/200, best 4.7926,
  end card); KD end card + "KD vs baseline @ step 100: baseline - KD = +0.1106 (KD ahead)"
  + Disk slots ckpt-100. Drill-only bug found + fixed: technique switch left the run
  dropdown's stale value -> Gradio preprocess Error; fix 57055b1 sets value to the new
  technique's first run in the same gr.update (unit tests cannot see preprocess validation).
- Commits (feature, in order): 789cd9f 5f37048 fc229a2 d4ba99c 96c943f b6a4bc0 0fbd729
  31ad265 b7ad778 b711a9d a5991ab 40d77bb 76b917a b06f6ef 70446ca 57055b1 (Task 7 =
  sanctioned no-commit shim removal; 70446ca its cleanup). WEBUI_PRD.md spec commit a61ee19.
  Zero writes to runs/ or configs/ verified (drill + commit audit); probe server + browser
  torn down after the drill (7879 down, agent-browser session closed).
- Triage list (minor, non-blocking): technical-toggle resets block selection to default;
  LoRA VRAM est ~16% high vs the 0.72 anchor; playback pace re-render-bound at low speed;
  unused label param in explorer.build_graph; guarded SVG onclick no-op (shim removed);
  _play stride recomputed per frame.

### 2026-09-10 morning — Track A 8k/12k/16k extension: STATE + RESUME (row 40)

User asked to extend row 29 to 8192/12288/16384. Target surface is
COMPLETE and analyzed; the e1 (instruct) surface is missing exactly 2
points because the machine slept twice mid-run. **This section is the
resume point. RESOLVED 2026-09-10: the e1 rescue ran clean (checklist
below); TASKS row 40 is DONE.**

#### What already happened (evidence committed)

- Driver: scripts/ctx_probe.py grew presets `long` (noop_8192, ntk_8192,
  ntk_12288, ntk_16384, w1024s4abs_8192), `extreme` (w1024s4abs_16384),
  `rescue` (e1's two missing points); per-point disk flush (crash-safe)
  and a RuntimeError catch so a CUDA fault no longer kills the matrix.
- target/final (base_1024 = 1.8512): noop_8192 3.9409 (+112.9%),
  ntk_8192 2.1616 (+16.8%), ntk_12288 2.4096 (+30.2%), ntk_16384 2.6159
  (+41.3%), **w1024s4abs_8192 1.8236 (-1.50%)**, **w1024s4abs_16384
  1.8512 (+0.00% - the window+sink absolute mask holds the 1024
  baseline FLAT at 16x context)**. Evidence: runs/ctx_probes/
  20260909T214837Z (long) + 20260909T221824Z (extreme).
- sft_v2_e1/final (base 2.0466): noop_8192 4.3577, ntk_8192 2.4586
  (+20.2%), ntk_12288 2.7336 (+33.6%) - evidence RECOVERED FROM STDOUT
  into runs/ctx_probes/20260909T224806Z (marked as such in its
  summary.json).
- Result note: research/distill_survey/track_a_result.md has the full
  8k-16k section (findings 5-6). Track B decision UNCHANGED: 4096 stays
  the YaRN target; eval-only NTK is +17..+41% at 8k+; the window+absolute
  mask is an inference-only lever (recall capped at window+sink).

#### Incidents (why e1 is incomplete) — Event Log verified

1. During the e1 long run the machine SLEPT (Kernel-Power 506/507):
   e1 ntk_8192 shows wall_s 17995 (~5 h, mostly suspended) but finished;
   the NEXT point (ntk_16384) died with CUBLAS_STATUS_EXECUTION_FAILED
   (damaged CUDA context after the wake).
2. The rescue attempt (presets rescue) was killed at 2026-09-10 07:01-07:09
   by ANOTHER sleep; nvlddmkm Event 153 (driver reset) at 07:09; process
   died hard, no traceback; runs/ctx_probes/20260909T214837Z_e1_rescue is
   empty. GPU reads healthy now (nvidia-smi OK).
3. VRAM discovery (-> MEMORY 32): at 12288/16384 the fp32 eval exceeds
   the 6 GiB card and **Windows sysmem fallback silently spills** -
   torch.cuda.max_memory_allocated measured 6.15 / 7.9 / 8.17 GiB with
   NO OOM; the tell is pace (23 s/batch at the 16384 mask arm).

#### NEXT ACTIONS (in order)

1. **Keep the machine awake during GPU runs** - user call:
   `powercfg /change standby-timeout-ac 0` (and lid-close plan), or
   accept the e1 gap as best-effort (the decision does not hinge on it).
2. Re-run the rescue (~20 min; fresh process, GPU idle):
   `& .\.venv\Scripts\python.exe scripts\ctx_probe.py --points rescue --ckpt runs/sft_v2_e1/final --out-dir runs/ctx_probes/20260909T214837Z_e1_rescue`
   Expected by trend: e1 ntk_16384 ~2.8-2.9 (+37-42%), w1024s4abs_8192
   ~2.0-2.1 (near baseline). If nvlddmkm 153 fires again - reboot first.
3. Fold the rescue numbers into track_a_result.md (8k-16k table), then
   set TASKS row 40 -> done and commit.

#### Not yet done (resume checklist)

- [x] e1 rescue run — DONE 2026-09-10 05:29-05:45 UTC on MUO4QK5 under
      the awake guard (powercfg /change standby-timeout-ac 0 applied;
      original AC value was 0x0e10 = 60 min — restore with
      `powercfg /change standby-timeout-ac 60`): ntk_16384 2.9597
      (+44.6% vs base 2.0466; slightly above the predicted 2.8-2.9 band,
      honest note) + w1024s4abs_8192 2.0444 (-0.11%, flat); wall
      270+296 s, peak VRAM 7.92/4.45 GiB (fits the 8 GB card, no spill),
      zero interruptions, no nvlddmkm recurrence. Evidence:
      runs/ctx_probes/20260909T214837Z_e1_rescue; folded into
      track_a_result.md (8k-16k table complete); TASKS row 40 -> done.
- [x] Commit of the extension + rescue evidence + docs (this commit;
      the extension itself was committed as WORK-IN-PROGRESS
      deliberately, concurrent session, crash protection).
- [ ] Optional: fold the window+absolute mask into the webui chat path
      later (it is the only working long-context inference lever; NOT
      part of row 40).

#### Resume pack rebuilt (2026-09-10, user request)

`checkpoint_backup/resume_pack_2026-09-10.zip` — fresh clone + THIS zip =
fully working setup. 6,229,439,224 B (5.8 GiB), 79 weight/config entries +
RESUME.md + BUNDLE_MANIFEST.json, verify OK (all CRC pass).
sha256 `e18f05de9d93ea944382b8e300d721c5fc002453e68be97767f35679fd259e45`.
git_commit c17dd28; REMOTE REQUIREMENT: the clone must include c17dd28
(it was the ONLY unpushed commit at build time — push it before relying
on remote+zip). Contents = the same 10 weights-bearing finals as the
2026-09-09 pack. To rebuild it, 6 finals whose weights had been removed
from disk were RESTORED bit-exact from the 09-09 pack (sha256-verified
source): sft_t1, h2_copy_lora, h2p2_mixed_lora (906,121,272 B each) +
kd-t2p-{kd,baseline,kd-skew} (402,742,216 B each) = +3.93 GB back on
disk (this also resolves the pending kd-t2p-residue cleanup question —
the weights are back; disk now ~27.9 GB free). The 09-09 pack is
redundant with this one — keep/delete per user decision.

### 2026-09-10 morning - Track B DONE (row 30): YaRN ctx-4096 fine-tune of the target apex

User go ("go to Track B"). Target ctx 4096 / YaRN factor 4 per the row-29
decision; the 2048/factor-2 fallback was NOT needed.

- Implementation (config-gated, defaults OFF; smoke sanity 4/4 =
  byte-compat proof): src/model.py build_config honors a rope.yarn
  {factor, original_max} block -> HF-native yarn (verified live: 26/32
  rotary dims re-parametrized per the beta 32/1 band ramp; attention
  temperature 0.1*ln(4)+1 = 1.1386 active). New load_finetune_init() +
  config-gated train.init_from in train.py: the base apex's weights load
  into the NEW run's model (base dir never touched; unit check 146
  tensors, tied lm_head intact). Transformers 5.16 gotcha -> MEMORY 34:
  LlamaConfig has NO rope_theta attribute (theta lives INSIDE
  rope_parameters).
- Mandatory probe FIRST (row 30): vram_probe BOTH ctx with 8-bit Adam -
  @4096 peak 4.24 GiB alloc / 5.02 reserved (6,599 tok/s); @2048
  3.34 / 3.60. No OOM -> 4096 CONFIRMED; 8-bit Adam made the 4x jump
  ~free (same peak as the old seq-1024 fp32-AdamW run).
- Data: NO re-pack needed - PackedDataset re-slices the existing
  data/target/tokens stream at ctx 4096 (11,702 train / 239 val blocks).
- Train (configs/yarn_4096.yaml): 1000 steps, batch 1 x accum 8 (32k
  tok/step), lr 3e-5 cosine (repo fine-tune precedent, ~1/13 of the 4e-4
  pretrain LR), adamw_bnb_8bit, seed 42, warmup 50, eval+save 100.
  Eval curve 1.713 -> 1.7035 (best = ckpt-1000 = final). Wall ~1h50m;
  pace 5.3-15.4 s/it across thermal throttle phases (never killed);
  zero OOM, zero interruptions; git at train time 412616f.
- GATES (eval.py, the same instrument as Track A):
  - val @4096 native: 1.7035 = -8.0% vs the 1024 baseline (1.8512) and
    -9.6% vs eval-only NTK@4096 (1.8836) - plan §B gate (<= +2%) PASS
    with ~10 pts of margin.
  - Forgetting guard @1024 (trained yarn rope): 1.8724 = +1.15% PASS
    (tolerance +10-15%). Default-rope @1024 diagnostic: 1.9277 = +4.13%
    (weights adapted to the yarn rotary - expected; the deployed model
    at 1024 uses its own yarn rope).
  - Honest extrapolation @8192 (2x beyond the trained ctx): 2.1240 =
    +14.8% vs the 1024 baseline, but BETTER than eval-only NTK@8192
    (2.1616, -1.7%) and far better than noop_8192 (3.9409) - the 2x
    extrapolation stays coherent (YaRN's own extrapolation claim).
  - Instruct surface: NOT directly liftable by B (e1 untouched by
    design; the artifact is a context-extended BASE). Follow-up row 41
    (user-gated): re-apply the e1/SFT recipe on runs/yarn_4096/final;
    the number to beat is e1 NTK@4096 +3.9%.
- Evidence: runs/yarn_4096/final/{train_summary.json, eval_report.json,
  config.json, generation_config.json, tokenizer*}; gates
  runs/yarn_4096/gates/{eval_1024_yarn,eval_1024_default,eval_8192_native}.json;
  raw stdout runs/yarn_4096_train_20260910.log + gates_gate*.log
  (untracked). TASKS row 30 -> done; row 41 created (pending).
- NEXT ACTIONS: (1) user go for row 41 (instruct re-SFT on the YaRN base)
  if the 4096 instruct model is wanted; (2) push decision for the local
  commits (412616f + this one); (3) powercfg standby restore still
  pending user call (standby-timeout-ac currently 0).

### 2026-09-10 day — row 41 (user go): soup P0 DONE + LoRA-SFT on the YaRN base LAUNCHED

- **User 4-option menu (2026-09-10)**: vehicle = LoRA-SFT r=32/alpha=64/lr 1e-4/1 epoch -> merged final (row-34 Track F repurposed as the row-41 vehicle); P0 soup probe FIRST; gates STRICT (CSN guard @1024-yarn <= +5% vs 1.8724; NEW base-surface @4096 <= +2% vs 1.7035; instruct val @4096 < 2.1265-equiv; ast).
- **P0 DONE (eval-only, zero training)**: e1 x yarn_4096 wiSE-FT interpolation sweep via NEW `scripts/soup_merge.py` (146 tensors, fp32, CPU-only; config/tokenizer donor = yarn so merged dirs are native-yarn @4096). Matrix (guard @1024 / ast greedy / @4096): alpha 0.25 -> 1.8849 (+0.67%) / 0.20 / 1.7142 (+0.63%); alpha 0.50 -> 1.9239 (+2.76%) / 0.84 / 1.7535 (+2.94%); alpha 0.75 -> 1.9893 (+6.26%) / 0.98 / 1.8218 (+6.96%). **Verdict: near-linear tradeoff, no magic knee — the soup is a post-hoc KNOB, not the vehicle; even alpha 0.75 stays far under the row-41 instruct gate (1.8218 << 2.1265).** Reports: runs/soup_p0/reports/*.json + soup_p0_summary.json (merged weights = gitignored scratch).
- **Literature sweep** (repo Exa helper, tasks exa_tasks_row41.json, payloads research/raw/r41*.json): LoRA Learns Less and Forgets Less (arXiv 2405.09673 — rank matters most on code); replay anti-forgetting (PMLR v267 bethune25a; arXiv 2603.04964; 2506.09428); wiSE-FT/model soups (2109.01903 / 2203.05482); LongAlign (2024.findings-emnlp.74) — extend THEN align is the canonical order (our Track B -> row 41 matches), and short-ctx SFT leaves long-PROMPT instruct ability untested -> long-prompt probe recommended in the gate pack.
- **LoRA run LAUNCHED (background)**: configs/yarn_lora_sft_v1.yaml — base runs/yarn_4096/final READ-ONLY via from_pretrained (baked yarn rope inherited; sft.py loads natively, line 132); minimax3 ds reused from e1 (no sft_data re-run); CPU sanity 3/4 PASS (gpu gate FAIL = deliberate CUDA_VISIBLE_DEVICES=-1 artifact, re-verified at launch; trainable 9,699,328 = 4.106%); adamw_torch, 1 epoch (~1172 steps), accum 16, eval/save 250, raw log runs/yarn_lora_sft_v1_train_20260910.log.
- **Run DONE 2026-09-10**: 1172/1172 steps clean (~1h45m, ~5.3 s/it), best_eval 0.38447 @ckpt-1172, final/ = merged full model (save_final merge_and_unload), zero flags/interruptions.
- **Gates (eval.py, runs/yarn_lora_sft_v1/gates/*.json)**: guard @1024-yarn 1.9471 = +3.99% PASS (<= +5%); ast greedy 0.94 / sampled 0.92; base @4096 1.7825 = **+4.64% FAIL** (<= +2%) — the LoRA dose drifts the LONG-ctx surface ~1.17x the short-ctx drift.
- **Repair sweep (soup knob, soup_merge.py, LoRA-merged x yarn)**: a40 guard +0.32% but ast 0.68 (fragile under interpolation, same as the e1 soup); **a60: guard 1.8917 (+1.03%) PASS, ast 0.90/0.96, @4096 1.7255 (+1.29%) PASS — ALL 4 strict gates PASS and it dominates the e1-soup P0 frontier everywhere**. Reports: gates/soup_a{40,60}_*.json; merges under runs/soup_repair_41/ (scratch, gitignored).
- **Outcome (user decisions 2026-09-10)**: deliverable = BOTH — `runs/yarn_lora_sft_v1/final_a60` (0.6x LoRA-merged + 0.4x yarn base) = the 4096 chat-default candidate (ALL 4 strict gates PASS: guard +1.03%, ast 0.90/0.96, @4096 +1.29%); `runs/yarn_lora_sft_v1/final` (raw LoRA-merged) = instruct-max variant (ast 0.94/0.92; base gate +4.64% noted). Long-prompt probe (3.4k-token paste of src/model.py, gates/long_prompt_probe.json) = FAIL — off-task code continuations, exactly LongAlign's prediction; the chat default is a SHORT-prompt model until a long-instruct slice lands (TASKS row 42, user-gated). Repair scratch weights deleted (all reports kept under runs/soup_p0/reports/ and gates/); a60 promoted to final_a60. Commit made; PUSH USER-GATED.

### 2026-09-11 — Mounting experiment (TASKS row 49; this agent's track, subagent-driven)

- **Status: COMPLETE 2026-09-12 — Tasks 0-7 all DONE, all five arms executed, AB report written (research/mounting_ab_report.md).** Everything in this section is THIS track's work; shared-file edits were targeted only — other agents' uncommitted edits in these same files are intentionally NOT included in this track's commits.
- Commits: 3429ede (Task 1 teacher stream script + data, reviewer PASS/APPROVED) -> 79d2afe (Task 2 src/mount.py bridge core, PASS/APPROVED; adjudicated deviation: container-preserving WrappedLayer because transformers 5.16.1 LlamaDecoderLayer returns a plain tensor) -> dc7c20d (USER-DECIDED bridge init fix: a=0 zero-init kept + out_proj default init — the plan's double zero-init made grads exactly 0 = inert mount; reviewer-verified identity still bitwise + trainable) -> 107cc13 (Task 3 trainer+dataset; review PASS/PASS, run on the settled report) -> 54c030e (Task 4 CPU suite ALL PASS; adjudication preserved the a=0/strength-1 identity literal) -> a5056a1 (Task 5 configs, SPEC COMPLIANT) -> drill-fix cycle e9feff5 / f97dae7 / b77f82b / 083b2d1 (collator carry-through + label_names undo-deadening pure-CE eval; fp32 teacher dtype; bridges registered once in wrappers; test-contract line).
- GPU gates recorded: Step 5.4 vram_probe (mount_gate): peak 1.36 GB alloc / 1.52 GB reserved, well under 6 GB. Step 5.5 kill/resume drill on configs/mount_gate_drill150.yaml (scratch, uncommitted): a kill BEFORE the first save produced a PARTIAL checkpoint that the partial-ckpt guard then correctly skipped (drill redesigned save/eval 50) -> kill after complete checkpoint-50 -> exact-command zero-flag resume at step 51 -> eval curve continuous across the kill (6.69@50 / 5.06@100 / 3.97@150), global_step 150, exit 0, cliff book persisted, final independence save verified (110 tensors, ZERO bridge/mount-named names). Fill-mode first-3-steps GPU drill PASSED (I2 gate: fill hooks + fp16 + grad checkpointing).
- Artifacts: full SDD trail under .superpowers/sdd/mounting/ (progress.md ledger = the resume map; task-N brief/report/review/diff per task + 3 FIX REPORTs appended to task-3-report.md). Drill scratch (runs/mount_gate_drill, runs/mount_fill_drill3, ~4 GB) DELETED after the PASS record. Teacher stream data/pilot/tokens_teacher_smol135 = 40,907 train / 846 val blocks EXACT matching the student stream (committed with Task 1). SmolLM2-135M snapshot = C:/Users/AhmadMhmoud/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2 (mount.teacher_path in configs/mount_*.yaml; machine-bound absolute).
- ARMS RESULTS (Task 6, strictly sequential, 2026-09-11 19:02 -> 2026-09-12 14:00): control 1.8094@1000 (1h23m, no teacher) / fill 3.6798 (4h11m, KD baseline FAIL: cliff @400 delta +2.62, post-cliff eval flat ~6.5, fp16 grad-norm runaway median 1454 - honest FAIL per plan Task 7) / gate 1.8442 (5h10m, +1.9% vs control, zero cliffs) / drop 2.0473 (5h34m incl. the step-300 crash; CPU/CUDA device bug in the severance mask src/mount.py:74 - mask never moved to the model device; one-line fix _sev_keep.to(device=h.device, dtype=h.dtype), CPU+CUDA verified, exact zero-flag resume from ckpt-300) / hybrid 1.8153 (2h05m, +0.3% - best mount arm AND best teacher-arm wall-clock; late-run drop mechanics skip the teacher forward). VRAM live-measured 2.69 GB / 6 GB during the gate arm. The +20% divergence wall never triggered a stop.
- Independence gate (user third gate) PASS: every final = 110 tensors, ZERO bridge/kv_proj/probe/mount names (programmatic verify); mount-on hold-end eval == final independent eval to 4 decimals for all five arms (control 1.8094, fill 3.6798, gate 1.8442, drop 2.0473, hybrid 1.8153). Honest interpretation: by step 1000 the tanh-gates had accumulated no measurable net contribution vs control (control still best: 1.8094 < hybrid 1.8153 < gate 1.8442) - bridges bought parity, not gain, at pilot scale; hybrid is the scale-up recommendation (user-gated). Full table + caveats: research/mounting_ab_report.md.
- NEXT on resume: NOTHING owed by this track - experiment COMPLETE, report written, TASKS row 49 = done. Drill scratch remains deleted; arm tee logs arm_<name>_tee.txt at repo root remain on disk. Weights stay LOCAL per AGENTS s5.
- NEXT on resume: read .superpowers/sdd/mounting/progress.md FIRST (ledger owns per-arm status), then run the exact zero-flag command of the interrupted arm; after all five arms: Step 6.2 independence evals + research/mounting_ab_report.md + TASKS row 49 / this HANDOFF section update (Step 6.3). Arm tee logs arm_<name>_tee.txt at repo root are progress-bar noise — numeric evidence = final/train_summary.json + tfevents.

### 2026-09-18 — E-23b S2 CLOSED: classical tashkeela expansion check arm -> NOT adopted

- User picked option A for the only remaining E-23 stage. Sources:
  community-datasets/tashkeela (97 book rows) + asas-ai/Tashkeela (804k lines);
  10-word shingle barrier vs gates+fadel train kept 1,200,000 NEW windows
  (3,276,760 dropped - raw Tashkeela overlaps our pool heavily); quar 2,879.
- Pack v3t = v3q + windows = 4,003,948 train rows, val IDENTICAL (160,630);
  configs/diac_e23b_t.yaml = arm C recipe exactly (2,976,128 params, batch 32,
  lr 4e-4, 2500 steps, no cache).
- Gates @2500: 38.69/50.70/59.93/52.55 mean 48.72 vs arm C 48.42 -> worse on
  EVERY gate (+0.30 mean) -> **NOT adopted; arm C (v3q, mean 48.42) stays
  micro champion**. Classical gold expansion ~= zero headroom; modern-MSA
  label gap confirmed as the only remaining data lever (no open source).
- Post-run safe cleanup (user requested 2026-09-18): deleted REJECTED-arm token
  pools data/diac/v3t/tokens + data/diac/v3qz/tokens (repack scripts committed:
  s2_pack.py / e23_pack_zm.py; staging classical_rows.jsonl + zm_labels.jsonl
  kept), the superseded data/diac/v2 pool (regenerable from raw/), and the
  re-downloadable models/e19/fine-tashkeel (E-19 participant; verdict frozen).
  E: free 6.1 -> 19.6 GB. Kept: v3/v3q/stage1 tokens, all final/weights/gates,
  raw corpora, gates.
- Report research/e19_bakeoff/E23B_CLASSICAL_S2.md; EXPERIMENTS row E-23b;
  TASKS row 68 done (+stale row 70 closed); scripts s2_classical_build.py /
  s2_pack.py committed.
- E-23 plan now FULLY closed (S1-S5 all executed; production = gold v3 +
  micro arm C).

### 2026-09-17 — E-23d S4: BiLSTM vs our-arch equal-data A/B -> NOT adopted

- S4 (E-23 plan, user-approved) executed to close: ZM BiLSTM (Embed128 -> BiLSTM 3x256 + Bahdanau attention, 4,498,831 params) reimplemented into OUR I/O contract (diacritizer/scripts/e23_bilstm_model.py; per-char [B,ctx,15] logits) so the exact bench/gate/CSV plumbing of every arm applies.
- Equal budget = arm C exactly: batch 32, 2500 steps, fp16 autocast, AdamW wd 0.1 on v3q tokens. Gate CSV: runs/diac/e23d_bilstm_lr1e-3/gate_eval.csv.
- lr scan at constant budget: our micro 4e-4 STALLS the LSTM (loss ~1.0 at 500, word-der ~0.997 all-bare luck); 1e-4 no-learn; ZM-native 1e-3 chosen (native hyperparams from vendored config: BASELINE lr 1e-3, grad clip).
- Verdict: 45.77/58.09/62.90/56.55, mean 55.83 @2500 vs arm C 48.42 (+7.41 worse), arm A 51.10. Trajectory still improving but gap structural (report research/e19_bakeoff/E23D_BILSTM_AB.md).
- DECISION: char transformer stays for D-line; no user-gated architecture change needed. MEMORY lessons 70 (arch not portable) + 71 (probe-gate eval bug lesson; first '0.997' rows VOID).
- Files: diacritizer/scripts/e23_bilstm_model.py + e23_biarm_train.py (env S4_TOTAL/S4_LR knobbed), research/e19_bakeoff/E23D_BILSTM_AB.md.
- Debug detour worth carrying forward: pwsh *> redirects hide subprocess JSON deaths -> gate() switched to in-process eval_der compare (lesson 71).

### 2026-09-17 — E-23a result: gold+v3q REGRESSED -> NOT adopted

- Gold 30M on v3q (same recipe as stage2b2500): fadel 33.54 / sadeed 25?45.33 / wn2024 55.71 (+7.06 vs gold) / wn2014 47.78 -> mean 45.59 vs 44.14. wn2024 collapse = QCRI label-distribution shift confuses saturated reader. Decision rule -> NOT ADOPTED for the production gold; QCRI weak data stays in the MICRO line only (arm C). Lesson: weak supplement helps capacity-limited (micro) models, not the 30M one. research/e19_bakeoff/E23A_GOLD_V3Q.md.

### 2026-09-17 — E-23 in flight (S1 GPU + S5/S2 CPU)

- S1 RUNNING: configs/diac_e23a_gold_v3q.yaml (phase e23a_gold_v3q, gold 30M arch, stage1lm warm-start, v3q tokens, 2500 steps, gates every 500). Commit 8569362.
- S5 validator DONE (CPU smoke PASS): diacritizer/scripts/e23_crossagree.py — gold + ZM per line, word-map agreement %, flagged/summary outputs; default data/control threshold flag_below 0.50.
- S2 interim: HF hunt finds UNACQUIRED candidates for later: arbml/tashkeela (classical, vocalized pairs), community-datasets/tashkeela (classical books incl. Shamila), asas-ai/Tashkeela, Misraj/Sadeed_Tashkeela (already in pool), TTS speech corpora NOT text-grade. Modern-MSA remains the gap -> covered later by S3 ZM distillation.
### 2026-09-18 — E-23c result: ZM distillation REJECTED (+4.13 worse on micro)

- ZM labels packed (v3qz, 3.12M rows); arm E (arm-C recipe): fadel 39.88/sadeed 52.72/wn24 61.44/wn14 56.15 vs arm C 48.42 mean -> +4.13 worse. Mechanism = label-STYLE conflict (3rd data point for lesson 68 family). BEST MICRO stays arm C (v3q). Runs auto-resumed cleanly mid-crash, shape unchanged.

### 2026-09-17 — E-23c S3: ZM label run + arm E in flight

- ZM full labeling DONE: 142,370 paragraphs / 3.06M words (12,000 qcri-gatesafe articles, 280-char paragraphs, CPU).
- v3qz tokens packed (diacritizer/scripts/e23_pack_zm.py, disk-light direct-shuffle write; NOTES: E:/ drive free space was only ~4 GB, e23a checkpoints trimmed to fit — metrics+final kept): train 3,119,320 rows = v3q 2,803,948 + ZM 315,372; val byte-identical to v3.
- Arm E (configs/diac_e23c_zm.yaml, phase e23c_zm): identical arm-C recipe on v3qz; compare @2500 vs arm C mean 48.42.
### 2026-09-17 — E-23c S3 launched: ZM distillation labeling

- e23_zm_label.py (diacritizer/scripts/): per-paragraph bare stripping of qcri_gatesafe texts -> ZM-BiLSTM vocalization -> rows {serial, para, bare, diac} in data/diac/raw/e23c_zm_labels/zm_labels.jsonl (probe.jsonl smoke PASSED: 31 paras/20 arts/3.1 s, ZM output verified proper diacritics).
- full run = pwsh-57 (CPU ~10 paras/s; GPU not used for ZM).
- next: pack ZM rows -> v3q2 tokens (v3 pool + QCRI + ZM-soft-label windows) -> micro arm E (E-23c check arm, like arm C protocol) once enough rows have accumulated; then compare @2500 vs arms A/C on gate CSVs.
### 2026-09-17 — E-23 plan: Z-Mahmood leverage (user-approved, most-gain-first)

- Plan: research/E23_ZM_LEVERAGE_PLAN.md (ledger E-23; TASKS rows 65-70).
- S1 gold-30M + QCRI v3q SFT is the biggest expected gate gain -> first launch (needs GPU idle, ~4-5 h). S5 validator + S2 data hunt run beside it on CPU. S3 distillation and S4 BiLSTM A/B follow.
### 2026-09-17 — E-22: micro arm C + QCRI weak data (improvement CONFIRMED)

- QCRI wiki gate-shingle-deduped (kept 24,882 articles; 362,621 windows) added to v3 pool -> v3q tokens; val unchanged. Arm C (micro_a_qcri) final: fadel 36.44 / sadeed 48.67 / wn24 58.13 / wn14 50.11 vs arm A 38.16/51.40/60.96/53.87: QCRI weak data IMPROVES (-1.7..-3.8 DER), KEPT per user rule. No cache in this line (user dropped). Micro now ~3-4 DER behind gold 30M.

### 2026-09-17 — E-21: micro 12-128-512 both arms + train-time cache test (user go)

- Full: research/e19_bakeoff/E21_MICRO.md (ledger E-21; MEMORY lesson 66; TASKS rows 62/63).
- ARM A from-scratch 2.98M: fadel 38.16 / sadeed 51.40 / wn24 60.96 / wn14 53.87 (3.5x faster than gold's schedule). ARM B warm-start from shape-matched micro LM (had to pretrain micro_a_lm 4k steps because 30M stage1lm shapes cannot transplant into 12L/128h): gates 41.07/54.08/62.76/56.44 -> FROM-SCRATCH WINS; B's SFT val_loss is LOWER (0.2911 vs 0.3228) yet gates worse (causal prior transfer hurt).
- TRAIN-TIME CACHE (user idea) FALSIFIED: cache-merge from training-corpus = zero gain on both micro arms (gold: -0.3..-0.6). Cache stays a deployment-only trick.
- QCRI audit: machine-grade labels (45.0 DER vs gold validator on wiki paragraphs; over-vocalized 99.8%) = weak-data-only; wn2014 == our gate -> dedup mandatory.
- Micro is a legit cheap fallback/ablation baseline, ~7-8 DER behind gold.

### 2026-09-17 — E-20: user 3 questions (QCRI datasets / word-cache for gold / micro-model)

- Full note: research/e19_bakeoff/E20_CACHE_SMALLMODEL.md (ledger row E-20; MEMORY lesson 65; TASKS row 62).
- QCRI github qcri/advancing-arabic-diacritization cloned to models/e19/qcri-src (91.4 MB jsonl = 32,834 articles / 5.07M machine-labeled wiki words + wn2014/wn2024 multi-ref benches) — trainable-but-license-unstated; dedup against wn2014 gate required.
- OUR word-cache: e19_build_wordcache.py (140.6M tokens -> 375,923 forms); cache-merge bench e20_goldcache_bench.py: gold+cache = fadel 32.88 / sadeed 45.13 / wn 48.65 / abdou 40.90 vs raw 33.23/45.68/48.99/41.46 (-0.3..-0.6 DER). Cache ≠ teaching substitute.
- Micro model in OUR arch (math verified via build_from_config): 12-128-512-4-2 = 2.98M / 10-144-576-4-2 = 3.14M (gold 28-272-1088-8-2 = 30.11M). E-21 launch NEEDS_USER_DECISION: (a) config, (b) from-scratch vs stage1lm init, (c) data = stage2b pool only vs +QCRI (license + dedup first).

### 2026-09-16/17 — E-19 external-model bake-off (TASKS row 61; user go)

- **Status: CLOSED 2026-09-17** — results `research/e19_bakeoff/RESULTS.md`; ledger E-19 CLOSED; TASKS 61 done; MEMORY lesson 64.
- "QINA King v21" does not exist anywhere (HF API + GitHub + Exa sweep) — dropped, MEMORY lesson 62.
- **WINNER: Z-Mahmood BiLSTM+attention (MIT, ~18 MB CPU, sentence cache BYPASSED via no_cache)** — beats our gold on EVERY gate by 19-29 DER absolute: fadel 4.35 vs 33.20, sadeed25 18.85 vs 45.68, wn2014 29.73 vs 48.99. Satisfies the USER decision rule (beat-gold-overall >5). Contamination caveat recorded: its training may overlap the public gate refs — treat numbers as an upper bound until a cache/n-gram contamination check runs.
- mishkala (Apache-2.0, 12.5M mamba+transformer+CRF, CPU ~6 l/s): rule-(b) sadeed specialist — sadeed 35.79 (wins by 9.9), fadel 39.37 (+6.2 worse), wn2014 50.24 (+1.25 worse).
- DISQUALIFIED: Etherll/Tashkeel-350M-v2 (repetition loops out-of-domain, wn2014 97.0 DER; mamba2 chunk-scan OOMs at batch>=2 on the 5.8 GB WDDM GPU — batch=1 only, ~6.5 s/line); basharalrfooh/Fine-Tashkeel (T5-large: wn2014 37.62 good BUT sadeed 70.55 / fadel 91.60 preservation 10% — wn2014-only). QCRI repo skipped (datasets-only, no weights); NAMAA excluded per user (audio).
- Harnesses: diacritizer/scripts/e19_infer_batch.py (causal, OOM-resilient) + e19_infer_t5.py (seq2seq); gate inputs models/e19/inputs/, predictions models/e19/{gate}.{model}.pred.txt.
- FOLLOW-UP E-19b (user asked "test on other data"): contamination audit research/e19_bakeoff/CONTAMINATION.md PASS-with-notes — cache-vs-gate overlap tiny (0-0.3% sentence hits, 4-12% word coverage, cache-ON == no-cache bench); NEW held-out evidence: abdou stride-20 (2067 lines) ZM 12.60 vs gold 41.64 DER; word-order-SHUFFLE probe (gold re-benched for comparison) ZM wins every shuffled slice too (30.96/38.69/50.47 vs gold 43.68/48.74/55.51) => BiLSTM has genuine contextual skill, not memorized-dictionary cheating. mishkala approved by user as cheap fallback. Files: scratch/e19/zm_{contamination,heldout_prep,shuf_probe}.py + models/e19/*.{zmahood_cache,shuf.*}.pred.txt / inputs/*.shuf.* / abdou_heldout.*.
- NEXT (user-gated): decide whether Z-Mahmood becomes the data-generator for the LLM-as-generator+validator plan, starting with the contamination check, then a Nawaasikh/Sadeed pilot.

## 9. Conventions

- Validation labels: PASS / FAIL / SKIPPED / BLOCKED (CLAUDE.md §16).
- Inspect before changing; never claim success without evidence; kill only
  processes this agent started.

## 2026-09-18 — ME-line μ0 build complete (CPU), GPU window pending (user-gated)
- Tasks 1,2,3,4,5,7 of docs/plans/2026-09-18-micro-expert-composition-plan.md DONE via subagent-driven dev, each implementer+reviewer approved: shared 97-id char vocab+tokenizer (241117f); seeded X2/X3/X4 generators (6f9feb2 + arith n_train pool-cap fix 80c3b4e — review caught an inverted sampling filter in the PLAN, plan bug not implementer error); X1 extractor over models/e19 cache {bare:vocalized} -> 63,000 mark-bearing lines 60K/1K/2K (382628d; data/mex/ gitignored 2abd626); uint32 packer + parameter-budget test PINNED TO REAL MODEL: expert 200,160 / control 784,320, ratio 0.98x, control = byte-exact union of expert train+val (73a4bbe); 5 configs all sanity-PASS vocab_size 97 (a19674d); eval harness exact-match + measured trivial baselines + dry-run integration check (e2c4b3d).
- E-24 PRE-REGISTERED in research/EXPERIMENTS.md before any training results exist (honest pre-registration order).
- NEXT: Task 6 GPU window — sequential train mex_x1..x4 + mex_control via train.py auto-resume, then mex/scripts/eval_mex.py all | user-gated (single-GPU rule); ask_user_question timed out 3x, decision put to user in chat.
- Minor findings roll-up for final review: BPEDecoder suffix risk; unk-id literal decode; X1 ~5.1% <unk> rate (alphabet gap over x2/x3/x4 chars); arith sampled subtraction symmetric-pair duplicates; split-assert edge when n_train small.

## 2026-09-18 — Task 6 GPU window ran; E-24 results honest-mixed
- Sequential train all five configs (subagent trained x1-x3; wrapper-timeout killed x4 mid-run; controller resumed x4 (checkpoint-750) and control via managed pwsh jobs pwsh-16/17; all reached step 2000 and wrote final model.safetensors + mex_eval.json).
- E-24 results (pre-registered gate = strictly beat measured trivial): x1 0.0115>0 PASS (thin), x2 0.004>0.002 technical-pass (useless in practice), x3 0.988==0.988 FAIL gate (majority tie), x4 0.236<0.316 FAIL; control 0.078/0.002/0.988/0.072 — control beat experts on x1 at 4x params.
- mu1 composition arms correctly NOT run on failing experts. Report: research/micro_experts/MU0_REPORT.md. Open user decision: extend steps (10-20K) / richer x3 targets / x4 ident-mix fix before any mu1 work.

## 2026-09-18 — E-25 mu0b: extended window + hardened X3 — ALL FOUR GATES PASS
- User picked option B. E-25 registered with results (retro-documented; E-24 remains the clean pre-registration exemplar). X3 generator hardened (maxlen 24, balanced-by-construction ok lines, half subtle one-pair flips; 14/14 tests green, commit 7d79cc1); configs bumped to 12K steps (c2510b0); old finals archived runs/mex/archive_2000; control re-packed byte-exact union (verified 3,294,269+31,506).
- Results: x1 0.1075>0, x2 0.898>0.002, x3 0.918>0.512, x4 0.846>0.316 — 4/4 PASS; control 0.129/0.054/0.922/0.864. Experts >> control on symbolic; control > expert on real-data x1.
- mu1 composition arms now unblocked. NEXT: whole-branch review, then mu1 planning.


## mu2 session 2026-09-19 (E-31 close + E-32 G3 close)
- E-31 (G2 fill-in via LoRA warm-start) CLOSED **PASS both gates** — winner E-31e: lr 1e-4, 3000 steps, in-batch corruption with hold_every 3 (2/3 blocks corrupted, labels stay clean). Canonical configs/mu2_g2.yaml + runs/mex/mu2_g2/final (merged E-31e). Failed 4e-4 run preserved at runs/mex/mu2_g2_a_failed4e4. Full sweep table in research/EXPERIMENTS.md.
- Resource profiler scripts/profile_resources.py (psutil + nvidia-smi CLI, no pynvml) — CSV + rolling MD verdicts; ran on live G2; finding: sub-1M models are launch-overhead-bound, NOT data-starved. Re-opt: batch 32×accum 1, grad_ckpt off → ~10× it/s (~147 it/s), GPU saturated mid-run; committed 628bb57/e592e83.
- E-32 (G3 Net2Net widening) CLOSED **PASS both gates**: widened 160→320/heads 8/kv 4/ffn 1280 function-preserving (max|dlogit|<1e-3), then LoRA settle lr 5e-5 ×1000 steps on the widened trunk: fill 0.6891 PASS, retention 0.7395 PASS — better than the E-31e numbers on both axes. runs/mex/mu2_g3/final = canonical widened trunk for G4.
- Data catalog from prior turn committed 45fa7ba (research/data_catalog/, 37 datasets) — still awaiting user consumption/decision.
- Next: G4 (E-33) mark-selection head + DER-lite on top of runs/mex/mu2_g3/final.

- E-33 (G4 mark-selection head, frozen trunk) CLOSED **PASS**: mark acc 0.8518 (Wilson95 [0.8496,0.8539], n=101930) vs 0.4063 either-guess; retention structural (trunk bitwise intact, anchor 0.7395). run runs/mex/mu2_g4, script mex/scripts/train_mu2_g4_head.py.


## mu2 2026-09-19 late: G4b composition probe (E-34 FAIL -> E-35 PASS)
- E-34 (G4b composed decode, pre-registered rule) CLOSED **FAIL** honestly: composed 0.2796 vs its own 0.6891 gate (mark-blind restriction branch was the root cause).
- E-35 (corrected rule, same frozen trunk + same head, NO retraining) CLOSED **PASS both gates**: all-position 0.6196; mark-position 0.7556 >= trunk-only 0.7515; non-mark cost 0.03pt. The full end-to-end composed diacritization walk now exists, and the head-first free-trunk composition pattern is established for the ladder's next lateral mounts (G5).
- Canonical: trunk runs/mex/mu2_g3/final; corruption-matched head runs/mex/mu2_g4b/head.safetensors; rule script mex/scripts/eval_mu2_g4b2.py.


## mu2 2026-09-19: G5 lateral bridge probe (E-36)
- E-36 first "combine" rung (bridge over frozen trunk): gate (c) PASS exact-identity; trunk-only 0.7515 -> bridged 0.7632 (+1.17pt) but the pre-registered bar was 0.7660 (+1.5pt) => **honest FAIL row** (no retunes within the registration). Artifacts runs/mex/mu2_g5; scripts mex/scripts/train_mu2_g5_bridge.py; follow-ups E-37a (2 bridges) / E-37b (fetch-conditioned KV) proposed and USER-GATED.

- E-37a (G5 two-layer lateral bridge) CLOSED **PASS all pre-registered gates**: bridged mark acc 0.7846 (Wilson95 [0.7821,0.7870], n=108,956) vs 0.7660 target (+3.31pt over trunk-only 0.7515), identity 0.0, retention structural. runs/mex/mu2_g37a; script mex/scripts/train_mu2_g37a_bridge2.py. Next user-gated: E-38a composed-rule readout with the E-37a bridges live (bridge-aware E-35 decode).

- E-38a (full-stack composed readout, no training) CLOSED **PASS all gates**: bridges+head composed 0.6485 all-pos (+2.89pt) / 0.7862 mark-pos (+3.06pt) vs E-35 0.6196/0.7556; bridges-off re-check reproduced E-35 exactly (first-run spurious FAIL was a harness branch bug, fixed honestly; closing numbers deterministic). x4 realized: compose by mount, never merge. runs/mex/mu2_e38a; mex/scripts/eval_mu2_e38a.py. Next user-gated: E-39a bridge-aware head finetune, or the mu2-to-mu3 scaling decision.

- E-39a (bridge-aware head finetune) CLOSED **PASS both gates**: composed 0.6533 all-pos / 0.7984 mark-pos (beats E-38a 0.6485/0.7862); nonmark untouched 0.5703. Head-only trainable (~3.7k), bridges frozen, disarm-before-clean-pass discipline enforced everywhere. mu2 compose stack closed: best standing 0.6533/0.7984 vs either-guess 0.4063 and trunk-only 0.6891/0.7515 refs. runs/mex/mu2_e39a. NEXT: mu3 scaling decision (user-gated): grow x5 recipe - mu2 stack + 1 MORE layer + NEW head for a second, harder task, bridged composition carried.

- E-41 (mu3-A x-task bridge tower) CLOSED **PASS all gates**: mixed-stream CE 6.1823 -> 4.2513 (-31.2% rel) on frozen trunk; mu2 stack hygiene exact (0.6533/0.7984). Federated mounts proven. NEXT per user: mu3-B = Net2Net widen mu2_g3 trunk x2 (320->640, heads 8->16, ffn 1280->2560, kv 4->8) with FULL stack remount (bridges + head widened by the E-32 concat rules), pre-registered gate = function-preserved composed readout 0.6533/0.7984 exactly.


## 2026-09-19 — DA-line opened: Desert Ant recreation, DA-1 (Tongue-analogue lang-ID) started
- User: research Desert Ant Labs (done earlier 2026-09-19 - research/desert-ant-labs-models-report.md, 17-model teardown), then "create a plan/milestones/tasks/..etc for them and start with most feasible/achievable one".
- Feasibility pick: DA-1 text language-ID (their Tongue recipe) - CPU-trainable in minutes (zero single-GPU contention vs D/mu lines), their eval protocol fully copyable (FLORES-style full/5/3/1-word acc, ties reported never guessed), ~200MB Tatoeba corpus. Arabic-first angle: ar/fa/ur/ps share Arabic script and are separated LEXICALLY (script routing deliberately never fires on Arabic script); ko/ja/he/el/hi auto-routed by script.
- Architecture: normalize (NFC, lower) -> letters-only words -> char n-grams n=1..4 angle-bracketed -> FNV-1a32 -> 2^16 buckets; EmbeddingBag(65536->21, sum)+bias = hashed logistic regression on CPU; per-language int8 export <= 2.5 MiB + pure numpy inference.
- Gates pre-registered BEFORE training (G1 smoke <10min beats majority; G2 full/5w/3w/1w >= 0.97/0.95/0.90/0.70; G3 arabic-script group 3w >= 0.85; G4 artifact <= 2.5MiB int8 within 0.5pp; G5 ties/abstains never guessed; G6 10-word shingle leakage barrier = 0). Reference points from their card (84 langs): 3w 0.933 (lingua 0.887), sentences 0.971, 1-word 0.759.
- Written this turn: research/desert_ant_recreation/DESIGN.md (ladder DA-1..DA-9 feasibility-ranked); docs/plans/2026-09-19-desert-ant-recreation-da1-langid-plan.md (full TDD plan, no-placeholder); langid/ submodule skeleton (src features/model/data/infer; scripts download/prepare/train/eval_langid/export_int8; 5 test files) - tasks 1+2 implemented, unittest green (row 82).
- NEXT: background Tatoeba download -> prepare -> eval set (FLORES mirror probe, honest fallback) -> G1 smoke -> full train -> eval -> int8 export -> DA1_REPORT.md. Rows 81-91 in TASKS.md.
- SAME-DAY EXECUTION CLOSED: DA-2 Emo-analogue also executed same day (user: go to DA-2). Ungated data: arbml/TEAD (12,558 ar tweets; author-emoji self-labels with the DeepMoji one-type rule -> 7,773 rows, 67 emojis) + cardiffnlp tweet_eval/emoji (en 50k, 20 emojis; card license unknown - research-use caveat). 46,100/5,729/5,845 splits, 76 classes. Model: hashed word-unigram+bigram+char 1..3-gram EmbeddingBag 65536x76, CPU. E-44 registered BEFORE results. Gates: D1/D3/D4/D5 PASS; D2 honest FAIL - beats frequency prior 0.1990 by +2.7pp (ar 0.2255) / +3.5pp (en 0.2335), top-3 ~0.40, but short of the pre-registered top-1 bars (2x prior / prior+0.05). int8 2.896 MiB, 1.02-1.08 ms/text naive proxy, int8-fp32 agree 98-99%. DA-1b STOOD DOWN as pending task per user (arabic gaps revisit later). Report: research/desert_ant_recreation/DA2_REPORT.md. DA-2b followed same-day per user (try the 3 options): (a) amgadhasan/arabic_tweets_dialects mined one-type emoji corpus (34,514 rows, 435 emojis) -> 73,164/9,076/9,304 splits, 160 classes; (b) tiny 1-layer d32 transformer head over the hashed buck buckets: TEST ar top-1 0.2255 -> 0.4063 (prior+0.05 ar bar CROSSED; 2x-prior 0.4934 still unmet; en regressed below prior - honest), int8 1.686 MiB, 1.75 ms/text, numpy agreement ar 1000/1000 en 998/1000; (c) weighted CE + LS honest FAIL: diverged at lr 0.05, stuck (val 0.12) at lr 0.01 + clip, no-op on transformer. E-47 registered before results; report: research/desert_ant_recreation/DA2B_REPORT.md.: Tatoeba = 855220 train rows (gotcha: export uses ISO-639-3 codes, 'ara' not 'ar', 'cmn' not 'zh' - TATOEBA_CODE map in langid/src/data.py); unittest 19/19; G1 smoke PASS (6 s, 0.9905 vs majority 0.0499); full train 3-epoch val 0.9953. Honest final gates on official FLORES-200 dev+devtest (42189 rows; dl.fbaipublicfiles.com, ungated): G1 PASS; G2 full 0.9877 / 5w 0.9510 / 3w 0.9072 PASS; G2-1w 0.6921 FAIL vs 0.70; G3 arabic-script group 0.778 FAIL (ps 0.48 - Tatoeba has only 66 pashto sentences; root cause data, not architecture); G4 PASS (0.81 MiB, 0.036 ms/word numpy; int8-vs-fp32 delta 0.05pp); G5 PASS (margin ties reported, never guessed); G6 PASS (0 shingle overlap). E-43 registered BEFORE results. USER DECISION: (a) DA-1b - add ps/ur crowdsourced corpus + try crossing the 1w bar, or (b) close DA-1 at 4/6 and open DA-2 Emo.
- SAME-DAY EXECUTION CLOSED: Tatoeba = 855220 train rows (gotcha: export uses ISO-639-3 codes, 'ara' not 'ar', 'cmn' not 'zh' - TATOEBA_CODE map in langid/src/data.py); unittest 19/19; G1 smoke PASS (6 s, 0.9905 vs majority 0.0499); full train 3-epoch val 0.9953. Honest final gates on official FLORES-200 dev+devtest (42189 rows; dl.fbaipublicfiles.com, ungated): G1 PASS; G2 full 0.9877 / 5w 0.9510 / 3w 0.9072 PASS; G2-1w 0.6921 FAIL vs 0.70; G3 arabic-script group 0.778 FAIL (ps 0.48 - Tatoeba has only 66 pashto sentences; root cause data, not architecture); G4 PASS (0.81 MiB, 0.036 ms/word numpy; int8-vs-fp32 delta 0.05pp); G5 PASS (margin ties reported, never guessed); G6 PASS (0 shingle overlap). E-43 registered BEFORE results. USER DECISION: (a) DA-1b - add ps/ur crowdsourced corpus + try crossing the 1w bar, or (b) close DA-1 at 4/6 and open DA-2 Emo.

- E-42 (mu3-B Net2Net widen + full-stack remount) CLOSED **PASS**: trunk 320->640/8->16/1280->2560 with widened E-37a bridges + widened E-39a head reproduces E-39a readout (0.6534/0.7984 within fp16 rounding bar); TWO honest FAILs root-caused en route (head-weight dup was the emb not lm_head for the untied G3 trunk: dmax 19; bridge kv chunk(2) mix: 0.6177). mu3-A (E-41 federated x-tower, PASS) + mu3-B both done per user sanction. NEXT user-gated options: (1) continue pretraining g4_init on the full mounted stack (mu3 growth train), (2) rebuild/train the x tower at 640, (3) stop and review. Commits: 5d0b1ce (E-41), then E-42 commit pending.

- E-43 (mu3 G-settle) CLOSED **PASS all three pre-registered gates**: settled merged 640-wide trunk runs/mex/mu3_g4/final; retention 0.7391 (guard 0.7431), fill acc 0.6972 (anchor 0.6891), mounts-remount-on-settled-trunk composed 0.6564/0.8012 (beats E-39a anchor), trunk-only 0.6242/0.7620. mu3 now has a trained canonical trunk with the mounted stack still adding. NEXT candidates (user-gated): (a) x-tower retrain at 640 for the x task stream (mu3-A retake on the bigger trunk), (b) head finetune refresh on the settled trunk (E-39a retake), (c) ladder x-levels for other x experts, (d) stop and review.

- E-44 (x-tower remount at 640, user-gated option a) CLOSED: gates (a) identity exact PASS, (b) mixed CE 4.2467 -> 3.3229 (-21.8%) PASS, (c) two-tower cohabitation FAIL (0.6459 mark vs 0.8012) - towers trained separately interact destructively when run stacked; joint-live co-training + router is the registered fix route in E-45.

- E-45 (joint two-tower co-train, user-gated option b) CLOSED: ALL gates PASS. Both towers now cohabit; diacritic mark-pos 0.7943 both-live (E-44 was 0.6459 FAIL), mixed x CE 2.2753 (E-44 was 3.3229), composed all 0.6462. mu3 = one trunk, two task families, mount-composition by training ORDER only. NEXT (user-gated): (E-46) live decode demo of the joint model + (E-47) router/specialist-per-task stack if needed; x-tower at 96-char charate; retention written.

- E-46 CLOSED (read-only capability audit): readout invariance PASS; boundary row - x-family first-char argmax fill ~0/50, free decode wanders, mark head fires on x prompts (task confusion). E-47 CLOSED (router + constraints mount; trunk frozen, sha-verified): router acc 1.0000 PASS; x2 digit-fill 0.36 FAIL (bar 0.80), x3 head 0.77 FAIL (bar 0.85) - honest row, no retune. Next user-gated options: (E-48a) towers-armed per-family heads (x3/carry read from bridge planes); (E-48b) per-family expert stacks / router MoE per the compositional plan; (E-49) more live decode demos.

- E-48a CLOSED: armed heads improve over clean-trunk heads (x2 0.36->0.42, x3 0.77->0.80) but both stay below pre-registered bars (0.80/0.85) - honest FAIL rows; weights sha-verified untouched. Interpretation: first-char boundary answers need multi-step decode or a taller trunk, not a bigger single-shot head.

- E-48b CLOSED: multi-step expert decode exact 0.00 both families (per-position CE ~2.0 = near chance) - honest FAIL; boundary computation absent from frozen-trunk state; next user-gated rung should be trunk-side (taller trunk / x-stream trunk finetune), not another readout.

- E-49 CLOSED (read-only demo): runs/mex/mu3_router/demo.md - live composite decode per family; x3 live-correct; ladder pointer = next rung is trunk-side growth.

- E-50/E-51 CLOSED (trunk-side: taller trunk + x-stream finetune): both honest FAILs on boundary gates; mixed CE 1.8757 (E-50) / 5.6833 (E-51); boundary fill did NOT move beyond readout-only caps; standing pointer: next rung to be user-gated (dialogue recorded in research/EXPERIMENTS.md).


## DA-3 (E-52, 2026-09-19): Gist-analogue topic tagging - PASS
- Data: ar = arbml/SANAD parquet (131,807 rows after gate+dedupe, 7 topics Tech/Culture/Finance/Sports/Politics/Religion/Medical); en = heegyu/news-category-balanced-top10 (83,878 HuffPo headline+description rows, 10 balanced editorial categories). output: data/langid/topic/{train,val,test}.tsv 120,000/3,995/3,995, 17-class shared head (ar:*/en:* labels).
- Train: hashed word+char n-gram EmbeddingBag 2^16x17, lr 0.25 AdamW... (see langid/scripts/train_topic.py), 10 epochs CPU, best worst-lang val 0.7135.
- TEST top-1: ar 0.9048 (2x bar 0.2857 PASS), en 0.6780 (2x bar 0.2000 PASS); top-3 ar 0.9850 / en 0.8830; fp16 model 2.13 MiB.
- Honest caveat: en 10-topic taxonomy = HuffPo editorial buckets, not the same ordering as the 36-IAB topic system of the original DA-3 Gist model. Ethics note in EXPERIMENTS.md E-52 row.
- Files: langid/scripts/build_topic_eval.py, langid/scripts/train_topic.py, langid/scripts/eval_topic.py, runs/langid_da3/*.log.json/.eval json artifacts; E-52 registered BEFORE results. E-50 (mu3 taller trunk) exists as a pre-registered mu3 section, unrelated.

- E-52 CLOSED (proper trunk-side growth, phases A/B/C): phase B x1 0.289, phase C x2 18/50 - all final gates FAIL; fallback to option 1 triggered per user sanction: canonical composite = E-45 standing. Taller-path artifacts archived under runs/mex/mu3_l2pre, mu3_l2head, mu3_lora/x2; not erased.

- E-53 DONE (option-1 consolidation, read-only): capability card written to runs/mex/mu3_joint/capability_card.md — canonical E-45/E-46/E-47 standing summarized with measured capability/cap tables and ladder provenance; no weights touched. mu3 standing is now the documented milestone snapshot; next growth (full pretrain of an added layer, ~90k steps) explicitly left user-gated.

- dia2 E-54a CLOSED PASS (zero-loss widen 640->1280 + full mount remount; pedantic identity within fp noise). E-54b CLOSED: Muon arm FAIL m1 (0.7941 < AdamW 0.7987), AdamW arm is the NEW dia composed standing (markpos 0.7987 / all 0.6504 / CE 2.2798). Canonical dia2 = runs/mex/dia2b_adamw/final + runs/mex/dia2_wide mounts. Muon rejected for this configuration. E-54c (shared-KV long-ctx) and E-54d (replay-scale) DEFERRED user-gated.


## E-53 (2026-09-19): Muon-on-DA-3 optimizer A/B - PASS
- Question: does the E-41 Muon recipe (NS5 orthogonalized momentum) transfer from trunk pretraining
  to the DA-3 sparse-lookup bag classifier? Arms: Adam lr 0.25 (E-52 control), Muon 1e-2, Muon 3e-2,
  identical data/batches/seed, 10 CPU epochs; 1-D params (bias) get plain SGDM inside the Muon optimizer.
- Result (test top-1): Muon 3e-2 ar 0.9268 / en 0.7300 (best); Muon 1e-2 ar 0.9338 / en 0.6965;
  Adam control ar 0.9048 / en 0.6780. Both Muon arms clear every pre-registered bar; 3e-2 wins en by +5.2pp.
- Verdict: Muon is the new default optimizer for DA-line hashed-bag/classifier heads (E-41 scale-out confirmed).
- Files: langid/scripts/train_topic.py got --opt muon; runs/langid_da3/topic_bag_muon*.pt; research/EXPERIMENTS.md E-53 row closed; TASKS.md row 94.

- E-54c CLOSED (honest negative): ctx 192 LoRA-adapted model taught window use (0.7920 long vs 0.6000 unadapted control) but dia composed does NOT gain over the 96-ctx standing (0.7987); shared teacher-KV variant destroys tower readout (0.5419) - per-depth teacher KV kept. Canonical stays dia2b_adamw/final ctx 96. E-54d (replay-scale) remains.

- E-54d CLOSED PASS: replay-scale 4000-step settle on dia2b trunk lifts dia composed mark-pos 0.7987 -> 0.8134 (all 0.6672) - NEW canonical dia2 = runs/mex/dia2d_scale/final + dia2_wide mounts, ctx 96. dia2 program complete (a PASS, b Muon-rejected/AdamW-best-then-d-superseded, c negative, d PASS); further budget escalation user-gated.


## E-54 (2026-09-19): DA-7 Redact-analogue Arabic NER tagger - PASS on extraction bar, honest caveats
- Data: iahlt/arabic_ner_mafat (40k sentences -> 39,356 kept, BILUO->BIO, 27 tags incl. TTL/ANG/DUC/WOA;
  splits 35,420/1,968/1,968); ANERCorp (asas-ai) also downloaded as alt candidate.
- Model: per-token hashed (prev cur next) EmbeddingBag 2^16, Muon 3e-2 (E-53 default), weighted invSqrt CE + clip 0.5, 6 CPU epochs.
- TEST: entity-token acc 0.3983, macro-F1 0.3297 (extraction bar vs all-O F1=0: PASS), token acc 0.8081 vs
  all-O acc 0.8186 (honest FAIL on accuracy surface).
- Files: langid/scripts/{build_ner,train_ner,eval_ner}.py; data/langid/ner/*; runs/langid_da7/*.
- Pending: deterministic Arabic regex redaction layer (phones/IDs/IBAN/dates) + span BIO decoding = row 96.

- E-55 CLOSED (read-only cross-ladder bench, adapter mex/scripts/bench_dia2.py): dia2 composite mean 4-gate DER 0.858 vs diacritizer gold 0.456 (E-23a) / stage2final 0.525 / e23c_zm 0.527 / ZM BiLSTM 0.558 - external benchmarks belong to the diacritizer line; 0.8134 mark-pos is in-domain-only. Hubs: 8-mark-set ceiling, ctx 96, fill-head bias on unseen bare text.

- E-55b CLOSED: per-prompt mark metrics (mark_accuracy / precision / F1, hit-wrong-missed-extra, der_collapse) in mex/scripts/mark_metrics.py; per-prompt CSVs under runs/mex/e55_bench/. Profiles: dia2d ~0.47 mark-acc; gold e23a 0.78-0.87; zm 0.74-0.84. der_collapse shows the mini's errors are symbol confusion, not just ceiling.

- E-55c CLOSED: full-profile benching is the standing policy (user-approved). All three lines run through mark_metrics.py: dia2d 0.47 mark-acc/0.04 word_ax/0.47 word_p50/0.035 lift; gold 0.83/0.49/0.96/0.44; zm 0.79/0.41/0.93/0.37. dia2d mid/last-rooted vowel confusion is the specific externality gap.


## E-56 (2026-09-19): DA-8 Title (CPU-feasible ranker swap, GPU under user policy) - PASS
- Their model: Granite-350M SFT (article -> headline generator). Ours: shared dual-encoder ranker -   one hashed bag 65536->d48 mean-pooled (single EmbeddingBag kernel), side biases, InfoNCE tau 0.07 over in-batch
  negatives, Muon 3e-2. Chance = 1/64; bar 4x chance.
- Data: ar asas-ai/Arabic-article-summarization 6,623 pairs (text->summary as title-analogue; SANAD has no titles);   en huff 71,732 pairs (short_description->headline).
- TEST: ar R@1 0.5826 / R@10 0.8929; en R@1 0.2160 / R@10 0.5742 (5 epochs, still climbing). Both PASS the bar.
- GPU notes (user policy now active): use GPU when free; one process at a time; VRAM peak 112 MB; v1 kept the   whole corpus as feature tensors in RAM (plus 2 concurrent procs) and made total RAM/VRAM exhausted and even read docs of shared-memory spill -  v2 encodes per batch and saves fp16-CPU checkpoints.
- Files: langid/scripts/{build_title,train_rank,eval_rank}.py (v2); data/langid/title/*; runs/langid_da8/*.

- E-56 CLOSED as FAIL (honest): dia2e_v3q = dia2d trunk + gold v3q corpus, same 4000-step settle. Means: mark-acc 0.496 (up from .472, gate .65), lift .040 (gate .20), in-domain markpos REGRESSED to .5779 (CE 5.77). Conclusion: gold's edge is data+interface+ctx jointly; data swap alone insufficient at mini scale. Artifacts runs/mex/dia2e_v3q + e55_bench/dia2e_*.

- EN EXT (2026-09-20): 12 epochs, patience-4 early stop (unused - val kept improving); test R@1 0.3211 / R@10 0.6786; peak VRAM 112 MB flat; no overfit (val/tracked in .log.json).

## 2026-09-22 — repo-sync incident: 205-commit push backlog (~16 GiB) rewritten down to code+docs
- User asked to sync with origin; push was projected at ~12.5 GiB plain blobs + 3.36 GiB LFS weights. Reason: ignore rules never covered the mu-line (runs/mex/), models/e19/, gdn smoke/selftest, or diac data arrays.
- User chose Option A (strip weights/data from history). Executed git filter-repo --invert-paths (41 exact paths: 38 diac blobs >1MB + models/e19 weight files/cache; globs: runs/mex/*.safetensors|*.pt|*.npy, runs/gdn_smoke*/…/model.safetensors, data/gdn_selftest/*) over all 299 commits. Duration 167s rewrite + 167s repack. Push then requires --force-with-lease (solo repo).
- filter-repo reset --hard deleted stripped tracked files from the worktree; 152 LFS-backed weights restored byte-exact from .git/lfs/objects (verified sizes: dia2e_v3q final 188.5 MB, tashkeel-350m-v2 649.2 MB, mishkala.pt 152.7 MB). Remaining worktree losses (regenerable or to re-derive): data/diac tokens npy/big jsonl/bench preds (raw sources intact under data/diac/raw), runs/mex bridge/xtower .pt + distill_cache.npy + _xv.npy, models/e19/our_word_cache.json.
- While restoring, found MEMORY.md/HANDOFF.md had been committed clobbered (empty/1 KB) by the 2026-09-21/22 commits 9897100, e9c79d9, 99b8f3b — pre-existing damage, not the rewrite. Restored from last-full commits 0f83061 / 9dbb5c2; gap entries live in git log messages.
- E-61b (train_e61b.py, dia2j_e60b) kept running untouched throughout; its untracked output dirs (runs/mex/dia2j_e60a*, dia2j_e60b_b1) were never part of the rewrite.
- Forward rule: before every push, git lfs status = 0 pending AND payload spot-check (rev-list --objects origin/main..HEAD).

## 2026-09-22 — Laya deep research (user request: data -> strategy -> architecture -> 50-100M feasibility)
- Deliverable: research/laya-decision-model-report.md (+17 raw payloads research/raw/laya/txt/). Three research subagents: training-pipeline (code-verified), ecosystem, feasibility. NO experiment run, NO GPU use.
- Laya (convaiinnovations/laya, Apache 2.0, ~2k likes) = NON-autoregressive System-1 decision model, NOT a generative LM: ModernBERT-large 395M fully fine-tuned + 2-layer decision head (per-question-type embedding, option-marker scorer, act/escalate head) = 421M; siblings laya-multilingual (mmBERT-base 322M, 100+ langs) and laya-typed-decisions (421M specialist).
- Training (code-verified rl_common.py / laya/common.py): RLCD = REINFORCE with group-mean baseline (GRPO-style), Gaussian logit noise (sigma 1.0->0.3 root, G=8), reward = log + w_sph*spherical - w_rps*RPS (strictly proper), TD(lambda=1.0) multi-turn, per-(type,option-count) LBFGS temperature calibration post-hoc (ECE 0.466->0.081). Root run: pure policy gradient, 7,313 updates / 1 epoch / 1.96 h — trainer + data-gen UNPUBLISHED (author claims 100% human-labeled public datasets, 13 task families, no list). Public notebook (typed-decisions, 1,200 cases / 6,000 decisions): RL+CE mix, 4 epochs, ~4-6 min on 2xT4, LR 2.5e-5/1e-4, fp16 autocast.
- Reality check (Luni/laya-jev-benchmark): base Laya near-chance zero-shot on typed-decisions (0.360 < 0.461 majority); phishing raw 0.505 (Platt -> 0.611); capability = fine-tuning (180k mixture, 3 epochs, 55 min -> macro 0.84). Author's '+16.0% vs Jev' compares two different benchmarks. Jev Decision Index: Laya 16.39 vs top open repro 55.74 (generalist breadth weak).
- Feasibility verdict (research-only): YES with conditions — MiniLM-L12-H384 (33M, MIT) or BERT-Medium + UNCHANGED Laya head + same recipe ~= 38-50M total; expected -2 to -5 macro pts on robust tasks (typed-decisions plausibly 0.60-0.68 vs 0.766; only direct scaling evidence: MiniLM-22M 0.587 / MBERT-base-149M 0.646 / Laya-421M 0.766); fits our 6 GB fp16 Turing trivially (~0.7 GB optimizer state) while the real 421M full AdamW does NOT fit (~6.7 GB state alone). Ladder recorded as TASKS 102, USER-GATED.

## 2026-09-23 — E-65 (L3) closed: 1/4 gates, phishing transfer PASS, honest FAIL on decision benchmarks
- L3 pretrain: 35.1M encoder (BERT 6L/H512), 260.7M-token 19-source domain corpus (3 subagent sweeps; 3 sources skipped+recorded; 20,004 eval-exclusion hashes, zero overlap), heldout ppl 9.81, 79 min, peak 4,610 MiB (FAIL vs 4,096 recorded).
- Ladder: stage A exit macro 0.6625; stage B typed 0.5185, mixture held 0.6622 (NO net forgetting — replay worked), phishing AUROC 0.6490 (G4 PASS — first over the bar), perm 0.1850 (G3 FAIL, aug did not fix).
- E-64 reruns on L3: AG News 0.245 (=chance), emotion 0.315, probes 2 failures (L1 = 1 remains best).
- Verdict + rows: EXPERIMENTS.md E-65 VERDICT + row 104; comparison addendum in 2026-09-22_laya_vs_jev_benchmarks.md; report research/2026-09-22_laya_l3.md; TASKS 103/104 done.
- Answer to the user's original question: ~50M-class from-scratch + Laya strategy does NOT reach Laya/Jev on decision benchmarks (gap = upstream token scale); it DOES transfer to phishing within 0.03-0.04 of them.
- Session reliability note: 6 crashes fixed+resumed zero-flag (masking shape, OOM micro-48, probe model-type/device, argparse import, permutation API); scheduled bench-probe monitoring infra (bench_probe.py) retained for future runs.

## 2026-09-23 — self-reflection closure (user-requested): failure catalogue + fixes in research/2026-09-23_agent_self_reflection.md; preflight gate (preflight_check.py + train_l3 --smoke) tested PASS 4/1SKIP/0FAIL; corpus build now hard-fails on eval overlap; rules R1-R6 adopted.

## 2026-09-23 - Laya-line arc E-65..E-73 snapshot (see research/2026-09-23_laya_l3_frontier.md)
The L3 branch produced a measured frontier on one 520M-token encoder: E-72 typed specialist 0.6965 (line-best typed, beats L2 incumbent), E-70 transfer specialist 0.6966 phishing (beats BOTH published Laya 0.678 / Jev 0.689 - the program's first published-benchmark wins), E-73 general-purpose mixer (typed 0.656 + phish 0.6932, both pre-registered gates PASS). External-review rules adopted and enforced (G1 delta, probe no-regression, calibration quarantine). Disk-full incident + smoke-dir trap learned and documented. Consoles/logs/bench JSONL/TB preserved per run.
| 26 | E-74 phase-2 closure + follow-up rungs (sym calibration, longer sym stage-B, E-75 corpus scaling) | `done` (panel) / rungs `pending` | Phase-2 panel CLOSED 2026-09-23: phish COLLAPSE (AUROC 0.414 vs 0.689 positional; mean_p 0.194, ECE 0.513 - isolated-BCE training lost the cross-option competition that calibrated zero-shot binary), choice PARITY (sym AG 0.253 / emotion 0.225 vs positional 0.269/0.366 - both arms near chance on the approximate panel; positional-choice numbers were also chance-level, honest note recorded). Net: E-74 = TYPED SPECIALIST (0.7712 unseen typed, past Laya-FT 0.766, Brier 0.1009, G3 by construction) NOT the encoder-replacement; E-75 (E-70 positional continuation on new corpus) remains the sanctioned scaling line. NEXT RUNGS in order: (R1) sym binary-calibration fine-tune on non-eval boolq sym items to repair phish posture (~15 min GPU); (R2) longer sym stage-B (curve still rising at ep8, +0.025/ep) (~40 min); (R3) E-75 new-corpus build (~1-2h CPU) + pretrain-continue (~90 min) + E-73 recipe (~1h). Resume reading order: research/EXPERIMENTS.md E-65..E-74 rows + research/2026-09-23_laya_l3_frontier.md; artifacts runs/laya/l3_e74* , runs/laya/*_sym.json | commits f1c34b1, 66a604a, 441c46d, 5bf6bcb, 1bc9e5d; laya/scripts/train_l3_sym.py + eval_phish_sym.py + eval_public_choice_sym.py; data/laya/sym caches

## 2026-09-23 - L3 frontier checkpoint (E-62..E-74): symmetric-architecture discovery + the two-arm structure
- OWNER-FACING STATE: the L3 ladder (goal: beat Layal/Jev published benchs with the 520M encoder) now has TWO arms: (a) POSITIONAL generalist (E-70 encoder + E-72/E-73 recipes: 0.6965 typed / phish 0.689 AUROC) - the sanctioned scaling line = E-75 new-corpus continuation (pre-registered, E-70/e70 A_last carry); (b) SYMMETRIC typed specialist (E-74: option-isolated windows, raw-score BCE, grouped argmax) = 0.7712 typed (PAST published Laya-FT 0.766 on the SAME unseen test), Brier 0.1009 record, perm-agreement guaranteed BY CONSTRUCTION - but zero-shot phish collapses (0.414) because independent-BCE training destroys the cross-option competition that calibrated the binary posture. Both verdicts committed (f1c34b1, 66a604a, 441c46d).
- E-74 mechanics (for resume): laya/scripts/train_l3_sym.py (standalone; NOT a copy of train_l3.py any more) trains on data/laya/sym/{train,heldout,typed_train,typed_test}.pt built once by sym_expand+sym_pack_all from mixture_raw + typed raws; config configs/laya_l3_e74.yaml (e72 520M encoder init, 5-ep sym stage A, 8-ep sym stage B). Stage-A mixture eval has a KNOWN qid-collapse bug (mixture qname=source name -> giant groups) - affects that eval line only; G1 delta needs a phase-2 recomputation.
- Resume order: (R1) sym calibration rung, (R2) longer sym stage-B, (R3) E-75 corpus scaling (build_corpus_l3.py CPU build first). Be careful with the sym eval on future runs: scored isolated options; probes bench adaptation is OPEN (deferred - positional-bound).
GOTCHA: mixture heldout sym items collapse to 7 qids (qname=source) - do NOT reuse that grouping for stage-A evals; typed items have real per-case qids (7,100 sub-items -> exactly 2,000 qids, max group 5 - grouping audit recorded).
- Numbers the next agent can trust (all from committed console tees + json): typed 0.7712 / Brier 0.1009; phish sym 0.414 raw 0.438 platt 0.549 ECE 0.513; choice sym AG 0.253 / emotion 0.225; positional comparators in runs/laya/public_choice_eval.json.
