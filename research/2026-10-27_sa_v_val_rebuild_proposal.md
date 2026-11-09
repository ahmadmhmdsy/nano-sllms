# SA-V duplicate-aware val rebuild proposal (DRAFT for user review, 2026-10-27)

Status: proposal only. data/sa2 and data/sa4 are untouched. Nothing is launched. RL run 5 is paused at part 3 step 107 (state in runs/sa6_r5_part3).
Why: 424 of the 567 §5 dev-gate tasks have a strong duplicate in train (HANDOFF 2026-10-27b, research/raw/sa_val_leak_audit.json). That leaves 143 clean tasks, only 3 of them fix/repair, so the gate cannot measure the gains RL is aiming for.

## 0. Finding that changes the plan

A fresh duplicate-aware split only gives a clean val **for models trained on the new train split**.

Every checkpoint we have (SA-2 → SA-4 → sa4_scale075 → SA-6) was trained on the old `data/sa2/train.jsonl`. That file holds about 99 % of the rows any new val would contain: every new val row that was not already in old val.

So re-splitting data/sa2 and re-baselining sa4_scale075 on the new val would measure memorisation again, only more of it. A full re-split pays off only when the SFT chain is retrained on it.

This splits SA-V into two parts:

- **SA-V.1 (now, small).** A clean gate carved from the *current* val: old-val rows whose whole duplicate cluster lies in old val, with no near-copy in train. This gate is valid for every existing checkpoint.
- **SA-V.2 (deferred).** A cluster-level re-split of data/sa2 → data/sa4, plus a train↔val guard in sa2_build. This runs only together with the next retrain of the SFT chain.

## 1. What was measured (CPU, read-only, nothing written to data/)

The simulation script is `cluster_sim.py` / `cluster_optA.py` in this session's scratchpad. It union-finds all 141,338 data/sa2 rows (train + val) over four exact keys:

- parent id (`x/fix` → `x`)
- assert set
- whitespace-collapsed completion
- whitespace-collapsed prompt

**Clusters**

- 80,282 clusters.
- Links by key: parent 28,081, tests 18,661, ref 13,983, prompt 331.
- Size histogram:

  | cluster size | count |
  |---|---|
  | 1 | 58,616 |
  | 2 | 16,735 |
  | 3–9 | 4,311 |
  | 10–49 | 515 |
  | ≥ 50 | 105 |

- The largest clusters are classic problems that OCI repeats over and over, under different ids and with near-identical tests:

  | problem | rows |
  |---|---|
  | fibonacci | 1,738 |
  | factorial | 1,713 |
  | is_palindrome | 821 |
  | binary_search | 796 |
  | fibonacci_sequence | 740 |

  These are real duplicate families, not chaining artefacts.

**How leaky the current val is**

Of the 1,382 old-val rows, 836 (60 %) sit in one of 672 clusters that also have rows in train. That counts exact keys alone, before any 13-gram check.

**Option SA-V.1: pure-cluster old-val rows**

418 code rows have their whole cluster in old val. After dropping rows with 13-gram coverage against old train, and keeping only rows with ≥ 3 asserts:

| 13-gram cut | he | mbpp | fix/repair |
|---|---|---|---|
| coverage < 0.5 | 182 | 196 | **0** |
| coverage < 0.8 | 190 | 225 | **0** |

No fix row survives. Every val fix row's parent function was split by its own id hash and landed in train. **Fix/repair cannot be gated cleanly on any existing checkpoint from SA data.** The held-out tier-fix eval (research/raw/*_tier_fix.json) stays the fix signal.

**Option SA-V.2: cluster-hash re-split at 1 %**

- New val: 1,267 rows (he 471, mbpp 411, fix 249, tool 136).
- With ≥ 3 asserts: he 470, mbpp 410, fix 249. After the SA-4 fix/repair split that is about 125 + 125, below the gate's 267, so the val fraction or the gate's fix count has to change.
- Residual 13-gram coverage of the 1,131 new-val code rows against new train: ≥ 0.5 on 89, ≥ 0.8 on 8, = 1.0 on 3. Exact keys alone are not enough; the near-dup pass in §3 is needed.
- None of the 5 largest clusters hashed into val. A rule still has to make that guaranteed (§3).

## 2. SA-V.1: clean gate on the current val (recommended now)

1. **Gate list (CPU).** Make `scratch/sa_v_clean_gate.py` from the simulation. It writes `research/raw/sa_v_clean_gate.json`: the ids, the rule, the counts, and the per-row max coverage. Rule: old-val he/mbpp rows, ≥ 3 asserts, whole cluster in old val, 13-gram coverage < **0.5** against old sa2+sa4 train. Expected: **378 tasks** (he 182, mbpp 196).
2. **Gate flag (code, additive).** Add `--task-ids <json>` to `sa5_rft.py`. With `--split val`, it restricts the code pool to those ids. The default behaviour and the 567-task gate are unchanged, so old results stay comparable.
3. **Acceptance (CPU).**
   - Re-run `scratch/sa_val_leak_audit.py` on the 378 ids (it needs a `--ids` option). Expect exact_prompt = exact_ref = same_tests = 0 and coverage ≥ 0.5 = 0.
   - Run `sa5_rft --dry-run` on the 378 ids. Every reference must pass.
4. **Re-baseline (GPU, separately gated).** Gate sa4_scale075 and c107 (runs/sa6_r5_part3/checkpoint-107) on the 378 tasks, k = 4, as on 2026-10-27, with the VRAM cap recomputed at launch. Then run the paired bootstrap.
   - At 378 tasks, the 95 % CI half-width should be about ±0.008, against ±0.012 on the 143-task clean subset.
   - This is the number that decides whether RL run 5 resumes.
5. **Docs.** Update the PLAN §5 gate definition: the clean gate is primary for code; the 567 gate is reported only for continuity; fix comes from tier-fix only.

Cost: CPU minutes plus one gate run per checkpoint. No data/ file changes.

## 3. SA-V.2: cluster re-split (deferred until the next SFT-chain retrain)

These steps are written now so the guard exists when it is needed. None runs before that retrain is approved.

1. **Split function in sa2_build.py.** It replaces the id-hash lines at sa2_build.py:634-635.
   - Union-find over the four keys in §1. Cluster id = smallest member id, so the split is stable under row order.
   - A cluster is val if `sha1(cluster_id) % 100 < --val-pct`.
   - Clusters larger than `--max-val-cluster` (default 50) always go to train. Common concepts like fibonacci are poor dev tasks, and one of them would swamp val.
2. **Near-dup pass.** Compute each val candidate's 13-gram coverage against train with the audit rule (n-grams shared by 2+ val rows are boilerplate). A candidate with coverage ≥ 0.5 moves to train together with its cluster.
3. **Hard guard.** After the split, assert no shared id, parent, assert set, reference or prompt across train/val. On any overlap, fail the build. Write `report["split"]` with cluster stats.
4. **New directories, old kept.** Write `data/sa2v` (a standalone resplit of the existing rows, no re-verification) and then `sa4_data.py --src data/sa2v --out data/sa4v`. data/sa2 and data/sa4 stay as they are, so this is reversible and the old runs stay reproducible.
   - Note: sa4's `--plain-frac` sample draws from the new train pool, so data/sa4v train is not the set SA-4 trained on.
5. **Val size.** Recommend `--val-pct 2`. That gives about 2,500 rows and about 250 fix + 250 repair, enough for a 300 + 267 gate. Train shrinks by about 1 %.
6. **Acceptance.** Run the leak audit with `--sa2 data/sa2v --sa4 data/sa4v`: zero exact-key leaks, coverage ≥ 0.5 = 0. Then the sa5 dry-run, then sa3_data tokenisation of data/sa4v.

## 4. Optional bundle: grader timeout fix (pending, not part of SA-V)

`run_guarded` timeouts and RSS kills are scored 0 in `sa5_rft.score` and `rl_smollm2.sa5_reward`, and `VerifyCache` does not cache them. Proposed fix:

- Retry once with a longer limit.
- If the retry passes, mark the sample NOT_EVALUATED: excluded from the gate mean and dropped from the RLOO group.
- If it still fails, score 0.

This changes scores, so if it goes in, it goes in **before** the §2.4 re-baseline and both arms are scored with it.

## 5. Decisions needed

1. Approve SA-V.1 steps 1–3 (CPU and code only)? *Recommend yes.*
2. Coverage cut for the clean gate: 0.5 (378 tasks) or 0.8 (415 tasks)? *Recommend 0.5, the same bar as sa2_build's bench decontam.*
3. Bundle the grader timeout fix before the re-baseline? *Recommend yes: one re-baseline instead of two.*
4. SA-V.2 now or deferred to the next SFT retrain? *Recommend deferred: it cannot help any existing checkpoint.*
5. The GPU re-baseline (§2.4) is a separate approval after steps 1–3 pass.
