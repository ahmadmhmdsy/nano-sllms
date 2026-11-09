# SA-6 RL run 5 proposal (DRAFT for user review, 2026-10-26)

Status: proposal only. Nothing is built, probed or launched.
Init and KL reference: **runs/sa4_scale075**. SA-5 stopped after round 1 (HANDOFF 2026-10-26e), so the SA-5 output is the SA-4 incumbent.
Plan v2 row: "RL run 5 with the fixes in §4", ≤ 5 h, gate "≥ SA-5 on `final`".

## 0. Mistakes to avoid (found while planning)

1. **Prompt-format mismatch.** `scripts/rl_smollm2.py` trains on playground `rl`-split prompts: raw `# comment` tasks plus docstring and MBPP renderings. Since run 4, the init was SFT'd on SA-2/SA-4 prompt shapes (he, mbpp, fix, repair, tool). Running RL on the old pool would pull the model back to the old formats. It would also train on the same distribution the B1/W2 ladders measure, which is how run 3/4 breadth gains ended up in-distribution.
   → Recommend RL on the **SA-5 task pool**: same prompts, same scorer (`sa5_rft.score`), and a disjoint train slice. The playground ladders then become a held-out check.
2. **SA-5 showed that self-sampled positives alone do not move this model.** RL differs in two ways:
   - It also pushes down failing samples (negative advantage). RFT never sees those.
   - It sharpens the distribution toward what passes. That usually helps greedy/pass@1 first, and greedy HE+MBPP is exactly the metric SA-5 failed.

   This is a reason to try, not a promise. §4 has an early stop/go so a flat run costs ~1.5 h, not 5.
3. **Constrained format 145/146 is a grammar gap, not a model gap.** The SA-7 grammar lets a call list repeat the same call until the budget runs out (xlam/49334). If the SA-6 gate keeps 146/146, a run can fail on a row the grammar should block.
   → Recommend a CPU-only grammar fix first: max calls per list, plus no exact-duplicate call. Then re-score init and the candidate with the same grammar.
4. **Throughput.** The current loop generates one prompt × 8 per step: ~11 s/group at 256 tokens in run 4. SA-5 measured batched generation as much cheaper per sample. Without batching, 5 h buys only ~1,000 groups at 384 tokens. With batching it buys several times that.

## 1. Task pool and reward (no download)

**Pool.** Use `sa5_rft.pools()` + `round_slice()`:
- Code: data/sa2 he/mbpp rows with ≥ 3 asserts.
- Fix: SA-4 fix + repair rows.
- Tool: SA-4 xLAM rows.

Order of use:
- **Slice 1 first: the 1,100 round-1 tasks.** Their k=8 results from runs/sa4_scale075 are already in data/sa5_r1/samples.jsonl, and SA-5 round 1 was rejected, so the init never trained on them. 441 of the 1,100 have mixed outcomes (1–7 of 8 pass), which is the RL signal:

  | shape | all fail | mixed | all pass |
  |---|---|---|---|
  | he | 120 | 42 | 7 |
  | mbpp | 96 | 33 | 2 |
  | fix | 64 | 124 | 72 |
  | repair | 58 | 114 | 68 |
  | tool | 136 | 128 | 36 |

- **Then slices 2+ (new tasks).** These are filtered online by dynamic sampling (§2).
- Mix target per step: code 40 %, fix/repair 35 %, tool 25 %. Code is overweighted because it has the fewest mixed groups and is what greedy HE/MBPP measure.
- `val` and `final` are never trained on. The 567-task val subset from the SA-5 gate stays the primary metric.

**Reward.**
- Code/fix: binary, 1 if every assert passes under `run_guarded`, else 0. Same rule as SA-5.
- Tool: 1 if the call list matches the reference exactly.
- A sample that never stops (no EOS and nothing cut) gets −0.1, the existing invalid-penalty value. This replaces SA-5's "drop it" rule.
- No length penalty (plan v2 §4).

## 2. Changes to scripts/rl_smollm2.py

All behind flags. Defaults reproduce run 4 exactly.

| flag | what | why |
|---|---|---|
| `--source {playground,sa5}` | sa5 = the pool and scorer above, imported from sa5_rft.py, not copied | §0.1 |
| `--prompts-per-step P` (default 1) | generate P prompts × 8 in one batch; one optimizer step over all non-flat groups | §0.4 |
| `--loss-norm {seq,const}` (default seq) | const = Dr. GRPO: divide by max_new_tokens, not by each sample's own length | plan v2 §4: per-length normalisation favours short answers (the one-line collapse) |
| `--dynamic-sampling` | drop flat groups (all equal rewards) and fill the step with fresh prompts, up to 3 refill rounds | DAPO; plan v2 §4 |
| `--slice-start` | which round_slice to start from (1 = the measured 1,100) | §1 |
| logging | per step: mean token entropy of the sampled tokens, mean and p90 completion length, flat-group share, pass rate per shape | plan v2 §4 |
| `--stop-entropy F`, `--stop-length F` | stop cleanly (save state, write final) if the 50-step mean entropy or mean length falls below F × its first-50-step value | collapse guard (plan v2 §4) |

**Fixed settings.** Most come from run 4:
- LoRA r16 on q/k/v/o, merged into the init.
- lr 2e-5, group size 8, temperature 0.8, full-distribution sampling (no top-p, so the run stays on-policy).
- KL β 0.05 to the init.
- max_new_tokens 384 for code/fix and 256 for tool.
- VRAM cap = fresh free − 0.75 GiB.

**Validation before any GPU use.**
- CPU dry run, `--source sa5 --dry-run`: every reference completion must score 1.
- Unit check: with P=1, `seq` and no dynamic sampling, the loss equals the old code path on a fixed toy batch.
- `ast.parse` check, plus the existing pytest suite.

## 3. Evaluation during the run

- `--checkpoint-every`: a merged checkpoint every ~25 % of the run, scored with the existing greedy probe (first 50 HE + 50 MBPP).
- **Selection** uses dev/val only. Candidates are the best checkpoint by sampled pass@1 on the 567-task val subset, and `final`. Both get the full L0 battery.
- Also LoRA-scale the RL adapter at 0.75 and 1.0, as at SA-3/4/5. This is cheap, and 0.75 won before.

## 4. Budget (≤ 5 h, with a stop/go point)

| step | time | note |
|---|---|---|
| Build + CPU dry run | 0 GPU | user reviews the diff |
| GPU probe: 40 steps, P = 8 | ~10 min | measures s/step, peak VRAM, flat share, entropy; picks P |
| RL part 1 | ~1 h | the 1,100-task slice, mostly informative |
| **Stop/go** | — | val pass@1 at the 1-h checkpoint (≈ 10 min). Continue only if it is above init's 29.5 % (paired, see §5); otherwise stop and report |
| RL part 2 | ~1.5 h | new slices, dynamic sampling |
| Gate | ~1 h | val pass@1 for 2 arms, L0 battery as at SA-5, then `final` once |

Disk: 24 GB free. Expect about 2 GB (four merged checkpoints of 0.27 GB each plus two scale merges). Checkpoints that are not chosen can be deleted after the gate, with your OK.

## 5. Gate (against runs/sa4_scale075; init numbers are already measured)

Up:

- **Primary:** sampled pass@1 (k=4) on the same 567 val tasks. Init: 29.5 %.
  - A gain counts only if a paired bootstrap over tasks gives a 95 % CI for the delta above 0 (plan v2 §3 statistics rule).
  - The SA-5 deltas (±0.3 pp) would not pass this.
- Greedy HE + MBPP ≥ 71 (init 14 + 57).

No L0 regression:

- English BPB ≤ +3 % vs init (0.5269).
- Constrained format: 146/146 with the fixed grammar (§0.3), or ≥ init under the same grammar.
- Unconstrained format valid ≥ 140.
- Fix tier ≥ 27.
- Tool exact ≥ 50.

Plan v2's SA-6 gate is "≥ SA-5 on `final`":

- After choosing on dev, run once on the `final` split for init and the chosen candidate:
  - `bench_ladder --split final`, B1 and W2.
  - The `final` playground fix and tool tiers.
- The candidate must be ≥ init on each.

If the gate fails, runs/sa4_scale075 stays the incumbent and goes to SA-7.

## 6. Points for you to decide

1. **Task pool:** the SA-5 pool (recommended, §0.1) or the playground pool as in runs 1–4?
2. **Grammar fix before the gate** (CPU only, recommended), or keep 146/146 under the current grammar?
3. **Batching (`--prompts-per-step`)** changes the update cadence compared with runs 1–4, so results are not directly comparable to those runs. OK?
