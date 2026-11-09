# Mounting A/B report — 5 arms at pilot dims (TASKS row 49)

Executed 2026-09-11 -> 2026-09-12, per [docs/plans/2026-09-11_mounting_impl_plan.md](../docs/plans/2026-09-11_mounting_impl_plan.md).
Frozen SmolLM2-135M teacher, dual-tokenized teacher stream (train 40,907 / val 846 blocks, ctx 512),
12L/768 from-scratch pilot student, 1000 steps, b1xaccum32, lr 4e-4, fp16, seed 42, single sm_75 GPU.

## Headline: wall-clock to matched-eval parity

Wall-clock per arm (tee-file first->last write, includes in-arm evals/saves; drop includes the
step-300 crash + device-bug fix + zero-flag resume):

| arm | wall-clock | in-run eval @500 | final @1000 (trainer curve) | independent eval (Step 6.2, eval_report.json) | ppl |
|---|---|---|---|---|---|
| control (train.py, no mount) | 1h23m (19:02->20:24) | 2.2215 | 1.8094 | 1.8094 | 6.11 |
| fill (annealed KD, no bridges) | 4h11m (20:24->00:35, incl. cliff pauses) | 6.4729 | 3.6798** | 3.6798 | 39.64 |
| gate (bridge, tanh-gate anneal) | 5h10m (00:36->05:46) | 2.3461 | 1.8442 | 1.8442 | 6.32 |
| drop (stochastic head severance) | 5h34m (05:46->11:20, incl. crash) | 2.3972 | 2.0473** | 2.0473 | 7.75 |
| hybrid (gate + drop mechanics) | 2h05m (11:55->14:00) | 2.2984 | 1.8153 | 1.8153 | 6.14 |

** = final_eval from train_summary.json, not the checkpoint-1000 trainer-state value (see caveats).

Wall-clock is the headline metric per plan/honesty gates: teacher-touching arms pay ~3-4x over the
plain control for the same 1000 steps. At the eval trajectory level, gate/hybrid track control within
+1.9% / +0.3% at 1000 - the Mount A/B/C wiring costs wall-clock but does not degrade beyond
truncation-level error at this scale, while fill (pure-KD probe baseline) collapses.

## Full results table

| arm | student params | cliff events | paused-extra | resumed | fp16 grad-norm (median/max) | NaN | best eval (run) | independent eval |
|---|---|---|---|---|---|---|---|---|
| control | 106.0M | - | - | - | 0.74 / 2.1 | 0 | 1.8094 | 1.8094 |
| fill | 106.0M | 2 (step 400 delta +2.62, step 500 delta +0.17) | +300 | - | 1453.7 / 20399.9 | 0 | 3.6798 | 3.6798 |
| gate | 134.35M | 0 | - | - | 0.99 / 2.1 | 0 | 1.8442 | 1.8442 |
| drop | 134.35M | 0 | - | ckpt-300 (device-bug crash) | 1.03 / 2.1 | 0 | 2.2881@900** | 2.0473 |
| hybrid | 134.35M | 0 | - | - | 0.85 / 2.1 | 0 | 1.8153 | 1.8153 |

(*) drop quirk: stored "best" (ckpt-900 = 2.2881) is worse than the saved-final independent eval (2.0473);
best-tracker used the mount-aware in-run metric while the independent metric differs - both reported.
fill quirk: the in-run curve diverges after the cliff (flat ~6.5), the final artifact evaluates at the
best-restore level (3.6798, bitwise equal to its step-300 eval) - consistency noted, not claimed as recovery.

## Independence gate (user third gate) - PASS

- Every arm's saved final artifact: 110 tensors, zero names containing bridge / kv_proj / probe / _mount
  (verified programmatically; the save-final contract held).
- Stage-1 mount-on hold-end eval vs Step 6.2 independent eval:
  control 1.8094->1.8094, gate 1.8442->1.8442, hybrid 1.8153->1.8153, drop 2.0473->2.0473, fill 3.6798->3.6798.
  Identical to 4 decimals: detaching the bridges costs nothing measurable at end state. Honest reading -
  by step 1000 the tanh-gates had not accumulated a measurable net contribution (control still edges all
  mount arms: 1.8094 < 1.8153 hybrid < 1.8442 gate); bridges traded wall-clock for parity, not gain, at this scale.

## Honest gates (Task 7)

- +20% matched-step wall: never triggered for gate/hybrid/drop (drop peaked at +8.6% @700). Fill diverged
  beyond the wall at matched steps (e.g. @500 6.47 vs 2.22) but finished its scheduled anneal naturally;
  its final is reported FAIL as a KD baseline, not a stop-flag case.
- Wall-clock: headline confirmed (teacher-touch arms 4-5h+ vs control 1h23m; fill pauses cost extra).
- VRAM peak: measured 2,689 MiB / 6,144 MiB during a live gate arm - well under the plan's 6 GB bound.
- fp16 stability: zero non-finite log entries in all five arms; only fill's grad-norm runaway
  (median 1454, max 20400) - consistent with its diverged post-cliff eval.

## Per-arm notes

- control: plain trainer, no mount block; anchors 100..1000 exactly on the plan table; own pace much faster
  (no teacher forward).
- fill (annealed KD baseline): probes without bridges; the plan-predicted cliff arrived at step 400
  (delta 2.62 right after the 300->400 anneal transition); probe pauses (+150, then +300) halted
  teacher-stream updates but the student never re-entered control territory; grad-norm runaway indicates
  the anneal-off transition destabilized fp16 training.
- gate (Mount A): zero cliff events; smooth monotone descent 2.8305@300 -> 1.8442@1000; the arm paying the
  full teacher-forward cost every step (~19-20 s/it -> 5h10m).
- drop (Mount B): post-severance teacher-skip translated into real per-step pace gains once severance ramps;
  stochastic keep-one rule prevented full severance (as designed - never claim full severance).
- hybrid (Mount C): best mount arm at both 900 (1.8249) and 1000 (1.8153) - gate anneal first, then drop
  mechanics late-run enable teacher-forward skipping, producing the best wall-clock of all teacher arms
  (2h05m) - the mixed schedule does not stack the gate arm's full teacher bill. If one arm scales up, it is this one.

## Pros/cons mapped to later use cases

- gate: simplest wiring, smoothest KD injection; cost = full teacher forward every step. Choose when
  wall-clock budget is generous or the column-teacher stream is cheap to serve.
- drop: cheapest mid-run teacher cost; accepts a few tenths of eval loss plus a best-tracker mismatch
  (ops friction). Choose under tight GPU budget.
- hybrid: near-control quality (delta +0.3%) at drop-level late-run cost and the best bridge-utility per
  wall-clock hour overall. Default recommendation for the target scale.

## Scale-transfer caveats (honesty)

- All arms at pilot dims with the SmolLM2-135M teacher; no cross-scale claim beyond this rig.
- Eval set = the dual-tokenized teacher-stream val (846 blocks); Step 6.2 standalone eval.py used the same
  stream at ctx 512 = the trainer-side eval points, so numbers are directly comparable.
- Full severance was NOT reached (keep-one rule); deterministic full-severance quality cannot be claimed.
- Eval methodology note: for fill/drop, final_eval (trainer.evaluate after best/restore bookkeeping) differs
  from the ckpt-1000 trainer-state curve; the authoritative independence numbers are the Step 6.2
  eval_report.json values. Both sources are cited rather than silently merged.
- Kill/resume: exact zero-flag resume verified twice in this experiment - the Step 5.5 drill and the
  in-run drop crash (trajectory continuity at checkpoint-300; device-bug fix landed before resume).

## Artifacts

- runs/mount_{control,fill,gate,drop,hybrid}/final/{model.safetensors, train_summary.json, eval_report.json}
- runs/mount_*/checkpoint-*/trainer_state.json (curves); runs/mount_*/logs (tfevents)
- Raw consoles: arm_{control,fill,gate,drop,hybrid}_tee.txt
- SDD trail: .superpowers/sdd/mounting/progress.md + task briefs/reviews
- Weights stay LOCAL per AGENTS s5 (no LFS upload).
