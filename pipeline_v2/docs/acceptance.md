# Release 1 acceptance runbook

This runbook is the procedure. It is not permission. Nothing in it may be run
until a **user-opened GPU window** exists, and no step here may be started as a
side effect of ordinary planning or CPU work.

## The gate that governs every GPU step

A GPU command runs only when all of the following are true:

1. The user has opened a GPU window. Nothing else opens one.
2. No repository training job is already running, and no second job is co-running.
3. A pre-registration record exists in the experiment registry naming the exact
   namespace, config path, resolved-config hash, exact command, requested device,
   expected evaluator suite and gates, metric source, data/tokenizer/model
   identities, code and dependency hash, and read-back paths.
4. A fresh `preflight --gpu` has just recorded the current device, driver, free
   VRAM, free disk, and the decision with its reason.

If any of the four is untrue, the step does not happen. The correct result is
`BLOCKED` or `NEEDS_USER_DECISION`, never a quiet attempt and never a PASS.

`preflight --gpu` is the only V2 command that probes a device. `doctor` is
CPU-only. No plan, validate, explain, or diff command touches a device.

## OOM policy, decided once and never improvised

If an OWN job runs out of memory:

- write the health event,
- preserve the last valid checkpoint,
- release the lease only after the owner process is gone,
- set the run status to `BLOCKED`.

Batch size, context length, optimizer, precision, and the command are NOT
changed. A reduced next attempt requires a separately approved versioned launch
profile AND a new run identity recorded in the experiment record. If that
approval is absent, the run stays `BLOCKED` and reports
`NEEDS_USER_DECISION`. It does not retry.

No foreign process is ever killed. A busy or uncertain device is `BLOCKED`, and
the recovery procedure must prove the owner is dead before the lease is reused.

A long run can also be halted without killing the process: `run --stop`
places a STOP flag that the train stage honours at its next optimizer-step
boundary, preserving the checkpoint from that step (cooperative stop).

## Exact-command resume

Resume is the original command with **zero additional flags**. Not `--resume`,
not `--seed`, not `--device`. The exact same invocation.

Any change to the config, the code, the dependency set, the environment, the
device, or the command produces a NEW run identity. It is never an in-place
continuation, and it is never a silent restart that looks like a resume.

## The canonical commands

The canonical V2 smoke command is exactly:

```powershell
& .\.venv\Scripts\python.exe pipeline_v2/run.py run --config pipeline_v2/configs/experiments/smoke.yaml
```

The full acceptance sequence, run one stage at a time, each gated on the
previous stage's evidence being PASS:

```powershell
& .\.venv\Scripts\python.exe pipeline_v2/run.py preflight --gpu --config pipeline_v2/configs/experiments/smoke.yaml
& .\.venv\Scripts\python.exe pipeline_v2/run.py run --config pipeline_v2/configs/experiments/smoke.yaml
& .\.venv\Scripts\python.exe pipeline_v2/run.py preflight --gpu --config pipeline_v2/configs/experiments/pilot.yaml
& .\.venv\Scripts\python.exe pipeline_v2/run.py run --config pipeline_v2/configs/experiments/pilot.yaml
& .\.venv\Scripts\python.exe pipeline_v2/run.py preflight --gpu --config pipeline_v2/configs/experiments/target.yaml
& .\.venv\Scripts\python.exe pipeline_v2/run.py run --config pipeline_v2/configs/experiments/target.yaml
```

## Required read-back paths

After a stage run, every one of these must exist and be readable under
`runs/v2/<namespace>/<run_id>/`:

- `external_gates/reports/COMPLETE.json`
- `publish_final/COMPLETE.json`
- `resolve_final_identity/manifests/run_identity.json`
- `publish_final/manifests/final_publication.json`
- `train/logs/log-manifest.json` (the final log index)
- `train/checkpoints/step-XXXXXXXX/manifest.json` (the selected checkpoint manifest)

A missing artifact is `BLOCKED` or `FAIL`. A run without `COMPLETE.json` is
not a successful run, and no marker is ever written to imply otherwise.

For every promotion, record the predecessor run ID, the checkpoint, the data
manifest, the evaluator set, the gate result, and the hashes. A promotion whose
predecessor is not accepted may not be registered.

## The Release 1 acceptance checklist

Each criterion is verified by a repository-relative artifact, never from memory
and never from a console line.

| Criterion | Verified by |
|---|---|
| an invalid config fails | `validate` returns FAIL with a reason |
| data is pinned and fresh | `manifests/data.json` and its provenance |
| tokenizer and shard identity | `manifests/tokenizer.json`, `manifests/tokenized_data.json` |
| no train/validation overlap | dataset validation report |
| resume rejects changed code, dependency, or launch profile | `assert_exact_resume` refusal |
| exact cursor restore | `CheckPointState` round trip and the resumed cursor |
| a busy or uncertain GPU blocks | lease `BLOCKED` with a reason |
| bootstrap handoff is atomic | create-if-absent CAS, no lost update |
| metrics and health are gap-free | `verify_stream` on every stream |
| NaN or Inf is rejected | health classification, report FAIL |
| a missing metric or report is rejected | `require_complete_suite` |
| an evaluator mismatch is rejected | `require_complete_suite` |
| final model, tokenizer, manifests, reports read back | `read_back_final_publication` |
| S/P/T promotion references are recorded | `features.promotion` plus the registry |
| V1 files and artifacts are unchanged | the frozen V1 inventory hashes |
| experiments are created in YAML only | no Python edit required for an experiment |

## Current status of this release

The CPU lanes are complete and passing. The GPU acceptance sequence HAS been
executed against the published runs under `runs/v2/core/`:

- `runs/v2/core/gpu_smoke/status.json` - PASS: the full 13-stage sequence
  completed on the GPU, ending in `publish_final`.
- `runs/v2/core/nano_codesechnet/status.json` - PASS: the full sequence,
  including `train` and the external gates.
- `runs/v2/core/gpu_smoke_post_remediation/status.json` - PASS: the full
  sequence re-run AFTER the remediation landed, on the remediated loop. The
  train record (`train/checkpoints/COMPLETE.json`) proves the schedule was
  applied, not just declared: learning_rate 0.003 decrease to a
  `final_learning_rate` of 0.0 under the declared cosine scheduler,
  `optimizer: adamw`, `selected_by: declared-policy`, held-out blocks kept
  out of the train stream. This is the run any NEW claim should cite.
- `runs/v2/core/micro_full/status.json` - PASS: the full sequence.
- `runs/v2/core/gpu_smoke-failed-fp16-step2/status.json` - FAIL at `train`:
  a non-finite loss at step 2, the pure-fp16 optimizer event the fp32
  master-weights rule was adopted from; kept as failure evidence.
- `runs/v2/core/smoke/status.json` - BLOCKED at `prepare` (no declared
  source); kept as the fail-closed refusal evidence.
- `runs/v2/core/pilot/status.json` - BLOCKED at `train`: the run ran out of
  memory at step 317; the last valid checkpoint is preserved and batch,
  context, optimizer, precision, and command are unchanged.
- `runs/v2/core/target/status.json` - BLOCKED at `train`: the run ran out of
  memory at step 2; same OOM policy, same preserved checkpoint.
- `runs/v2/core/pilot_ctx512/status.json` - FAIL at `train`: the ctx-512
  OOM-shrink re-entry (`configs/experiments/pilot_r3.yaml`) hit a
  checkpoint-publish collision (`FileExistsError` on
  `train/checkpoints/step-00000200/state/model.pt`); kept as failure evidence.

Every run criterion a published PASS has settled is read from those
status.json artifacts, never asserted from memory. The pilot and target
promotion sequences HAVE been attempted: both are published as BLOCKED at
`train` (OOM), so the criteria they were meant to settle remain
`NEEDS_USER_DECISION` until a pre-registered run completes the sequence.

Historical note: the previous text of this section stated the GPU lane was
unrun; it was stale against the published runs above, and the contract test
that pinned that wording was removed in the same change. The published runs
other than `gpu_smoke_post_remediation` were trained under the
pre-remediation loop (static LR, no gradient accumulation): read them as
evidence the SEQUENCE ran, not that the current schedule semantics were
exercised.

- 2026-09-26 pipeline-validation small run: the full 13-stage graph plus train completed end to end on a governed small config (runs/v2/core/pilot_small/status.json), checkpoints at steps 10 and 20 published with COMPLETE markers, val_loss/val_perplexity recorded in runs/v2/core/pilot_small/train/checkpoints/scores.json, and the internal evaluation gate refused the undertrained model at its declared threshold (val_loss 7.46259 > 7.0). PASS as pipeline-mechanics evidence; FAIL as a model gate, exactly as designed. Config: pipeline_v2/configs/experiments/pilot_small.yaml.
- 2026-09-26 memory changes: clip_grad_norm_ runs non-batched (foreach=False), the held-out-loss autocast disables its fp16 weight-cast cache (cache_enabled=False), and an interrupted checkpoint publish now reclaims its own debris (training/checkpoint.py). Rules/gpu.md OOM shrink declared and recorded by configs/experiments/pilot_r3.yaml: context 1024 -> 512 (model, tokenizer, and packing in lockstep), batch / optimizer / precision / checkpoint policy unchanged. The ctx-512 re-entry is published as `runs/v2/core/pilot_ctx512` (FAIL at `train`, a checkpoint-publish collision); no completed ctx-512 run exists, so no VRAM figure is asserted here.

- 2026-09-26 checkpoint publish now reclaims an interrupted destination missing its completion marker and never touches a complete one (training/checkpoint.py; unit tests in tests/unit/test_resume.py).

## Operator-friction remediation (2026-09-26) — phase entries

Each entry is anchored in a test-output artifact under `scratch/` (validation
consoles are not durable evidence) and the unit names run:

### Phase 1 - VRAM observability
- `vram_probe` and `oom_accounting` events declared on the monitor stream
  (slm_train/monitoring/bus.py, slm_train/resources/gpu_probe.py). Unit
  coverage: test_monitoring.py declared-event, refused-name, and flush-on-del
  tests; handlers emit at every completed step and at the micro-OOM site.
- Evidence: scratch/gate_test14.out (33 passed: test_handlers.py full file at
  the time), scratch/gate_test18.out, and the pyc-recovery oracle
  scratch/dis_dump.txt (recovered-set disassembly, backed up first at
  scratch/handlers_recovery.pyc).
- Known limitation vs the recovered bytecode oracle: the micro-batch
  accumulation groups, the cooperative STOP-file stop, the declared-policy
  selection label, and the inter-step evaluation_completed emit from the
  pre-truncation handlers.py are NOT re-implemented (behavior attested only by
  the disassembly, not by any live test). Recorded as known fidelity gaps.

### Phase 2 - stop errors before the run wastes an hour
- Disk gate (slm_train/resources/disk_gate.py; observation only, never deletes)
  applied in stage_train (after run_started) and stage_model_validation; unit
  coverage test_disk_gate_refuses_a_run_root_with_no_room, test_disk_gate_passes_when_space_exceeds_the_requirement, test_required_free_bytes_scales_with_kept_checkpoints, and
  test_model_validation_gate_refuses_an_undersized_volume
  (evidence: scratch/gate_test19.out - 2 passed).
- Resume summary artifact `train/logs/resume_summary.json`
  (resumed_from_step, source_checkpoint, identity_keys_compared,
  retrain_reason; a retry overwrites it) plus the train stage record field
  `resumed_from_step` in `train/checkpoints/COMPLETE.json`. Coverage:
  test_a_deleted_train_record_resumes_instead_of_retraining
  (evidence: scratch/gate_test18.out - 1 passed).
- Loud prepared-data invalidation: stage_train refuses a run whose tokenized
  data context disagrees with packing.sequence_length
  (test_prepared_context_mismatch_is_an_honest_fail, evidence:
  scratch/gate_test16.out).
- Real scheduler state: optimizer and scheduler built through
  slm_train/training/schedule factories so a resumed run reconstructs the
  declared LR curve; the scheduler state carried in every checkpoint is the
  scheduler's own state_dict, not a stub.

### Phase 3 - ergonomics
- One-key context alias: `context:` on ConfigDocument validated against
  model/tokenizer/packing, with materialize_context filling the sections from
  the alias (slm_train/config/models.py). Coverage:
  test_config_context_alias_must_agree_with_the_three_sections,
  test_materialize_context_fills_the_sections_from_the_alias (evidence:
  scratch/full_suite_p2.out).
- Event-name safety net: emit_or_quarantine published under
  rejected_event:... on the monitor stream instead of raising (bus.py);
  test_emit_or_quarantine_publishes_a_refused_name_instead_of_raising and its
  siblings in test_monitoring.py.
