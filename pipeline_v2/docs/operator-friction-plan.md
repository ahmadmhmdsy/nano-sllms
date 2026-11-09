# pipeline_v2 - operator-friction remediation plan

Status: **PLAN (drafted 2026-09-26).** Derived from the observed session while
running the Release-1 smoke, post-remediation promotion attempts, the OOM
diagnosis, and the pipeline-validation small run
(runs/v2/core/pilot_small/status.json). Scope: operator-evidenced frictions
only; SFT/LoRA/KD/GDN/mounting remain Release 2.

Ordering principle: fix blind diagnosis first, then waste-prevention, then
ergonomics. Every numbered item carries its own test and validation label, and
nothing here changes declared numerics (batch, optimizer, precision,
checkpoint policy).

---

## Phase 1 - diagnosis you can read (P0: every OOM costs a run cycle today)

1. **Per-micro VRAM probe as a governed event.**
   Replace the throwaway env-gated file tracing that had to be written into
   orchestration/handlers.py during the 2026-09-26 OOM investigation with a
   first-class resources probe event.
   - contracts: declare an event in code via pipeline_v2/slm_train/monitoring/bus.py
     (a reserved diagnostic stream) and run_event.ts's event names; declare event shape
     (step, micro, allocated_bytes, reserved_bytes, ceiling_bytes).
   - handlers: emit at group boundaries, not every micro, so the monitor volume
     stays bounded (~125 events per 1000 steps at accumulation 8).
   - test: unit test asserting a probe event per N micros with a monotone
     step counter and nonzero ceiling (test_handlers.py).
2. **OOM refusal records demand vs. cap.**
   - slm_train/resources/gpu_probe.py: expose an attempt_accounting(device,
     requested_bytes) helper invoked at the OOM catch sites in handlers.py.
   - Payload: requested_bytes, reserved_bytes, allocated_bytes,
     ceiling_bytes, free_driver_bytes (probe_gpu), so "0 bytes free" is never
     the only line an operator sees.
   - test: classify_health + payload encoding, in test_handlers.py.
3. **flush-on-close for trace streams.**
   - bus.py: group emits already buffered; ensure every emit path that writes a
     file makes flush implicit (emit() flushes the active stream before return,
     or a ctx manager guarantees it). Removes the silent-loss class seen twice.
   - test: test_monitoring.py - emit then immediately read the file back.

## Phase 2 - stop errors before the run wastes an hour

4. **Disk-space gate before train.**
   - slm_train/resources/ or a new slim module: required_free_bytes(run_root) =
     keep_last_checkpoints x measured_checkpoint_bytes(from a smoke prefix or
     the model manifest parameter count) x 2 (current + in-flight copy).
   - Called in stage_train entry (and model_validation). Fails fast with a
     declared label and the arithmetic in the reason, not Errno 28 after step 200.
   - config: keep it non-configurable first phase; v2 exposes a knob only if a
     real workload disagrees.
   - test: test_handlers.py - a run_root on a mock small disk refuses to start.
5. **Resume summary artifact.**
   - Auto-resume already decides "resume vs. retrain" inside
     _resume_from; publish that decision as
     train/logs/resume_summary.json (resumed_from_step, source checkpoint,
     identity keys compared, keep_last effect, and the retrain reason when no
     checkpoint qualified).
   - status.json gains a "resumed_from_step" field for the train stage so
     a BAD run no longer looks like a fresh attempt.
   - test: test_resume.py - resume produces the summary; fresh run produces
     summary with resumed_from_step = null and the reason.
6. **Loud prepared-data invalidation.**
   - stage_dataset_validation writes
     dataset_validation/manifests/tokenized_data.json with a data_context
     field. stage_train reads it and, if the value differs from
     document.packing.sequence_length today, FAILs the run with a single-line
     reason (e.g. "prepared data context 1024 != configured 512; rename the
     run or regenerate").
   - This turns the silent ids_shape=(1,1024) trap into an honest FAIL.
   - test: handlers.py unit test covers the mismatch case directly.

## Phase 3 - ergonomics (smallest diffs, after 1-5 land)

7. **One-key context declaration in config.**
   - config/models.py: allow an optional top-level "context:" alias that, when
     present, validates equal to the three existing fields; a helper
     "materialize_context" sets model/tokenizer/packing if the keys are absent.
     Existing configs keep working; no schema break.
   - test: test_config.py - three-key mismatch still FAILs; alias path passes.
8. **Event-name safety net.**
   - bus.py: unknown-name EventBusError gains a sibling emit_or_quarantine
     event that writes the refused payload into monitor stream under an
     event name with a "rejected_event:" prefix, so a live diagnostic never
     silently disappears again (the original unknown-name class).
   - test: test_monitoring.py - unknown name is written to the quarantine
     stream, not raised, when called via the safe path.

## Phase ordering and validation pace

- Phase 1 items 1-3: one PR each? No - one combined change, tests inside
  pipeline_v2/tests/unit/test_handlers.py + test_monitoring.py; regression
  validation: full pipeline_v2 suite stays green (797 passed baseline).
- Phase 2 items 4-6: follow-up, same rule.
- Phase 3: never mixes with a release claim; it can land between P and T runs
  without touching training numerics.
- Every phase updates docs/acceptance.md with the artifact evidence
  (test output paths), never console-only claims.
- The 2026-09-26 temporary throwaway scaffold (scratch/*.py probe files
  scratch/vram_trace.log writes in handlers) stays out; zero regressions.

## Explicit non-goals

- No change to batch_size, gradient_accumulation, precision, optimizer, or
  checkpoint policy - all of that stays the OOM-shrink vocabulary already
  recorded in configs/experiments/pilot_r3.yaml.
- No SFT/LoRA/KD/GDN/mounting work; Release 2 boundary holds.
- No new dependency: every fix uses torch + stdlib only (contract
  test_only_preflight_gpu_probes_a_device in
  pipeline_v2/tests/contract/test_release_boundaries.py continues to pin the
  gpu_boundary).
