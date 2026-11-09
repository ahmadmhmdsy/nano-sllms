# pipeline_v2 Remediation Plan

Status: **active** — derived from the 2026-09-28 code review.
Owner: sub-agent tasks, one milestone at a time.

Each milestone is a self-contained unit of work a single sub-agent can complete.
Milestones are ordered by dependency: later milestones assume earlier ones landed.

---

## Milestone 1: Training Loop Correctness (CRITICAL)

**Goal:** The training loop does what the config declares. No false certifications.

### Task 1.1 — Call `scheduler.step()` in the training loop

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `build_scheduler` is called at line 1522 and the scheduler is serialized
into checkpoints, but `scheduler.step()` is never called. The learning rate is flat
for the entire run. The checkpoint fallback `{"kind": "cosine-constant-lr"}` is a lie.

**Fix:**
1. After `optimizer.step()` at line 1765, add `scheduler.step()`.
2. Remove the misleading fallback in `_real_payloads` (line 2087): the scheduler
   always exists, so the `if scheduler is not None` branch is dead. Replace with
   `dump(scheduler.state_dict())`.
3. Add a unit test that builds a scheduler, steps it N times, and asserts the LR
   follows the declared curve.

**Acceptance criteria:** `scheduler.step()` is called exactly once per optimizer
step. A test proves the LR changes over a run. The checkpoint scheduler state is
a real `LambdaLR` state dict, not a placeholder.

---

### Task 1.2 — Implement gradient accumulation

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `train.gradient_accumulation` is declared in the config (default 8) but
never read. The training loop does one `backward()` + `optimizer.step()` per batch.
The effective batch is `batch_size` (1), not `batch_size * accum` (8).

**Fix:**
1. Read `accum = int(document.train.gradient_accumulation)` at the top of
   `stage_train`.
2. In the training loop, divide the loss by `accum` before `backward()`.
3. Only call `optimizer.step()` and `scheduler.step()` every `accum` batches.
4. Track a micro-batch counter; reset it after each optimizer step.
5. Handle the final partial accumulation group (if the number of batches is not
   divisible by `accum`, step with the partial group).
6. Add a test that proves the effective batch size matches the config.

**Acceptance criteria:** A config with `batch_size: 1` and `gradient_accumulation: 8`
produces the same parameter updates as a config with `batch_size: 8` and
`gradient_accumulation: 1` (given the same data and seed).

---

### Task 1.3 — Implement cooperative stop

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `cli.py:361` writes a STOP flag file. The README promises the train
stage honours it. But no code in `handlers.py` reads the flag. A long run cannot
be halted without killing the process.

**Fix:**
1. At the start of each optimizer-step iteration, check for the STOP flag file
   at `stage_root / "STOP"`.
2. If the flag exists, set `stop = True` and break out of the loop.
3. The existing checkpoint at the current step is preserved (the loop already
   publishes at `checkpoint_every_steps` boundaries).
4. Add a test that writes the flag mid-run and asserts the loop exits cleanly
   with a valid checkpoint.

**Acceptance criteria:** A run with a STOP flag present exits at the next
optimizer-step boundary with a valid checkpoint. The run status is PASS (or
SKIPPED), not FAIL.

---

### Task 1.4 — Add GradScaler for fp16 training

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** No GradScaler is used. The scaler checkpoint is
`{"kind": "none", "reason": "fp32 master weights on CPU"}`. fp16 gradients can
underflow without loss scaling.

**Fix:**
1. When `autocast_enabled` is true, create a `torch.cuda.amp.GradScaler`.
2. Wrap the forward pass in `scaler.scale(loss).backward()`.
3. Call `scaler.step(optimizer)` instead of `optimizer.step()`.
4. Call `scaler.update()` after each step.
5. Serialize the scaler state dict in `_real_payloads`.
6. Restore the scaler state in `_resume_from`.

**Acceptance criteria:** A fp16 run produces finite gradients at every step. The
scaler state is serialized and restored on resume.

---

## Milestone 2: Data Pipeline Correctness (CRITICAL)

**Goal:** The data pipeline applies what the config declares. No false certifications.

### Task 2.1 — Pass declared filters to `prepare_generation`

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `stage_prepare` hardcodes `"min_chars": 0` at line 625. The declared
`data.filters` are recorded in the manifest as `filters={"declared": ...}` but
never applied. A config declaring `filters: ["min_chars:100"]` certifies filtering
that never happened.

**Fix:**
1. Parse `data.filters` in `stage_prepare` to extract `min_chars` and `max_chars`.
2. Pass them to `prepare_generation` via the options dict.
3. Remove the hardcoded `"min_chars": 0`.
4. Add a test that declares a filter and asserts the manifest records the applied
   filter value, not just the declared one.

**Acceptance criteria:** A config with `filters: ["min_chars:100"]` produces a
manifest where the applied `min_chars` is 100, and short documents are rejected.

---

### Task 2.2 — Refuse or implement multi-source preparation

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** Only `data.sources[0]` is read. A two-source config publishes a
manifest claiming two pinned sources while only one was read.

**Fix (choose one):**
- **Option A (refuse):** Add a validator in `config/models.py` that rejects
  configs with more than one source. Update the error message to say "R1
  preparation consumes exactly one declared source".
- **Option B (implement):** Implement multi-source preparation in `stage_prepare`:
  read all sources, merge the documents, deduplicate across sources, and assign
  splits across the merged set.

**Acceptance criteria (Option A):** A config with two sources fails validation
with a clear error message.

**Acceptance criteria (Option B):** A config with two sources produces a manifest
that correctly records both sources, and the training data contains documents
from both.

---

### Task 2.3 — Refuse or implement Hugging Face sources

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `kind: huggingface` is allowed in the config. `HFSourceReader` exists
but is never used. `_source_reader` always constructs `LocalSourceReader`.

**Fix (choose one):**
- **Option A (refuse):** Add a validator that rejects `kind: huggingface` in R1.
- **Option B (implement):** Add a branch in `_source_reader` that constructs
  `HFSourceReader` when `declaration.kind == "huggingface"`.

**Acceptance criteria (Option A):** A config with `kind: huggingface` fails
validation with a clear error message.

**Acceptance criteria (Option B):** A config with `kind: huggingface` successfully
reads from the declared HF repo.

---

### Task 2.4 — Refuse or implement held-out source roles

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `role: test` and `role: external-gate` are allowed in the config.
`assign_split` can only return `"train"` or `"validation"`. A corpus declared as
held-out evaluation data is trained on.

**Fix (choose one):**
- **Option A (refuse):** Add a validator that rejects `role: test` and
  `role: external-gate` in R1.
- **Option B (implement):** Add a third branch in `assign_split` that returns
  `"held-out"` for these roles, and ensure the training loop skips held-out
  documents.

**Acceptance criteria (Option A):** A config with `role: test` fails validation
with a clear error message.

**Acceptance criteria (Option B):** A config with `role: test` produces a manifest
that correctly records the held-out split, and the training data excludes those
documents.

---

## Milestone 3: Config and Migration Correctness (CRITICAL)

**Goal:** The config validates what it declares. The migration command produces
a valid config.

### Task 3.1 — Fix `migrate-v1` to produce a valid config

**File:** `pipeline_v2/slm_train/adapters/legacy_v1.py`

**Problem:** The command reports PASS but the output config always fails
validation. Only 2 of ~50 fields are migrated. The hand-rolled YAML parser is
broken. `DIRECT_FIELDS` maps to V2 paths that don't exist. The output emits
`provenance:` (forbidden) and omits `schema_version: 2` (required).

**Fix:**
1. Replace `_read_mapping` with a real YAML parser (PyYAML is already a dependency).
2. Update `DIRECT_FIELDS` to map to real V2 paths.
3. Add `schema_version: 2` to the output.
4. Remove the `provenance:` key from the output.
5. Add a test that runs `migrate-v1` on a real V1 config and asserts the output
   passes `validate`.

**Acceptance criteria:** `migrate-v1 --input <v1.yaml> --output <v2.yaml>` produces
a config that passes `validate --config <v2.yaml>`.

---

### Task 3.2 — Fix `evaluation_cadence` to apply independently

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `evaluation_cadence` is declared separately from `checkpoint_every_steps`
but held-out loss is only measured inside the `checkpoint_every_steps` branch.

**Fix:**
1. Add a separate check: `if final_step % evaluation_cadence == 0` for evaluation.
2. Keep the checkpoint publishing at `checkpoint_every_steps`.
3. Add a test that proves evaluation happens at the declared cadence, not just at
   checkpoint boundaries.

**Acceptance criteria:** A config with `evaluation_cadence: 100` and
`checkpoint_every_steps: 500` produces evaluation metrics at steps 100, 200, 300,
400, 500, etc.

---

### Task 3.3 — Fix `worker_count` to be read or removed

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `worker_count` is declared in the config but never read. No DataLoader
is built.

**Fix (choose one):**
- **Option A (remove):** Remove `worker_count` from `RuntimeConfig` and all configs.
- **Option B (implement):** Build a DataLoader with the declared worker count.

**Acceptance criteria (Option A):** The config no longer declares `worker_count`.

**Acceptance criteria (Option B):** A config with `worker_count: 4` uses a DataLoader
with 4 workers.

---

### Task 3.4 — Fix checkpoint selection to honor `best` policy

**File:** `pipeline_v2/slm_train/training/selection.py`

**Problem:** `select_checkpoint` picks the highest-numbered verified checkpoint,
ignoring the `checkpoint_policy: "best"` declaration.

**Fix:**
1. Read the `checkpoint_policy` from the train stage's published config.
2. When policy is `"best"`, select the checkpoint with the lowest validation loss
   (from `scores.json`).
3. When policy is `"last"`, select the highest-numbered checkpoint (current
   behavior).
4. When policy is `"complete_only"`, select the final step's checkpoint.

**Acceptance criteria:** A config with `checkpoint_policy: "best"` selects the
checkpoint with the lowest validation loss, not the highest-numbered one.

---

## Milestone 4: Observability and Operations (MEDIUM)

**Goal:** An operator can monitor and diagnose a run.

### Task 4.1 — Add learning rate logging

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** The training loop doesn't log the current learning rate. This makes it
impossible to verify the scheduler is working.

**Fix:**
1. After each `optimizer.step()`, read `optimizer.param_groups[0]["lr"]`.
2. Emit it in the `step_completed` event payload.
3. Add a test that proves the LR changes over a run.

**Acceptance criteria:** The `step_completed` event includes the current LR.

---

### Task 4.2 — Add training progress logging

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** The training loop doesn't log progress (step X of Y). This makes it
hard to monitor long runs.

**Fix:**
1. Every N steps (e.g., 10), emit a `step_completed` event with
   `{"step": step, "max_steps": max_steps, "progress": step / max_steps}`.
2. Add a test that proves progress is logged.

**Acceptance criteria:** The `step_completed` event includes progress information.

---

### Task 4.3 — Add TensorBoard/tfevents support

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** No tfevents are ever written. Curves exist only in JSONL.

**Fix:**
1. Add a `SummaryWriter` that writes to `train/logs/tensorboard/`.
2. Log loss, LR, and gradient norm at each step.
3. Add a test that proves tfevents are written.

**Acceptance criteria:** A run produces tfevents files that TensorBoard can read.

---

### Task 4.4 — Add multi-run status dashboard

**File:** `pipeline_v2/slm_train/cli.py`

**Problem:** `status --run` reads one published status. No curve, no cross-run
comparison.

**Fix:**
1. Add a `status --runs` command that reads all runs under a namespace.
2. For each run, read the loss curve from `train/checkpoints/scores.json`.
3. Print a summary table with run name, status, best val loss, and final val loss.
4. Add a test that proves the command works.

**Acceptance criteria:** `status --runs --namespace <ns>` prints a summary table
for all runs in the namespace.

---

## Milestone 5: Dead Code and Maintenance (MEDIUM)

**Goal:** Remove dead code and reduce maintenance burden.

### Task 5.1 — Remove or implement HFCausalLMTrainer

**File:** `pipeline_v2/slm_train/training/hf_causal_lm.py`

**Problem:** `HFCausalLMTrainer` implements only `prepare()`. All other methods
raise `PipelineError`. The module is never used by any stage.

**Fix (choose one):**
- **Option A (remove):** Delete the module and its tests.
- **Option B (implement):** Implement `fit`, `evaluate`, `checkpoint`, `resume`,
  and `finalize` using the hand-rolled training loop as a reference.

**Acceptance criteria (Option A):** The module and its tests are deleted.

**Acceptance criteria (Option B):** The module is used by `stage_train` and all
methods work.

---

### Task 5.2 — Remove or implement TrainerBackend

**File:** `pipeline_v2/slm_train/training/backend.py`

**Problem:** `TrainerBackend` is declared but never implemented. The real loop is
the hand-rolled one in `handlers.py`.

**Fix (choose one):**
- **Option A (remove):** Delete the module and its references.
- **Option B (implement):** Refactor the hand-rolled loop into a `TrainerBackend`
  implementation.

**Acceptance criteria (Option A):** The module and its references are deleted.

**Acceptance criteria (Option B):** `stage_train` uses `TrainerBackend` and the
training loop is unchanged.

---

### Task 5.3 — Fix KD teacher cache to be per-run

**File:** `pipeline_v2/slm_train/orchestration/handlers.py`

**Problem:** `_KD_TEACHER_CACHE` is a module-level dict. This is a global mutable
state that could cause issues in concurrent scenarios.

**Fix:**
1. Move the cache into the `stage_train` function as a local variable.
2. Pass it to `_kd_teacher_kl` as a parameter.
3. Add a test that proves the cache is per-run.

**Acceptance criteria:** The cache is a local variable, not a module-level dict.

---

## Dependency Graph

```
Milestone 1 (Training Loop Correctness)
  ├── Task 1.1 (scheduler.step)
  ├── Task 1.2 (gradient accumulation)
  ├── Task 1.3 (cooperative stop)
  └── Task 1.4 (GradScaler)

Milestone 2 (Data Pipeline Correctness)
  ├── Task 2.1 (declared filters)
  ├── Task 2.2 (multi-source)
  ├── Task 2.3 (HF sources)
  └── Task 2.4 (held-out roles)

Milestone 3 (Config and Migration Correctness)
  ├── Task 3.1 (migrate-v1)
  ├── Task 3.2 (evaluation_cadence)
  ├── Task 3.3 (worker_count)
  └── Task 3.4 (checkpoint selection)

Milestone 4 (Observability and Operations)
  ├── Task 4.1 (LR logging)
  ├── Task 4.2 (progress logging)
  ├── Task 4.3 (TensorBoard)
  └── Task 4.4 (multi-run status)

Milestone 5 (Dead Code and Maintenance)
  ├── Task 5.1 (HFCausalLMTrainer)
  ├── Task 5.2 (TrainerBackend)
  └── Task 5.3 (KD teacher cache)
```

**Recommended order:** Milestone 1 → Milestone 2 → Milestone 3 → Milestone 4 → Milestone 5

**Parallelizable:** Tasks within a milestone are independent and can be assigned
to different sub-agents. Milestones 1 and 2 can run in parallel. Milestones 3, 4,
and 5 can run in parallel after Milestone 1 lands.

---

## Acceptance Criteria for the Entire Plan

1. Every config field that V2 declares is either applied or removed.
2. Every manifest that V2 publishes certifies what actually happened.
3. Every command that reports PASS produces a valid artifact.
4. The training loop does what the config declares: scheduler, gradient
   accumulation, warmup, gradient clipping, and checkpoint selection.
5. The data pipeline does what the config declares: filters, normalization,
   multi-source, and held-out roles.
6. An operator can monitor a run: LR, progress, loss curves, and cross-run
   comparison.
7. No dead code remains: every module is either used or deleted.
