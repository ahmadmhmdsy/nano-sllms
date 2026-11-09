# pipeline_v2 Phase 3: Missing Features Plan

Status: **active** — extends the 2026-09-28 remediation plan.
Owner: sub-agent tasks, one milestone at a time.

These are MISSING FEATURES, not false certifications. They are MEDIUM priority
because they block evaluation or observability but do not produce false evidence.

---

## Milestone 6: Dry-Run and Observability (MEDIUM)

**Goal:** An operator can preview what a run will do before launching it.

### Task 6.1 — Add dry-run to the plan command

**File:** pipeline_v2/slm_train/cli.py

**Problem:** plan prints a static stage list, not resolved commands. An operator
cannot preview what a run will do before launching it.

**Fix:**
1. Add a --dry-run flag to the plan command
2. When set, print the resolved config (all sections) alongside the stage list
3. Print the resolved command that run would execute
4. Add a test that proves the dry-run output contains the resolved config

**Acceptance criteria:** plan --config <cfg> --dry-run prints the resolved config
and the command that run would execute.

---

### Task 6.2 — Add hardware state report

**File:** pipeline_v2/slm_train/cli.py

**Problem:** doctor is an env/import check only. An operator cannot diagnose
hardware issues.

**Fix:**
1. Add a hardware-state command that reports CPU, RAM, disk, GPU, VRAM
2. Use existing resources/ modules where possible
3. Add a test that proves the command works

**Acceptance criteria:** hardware-state prints a report with CPU, RAM, disk, GPU,
and VRAM information.

---

### Task 6.3 — Add resource profiling over time

**File:** pipeline_v2/slm_train/orchestration/handlers.py

**Problem:** Only a single peak_reserved_bytes and wall-clock at end of stage.
An operator cannot diagnose memory leaks or performance degradation.

**Fix:**
1. Add a periodic resource probe that logs CPU, RAM, GPU, VRAM every N seconds
2. Write the timeseries to train/logs/resource_timeseries.jsonl
3. Add a test that proves the timeseries is written

**Acceptance criteria:** A run produces a resource timeseries file with periodic
CPU, RAM, GPU, and VRAM measurements.

---

## Milestone 7: Evaluation Features (MEDIUM)

**Goal:** An operator can evaluate the model with sampling-based and instruction-style
evaluations.

### Task 7.1 — Add generation sampling parameters

**File:** pipeline_v2/slm_train/evaluation/generation.py

**Problem:** Greedy only; no temperature, top_k, top_p. Blocks any sampling-based
evaluator.

**Fix:**
1. Add temperature, top_k, top_p parameters to the generation function
2. Pass them to model.generate()
3. Add a test that proves sampling works

**Acceptance criteria:** A generation suite can sample with temperature=0.8,
top_k=50, top_p=0.95.

---

### Task 7.2 — Add instruction-style eval

**File:** pipeline_v2/slm_train/evaluation/internal.py

**Problem:** Generation and syntax smoke only. Cannot evaluate instruction-following
capability.

**Fix:**
1. Add an instruction eval suite that measures instruction-following accuracy
2. Use a set of instruction-following prompts with known correct answers
3. Add a test that proves the suite works

**Acceptance criteria:** An instruction eval suite produces a pass/fail metric
for instruction-following.

---

### Task 7.3 — Add context extrapolation probes

**File:** pipeline_v2/slm_train/config/models.py, pipeline_v2/slm_train/models/causal_lm.py

**Problem:** No rope scaling, no window, no sink tokens. Cannot measure long-context
claims.

**Fix:**
1. Add rope_theta, rope_scaling, head_dim fields to ModelConfig
2. Pass them to the model builder
3. Add a context extrapolation eval suite
4. Add a test that proves the probes work

**Acceptance criteria:** A config can declare rope_scaling and the model uses it.
A context extrapolation eval suite produces metrics at different context lengths.

---

### Task 7.4 — Add YaRN context extension

**File:** pipeline_v2/slm_train/config/models.py, pipeline_v2/slm_train/models/causal_lm.py

**Problem:** No rope_theta, no rope_scaling, no head_dim in ModelConfig. Cannot extend
context beyond the trained length.

**Fix:**
1. Add rope_theta, rope_scaling, head_dim fields to ModelConfig
2. Pass them to the model builder
3. Add a test that proves YaRN works

**Acceptance criteria:** A config with rope_scaling: yarn produces a model that
can handle context beyond the trained length.

---

## Dependency Graph

Milestone 6 (Observability)
  - Task 6.1 (dry-run)
  - Task 6.2 (hardware state)
  - Task 6.3 (resource profiling)

Milestone 7 (Evaluation)
  - Task 7.1 (generation sampling)
  - Task 7.2 (instruction eval)
  - Task 7.3 (context extrapolation)
  - Task 7.4 (YaRN extension)

**Recommended order:** Milestone 6 then Milestone 7

**Parallelizable:** Tasks within a milestone are independent. Milestones 6 and 7
can run in parallel.

---

## Acceptance Criteria for the Entire Plan

1. An operator can preview a run with plan --dry-run
2. An operator can diagnose hardware issues with hardware-state
3. A run produces a resource timeseries for diagnosing memory leaks
4. An operator can evaluate the model with sampling-based generation
5. An operator can evaluate instruction-following capability
6. An operator can measure long-context claims with context extrapolation probes
7. A config can declare YaRN context extension and the model uses it
