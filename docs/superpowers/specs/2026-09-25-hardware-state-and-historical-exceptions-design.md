# Hardware State and Historical Exception Registry Design

**Status:** Approved design  
**Date:** 2026-09-25  
**Scope:** nano_SLMs repository

## 1. Goal

Add two repository capabilities:

1. Record the four known context-linter findings as explicitly accepted historical exceptions without rewriting the historical documents.
2. Provide a read-only, on-demand hardware-state command that lets an agent request current CPU, memory, disk, network, GPU, dedicated-VRAM, shared-GPU-memory, and per-process information only when it needs that information.

The hardware command must keep dedicated VRAM and shared GPU memory as separate measurements. It must never present them as one interchangeable number.

## 2. Approved decisions

- The historical findings are recorded in a dedicated registry: `state/context_exceptions.yaml`.
- The historical documents remain unchanged.
- Hardware inspection is implemented as a selective read-only CLI: `scripts/hardware_state.py`.
- No static live hardware snapshot is generated automatically.
- The agent chooses the requested sections; detailed process information is not returned by default.
- All hardware measurements are live-only and are never copied into stable policy files.
- A real hardware probe is not run during automated implementation validation; tests use mocked command results.
- No training, inference, server, network request, process termination, or continuous monitoring is part of the command.

## 3. Non-goals

This design does not:

- edit `TASKS.md` or `research/EXPERIMENTS.md`;
- suppress arbitrary linter findings;
- claim that an exception makes a historical document correct;
- provide a global process, ownership, or network audit;
- provide packet capture or traffic interception;
- combine dedicated VRAM and shared GPU memory;
- guess missing measurements;
- install new dependencies;
- launch a GPU workload;
- write `CURRENT_STATE.md`, `state/current.json`, or `state/environment.json`;
- continuously poll hardware in the background;
- kill or modify any process.

## 4. Historical exception registry

### 4.1 File and schema

Add `state/context_exceptions.yaml` with schema version 1. Each record contains:

```yaml
id: context-exception/tasks-60
status: accepted_historical
path: TASKS.md
line: 60
kind: malformed_table_row
reason: Legacy table formatting is preserved as historical evidence.
accepted_at: "2026-09-25"
decision_source: direct-user-instruction
evidence:
  - docs/agent/reports/context-validation-2026-09-24.md
```

For a broken-link record, add the exact relative `target` field. Every record must use concrete values; placeholders are not valid registry data.

The registry must not contain absolute paths, UNC paths, credentials, command output, or unbounded exception text.

### 4.2 Exact-match rule

An incoming linter finding is accepted only when its path, line, kind, and link target (where applicable) exactly match one registry record.

The following must not be suppressed:

- a new finding at another line;
- a different finding type at the same line;
- a changed broken-link target;
- a shifted line;
- a finding with a different path;
- a malformed or duplicate registry record.

A changed or shifted finding fails and requires a new review. This makes acceptance narrow and auditable.

### 4.3 Linter behavior

In normal policy-enabled mode, the linter loads the registry before classification.

It prints every accepted item explicitly, for example:

```text
ACCEPTED HISTORICAL: TASKS.md:60 (malformed_table_row)
```

The accepted items are not silently discarded. The linter exits successfully when every remaining finding is covered by an exact accepted record.

The explicit `--no-policy` record-only mode remains separate and cannot be used to claim a clean policy result.

A missing, unreadable, malformed, duplicate, or unsafe exception registry fails closed. The linter must not continue with an unvalidated exception set.

The four initial records correspond exactly to:

- `TASKS.md:60`, malformed table row;
- `TASKS.md:196`, malformed table row;
- `research/EXPERIMENTS.md:25`, broken link to `research/_template_experiment.md`;
- `research/EXPERIMENTS.md:28`, broken link to `research/WHAT_WORKS.md`.

The validation report must distinguish the now-accepted historical linter items from the separate global-audit `BLOCKED` status. It must not describe the global audit as passed.

## 5. Hardware-state CLI

### 5.1 Invocation

The command is:

```powershell
& .\.venv\Scripts\python.exe scripts\hardware_state.py [options]
```

Default behavior is a small overview. Detailed sections are opt-in:

```powershell
& .\.venv\Scripts\python.exe scripts\hardware_state.py --section cpu --section memory
& .\.venv\Scripts\python.exe scripts\hardware_state.py --section gpu --section gpu-processes
& .\.venv\Scripts\python.exe scripts\hardware_state.py --all
```

Supported section names are:

```text
overview
cpu
memory
gpu
gpu-processes
disk
network
all
```

The default overview contains high-level availability and totals only. It does not contain detailed per-core or per-process data.

JSON is the only output format in this change. It is the authoritative machine interface; human-readable formatting is out of scope.

### 5.2 Output envelope

Every invocation emits one valid JSON document unless argument parsing itself fails:

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-25T12:34:56Z",
  "sections": {
    "gpu": {
      "status": "available",
      "data": {},
      "diagnostics": []
    }
  }
}
```

Rules:

- unavailable measurements are `null` or absent with an explicit section status;
- missing values are never represented as zero;
- diagnostics are bounded and contain no raw command text, traceback, secret, or machine-specific path;
- output is ASCII-safe when the output stream cannot encode ordinary Unicode;
- process names are bounded and sanitized;
- command lines, environment variables, and secrets are never returned.

Exit status:

- `0`: all requested sections are available;
- `2`: output was produced but one or more requested sections are unavailable or partial;
- `1`: invalid arguments or an internal output/validation failure.

The JSON document is still emitted for status `2` so an agent can safely inspect partial results.

### 5.3 CPU and system memory

The `cpu` section may report:

- CPU model and logical-core count;
- overall utilization;
- per-core utilization;
- load average where the platform exposes it;
- sampling timestamp and duration.

The `memory` section reports system RAM as:

- total bytes;
- used bytes;
- available bytes;
- percentage used.

These values are not GPU memory and are never placed in the VRAM fields.

### 5.4 GPU devices

For each NVIDIA device, the `gpu` section reports:

- device index and model;
- driver and CUDA information when available;
- utilization;
- temperature and power when available;
- total dedicated VRAM;
- used dedicated VRAM;
- available dedicated VRAM;
- shared GPU memory when the platform exposes it.

The following fields are always distinct:

```text
dedicated_vram
shared_memory
```

The command never sums these fields and never labels shared memory as VRAM.

### 5.5 Per-process GPU state

The `gpu-processes` section reports each discovered GPU process with:

- PID;
- bounded process name;
- dedicated VRAM used;
- shared GPU memory used;
- GPU utilization when the platform can attribute it;
- source and availability for each measurement.

On Windows, the implementation may join:

- `nvidia-smi --query-compute-apps=pid,process_name,used_memory`;
- the Windows `GPU Process Memory(*)\\Shared Usage` performance counter.

Records are joined by PID. If a PID appears in only one source, the missing side remains unavailable; the tool does not infer or combine values. If a process cannot be safely named, the process name is a bounded placeholder.

The command reports all discovered GPU processes so an agent can detect co-running work. It does not decide ownership, kill a process, or infer that a process is safe to stop.

### 5.6 Disk and network

The `disk` section reports local volumes with total, used, and available bytes.

The `network` section reports interfaces, addresses, and link state only. It does not capture packets, inspect traffic, or make network requests.

### 5.7 Platform behavior

The current repository runtime is Windows, but the command should degrade safely on other platforms:

- unsupported sections are marked unavailable;
- a missing optional command is not a crash;
- platform-specific shared-memory collection is optional;
- no fallback guesses a value.

No new third-party dependency is required. Standard-library facilities, `nvidia-smi`, and the existing Windows performance-counter pattern are sufficient.

## 6. Documentation integration

Add `docs/agent/HARDWARE_STATE.md` with:

- invocation examples;
- section definitions;
- units and null semantics;
- dedicated-VRAM versus shared-memory explanation;
- per-process interpretation;
- platform limitations;
- the rule that a live probe is only run in a user-opened hardware window;
- the distinction between a local measurement and a global process audit.

Add a minimal pointer to the repository router and the GPU operations documentation. Do not add volatile values to `rules/`, `ENVIRONMENT.md`, or generated state.

Update the context-validation report to record the accepted historical exception behavior and the hardware-tool validation boundary. Do not claim a live hardware run or a global process audit.

## 7. Testing and validation

Implementation follows test-driven development.

### 7.1 Exception tests

Add tests for:

- all four exact matches;
- changed line, kind, target, and path;
- duplicate and malformed registry records;
- absolute or unsafe paths;
- default policy mode;
- record-only mode;
- explicit accepted output;
- no silent suppression.

### 7.2 Hardware tests

Use injected/faked command runners and fixtures. Test:

- section selection and default overview;
- JSON envelope and schema version;
- dedicated VRAM and shared-memory separation;
- per-process PID joining;
- partial and unavailable sources;
- malformed command output;
- ASCII-safe output;
- bounded diagnostics;
- no writes;
- no process termination;
- no network operations.

### 7.3 Automated validation

Run through the repository virtual environment:

- focused exception and hardware tests;
- full `tests/agent_context` discovery;
- `py_compile` for changed scripts and tests;
- context linter;
- renderer `--check`;
- intended-path and cached whitespace checks;
- protected-path diff checks.

A live `hardware_state.py --all` probe is a separate user-opened validation step. Until that step is explicitly authorized and run, the report records it as `SKIPPED`, not `PASS`.

## 8. Worktree and rollback boundary

The pre-existing dirty worktree is preserved.

The implementation must not:

- reset, stash, clean, or bulk-delete;
- stage, commit, merge, or push;
- delete or rewrite historical documents;
- adopt a lesson or policy candidate automatically;
- modify unrelated dirty paths.

If validation fails, only files belonging to this design may be adjusted. No destructive rollback is permitted.

## 9. Acceptance criteria

The design is implemented when:

1. The four named findings are represented in the dedicated exception registry and are visibly accepted by the linter.
2. Any new, shifted, changed, or unlisted finding still fails.
3. The hardware CLI returns selective, structured, read-only measurements.
4. Dedicated VRAM and shared GPU memory remain separate fields at device and process levels.
5. Missing data is explicit and never guessed.
6. Tests prove no writes, process termination, or network activity.
7. Documentation tells an agent exactly how and when to request the information.
8. The final report distinguishes local hardware readiness from the still-unavailable global audit.
9. No live hardware result is claimed unless it was actually measured in an explicitly opened window.
10. The dirty worktree remains preserved.
