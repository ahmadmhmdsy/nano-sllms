
# Agent operations — launch gates, read-back, resume, and safety

Operational procedures shared by every component line. Per-component hazards
live in `rules/components/*.md`; engineering-wide policy lives in
`rules/`; record fields and vocabularies live in
`docs/agent/SCHEMA.md`.

## Launch gate

Every GPU-bound, network-bound, or long-running launch passes a gate before
it starts:

1. The task scope is identified and the relevant `rules/` file has been
   read for this task class.
2. The exact command and config are recorded in the task record before
   launch.
3. A proven preflight gate is run for unattended long launches (config
   validation: duplicate keys, static checks, --help smoke, data state, GPU
   smoke) using the repo's documented preflight script under laya/scripts/.
4. Free-VRAM, disk headroom, and the process list are re-measured right now
   (never remembered); the measured numbers are what the launch margin is
   sized against, per `rules/gpu.md`.
5. The launch is one exact, reproducible, venv-only command with an explicit
   config path. A gate that cannot be satisfied produces a BLOCKED launch —
   never a silent launch-anyway.

## Read-back

Launch output is not trusted on faith; the artifacts are read back:

- Training: after the run settles, re-open the TensorBoard event files and
  confirm what the logs claim (loss direction, steps per second, eval
  metrics).
- Data-era and eval runs: re-parse the written metrics JSON or output files
  and cross-check them against the registered gate values.
- Checkpoints: verify the checkpoint directory contents (checkpoint step,
  status files) rather than assuming the trainer saved what it printed.
- Every read-back result is recorded with its own timestamp and the exact
  artifact path, per `rules/evidence.md` (no remembered numbers).

## Smoke validity

- A smoke run's verdict is valid only for the exact config, data slice, and
  code path that ran; the claim never transfers to a different phase,
  config, or dataset size.
- A smoke run must demonstrably advance the update path (loss moved, a
  shard or checkpoint landed, metrics were written). An exit code alone is
  not a verdict.
- A smoke verdict stays valid only while the boundary it exercised is
  unchanged: a new config, data slice, or trainer edit invalidates it.

## Safe resume

- Training jobs auto-resume. After any crash, kill, or timeout, re-run the
  exact original command with zero additional flags; never add a resume or
  skip flag on a retry.
- Before resuming, verify that the checkpoint directory exists and matches
  the crashed run (same config, same phase). If this verification cannot be
  done, the launch is BLOCKED, not a guess.
- Never resume a foreign machine's checkpoint on this machine without
  explicit user direction, and never resume a run whose checkpoint artifacts
  cannot be located (see `rules/context.md`).

## GPU operations

- GPU policy lives in `rules/gpu.md`.
- The operational layer: measure before launching, cite the measured
  numbers, never claim GPU headroom from memory, and confirm the prior
  job's read-back is available before the next validated launch.

## Hardware state probe

- 'scripts/hardware_state.py' is read-only, selective, and local. It reports
  CPU, memory, disk, network, GPU, dedicated VRAM, shared GPU memory, and
  opt-in per-process GPU rows as JSON; see `docs/agent/HARDWARE_STATE.md`.
- Run it only inside a hardware window the user has explicitly opened, and
  keep the label SKIPPED when that window was not opened.
- Its output is a local measurement, not global adoption evidence. It does
  not satisfy the global process, network, and adoption audit, which stays
  BLOCKED.
- It never writes state, never stops a process, and never contacts the
  network, so it needs no launch gate of its own; a per-process row never
  authorizes stopping anything.

## Data safety in operations

- Datasets, checkpoints, and logs are preserved per `rules/data.md`
  (preserve source data, bounded chunking, regeneration recorded, atomic
  and mmap-safe writes, no user-data deletion).
- Operations never bulk-delete anything: cleanups happen only under the
  data policy's approved boundary, and every deletion is reported first.
