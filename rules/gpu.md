---
id: policy/gpu
status: active
scope: Any workload that inspects, launches, resumes, or co-schedules GPU work in this repository.
applies_when: The task touches a training job, checkpoint, evaluation, inference, VRAM probe, or SFT run.
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-10-06"
supersedes: []
evidence:
  - AGENTS.md
  - docs/agent/OPERATIONS.md
  - state/environment.json
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# GPU policy

## Hardware availability — use the GPU and CPU when they are free
Using the machine's compute is the default goal, not the exception. Three
states decide what may launch (each measured right now, never remembered):

1. **Fully free (GPU or CPU).** No protected training run active and no
   meaningful load: launch normally, subject to the launch gate in
   docs/agent/OPERATIONS.md and the measurement rules below.
2. **Partially free (small incumbent usage).** A small job may be launched
   into the spare slice, sized to stay comfortably inside the free VRAM
   (cap 60% of currently free VRAM) and the free compute fraction, so the
   incumbent is never slowed or OOM-impacted. Leftover-compute scavenging
   (below) owns the details; it applies only when the incumbent is a
   foreign, non-training workload — never against this repository's own
   active train/SFT run.
3. **Fully used.** Do not compete. End the turn, run CPU work, or leave a
   background watcher; never busy-poll and never kill or starve the
   incumbent.

CPU is always usable when it is free or partially free: bulk-parallel work
(dataloaders, data preparation, tokenization, lint/test batches) may use
multiple cores, but keep enough headroom that an active training run's
dataloader and any foreign process are not starved.

## The one-job rule
**There is exactly one protected repository GPU training job at a time.**
While a repository train or SFT run is active, no repository GPU job may
**co-run** with it: no eval, no infer, no VRAM probe, no second training,
no unrelated component launch. CPU-only work that touches nothing under
runs/ is exempt.

## No foreign-process kills
**Never kill a foreign process**, and never cause one to OOM or fail. Only a
process this agent started may be stopped, and then only as part of this
task's own launch control. Every launch decision preserves a safety margin
measured on the GPU right now, not remembered from a previous session.

## No shared-memory spill
Monitoring must confirm the job never spills into shared GPU memory. Spilling
to shared memory is treated as an OOM-equivalent event and immediately
downscales the next attempt. Enforce a hard allocator ceiling (per-process
memory fraction or an explicit cap passed to the trainer) instead of trusting
driver fallback behaviour.

## OOM response
On an own-job OOM: shrink the next attempt (smaller batch, shorter chunks),
keep the checkpoint, and re-run the exact launch command with zero
state-reset flags. Never kill a run for pace alone — sustained-load thermal
throttling is normal on this GPU and expected; compare pace against the
run's own recorded history, not against other jobs.

## Leftover-compute scavenging
Scavenging spare GPU capacity applies **only to foreign, non-training
workloads** (desktop apps, webview, webui leftovers, foreign
interpreter processes). It never applies while the incumbent is this repository's own
active `train.py`/`sft.py` run — the one-job rule above stands for those.

- **Free-VRAM cap: 60%.** Launch at most 60% of *currently free* VRAM. On
  any own-job OOM, shrink the next attempt (smaller batch / shorter
  chunks) and shrink the cap to 40%.
- **No shared-memory fallback.** Spill into shared GPU memory is treated
  as an OOM-equivalent event: rescale immediately and enforce a hard
  allocator ceiling (per-process memory fraction or an explicit cap).
- **TOCTOU re-check.** Measure free VRAM immediately before launch, start
  at reduced load, and scale up only after a few stable minutes.
- If the GPU is fully used, do not compete: end the turn or leave a
  background watcher; never busy-poll.

## Measurement before any launch
- Utilisation is a **windowed average (10-30 s)**, never an instantaneous
  reading, and is re-measured before every launch.
- Probe with `scripts/vram_probe.py` / vram-smoke configs. Never re-run a
  probe whose verdict is already recorded: log once, reuse.
- Bottleneck-first escalation: compute (batch/accum) -> data (workers,
  pin_memory, prefetch) -> memory (chunking).

## CPU fallback and job integrity
- CPU work is acceptable when the machine is free or partially free; keep
  total CPU pressure moderate while any training run or foreign workload
  is active (leave dataloader and foreign-process headroom). A GPU-sized
  job run on CPU only when a measured probe-based estimate puts it
  at roughly a couple dozen minutes or less — never a guess.
- If the GPU frees while a CPU job runs, let it finish; use the GPU for
  the next task. **Never migrate a running job between devices.**
- Prefer proper dataloader settings (num_workers, pin_memory,
  persistent_workers) and parallel data preparation regardless of GPU
  contention — cheap, contention-independent wins.

## Checkpoint frequency for long-running training
**Long-running training jobs (expected > 30 minutes) must save checkpoints at small intervals.**
A process that dies between checkpoints loses all progress since the last save. For any
training run expected to take more than 30 minutes:

- `checkpoint_policy` MUST be `last` (not `complete_only`) so checkpoints are saved
during training, not only at completion.
- `checkpoint_every_steps` MUST be ≤ 100 (save at least every 100 steps).
- `keep_last_checkpoints` MUST be ≥ 2 so a resume always has a valid checkpoint.
- The resume command is always the exact original command with zero additional flags.

Evidence: `u5_full_long2` lost 1,389 steps (4,089 → 2,700) because
`checkpoint_policy: complete_only` prevented mid-training saves.

## Component scoping
Component-specific GPU hazards live in the matching rules/components/ file,
never here. Launch-gate procedure, the read-back requirement, smoke validity,
and safe resume live in docs/agent/OPERATIONS.md.

## Performance policy (user-adopted, 2026-09-28)

- **Never launch a slow path**: gate/probe processes run batched generation (chunks of 8) and parallel CPU validation (`ThreadPoolExecutor` over all cores) as the forced defaults. A process that has to be stopped and re-run for speed-up is a waste of both GPU and user time.
- **Max hardware utilization**: before every launch, read `state/hardware_monitor.yaml` and the live nvidia-smi numbers. If measured GPU utilization is below 65%, a second process may be launched (co-run) — the second process must be read-only or tiny (verifier/smoke/validator) so neither starves the other.
- **Hardware picture is a file**: `state/hardware_monitor.yaml` records the current GPU model, VRAM cap, measured utilization baselines, and the co-run decision log. Every session updates it as the source of truth before launching anything.
- Evidence rule unchanged: co-run still never applies against this repository's own protected training jobs (one training job at a time stands; rules/gpu.md remains the safety boundary).

## Measure a contention theory before acting on it
When a run's pace looks degraded, a contention theory is a hypothesis, not
a diagnosis. Measure it — the rate with and without the suspected cause —
before acting, and never stop a job on an unmeasured theory. A theory that
does not survive its own measurement leaves the run stopped for no gain.
This supplements the no-foreign-kills and never-kill-for-pace-alone rules:
those forbid the action, this one forbids acting on the guess.
