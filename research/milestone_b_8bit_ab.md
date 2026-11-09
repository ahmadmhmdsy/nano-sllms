# Milestone B - Throughput A/B protocol: 8-bit Adam + batch/accum retune

Status: protocol written 2026-09-06 (CPU-safe parts done beside live M3).
**All GPU parts are post-M3 only** - never co-run an A/B arm with the live
M3 trainer (TASKS row 1 / the never-co-run rule).

## Verified preconditions (2026-09-06)

- bitsandbytes **0.50.2 is installed** in the project venv (checked via
  importlib.metadata - no CUDA import needed while M3 is live). sm_75
  support for 8-bit Adam on this exact GPU class was verified in the
  C12 Tier 2 readiness brief (research/c12_tier2_brief.md, MEMORY bnb row).
- CPU-side full import/CUDA-context test of bitsandbytes is deliberately
  deferred: `import bitsandbytes` may create a CUDA context, which would
  add VRAM pressure to the live M3 run. First post-M3 action: import check.
- train.py now stamps an env fingerprint (git commit, python/platform,
  package versions incl. torch/transformers/peft/bitsandbytes, GPU name,
  resolved config) into runs/<phase>/final/train_summary.json - additive
  fields only, auto-resume contract unaffected (py_compile PASS).

## Arms (pilot-scale, equal steps, isolated run dirs)

| Arm | Config | Run dir | Change vs M2 pilot |
|---|---|---|---|
| A (control) | configs/pilot_ab_base.yaml | runs/pilot-ab-base | fp32 AdamW, 500-step window |
| B | configs/pilot_b8.yaml | runs/pilot-b8 | optim: adamw_bnb_8bit |
| C | configs/pilot_b8_r2x16.yaml | runs/pilot-b8-r2x16 | 8-bit + batch 2 / accum 16 (same effective 16,384 tok/step) |

All arms share the M2 pilot data (data/pilot/tokens), seed 42, cosine LR,
warmup 50, ctx 512, and a 500-step window with logging_steps 10 (fine-grained
s/it + tokens/sec via scripts/status.py).

## Protocol (run only when the GPU is free)

1. GPU-free check: nvidia-smi shows no compute process; M3 finished and
   runs/target/final exists (or the user explicitly re-scheduled M3).
2. Import gate: `.venv/Scripts/python -c "import bitsandbytes; print(bitsandbytes.__version__)"`
   must PASS before any arm.
3. Run arms sequentially (one at a time):
   `& .\.venv\Scripts\python.exe scripts/train.py --config configs/pilot_ab_base.yaml` (A)
   then the same for pilot_b8.yaml (B) and pilot_b8_r2x16.yaml (C).
4. Kill/resume drill per arm (auto-resume re-proof with a quantized
   optimizer state): mid-arm, kill the trainer, re-run the exact command
   with zero flags, confirm it resumes from checkpoint-250 and finishes.
5. Collect: scripts/status.py (tokens/sec lines per run dir), eval loss at
   step 500 from each arm's trainer_state/final summary, VRAM peak
   (scripts/vram_probe.py --config <arm> for a measured peak per arm).
6. Decision + docs: record the table below into TASKS row 11 evidence and
   MEMORY; the winning optim config then feeds any future full retrain.

## Acceptance gates (measured, not estimated)

- Loss parity: B eval_loss@500 within +/-0.02 of A. FAIL -> keep fp32 AdamW.
- Pace: B mean tokens/sec >= 0.9x A. FAIL -> keep fp32 AdamW.
- VRAM: B peak < A peak with >= 500 MiB headroom, or C fits batch 2 at
  ctx 1024-class workloads where batch 1-only would OOM.
- Auto-resume: kill/resume drill PASS on every arm that wins anything.
- Any arm that passes all gates becomes the recorded default for the NEXT
  pretrain; nothing in this A/B changes the running M3 or its resume path.

## Risks

- 8-bit quantized optimizer moments + fp16 master weights: watch for loss
  spikes in the first 100 steps (logging_steps 10 makes this visible).
- bitsandbytes resume path differs from torch AdamW state dicts - the
  drill in step 4 is the proof, not an assumption.
- A/B run dirs (runs/pilot-ab-base, runs/pilot-b8, runs/pilot-b8-r2x16)
  are disposable; deleting them afterwards needs user approval per policy.

## Non-goals

- No change to the live M3 run, its config, or its checkpoints.
- No 8-bit switch on the SFT path in this milestone (SFT VRAM is tracked
  separately in the C12 runbook).
