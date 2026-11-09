# ENVIRONMENT.md — stable environment policy and historical machine profiles

This file is STABLE: it holds policy, durable constraints, and the
historical machine profiles kept for reference. It deliberately does NOT
hold volatile current facts. Where they live instead:

- Current machine, GPU, driver, disk, and availability:
  [CURRENT_STATE.md](CURRENT_STATE.md) (generated; do not hand-edit) and
  state/current.json — they answer "what is true right now", and default
  to `UNKNOWN` until a session verifies them.
- Stable software inventory (venv path, versions, precision policy):
  [state/environment.json](state/environment.json).
- Verify at session start before any launch: nvidia-smi plus
  $env:COMPUTERNAME; the box pair-flips, so treat yesterday's machine
  name as UNKNOWN for a new session.

## Command and dependency policy (stable)

- Every interpreter command goes through the repo venv: & .\.venv\Scripts\python.exe
- Never bare python, never bare pip, never outside the venv.
- Dependency ownership: .venv is built and installed by the **uv**
  manager (rebuild recipe is recorded in HANDOFF.md §4; run it exactly
  as written). Missing dependencies are installed into .venv by the uv
  manager and then recorded in [state/environment.json](state/environment.json) —
  nothing is installed "just in case".
- pyflakes is the one recorded exception-style venv-level dev tool:
  installed into .venv by uv (2026-09-23, 4.0.0, not in pyproject); it
  powers preflight_check.py's C2 undefined-name scan.
- flow-line dev stack is owned by FlowServer requirements: fastapi +
  uvicorn pinned in flow/server/requirements.txt (already present in
  .venv as of 2026-09-13); Node frontends run through pnpm 12
  (pnpm approve-builds esbuild was accepted once in
  flow/pnpm-workspace.yaml, else ERR_PNPM_IGNORED_BUILDS); the flow
  server default port is 3010 (--port or FLOW_PORT).

## Hardware policy (stable)

- Both cards seen on this project are Turing (sm_75): **fp16 only,
  no bf16 (Turing has no bf16), flash-attn not used — PyTorch SDPA**.
  This precision policy does NOT change when the box pair-flips.
- Training is the only GPU-heavy activity; the never-co-run policy
  (one protected training job, GPU co-rules) is owned by
  [rules/gpu.md](rules/gpu.md), not by this file.

## Historical machine profiles (kept for reference, not current truth)

These are recorded sessions, in date order. None of them is "the
current machine": check CURRENT_STATE.md.

- DESKTOP-MUO4QK5 (Quadro RTX 4000, 8 GB, driver 595.97): M3 fresh run
  2026-09-06; re-verified live 2026-09-22 during the laya-line E-62
  session (driver 616.92, 8192 MiB, E: free 18.7 GB at that moment).
- DESKTOP-TU09FBO (RTX 3000, 6144 MiB, driver 580.92): Track A session
  2026-09-09 (M2 + M3 probe ran here: pilot ~3.3 GB; target-config probe
  4.24 GB allocated / 4.4 GB reserved at seqs 512 AND 1024); M3 resumed
  at step 2000 on this box vs. its fresh start on MUO4QK5.
- Machine-move history: the box pair flip is real and expected;
  bundle trainer_state paths carry the source machine's root (the
  2026-09-09 Track C path had a space, no space since); patch on arrival.
- Thermal behavior: sustained cycles 84–90 °C with SM-clock throttling;
  pace swings are normal (see MEMORY.md Lessons).
- Historical VRAM datapoints (verify-a-launch-time, never assumed):
  teacher-touching trainer arm measured 2.69 GB / 6 GB (frozen
  SmolLM2-135M fp32 + fp16 student, ctx 512, b1/accum 32, row 49
  2026-09-11/12); sustained ~19–20 s/it at 1000-step arms; no NaN across
  5 arms.

## Capability table (durable, historical)

| Capability | State |
|---|---|
| CUDA training | proven on both cards; auto-resume proven (M0 kill/resume, M2 step-1001) |
| Gated HF datasets | HF_TOKEN in .env WORKS for accepted repos (2026-09-09); 403 with a VALID token = terms not accepted, not a token problem |
| Network | fast in practice (2026-09-09 measurement); never re-download what data/*/tokens/*.bin already holds |
| LFS | ~790 MB of 1 GB used at last check (2026-09-09) — no new >100 MB files to LFS; large weights stay local by user policy |
| Disk headroom | volatile value — live in CURRENT_STATE.md, not here |
