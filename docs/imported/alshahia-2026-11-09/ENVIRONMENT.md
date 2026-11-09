# ENVIRONMENT.md — what we run on

Verified facts about the machine this repo trains on. Update this file the
moment anything below changes (machine move, venv rebuild, version bump,
disk event). Last verified: 2026-09-09 (Track A session — machine moved
back to TU09FBO; verify at every session start, the box pair-flips).

## Machine

> **2026-10-04 (OCR-line, docs-only session):** live machine = Quadro RTX 4000
> 8 GB (sm_75, fp16 only), **E: free 8.0 GB** - down from 18.7 GB on 2026-09-22
> because the TrOCR fine-tune v2 consumed 4.3 GB (a 2.15 GB top-level
> safetensors plus a byte-duplicate `checkpoint-189/`, the latter a
> reclaim candidate). Disk is the binding constraint on the approved Arabic OCR
> plan: 500k rendered images (~25 GB) cannot be stored, so rendering must be
> chunked or done inside dataloader workers.

> **2026-09-22 (laya-line E-62):** live machine verified as DESKTOP-MUO4QK5 — Quadro RTX 4000 8 GB (driver 616.92, 8192 MiB), E: free 18.7 GB. Pair-flip confirmed by nvidia-smi + COMPUTERNAME at session start. The TU09FBO numbers below apply when the pair flips back; re-verify before any launch.

- **Current session (verified 2026-09-09, Track A start): DESKTOP-TU09FBO —
  RTX 3000, 6144 MiB, driver 580.92** (nvidia-smi + $env:COMPUTERNAME);
  project root `E:\python_projects\nano_SLMs` (no space); drive E: **36.30 GB
  free** (2026-09-09 evening, rows-2/13/16 session: −868 MB approved deletion
  of checkpoint-sft_v2_e1-final.zip, then milestone-D measurement −50 MB).
  CONCURRENT-SESSION NOTE (2026-09-09 evening): a Track A context-probe
  session and the rows-13/16 session share this working tree — edit shared
  files (src/model.py, docs) only as targeted re-read edits, never whole-file
  writes from memory.
- Prior session (2026-09-09 Track C): MUO4QK5, Quadro RTX 4000 8 GB,
  driver 595.97, root `E:\python projects\nano_SLMs` (WITH a space; bundle
  trainer_state paths carry the source machine's root, patch on arrival).
- Turing sm_75 on BOTH cards → **fp16 only**, no bf16, no flash-attn (SDPA).
- Machine-move history: DESKTOP-MUO4QK5 (RTX 4000 8 GB, driver 595.97) ↔
  DESKTOP-TU09FBO (RTX 3000 6 GB, driver 580.92). M3 ran fresh on MUO4QK5
  (2026-09-06) and resumed at step 2000 on TU09FBO; M2 + the M3 probe ran on
  the 6 GB card: pilot ~3.3 GB, target-config probe 4.24 GB allocated /
  4.4 GB reserved (seq 512 AND 1024).
- Thermal behavior: sustained training cycles 84–90 °C with SM-clock throttle;
  pace swings are normal (see MEMORY.md §Lessons).
- Mounting arms (row 49, 2026-09-11→12): teacher-touching trainer live VRAM measured
  2.69 GB / 6 GB (frozen SmolLM2-135M fp32 + fp16 student, ctx 512, b1/accum 32) —
  well under the 6 GB bound; sustained ~19–20 s/it at 1000-step arms; no NaN
  across 5 arms (honest note: fill broke its own fp16 grad norms, not VRAM).

## Software (verified this session by import)

> **2026-10-04 (verified by import probe, OCR-line):** absent from `.venv` -
> `uharfbuzz`, `arabic_reshaper`, `python-bidi`, `cv2`
> (opencv), `shapely`, `albumentations`, `paddleocr` (deliberately
> **not** needed — see below). Present: `paddlepaddle 3.3.1` (**CPU build**,
> installed 2026-10-04 for the Stage 0 bake-off; `uv pip install --python .venv
> paddlepaddle` resolved only `paddlepaddle` + `opt-einsum`, no numpy/opencv/torch
> conflict), `fontTools` (glyph outlines and
> advances). 20+ real Arabic fonts are on disk (Amiri, AmiriQuran, arabtype,
> Candarab, GARABD, Noto Kufi/Naskh/Sans Arabic, ReemKufi, Scheherazade, simpo,
> tradbdo, trado). The list below is the 2026-09-09 baseline and still holds for
> those packages.
>
> **Second venv for the OCR renderer (2026-10-04):** `.venv-paddle-gpu` is a SEPARATE
> uv venv holding what `.venv` lacks - `uharfbuzz 0.56.2` (HarfBuzz shaping),
> `freetype-py 2.5.1` (rasterization), `scipy`, and `paddlepaddle-gpu 3.3.1` (cu126,
> from Paddle's own CDN, not PyPI). It has **no torch**, so it cannot train. Conversely
> `.venv` has **no scipy**, so the Stage 2 label decoder is a hand-rolled union-find
> rather than `scipy.ndimage.label`. Rule: **render with `.venv-paddle-gpu`, train with
> `.venv`** - each lacks what the other has. `PIL` is present in both but is built
> **without Raqm** (`PIL.ImageFont.core.HAVE_RAQM is False`), so it cannot shape Arabic at
> all and is NOT a fallback for the renderer; HarfBuzz + FreeType is the only route.


- venv: `.venv` — uv-managed CPython 3.12.9. **Never pip**; use
  `uv pip install --python .venv ...` if something is missing.
- **PaddleOCR inference models load without the `paddleocr` package** (2026-10-04):
  `paddle.jit.load("arabic_ocr/models/<name>/inference")` reads the PIR
  `inference.json` + `inference.pdiparams` pair directly. Two Stage 0 recognizers
  are on disk under `arabic_ocr/models/` (49 MB, **local-only weights**, not
  committed): `ppocrv5_mobile_rec/` (PaddlePaddle/arabic_PP-OCRv5_mobile_rec, 7.8 MB)
  and `ppocrv6_small_rec/inference/` (medyas/arabic_PP-OCRv6_small_rec, 41.3 MB).
  Both export the **same** shape — logits `(B, T, 749)`, alphabet
  `['blank'] + inference.yml dict + [' ']`, output in **visual (mirrored) order**.
  Details and traps: MEMORY.md 2026-10-04.
- torch 2.14.0+cu126 (CUDA 12.6; `torch.cuda.is_available()` → True)
- transformers 5.16.1 · datasets 5.0.1 · accelerate 1.14.0 ·
  bitsandbytes 0.50.2 · tensorboard 2.21.0 · pyyaml · exa-py 2.20.0
- git-lfs 3.6.0 (system). Git remote: github.com/alshahia/nano_SLMs (main).

### Nested reference repo + dev tools (2026-10-08)

- **E-113 dense re-run (2026-10-08):** the repeat 400-step pretrain in `glm-5.3-flash-from-scratch/runs/pretrain-dense` produced `checkpoint-0100` and `checkpoint-0400` **sha256-identical** to the E-112 run, so this pretrain is bit-reproducible within this venv/torch build (2.14.0+cu126, seed 42). Disk: 11 extra dense checkpoints plus the two E-113 RL run dirs (`runs/rl-regime-a1`, `runs/rl-regime-a3`); all under the nested repo's gitignored `runs/`.

- `glm-5.3-flash-from-scratch/` is a **nested git repo tracked as a gitlink** at parent
  commit `2bd733b` (no `.gitmodules`, deliberately NOT a submodule; its own HEAD
  `96d9c7c`, its `origin` = vukrosic - **never push**). It runs INSIDE `.venv`: its
  `requirements.txt` no longer pins `torch==2.5.1` and a second venv must not be
  created. From inside it, every run is `..\.venv\Scripts\python.exe scripts\...`.
- `pytest 9.1.1` (+ iniconfig 2.3.1, pluggy 1.6.0) installed into `.venv` on
  2026-10-08 via `uv pip install --python .venv pytest`. Its default `--basetemp`
  under `%TEMP%` hits `WinError 5` in `cleanup_dead_symlinks` on this machine, so pass
  `--basetemp runs\_pytest_tmp`.
- GPU free-VRAM observations 2026-10-08 (8 GiB Quadro RTX 4000): 809 MiB free with the
  desktop + foreign consumers resident; **7,704 MiB free / 291 MiB used / 0% util after
  the user closed them**. The 25.7M fp16 recipe peaks at 2.141 GiB (pretrain bs32) and
  0.939 GiB (RL), so it fits comfortably once the desktop is clear - but a 60% cap on
  the *contended* 809 MiB (0.47 GiB) OOMs at batch 1, so the cap must be recomputed
  from a fresh reading, never reused.
- `torch 2.14.0+cu126` / `python 3.12.9` here vs the clone's reference machine
  (`torch 2.5.1+cu121`, py 3.12.3): the source of every numeric deviation in E-112.

- flow/ dev stack (2026-09-13): fastapi 0.141.1 + uvicorn 0.52.4 already
  present in `.venv` (no install needed, pinned in `flow/server/requirements.txt`);
  Node frontends run through **pnpm 12** (`pnpm approve-builds esbuild`
  persisted once in `flow/pnpm-workspace.yaml`, else ERR_PNPM_IGNORED_BUILDS);
  server default port **3010** (`--port` or `FLOW_PORT`); E2E UI drill via
  agent-browser 0.34.0 (`agent-browser install` downloaded Chrome 153; the
  daemon relaunches per pwsh call, so UI drills must chain in one call).

## venv rebuild recipe (proven once after a machine move)

See HANDOFF.md §4 — do not improvise. After any rebuild validate with
`python -c "import torch; print(torch.cuda.is_available())"` then
`scripts\sanity_check.py --config configs\smoke.yaml` (four PASS gates).

## Capabilities & limits

- pyflakes 4.0.0 installed into .venv via `uv pip install pyflakes` (2026-09-23, venv-level dev tool, not in pyproject) — powers preflight_check.py C2 undefined-name scan.

| Capability | State |
|---|---|
| CUDA training | working — M0/M1/M2 PASSED; M3 resumed at step 2000/5000 on TU09FBO (best eval_loss 1.9972) |
| Auto-resume | proven — M0 kill/resume drill; M2 resumed at exactly step 1001 |
| Disk headroom | 36.0 GB free at the step-2000 resume (was ~10 GB on MUO4QK5); rotation keeps ≤3 checkpoints (~2.6 GB each, ~7.8 GB steady state); report before deleting anything; **2026-10-04: 26.52 GB free** after the user-approved cleanup deleted 42 paths / 18.60 GB (audit TIER 0-4: the TrOCR v2 duplicate `checkpoint-189/`, the v1 TrOCR run, the teacher-forced-era OCR runs, superseded checkpoints, and the superseded OCR image mixes - whose Arabic text was harvested to `data/ocr_textpool/` first); was 8.0 GB before it. `data/smoke` (tokens/*.bin) and `data/linecorpus_smoke` were NOT deleted - the audit had mislabeled them as OCR mixes. Later same day the Stage 0 prep (paddlepaddle CPU ~400 MB installed + 49 MB of model weights) brought it to **26.1 GB** |
| Gated HF datasets | HF_TOKEN in .env WORKS for accepted repos (2026-09-09: user accepted bigcode/starcoderdata terms → content reads OK; the-stack-smol ungated since the namespace moves; both bigcode repos have NO per-language builder config — use `data_dir:`). GatedRepoError 403 with a VALID token = terms not accepted for this account, not a token problem |
| Network | fast in practice (2026-09-09: starcoderdata 2k-row stream <2 min; 1.7 GB teacher ~3 min; the old ~230 KB/s estimate was wrong) — still never re-download what `data/*/tokens/*.bin` already holds |
| Teacher APIs (2026-10-22) | GEMINI_API_KEY = environment variable, Gemini free tier (user-confirmed no billing); limits are per project per model, the daily quota resets at midnight Pacific, and exact numbers show only in AI Studio. OPENCODE_GO_API_KEY in .env: only minimax-m3.1-flash-free usable (429 after ~35 requests). Free models only - `scripts/teacher_client.py` allowlist. Disk ~32 GB free after the Tier 1/2 checkpoint cleanup |
| LFS quota | ~790 MB of 1 GB used — no new >100 MB files to LFS; M3 weights local-only (user decision) |
