# AGENTS.md — agent guide & document map

Any agent working in this repo continues one long-running project: train
small language models (S smoke → P pilot → T target) on a single consumer
GPU with an unattended-safe, auto-resuming pipeline.

[CLAUDE.md](./CLAUDE.md) is the **operating system** — read it first. This
file is the **repo supplement**: it maps every durable document to its job
and fixes the repo-specific conventions CLAUDE.md points here for.

> CLAUDE.md's read-order table and cross-references were trimmed
> (2026-09-07) to match this map — the harness-internal targets
> (`packages/`, `examples/`, `vendor/`, `planning/`, `docs/`,
> `.agents/notes/`) do not exist in this repository. If you find a stale
> cross-link in CLAUDE.md, report it — do not follow it. The map below is
> the real one.

## 1) Document map — one authoritative home per concern

| Concern | File | Read it when | Update it when |
|---|---|---|---|
| Operating rules | [CLAUDE.md](./CLAUDE.md) | every session | only for deliberate OS changes (rare) |
| **Plan** (the spec) | [PLAN.md](./PLAN.md) | before any pipeline/model/data work | only for user-approved spec changes |
| **State** (where we are) | [HANDOFF.md](./HANDOFF.md) | at session start | as milestones/incidents land — snapshot style, newest entries win |
| **Tasks** (what's next) | [TASKS.md](./TASKS.md) | at session start | whenever a task starts/finishes/gets gated |
| **Memory** (lessons & decisions) | [MEMORY.md](./MEMORY.md) | before touching anything non-obvious | every hard-won gotcha or user decision, immediately |
| **Environment** (what we run on) | [ENVIRONMENT.md](./ENVIRONMENT.md) | before running anything | after any machine/venv/version/disk change |
| Usage (humans) | [README.md](./README.md) | for command syntax | when commands or usage change |
| Research | [research/](research) | when touching architecture/data direction | new studies live here; raw payloads under `research/raw/` |
| **Experiments ledger** | [research/EXPERIMENTS.md](research/EXPERIMENTS.md) | **before planning any experiment** (what exists, what it earned) | one row whenever a task's experiment closes; report = research/_template_experiment.md skeleton |
| Proven levers | [research/WHAT_WORKS.md](research/WHAT_WORKS.md) | before planning/launching any new training run | whenever a verdict changes what future training should adopt/avoid |
| **Web UI spec** | [WEBUI_PRD.md](./WEBUI_PRD.md) | before any `webui/` work | when a UI decision or milestone scope changes (user-gated) |
| Implementation plans | [docs/plans/](docs/plans) | before executing a planned feature | when a plan is created or completed (kept as historical record) |
| **Arabic OCR line** (status + audit) | [arabic_ocr/STATUS.md](arabic_ocr/STATUS.md) · [arabic_ocr/RETROSPECTIVE.md](arabic_ocr/RETROSPECTIVE.md) | before any OCR-stage work | as each OCR stage/gate lands |

The plan / state / task / memory / environment docs are the five an agent
needs before doing anything; the table is how they chain together. Mnemonic:

    PLAN = what to build      HANDOFF = where we are      TASKS = what's next
    MEMORY = what we learned  ENVIRONMENT = what we run on

## 2) Repository layout

```
nano_SLMs/
├─ AGENTS.md · CLAUDE.md · PLAN.md · HANDOFF.md · TASKS.md · MEMORY.md · ENVIRONMENT.md · README.md
├─ configs/            smoke.yaml · pilot.yaml · target.yaml · sft_t1.yaml
├─ scripts/            sanity_check · prepare_data · tokenize_data · train · eval · infer ·
│                      vram_probe · sft_data · sft · exa_research (+ exa_search.py/.bat at root)
├─ src/                model.py (GQA decoder) · data.py (memmap packed-shard dataset)
├─ data/               <phase>/raw/ (gitignored) · <phase>/tokens/*.bin (tracked) · sft/evol/
├─ runs/               <phase>/checkpoint-* · <phase>/final · <phase>/logs (TensorBoard)
├─ resources/          original user-supplied study notes (pseudo-code skeletons)
├─ research/           web-research notes + raw Exa payloads (c12 distillation plan/report)
├─ webui/              Gradio web UI (spec: WEBUI_PRD.md; milestones U1-U5)
├─ viz/                React model-architecture explorer — config-driven, pnpm only (docs: viz/README.md; plan: docs/plans/2026-09-10-model-viz-react-platform.md)
├─ flow/               Visual flow editor MVP — drag-and-drop pipeline graph (dataset→prepare→tokenize→train→eval/infer), FastAPI backend (docs: flow/README.md; plan: docs/plans/2026-09-13-flow-editor-mvp-plan.md); `.flow.json` format flow/0.1
├─ langid/             DA-line: text language ID - Tongue-analogue (DESIGN: research/desert_ant_recreation/DESIGN.md; plan: docs/plans/2026-09-19-desert-ant-recreation-da1-langid-plan.md); CPU-only, gates pre-registered (rows 81-87)
├─ diacritizer/        D-line: Arabic-diacritization submodule (own README.md; live GPU train runs under diacritizer/scripts/)
├─ mex/                μ-line micro-expert sandbox (μ0/μ1/μ2; train_mu2_lora.py --config configs/dia2f_replay.yaml live as of 2026-09-20)
├─ arabic_ocr/         Arabic OCR two-stage nano model (line status: STATUS.md; audit: RETROSPECTIVE.md; plan: docs/plans/2026-09-29-arabic-ocr-nano-model.md); own models/ (gitignored) + .venv-paddle-gpu
├─ scratch/            throwaway experiment scripts (NON-durable: never map knowledge here)
├─ models/ + tests/    shared model weights/tests (referenced from flow/ and line submodules)
├─ checkpoint_backup/  untracked; user-staged machine-move checkpoint zips (HANDOFF §3b) — check for a newer checkpoint-*.zip first
└─ .venv/              uv-managed CPython 3.12.9 (never pip)
```

## 3) Commands

All Python runs through the venv — never bare `python`, never `pip`:

```powershell
& .\.venv\Scripts\python.exe scripts\sanity_check.py --config configs\<phase>.yaml
& .\.venv\Scripts\python.exe scripts\prepare_data.py  --config configs\<phase>.yaml
& .\.venv\Scripts\python.exe scripts\tokenize_data.py --config configs\<phase>.yaml
& .\.venv\Scripts\python.exe scripts\train.py         --config configs\<phase>.yaml
& .\.venv\Scripts\python.exe scripts\eval.py          --config configs\<phase>.yaml
& .\.venv\Scripts\python.exe scripts\vram_probe.py    --config configs\target.yaml
& .\.venv\Scripts\python.exe scripts\sft_data.py      --config configs\sft_t1.yaml
& .\.venv\Scripts\python.exe scripts\sft.py           --config configs\sft_t1.yaml --pilot
& .\.venv\Scripts\python.exe scripts\bench_passk.py   --arm NAME=PATH --device cuda --vram-cap-gib 4.5   # HumanEval+MBPP pass@1 (GPU)
& .\.venv\Scripts\python.exe scripts\bench_ladder.py  --arm NAME=hf:PATH --device cuda --up-to L4 --out-dir research/raw/ladder_NAME   # stepped battery: L0-L4 71→1,056 robustness; B1 192 / W2 560 / W4 1,120 breadth (PLAN §11.2); --dry-run = counts only
& .\.venv\Scripts\python.exe -I scripts\bench_ns_eval.py --model PATH --name NAME --quick --batch 128 --vram-cap-gib CAP   # never-seen bench, 11 separate sets (PLAN §12); drop --quick for the full tier; resumes per set; CAP = fresh free VRAM - 0.75 GiB
& .\.venv\Scripts\python.exe scripts\rl_smollm2.py    --init runs/sft_smollm2_cpt_pg/final --output runs/rl_smollm2 --groups 2000 --vram-cap-gib 4.5   # SA-6 RLOO on playground `rl` split (GPU; auto-resumes); --dry-run = reward self-check
& .\.venv\Scripts\python.exe laya/scripts/preflight_check.py --script <trainer>.py --config <cfg>.yaml   # REQUIRED gate before ANY unattended long launch (dup-keys, pyflakes, --help, data, GPU smoke)
& .\.venv\Scripts\python.exe laya/scripts/param_count.py --layers 6 --hidden 512 --ffn 2048 --heads 8 --vocab 30522 --target_m 50   # pre-registration param check (rule R5)
# flow/ visual flow editor (development build + server; port 3010 via FLOW_PORT)
cd flow; pnpm build                                     # frontend -> flow/dist
& .\.venv\Scripts\python.exe -m flow.server.app                # port 3010 (or --port)
cd flow; pnpm test                                      # vitest (148 tests)
& .\.venv\Scripts\python.exe -m unittest discover -s flow/tests -t flow  # server tests (145)
```

- `sft_data.py` is CPU + network only — safe to run while a train job is
  active (it touches nothing under `runs/`).

- `train.py` auto-resumes: after ANY crash/kill re-run the exact command —
  zero flags (the user's hard requirement, PLAN §5.3).
- **Net/web research goes through the Exa helper** (root `exa_search.py` /
  `exa_search.bat`; API key `EXA_API_KEY` in project-root `.env`, never
  committed; batch mode `scripts/exa_research.py --tasks <file.json>` saves raw
  payloads under `research/raw/`). The harness `web_search` tool failed
  repeatedly here (2026-09-06) — prefer the helper.
- Full usage (infer REPL, `ask_model.bat`, Exa web-search helper):
  [README.md](./README.md).

## 4) Environment constraints (hard facts — details in [ENVIRONMENT.md](./ENVIRONMENT.md))

1. Single Turing GPU (sm_75): **fp16 only** — never bf16; no flash-attn (PyTorch SDPA instead).
2. transformers 5.16.1 API drift: no `logging_dir`/`save_safetensors` args; `processing_class=`;
   `eval_strategy=`; no console loss lines — read TensorBoard events instead (see MEMORY.md).
3. Disk headroom swings (~10 GB during the MUO4QK5 window; 36 GB at the TU09FBO step-2000 resume): checkpoint rotation (`save_total_limit=3`) +
   local-only weights policy; report before deleting anything.
4. The bigcode datasets are gated; ungated fallbacks and the dataset-namespace
   moves are recorded in [MEMORY.md](./MEMORY.md).
5. The GPU thermally throttles under sustained training — pace swings are normal;
   never kill a run for pace alone.
6. The GPU is a single shared resource: never launch another GPU job
   (`eval.py`, `infer`, `vram_probe.py`, `sft.py`) while a `train.py` run is
   active — it can OOM and kill the training run. CPU-only scripts
   (`sft_data.py`) are exempt. Leftover-compute co-runs against FOREIGN
   non-training incumbents are governed only by CLAUDE.md §20 (first
   evidence row = TASKS 98, still pending).

## 5) Git / LFS rules (summary — details in HANDOFF §7)

- `optimizer.pt` is gitignored on purpose (free LFS ~1 GB quota). M3/target
  weights stay **local by user decision**; metrics, logs, small final-dir files
  and tfevents are tracked.
- Keep `data/*/raw/` out of git (regenerable); keep `data/*/tokens/*.bin` in
  (re-downloading on the slow network is the expensive part).
- Before any push: `git lfs status`; no >100 MB file outside LFS.
- Only processes this agent started may be stopped; a running train job is
  never collateral.

## 6) Conventions

- Validation labels PASS / FAIL / SKIPPED / BLOCKED; honest reporting over
  optimistic claims (CLAUDE.md §10, §16).
- Task statuses `pending` / `in_progress` / `blocked` / `done` in
  [TASKS.md](./TASKS.md); user-gated items say so explicitly.
- Keep dates machine-verifiable (git log, tfevents, logs) rather than vague.

## 7) Keeping the map honest

- New durable knowledge → [MEMORY.md](./MEMORY.md). New work →
  [TASKS.md](./TASKS.md). A new *kind* of document must be added to the map
  in §1 — the map is the index; a doc outside the map does not exist for the
  next agent.
- HANDOFF.md and TASKS.md may describe the same next step: TASKS.md owns the
  live status, HANDOFF owns the narrative and evidence.
