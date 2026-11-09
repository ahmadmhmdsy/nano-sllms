---
id: components/core
status: active
scope: core
applies_when: "The task runs or extends the core SLM pipeline: model code, data preparation, tokenization, training, evaluation, or inference."
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - AGENTS.md
  - PLAN.md
  - src/model.py
  - src/data.py
  - configs/
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
---

# core — SLM pipeline component rule

## Scope and entry points
- Model and data code: `src/model.py` (GQA decoder) and `src/data.py`
  (memmap packed-shard dataset).
- Pipeline scripts under `scripts/` (sanity_check, prepare_data,
  tokenize_data, train, eval, infer, vram_probe, sft_data, sft) always run
  through the venv with the matching phase config from `configs/`.
- The five-document chain (PLAN, HANDOFF, TASKS, MEMORY, ENVIRONMENT) is read
  before pipeline, model, or data work.
- **pipeline_v2** (Release 1: pretrain only): `pipeline_v2/run.py` plus the
  `slm_train` package. Entry: `.\.venv\Scripts\python.exe pipeline_v2/run.py <command> --config
  pipeline_v2/configs/experiments/<run>.yaml`. Its slice of the truth lives in
  `pipeline_v2/README.md`, `pipeline_v2/docs/acceptance.md`, and
  `pipeline_v2/docs/gap-analysis.md`; the remediation plan is
  `pipeline_v2/docs/remediation-plan.md`.

## Smallest validation
For code changes: the affected script's own gate (config sanity check or a
--help smoke) plus the existing unit tests covering the touched modules.
For a pipeline_v2 code change:
`& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit pipeline_v2/tests/contract -q`
(CPU-only); a config change additionally runs `.\.venv\Scripts\python.exe pipeline_v2/run.py
validate --config pipeline_v2/configs/experiments/smoke.yaml`.
No GPU claim without `rules/gpu.md` having been read for the task.

## Boundaries
- Single Turing GPU (sm_75): fp16 only; no bf16; no flash-attn (PyTorch SDPA
  instead) — recorded facts live in ENVIRONMENT.md, not re-derived here.
- Training auto-resumes: after any crash or kill, re-run the exact command
  with zero flags.
- transformers 5.x API drift facts are recorded in MEMORY.md; cite them,
  do not re-enumerate.

## Hazards
- Never launch a second GPU job while a repository training run is active:
  governed by `rules/gpu.md`.
- Never delete or rewrite raw or tokenized data: governed by
  `rules/data.md`.
- `scratch/` is throwaway; it is never treated as durable knowledge and
  is never mapped into this document map.

## Evidence sources
- Root chain documents (PLAN.md, HANDOFF.md, TASKS.md, MEMORY.md,
  ENVIRONMENT.md).
- `research/EXPERIMENTS.md` and `research/WHAT_WORKS.md` for run
  verdicts.
