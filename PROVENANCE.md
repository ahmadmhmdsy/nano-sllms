# Provenance - fresh-history public snapshot

This repository is a curated, fresh-history snapshot published on the `ahmadmhmdsy` account.
It merges two lines that diverged from commit `9f4a87b` (2026-09-24):

- **Local line**: this machine's `main` at `4f63dbd` plus the uncommitted working-tree state (118 commits after the base).
- **Imported line**: `alshahia/nano_SLMs` `origin/main` at `534196ca` (2026-11-09, 68 commits after the base).

## What was imported from `alshahia/nano_SLMs`

- **SA/RL pipeline** (`scripts/sa*.py`, `scripts/rl_smollm2.py`, `scripts/sa7_*.py`, `scripts/tool_*.py`, `scripts/teacher_client.py`,
  `scripts/bench_*.py`, `scripts/e2_verifier*.py`): sandboxed verifier, SFT/RFT stages, RLOO RL sandbox, SA-7 runtime/agent, never-seen benches.
- **Arabic OCR line** (`arabic_ocr/`): two-stage nano OCR experiments, code, configs and honest status/retrospective documents.
- **Shared-core improvements** (`scripts/sft.py`, `scripts/train.py`, `scripts/eval.py`, `scripts/prepare_data.py`, `scripts/tokenize_data.py`,
  `src/model.py`, `src/gdn.py`, plus new configs/tests) and their research notes under `research/` and `docs/plans/`.
- **Remote ledgers**: verbatim copies under `docs/imported/alshahia-2026-11-09/` (`TASKS.md`, `HANDOFF.md`, `MEMORY.md`, `AGENTS.md`,
  `ENVIRONMENT.md`, `PLAN.md`, `research/EXPERIMENTS.md`, `.gitignore`).

## GLM small-model line

The original repository tracked `glm-5.3-flash-from-scratch` as a bare gitlink (mode 160000, commit `a7e7bd45...`) with no `.gitmodules`.
That pinned tree was never published and was not present on this machine, so the pointer is unclonable. A verbatim snapshot of the public MIT
upstream `https://github.com/vukrosic/glm-5.3-flash-from-scratch` at commit `96d9c7c3bf850ca2500a41b8184b4efb8e07890f` is vendored in
`glm-5.3-flash-from-scratch/` (see its `PROVENANCE.md`). The later unpublished local extensions (playground ladders, sandbox runner,
aspect LoRA, byte-tokenizer arms) are not recoverable and are not part of this snapshot; parent scripts that import them are marked in the
status notes and remain non-runnable until those modules are restored.

## What was excluded

Weights and checkpoints (the old line used ~815 MB of Git LFS across three `*.safetensors` files), `runs/`, `data/`, `resources/`,
`ui/data/` generated corpora, `node_modules/`, machine-local caches, agent scratch, root logs, and other regenerable artifacts.
`.gitattributes` LFS mapping was removed because no LFS objects are tracked here.

## History

This repository starts from a single root commit. It carries no Git LFS objects and no history from the old repository;
the old repository remains untouched under `alshahia/nano_SLMs`.

