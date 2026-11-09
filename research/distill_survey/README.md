# distill_survey/ — distillation + context research (2026-09-08)

User-requested survey (2026-09-08): how frontier labs distill (and why
distill > pretrain), per-stage data recipes, the Sept-2026 model landscape
(DeepSeek-V4, Qwen3.8), distillation from cloud models with text-only
access, and effective context beyond the 1024/6 GB VRAM cap (compression,
in-model memory).

## Map

| File | Covers |
|---|---|
| `REPORT.md` | Synthesis + prioritized, user-gated adoption plan |
| `adoption_plan.md` | Tracks A-H: concrete specs, gates, costs, files touched — integrated with `research/training_analysis_2026-09-08.md` (standing 3-surface gates, knee-stop, 1-epoch rule); +2026-09-08 COMPARISON addendum (Track H, D/G data-prep bullets, E pointers) |
| `COMPARISON.md` | 2026-09-08 comparison of this survey vs the three parallel runs saved under `other_research/` (Claude / Perplexity / ChatGPT): verdict table, cross-validation, errors found, merge candidates (user-gated) |
| `notes/01_distillation_methods.md` | Trace SFT / SeqKD / GKD / MiniLLM / DistiLLM / on-policy (TM blog); Qwen3 + R1 recipes; DeepSeek-V4 + Qwen3.8 landscape |
| `notes/02_api_only_distillation.md` | Black-box taxonomy: SeqKD, CoT distill (DSS, SCoTD), GAD, SODA, judge-rerank, DPO-from-scores; V2 options for Tier 2 |
| `notes/03_training_stages_data.md` | Stage map (bulk → mid-train → anneal → ctx ext → SFT → RL → distill); SmolLM2/3 + Qwen3 + DeepSeek-V3 recipes; LR-decay/HQ-last conflict; MTP |
| `notes/04_context_extension_memory.md` | NTK/YaRN/StreamingLLM; KV math for OUR model; ICAE-family compression; RMT/Infini-attention/Titans memory; ranked feasibility |
| `exa_tasks_round6.json` | The 22 Exa tasks that produced the payloads |
| `ver_tasks_round7.json` | 2026-09-08 verification round (12 tasks) for the COMPARISON merge pointers — payloads in `research/raw/ver7_*.json` |

Raw payloads: `research/raw/` (kd1–kd7, stg1–stg5, ctx1–ctx5, chk1–chk2,
ans1–ans3 — repo convention keeps all raw Exa output there).

Re-run: `& .\.venv\Scripts\python.exe scripts\exa_research.py --tasks research\distill_survey\exa_tasks_round6.json`

Conventions: VERIFIED (payload text) / REPORTED (single source) /
UNVERIFIED (no source found) labels throughout; COMPUTED marks our own
arithmetic. Related repo docs: `research/c12_distillation_report.md`,
`research/c12_tier2_brief.md`, `research/qwen3_8_flash_next_research.md`,
`research/gdn_sandbox_design.md`, `research/pretrain_mix_proposal.md`.
