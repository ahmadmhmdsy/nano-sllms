# μ1 Composition Bake-off — Implementation Plan (ME-line Stage 1)

Date: 2026-09-19 · Owner: this agent · Base doc: research/micro_experts/DESIGN.md §4
Parent plans: 2026-09-18-micro-expert-composition-plan.md (μ0, closed mu0c) · report: MU0_REPORT.md

## Facts inherited from μ0 (benefit of prior results)
- 12K-step finals are the best generalizers for every arm's expert material
  (mu0c showed min-eval at 6-8K steps; 120K runs overfit mid-run, exact-match flat/worse).
  Expert material for ALL mu1 arms: runs/mex/archive_12000/<t>/final (E-25/E-26 experts).
- USER-APPROVED (2026-09-19): composition students train 12K steps (cosine-tail
  saturation point; a longer window bought nothing on MUO4QK5 evidence).
- Control reference: runs/mex/control/final (120K, eval loss 1.0601, val
  loss-curve min at 88K). Cross-task exact-match baseline recorded in E-26.
- GPU parallelism was user-authorized for these tiny models; sequential GPU rule
  still applies to unrelated heavy jobs.

## Metrics (frozen in the E-xx pre-registrations before each arm's results)
- per-task exact match on the same held-out mixes used by eval_mex.py
  (x1 500 items; x2/x3/x4 2000 items each?). Follow the eval_mex.py loading.
- routing accuracy on UNLABELED mixed inputs (arm C headline).
- composition premium = best arm - dense control, per task and mixed.
- binomial 95% CI on every exact-match claim (500-sample gate discipline,
  TASKS row 80 follow-up) — added to the eval harness once, used by all arms.
- wall-clock + params + tokens bookkeeping for every arm (ME-D5).

## Arms in scope (per user decision 1: all four)
| arm | mechanism | status |
|---|---|---|
| A | weight merge (uniform soup first, TIES sign-election fallback) from the four 12K finals | in scope, this stage |
| B | MoE merge (experts -> FFN slots + learned router) | USER-GATED arch change — requires new MoE block in src/model.py + trainer path; presented to the user at the A/C/D readout |
| C | dispatch router: whole input -> one expert; router trained on labeled union (task id), invoked UNLABELED | in scope, this stage |
| D | committee distill: dense student (control shape 784,320 params), loss 0.5*KL(teacher_ensemble)+0.5*CE on the union data, 12K steps, cosine 1e-3, seed 42; teachers = 12K expert finals | in scope, this stage |

## Mixed eval set
Unlabeled mixed prompts (no task label in the text) = deterministic seeded mix
from the existing generators + the x1 wordlist val slice:
- 200 items: 50 per task, interleaved in seed order, saved to
  data/mex/mixed/val.jsonl with true task id for scoring (label kept out of model input).
- Per-task held-out mixes are the same ones eval_mex.py already rebuilds from
  generators/wordlist (no new data download; CPU-only; deterministic).

## Execution order
1. CPU harness first: mixed eval set builder + routing scorer + binomial CI
   wired into eval_mex.py (test in mex/tests).
2. Arm A (cheap, no training): soup script (uniform + TIES), eval vs experts.
3. Arm C (small classifier training, GPU-light): route model on labeled union.
4. Arm D (GPU window): train_distill.py student, 12K steps; then eval.
5. REPORTING: report + E-rows results commits per arm (row-first discipline kept:
   the E-27/E-28/E-29 rows land BEFORE any of their arm's results).
6. Present arm B (MoE arch change) as the user gate with A/C/D evidence.

## Risks
- Merge conflicts across experts (char embeddings fine; attention heads may
  disagree) - measure, do not tune away; a failed soup is a result.
- Router OOD: train router on the same mixed distribution it is evaluated on,
  never on labels present in the input text.
- Distill students inherit committee blind spots - report as measured.
- GPU contention: no new GPU job concurrent with unrelated heavy jobs;
  mu-level parallelism within micromodels only (prior user authorization).
