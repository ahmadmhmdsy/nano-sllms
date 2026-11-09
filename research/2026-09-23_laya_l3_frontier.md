# Laya-line L3: applied-research report (E-65 → E-73, consolidated 2026-09-23)

Public goal: train our own small (~35–42M) in-repo decision model toward Laya/Jev-class results without any published weights.

## Headline scoreboard (all official `eval_l3`, sources = each run's eval_report.json)

| run | change | G1 macro (delta rule) | typed acc | phish AUROC | Brier | probes fail | verdict |
|---|---|---|---|---|---|---|---|
| E-66 (origin) | perm-dup aug | 0.6365 (?) | 0.5630 | 0.6086 | — | 4 | BASELINE |
| E-67 | eval-side calibration | - | 0.5615 | pos-prior is SIGNAL | — | — | negative = finding |
| E-68 | stage B 4->8 ep | 0.6592 | 0.6530 | 0.5878 | 0.1382 | 2 | recipe WIN |
| E-69 | KD distill L2 teacher | — | 0.5490 | — | — | 4 | FAIL (teacher=encoder priors) |
| E-70 | token scale 260M->520M | +0.0032 | 0.6420 | **0.6966** | 0.1415 | 4 | PARTIAL (transfer WIN) |
| E-71 | balanced replay | +0.0048 | 0.6705 | 0.6690 | 0.1325 | 4 | coupling SOFTENER |
| E-72 | stage B 8->12 ep | +0.0034 | **0.6965** | 0.6092 | 0.1312 | 4 | typed WIN |
| E-73 | mixer (+ phish early stop) | -0.0025 | 0.6560 | **0.6932** | 0.1367 | 4 | WIN (both gates) |
Als available: L1 0.6205 / L2 (MiniLM pretrained) 0.6585 typed, 0.6856 macro.
Published class: Laya-FT 0.766 typed / 0.678 phish, Jev 0.727 / 0.689, teacher ceiling 0.735.

## The measured frontier (one 520M-token encoder)
E-70 (0.642, 0.697) - E-71 (0.671, 0.669) - E-73 (0.656, 0.693) - E-72 (0.697, 0.609).
Typed and transfer are genuinely one dial: length +typed / -transfer; token scale +transfer / -typed; balanced replay (+ stratification) and a metric-pegged early stop redistribute the trade instead of removing it.

## Production candidates
- `runs/laya/l3_e73/final/model.pt` - general purpose (both gates pass: published-class transfer AND typed hold; retention preserved).
- `runs/laya/l3_e72/final/model.pt` - typed specialist (0.6965, line-best accuracy).
- `runs/laya/l3_e70/final/model.pt` - transfer specialist (published-beating phishing).
Encoder lineage: `runs/laya/l3_e70_pretrain/pretrain_final/encoder` (35.1M params, ~520M effective domain tokens).

## What learned (levers, one per rung)
1. Perm-duplicate aug: +4.45 typed at equal cost (the recipe baseline).
2. Eval-side calibration: NEVER pays (three failures now recorded: per-group T, global T, position prior). Fitted calibration requires acc AND ECE to BOTH improve on held-out data.
3. Stage-B length 4->8: largest recipe-class typed gain but the curve was unfinished (both later show >=8 still rising).
4. KD from a teacher with different encoder priors: subtracts value (soft targets carry the teacher's biases, not information).
5. Domain token scale (260M->520M double pass): transfer +0.109 AUROC class, typed -1.1. General-process quality (MLM ppl 10.97->8.12) benefits distant zero-shot tasks more than in-domain heads.
6. Per-source balanced replay (flag replay_balanced, default-off): recovers 2/3 of length-induced transfer decay, costs typed vs the pure length champion.
7. Metric-pegged early stop (flag stageB_auroc_stop, default-off): converts losing downstream epochs into saved compute and preserves the protected metric — zero extra eval cost because the stop reuses the existing per-epoch bench numbers.
8. G3 perm-agreement is architectural, not a recipe: ~0.19-0.24 across four recipes vs incumbent 0.385; record-only per E-67 standing decision. Only an architecture-level rung (option-isolated packing) reaches it.

## Discipline lessons (permanent)
Pre-register with fixed gates; single-lever variation; official eval only when training completes; never touch eval sets; parse numbers from run artifacts (R2); frozen-head probes predict downstream transfer before any fine-tune; preflight C1-C5 + free-disk before launch (0-GB disk once mimicked checkpoint corruption); pwsh-timeout kills smoke mid-save and poison-resumes later runs (always purge *_smoke dirs); checkpoint-carry belongs in config generation; one genre/channel per file (TASKS.md is pwsh-written).

## Artifacts & logs (all teed consoles + per-epoch bench JSONL + TensorBoard)
Consoles: runs/laya/l3_e6{6,8,9}, l3_e7{0,1,2,3}_console.log (+ l3_e70_pretrain). Bench: each run dir's bench_log.jsonl. Reports: each out_dir's {train_summary.json,eval_report.json}.