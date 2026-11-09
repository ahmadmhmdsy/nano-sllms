# laya/ - Laya-line: non-autoregressive typed-decision models

Recreation study of convaiinnovations/laya (System-1 decision model) at
50M-class scale. Research basis: research/laya-decision-model-report.md.
Pre-registration: research/EXPERIMENTS.md E-62 (L1) / E-63 (L2).
Experiment report: research/2026-09-22_laya_l1.md.

## Status (2026-09-22)

- L1 (E-62, user-approved): MiniLM-L12-H384 + Laya decision head (unchanged)
  + soft-CE on LocalLLaMA/typed-decisions train split. Gate: test acc >= 0.587.
- L2 (E-63, user-approved, starts after L1 closes): RLCD (REINFORCE +
  group-mean baseline, proper reward) + temperature calibration + generalist
  mixture (BoolQ/SQuADv2/SNLI/MNLI/ANLI/SciTail/yelp).
- L3 (50-100M scale / from-scratch pretrain): deferred on L1/L2 results.

## Files

- laya_head.py          model port + packing + collate + proper reward
- scripts/build_data.py download typed-decisions -> packed .pt (CPU+net only)
- scripts/train_l1.py   soft-CE trainer (auto-resume at epoch boundaries)
- scripts/eval_l1.py    test-split metrics + gates -> runs/laya/l1/eval_report.json

## Commands

    & .\.venv\Scripts\python.exe laya/scripts/build_data.py --config configs/laya_l1.yaml
    & .\.venv\Scripts\python.exe laya/scripts/train_l1.py --config configs/laya_l1.yaml --probe
    & .\.venv\Scripts\python.exe laya/scripts/train_l1.py --config configs/laya_l1.yaml
    & .\.venv\Scripts\python.exe laya/scripts/eval_l1.py --config configs/laya_l1.yaml

## Honest deviations from the original head

- Option marker = pretrained [MASK] token (Laya adds a dedicated <opt> token).
- State truncated right (keep beginning); Laya truncates left for episodes.
- act/escalate head present but untrained (matches the published notebook).
- Weights stay local: runs/laya/**/*.pt gitignored (regenerable, ~400 MB).
