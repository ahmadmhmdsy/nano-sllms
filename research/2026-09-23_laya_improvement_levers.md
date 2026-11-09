# Score/accuracy improvement levers — Laya line (2026-09-23, net-researched during E-66)

User request: "search for how to enhance/improve the score/acc from the net all possible
way". Protocol notes honored: every lever below is validated on the FULL multi-metric
panel (acc, soft_acc, Brier, ECE, perm agreement, phishing AUROC+raw+Platt, probe suite,
AG News/emotion transfer) — never a single metric (E-62 proved acc hides calibration rot;
E-65 proved acc hides order bias; phishing raw acc hid ranking signal). All training
logs stream to bench_log.jsonl + TensorBoard + console Tee for post-hoc analysis.

Measured gaps being attacked: typed 0.5185 (vs L2 0.6585, Laya 0.766, Jev 0.727);
perm agreement 0.185 (bar 0.90); phishing AUROC 0.649 (Laya 0.678, Jev 0.689); AG News
0.245 (chance); emotion 0.315.

## P1 — Distill from our own incumbent (highest confidence for the G2 gap)
- Evidence: MiniLMv2 distillation gains at small scale ([ACL 2021 findings](https://aclanthology.org/2021.findings-acl.188.pdf)); "plan distillation early" for small specialized LMs ([arXiv:2402.01093](https://arxiv.org/html/2402.01093v2)).
- Mechanism: L2 (0.6585) already runs on our infra — dump its soft distributions over mixture+typed TRAIN and train L3's head+encoder on them (the Laya head already trains soft-CE on gold distributions, so the loss path exists).
- Cost: one L2 inference pass (~minutes) + one stage-B rerun (~15 min). Attacks: G2, soft_acc, Brier.

## P2 — Scale pretraining tokens 260M → 600M-1B (DAPT evidence)
- Evidence: 125M model continued-pretrained on 400M tokens gained MMLU +8.1 / HellaSwag +7.6, with 1B better still and forgetting-mitigation analyzed ([arXiv:2504.09687](https://arxiv.org/abs/2504.09687)).
- Mechanism: extend E-65 pretraining (same corpus, second epoch + filler), keep replay into stage A. Cost: ~2-5 h GPU. Attacks: G1 ceiling (0.6625), G2, G4 margin.

## P3 — Selection-bias calibration at EVAL (G3, zero training)
- Evidence: MCQA selection-bias calibration ([ACL 2025 long.162](https://aclanthology.org/2025.acl-long.162/)); order sensitivity ([NAACL 2024 findings](https://aclanthology.org/2024.findings-naacl.130.pdf)); PMI/surface-form debiasing ([Holtzman 2021](https://ar5iv.labs.arxiv.org/html/2104.08315)); caveat — validate on held-out, probability fixes can backfire ([arXiv:2305.14596](https://ar5iv.labs.arxiv.org/html/2305.14596)).
- Mechanism: permutation-averaged scoring (score each item under several option orders and average — literally "multi-eval, never trust one view") or per-position prior correction fitted on TRAIN only.
- Cost: eval-side only, minutes. Attacks: G3 (0.185), likely G2 too (bias removal helps acc).

## P4 — Fine-tune recipe upgrades (G2, cheap)
- Evidence: few-sample fine-tuning levers — re-init top layers / layer-wise LR decay / longer warmup ([arXiv:2006.05987](https://arxiv.org/pdf/2006.05987v3.pdf)); warm-started runs need LR reset, not continuation ([NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf)).
- Mechanism: on E-66/E-67 stage B: reset LR schedule on resume, 6-8 epochs with early stop on a TRAIN-heldout slice (never the test), optional EMA of weights.
- Cost: minutes. Note: E-65 G2 was still rising monotonically at epoch 4 — stage B is under-trained, this is the cheapest G2 lever.

## P5 — Continuous training over rebuilds
- Evidence: continuous training surpasses retraining from scratch at equal accuracy, ~2x faster ([arXiv:2502.21147](https://arxiv.org/html/2502.21147v1)).
- Mechanism: keep the E-66 loop incremental (checkpoint carry-over like today) instead of full rebuilds for every lever test.

## P6 — Ranking/contrastive head over options (G3+G2, one experiment)
- Evidence: contrastive option-interaction modeling for multi-choice ([CoLISA, ECIR 2023](https://oar.a-star.edu.sg/storage/2/2rj28m5jn0/2023-ecir-dmx-colisa.pdf)).
- Mechanism: add a ranking term (gold above each distractor by margin) to the soft-CE loss. Cost: one experiment (~15 min).

## Priority order (expected ROI / cost)
1. P3 eval-calibration (minutes, attacks the worst gap) → 2. P4 stage-B recipe (minutes) → 3. P1 distillation (~30 min) → 4. E-66 verdict informs G3 training-side → 5. P2 token scale (hours, biggest ceiling move) → 6. P6 ranking loss.
Each lever = one pre-registered experiment, one lever at a time (E-66 discipline), full metric panel, gates written before launch.
