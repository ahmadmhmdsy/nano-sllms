# E-19 external-model bake-off — bench results (DER %, lower is better)

All numbers measured with our own `eval.py compare` on OUR gate slices.
Gate refs come from the stage2b2500 gate-probe refs (bare = marks stripped).
Gold = runs/diac/stage2b2500/final (fadel 33.20 / sadeed25 45.68 / wn2014 48.99 / abdou HO 41.64).

USER RULE (E-19 registration): a model may generate data for gold only if
(a) it beats gold OVERALL by >5 DER absolute, or
(b) DOMAIN specialist: wins its domain by margin AND stays near-gold on all other gates.

## Results

| candidate | license | hardware | fadel | sadeed25 | wn2014 | verdict vs rule |
|---|---|---|---|---|---|---|
| gold stage2b2500 | - | - | 33.20 | 45.68 | 48.99 | reference |
| Z-Mahmood BiLSTM+attention (model-only, cache bypassed) | MIT | CPU | **4.35** | **18.85** | **29.73** | QUALIFIES (a): beats gold by 19-29 DER on every gate |
| Etherll/Tashkeel-350M-v2 (granitemoehybrid) | Apache-2.0 | GPU batch=1 | 49.7 (20-line probe) | - | 97.0 (60-line slice) | FAILED: repetition loops on out-of-distribution text; mamba2 chunk-scan OOMs at batch>=2 |
| flokymind/mishkala (mamba+transformer+CRF) | Apache-2.0 | CPU | 39.37 | 35.79 | 50.24 | Rule (b) profile only as sadeed-domain specialist (wins sadeed by 9.9); fadel +6.2 worse than gold - borderline |
| basharalrfooh/Fine-Tashkeel (T5-large enc-dec) | repo (FA checkpoints) | GPU batch 4-8 | 91.60 (256-line slice), preservation 10% | 70.55 (500-line slice) | 37.62 (full) | FAILS (rule b): wn2014-only; other gates catastrophically worse than gold |
| NAMAA/Cohere-Speech-Tashkeel-2B | - | - | - | - | - | EXCLUDED by user (audio modality) |
| QCRI advancing-arabic-diacritization (EMNLP 2025) | - | - | - | - | - | SKIPPED: repo is datasets-only fork, no released weights at time of bench |

Runtimes: Etherll ~6.5 s/line (batch 1 forced); T5 Fine-Tashkeel ~4.7 s/line (batch 8, WDDM 5.8 GB cap); mishkala ~6-9 lines/s CPU; Z-Mahmood ~5-8 lines/s CPU.

## Caveats / contamination risk

- Z-Mahmood ships a 29 MB sentence cache trained on Tashkeela-family data; we bypassed it (model-only), but its TRAINING set may overlap our gate reference sentences (fadel/sadeed/wn2014 all public). Treat 4.35/18.85/29.73 as an UPPER BOUND on generalizable quality until a contamination check (n-gram overlap between training cache and our gates) is run.
- Our gate refs are the SAME refs used to score gold, so head-to-head is apples-to-apples.
- Etherll numbers used greedy decoding, fp16, card-exact chat prompt; mismatched-base errors and loops are the failure mode.

## Raw artifacts

- Gate inputs: `models/e19/inputs/*.{bare,ref}.txt`
- Predictions: `models/e19/{gate}.{model}.pred.txt`
- Harnesses: `diacritizer/scripts/e19_infer_batch.py` (causal), `e19_infer_t5.py` (seq2seq), inline runs for CPU models.
