# E-23d - S4 BiLSTM vs our-arch equal-data A/B (arm D1)

**Date:** 2026-09-17 **Stage:** E-23 (S4, user-approved) **Status:** CLOSED - NOT adopted

## Question
E-23 plan S4: does Z-Mahmood's architecture (3x256 BiLSTM + Bahdanau
attention, MIT-rule-(a)) beat our micro char transformer at EQUAL data and
EQUAL step budget on the v3q token pool (arm C tokens)?

## Setup
- Model (arm D1): reimplementation of the ZM baseline into our pipeline I/O
  contract (diacritizer/scripts/e23_bilstm_model.py): same TK.VOCAB_SIZE=171
  tokens, our 15-class labels, per-char logits - so the gate plumbing is
  byte-identical to every other arm. 4,498,831 params.
  Embedding(128) -> BiLSTM(256x2, 3 layers, dropout 0.3) -> Bahdanau
  self-attention -> Linear(15). ZM native hyperparams: lr 1e-3, grad clip 1.0.
- Budget: EXACTLY arm C's: 2500 steps, batch 32, fp16 autocast, AdamW
  wd 0.1, single GPU.
- Tokens: data/diac/v3q (2,803,948 train rows = v3 + QCRI gatesafe).
- Gates: fadel/sadeed/wn2024/wn2014 every 500 steps, eval_der compare,
  same CSV rows as every arm (runs/diac/e23d_bilstm_lr1e-3/gate_eval.csv).

## lr behavior at equal budget (batch 32)
| lr | loss@500 | mean DER@500 |
|---|---|---|
| 4e-4 (our micro lr) | 1.03 | ~0.997 (bare-collapsed word luck) |
| 1e-4 | 1.75 | (killed; no learning) |
| **1e-3 (ZM native)** | 0.70 | 62.84 -> improving |

ZM's native lr 1e-3 is what makes the BiLSTM learn at all on this batch
size; our micro lr 4e-4 stalls it (recorded as MEMORY lesson).

## Arm D1 trajectory (fadel / sadeed / wn2024 / wn2014, % DER)
| step | fadel | sadeed | wn2024 | wn2014 | mean |
|---|---|---|---|---|---|
| 500  | 53.95 | 65.86 | 67.79 | 63.78 | 62.84 |
| 1000 | 48.61 | 60.35 | 64.81 | 59.08 | 58.21 |
| 1500 | 47.65 | 59.54 | 63.52 | 57.93 | 57.16 |
| 2000 | 45.95 | 58.08 | 63.45 | 57.02 | 56.12 |
| 2500 | 45.77 | 58.09 | 62.90 | 56.55 | **55.83** |

## Verdict (vs frozen yardsticks)
| arm | mean DER % |
|---|---|
| gold model (yardstick) | 46.37 |
| E-23a S1 gold (WIN - corrected 2026-10-06) | 45.59 |
| **micro arm C (champion, v3q)** | **48.34** |
| micro arm A (v3 only) | 51.10 |
| **arm D1 BiLSTM (this)** | **55.83** |
| E-23c arm E (v3qz, NOT adopted) | 52.55 |

BiLSTM LOSES to our micro transformer by **+7.49 mean DER** at equal data
and budget (55.83 vs 48.34), and even loses to arm A (51.10, less data).
It does beat arm E (52.55), the label-style-conflicted ZM-distill arm.
Trajectory is still improving slowly at 2500, but the gap is structural,
not a lr artifact (ZM-native lr tested).

**Decision: D-line keeps the char transformer.** Z-Mahmood's baked-off edge
on MIT rule-(a) is a data-scale + train-duration property (their 3.5 h full
pass on their pool), not an architecture to port into nano_SLMs at micro
scale. Vendored reference stays in models/e19/zmahood-src for provenance.

## Side notes
- The FIRST BiLSTM arm numbers (~0.997 mean DER) were a self-inflicted bug
  in the new probe gate (tab-joined refs double-iterated per character) and
  are VOID. Rows above come from the fixed in-process compare (same
  eval_der functions as eval.py compare; parity verified on crash files).
- Run dirs kept: runs/diac/e23d_bilstm (4e-4, void, for the record),
  runs/diac/e23d_bilstm_lr1e-3 (verdict). Trim weight checkpoints per
  post-close policy; keep best.pt / final / gate_eval.csv.
- User-gated items unchanged: S2 classical-only decision; QCRI/HBKU email.
