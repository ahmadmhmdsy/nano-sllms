# D-line (diacritizer) — refuted arms, tracks and techniques

Archived 2026-10-06 from the verdict audit (`research/verdict_audit.md`,
`research/verdict_audit/parts/`). **Read-only history.** Current truth is
`research/experiments.yaml`.

Two failure kinds are separated on purpose, because they mean different things:
**REFUTED** (the lever genuinely does not work here) vs **SUPERSEDED** (it worked
but a later arm beat it) vs **PROCESS** (the result was mis-measured or
mis-recorded, so the verdict was wrong in one direction or the other).

---

## A. REFUTED levers — do not refight without new evidence

| # | what was tried | measured | lesson | origin |
|---|---|---|---|---|
| 1 | Skew-KL (alpha-SKL) distillation | +2.29 % over plain-KD; led only in a 200-step window | Use plain-KD; fp16 grad norms were healthy, so the skew floor was never needed | `research/EXPERIMENTS.md` E-03 |
| 2 | Mount-fill KD with a hard teacher-off anneal | collapse, eval flat ~6.5, grad-norm median 1454 | Make the teacher-off transition an lr-warmed phase, not a hard shutoff | E-11 |
| 3 | Stage-1 warm start WITHOUT reset-last-2 (arm A) | Fadel 41.6 / Sadeed 54.5 / WN24 60.9 / WN14 54.1 — worse than the from-scratch specialist on ALL FOUR gates | A stage-1 LM is not automatically a good stage-2 init; the reset trick was doing real work | E-14 |
| 4 | Whole-corpus v4 init for the final model | Held-out abdou 49.4 DER vs stage2b2500 41.6 = **−7.8 pp worse** | More init data is a statistical wash on gates and clearly worse on fresh held-out. Init scale is NOT the lever — 3rd confirmation | E-17 |
| 5 | Postproc: dedup-marks + legality-repair | outputs BYTE-IDENTICAL to baseline; 0 doubled marks in real predictions | There was no defect to repair. Postproc was dropped from the model pipeline | E-18 |
| 6 | ZM BiLSTM+attention AS OUR ARCHITECTURE | mean 55.83 vs arm C 48.42 = **+7.41 worse** | A stronger external model does not transfer its architecture. Char transformer stays | `research/e19_bakeoff/E23D_BILSTM_AB.md` |
| 7 | Classical (tashkeela-family) data expansion, +1.2 M gate-deduped windows | mean 48.72 vs arm C 48.42 — worse on EVERY gate | Classical gold expansion measured ≈ zero. The headroom is modern-MSA labels | `research/e19_bakeoff/E23B_CLASSICAL_S2.md` |
| 8 | ZM distillation into the micro model (315 k windows) | mean 52.55 vs arm C 48.42 = **+4.13 worse** | A NEW label style hurts a saturated pool REGARDLESS of teacher quality. Supplement quantity helped; a new label style did not | `research/e19_bakeoff/E23C_ZM_DISTILL.md` |
| 9 | QCRI EMNLP-2025 supplement | +2.7 micro (weak) | Below the adoption bar | `research/e19_bakeoff/E22_QCRI_WEAK.md` |
| 10 | Label smoothing on the full-data run | 0.5810 vs S4 0.3996 — **harmful** | LS was tuned on a different regime and does not transfer to full data | `runs/diac/stage3best_ls` |
| 11 | Scaling params alone (b65 76 M, v2d28 30 M) | both arch axes together ≈ **1/10** of the SOTA gap; 2.5× params bought 1–4.6 DER pp | Data breadth / domain routing >> depth-per-param > raw params | E-13 |
| 12 | MTP-style aux next-token-2 head | negative at every aux weight 0.02–0.3 | Retry only with a decoupled aux LR + head warmup at longer budget | E-42 (core) |

## B. SUPERSEDED — kept for the record, not wrong

| # | arm | result | why it is here | origin |
|---|---|---|---|---|
| 13 | E-068 C1 (ctx 128, full v3) | 0.382541 @90000 — **was the project best** | Demoted when E-070 C4b reached 0.3748 and E-071 voting reached 0.3697. Still the diagnostic anchor: it proved the warm start was the real lever | `runs/diac/e67_c1_ft64_fullv3` |
| 14 | E-069 C4 (ctx 256, 60 k steps) | 0.387245 @60000; FAIL on its own rule trigger (−3.4…−4.5 % at matched tokens) | Superseded by C4b. Its "under-trained" diagnosis was VINDICATED, but "budget was the whole story" was OVERSTATED — past step 15000 the extra 45 k steps bought nothing | `runs/diac/e69_c4_ctx256` |
| 15 | E-066 R1 (long-budget arm) | 0.4205 vs F2 0.3974 — FAIL on peak | Killed at step 109500, not early-stopped | `runs/diac/e66_r1` |
| 16 | E-065 F2 | 0.3974, but from RANDOM init while its config claimed a warm start | The false result that motivated C1. Kept as the canonical example of a silently-defaulted lever | `runs/diac/e65_f2` |

## C. PROCESS failures — the verdict itself was wrong

These are the most valuable entries: the experiment did not fail, our *reading*
of it did. Both directions were found.

| # | defect | direction | what it cost | origin |
|---|---|---|---|---|
| 17 | Registered metric DIRECTION inverted in a resolved config | false reading | Any verdict derived from it is sign-flipped | `runs/v2/core/micro_full/validate/config/resolved.json` |
| 18 | E-19 Fine-Tashkeel scored on a slice, not the gate | **false FAIL** | A workable model was nearly discarded | E-19 |
| 19 | E-23a gold wn2024 misread as 48.65 (actual 57.58) | **false NOT-ADOPTED** | A data arm was rejected on a wrong baseline | E-23a |
| 20 | False PASSes at E-38a, E-45, E-54c/d, E-63c, E-63s S3-best | **false PASS** | Wrong arms looked adoptable | ledger |
| 21 | Silent warm-start no-op (D-12) | false premise | A 10-hour run trained from random init while the config claimed otherwise | `runs/diac/stage1lm` |
| 22 | PAD-convention drift on rebuild (D-13) | incomparable losses | Fresh tokenize reports a different loss/val_loss than the whole ledger; gates unaffected | `diacritizer/scripts/tokenize_data.py` |
| 23 | Rebuild command not reproducing its own bed (D-14) | unreproducible | The bed could not be regenerated from its recorded command | `data/diac/prepared_rebuilt/` |
| 24 | Best-of-run selected over NOISY gate probes | selection risk | E-072 later tested it on held-out abdou and the ordering held — so this one was a *near miss*, not a loss | `research/verdict_audit.md` |

## D. What these add up to

- **The D-line's real levers are warm start, context length, selection discipline
  and inference-time voting — not architecture and not parameter count.**
  Failures 3, 4, 6, 11 all attack the architecture/init-scale axis and all lose.
- **Two of the three "big data" moves failed** (classical expansion, ZM
  distillation) while the *domain-specialist* move historically won big
  (Fadel 47.1 → 34.2). Data helps when it matches the target domain, not when it
  merely adds volume or a new label style.
- **A mis-read verdict is more expensive than a failed run**, because it
  propagates. That is why the audit checked BOTH directions and why
  `research/verdict_audit.md` exists.
