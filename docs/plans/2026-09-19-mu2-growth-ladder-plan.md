# μ2 Growth Ladder — Progressive Composition of the Micro-Expert Line (ME Stage 2)

Date: 2026-09-19 · Owner: this agent · Prior: MU1_REPORT.md (mu1 closed), MU0_REPORT.md (mu0c closed)
USER DECISIONS (2026-09-19 verbatim basis):
- "proceed to next stage (benefit from the past result"
- ladder idea: x1,x2 tiny students -> x3 bigger -> x4 = x1+x2 + new layer for a
  related task (LoRA/transfer/router/MoE..) -> x5 = x4+x3+x1 + layer ... growing up.
- gated choice: A (warm-start + LoRA frozen trunk) first, then B (Net2Net widening),
  then C (progressive lateral blocks): "A+B+C worth try, start with your RECOMMENDATION".

## Core rule (from the E-27 post-mortem)
Composition happens through INITIALIZATION ORDER, never by post-hoc merging of
independently-trained branches. Every rung starts from the previous rung's
weights (one basin by construction). Post-hoc merge arms (mu1 A) are closed as
verified failures at this scale; routing/distill arms recorded; the ladder is
the sanctioned path.

## Ladder (target: sentence-level Arabic diacritization, DER metric)
| rung | task | mechanism | data source | param budget |
|---|---|---|---|---|
| G1 | char-LM pretrain on VOCALIZED Arabic text (alphabet+word priors; user idea ids 1+3) | from scratch, expert-arch (2L/hidden80/ffn320) | raw vocalized text from data/diac/raw (wikinews+fadel+abdou; READ-ONLY from D-line staging, copied into data/mex/mu2/g1; license pre-check per corpus) | ~200K |
| G2 | letter-mask fill-in (user id 4) | G1 warm-start + LoRA adapter on frozen trunk, <=12K steps | same corpus, mask transformation | +LoRA deltas only |
| G3 | Net2Net function-preserving widening (hidden 80->160, 2L->4L) + word-mask fill-in (user id 5) | B-mech rung (USER pre-approved) | same + word-mask | ~800K |
| G4 | mark-selection head: per-letter {haraka|sukun|none} + DER metric (user ids 2+6 collapsed) | G3 warm-start (+LoRA; C lateral blocks stay USER-GATED) | x1 wordlist train + der-lite metric vs known val | user target band |

Per-rung gate (frozen here, pre-registration rows E-30..E-33 land BEFORE each rung's results):
1. rung's own task metric beats its measured trivial baseline with Wilson 95% CI;
2. RETENTION: rung's held-out loss on all ancestor tasks within +5% of the ancestor's
   own best (replay mix 10-20% continues at every rung, no exceptions);
3. DER-lite reported for every rung from G2 on (per-char mark accuracy);
4. row-first commit discipline per rung.

Risks: ladder reaggravates only stage-4 gating; sadeed_tashkeela license
(research OK, commercial = USER FLAG); wikinews/fadel fine for research.
Wall-clock: 12K steps/rung ~17min GPU each; tape all four ~70min plus evals.

Out of scope: nothing in M3/D-line src/configs; no edits to data/diac writes.
