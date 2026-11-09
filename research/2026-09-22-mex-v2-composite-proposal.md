# MEX v2 Composite — Design Proposal (user-facing draft, 2026-09-22)

Status: DRAFT for user review - not a launch pre-register. Written after the E-60b rung-B retirement (frozen-growth #1, co-train divergence #2 per R5) and after the 2026-09-22 agent reflection (docs/plans/2026-09-22-agent-reflection-lessons.md).

## 0. The question the user posed
Combine the GOLD models (current standing best), the ladder discipline, the expert towers, the shared KB, and the other proven techniques into a SECOND-VERSION composite whose result is solid on SEVERAL metrics over SEVERAL test sets - never one number. Rank the ingredients: MUST-have (base identity) vs NICE-to-have (confirmed but not guaranteed).

## 1. Inventory of what is already PROVEN (foundations we do not re-litigate)

| Asset | Status | Evidence |
|---|---|---|
| Gold diacritizer-line models (external gates) | standing best externally | bench rows: gold DER/WER dominate; dia2 composite far behind externally |
| mu3 12L joint trunk + mu2/x towers + mark head + router (~19.5M mounted) | E-53 capability card; dia composed 0.7943 in-domain | E-41..E-53 chain |
| dia2d_scale trunk (~50M) + dia2_wide mounts (canonical composite) | strongest dia2 composite | E-54d PASS: 'wider trunk not saturated; scale replay buys a large composed gain' |
| Muon optimizer | REJECTED for settled-LoRA work (E-54b honest negative: AdamW arm NEW dia standing best) | do not re-adopt without a new hypothesis |
| Context extension / shared-KV bridging | REJECTED as dia levers at this scale (E-54c) | does not re-enter v2 unless re-gated |
| Expert KD harness (c12 kd.py pattern + mex KD scripts) | exists and is proven plumbing | E-60b B2 arm design was ready |
| e23a_gold_v3q pseudo-label teachers | the shared-KB ingredient; provenance caveat: e23a_gold_v3q was REGRESSED vs production gold as a standalone (mean 45.59 vs 44.14, wn24 collapse from QCRI label shift) - usable as teacher for IN-DOMAIN replay rows, but stage2final is the safer teacher candidate; cross-agreement filter required either way | HANDOFF 2026-09-17, E-23a row |
| v3q token pack | REBUILT + VERIFIED this session (byte-equivalent val, exact stats) | scratch/rebuild_v3q_report_2026-09-22.md |
| Full metric profile bench (strict DER, per-class 15-way, hit/wrong/missed/extra, word buckets, der_collapse, contrastive_lift, per-prompt CSV) | standing output of every gate eval (E-55c, user-set policy) | the 'several metrics' the user wants, ALREADY a standing rule |
| VRAM-only residency guard + WDDM spill watchdog | adopted this session (M8) | train_e61b.py guard + scratch/gpu_shared_watch.py |

The tie-breaking fact from the whole chain: **the dia2 mini composite lost to gold mostly on KNOWLEDGE-shaped failures (E-55c mid/last mark confusion), not capacity-shaped ones, and E-56 proved data parity alone does NOT close that gap. E-57 proved replay mixing alone does not either. What actually worked historically was CO-TRAINING (A/E-60a: co-trained-from-step-0 recipe builds real mark knowledge in-domain) and COMPOSED scale replay on an UN-frozen joint trunk (E-54d).**

## 2. v2 spine: ONE trunk, START JOINT, COMPOSE, then GROW BY COPIES (not by zero-insertion)

The two structural retirements say: depth-GROWTH by insertion is dead. Growth in v2 = **launch the next rung as a FRESH DEEPER model initialized by composition (init-order mounting) from a proven smaller checkpoint where the new rung is CO-TRAINED from step 0 with the SAME joint objective** - not weight-insertion into a warmed transformer. Concretely:

### Rung V2-1: 'KD-plus-attempt-3' (MUST-have) - single variable: the KD term
- Control arm = attempt-3's RECORDED run (markpos 0.6496 @3000, per-class vector, external profile) - no re-run needed, it IS the anchor-set control.
- Treatment arm = EXACTLY the attempt-3 recipe (fresh head on attempt-2 trunk, logit-adjusted CE tau=1, lr 1e-4 / head 6e-4, 3000 steps, bs 512-equiv) PLUS the GDP KD term: soft mark-class targets from the chosen teacher on replay rows. One variable, R5-clean, ~2h GPU.
- THE ONE UNFREEZING CHANGE: initialize from A3 final (like B1 did) BUT train from a cosine schedule whose warmup tops at lr 1e-4 (half) AND extends the linear-decay phase: B1' divergence is an optimizer-stability event, not an architecture event. This is ONE variable vs A3 (init position), R5-clean.
- Expert term: B2-style KD arm from e23a_gold_v3q pseudo-labels as soft mark-class targets on replay rows (the A/B distill-vs-alone rule from R6): measured, not decorative. Either it beats the arm by 2% mark-acc on the 4-gate mean or it gets dropped.
- Gates (same pre-registered style): 4-gate mean mark-acc >= 0.65, contrastive lift >= 0.20, in-domain markpos >= 0.70, ALL BY THE FULL PROFILE, never by a single number.

### Rung V2-2: 'replay + KD on the standing composite' (MUST-have if V2-1 passes)
- Control arm = E-54d's RECORDED run (the composite with scale replay, the only large composed gain).
- Treatment arm = EXACTLY that recipe PLUS the v2-1-proven KD term (only runs if V2-1 KD beats control by >= 2% externally). One variable vs E-54d; the wider trunk still has unsaturated capacity per E-54d's own conclusion.

### Rung V2-3: multi-trunk composition (NICE-to-have, only if V2-2 floors)
- NOT transplant towers into a settled trunk (E-44 gate (c) FAIL). Instead: train mu3-style towers WITH shared co-trained KV from a settled mu3 trunk, then mount, per E-53's capability-card path.
- Shared KB readout (E-54c) is rejected as a primary lever; keep only as a benchmark for composition efficiency.

## 3. The MULTI-METRIC frame is a MUST-HAVE, pre-registered for every rung

Standing: every gate emits the FULL BILL of health in one artifact (E-55c pattern), on SEVERAL test sets, not one:

| Metric family | What it detects |
|---|---|
| External gates (fadel, sadeed, wn24, wn14) mark-acc + DER | knowledge-transfer; the gold standard demanded by the gates |
| In-domain val markpos (same-index head-through protocol, ANCHORED to A3 0.6496 and B1 0.0889 every run) | in-domain learning headroom |
| Per-class 15-way accuracy (cls vector) + prior | per-class collapse (shadda-class 0.02 was the FIRST Canary in E-60a) |
| mark hit/wrong/missed/extra | whether prediction adds spuriously-wrong marks |
| word_ax / word_partial50 | word-level usability, detects mid/last failures E-55 found |
| der_collapse + zero_mark_baseline_der + contrastive_lift | the R1..R4 sanity: is composite beating dead baselines? |
| Per-prompt right/wrong CSVs | deployed debugging; the user's explicit ask |

Anchored eval invariant (from lesson L1): any new metric or test set is first validated by scoring an existing final model against its recorded anchor value before it is allowed to gate anything.

## 4. Technical levers, ranked

MUST (validated by our chain or the standing policy):
- M1 joint char+mark co-training from step 0 (E-60a attempt-3 evidence);
- M2 logit-adjusted CE, tau=1, with weighted-CE variant because E-60 attempt-2 vs 3 established the tau=1 form was what revived dead classes;
- M3 full metric profile per gate (E-55c);
- M4 KD from e23a_gold_v3q pseudo-labels as the only sanctioned expert signal (R6/R3);
- M5 scale-replay co-training on un-frozen joint trunks (E-54d);
- M6 residency guard + RAM-bounded data packers (this session's infrastructure fix);
- M7 pre-registered one-variable rungs with structural-retirement rules (R5) - already honored, keep honoring;
- M8 V2-1 un-frozen fresh-deep CO-TRAIN from a PRE-WARMED init with the attempt-3 recipe (frozen-insertion is retired).

NICE (validated direction, needs its own rung, not base assumptions):
- N1 Muon as a SECOND optimizer arm for the joint-fresh path only (not settled-LoRA; E-54b rejection precedent is for settled work);
- N2 weight copies of the A3 trunk while training THE COPY (skip init-order zero-insertion entirely - B1/B1' agent lesson; only feasible if the trunk pattern is co-trained);
- N3 per-class prior re-estimation online during training with a small momentum - the mark prior shift mid-run could destabilize;
- N4 curriculum ordering across the token pack (unverified in this repo, cite as a hypothesis only).

EXPLICITLY OUT unless re-gated: shared-KV towers (E-54c), context extension (E-54c), settled-trunk tower transplants (E-44c), the retired zero-insert growth recipe (R5).

## 5. What makes v2 'solid result' instead of 'better single number'

1. Every rung pre-registers BOTH the primary metric AND the metric profile acceptance bands; verdicts read on the BAND, never on one number. So a 'does it actually improve' question cannot be gamed by a lucky gate.
2. The same architecture must hold across 4 gate corpora + the in-domain val to be called improved; a single-corpus gain gets the label 'must-have if it survives all metric families', not 'PASS'.
3. An INSTRUMENT first, always: any new metric or fixture passes the anchor-check before it can veto.
4. The B2 KD A/B arm is MANDATORY for any expert/KB claim (R6): no 'experts help' claim without the measured distill-vs-alone delta.
5. Frame health (VRAM residency, data-regen timestamp, seed chain) is a RUNG-GATE prerequisite visible in the ledger row, so 'v2b-class' silent rot cannot recur (M4/L4).

## 6. Costs and risks to state up front
- GPU-time: V2-1 ~1.5-2.5 h at measured ~830 tok/s on the 6GB card; V2-2 ~2 h; external 4-gate bench ~30 min per profiled run.
- Risk: divergence reopened if LR is mis-picked for the V2-1 form; mitigated by running the smaller 256x2 gradient-equivalent shape under cap (already proven this session) and by the new divergence runtime gate.
- Risk: over-claiming v2 if external gates do NOT move the composite's profile beyond gold. The honest pre-registered verdict criteria keep the chain's file honest the way the last 48h did.

## 7. What I need from the user to commit
1. Approve the v2 spine (Rung V2-1 and V2-2), or tell me which ingredients to change BEFORE the full plan doc is formalized.
2. Confirm which gold model is canonical for v2 teachers (e23a_gold_v3q; if there's a better successor, name it).
3. Confirm the kill criterion for V2-1 divergence runtime gate: val eval at the NEXT 500-step eval after monotone-rise detection instead of completing the log.
4. Whether to keep checkpoints of retired runs (B1-prime checkpoint-01000) or archive them off the GPU budget.
## 8. FINAL REVIEW PASS (2026-09-22, pre-launch corrections)

Mistake 1 ( MATERIAL ): teacher naming. The E-23a row (HANDOFF 2026-09-17) records e23a_gold_v3q as REGRESSED / NOT adopted as production gold (mean 45.59 vs stage2f 44.14; wn2024 collapse attributed to QCRI label distribution shift on the saturated reader). The v2 KD teacher MUST therefore be announced at pre-register as EITHER (a) stage2final (production gold, cleaner on wn24) OR (b) e23a_gold_v3q SOLELY as in-domain-rows teacher with the wn24/QCRI confound removed from its label surface (cross-agreement filter below). NOT两者 mixed. This is the user's one remaining pre-register decision (proposed: stage2final).

Omission 2: multi-teacher AGREEMENT FILTER. The E-23 S5 validator (e23_crossagree.py: gold + ZM per-line word-map agreement, flag_below threshold) is proven plumbing and was not in the draft. v2 KD labels are ADMITTED ONLY on teacher-agreement (>= threshold per line/word); rescored-against-disagreement bucket tracks KB quality over time. KD rows from teachers that wiki the end-test distribution shift stay quarantined (the lesson of the wn24 collapse).

Omission 3: REGULAR EVAL CADENCE (user-approved requirement, explicit now):
- In-run cadence: in-domain multi-metric (markpos + per-class cls vector + val_ce + mark hit/wrong/missed/extra) at EVERY eval step (500) on the REBUILT v3q val - NO GPU co-run conflict since it's in-process.
- Rung cadence: full external 4-gate profile (strict DER, word buckets, der_collapse, contrastive_lift, per-prompt CSV) at ANY save step known >= 0.6 * max_steps AND at close - never from a single saved hidden state mid-warm.
- Anchor rule: the in-domain metric line is asserted against A3 0.6496 / B1 0.0889 reusable anchors within tolerance at the start of each rung before external benches are allowed to gate.

Omission 4: seed/env discipline for A/B arms: R5 single-variable arms MATCH control on seed/batch/accum (this session's verified 256x2 == 512x1 equivalence in fill-length rows makes that safe; restate as a rule for any future variable-length pack).

Confirmed-good (nothing else missed in draft): VRAM-residency prerequisite, RAM-bounded packers, R5 retirement closure for the insert recipe, ending the One-of-many 'improved' definition. Quality verdict: the spine holds; correct the teacher naming and pre-register with cross-agree filtering.