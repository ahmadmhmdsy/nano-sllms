# UI-line U-2 plan: pipeline_v2 full-train profile, dataset audit, architecture review, SOTA path

Status: living report (updated as we discuss / research / work).
Labels use the mandatory vocabulary (PASS / FAIL / SKIPPED / BLOCKED / NEEDS_USER_DECISION).

## Executive status 2026-09-27

- Smokes launched with pre-registration before any launch (rules/experiments.md).
- Full-train and any R2 capability stays BLOCKED on an explicit user go/no-go.

| item | verdict | evidence |
|---|---|---|
| pipeline_v2 smoke on U-1 chain corpus | PASS, 13/13 R1 stages | runs/v2/ui/u1_v2_smoke_b/status.json |
| loss finite and decreasing | PASS | runs/v2/ui/u1_v2_smoke_b/train/logs/metrics/segment-000001.jsonl and segment-000030.jsonl |
| checkpoint published at final step | PASS | runs/v2/ui/u1_v2_smoke_b/train/checkpoints/step-00000030/COMPLETE.json |
| OOM events | PASS, zero emitted | runs/v2/ui/u1_v2_smoke_b/train/logs/monitor |
| first smoke attempt (bad gate floor) | FAIL, kept as evidence | runs/v2/ui/u1_v2_smoke/status.json |
| dataset quality / count | PASS main, PARTIAL realism | scratch/uidata_analysis.out |
| full-scale train launch | BLOCKED on user go/no-go | state/tasks.yaml ui/task-u1-v2-smoke |

## Measured smoke facts for sizing

- Model: llama hidden 256 / 8 layers / 8 heads / 2 kv / ctx 512 / tied embeddings, fp16 + SDPA, cuda:0 (Quadro RTX 3000 6 GiB).
- Batch 2 x accum 2. Train loss 10.45 -> 4.86 over 30 steps.
- VRAM: peak allocated ~259 MB, peak reserved ~1.07 GB, zero OOM events emitted.
- Wall clock: prepare+tokenize 800 rows (~303k tokens) < 1 s; 30 train steps ~17 s incl. warmup; whole graph ~25 s.
- tokens_per_second / peak_vram_bytes are published only via the resource_probe suite (see micro_full.yaml), not by the train stage; enable evaluation.suites: [validation, resources] in the full profile so throughput is a measured artifact.

## Dataset audit (scratch/uidata_analysis.out)

- data/u1/raw/u1_main_50k.jsonl: 50,000 rows (40,000/7,504/2,496 splits), 8 archetypes, synthetic generator, validated at birth, 0 exact dupes (trustworthy), avg 379 tok/row, est 19.0M tokens.
- data/u1/full/raw/train.jsonl: 40,000-row train slice = 15.63M packed tokens, matches data/u1/full/tokens/meta.json exactly; p99 414 tokens, max 619 -> ctx 512 comfortable.
- data/u1/raw/side/omen_alpha_all.jsonl: 50,980 rows total but 50,000 = a copy of the main corpus; only the 980 LLM-authored rows are net new (about 0.49M tokens, avg 498 tok/row via spec_to_chain), batches 001/002hard/003, all gate-clean.
- ui/data/space_bunny/production_50k/corpus.jsonl: 50,000 rows, validation_report.json PASS, 6,250 per archetype, 11 domains; near-duplicate issue documented as observational, not a hard rejection.

Sufficiency: for the registered U-1 structural-validity objective, the 50,000 main rows are enough (measured greedy 1.00 / pass@4 1.00 on 40 held-out tasks, runs/u1_full). For realism beyond the 8 archetypes it is NOT enough: no paraphrase augmentation exists; the 980-row side layer is the only naturalistic coverage and is NOT fused (rung U-1c, state/tasks.yaml).

## Architecture review — hypotheses to test, not beliefs

Per the new rule in rules/experiments.md (never believe info alone), each item must become a paired arm on real data:

1. Document-boundary bleed in concat packing: handlers.py packs documents back to back with no EOS inside one 512-token window. HYPOTHESIS (code-read). Test: paired arm inserting EOS (id 2) at document seams in the tokenize stage; gate val_loss + pass@k at the anchor; smallest change first.
2. Prompt-masked loss: the chain surface is prompt -> spec but R1 trains loss on the full text; document_masked packing already carries a per-token loss mask. HYPOTHESIS (code-read). Test: paired arm.
3. Optimizer plainness: keep AdamW + cosine. Lesson diacritizer/E-63: a winner on the sliced bed overfit to distribution on the full-data pass and regressed every external gate.
4. Post-hoc checkpoint averaging (TASKS row 67): cheap; test as an add-on arm gating on pass@4 at the anchor.
5. Vocab efficiency: 32k CodeLlama embeddings cost ~10.5M of 27.8M params (38%) on a corpus whose alphabet is tiny. Test: dedicated 2k-4k BPE tokenizer trained on the chain corpus; byte-level fallback. NOT a blocker for the first full run.

## SOTA-scaling path (pretrained teacher / adapters)

- Recursive-depth idea: the tiny ~28M recursive-model reports (arXiv 2608.29376) suggest depth has headroom even at this scale. UNTESTED here; candidate rung, must pass a paired arm.
- Constrained repair: post-decode repair via the existing codec + validator (round-trip asserted by tests) can lift validity without touching the model. UNTESTED; candidate rung.
- Mounting / LoRA / freeze pretrained base: fights the U-line thesis (the chain surface exists precisely because at <150M params the model is capacity-bound and should pay for real decisions, not generic language); also forces a tokenizer swap and loses the 15.6M-token packing head start. Verdict: not the next step at this scale.
- KD from u1_full (the 1.00 pass@4 teacher) into an even smaller student (e.g., 8M) is the only KD direction with a real registered teacher; R2-class, currently BLOCKED at pipeline_v2 R1's capability gate; needs an explicit user decision to open R2.

Recommended sequence, each pre-registered and tested: (a) fuse the 980 side rows (U-1c), (b) EOS-separator arm, (c) only then consider R2 escalation for the KD student.

## Mistakes / corrections I owe you (direct mandate)

- The '50,980 side rows' is misleading at first read: 50,000 of them duplicate the main corpus; only 980 are net new diversity.
- The large dup numbers in scratch/uidata_analysis.out for files whose rows carry the field text (not chain) are an artifact of my analyzer keying on a missing field and are NOT evidence; only files with a real chain field have a valid dup count.
- Zero oom_accounting events proves only that nothing was EMITTED, not that no OOM could occur; phrase future claims as 'no OOM event emitted', not 'no OOM occurred'.
- The new experiments rule above stands: any past-experience or literature claim cited in this file is a hypothesis to test, not a verdict.

## Discussion points for the user

1. Full-train target: reproduce-plus (U-1 gates + fused side data + resource_probe metrics) before any new-arch jump?
2. EOS-separator arm before or after the fusion arm?
3. Is the KD-student direction worth a user decision to open R2 capabilities, or does the chain-surface thesis keep the from-scratch pico line the bounded scope for now?


## Results (2026-09-27, all arms run for real on the user-opened GPU window)

Fixes that had to land before the arms produced trustworthy artifacts
(each verified by a unit test where applicable):

1. Corpus builder text_of used an absent "task" field -> corpora briefly had
   empty task prefixes; rebuilt with the task/body partition
   (scratch/build_arm_corpus.py, seed 270927).
2. Packer retained-tail mask truncation zeroed all loss bits for sub-block
   documents (`_plan_document_masked`; emit_mask now keeps real bits) —
   regression test test_document_masked_retained_tail_keeps_real_loss_bits.
3. The prepare stage silently dropped the DECLARED normalization (options
   never received data.normalization -> ws-collapse default) while the
   handler-level generation id certified ui_task_nl: a published-but-miscertified
   generation (now quarantined at
   data/v2/prepared/ui_chain_mask/.broken-ws-collapse-g-8fb3a7eb6e0c59af-
   renamed-20261227) had already fed poisoned run roots (u2b_/u2c_/u2d_/u2e_,
   quarantined at runs/v2/ui/.broken-partial-tokens-*). Plumber: handlers.py
   prepare options now carry "normalization"; a new mode ui_task_nl
   (slm_train/data/dedupe.py) keeps the task/chain newline so the document
   mask has a boundary; probe list extended; test
   test_ui_task_nl_keeps_the_task_chain_newline.

Fresh run roots and gates (40 tasks, greedy + 4x temp 0.8 / top_p 0.95):

| arm | run root | internal val_loss (basis) | gate greedy | gate pass@4 | gate artifact |
|---|---|---|---|---|---|
| base (concat) | runs/v2/ui/u2g_arm_base | 5.476 (all tokens) | 0/40 | 0/40 | runs/v2/ui/u2_arms/gates/baseu.json |
| eos (concat + '</s>' seams) | runs/v2/ui/u2g_arm_eos | 4.190 (all tokens) | 24/40 | 35/40 | runs/v2/ui/u2_arms/gates/eosu.json |
| mask (document_masked) | runs/v2/ui/u2f_arm_mask | 0.608 (chain tokens only, NOT comparable) | 0/40 | 0/40 | runs/v2/ui/u2_arms/gates/mask.json |
| fuse (concat + 980 side rows) | runs/v2/ui/u2g_arm_fuse | 1.756 (all tokens) | 0/40 | 0/40 | runs/v2/ui/u2_arms/gates/fuseu.json |
| combo (fused + eos + masked) | runs/v2/ui/u2f_arm_combo | 1.879 (chain tokens only) | 13/40 | 14/40 | runs/v2/ui/u2_arms/gates/combo.json |
| eos + 2-ckpt average | runs/v2/ui/u2g_arm_eos/train/checkpoints_avg | - | 40/40 | 40/40 | runs/v2/ui/u2_arms/gates/eosu_avg.json |
| combo + 2-ckpt average | runs/v2/ui/u2f_arm_combo/train/checkpoints_avg | - | 40/40 | 40/40 | runs/v2/ui/u2_arms/gates/combo_avg.json |

Mask-bits integrity: the final mask generation
data/v2/tokens/81a0ca996565b74d has 391 ones in shard0's 512 tokens
(scratch/check_gen_masks.py), i.e. the TASK prefix really is masked off and
the chain tokens carry loss.

Verdicts against the pre-registered questions:

- Q1 EOS: decisive YES at this scale. The literal `</s>` seam is the only
  single arm that produced valid generations alone; obsession-level jump
  0/40 -> 35/40 pass@4 with the same skeleton and step budget.
- Q2 prompt-mask: the mask trains (0.608 masked-token loss) and the mask
  artifacts are correct, but generation alone never satisfies the validator
  at 600 steps; masked-basis val_loss is NOT comparable and must be read
  with pass@k (exactly the pre-registered caution).
- Q3 side-data fusion: lowers the concat val_loss (5.48 -> 1.76) but does
  NOT convert into valid@4 (0/40): loss is not the gate.
- Q4 combination: eos+mask (14/40) is WORSE than eos alone (35/40) at the
  same 600-step budget: real interference, not an additive win.
- Q5 checkpoint averaging: post-hoc average of the two surviving checkpoints
  (500/600) takes eos 35/40 -> 40/40 INCLUDING greedy 40/40 and combo
  14/40 -> 40/40. The largest single effect measured in this whole task.

R2 recommendation (user decision): train eos-arm style corpora (eos seams,
possibly + fused side rows) as the base recipe, adopt checkpoint averaging in
train policy, keep document_masked OFF until max_steps grows (it needs more
steps: the combo at 600 steps lags its components' averages). Real full-scale
pilot/target runs remain the user-gated next window.


## R2 adoption results (2026-09-27, second session window)

Adopted and verified END-TO-END through the pipeline (not scratch):

- TRAIN POLICY: train.checkpoint_averaging = off | final_n | progressive
  and checkpoint_averaging_n (1..8) are now first-class config fields
  (pipeline_v2/slm_train/config/models.py). final_n publishes
  train/checkpoints_avg/weight-average.pt (verified artifact: manifest
  binds every source checkpoint by step + sha256; completion marker), and
  the evaluation stages SCORE THE AVERAGED WEIGHTS
  (slm_train/training/averaging.py; handlers.py loaders wired). The publish
  is immutable and idempotent on rerun.
- eos-seam corpus recipe: ui/scripts/prepare_u1.py --eos-seam appends the
  literal '</s>' chain terminator per row (test:
  ui/tests/test_prepare_u1.py). Regeneration commands change when the flag
  is set; it is part of the recorded pilot recipe.

Real pipeline runs adopting the recipe (config u2h/u2i/u2j_eos_*.yaml):

| run | policy | internal val_loss | gate greedy | gate pass@4 | gate artifact |
|---|---|---|---|---|---|
| u2h_arm_eos_avg | final_n=2 | 4.0684 | 40/40 | 40/40 | gates/eos_pipeline_avg.json |
| u2i_arm_eos_avg5 | final_n=5 | - | 40/40 | 40/40 | gates/eos_avg5.json |
| u2j_arm_eos_progavg | progressive n2 (applied at steps 200..600) | 3.9345 | 34/40 | 39/40 | gates/eos_progavg.json |

Verdict on the user-directed averaging questions: KEEP-last-5 with a final
average of n=5 matches n=2 (both 40/40); PROGRESSIVE average-then-continue
(also applied 5 times during training, evidence
u2j_arm_eos_progavg/train/progressive_averaging.jsonl) is CLOSE but not
better (39/40, greedy 0.85 vs 1.0): the running average slightly trades
greedy sharpness for smoothness on this 600-step bed. ADOPTED BASE RECIPE:
eos seams + checkpoint_averaging=final_n (n=2, keep_last 5 is harmless);
progressive stays available but off the base recipe.

Dataset adequacy (user-directed, written while u2j trained):
research/ui_dataset_adequacy_r2.md — 15.6M train tokens, 100% chain
contract validity, 17 component types, ZERO tokenizer unknowns; the weak
rung for a near-SOTA claim is diversity (36 unique task phrasings / 19
unique titles across 40000 rows), so pass@k here is family-fit, not
natural-language generalization; a diversity escalation is pre-registered
before any full SOTA claim.


## U-3 diversity rung: two-column pass@k verdict (2026-09-27, CLOSED)

Pre-registration (user-directed 2026-09-27): diversity remedies are worth trying; LLM
content generated ONLY by opencode3/space-bunny-free subagent models; two-column
pass@k reporting mandatory (family-split vs novel-phrase split); only the novel
column may back a generalization claim.

### Data created (all CPU-side, regeneration by builder commands)
- u3 corpus: data/u3/raw/u3_diverse_50k.jsonl - 40000 rows, 0 gen fails
  (ui/src/generator.py per-arch 5000, seeded, value/label/diversity pools widened).
- Prepared pipeline input: data/u3/prepared/raw/ train 32000 / val 1500 / test 500
  (ui_task_nl normalization, eos-seam recipe, dataset_id ui_chain_eos).
- LLM novel phrasings: data/u3/novel/phrasings_a..h.jsonl = 580 + 800 = 1380 rows
  across 8 archetypes (a-d 2026-09-27 earlier, e-h this slice: gym/college/
  recipes/flashcards, bookshop/vet/gallery/film fest, music/photography/tutoring/
  hardware, Airbnb/renovation/nursery/daycare). Independent validation PASS: 0 JSON
  errors, per-arch 100 each in e-h, 0 exact dups; ONE near-dup pair inside e-h
  (shared "id, customer, total and status" column skeleton) - observational.
- Novel gate slices: data/u3/novel/gate_novel_a.jsonl / gate_novel_b.jsonl (40 rows
  each, spec_to_chain, 0 invalid) and data/u3/novel/novel_gate_tasks_eh.jsonl
  (40 tasks, 5 per archetype, rotated across e-h) with scratch/gate_u3_novel.py.

### Run (adopted recipe: eos seams + final_n=2 checkpoint averaging)
- Config: pipeline_v2/configs/experiments/u3_arm_eos_avg.yaml; run root
  runs/v2/ui/u3_arm_eos_avg; throttle ceiling 0.55; single GPU job, no co-run.
- INCIDENT, kept as evidence: the first launch died silently after step 493
  (last metric segment 2026-09-27T06:37:50Z, no monitor anomaly, no OOM event,
  no train COMPLETE). Re-run of the EXACT original command auto-resumed at
  step 400; final loss 0.4858 @ step 600; all 13 R1 stages PASS
  (runs/v2/ui/u3_arm_eos_avg/status.json).
- Checkpoints: best_val_loss 4.1156 @ step-00000400; final step-00000600;
  weight-average.pt = average steps {500, 600} (binds both by sha256,
  train/checkpoints_avg/manifest.json).
- Internal eval: val_loss 4.2477, ppl 69.95
  (internal_evaluation/evaluations/COMPLETE.json; concat all-token basis, NOT the
  mask basis of the masked arms).

### Two-column gate (40 tasks each, greedy + 4x temp 0.8 / top_p 0.95, avg weights)
| column | greedy_valid | pass@4 | artifact |
|---|---|---|---|
| family (data/u1/full/raw/val.jsonl slice) | 38/40 (0.95) | 39/40 (0.975) | runs/v2/ui/u2_arms/gates/u3_eos_avg+avg.json |
| novel phrases (data/u3/novel/novel_gate_tasks_eh.jsonl) | 17/40 (0.425) | 20/40 (0.50) | runs/v2/ui/u2_arms/gates/u3_eos_avg+avg_novel.json |

### Verdict: PARTIAL - structural validity holds; natural-language generalization does NOT
- FAMILY column holds near the u2 level (39/40 vs 40/40 pass@4 with a smaller
  32k train corpus at the same 600-step budget): the adopted recipe reproduces.
- NOVEL column is the decider: pass@4 only 0.50 on phrasings from domains the
  model never saw. The chain grammar transfers, the TASK-TO-ARCHETYPE map does
  not. NO near-SOTA generalization claim is available from this line; any pass@k
  number without the novel column is family-fit only (exactly as
  research/ui_dataset_adequacy_r2.md pre-registered).
- Diversity remedy DOWNSTREAM (u3 corpus value mix) is working as registered
  (0 fails, 100% validated-at-birth), but phrasing-text diversity is the remaining
  gap. Follow-ups, each needs its own pre-register: (a) mix the 1380 LLM phrasing
  rows INTO the training corpus and re-gate the novel column (currently only 80
  gate slices were built, none fused into training); (b) semantic-follow gate
  (ui/task-u1b) to measure the task->chain semantic headroom separately from
  validity; (c) more novel-phrasing rounds from space-bunny-free subagents fused
  with the same two-column gate before any full train.
