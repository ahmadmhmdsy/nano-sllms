# PLAN — D-line Stage 1/Stage 2 transfer: char-LM Arabic pretraining -> diacritization specialist

Date: 2026-09-13 · Status: SPEC FOR USER REVIEW (not started; GPU-days commitment = user-gated)
Owner: ARABIC-DIACRITIZATION agent. Supersedes nothing; completes the 3-axis ablation chain
(E-12/E-13: data/domain routing >> depth-per-param > raw params; ALL arch polish ~1/10 of the
SOTA gap -> transfer/pretrained-init is the frontier; SOTA systems Fine-Tashkeel(CATT etc.)
all start from pretrained weights).

## 0) The registered hypothesis (falsifiable)

H: A diacritizer warm-started from a char-LM Arabic-pretrained stage-1 model beats every
from-scratch D-line variant (best gate DERs: fadel_spec 34.2 Sadeed 51.4 WN24 62.9 WN14 57.4)
on >= 3 of the 4 gates by >= 2 pp, at the same ~30M param budget.

Secondary read: if H fails, D-line's near-term frontier stays data breadth (v3 corpus) and the
specialist-ensemble route.

## 1) Stage 1 — char-level Arabic LM pretraining

### 1.1 Corpus mix (target ~80-120M chars raw Arabic, one-time, regenerable)
Raw (undiacritized) vs diac: strip diacritics for the MARKED sources to build plain text:
- abdou_tashkeel train parquet -> strip marks (1.44M rows, modern/MSA wikipedia-ish) ~ 300M chars?
  (measure first; slice if oversized, keep source-balanced)
- sadeed_tashkeela input column (bare classical, EXACTLY the Sadeed-25 domain): 1.04M rows
- qcri wiki (strip marks from the diac jsonl)
- fadel train (strip marks or the input side)
Mix ratio sanity: keep classical <= 60% (Sadeed dominates rows; cap by CHAR budget per
source to avoid Abdou drowning), dedupe at the doc-hash level on normalized text.
NO new downloads required; all four sources are on disk.
Character coverage check: reuse the SAME 171-char diac tokenizer (TK.VOCAB_SIZE) - byte-identical
char handling, no mixed scripts surprises. Report the char-census (letters vs punct vs PASSTHROUGH
classes) before packing.

### 1.2 Packing
- New flag in diacritizer/scripts/prepare_data.py: kind "rawlm" -> pack PLAIN CHAR TEXT
  (no labels; y all -1) into ctx 512 windows (NOT 128: LM pretraining wants context; the
  final fine-tune stays at its own ctx 128 and stage-2 handles the length gap).
- data/diac/stage1/tokens/{train_ids,val_ids}.npy, val = doc-stratified 1%.
- Deterministic seeds (20260911); provenance jsonl like the marked corpus (per-source +
  doc_hash so stage-2's J-batch pairing can verify NO MARK-STRIP artifacts).

### 1.3 Stage-1 model & trainer
- SAME arch as the stage-2 v2d28 diacritizer (28L/272h/8h/2kv/ffn1088): 30,103,600 params.
  Rationale: warm-start = load the SAME state_dict; zero shape surgery. (The v2d28 gate win
  makes depth the arch default for this budget.)
- New diacritizer/scripts/train_lm.py (extends train.py, ~80 lines): causal mask ON,
  TRANSPOSED LM head (tied to embed), plain CE loss, same fp16 contract/AMP/zero-flag
  auto-resume contract (checkpoint-* rotation + best.pt + final/=best + config.yaml).
- budget: M-2 pilot classprobe: VRAM/step check + 1000-step smoke, then USER GO for the
  full run. Planned compute: ~6 h/45M chars on the RTX 3000 (measured 1.38 s/step at ctx
  128; scale leagues measured live before the number is final - 512 ctx substitutions).
- lr 2e-4 cosine-ish fallback = constant + early_stop patience 12 (same discipline); the
  pretraining budget is CPU-cheap relative to checkpoint-backup duty.

### 1.3b Stage-1 model feels "done" when...
- val CE plateaus (patience 12 similar) AND plain-text anneal smoke samples read as plausible
  Arabic flow (spot-check top-5 continuation of 5 held-out bare openings - this is a
  language-model sanity gate, not a gate for the diacritizer head).

## 2) Stage 2 — warm-start the bidirectional diacritizer + fine-tune

### 2.1 Corpus
- The same v3 build the OTHER open row already wants: v3 = v2b corpus + sadeed_tashkeela
  (adds 1.04M windows classical-books; total ~2.49M train windows expected).
- MANDATORY FIRST (cheap, CPU): dedup check Sadeed_Tashkeela train text vs SadeedDiac-25
  gate rows (longest-common-substring / mark-stripped exact + relaxed match report);
  if overlap exists, the Sadeed gate gets an "overlap-flagged" caveat in every report.

### 2.2 Trainer change (one hook)
- diacritizer/scripts/train.py: add pretrained_init: path (torch.load -> strict=False into
  the bidirectional model). The 15-class head + head/final proceed fresh (the transplant is
  embeddings + encoder stack; label head has no pretrained counterpart).
- Tie break: none - everything else stays the SAME frozen data/lr/batch/seed/patience as
  the v2b control EXCEPT total_steps can be 8000 (fine-tune class; early stop patience 8)
  -> runnable in ~3 h.

### 2.3 Ablation arms (cheap, queuing after the primary)
- A: warm-start every layer, bidirectional fine-tune (primary; the bet).
- B (open question, one extra run if A wins): ALSO reset the last 2 encoder layers before
  fine-tune (CATT's reset-last-layer trick) - tests whether deep bidirectional mismatch
  hurts the transferred causal stack.
- Dedup/contamination gates on fadel_test text vs stage-1 corpus (overlap check via
  the same dedup report helper as 2.1).

## 3) Gates & decision rules (registered BEFORE results)
- Same 4-gate pack (bench.py --cfg, eval.py compare): fadel_test, sadeed25,
  wikinews2024, wikinews2014; text preservation hard gate = 1.0 everywhere.
- DECISIVE if >= 2 pp DER improvement on >= 3 gates vs the best from-scratch arm
  (fadel_spec on its own gates / v2d28 on the WN gates).
- Publication of the honest model-card table stays the standing deliverable; GPL-2 commercial
  use FLAG stays user-gated and unchanged by this plan.

## 4) Budget & sequencing
- S1 prep + smoke: ~1 h CPU + 0.5 h GPU (user go).
- S1 pretrain: ~6-8 h GPU (32k steps at ctx 512 measured live; auto-resume contract).
- S2 A: ~3 h GPU (8k steps). S2 B optional: +3 h.
- Total: ~12 h GPU + ~1.5 h CPU. CPU-only stages (prep, Sadeed dedup) can run under any
  live train job (GPU-pool exempt per repo rule).
- Everything reuses the v2b matched-window infra (tokens (tokens_phase) hook, bench --config,
  final/config.yaml shipping, zero-flag resume).

## 5) Files touched (on approval)
- NEW: diacritizer/scripts/train_lm.py, configs/diac_stage1_lm.yaml
- diacritizer/scripts/prepare_data.py: kind rawlm + --out-dir already in place
- diacritizer/scripts/train.py: pretrained_init hook (~15 lines)
- configs/diac_stage2_warm.yaml, configs/diac_stage2_reset2.yaml
- docs updates: TASKS row 57 (closure), HANDOFF section, EXPERIMENTS E-14 row, MEMORY lessons
  as they fall out (the transfer mismatch / warm-vs-transplant delta reading).

## 6) Explicit user gates remaining
- G1: approve the Stage-1 corpus mix + budget (~6-8 h GPU for pretraining).
- G2: give the go to launch the S1 pretrain (one command, SSD stable, no co-run).
- G3: after S1, choose warm-start arm A now vs +counterfactual reset2 arm B (+3 h).
- Standing: GPL-2 decision unaffected; push remains gated.
