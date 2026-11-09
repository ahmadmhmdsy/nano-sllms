# Implementation Plan — Arabic Diacritization Specialist (D-line, submodule-form)

Date: 2026-09-11 · Status: APPROVED SPEC (user); execution pending task-row gates
Design (approved): research/arabic_diacritization/DESIGN.md — read it first; this
file decomposes it into buildable tasks with validation gates. Evidence anchors:
research/arabic_diacritization/arabic_diacritization_deep_research_*.md.

## 0. What this is (and is not)

- A NEW specialist model inside this repo ("submodule-form"): char-level
  bidirectional Transformer encoder + 15-class diacritic head (D1/D2 DESIGN).
- NOT an M3/e1 fine-tune (English/code weights; decoder-LM framing is the
  Sadeed failure mode). NOT a CATT fork (stays an external baseline).
- GPU discipline inherited: fp16-only, SDPA, auto-resume zero-flag contract,
  single-GPU exclusivity. A0 is CPU-ONLY by design (safe beside any live run).

## 1. Repo layout (new, self-contained)

```
diacritizer/
  README.md                 usage + gates summary
  src/
    tokenizer.py            char vocab (~250-300 ids) + encode/strip/reassemble
    labels.py               15-class label schema + diacritic regex tables
    model.py                bidirectional encoder wrapper + DiacritizerModel
                            (reuses/re-exports our GQA block set; causal mask OFF;
                             per-position linear head on Arabic base letters only)
    data.py                 labeling pipeline: line -> (char_ids, labels),
                            alignment validation, quarantine, provenance fields
    passthrough.py          non-Arabic span detection + byte-exact reassembly
    eval_der.py             DER/WER (+- case endings), text-preservation,
                            preservation rate; QCRI-script-compatible semantics
  scripts/
    download_data.py        Fadel split + Sadeed Tashkeela + WikiNews-2024 +
                            SadeedDiac-25 (sources w/ licenses; resumable)
    prepare_data.py         clean -> align -> windows (<=1024 chars, word
                            boundaries) -> memmap shards  data/diac/<phase>/
    tokenize_data.py        pack [(char_ids,label_ids)] blocks -> tokens bin
    train.py                SAME contract as repo train.py (auto-resume zero
                            flags, fp16, rotation; new config family)
    eval.py                 eval over the 3-benchmark pack + report json
    selftest.py             one command: tokenizer+labels+passthrough+data+
                            eval selftests  (A0 gate runner)
configs/
  diac_smoke.yaml           ~4M: 4L, hidden 192, ctx 512
  diac_pilot.yaml           ~30M: 6L, hidden 384, heads 6, ffn 1536, ctx 1024
  diac_target.yaml          (deferred; only if pilot < gates)
data/diac/<phase>/{raw,tokens}/   same gitignore policy as data/<phase>
runs/diac/<phase>/                 checkpoint rotation + logs + gates reports
tests? -> folded into diacritizer/src selftests (repo stdlib-style, pytest-free)
```

Shared-file touch policy: NO edits to src/model.py, train.py, data.py of the
M3 line. Reuse = import of block-building pieces where feasible; otherwise
diacritizer/src/model.py is local code with the same block math. Anything that
 WILL touch an M3-line file gets its own TASKS row + diff list first.

## 2. Tasks (map to TASKS.md rows 50-55)

### R50 - A0a scaffolding + tokenizer + labels + passthrough (CPU)
- char vocab: Arabic letters (incl. hamza/alif variants per normalization spec),
  digits, punctuation sentinels, specials (PAD/BOS/UNK + <src:tier> sidecar
  metadata live OUTSIDE the vocab in data provenance).
- labels.py: the 15 classes; decode(label) -> combining marks; label extraction
  from a diacritized char stream (ong-ahead parse like the CATT schema).
- passthrough.py: greedy scanner classifying every codepoint: ARABIC_BASE |
  DIACRITIC | TATWEEL (passthrough) | QURANIC_MARK (passthrough stage-1) |
  INVISIBLE_RLM_LRZ_ZWJ (passthrough) | OTHER (passthrough, context-visible).
  Output guarantees byte-exact segment recovery.
- VALIDATE: selftest corpus (emoji soup, mixed script, RLM/ZWJ/ZWSP, lone
  marks, tatweel, Quranic marks, Arabic-Indic + Western digits, URLs/numbers
  spanning chunk borders) -> reconstruct(input) === input for every case; round
  trip strip->relabel->reconstruct vs hand-built gold labels.

### R51 - A0b data pipeline (CPU; network download is CPU-safe)
- download_data.py: Fadel github (AliOsm/arabic-text-diacritization),
  QCRI advancing-arabic-diacritization (WikiNews-2024 + scoring reference),
  Misraj/Sadeed_Tashkeela + Misraj/SadeedDiac-25 (HF). Record_license field on
  every source; Tashkeela-derived = GPL-2 noted in provenance.
- prepare_data.py: clean -> per-line alignment (+quarantine) -> split at word
  boundaries, window cap 1024 -> document-level dedupe -> document/source-
  stratified train/val split -> provenance jsonl.
- Mix record: Phase-1 target ~65/35 classical/MSA; exact per-source counts
  land in prepare_stats.json (must be reproducible from seed).
- VALIDATE: selftest on 100-line hand-set; split-leak check (same doc never on
  both sides); all 3 benchmarks load + score harness on canned examples
  (known-good and known-bad cases, PASS/FAIL).

### R52 - A0c eval harness + Gate-Runner (CPU)
- eval_der.py: DER, WER, each +-case-ending; text-preservation (identity-out
  gate); preservation metric staged but inactive until A3; multi-reference
  support (WikiNews-2024/SadeedDiac-25).
- MUST show, on synthetic pairs: perfect model => 0.0 all metrics; fixed 1-
  char-wrong gold correction => expected exact numbers (regression-locked).
- VALIDATE: scripts/selftest.py --all = 4/4 PASS IS THE A0 EXIT GATE.

### R53 - A1 D-smoke (GPU, queued behind live runs)
- config diac_smoke.yaml, ~10-50k-line clean slice.
- EXT gate: pipeline end-to-end reproducible (loss curve monotonic, no NaN,
  auto-resume drill kill->re-run zero-flags PASS), deterministic patterns
  qualitative-eval spot checks show expected patterns; no gate on benchmark numbers.

### R54 - A2 D-pilot (GPU; the deliverable rung)
- config diac_pilot.yaml (30M ctx 1024).
- Gate: 3-benchmark pack ALL reported; vs recorded external baselines table
  (CATT EO per paper claims + Fine-Tashkeel + Turath1.0 claims — honest model-
  card vs our-measured annotations). Gate metric per DESIGN §5.
- BYPRODUCT: MSA-regression pack (fixed WikiNews/Sadeed slices) saved as the
  standing regression gate for all later stages.

### R55 - A3 preservation (two-stream) + A4 dual-mode — designed AFTER A2 gates
- two-stream input (base chars + observed-diacritic channel), curriculum on
  known-share 0/10/25/50/75%/word-spans; preservation gate =
  known_diacritic_accuracy == 1.0 on held-preserved eval cases;
- A4 = modes: preserve-mode fixtures keep known marks identical; "rewrite" mode is a
  config/CLI switch w/ separate eval flags.

## 3. Standing risks / notes

- VRAM: 30M encoder ctx 1024 is trivial (<2 GB projected incl. fp16 grads);
  probes still run per protocol (mirrors M1 discipline).
- Dataset legality: Tashkeela-derived GPL-2 — record license, training-use OK
  for research; commercial use = user decision point flag in A2 report.
- Known failure surface = case endings on MSA; measure and report them
  separately every time (SadeedDiac-25 w/ and w/o case endings).
- The recorded repo-wide "no crash while a train runs" and single-GPU rules
  apply (AGENTS.md §4); A0 is co-run safe.
- transformers 5.16.1 drift items (MEMORY 1-14) apply (no
  logging_dir/save_safetensors args; processing_class=; eval_strategy=).

## 4. Order of execution

A0a -> A0b -> A0c -> (SELFTEST GATE) -> A1 -> A2 -> A3/A4 (design refresh), all
task-gated in TASKS.md; GPU rows queued behind the currently-active KT-2
round 2 (row 48). Commits: one per task row close; pushes remain user-gated.
