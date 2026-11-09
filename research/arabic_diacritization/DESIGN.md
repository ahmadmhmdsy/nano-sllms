# Arabic Diacritization (تشكيل) Specialist Model — Design (v1, APPROVED)

Date: 2026-09 (session-confirmed)
Status: design approved by user; implementation plan next
Location: submodule-form under the repo (per user decision)

## 0. Goal

Build a small, high-quality Arabic diacritization model in this repo. Input: Arabic
text (bare, partially, or fully diacritized). Output: the same text with accurate
diacritics. Primary task = per-Arabic-letter diacritic prediction; everything
non-Arabic passes through byte-exact.

## 1. Decisions (user-confirmed)

| # | Decision | Value |
|---|---|---|
| D1 | Build route | OUR architecture blocks (src/model.py GQA blocks) as a NEW bidirectional encoder + per-char classification head. NOT a CATT fork, NOT M3/e1 fine-tune (their English/code weights don't transfer; CATT stays as external baseline). |
| D2 | Pilot/target size | ~30M params (evidence sweet spot; CATT-scale, rababa-scale). D-target (12L/512d ~100M) only if 30M plateaus below gates. |
| D3 | Repo placement | Sibling submodule-form project dir with our usual structure (own configs/, scripts/, src additions, data/diac/*, runs/diac/*). Cleanly separated from the M3 pipeline files. |
| D4 | Policy ladder (existing diacritics in input) | Stage 1 = strip-and-rediacritize -> Stage 2 = preserve-known+fill-gaps (two-stream input + synthetic partial-diacritic curriculum) -> Stage 3 = dual modes (preserve + explicit rewrite/correct mode). Each stage gated on previous stage's eval PASS. |
| D5 | Target domain | Phase 1 = Mixed general (SadeedDiac-25-shaped, ~50/50 MSA/Classical). Scale-up path to "everything" incl. poetry & Quran via later domain stages (row-region Stage 4). |
| D6 | Non-Arabic / edge input | Copy-by-construction: context-visible, zero-predictability, byte-exact reassembly. See §4. |

## 2. Model

### Architecture

- Bidirectional char Transformer encoder: Llama-style blocks (RMSNorm + RoPE +
  SwiGLU + GQA, SDPA for sm_75/turing) with the causal mask REMOVED —
  LlamaModel-as-encoder — plus a shared linear head (hidden -> 15) applied at
  every Arabic base-letter position; non-Arabic positions are never classified.
- Heads: flat 15-class softmax first (CATT/Fadel precedent):
  {bare, fatha, damma, kasra, sukun, fathatan, dammatan, kasratan,
   shadda, shadda+fatha, shadda+damma, shadda+kasra,
   shadda+fathatan, shadda+dammatan, shadda+kasratan}
  Two-head factorization (vowel x shadda-flag) = later ablation. CRF = later ablation only.

### Ladder (S->P->T analog)

| Phase | Config | Purpose |
|---|---|---|
| D-smoke | ~4M (4L, hidden 192, ctx 512) | pipeline validation only; learns deterministic patterns (prefixes, article, shadda forms) |
| D-pilot | ~30M (6L, 384d, 6 heads, ffn ~4x, ctx 1024) | PRIMARY model; CATT/rababa scale |
| D-target | ~100M stretch (12L, 512d) | only if D-pilot plateaus below gates |

Training discipline reuses repo conventions: fp16-only, SDPA, auto-resume
contract, configs in configs/, checkpoint rotation, single-GPU exclusivity
(never co-run with an active train.py).

## 3. Data & tokenizer

### Tokenizer

- Char-level, small (~250-300 IDs): Arabic letters + hamza/alif variants +
  digits + punctuation + specials/PAD/UNK/BOS; NOT CodeLlama BPE.
- Two alignment-invariant streams are derived from each training line:
  (a) base-character sequence (diacritics stripped), (b) per-char 15-class label.
  Alignment validated sample-by-sample; corrupt samples quarantined.

### Corpus (Stage-1 strip mode)

- Backbone: cleaned Tashkeela-derived (Fadel split / Sadeed Tashkeela) —
  never raw Tashkeela.
- MSA slices: QCRI diacritized Wikipedia + WikiNews-style prose.
- Phase-1 mixture target ~65/35 Classical/MSA (tunable; recorded per run).
- Windows: <=1024 char positions, split at word/space boundaries
  (truncate-at-last-space rule), never mid-word or inside numbers/URLs.
- Document-level dedupe; document/source-stratified split (no random sentence
  leakage); provenance + domain + tier (gold/weak) metadata on every line.
- Weak-labeled data (QCRI model-diacritized Wikipedia etc.) enters only in
  later stages, confidence-filtered, never blindly mixed with gold.

### Preprocessing module (its own tested unit)

Normalize (Unicode NFC, hamza/alif forms ONLY if spec demands), tatweel
handling, base/diacritic separation, label alignment checks, dedupe, split,
provenance tracking. Selftest corpus with pathological input (see §4).

## 4. Edge cases & non-Arabic policy (D6)

Principle: NON-ARABIC IS NEVER PREDICTED. The classifier exists only on
Arabic base letters. Non-Arabic units are fed to the encoder as context
tokens (syntactic signal for case endings) but carry no prediction head;
output assembly re-inserts them BYTE-EXACT from the original input.
Result: non-Arabic hallucination is structurally impossible; a
post-processor deterministically reconstructs the full string and a scored
gate (text-preservation rate) still verifies it.

| Case | Rule |
|---|---|
| Digits (0-9, ٠-٩) | passthrough byte-exact; context-visible |
| Latin/Cyrillic/Greek, mixed-script (WiFi, GPT-4o) | passthrough byte-exact; context-visible |
| Emoji, CJK, symbols | passthrough byte-exact; context-visible |
| Directional/invisibles: RLM U+200F, LRM U+200E, ZWJ U+200D, ZWNJ U+200C, ZWSP, BOM | passthrough byte-exact (formatting; worst thing to silently drop) |
| Tatweel U+0640 | passthrough unchanged, NEVER diacritized (decided) |
| Lone combining mark with no base letter | passthrough untouched; never a training target |
| Quranic annotation symbols (U+06D6-U+06ED etc.), footnote markers | passthrough in Stages 1-3; revisit in Quran domain stage |
| Pre-existing Arabic diacritics on input words | NOT passthrough — governed by the D4 policy ladder (Stage 1 strips; Stage 2 preserves; Stage 3 dual-mode) |
| Sentence/window boundaries | structural rule, not vocab: chunk splits at word boundaries only; explicit sentinel punctuation list so windows never cut inside a number or URL |

Long documents are processed as independent <=1024-char windows with zero
cross-window drift; case endings are locally resolvable (rababa evidence:
longer paragraph context helped one benchmark but HURT WikiNews — 1024 is
enough; longer ctx = optional later ablation, not a design dependency).

## 5. Evaluation harness (built FIRST, before any model training)

- DER + WER, each +- case endings (QCRI scoring-script-compatible definitions;
  multi-reference support where the benchmark allows).
- Text-preservation / hallucination rate (changed base chars, dropped/inserted
  spans, punctuation damage) — scored as a HARD gate even though it should be
  ~0 by construction.
- Preservation gate (Stage 2+): known_diacritic_accuracy on partially
  diacritized inputs.
- Benchmark pack — ALL THREE must be reported together; any single one is
  meaningless on its own:
  1. Fadel test 2,500 lines (cleaned classical)
  2. WikiNews-2024 (modern MSA, multi-reference)
  3. SadeedDiac-25 (mixed 50/50, expert-reviewed)
- Phase gate style: mixed-benchmark result must BEAT a recorded baseline
  (CATT EO published numbers + our own MSA regression pack) kept in the report
  for comparability.

## 6. Phase ladder (task rows)

| Phase | Content | Exit gate |
|---|---|---|
| A0 | submodule scaffolding + char tokenizer + preprocessing module + data pipeline + eval harness — ALL CPU-only | corpus built; 3 benchmarks loadable; eval selftest PASS on known examples |
| A1 | D-smoke training on a small clean slice (~10-50k lines) | end-to-end reproducible; learns deterministic patterns; loss curves sane |
| A2 | D-pilot full training (strip mode) | Fadel + WikiNews + SadeedDiac all reported; honest internal baseline recorded |
| A3 | Stage 2 preservation (two-stream input, synthetic partial-diacritic curriculum at 0/10/25/50/75%/word-spans) | preservation gate PASS |
| A4 | Stage 3 dual-modes + (optional start of) domain extras (poetry/Quran) | mode behavior correct; subdomain checks |
| later | YaRN/ctx scaling tests, LoRA, soup-style post-hoc knobs, noisy-student weak-data expansion | evidence-driven only |

## 7. Reuse map (what stays ours vs what is new)

| Asset | Status |
|---|---|
| GQA blocks (RMSNorm/RoPE/SwiGLU/GQA, SDPA) | reused as bidirectional encoder core |
| train.py discipline (auto-resume, fp16, 8-bit-Adam option, rotation) | reused (adapted, new config family) |
| src/data.py memmap shard streaming | adapted: new packing format ((char_ids, labels) word-boundary windows) |
| YaRN / LoRA / soup_merge | inherited as later-phase options |
| eval.py / DER-WER scoring | NEW (QCRI-compatible), built in A0 |
| char tokenizer + labeling + alignment validator | NEW, A0 |
| two-stream partial-diacritic input | NEW, A3 |
| M3/e1 weights | NOT used (domain mismatch; Sadeed result shows decoder-LM framing is worse for this task) |

## 8. Source evidence anchors

- research/arabic_diacritization/arabic_diacritization_deep_research_chatgpt.md
- research/arabic_diacritization/arabic_diacritization_deep_research_claude.md
- research/arabic_diacritization/arabic_diacritization_deep_research_perplexity.md
Key anchors: CATT (abjadai/catt, Apache-2.0, 2407.03236); Fine-Tashkeel (2303.14588);
Deep Diacritization D3 partial-diacritic evidence (2011.00538); QCRI 2025 EMNLP
(advancing-arabic-diacritization; WikiNews-2024); Sadeed (2504.21635;
SadeedDiac-25; decoder-LM hallucination finding); Mishkala 12.5M 1.66% DER
(model-card evidence); rababa ~30M 0.99% DER (engineering evidence);
Tashkeela domain-skew and mandatory cleaning (10.1016/j.dib.2017.01.011).
