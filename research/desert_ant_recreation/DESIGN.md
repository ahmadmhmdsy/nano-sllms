# DA-line - Desert Ant recreation: design

Status: opened 2026-09-19 with explicit user approval ("create a
plan/milestones/tasks/..etc for them and start with most feasible/achievable
one"). Reference teardown: research/desert-ant-labs-models-report.md.

## 1. Purpose

Recreate Desert Ant Labs' one-tiny-model-per-task portfolio as an
Arabic-first on-device product line, using the teardown report as the
reference for architecture, data, and benchmark protocol.

## 2. Policy (inherits repo discipline)

- Self-contained submodules per target (same placement discipline as the
  D-line's diacritizer/): DA-1 lives in langid/; no edits to src/, configs/,
  diacritizer/, mex/ of the M3/D/mu lines.
- Gates and the eval harness are built BEFORE any training (D-line A0
  discipline); gates are pre-registered in the plan before any results.
- CPU-only by default; GPU only for the audio tier, and only in a genuinely
  free GPU window (never beside a live run).
- Honest reporting (PASS/FAIL/SKIPPED/BLOCKED); one EXPERIMENTS.md row per
  closed experiment, registered before results.

## 3. Ladder (feasibility-ranked)

| # | Target (their model) | Their recipe (one line) | Our angle | Feasibility here |
|---|---|---|---|---|
| DA-1 | Tongue (text lang-ID) | hashed char n-grams + script routing, 2 MB, 84 langs | 21-lang lexical model incl. the Arabic-script group ar/fa/ur/ps; FLORES-style held-out protocol | HIGH: CPU-trainable in minutes, eval fully copyable - START HERE (approved) |
| DA-2 | Emo (emoji suggestion) | small text classifier, 812 emojis, 22 langs | Arabic is in their 22 langs - direct baseline exists | MEDIUM: needs a labeled emoji-text corpus we must build ourselves |
| DA-3 | Gist (topic tagging) | static int8 embedding + hashed n-grams + fp16 head, 36 topics / 101 langs | reuse their 36-topic IAB taxonomy; Arabic news topic data (SADA, ArNews) | MEDIUM |
| DA-4 | Uhm (filler detection) | DistilHuBERT fine-tune, per-20ms 6-class | Arabic fillers (ya3ni, eeh, umm) are acoustically distinct; needs podcast audio | MEDIUM-LOW (GPU + audio collection) |
| DA-5 | Ear (spoken lang-ID) | Whisper-tiny pruned to a language head + syllable-rate windows | Arabic DIALECT ID (MSA / Egyptian / Gulf / Levant / Maghrebi) - unshipped niche, differentiator | MEDIUM-LOW (GPU) |
| DA-6 | Voz (ASR) | Parakeet TDT 0.6B v3 -> Core ML | Arabic ASR - largest value gap, largest lift | LOW (GPU-heavy; likely fine-tune an existing checkpoint) |
| DA-7 | Redact (PII) | 23M BERT token classifier + deterministic rules | Arabic IDs/IBAN/phone rules; WikiANN/MultiNERD have Arabic | MEDIUM |
| DA-8 | Title (titles) | Granite-350m SFT | maps onto our existing SFT pipeline + D-line infra | MEDIUM |
| DA-9 | Schemer (schema extraction) | pruned mmBERT + per-type heads + deterministic harness | hardest: Arabic relative dates, Arabic-Indic numerals, Hijri/Gregorian | LOW (last) |

Each rung: a research/<line>/DESIGN.md section (this file for DA-1) plus a
docs/plans/ plan, and an EXPERIMENTS.md row registered before training.

## 4. DA-1 spec (Tongue-analogue, name: "langid" / lisan)

Scope: 21 languages - Arabic-script group first-class:
ar, fa, ur, ps, en, de, fr, es, it, pt, nl, tr, ru, uk, pl, el, he, hi,
ko, ja, zh.

Recipe (mirrors theirs):
- normalize: NFC + lowercase; letters-only words (digits/punctuation carry
  no language signal and poison the 1-word regime).
- features: per-word char n-grams n=1..4 wrapped in angle brackets
  ("<wor", "word", ... word-boundary markers), feature-hashed with FNV-1a
  32-bit into 2^16 buckets; word order ignored (bag of n-grams). No
  tokenizer file - the host hashes (their stated design).
- script routing (no model involved): Hangul -> ko, Kana -> ja,
  Hebrew -> he, Greek -> el, Devanagari -> hi (at least 2 matching chars).
  Arabic script is NEVER routed: ar/fa/ur/ps share it and must be separated
  by the lexical model - that is the research value of DA-1 for the
  Arabic-first mission.
- model: EmbeddingBag(65536 -> 21, sum) + bias = multinomial logistic
  regression on hashed features; zero-init; Adam lr 0.05; CPU torch.
- data: Tatoeba sentences export (CC-BY), cap 50k/lang, dedupe on the
  alphanumeric-lowercase key, seeded shuffle (42), val = 10% (max 500/lang)
  from the shuffled tail.
- eval: FLORES-200 dev held-out as primary (ungated-mirror probe; honest
  fallback = wikimedia/wikipedia first-sentence slices, same wiki domain);
  protocol = full / 5 / 3 / 1-word accuracy with seeded word windows;
  tie = top1-top2 logit margin < 0.5 (reported, never counted correct);
  abstain = no letter features (reported, never counted correct).
- export: per-language-column int8 (max-abs scale), pure numpy inference,
  artifact <= 2.5 MiB, latency <= 1 ms/word on this CPU.

Pre-registered gates (registered here BEFORE any training):

- G1 smoke: a 2k/lang x 1-epoch CPU run finishes < 10 min and beats the
  majority-class baseline on the val split.
- G2 (final eval set): full >= 0.97; 5-word >= 0.95; 3-word >= 0.90;
  1-word >= 0.70.
- G3: Arabic-script group (ar/fa/ur/ps) mean 3-word acc >= 0.85.
- G4: artifact <= 2.5 MiB; int8 within 0.5 pp of fp32 on the eval set.
- G5: tie/abstain behavior verified - ties and no-letter inputs are
  reported, never guessed.
- G6: leakage barrier - 10-word shingle overlap between train and eval
  sets == 0.

Reference points from their card (84 langs): FLORES 3-word 0.933 (lingua-py
0.887), 5-word 0.974, held-out sentences 0.971, held-out single words 0.759.

## 5. Risks

- Tatoeba's short-sentence domain vs FLORES's news domain: conservative gate
  numbers + wiki-domain fallback eval.
- ps (Pashto) Tatoeba volume is small; cap applies; G3 is additionally
  reported per-language so a weak member is visible.
- fa/ur 1-word classification is genuinely hard (vocabulary shared with
  Arabic across script families); the gates price this in honestly.
- zh/ja Han overlap: Kana presence routes ja; Han-only text stays lexical.

## 6. Relations to existing lines

- Independent of M3/D/mu lines: no shared weights, no GPU time, zero
  contention by construction (the single-GPU correlation risk does not apply).
- Complements the D-line: D-line models diacritized Arabic text; DA-1 only
  identifies language/script.
- mu2's G1 char-LM is generative on vocalized Arabic; DA-1 is a classifier
  on raw text - different task, no artifact sharing.
