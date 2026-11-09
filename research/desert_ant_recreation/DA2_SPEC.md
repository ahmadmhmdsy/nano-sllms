# DA-2 spec (Emo-analogue, name: "emo" / masha'ir-al-ima'ji)

Status: opened 2026-09-19 (user: "go to DA-2 Emo"). DA-1b (Arabic gaps for
langid) STOOD DOWN as a pending task - revisit later (user decision).

## Scope (v1, honest)

- Task: short-text -> emoji suggestion (top-1 / top-3), Arabic-first.
- Languages v1: **ar + en** (their model has 22 langs; ar is the priority
  language and en is the feasibility anchor; adding more langs is a later
  goal, not v1).
- Vocabulary: capped at top-K emoji types by training-set frequency
  (target K=150; their 812 needs a bigger tweet dump - documented as
  DATA-LIMIT, not architecture; vocabulary is a config field, so the model
  scales by retraining).
- Domain: tweets / short intent text (matches their "short intent-oriented"
  product note).

## Recipe (mirrors DeepMoji-style self-labeling)

- self-labeling: for TEAD rows the label is the emoji the author themselves
  chose; only rows with exactly ONE emoji type are kept (DeepMoji rule;
  multiple emoji types = unresolvable intent for top-1).
- normalize: NFC + lowercase, strip URLs/mentions/hashtag marks; keep
  hashtags text (no '#') - they carry intent.
- features: word + char-n-gram (n=1..3) bag, FNV-1a 32-bit -> 2^16 buckets
  (same machinery as DA-1 features - honest reuse).
- model: EmbeddingBag(65536 -> K) + bias, multinomial LR (same as DA-1).
- export: per-emoji int8 + pure numpy infer <= 3 MiB.

## Data sources (ungated only)

| Source | Language | Rows | Labels | License | Use |
|---|---|---|---|---|---|
| arbml/TEAD (HF) | ar | 12,558 (astral emoji in 11,046) | 469 types, self | research (Abdellaoui & Zrigui 2018); verify per-row source policy | main Arabic |
| cardiffnlp tweet_eval subset "emoji" | en | ~50k train / 10k val / 50k test | 20 emoji labels | apache-2.0 | English |

- rejected: snakers4 emoji-sentiment-dataset (11 langs incl. ar but a
  CC-BY-NC license and its hosting 404s - both disqualifying for us);
  SemEval-2018-T2 multilingual (de/es/fr/it only - no Arabic).

## Pre-registered gates (BEFORE any training)

| # | Gate | Bar |
|---|---|---|
| D1 | dataset | >= 12k train + >= 2k eval rows; >= 40 distinct eval emojis not zero in the eval split; leakage split by source+date-hash so no near-duple lies |
| D2 | skill vs prior | top-1 >= 2.0x the label-frequency prior (prior measured on the TRAIN marginal, honest), and top-1 >= freq-prior + 0.05 absolute |
| D3 | per-language row | both ar and en eval top-1/macro-F1 reported independently; neither language silently dropped |
| D4 | artifact | <= 3 MiB int8, numpy infer <= 1 ms/text ar CPU |
| D5 | degenerate check | for 5 k-samples human-readable confusion table: not >90% mass on one emoji (sanity, not gate) |

Failure analysis must be honest: if D2 fails we report FAIL and the per-
language rows, plus a "what data would fix it" note, NOT a silently-scaled
back gate.

## Note on ties (from DA-1 discipline)

top-1 vs top-2 candidates within margin 1e-3 are REPORTED (polysemy
honored) but never counted as right or wrong - same E-43 tie rule.
