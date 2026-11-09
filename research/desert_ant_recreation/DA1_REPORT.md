# DA-1 report - Tongue-analogue text language ID (langid/)

Date: 2026-09-19. Design: research/desert_ant_recreation/DESIGN.md. Plan: docs/plans/2026-09-19-desert-ant-recreation-da1-langid-plan.md.

## What was built

Full recreate of Desert Ant Labs' "Tongue" recipe (their card, 84 langs): letters-only
NFC+lowercase words, char n-grams n=1..4 angle-bracketed per word, FNV-1a 32-bit hash
-> 2^16 buckets, bag-of-ngrams (no tokenizer), torch EmbeddingBag(65536->21, sum)+bias
(multinomial logistic regression), script router (ko/ja/he/el/hi; Arabic script NEVER
routed - ar/fa/ur/ps separated lexically, the Arabic-first requirement). Per-language
int8 export + pure numpy inference. CPU-only end to end; no GPU used at any step.

## Data

- Train: Tatoeba export sentences.tar.bz2 (218.6 MB), 855,220 train + 9,790 val rows
  over 21 langs (cap 50k/lang, dedupe alnum-lower key, seed 42). Availability limits
  recorded: ur 2,851 rows, ps 66 rows (!), hin 16,465 in the raw tally.
- Eval (held out, never trained on): official FLORES-200 archive dev+devtest,
  42,189 rows, 2009/lang, manifest data/langid/eval_manifest.json with source string.

## Gates (pre-registered BEFORE training) - honest results

| Gate | Bar | 2-epoch model | 3-epoch model | Verdict |
|---|---|---|---|---|
| G1 smoke | <10 min, beats majority | 6 s, val 0.9905 vs baseline 0.0499 | - | PASS |
| G2 full | >= 0.97 | 0.9850 | 0.9877 | PASS |
| G2 5-word | >= 0.95 | 0.9519 | 0.9510 | PASS |
| G2 3-word | >= 0.90 | 0.9088 | 0.9072 | PASS |
| G2 1-word | >= 0.70 | 0.6953 | 0.6921 | FAIL (tie rate 3.8-6.6% reported, never guessed) |
| G3 arabic-script 3-word mean | >= 0.85 | 0.789 (ar .974 fa .945 ur .781 ps .457) | 0.778 (ar .987 fa .915 ur .729 ps .483) | FAIL - ps has only 66 training sentences |
| G4 artifact | <= 2.5 MiB, <= 1 ms/word | 0.82 MiB, 0.033 ms/word (numpy CPU); int8 0.9875 vs fp32 0.9880, delta 0.05pp | same | PASS |
| G5 ties/abstains | never guessed | margin gates active; abstain only featureless | - | PASS |
| G6 leakage | 10-word shingle overlap 0 | overlap 0 (225979 x 567463 10-word shingles) | - | PASS |

Majority-class baseline on eval: 0.0476. Reference points from their 84-lang card:
FLORES 3-word 0.933 (lingua 0.887), 5-word 0.974, sentences 0.971, 1-word 0.759.
Our 3-word 0.907 (no comparison baseline installed - lingua-py install is USER-GATED).

## Verdict

G4 fully closed (int8-vs-fp32 delta 0.05pp) and G6 closed (0 leakage shingles). Trained-model quality is real (full-sentence 0.9877 vs 0.048 baseline, 12x int8-NUM
small artifact). 1-word accuracy is ~0.695 - just under the pre-registered 0.70 bar;
Partho (ps) is the G3 blocker with an honest data-limitation root cause: the Tatoeba
export has only 66 Pashto sentences. USER DECISION NEEDED: (a) accept 1-word margin
reporting + add a ps/ur data source (HF crowdsourced corpora) as a DA-1b increment,
or (b) close DA-1 at 3/6 pre-registered PASS and move to DA-2.

## Artifacts

runs/langid_da1_e3/{model_fp32.pt, train_summary.json, eval_report.json} (local),
langid_int8.npz 861,785 bytes in runs/langid_da1 (regenerable one-liner).
