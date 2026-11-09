# DA-2 (Emo-analogue) report - 2026-09-19

Status: CLOSED honest. Trained, evaluated, exported; gates read against the
pre-registered bars in research/desert_ant_recreation/DA2_SPEC.md. E-44.

## What was built (all CPU-only, zero GPU contention)

- Data pipeline: langid/scripts/build_emo_eval.py - TEAD (arbml/TEAD, HF,
  12,558 Arabic tweets, DeepMoji one-emoji-type self-label rule) +
  tweet_eval/emoji (cardiffnlp, 20-class English emoji tweets; card
  license field "unknown" - research-use caveat recorded). Output
  data/langid/emo/{train,val,test}.tsv + emo_vocab.json.
- Dataset: 46,100 train / 5,729 val / 5,845 test; 76 emoji classes
  (67 Arabic types + 20 English types, unioned).
- Model: langid/src/emo_model.py - hashed features (word unigrams +
  adjacent-word bigrams + char 1..3-grams, FNV-1a -> 2^16 buckets,
  prefix-disambiguated) -> EmbeddingBag(65536x76, sum) + bias, multinomial
  LR, CPU Adam.
- Eval: langid/scripts/eval_emo.py - top-1/top-3, per-language rows; ties
  (margin < 1e-3) REPORTED and never counted (E-43 rule carried).
- Export: langid/scripts/export_emo_int8.py - per-column symmetric int8 +
  numpy inference proxy; compressed npz artifact.

## Gates (pre-registered in DA2_SPEC.md, read honestly)

| gate | result | verdict |
|---|---|---|
| D1 dataset (>=12k train, >=2k eval, >=40 eval emojis) | 46,100 / 5,845 / 67 gold labels (ar) + 20 (en) | **PASS** |
| D2 skill vs prior, top-1 >= 2x freq-prior AND >= prior+0.05 | prior 0.1990; bars 0.3979 / 0.2490. Best val 0.2358 (ep4); TEST ar top-1 0.2255, en top-1 0.2335 | **FAIL both bars** (best val short of prior+0.05 by 1.3 pp; test short by 2.3 pp (ar) / 1.5 pp (en)) |
| D3 per-language rows reported independently | ar 0.2255 top-1 / 0.3866 top-3; en 0.2335 / 0.4358; both languages kept | **PASS** (reporting gate) |
| D4 artifact <=3 MiB + infer <= 2 ms/text | 2.896 MiB int8 npz; 1.02-1.08 ms/text in a NAIVE per-text python-loop proxy (vectorized numpy would be several x faster); int8-vs-fp32 agreement ar 831/1000 = 99.2% (t: 831/838 test rows = 99.2%), en 980/1000 = 98.0% | **PASS** |
| D5 degenerate check | distinct predicted emojis: ar 45/67 gold alive, en 21/20 - no one-emoji collapse | **PASS** |

## Honest reading

- The model beats the frequency prior on BOTH languages (ar +2.7 pp, en
  +3.5 pp over 0.1990) and reaches top-3 0.39-0.44 - useful as a
  suggestion strip - but the pre-registered "skill" bar (2x prior, and
  prior+5 pp) was set for a model that would actually drive a product
  top-1 suggestion, and we do NOT reach it.
- Root cause is data/recipe scope, not the hashing head per se:
  (a) 76-way vs their 812-way vocabulary (v1 honest scope),
  (b) self-labeled tweets are noisy - TEAD is a balanced sentiment-mining
  set (the one-emoji-type rule leaves 67/469 emojis alive over 7,773
  rows), TweetEval-en is 20 emojis only,
  (c) lexical hashed features cap intent capture; their card's size
  budget implies a compact transformer encoder inside 10 MB.
- Arabic is IN the product (0.226 top-1 vs 0.199 prior) but en-parity is
  not reached in v1; the ar gap traces to only 838 test rows and emoji
  diversity, not to encoding (Arabic script flows through the same
  featurizer with no routing - consistent with the DA-1 Arabic-first
  stance: emojis are script-agnostic).

## What would plausibly close D2 (next user-gated rung, not done here)

1. Bigger Arabic emoji-labeled dump (snakers4 multilingual dataset was
   both 404 and CC-BY-NC; re-mining a Twitter archive is feasible but
   heavy).
2. A tiny 1-2 layer transformer head over the same hashed features.
3. Class-balanced weighted CE + label smoothing.

## Artifacts

- runs/langid_da2/emo_best.pt (fp16 state dict + labels; val 0.2358)
- runs/langid_da2/emo_int8.npz (2.896 MiB compressed)
- runs/langid_da2/{train_log.json, eval_report.json}
- data/langid/emo/ (gitignored raw splits; regenerable via build script)
