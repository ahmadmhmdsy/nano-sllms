# E-20 — cache/n-gram lookup for OUR gold + small-model trim plan + QCRI datasets (2026-09-17)

Answers to the user's three questions after E-19/E-19b, with measured evidence.

## 1. Can we train on QCRI (EMNLP 2025)? — YES (with one gap to close)

Repo cloned as models/e19/qcri-src/. Contents (all free of our gate-experiments):

- **Wikipedia_Diacritized_Corpus/Wikipedia_20240420.diac.jsonl** — 91.4 MB,
  32,834 articles / **5.07M words**, fully diacritized BY THEIR BEST BiLSTM
  (WER ~3% on WikiNews-2014). Machine-labeled, not human gold.
- **WikiNews Benchmarks** — WikiNews-2014 (the SAME benchmark as our wn2014
  gate, multi-ref) + new WikiNews-2024 (~10k words, multi-ref).
- Evaluation script (Java) with multi-reference scoring.

Training-use caveats:
1. **License is NOT stated in the repo or README.** The paper is CC-visible;
   the data itself is "all publicly shareable" per README, but no LICENSE
   file ships. Research-only use is fine under our deployment lock, but a
   license question to qcrisupport@hbku.edu.qa is needed before any redist.
2. **Contamination**: their WikiNews-2014 bench IS our wn2014 gate — any
   training on the JSONL corpus must be shingle-deduped against ALL our
   gates first (we already have dedup_check.py). The corpus is Wikipedia
   (dump 2024-02-20), so sadeed/fadel (classical) overlap should be near zero.
3. Labels are model-predicted (~3% WER): fine as weak data for a SMALL model,
   not as replacement for human gold references.

Also notable: QCRI re-ran the wn2014 gate multi-ref, which will make our own
wn2014 gold-vs-candidate scoring slightly harsher single-references — worth
adopting their multi-ref verifier later.

## 2. Word-cache / n-gram lookup for OUR model — implemented + measured

New: diacritizer/scripts/e19_build_wordcache.py
- Built from OUR pool (sadeed_tashkeela train-*, abdou_tashkeel train-* 8
  shards; sadeed_25 row-schema mismatch skipped — TODO small): positional
  pairs (non_vocalized, vocalized) per row; **140.6M tokens -> 2,059,820
  unique stripped keys -> 375,923 cached forms** (majority >= 99.5% of votes,
  count >= 2 — same rules as zmahood's cache meta).

New: diacritizer/scripts/e20_goldcache_bench.py (gold B.predict_bare +
cache-merge on equal-length word positions).

| gate | gold raw DER | gold + our cache DER | delta |
|---|---|---|---|
| fadel2500 | 33.23 | 32.88 | -0.35 |
| sadeed2500 | 45.68 | 45.13 | -0.55 |
| wn2014 | 48.99 | 48.65 | -0.34 |
| abdou_heldout (2067) | 41.46 | 40.90 | -0.55 |

Honest read: **the cache helps, but tiny (~0.3-0.6 DER)**. Reason: our gold
already learns the same majority-vote word priors implicitly - the cache's
extra value over the model's own memorized forms is just when the model is
WRONG on a very frequent word. ZM's own cache gains ~0 on our gates too
(cache-On == cache-Off in E-19b). Conclusion: cache-lookup alone will NOT
close the 19-29 DER gap to the Z-Mahmood BiLSTM; it is a cheap deployment
add-on, not a teaching substitute.

## 3. A ~4.5M-param model in OUR architecture — YES, param math verified

build_from_config (diacritizer/src/model.py, GQA char-transformer):

| config (L-hidden-ffn-heads-kv, ctx 128) | params |
|---|---|
| our gold 28-272-1088-8-2 | 30.11M (matches the known 30M) |
| 12-128-512-4-2 | 2.98M |
| 10-144-576-4-2 | 3.14M |
| 8-128-512-4-2 | 2.00M |

Proposed E-21 "micro" run: **12 layers / hidden 128 / ffn 512 / heads 4 / kv 2**
(~2.98M params) or **10-144-576** (~3.14M) - either lands in Z-Mahmood's 4.5M
ballpark while keeping our RoPE/SwiGLU/GQA recipe. ctx 128 unchanged.
Expected to train ~5-8x faster per step than the gold's early schedule;
can train from scratch on the SAME stage2b pool (or add QCRI 5M-word wiki
corpus as extra weak data). Hardware cost: about 2.5 h on our GPU for a
2,500-step schedule at batch 32.

Decision needed from user before launching (training job = GPU hours):
- (a) micro 12-128-512 (~2.98M) vs (b) 10-144-576 (~3.14M) vs add ~1 layer to hit 4.5M
- (c) from-scratch vs SFT-stage2-style init from runs/diac/stage1lm/final
- (d) data: current stage2b pool only vs +QCRI wiki (after dedup + license OK)

## Artifacts

- models/e19/qcri-src/ (clone, untracked), our_word_cache.json
- models/e19/{gate}.gold.{raw,goldcache}.pred.txt (4 gates)
- diacritizer/scripts/{e19_build_wordcache,e20_goldcache_bench}.py
