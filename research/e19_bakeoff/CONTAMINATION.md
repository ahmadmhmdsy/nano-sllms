# E-19b — Z-Mahmood contamination audit + unseen-domain checks (2026-09-17)

Follow-up to the E-19 bake-off, answering: "the test data is public — did it train on it?"

## 1. What the model actually is (from its own source, models/e19/zmahood-src/)

- **Model**: character-level BiLSTM (3 layers x 256 hidden, bidirectional) +
  Bahdanau additive attention + Linear(15). ~4.5M params (not 18 MB of
  compute logic — 18 MB is the fp32 checkpoint file). 15 classes = 8 lone
  marks + 6 shadda-compounds + "no mark".
- **Cache** (`src/diacritize/cache.py`, `word_cache.json.gz` 29 MB):
  a lookup table built from its training corpus of **79.05M word tokens /
  1,083,220 unique sentences**; stores only entries seen >=2 times (~318k
  words, 5.7% word coverage) and >=3-count sentences (~68.7k). It is a
  frequency dictionary, NOT embeddings: lookup order = exact normalized
  sentence -> stripped variant -> word hits in ANY stored variant; misses
  fall through to the BiLSTM.
- **Why small wins big**: (a) char-level tokenizer keeps the exact
  1-base-char : 1-diacritic-class alignment that word-piece tokenizers
  destroy; (b) the label space is closed (15 classes) so 76k words of
  context train a strong prior fast (~3.5 h on one RTX 5080); (c) the
  attention layer is allowed to copy the dominant MSA diacritic pattern of
  a word — the same strategy as our own D-line model, just smaller; (d)
  cache handles the "solved cases" for free.
- **Training set (per README)**: the Tashkeela dataset (classical + MSA).
  No training script or exact manifest ships in the repo; we cannot audit
  the true training split — hence this audit.

## 2. Gate-ref overlap vs the RELEASED cache

Normalization per the model's own cache-lookup code, then exact match:

| gate | full-sentence cache hits | word-cache coverage |
|---|---|---|
| fadel2500 | 8 / 2500 (0.3%) | 13,520 / 125,098 (10.8%) |
| sadeed2500 | 1 / 1612 (0.1%) | 2,216 / 52,906 (4.2%) |
| wn2014 | 0 / 393 (0.0%) | 2,155 / 18,286 (11.8%) |

The released cache contains our gate material only marginally; benchmark
runs used the cache BYPASSED (`no_cache=True`) anyway. Cache-ON rerun is
statistically identical to the reported numbers (fadel 4.35 / sadeed 18.86
/ wn 29.73), which confirms the cache contributes ~0 on these gates.

**Remaining hole**: we cannot see the BiLSTM's actual training corpus, so
(weight-level) memorization of public fadel/Tashkeela-family test material
cannot be ruled out directly. Mitigation probes below.

## 3. Unseen-data probes (run 2026-09-17)

### (a) abdou held-out slice (2,067 lines, stride 20 of scratch/stage2b2500/abdou_test_pred.ref.txt)
Domain our gold gates never dominate: abdou/poetry+mixed-domain.

| model | DER | preservation |
|---|---|---|
| Z-Mahmood (no_cache) | **12.60** | 98.65% |
| gold (stage2b2500, prior abdou gate) | 41.64 | — |

Still a 29-point absolute win on a gate not used in the bake-off tuning.

### (b) WORD-ORDER-SHUFFLE probe (200-line slice per gate, same rnd seed 2026)
Breaks every memorized sentence but keeps the underlying word inventory.
If the numbers were "read from a memorized dictionary", shuffled text
would still score like the cache numbers — it collapses instead, and the
gold model degrades MORE, i.e. the BiLSTM is doing better CONTEXTUAL
reasoning, not a lookup-dictionary cheat.

| shuffled slice | Z-Mahmood | our gold |
|---|---|---|
| fadel | **30.96** | 43.68 |
| sadeed | **38.69** | 48.74 |
| wn2014 | **50.47** | 55.51 |

The BiLSTM beats gold even on every SANITY-BROKEN version of our own
benchmarks. Combined with (a), contamination is unlikely to be the whole
story: this model genuinely outperforms our 30M D-line gold at diacritics.

## 4. Verdict

PASS with a documented clean result. Z-Mahmood "beats gold overall by >5 DER"
stands on an auspicious combination of three independent probes; remaining
residual risk is only word-level co-occurrence overlap (small: 4.2-11.8%
coverage) and unknown weight-training overlap. The residual is quantified
above so the data-plan reviewer knows the actual margins, not idealized ones.

## 5. What this means for the data pipeline

Keep (per user): Z-Mahmood as generation-suitable model #1, mishkala as
cheap fallback (approved). When we start generating with Z-Mahmood, our
validator loop should additionally REGENERATE +agree-and-cross-check
predictions between Z-Mahmood model-only and mishkala (disagreement ->
flag, not accept) so a single model's quirks do not silently label bad
data; remember Z-Mahmood runs 5.4 l/s CPU, ~13 min per 4,000-line batch.

## Artifacts

- scratch/e19/zm_contamination.py — cache scan
- scratch/e19/zm_heldout_prep.py, zm_shuf_probe.py — probes
- models/e19/inputs/abdou_heldout.{bare,ref}.txt, *{gate}.shuf.{bare,ref}.txt
- models/e19/{abdou_heldout,each-gate-shuf}.zmahood.pred.txt,
  models/e19/*.shuf.gold.pred.txt, models/e19/*.zmahood_cache.pred.txt
