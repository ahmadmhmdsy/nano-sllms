# Arabic Datasets Catalog — index (2026-09)

Companion to `arabic_datasets.csv` (37 rows, UTF-8-BOM comma CSV). Full raw
Exa payloads: `research/raw/adc_*.json` (task list `research/adc_exa_tasks.json`).

## Top 15 rows by usefulness for the growth ladder

| # | Dataset | Ladder use | Why |
|---|---------|-----------|-----|
| 1 | Abdou Arabic Tashkeel (MIT, 3.72GB, 1.46M rows) | high | Largest staged, FT-cleared source (45% Tashkeela + 42% Shamela); Wikipedia slice synthetic — flag |
| 2 | Tashkeela via Misraj/Sadeed_Tashkeela (279.4M chars local) | high | Our G1/G2 workhorse; GPL-2 — redistribution = user decision |
| 3 | Fadel 2019 split (MIT, 37.7MB local txt) | high | Canonical eval; also G2 fill-in seed (hard-won) |
| 4 | QCRI Wikipedia Diacritized Corpus (~5M words, EMNLP 2025) | high | Largest free fully-vocalized MSA corpus (labels model-made); gold-source MSA — cheap |
| 5 | WikiNews-2014 multi-ref benchmark (local .diac) | high (eval) | The standard MSA gate (E-23 uses it) |
| 6 | WikiNews-2024 benchmark (local .diac) | high (eval) | Fresh 2024 gate; zero-holdout prep already wired |
| 7 | SadeedDiac-25 (1.2K paragraphs, local) | high (eval) | Cleanest 2025 MSA+CA benchmark; 46.8% overlap flagged EVAL-ONLY |
| 8 | CATT benchmark (Apache-2.0, 742 rows) | high (eval) | Perfect bare/vocalized pairs — ideal G2/G3 sanity palette |
| 9 | Tashkeela original (community-datasets/tashkeela, GPL-2, 1.08GB) | high | Book-level (97 docs) gold context field variant of #2 |
| 10 | Tashkeela arbml repack (1.58GB, GPL-2 upstream) | high | Sentence-split variant with clean held-out test |
| 11 | Roots-G5 ar_tashkeela (GPL-2, ROOTS) | high | Pre-sliced windows matching our G2 shakkelha style |
| 12 | Athar-Shamela4 (MIT, ~19GB classical extraction) | high (verify) | THE jackpot candidate: 19GB vocalized classical — must verify tashkeel rate |
| 13 | Shamela Repack ReligiousLLMs (fully vocalized classical) | high (dup) | Same content already in #1's 42% share — skip |
| 14 | Quranic Arabic Corpus (research-only GPL notes) | medium-high | Fully vocalized Quran + gold morphology for G2/G3 Quran rungs |
| 15 | qa-emmda eval / mOSCAR / wikimedia-aar | medium | Bare-text fill for G1 if mixture coarse-grained |

## License-risk flags (exact ids verified)

- **GPL-2 (viral)**: Tashkeela original, Misraj/Sadeed_Tashkeela (upstream),
  bigscience-data/roots_ar_tashkeela. Training a model is fine; redistributing
  the model or the corpus slice needs the user's explicit call.
- **CC-BY-ND 3.0**: Tanzil Quran text. No-derivatives on redistribution; use-in-training OK.
- **MIT**: Abdou tashkeel, AliOsm/Fadel repo, DIA2-Tashkeela-Gold, CAMeL code, ZAEBUC code, Kandil7 Athar-Shamela4 — but GPT-4o-mini-synthetic Wikipedia slice inside Abdou is not human gold.
- **CC-BY 4.0**: mOSCAR (bare), AraTweets (pain/Arabic-Tweets); **ODC-BY**: allenai/c4.
- **CC-BY-SA 3.0/GFDL**: wikimedia/wikipedia, wiki40b content.
- **no-license-file / unknown**: qcri/advancing-arabic-diacritization (provenance
  doc says CC terms per repo), Misraj/SadeedDiac-25 card, ArabicaQA, YAMAMA,
  ArabicCorpus2B ("other"), CALM/arwiki.
- **gated / research-only**: Misraj/mudd (manual), DIA2-Tashkeela-Gold (auto),
  KSUCCA (Sketch Engine), KACST KALD, Farasa models.

## What the catalog teaches

1. **Raw Arabic text is not the bottleneck** — bare-text oceans exist (mC4,
   OSCAR, wikimedia dumps). The bottleneck is **volume of reliable human-vocalized
   text**: we already stage ~38M chars (mu2 G1 = combined from Abdou + sadeed_tashkeela + qcri + fadel) and the next unlocking step is licensing-clean classical corpora beyond 100MB fully vocalized.
2. **Anything >100MB fully vocalized is the jackpot for our rungs** — only
   Abdou (3.72GB, verified) and sadeed_tashkeela (279MB, local) currently
   qualify, and the **Tashkeela tree (GPL-2) is behind a user-decision for
   anything beyond repo-internal training**. Athar-Shamela4 (~19GB, MIT) is the
   single most valuable unverified candidate; before deploying it we must split
   into one G1 pilot shard and run a tashkeel-rate gate on a sample window.
3. **Eval-only small benchmarks are the risky-zoniest to keep out of train**:
   SadeedDiac-25 (46.8% overlap flagged), and the E-23c ZM labels that already
   encode Sadeed/Fadel/WikiNews facts — the catalog reinforces our existing
   EVAL-ONLY gates.
