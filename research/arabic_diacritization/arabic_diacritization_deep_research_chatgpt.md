# Arabic Diacritization (تشكيل) Model — Deep Research Report

## Executive conclusion

The evidence strongly supports **building on existing work rather than training an Arabic diacritizer from scratch**. The task is best treated as character-level or byte-level conditional prediction: each Arabic letter receives a diacritic state, while punctuation, digits, Latin text, and existing user-supplied diacritics must be preserved.

Three implementation families dominate the open literature:

1. **Character-level sequence labeling / Transformer encoders** — especially CATT and recent compact character Transformers. This family gives exact input/output alignment, small vocabularies, efficient inference, and easy preservation of user-specified marks.
2. **Byte-level seq2seq models such as ByT5** — Fine-Tashkeel shows that pretrained byte-level models can reach very strong classical-Arabic results with relatively little task-specific training.
3. **RNN / hierarchical recurrence systems** — Shakkala, Shakkelha, and Deep Diacritization are strong historical baselines and remain useful references for data preparation, multi-diacritic labels, partially diacritized input, and efficiency.

For a new production-oriented project, the recommended first implementation is a **CATT-style character Transformer encoder**, initialized from a pretrained character model when available, with a 15-class diacritic head (or separate vowel + shadda heads), supporting partially diacritized input and a maximum sequence length of **1024 character positions**. The next step should be scaling to 2048 and 4096 positions with long-context/windowed training rather than replacing the model with a much larger LLM.

A second strong route is **Fine-Tashkeel/ByT5** as a quality baseline. A practical research program should train both a compact character model and a ByT5 baseline and compare them on the same multi-reference test sets.

The most important warning is evaluation leakage and domain mismatch. Tashkeela is overwhelmingly Classical/Islamic text, so a model can score extremely well on classical Arabic while underperforming on contemporary MSA. The 2025 QCRI work introduced WikiNews-2024 specifically to address this problem, and the SadeedDiac-25 benchmark further expands evaluation across MSA and Classical Arabic.

---

## 1. Problem definition

### Input

Arabic text that may be:

- completely undiacritized;
- partially diacritized;
- fully diacritized but requiring normalization or correction.

### Output

The same text with the correct Arabic diacritics restored while preserving:

- the original base letters;
- punctuation and whitespace;
- numbers and non-Arabic spans;
- user-provided diacritics when the application requires hard preservation.

### Why this is a difficult NLP task

Arabic diacritization is not just local vowel prediction. Correct internal vowels often depend on morphology and lexical context, while word-final case endings depend strongly on syntax. Ambiguous strings such as علم can have multiple valid readings depending on sentence context.

This is why the task benefits from context beyond a word, but it does **not** automatically require a huge language model. Several successful systems model the problem directly at the character level and obtain strong results with tens to hundreds of millions of parameters—or less.

---

# 2. Existing open-source systems

## 2.1 CATT — Character-based Arabic Tashkeel Transformer

**Repository:** https://github.com/abjadai/catt
**Paper:** https://arxiv.org/abs/2407.03236
**Current repository license:** Apache-2.0 after a stated license change.

CATT is one of the most relevant starting points for this project. It uses a character-based tokenizer and Transformer models in two configurations: encoder-only (EO) and encoder-decoder (ED). The authors first pretrained a character BERT, then fine-tuned EO and ED models for tashkeel, and finally used Noisy Student training to improve the best model. The paper reports state-of-the-art results against a set of commercial and open-source systems on two manually labeled benchmarks.

The released code is unusually useful for a new implementation because it contains the tokenizer, dataset code, training code, inference code, and ONNX export path. The training configuration exposes a `max_seq_len=1024`, batch size 32, and supports initialization from a pretrained character-BERT checkpoint. The released ED checkpoint uses a 3-layer Transformer; the EO model is deeper. The repository explicitly states that EO is faster while ED is more accurate.

**Why CATT is a strong base:**

- It matches the natural structure of Arabic diacritization: character → diacritic.
- A character vocabulary removes subword alignment problems.
- It already supports the user's initial 1,000-token/position scale.
- ONNX deployment is already demonstrated.
- It is much smaller and more controllable than a general-purpose decoder LLM.

For this project, CATT is the best codebase to study first.

---

## 2.2 Fine-Tashkeel — ByT5 / pretrained byte-level approach

**Model:** https://huggingface.co/basharalrfooh/Fine-Tashkeel
**Paper:** https://arxiv.org/abs/2303.14588

Fine-Tashkeel is important because it proves that diacritization does not have to be learned from a randomly initialized task-specific network. The paper fine-tunes pretrained token-free ByT5 models and reports strong results with minimal task-specific training and no hand-engineered linguistic features.

The released model reports approximately **0.95% DER and 2.49% WER** on its classical-Arabic benchmark, and the model card says it was fine-tuned for 13,000 steps on Tashkeela-derived data.

**Strengths:**

- pretrained knowledge transfers to diacritization;
- byte-level processing avoids Arabic subword fragmentation issues;
- very strong classical-Arabic baseline;
- straightforward Hugging Face integration.

**Weaknesses for the first production architecture:**

- seq2seq generation is more computationally expensive than direct character labeling;
- strict preservation of user-supplied diacritics is less natural than in a constrained classifier;
- generation can change more than the diacritics unless carefully constrained;
- the published model is specifically classical-Arabic, so its benchmark should not be treated as a universal Arabic score.

Recommendation: keep Fine-Tashkeel as a mandatory baseline, but do not make it the only architecture.

---

## 2.3 Shakkala

**Repository:** https://github.com/Barqawiz/Shakkala

Shakkala is a classic neural Arabic vocalizer built around character-level embeddings, three bidirectional LSTM layers, and dense layers. Its README says it was trained on more than one million sentences and that the majority of the data were historical Arabic books, with some modern internet text.

The 2019 Fadel benchmark measured Shakkala at **2.88% DER** on the released Fadel benchmark, substantially better than the Mishkal rule-based baseline reported in that paper.

Shakkala is valuable as a reproducible historical baseline and for understanding error patterns, but its RNN architecture is no longer the best place to invest new research effort.

---

## 2.4 Shakkelha

**Repository:** https://github.com/AliOsm/shakkelha

Shakkelha is the companion implementation released with Fadel et al.'s later work. The repository includes FFNN and RNN model families, CRF variants, comparison scripts, training logs, checkpoints, and an extra training dataset.

The open comparison results reported by OpenCodePapers show a Shakkelha result around **1.69% DER / 5.09% WER** on the historical Tashkeela benchmark.

This codebase is useful for understanding:

- sequence-labeling formulations;
- CRF-style decoding;
- class construction for Arabic diacritics;
- reproducible historical evaluation.

It is less attractive than CATT for a new Transformer-first implementation.

---

## 2.5 Deep Diacritization (D2 / D3)

**Paper:** https://arxiv.org/abs/2011.00538
**Repository:** https://github.com/BKHMSI/deep-diacritization

Deep Diacritization is particularly relevant because it explicitly models the hierarchy between **words and characters**. D2 uses a word-level recurrent encoder and independent character-level encoders, connected with cross-level attention. D3 adds a recurrent decoder and can consume **partially diacritized text** by feeding known previous diacritics into the decoder.

The paper reports a best WER of **5.34%**, with an efficiency advantage over fully character-level recurrent models.

The most important idea to carry forward is not the LSTM itself; it is the structural insight that:

> word-level semantic/syntactic context and character-level morphology can be represented separately and combined.

A modern Transformer version of this idea is worth testing later if a pure character Transformer plateaus.

---

## 2.6 Transformer Diacritization

**Repository:** https://github.com/mohammadKhalifa/transformer-diacritization

This older but very clear implementation frames Arabic diacritization as character-level sequence labeling and trains a 10-layer Transformer with embedding dimension 512, hidden dimension 512, 8 attention heads, maximum position 256, dropout 0.1, batch size 64, and learning rate 1e-4.

Its main value today is pedagogical: it shows that a directly trained Transformer works without a language-modeling pretraining phase and provides a clean bridge from older RNN systems to modern character Transformers.

---

## 2.7 Mishkal and Farasa

These are important non-neural baselines.

**Mishkal:** rule/dictionary/semantic candidate generation and ranking.

**Farasa:** QCRI's Arabic NLP toolkit contains a diacritization module plus a seq2seq diacritization service. The public Farasa page documents a standalone Java/JAR path and a REST API, but source/model access is more constrained than the fully open repositories above.

These tools remain useful as:

- fallback baselines;
- error analyzers;
- possible sources of weak labels;
- normalization/reference candidates.

They should not be the primary learned architecture.

---

## 2.8 Sadeed

**Paper:** https://arxiv.org/abs/2504.21635
**Dataset:** https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela
**Benchmark:** https://huggingface.co/datasets/Misraj/SadeedDiac-25

Sadeed is a major 2025 development because it asks whether a relatively small Arabic language model can be specialized for diacritization. It fine-tunes a decoder-only 1.5B Arabic language model on carefully curated diacritized data.

The important result for architecture selection is mixed. Sadeed demonstrated that a small Arabic LLM can perform the task, but the SadeedDiac-25 benchmark reports **7.29% DER with case endings and 5.26% without case endings**, plus a non-trivial hallucination rate. That is much weaker than the best specialized sequence-labeling systems on the same benchmark.

This is evidence against assuming that “larger generative model = better diacritizer.” For an exact character-preserving production system, a constrained character model remains a very attractive design.

---

## 2.9 Mishkala — very small 2026 model

**Model:** https://huggingface.co/flokymind/mishkala

Mishkala is a recent compact model reported at **12.5M parameters** using a hybrid Mamba + Transformer architecture with a CRF decoder. Its model card reports **1.66% DER** on the Sadeed test set.

This is highly relevant to the requirement “smaller but better.” The result is promising, although it should be treated as a model-card result until independently reproduced on the same evaluation protocol.

The lesson is encouraging: a small, task-specific model can be extremely competitive when the data and evaluation are good.

---

## 2.10 Current 2026 compact-Transformer evidence

A 2026 open-source project, `interscript/rababa`, reports a **~30M parameter, 6-layer, 384-dimensional, 6-head character Transformer** trained for 15 epochs on a combined corpus derived from cleaned Tashkeela plus Arabic Wikipedia and QCRI data. The repository reports approximately **0.99% DER** on a 52,224-example held-out set.

Its ablation is especially informative: a 75K-example corpus reportedly gives about **2.42% DER**, whereas a 2.1M-example corpus gives about **0.98–0.99% DER**. That is roughly a 28x data increase for a 2.5x error reduction. The repository also reports that paragraph context and domain adaptation helped, while later reinforcement-learning-style experiments did not materially improve the Arabic task.

This is not yet equivalent to a peer-reviewed benchmark result, so it should be treated as strong **engineering evidence**, not final scientific SOTA. It nevertheless strongly supports the idea that a ~30M parameter character Transformer plus high-quality data can be a very strong target.

---

# 3. What the existing systems teach us

| Family | Example | Input unit | Context | Strength | Main weakness |
|---|---|---|---|---|---|
| Rule/dictionary | Mishkal | word | mostly local + lexical | deterministic, interpretable | limited coverage/context |
| Classic RNN | Shakkala/Shakkelha | character | sentence | simple and proven | slower/older architecture |
| Hierarchical RNN | Deep Diacritization | word + character | sentence | efficient context modeling | recurrent architecture |
| Transformer | CATT | character | up to 1024+ positions | alignment, speed, deployment | needs careful pretrained/data strategy |
| Byte-level Seq2Seq | Fine-Tashkeel | bytes | long sequence | excellent pretrained transfer | heavier decoding, copy constraints |
| Decoder LLM | Sadeed | subword/LM | long | easy adaptation, broad knowledge | hallucination / copying problems |
| Compact modern hybrid | Mishkala | sequence | model-specific | very small footprint | independent validation still needed |

**Best fit for this project:** character Transformer encoder, with Fine-Tashkeel as a benchmark rather than the sole solution.

---

# 4. Arabic diacritization datasets

## 4.1 Tashkeela — the main backbone corpus

**Source paper:** https://doi.org/10.1016/j.dib.2017.01.011

Tashkeela remains the foundational open corpus. The original paper reports **75,629,921 words**, mostly Classical Arabic; 98.85% came from 97 Shamela books, and only about 1.15% was Modern Standard Arabic web/modern material. A later system paper reports more than 75.6M words and notes that more than 67.2M Arabic words are diacritized.

The corpus is therefore enormous, but its distribution is highly skewed toward Classical/Islamic text.

### Cleaning is mandatory

Published work identifies inconsistent formatting and some diacritization errors in the raw corpus. A documented cleaning pipeline removes empty/non-diacritic lines, splits sentences, fixes invalid diacritic combinations, removes malformed symbols, and filters sentences containing undiacritized words.

One published processed split contains roughly:

- 31.77M total training tokens;
- 1.76M validation tokens;
- 1.77M test tokens;
- 27.66M Arabic-word training tokens;
- 351K unique undiacritized Arabic training words.

This is a strong warning: **do not train directly on the raw Tashkeela archive.**

License must be reviewed carefully before commercial deployment; the commonly mirrored Tashkeela distribution is GPL-2.

---

## 4.2 Fadel Arabic Text Diacritization benchmark

**Repository:** https://github.com/AliOsm/arabic-text-diacritization

The Fadel benchmark is extracted and cleaned from Tashkeela and contains:

- 50,000 training lines;
- 2,500 validation lines;
- 2,500 test lines;
- roughly 2.3M words overall.

This dataset is extremely useful because it provides a stable historical baseline and has been used to compare Mishkal, Shakkala, Shakkelha and other systems.

It should be treated as a benchmark, not as the only training source.

---

## 4.3 Sadeed Tashkeela

**Dataset:** https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela

Sadeed provides a cleaned/curated representation of Tashkeela optimized for diacritization. It also provides evaluation tooling and a refined test set based on the Fadel benchmark.

This is probably the best “ready-to-run” Tashkeela-derived dataset for a first experiment because it avoids repeating much of the historical cleaning work.

The dataset is described as research-only and inherits terms from the original Tashkeela source, so licensing needs to be handled separately from pure technical suitability.

---

## 4.4 SadeedDiac-25

**Dataset:** https://huggingface.co/datasets/Misraj/SadeedDiac-25

SadeedDiac-25 contains **1,200 paragraphs**, intentionally mixing:

- 50% Modern Standard Arabic;
- 50% Classical Arabic;
- multiple domains including news, religion, politics, sports, and culinary content;
- expert-review and contamination-control procedures.

It reports results using DER and WER, with and without case endings.

This should be one of the primary evaluation sets for a new model because it addresses the biggest weakness of older Tashkeela-only benchmarks.

---

## 4.5 QCRI WikiNews-2024 and Wikipedia corpus

**Repository:** https://github.com/qcri/advancing-arabic-diacritization
**Paper:** https://aclanthology.org/2025.emnlp-main.846/

QCRI's 2025 work introduced:

- WikiNews-2014 with improved multi-reference annotations;
- WikiNews-2024, about 10K words, manually reviewed and multi-reference;
- a ~5M-word diacritized Arabic Wikipedia corpus generated by a strong BiLSTM system.

The key result is a BiLSTM system with WER of **3.12% on WikiNews-2014 and 2.70% on WikiNews-2024**.

The Wikipedia corpus is not gold human annotation; it is model-generated. Therefore it is better viewed as **weak/teacher-labeled data** rather than a perfect gold corpus.

This resource is nevertheless highly valuable for MSA domain adaptation.

---

## 4.6 DIA2 — new large MSA resource

**Paper:** OSACT7/LREC-COLING 2026
**Dataset:** https://huggingface.co/datasets/DIA2-Arabic/DIA2-Dataset-Non-Diacritized
**Gold subset:** https://huggingface.co/datasets/DIA2-Arabic/DIA2-Tashkeela-Gold

DIA2 is particularly relevant to this project because it explicitly targets diverse **native Modern Standard Arabic**, with books, news, encyclopedic content, and poetry, while avoiding machine-translated content.

The full DIA2 release is reported at more than **140 GB**, more than **26M unique words**, and about **41.9B tokens**. The preprocessing includes aggressive cleaning, deduplication, automatic diacritization, and a gold diacritized subset.

This is a powerful large-scale source, but it is **not the first dataset to train on**. Its scale is useful after the pipeline is proven. The gold subset is more important for evaluation and high-quality fine-tuning.

---

## 4.7 Abdou Arabic Tashkeel Dataset

**Dataset:** https://huggingface.co/datasets/Abdou/arabic-tashkeel-dataset

This is a useful aggregation containing about **1.5M rows / 1.76GB**, built from:

- Tashkeela;
- Shamela;
- Wikipedia;
- Arabic poetry datasets;
- Quran riwayat;
- a hadith corpus.

One important limitation is that its Wikipedia diacritics were generated using GPT-4o mini, and the dataset is more than 90% classical/religious text. Therefore it is best classified as a **mixed-quality training resource**, not a gold benchmark.

It may be useful for experimentation after stronger sources have been validated.

---

## 4.8 Arabic poetry / APCD / Ashaar

**APCD:** https://huggingface.co/datasets/arbml/APCD

APCD contains about **1.83M rows** of diacritized poetry. The broader Ashaar corpus contains millions of Arabic poetry lines. A newer `ashaar-tashkeel` dataset contains about **6.93M model-diacritized poetry verses**, but its labels were produced by Fine-Tashkeel rather than manually verified.

These datasets are valuable for a poetry-specific model or a second-stage domain adaptation model. They should not dominate a general MSA diacritizer, because poetry has its own vocabulary, orthography, and syntactic conventions.

Licensing is also restrictive for the original Ashaar corpus; it should not be assumed suitable for commercial training merely because a derived Hugging Face copy exists.

---

## 4.9 Quran and Hadith

Quranic data are exceptionally useful because the Arabic is highly vocalized and the text is relatively small and well constrained. However, Quranic Arabic is a specialized domain and should be treated as **high-quality supplemental data**, not as representative general Arabic.

The `arbml/LK_Hadith` resource provides 34K hadith records and can also be useful for religious-domain coverage.

For licensing, each underlying source must be checked individually; Quran datasets frequently combine text and derived annotations under different terms.

---

# 5. Dataset strategy: large corpus vs smaller high-quality corpus

The research points to a clear answer: **quality and domain balance matter more than raw volume once a reasonable amount of clean data exists.**

The best training mixture for a general-purpose Arabic diacritizer is therefore not “all of Tashkeela.” It is a staged mixture:

### Tier A — Gold / highest confidence

Use for evaluation and late-stage fine-tuning:

- SadeedDiac-25;
- WikiNews-2024;
- manually verified CATT benchmark;
- carefully selected Fadel test/training lines;
- verified Quran / Hadith subsets if the target domain includes them.

### Tier B — Clean human-vocalized training data

Use as the main supervised corpus:

- cleaned Tashkeela / Sadeed Tashkeela;
- verified poetry such as APCD;
- high-quality diacritized books and native Arabic sources.

### Tier C — Teacher-labeled / weak data

Use only after the model already works:

- QCRI's model-diacritized Wikipedia;
- model-labeled Ashaar;
- GPT/LLM-diacritized corpora;
- large automatic-diacritization expansions.

Weak data should never be mixed blindly with gold data. Track provenance and give gold samples higher sampling probability or loss weight.

---

# 6. Model architecture recommendation

## Recommended first model: Character Transformer Encoder

### Input representation

Normalize the Arabic text while preserving the base string. Convert it to character IDs from a small vocabulary containing:

- Arabic letters;
- Arabic presentation/hamza variants as appropriate;
- whitespace;
- punctuation;
- digits;
- special tokens.

Represent existing diacritics separately rather than stripping them without trace.

### Output representation

A robust initial label set is 15 classes:

1. no diacritic
2. fatha
3. damma
4. kasra
5. fathatan
6. dammatan
7. kasratan
8. sukun
9. shadda
10. shadda + fatha
11. shadda + damma
12. shadda + kasra
13. shadda + fathatan
14. shadda + dammatan
15. shadda + kasratan

An alternative is to predict two heads:

- primary vowel/ending;
- shadda flag.

The two-head formulation is attractive because Arabic can place shadda together with another diacritic, and separating the factors makes the task more structured.

### Recommended initial configuration

- 6 Transformer encoder layers
- hidden size: 384
- 6 attention heads
- feed-forward dimension: about 4× hidden size
- dropout: 0.1
- maximum sequence: 1024 character positions
- LayerNorm + residual connections
- learned or sinusoidal position embeddings
- linear classification head, optionally CRF

This is approximately the same scale as the successful 2026 compact-character evidence (~30M parameters), while remaining small enough for consumer GPUs.

### Loss

Start with weighted cross-entropy over diacritic classes.

Then test:

- label smoothing;
- focal loss for rare combinations;
- auxiliary shadda loss;
- constrained copy/preservation loss for already-diacritized characters.

Avoid adding CRF immediately. First establish whether the Transformer itself is sufficient. CRF can be an ablation.

---

# 7. Handling partially diacritized input

This is one of the most important product requirements and should be built into the architecture from day one.

Suppose the input is:

`كَانَ الطَّالِبُ يَذْهَبُ إلى المدرسة`

Some tokens may be complete, some partially marked, and some bare.

The model should receive **two aligned streams**:

1. the base character sequence;
2. a known-diacritic mask / existing-diacritic encoding.

For an already supplied diacritic:

- encode it as an observed feature;
- either force the output to match it, or strongly bias the corresponding class;
- never allow the model to silently replace user-provided marks unless operating in an explicit “correct/rewrite” mode.

This is preferable to feeding fully diacritized strings as ordinary characters because the model then learns that some marks are observations/constraints, not targets to reinvent.

Deep Diacritization's D3 is strong evidence that partial diacritics can be incorporated explicitly into the model.

---

# 8. Do we actually need a huge context window?

Not necessarily.

### 1,000-token/position starting point

Your 1,000-token requirement is sensible as a first experiment, but for a character-level model it should be interpreted carefully. A character Transformer sees **characters**, not LLM subword tokens.

A better specification is:

> Start with a 1024-character maximum sequence length.

That is close to the released CATT configuration and is a meaningful sentence/paragraph-scale context for Arabic.

### Scaling plan

Use:

- Stage 1: 256–512 characters for fast debugging;
- Stage 2: 1024 characters for the baseline;
- Stage 3: 2048 characters;
- Stage 4: 4096 characters;
- Stage 5: paragraph windowing / long-context attention if needed.

Do not jump immediately to 8K–32K context. Diacritics are predominantly resolved by local word structure plus sentence/paragraph syntax, so the best experiment is to measure the marginal benefit of context length.

Recent 2026 engineering results are particularly interesting here: paragraph context improved one benchmark but slightly hurt the cross-domain WikiNews benchmark, suggesting a real domain/context trade-off rather than “more context is always better.”

---

# 9. Training strategy

## Stage 0 — build the evaluation harness first

Before training the final model, create a deterministic evaluator supporting:

- DER with case endings;
- DER without case endings;
- WER with case endings;
- WER without case endings;
- percentage of changed base characters;
- user-diacritic preservation accuracy;
- hallucination / text-change rate.

Use at least:

- Fadel test;
- WikiNews-2024;
- SadeedDiac-25;
- CATT benchmark if accessible.

Do not report one aggregate number only.

---

## Stage 1 — tiny sanity model

Use approximately 50K–200K high-quality examples.

Train:

- 4–6 layer character Transformer;
- 256–384 hidden dimensions;
- 512–1024 character context.

Goal: confirm data alignment and loss implementation, not final quality.

Expected outcome: the model should quickly learn deterministic patterns such as:

- common prefixes/suffixes;
- definite article behavior;
- shadda patterns;
- frequent morphological forms;
- local vowel combinations.

---

## Stage 2 — main supervised training

Start with Sadeed/Tashkeela-clean data.

Recommended mixture:

- 70–80% clean Tashkeela-derived data;
- 10–20% MSA data from QCRI/DIA2/gold sources;
- 5–10% high-confidence specialized material.

Do not simply concatenate everything. Use domain-balanced sampling so that Classical Arabic does not overwhelm MSA.

Train until validation DER plateaus rather than selecting by loss alone.

---

## Stage 3 — domain balancing

Introduce clean MSA/news sources.

This stage is essential because raw Tashkeela is overwhelmingly Classical Arabic. QCRI's 2025 work directly demonstrates the value of refined data and modern multi-reference evaluation.

A good experiment is to compare:

A. Tashkeela-only;
B. Tashkeela + MSA;
C. Tashkeela + MSA + poetry;
D. Tashkeela + MSA + weak-label Wikipedia.

Measure each on both CA and MSA benchmarks.

---

## Stage 4 — weak-label expansion

Only after the clean model is strong should you add:

- QCRI's model-diacritized Wikipedia;
- model-diacritized poetry;
- automatically diacritized native MSA.

Use confidence filtering. A weak label should be accepted only if:

- teacher confidence is high;
- the output preserves the input skeleton;
- the teacher does not introduce or remove base characters;
- the sentence passes normalization checks.

Distinguish gold vs weak examples in metadata.

---

## Stage 5 — partially diacritized curriculum

Create synthetic mixed-input examples from the gold corpus.

For every fully diacritized target, generate inputs such as:

- 0% known diacritics;
- 10% known;
- 25% known;
- 50% known;
- 75% known;
- random word-level spans known.

Train the model to preserve known marks while restoring unknown ones.

This will make the production model much more robust than a model trained only on completely bare text.

---

## Stage 6 — hard-example mining

After the first strong model, collect errors such as:

- rare lexical forms;
- ambiguous internal vowels;
- case endings;
- proper names;
- foreign names embedded in Arabic text;
- numerals;
- poetry;
- punctuation edge cases;
- Quranic special marks;
- partially diacritized user text.

Oversample difficult examples instead of simply scaling total corpus size.

---

# 10. Pretraining from scratch vs fine-tuning

## Do not pretrain a large Arabic LLM just for this task

There is little evidence that creating a new 100M–1B Arabic language model from scratch is the best first move for tashkeel.

The strongest engineering options are:

### Option A — fine-tune a pretrained character encoder

Best quality/compute trade-off if a good Arabic/character checkpoint is available.

### Option B — train a compact character Transformer from scratch

Best experimental control and potentially best deployment model.

### Option C — fine-tune ByT5

Best “strong baseline with minimal research code.”

### Option D — fine-tune a small decoder LLM

Useful for comparison, but not the architecture I would choose for the primary production model because of copying/hallucination concerns.

---

# 11. Recommended experiments

## Experiment matrix

| Exp | Architecture | Params target | Data | Context | Purpose |
|---|---|---:|---|---:|---|
| A | CATT-style EO | ~10–30M | Fadel/Sadeed subset | 512 | sanity baseline |
| B | CATT-style EO | ~30M | Sadeed/Tashkeela | 1024 | primary baseline |
| C | CATT-style EO | ~30M | Sadeed + MSA | 1024 | domain balancing |
| D | CATT-style EO | ~30M | clean + weak MSA | 1024 | weak supervision |
| E | same model | ~30M | best mix | 2048 | context scaling |
| F | same model | ~30M | best mix | 4096 | long-context test |
| G | ByT5 baseline | 0.3–0.6B class | Tashkeela | 1024+ | pretrained seq2seq comparison |
| H | small hybrid/CRF | ~10–30M | best mix | 1024 | structured decoder test |

This produces an evidence-based answer to whether a tiny specialist can beat a larger pretrained model.

---

# 12. Metrics that matter

### DER — Diacritic Error Rate

Primary token/character-level accuracy measure.

### WER — Word Error Rate

Much more meaningful for product quality because a single wrong diacritic can make a word wrong.

### Case-ending excluded score

Essential because word-final grammatical case is an unusually difficult part of Arabic diacritization and is also one of the areas where multiple valid readings can exist in some evaluation settings.

### Multi-reference scoring

Use multi-reference benchmarks wherever possible. QCRI's WikiNews-2024 and SadeedDiac-25 were designed specifically to handle legitimate variation better than older single-reference tests.

### Preservation score

For partially diacritized input, add:

`known_diacritic_accuracy = preserved_known_marks / known_input_marks`

This should be a hard product requirement.

### Hallucination rate

Count cases where the model:

- changes base Arabic letters;
- changes punctuation unexpectedly;
- drops words;
- inserts words.

A character classifier should approach zero such errors by construction.

---

# 13. Preprocessing pipeline

The most reusable production pipeline should be:

1. Unicode normalization.
2. Detect and preserve Arabic base characters.
3. Separate base letters from combining diacritics.
4. Normalize equivalent hamza/alif forms only if the project specification demands it.
5. Preserve punctuation and whitespace offsets.
6. Produce `(base_text, observed_diacritics, target_diacritics)` triples.
7. Validate alignment between base characters and diacritic labels.
8. Remove or quarantine corrupt samples.
9. Deduplicate at document level before train/test splitting.
10. Split by document/source, not random sentence, whenever possible.
11. Track domain and provenance metadata.

The last two steps are critical. Randomly splitting neighboring sentences from the same book can make the benchmark unrealistically easy.

---

# 14. A better dataset than “all 75M Tashkeela words”

A high-quality first release could be much smaller than the raw Tashkeela corpus.

A reasonable first curated training corpus might contain roughly **10–20M high-confidence Arabic words**, balanced approximately as:

- 55–65% Classical Arabic;
- 25–35% MSA/news/general prose;
- 5–10% poetry or specialized domains.

Then create a very strong validation/test suite of perhaps **100K–300K words** sampled by domain, source, morphology, and difficulty, with multi-reference handling where possible.

The goal should not be “largest dataset.” The goal should be:

> **largest clean, diverse, non-leaky, linguistically trustworthy dataset that the model can actually learn from.**

The 2026 compact-model evidence also suggests strong returns from scaling clean examples from tens of thousands into the millions, but diminishing returns should be measured rather than assumed.

---

# 15. Proposed project architecture

```text
                 ┌────────────────────────┐
                 │ Raw Arabic input       │
                 └───────────┬────────────┘
                             │
                   Unicode / text parser
                             │
                 ┌───────────▼────────────┐
                 │ Base-character stream  │
                 │ + observed-diacritic   │
                 │   stream                │
                 └───────────┬────────────┘
                             │
                    Character tokenizer
                             │
                    1024-position window
                             │
                 ┌───────────▼────────────┐
                 │ Character Transformer  │
                 │ 6L × 384d × 6 heads   │
                 └───────────┬────────────┘
                             │
                    ┌────────▼────────┐
                    │ Diacritic heads │
                    │ vowel + shadda  │
                    └────────┬────────┘
                             │
                  known-mark constraints
                             │
                 ┌───────────▼────────────┐
                 │ Reconstruction layer  │
                 │ restore original text  │
                 └───────────┬────────────┘
                             │
                   Fully diacritized text
```

---

# 16. Why this architecture is preferable to a generic LLM

A specialized diacritizer has a strong structural advantage:

- the base text should not change;
- only diacritic labels change;
- the output vocabulary is tiny;
- alignment is deterministic;
- inference can be parallelized across all positions;
- long output generation is unnecessary;
- quantization is straightforward;
- ONNX/TensorRT-style deployment is possible.

A generic decoder LLM must solve a harder problem than necessary: generate the entire string while also deciding the diacritics. This creates failure modes unrelated to the target task.

This is one of the most important design conclusions from the current literature.

---

# 17. Recommended implementation roadmap

## Phase 1 — Reproduce existing models

1. Run CATT inference.
2. Run Fine-Tashkeel inference.
3. Run Shakkelha/Shakkala.
4. Run Mishkal/Farasa as non-neural baselines.
5. Run Mishkala if the environment permits.

Build one evaluation script around all of them.

## Phase 2 — Build the first custom model

Fork/study CATT but simplify the stack into a clean PyTorch project:

- tokenizer;
- alignment validator;
- dataset loader;
- character encoder;
- two-head diacritic classifier;
- training loop;
- evaluator;
- inference pipeline;
- ONNX exporter.

## Phase 3 — Train on clean data

Use Sadeed/Tashkeela-clean as the main source and add a controlled MSA slice.

## Phase 4 — 1024-character model

Make 1024 positions the first serious baseline.

## Phase 5 — 2048/4096 scaling

Measure the gain in DER/WER against GPU memory and latency.

## Phase 6 — weak supervision

Add QCRI Wikipedia and other teacher-labeled sources only after filtering.

## Phase 7 — hard-example learning

Use model errors to construct a curated challenge set and retraining loop.

---

# 18. Final recommendation

### Should a new model be trained from scratch?

**Not as the primary path.**

Start from or reproduce **CATT-style character Transformers**, and benchmark against Fine-Tashkeel. A custom model from scratch is still worthwhile as a research experiment because the model can be tiny and highly constrained, but it should be compared against pretrained initialization rather than assumed superior.

### Should we use 75M+ words?

**Eventually, but not blindly.**

Use cleaned Tashkeela as the foundation, then explicitly add MSA and diverse domains. The evidence strongly suggests that data quality, source balance and evaluation design matter more than simply throwing all available text into training.

### Can a much smaller model beat a larger model?

**Yes, plausibly.**

A 12.5M-parameter hybrid model is already reporting 1.66% DER on a Sadeed test set, while a 30M-parameter character Transformer project reports around 0.99% DER on its own held-out corpus. These are not yet apples-to-apples scientific comparisons, but they demonstrate that a specialist model does not need billions of parameters.

### What should the first serious model be?

**~30M parameter character Transformer, 1024 character positions, dual diacritic heads, partially-diacritized input support, trained on cleaned Tashkeela/Sadeed plus an MSA-balanced corpus.**

### What should be the first benchmark?

Use at least:

- Fadel benchmark;
- WikiNews-2024;
- SadeedDiac-25;
- CATT benchmark.

### What is the most promising path toward “small but superior quality”?

**High-quality character-level supervision + domain-balanced data + pretrained character initialization + hard-example mining + partial-diacritic constraints.**

This is a more promising route than building a large Arabic generative model solely for tashkeel.

---

# 19. Priority source list

1. Al-Rfooh, Al-Rfou, Abandah. *Fine-Tashkeel: Finetuning Byte-Level Models for Accurate Arabic Text Diacritization* (2023). https://arxiv.org/abs/2303.14588
2. Alasmary, Zaafarani, Ghannam. *CATT: Character-based Arabic Tashkeel Transformer* (2024). https://arxiv.org/abs/2407.03236
3. Mohamed, Mubarak. *Advancing Arabic Diacritization: Improved Datasets, Benchmarking, and State-of-the-Art Models* (EMNLP 2025). https://aclanthology.org/2025.emnlp-main.846/
4. Aldallal et al. *Sadeed: Advancing Arabic Diacritization Through Small Language Model* (2025). https://arxiv.org/abs/2504.21635
5. AlKhamissi, ElNokrashy, Gabr. *Deep Diacritization: Efficient Hierarchical Recurrence for Improved Arabic Diacritization* (2020). https://arxiv.org/abs/2011.00538
6. Fadel et al. *Arabic Text Diacritization Using Deep Neural Networks* (2019). https://arxiv.org/abs/1905.01965
7. Taha Zerrouki. *Tashkeela: Novel corpus of Arabic vocalized texts, data for auto-diacritization systems* (2017). https://pmc.ncbi.nlm.nih.gov/articles/PMC5310197/
8. Fadel et al. *Neural Arabic Text Diacritization: State of the Art Results and a Novel Approach for Machine Translation* (2019). https://aclanthology.org/D19-5229/
9. CATT official repository. https://github.com/abjadai/catt
10. Shakkelha official repository. https://github.com/AliOsm/shakkelha
11. Shakkala official repository. https://github.com/Barqawiz/Shakkala
12. Deep Diacritization repository. https://github.com/BKHMSI/deep-diacritization
13. QCRI Arabic diacritization datasets/tools. https://github.com/qcri/advancing-arabic-diacritization
14. Sadeed Tashkeela dataset. https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela
15. SadeedDiac-25. https://huggingface.co/datasets/Misraj/SadeedDiac-25
16. DIA2 dataset. https://huggingface.co/datasets/DIA2-Arabic/DIA2-Dataset-Non-Diacritized
17. DIA2 Gold. https://huggingface.co/datasets/DIA2-Arabic/DIA2-Tashkeela-Gold
18. Fadel benchmark repository. https://github.com/AliOsm/arabic-text-diacritization
19. Abdou Arabic Tashkeel dataset. https://huggingface.co/datasets/Abdou/arabic-tashkeel-dataset
20. CATT Tashkeel package. https://pypi.org/project/catt-tashkeel/
21. Farasa diacritization. https://farasa-api.qcri.org/diacritization/
22. Farasa seq2seq diacritization. https://farasa.qcri.org/seq2seq_diacritization/
23. 2026 compact-model engineering results: https://github.com/interscript/rababa/blob/main/docs/RESULTS.md
24. Mishkala model card. https://huggingface.co/flokymind/mishkala

