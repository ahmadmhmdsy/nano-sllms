<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Overview

You can absolutely build on existing open‑source Arabic diacritization work instead of starting from scratch. There are several strong baseline models (CATT, Shakkala, Fine‑Tashkeel, Turath, etc.), clean benchmark datasets (WikiNews, Tashkeela‑derived corpora, Wikipedia diacritized corpus), and ready‑to‑use codebases you can fine‑tune or adapt.[^1_1][^1_2][^1_3][^1_4][^1_5]

Below is a focused research summary: what exists, how they’re built, what datasets you can reuse, and a concrete path to train a small but high‑quality model starting at ~1k tokens and scaling up.

______________________________________________________________________

## 1) State‑of‑the‑art open‑source diacritizers

| Model / Project | Architecture | Key traits | License / Code | Reported quality |
| :-- | :-- | :-- | :-- | :-- |
| **CATT** (Character‑based Arabic Tashkeel Transformer) | Transformer encoder + linear head (character‑level, non‑autoregressive) | SOTA on several benchmarks; character‑level; handles long context better than fixed‑length LSTMs; used in production TTS/ASR pipelines | Apache‑2.0 (abjadai/CATT); inference code vendored in several repos | DER in low single digits on WikiNews/Tashkeela‑style benchmarks [^1_2][^1_6][^1_7] |
| **Fine‑Tashkeel** | Fine‑tuned ByT5 (seq2seq) | Strong seq2seq baseline; 40% WER reduction reported vs earlier baselines in shared tasks | Code released for KSAA‑2026 task (Jupyter/Python) [^1_2][^1_8] | Competitive WER/DER in KSAA‑2026 evaluations [^1_8] |
| **Shakkala** | 2‑layer BiLSTM (character‑level) + embedding | Very clean character‑level sequence labeling; simple, fast; input capped at 315 chars per segment | Code/descriptions public; model weights on HF (e.g., TigreGotico/shakkala‑diacritizer) [^1_2][^1_9][^1_4] | DER ≈ 2.88% on cleaned Tashkeela benchmark [^1_4] |
| **Turath1.0** | 6‑layer Transformer encoder (~8M params) | Lightweight, CPU‑friendly; trained on Shamela classical corpus; max 256 chars | MIT; HF model + inference script [^1_5] | DER 7.1% on Shamela test split; ~5ms CPU latency/sentence [^1_5] |
| **arabic‑diacritizer (PyPI)** | BiLSTM + Bahdanau attention + sentence cache | Claims ~6.6% DER; beats some large LLM baselines on their test | MIT/Apache‑style; GitHub + pip installable [^1_1] | DER 6.6% reported [^1_1] |
| **Mishkal** | Rule‑based + dictionary lookups | Deterministic, fast, good for MSA; struggles with ambiguity | Open source (Java/Python wrappers) [^1_2] | Higher DER than neural SOTA, but useful as a fallback/post‑processor [^1_2] |

**Takeaway:** For a new project, CATT (encoder‑only, character‑level) and ByT5‑style seq2seq (Fine‑Tashkeel) are the best foundations if you want modern accuracy and room to scale context. Shakkala/Turath are excellent if you want small, fast models you can train or fine‑tune quickly.[^1_2][^1_4][^1_5][^1_8]

______________________________________________________________________

## 2) How these models are built (patterns you can reuse)

Most top systems follow one of two patterns:

### A) Character‑level sequence labeling (Shakkala / BiLSTM style)

- **Input:** Raw Arabic characters (with or without existing diacritics stripped).
- **Vocab:** ~60–130 tokens (Arabic letters, punctuation, UNK, PAD).
- **Model:** Embedding → BiLSTM (1–2 layers) → Dense → softmax over diacritic classes per character.
- **Output:** One label per character (e.g., no‑diacritic, fatha, kasra, damma, tanwin, shadda combos).
- **Constraints:** Often fixed max length (e.g., 315 chars); longer text is chunked by sentence.[^1_4][^1_9]


### B) Transformer encoder / seq2seq (CATT / Fine‑Tashkeel / Turath)

- **Encoder‑only (CATT, Turath):** Character or subword tokens → Transformer encoder → per‑token classifier. Non‑autoregressive, fast inference.[^1_5][^1_10][^1_2]
- **Seq2seq (Fine‑Tashkeel, ByT5):** Input text → encoder–decoder generates fully diacritized text. Good for long‑range dependencies; can be fine‑tuned with parameter‑efficient methods.[^1_8][^1_2]
- **Diacritic label set:** Typically 12–18 classes covering: {no‑diacritic, َ ُ ِ ً ٌ ٍ ْ ّ and shadda+vowel combos}. Some models also predict case endings explicitly.[^1_9][^1_4][^1_5]

**Common preprocessing steps**

- Normalize Arabic orthography (unify alef forms, remove tatweel, standardize punctuation).
- Strip existing diacritics from input for training pairs (raw → fully diacritized target).
- Segment long documents into sentences/lines respecting max sequence length.
- Optionally inject noise (spelling errors, transliterated words) to improve robustness.[^1_3][^1_11][^1_4]

______________________________________________________________________

## 3) Datasets you can use (and how to start small)

### High‑quality public corpora

- **Tashkeela Corpus** (classical + MSA): ~2.3M words after cleaning; widely used benchmark source.[^1_4]
- **Wikipedia Diacritized Corpus** (QCRI, 2025): ~5M words from 32k+ articles, fully diacritized by a strong BiLSTM; JSONL format.[^1_3]
- **WikiNews Benchmarks** (2014 \& 2024): Multi‑reference annotated test sets (~10k words each) for fair evaluation.[^1_3]
- **Shamela‑based corpora**: Classical Islamic texts; used by Turath1.0; good for religious/classical domains.[^1_5]
- **Quranic recitation datasets** (e.g., everyayah, quran‑recitations‑asr): Ayah‑level audio + fully diacritized text; useful if you later want speech‑aware training or domain adaptation.[^1_12][^1_13]
- **ClArTTS** (Classical Arabic TTS): 12h, 9.5k utterances, fully diacritized; smaller but high quality.[^1_14]


### Starting with ~1,000 tokens

You don’t need to train on millions of tokens immediately. A practical plan:

- Sample 1,000–2,000 short sentences (≈1k–2k tokens) from Tashkeela or Wikipedia corpus.
- Ensure diversity: news, general prose, some classical if your target domain needs it.
- Use this small set to validate your pipeline (preprocessing, model, eval script) before scaling.[^1_4][^1_3]

Later, scale to 100k–1M+ tokens from the same sources, or mix domains (news + classical) depending on your use case.

______________________________________________________________________

## 4) Can you build on existing models instead of from scratch?

Yes—and it’s recommended.

**Best candidates to fine‑tune**

- **CATT (abjadai/CATT)**: Character‑level transformer encoder; strong SOTA; permissive license; already used in production ASR/TTS pipelines. Ideal if you want accuracy and the ability to increase context length.[^1_6][^1_7][^1_10][^1_2]
- **Fine‑Tashkeel (ByT5)**: If you prefer seq2seq generation and want to experiment with parameter‑efficient fine‑tuning (LoRA, etc.).[^1_2][^1_8]
- **Turath1.0**: If you want a tiny, fast model (~8M params) you can retrain quickly on your own domain (e.g., classical texts).[^1_5]
- **Shakkala‑style BiLSTM**: If you want maximum simplicity and very small models; good baseline and teaching model.[^1_9][^1_4]

**When to train from scratch**

- Your domain is very specialized (e.g., dialectal Arabic, poetry with strict meter, legal texts) and existing models underperform even after fine‑tuning.
- You need unusual output labels (e.g., explicit morphological tags alongside diacritics).

In most cases, start by **fine‑tuning CATT or a ByT5 variant**, then consider distilling into a smaller model (like Turath‑size) for deployment.[^1_8][^1_2][^1_5]

______________________________________________________________________

## 5) Context window and scaling strategy

- **Character‑level models** naturally handle longer raw text if you increase max sequence length, but VRAM and attention cost grow quadratically. CATT‑style encoders are a good base to experiment with longer contexts.[^1_10][^1_2]
- **Practical approach**:
    - Start with 256–512 characters per segment (≈1k tokens depending on tokenization).
    - Chunk documents at sentence boundaries; recombine outputs.
    - As you scale data, increase max length and/or move to a more efficient attention variant if needed.[^1_4][^1_5]

Your plan to begin with ~1,000 tokens and scale is aligned with how these projects evolved: small clean subsets first, then larger corpora (Wikipedia 5M words, Tashkeela 2.3M words).[^1_3][^1_4]

______________________________________________________________________

## 6) Concrete next steps (actionable)

1. **Pick a base model**
    - For best accuracy + future scaling: **CATT**.
    - For smallest trainable model: **Turath1.0** or a **Shakkala‑style BiLSTM**.[^1_10][^1_2][^1_5][^1_4]
2. **Assemble a mini‑dataset (~1k tokens)**
    - Sample sentences from **Tashkeela** and/or **Wikipedia Diacritized Corpus**.
    - Create pairs: `raw_text` (diacritics stripped) → `diacritized_text`.[^1_3][^1_4]
3. **Implement preprocessing**
    - Normalize Arabic characters.
    - Strip diacritics from input.
    - Segment into max‑length chunks (e.g., 256–512 chars).
    - Map to character IDs and diacritic labels (12–18 classes).[^1_9][^1_5][^1_4]
4. **Fine‑tune the base model**
    - Use cross‑entropy over per‑character diacritic classes.
    - Track **DER** (Diacritic Error Rate) and **WER** (Word Error Rate) using the QCRI scoring script or equivalent.[^1_4][^1_3]
5. **Iterate and scale**
    - Once the 1k‑token pipeline is stable, expand to 100k–1M+ tokens.
    - Optionally distill a large CATT/ByT5 model into a smaller Transformer (Turath‑size) for deployment.[^1_2][^1_8][^1_5]

______________________________________________________________________

## 7) Key repositories and resources to clone

- **QCRI diacritization datasets \& eval script**: [https://github.com/qcri/advancing-arabic-diacritization](https://github.com/qcri/advancing-arabic-diacritization)[^1_3]
- **CATT models \& code** (via Hugging Face and related repos): search “CATT Arabic diacritization” on HF; also used in NightPrince’s STT+diacritization pipeline.[^1_7][^1_6][^1_10]
- **Fine‑Tashkeel (KSAA‑2026)**: [https://github.com/mzx93M/KSAA2026-Fine-Tashkeel](https://github.com/mzx93M/KSAA2026-Fine-Tashkeel)[^1_8]
- **Shakkala / BiLSTM diacritizers**: Hugging Face models like `TigreGotico/shakkala-diacritizer`, `TigreGotico/catt-diacritizer`.[^1_10][^1_9]
- **Turath1.0 (lightweight Transformer)**: [https://huggingface.co/yuujiElfahkrany/turath1.0](https://huggingface.co/yuujiElfahkrany/turath1.0) (includes inference.py and training details).[^1_5]
- **Awesome Arabic NLP** (curated list including diacritizers): [https://github.com/h9-tec/Awesome_Arabic_NLP](https://github.com/h9-tec/Awesome_Arabic_NLP)[^1_2]

If you tell me your preferred stack (PyTorch vs TensorFlow, GPU constraints, target domain: MSA vs classical vs dialect), I can outline a minimal fine‑tuning script and data‑prep pipeline tailored to a 1k‑token starter experiment.

<span style="display:none">[^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35]</span>

<div align="center">⁂</div>

[^1_1]: https://pypi.org/project/arabic-diacritizer/

[^1_2]: https://github.com/h9-tec/Awesome_Arabic_NLP

[^1_3]: https://github.com/qcri/advancing-arabic-diacritization

[^1_4]: https://www.emergentmind.com/topics/shakkala-neural-system

[^1_5]: https://huggingface.co/yuujiElfahkrany/turath1.0

[^1_6]: https://huggingface.co/NightPrince/Nemo-Arabic-STT-Diacritized

[^1_7]: https://libraries.io/pypi/langchain-arabic

[^1_8]: https://github.com/mzx93M/KSAA2026-Fine-Tashkeel

[^1_9]: https://huggingface.co/TigreGotico/shakkala-diacritizer

[^1_10]: https://huggingface.co/TigreGotico/catt-diacritizer

[^1_11]: https://www.eurekalert.org/news-releases/1115332

[^1_12]: https://huggingface.co/datasets/Rdyh/everyayah/blob/main/README.md

[^1_13]: https://huggingface.co/datasets/FaresElmenshawi/quran-recitations-asr

[^1_14]: https://huggingface.co/azeddinShr/Spark-TTS-Arabic-Complete

[^1_15]: https://aclanthology.org/2025.emnlp-main.846/

[^1_16]: https://pypi.org/project/voicetut-tts/

[^1_17]: https://huggingface.co/silma-ai/silma-tts

[^1_18]: https://huggingface.co/rufaelfekadu/diac-transformer-text-asr-tashkeela-clartts-kssa

[^1_19]: https://huggingface.co/NightPrince/Fasih-TTS-V1

[^1_20]: https://huggingface.co/blog/Omartificial-Intelligence-Space/cohere-ar-tashkeel-tips

[^1_21]: https://huggingface.co/NightPrince/stt-arabic-whisper-finetuned-diactires/blob/main/README.md

[^1_22]: https://github.com/YazanAi-Dev3/RAG-Augmented-Prosody-Analyzer

[^1_23]: https://www.scribd.com/document/829237650/IEEE-Conference-Template1

[^1_24]: https://aclanthology.org/2026.findings-acl.1293/

[^1_25]: https://aclanthology.org/2025.arabicnlp-sharedtasks.17.pdf

[^1_26]: https://arxiv.org/abs/2605.25928v1

[^1_27]: https://huggingface.co/TigreGotico/rawi-v2

[^1_28]: https://huggingface.co/spaces/nipponjo/tts-arabic-demo/commit/4f3c00445c7ee8e18ee5eeeff21f9b53c24d3370

[^1_29]: https://huggingface.co/collections/TigreGotico/arabic-diacritizers-tashkeel

[^1_30]: https://huggingface.co/CUAIStudents/DeepAr

[^1_31]: https://huggingface.co/IbrahimSalah/Arabic-F5-TTS-v2

[^1_32]: https://huggingface.co/models?other=arabic\&p=32\&sort=trending

[^1_33]: https://huggingface.co/datasets/Rdyh/everyayah

[^1_34]: https://groups.google.com/g/sigarab/c/elC9KqPi0UM

[^1_35]: https://www.opentrain.ai/papers/more-data-fewer-diacritics-scaling-arabic-tts--arxiv-2603.01622/


---

## Deep dive: Arabic diacritization datasets (where to get them, size, properties, dates, licenses)

Below is a consolidated, practical catalog of the main datasets you can use to build or fine‑tune an Arabic diacritization ("تشكيل") model. For each resource you get: **what it is**, **where to download**, **size**, **domain / language variety**, **diacritic coverage**, **creation / release date**, and **license / usage notes**.[^2_1][^2_2][^2_3][^2_4][^2_5][^2_6][^2_7]

______________________________________________________________________

## 1) QCRI "Advancing Arabic Diacritization" release (2025)

**What:** The most directly relevant package for your task: a large diacritized training corpus plus standardized benchmarks and an evaluation script.[^2_3][^2_1]

- **Where:** GitHub – [https://github.com/qcri/advancing-arabic-diacritization](https://github.com/qcri/advancing-arabic-diacritization)[^2_1]
- **Contents:**
    - **Wikipedia Diacritized Corpus** (~5M words)
    - **WikiNews Benchmarks** (2014 multi‑reference + new 2024 benchmark, ~10k words)
    - **Scoring script** (WER/DER with multi‑reference support)[^2_1]
- **Size:** ~5 million words in the training corpus; benchmarks ~10k words each.[^2_1]
- **Domain / variety:** Modern Standard Arabic (MSA), news‑style Wikipedia articles.[^2_3][^2_1]
- **Diacritic coverage:** Fully diacritized text (every word fully voweled) produced by a strong BiLSTM model; intended as training data and evaluation gold.[^2_3][^2_1]
- **Date:** Wikipedia dump from **2024‑02‑20**; dataset and paper released with EMNLP 2025 (Nov 2025).[^2_3][^2_1]
- **License / notes:** Publicly shareable research dataset released with the paper; citation required if used.[^2_1]

**Why it matters:** This is your best "one‑stop" starting point: clean MSA, fully diacritized, with official metrics. You can sample ~1k tokens easily from the Wikipedia corpus for your initial experiments.[^2_1]

______________________________________________________________________

## 2) Tashkeela Corpus (Zerrouki \& Balla, 2017)

**What:** Classic, widely used diacritized Arabic corpus; foundation for many prior diacritization papers.[^2_2][^2_8]

- **Where:** Originally hosted on GitHub / research pages; often mirrored in diacritization repos (e.g., Shakkala evaluations). Search "Tashkeela corpus Zerrouki" or use via repos that bundle it for benchmarking.[^2_2]
- **Size:** ~75M tokens (words) after cleaning; commonly cited as ~2.3M words in cleaned subsets used for benchmarks.[^2_8][^2_2]
- **Domain / variety:** Mixed: Modern Standard Arabic + Classical Arabic (news, literature, religious texts).[^2_8]
- **Diacritic coverage:** Fully diacritized text; includes morphological and syntactic context annotations in some releases.[^2_8]
- **Date:** First major release around **2017** (Zerrouki \& Balla).[^2_8]
- **License:** Typically released for research; many downstream uses treat it as research‑only. Check the specific mirror you download from.[^2_8]

**Why it matters:** Still a gold standard for reporting DER/WER. Good for classical + MSA mix; excellent if you want your model to handle older texts.[^2_2][^2_8]

______________________________________________________________________

## 3) Shamela‑based classical corpora

Several large datasets derive from **al‑Maktaba al‑Shamela** (المكتبة الشاملة), a massive digital library of classical Islamic texts. These are ideal if your target domain is classical Arabic (tafsir, fiqh, hadith, etc.).[^2_4][^2_5]

### 3.1) AuthenticIlm/Shamela4_Full_DB

- **Where:** Hugging Face – [https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB](https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB)[^2_4]
- **Size:** ~19.8 GB; ~8,589 books; ~7.6M pages; ~7.6 million pages of Arabic text.[^2_4]
- **Domain / variety:** Classical Islamic sciences: Quran, tafsir, hadith, fiqh, aqeedah, Arabic language, history, etc.[^2_4]
- **Diacritic coverage:** Many books are fully or partially diacritized (Quran, hadith matn, classical prose); quality varies by book/edition.[^2_4]
- **Date:** Extracted from Shamela v4 on **2026‑04‑26**; dataset uploaded May 2026.[^2_4]
- **License / notes:** Derived from Shamela (public‑domain texts), but compilation/digitization rights belong to respective owners; intended for **research and personal use**, with respect for critical editions' IP.[^2_4]


### 3.2) Kandil7/Athar‑Datasets

- **Where:** Hugging Face – [https://huggingface.co/datasets/Kandil7/Athar-Datasets](https://huggingface.co/datasets/Kandil7/Athar-Datasets)[^2_5]
- **Size:** ~40 GB; **18.7M passages** across 10 collections (hadith, fiqh, tafsir, aqeedah, seerah, Arabic language, etc.).[^2_5]
- **Domain / variety:** Classical Arabic across all major Islamic disciplines; time span ~0–1400 AH.[^2_5]
- **Diacritic coverage:** Mixed; many passages include full diacritics (especially Quran/hadith), others partial.[^2_5]
- **Date:** Dataset page updated **April 2026**.[^2_5]
- **License:** **MIT** for the dataset structure; individual texts may have their own copyright (check per book).[^2_5]

**Why they matter:** If you care about classical/religious Arabic, these are by far the largest, richest sources. You can filter by category (e.g., `aqeedah_passages`, `hadith_passages`) and sample small subsets for early training.[^2_5][^2_4]

______________________________________________________________________

## 4) ClArTTS – Classical Arabic TTS corpus (MBZUAI)

**What:** High‑quality, fully diacritized Classical Arabic speech + text; great for TTS but also useful as clean diacritized text.[^2_6][^2_9]

- **Where:** Hugging Face – [https://huggingface.co/datasets/MBZUAI/ClArTTS](https://huggingface.co/datasets/MBZUAI/ClArTTS)[^2_6]
- **Size:** ~3.2 GB; **9,500 utterances** (~12 hours of speech).[^2_9][^2_6]
- **Domain / variety:** Classical Arabic (MSA‑like), single male speaker, audiobook‑style.[^2_9][^2_6]
- **Diacritic coverage:** Text field is **fully diacritized** (tashkeel required for TTS).[^2_6][^2_9]
- **Date:** Corpus published **2023** (Interspeech); HF dataset updated recently (2025–2026).[^2_6]
- **License:** Open‑source research corpus; cite the Interspeech 2023 paper.[^2_6]

**Why it matters:** Small but extremely clean diacritized Classical Arabic. Perfect for a 1k–10k token pilot, or for domain adaptation to classical style.[^2_6]

______________________________________________________________________

## 5) BAREC‑10M – Balanced Arabic Readability Corpus

**What:** Large, balanced MSA corpus with rich linguistic annotations (morphology, syntax, readability). Not primarily a diacritization corpus, but useful as additional text and for readability‑aware modeling.[^2_7]

- **Where:** Hugging Face – [https://huggingface.co/datasets/CAMeL-Lab/BAREC-10M](https://huggingface.co/datasets/CAMeL-Lab/BAREC-10M)[^2_7]
- **Size:** ~1.16 GB; **10M words** across multiple domains.[^2_7]
- **Domain / variety:** MSA; domains: Arts \& Humanities, Social Sciences, STEM; categories include educational, literature, media, academic, encyclopedic, religion \& philosophy.[^2_7]
- **Diacritic coverage:** Primarily **undiacritized** raw text, but with morphological and syntactic annotations; you'd need to diacritize it (or use it as extra unlabelled data).[^2_7]
- **Date:** LREC 2026 paper; dataset released in **2026**.[^2_7]
- **License:** Research use; cite the LREC 2026 paper.[^2_7]

**Why it matters:** Good for scaling up text volume and domain diversity once you have a diacritization pipeline; not a direct diacritized training set, but valuable context.[^2_7]

______________________________________________________________________

## 6) Quranic / Islamic speech datasets (optional, domain‑specific)

These are more specialized but useful if your model must excel on Quranic/classical recitation text.

- **everyayah / quran‑recitations‑asr** (Hugging Face): Ayah‑level Quranic recitations with fully diacritized transcripts; hundreds of thousands of ayahs, many hours of audio.[^2_10][^2_11]
- **Size:** e.g., quran‑recitations‑asr: ~581k ayahs, ~2,882 hours of audio.[^2_11]
- **Domain:** Quranic Arabic (fully diacritized by definition).[^2_10][^2_11]
- **Use:** Primarily ASR/TTS, but the text side is a clean diacritized corpus for Quranic domain adaptation.[^2_11][^2_10]

______________________________________________________________________

## 7) Large‑scale weakly diacritized TTS data (for scaling later)

**"More Data, Fewer Diacritics" (arXiv 2026)** describes a pipeline that produced ~4,000 hours of Arabic TTS data with automatic diacritization.[^2_12]

- **Status:** The paper describes the dataset; public release is planned but not fully detailed yet.[^2_12]
- **Relevance:** Shows that large‑scale, automatically diacritized data can train strong models; you can replicate a similar pipeline later using your own diacritizer + ASR.[^2_12]

______________________________________________________________________

## Practical recommendations for your 1,000‑token starter

To start small and iterate quickly:

1. **Primary seed corpus:**
    - Use the **QCRI Wikipedia Diacritized Corpus** (5M words) and sample ~1,000–2,000 tokens from diverse articles.[^2_1]
2. **Classical flavor (optional):**
    - Add a few hundred tokens from **ClArTTS** text (fully diacritized classical sentences) or a small slice of **Shamela4** (e.g., aqeedah or tafsir passages).[^2_6][^2_4]
3. **Evaluation:**
    - Use the **WikiNews‑2024** benchmark from QCRI (multi‑reference, ~10k words) to compute WER/DER once your model is ready.[^2_1]

All of these are downloadable today via GitHub or Hugging Face, with clear sizes and licenses, so you can begin with a tiny subset and scale systematically.[^2_4][^2_6][^2_1]

If you tell me your target domain (MSA news vs. classical Islamic texts vs. mixed), I can suggest an exact sampling plan (how many tokens from which dataset, and in what proportions) for your first 1k‑token training run.

<span style="display:none">[^2_13][^2_14][^2_15][^2_16][^2_17][^2_18][^2_19][^2_20][^2_21][^2_22][^2_23][^2_24][^2_25][^2_26][^2_27][^2_28][^2_29][^2_30][^2_31][^2_32][^2_33]</span>

<div align="center">⁂</div>

[^2_1]: https://github.com/qcri/advancing-arabic-diacritization

[^2_2]: https://www.emergentmind.com/topics/shakkala-neural-system

[^2_3]: https://aclanthology.org/2025.emnlp-main.846.pdf

[^2_4]: https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB

[^2_5]: https://huggingface.co/datasets/Kandil7/Athar-Datasets/blob/main/README.md

[^2_6]: https://huggingface.co/datasets/MBZUAI/ClArTTS

[^2_7]: https://huggingface.co/datasets/CAMeL-Lab/BAREC-10M

[^2_8]: https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.409

[^2_9]: https://huggingface.co/azeddinShr/Spark-TTS-Arabic-Complete/commit/859dd4bb6da9bd13a1e57f43097a049ea1839b7b

[^2_10]: https://huggingface.co/datasets/Rdyh/everyayah/blob/main/README.md

[^2_11]: https://huggingface.co/datasets/FaresElmenshawi/quran-recitations-asr

[^2_12]: https://arxiv.org/abs/2603.01622

[^2_13]: https://aclanthology.org/2026.osact-1.14/

[^2_14]: https://zenodo.org/records/17942113

[^2_15]: https://huggingface.co/datasets/Kandil7/Athar-Shamela4

[^2_16]: https://huggingface.co/datasets/Kandil7/shamela-database

[^2_17]: https://huggingface.co/datasets/bbasavar/wiki-corpus-v1

[^2_18]: https://huggingface.co/datasets/mhaamh19/shamela_books_text_full

[^2_19]: https://huggingface.co/Kandil7/Baligh-1.5B

[^2_20]: https://huggingface.co/datasets/BlueHybrid/qwen35-4b-arabic-blind-spots

[^2_21]: https://metatext.io/datasets/ieasybooks-org/shamela-waqfeya-library

[^2_22]: https://dumps.wikimedia.org/wikidatawiki/entities/20251229/

[^2_23]: https://huggingface.co/datasets/Kandil7/Athar-Datasets

[^2_24]: https://aclanthology.org/2026.eacl-long.102.pdf

[^2_25]: https://aclanthology.org/2026.abjadnlp-1.43/

[^2_26]: https://aclanthology.org/2026.findings-acl.575.pdf

[^2_27]: https://arxiv.org/abs/2607.18946v2

[^2_28]: https://kth.diva-portal.org/smash/record.jsf?dswid=3396\&pid=diva2:2013492

[^2_29]: https://huggingface.co/azeddinShr/Spark-TTS-Arabic

[^2_30]: https://huggingface.co/rufaelfekadu/diactc-wav2vec2-clartts

[^2_31]: https://www.scribd.com/document/916074339/The-International-Corpus-of-Arabic-Compi

[^2_32]: https://huggingface.co/azeddinShr/Spark-TTS-Arabic-Complete

[^2_33]: https://github.com/h9-tec/Awesome_Arabic_NLP


---

## Deep dive: Arabic diacritization models (architecture, training, performance, licensing)

Below is a detailed, side‑by‑side breakdown of the main open‑source models you can use or adapt for Arabic diacritization. For each model you get: **architecture**, **input/output**, **label set**, **training data \& objective**, **size \& speed**, **reported accuracy**, **license**, and **where to get it**.[^3_1][^3_2][^3_3][^3_4][^3_5][^3_6]

______________________________________________________________________

## 1) CATT (Character‑based Arabic Tashkeel Transformer) – Abjad AI

**Best for:** State‑of‑the‑art accuracy on MSA/classical; strong base for fine‑tuning and multimodal extensions.[^3_4][^3_1]

### Architecture

- **Two variants:**
    - **CATT‑EO (Encoder‑Only):** Transformer encoder + linear classification head; **non‑autoregressive**, token‑level tagging.[^3_1]
    - **CATT‑ED (Encoder‑Decoder):** Full seq2seq Transformer; **autoregressive** generation of diacritized text.[^3_1]
- **Tokenization:** Character‑level (Buckwalter encoding) with a dedicated tashkeel tokenizer; 18 diacritic tags.[^3_1]
- **Output:** Per‑token logits over 18 diacritic classes; applied by interleaving diacritics onto characters.[^3_1]


### Input / Output

- **Input:** Raw Arabic text (diacritics optionally present but typically stripped); encoded as Buckwalter character IDs.[^3_1]
- **Output:** Fully diacritized text (or tag sequence) at character level.[^3_1]


### Training data \& objective

- **Pretraining:** Large Arabic corpora (details in Abjad AI papers); trained as a general Arabic language model, then specialized for diacritization.[^3_4]
- **Fine‑tuning:** Diacritization treated as token classification (EO) or seq2seq generation (ED) with cross‑entropy over diacritic labels.[^3_4][^3_1]
- **Multimodal extension:** CATT‑Whisper fuses CATT text encoder with Whisper speech encoder for dialectal diacritic restoration; shows CATT’s encoder is robust and reusable.[^3_4]


### Size, speed, deployment

- **Model size:** ~tens of millions of parameters (encoder‑only variant); ONNX fp32 ~78 MB, int8 ~21.6 MB for CATT‑EO.[^3_1]
- **Speed:** Encoder‑only variant is fast and suitable for CPU/GPU; encoder‑decoder is slower due to autoregressive decoding.[^3_1]
- **Deployment:** Available as ONNX (fp32 \& int8); easy to embed in servers or mobile apps.[^3_1]


### Reported performance

- **DER:** ~4.27% on a broad test set (`TigreGotico/arabic_diacritized_text`); much lower on CATT’s own narrow benchmark (distribution‑dependent).[^3_1]
- **Dialectal task (CATT‑Whisper):** WER 0.55, CER 0.13 on NADI 2025 test set (dialectal + code‑switching).[^3_4]


### License \& availability

- **License:** **Apache‑2.0** (original CATT by Abjad AI).[^3_1]
- **Where:** Hugging Face (`abjadai/CATT` family); ONNX exports at `TigreGotico/catt-diacritizer`.[^3_1]

______________________________________________________________________

## 2) Shakkala (BiLSTM character‑level tagger)

**Best for:** Simple, small, fast baseline; easy to retrain or distill; good for teaching and quick experiments.[^3_2][^3_7]

### Architecture

- **Type:** Character‑level sequence labeling with BiLSTMs.[^3_2]
- **Stack:** Embedding(288) → 3× BiLSTM (288/144/96 per direction) + BatchNorm → Dense(28).[^3_2]
- **Parameters:** ~2.5M.[^3_2]
- **Output:** 28‑class logits per character (diacritic tags).[^3_2]


### Input / Output

- **Input:** Fixed‑length character ID sequence, **length 315**, post‑padded with 0.[^3_2]
- **Output:** `[batch, 315, 28]` logits; decode via `argmax` then interleave diacritics onto characters.[^3_2]


### Training data \& objective

- **Data:** Trained on diacritized Arabic corpora (Tashkeela‑style); version 3 is the most common public release.[^3_7][^3_2]
- **Objective:** Cross‑entropy over per‑character diacritic classes.[^3_7][^3_2]


### Size, speed, deployment

- **Model size:** ONNX fp32 ~10.2 MB; int8 ~10.1 MB.[^3_2]
- **Constraints:** Fixed 315‑char input; longer text must be chunked.[^3_2]
- **Deployment:** Very light; runs easily on CPU; ONNX export available.[^3_2]


### Reported performance

- **DER:** ~5.65% on the broad `TigreGotico/arabic_diacritized_text` test split.[^3_2]
- **Earlier benchmarks:** ~2.88% DER on cleaned Tashkeela‑style benchmarks (depends on preprocessing and test set).[^3_7]


### License \& availability

- **License:** **MIT** (original by Ahmad Barqawi).[^3_2]
- **Where:** Original Keras code on GitHub; ONNX exports at `TigreGotico/shakkala-diacritizer`.[^3_2]

______________________________________________________________________

## 3) Turath1.0 (lightweight Transformer for classical Arabic)

**Best for:** Classical Islamic texts (Shamela domain); very fast CPU inference; small model you can retrain quickly.[^3_3]

### Architecture

- **Type:** Transformer encoder (6 layers), character‑level tagging.[^3_3]
- **Config:** Embed dim 256, 4 attention heads, FFN dim 1024, ~8M params.[^3_3]
- **Vocab:** 130 characters.[^3_3]
- **Output:** 15 diacritic classes (no diacritic + basic vowels + tanwin + shadda combos).[^3_3]


### Input / Output

- **Input:** Up to **256 characters** per segment; longer texts must be chunked.[^3_3]
- **Output:** Per‑character diacritic label; interleaved to produce fully voweled text.[^3_3]


### Training data \& objective

- **Data:** Subset of **Shamela classical Islamic corpus** (tafsir, fiqh, hadith, etc.).[^3_3]
- **Objective:** Cross‑entropy over 15 diacritic labels.[^3_3]


### Size, speed, deployment

- **Model size:** ~8M params; PyTorch checkpoint + vocab + inference script on HF.[^3_3]
- **Latency:** ~5 ms per sentence on CPU (single sentence, no batching).[^3_3]
- **Deployment:** Single‑file inference script; easy to integrate.[^3_3]


### Reported performance (classical benchmark)

On a 1,000‑sentence Shamela test split:[^3_3]


| Model | DER | WER | CPU latency |
| :-- | :-- | :-- | :-- |
| flan‑t5‑small | 3.1% | 29.2% | 90 ms |
| catt‑eo | 5.2% | 30.1% | 31 ms |
| **turath1.0** | 7.1% | 29.7% | **5 ms** |
| mishkal | 20.6% | 85.3% | 36 ms |

Turath is slightly behind flan‑t5 on DER but **17× faster**, with nearly identical WER, making it ideal for high‑throughput classical text pipelines.[^3_3]

### License \& availability

- **License:** **MIT**.[^3_3]
- **Where:** Hugging Face – `yuujiElfahkrany/turath1.0` (model + inference.py).[^3_3]

______________________________________________________________________

## 4) Fine‑Tashkeel (ByT5 seq2seq for speech dictation diacritization)

**Best for:** Seq2seq generation from undiacritized transcripts (especially ASR output); strong performance on dialectal data when fine‑tuned.[^3_5]

### Architecture

- **Base:** `google/byt5-small` (byte‑level T5).[^3_6][^3_5]
- **Type:** Byte‑level **seq2seq** Transformer; maps undiacritized bytes → fully diacritized bytes.[^3_5][^3_6]
- **Input format:** Undiacritized Arabic text (often from ASR), optionally with language/dialect cues.[^3_5]
- **Output:** Fully diacritized text as a generated sequence.[^3_5]


### Training data \& objective

- **Task:** KSAA‑2026 Task 2 – automatic diacritization of Arabic speech dictation.[^3_5]
- **Data:** Speech‑derived transcripts (dialectal + MSA), paired with fully diacritized references.[^3_5]
- **Objective:** Sequence‑to‑sequence cross‑entropy over bytes/characters.[^3_5]


### Reported performance

- **Best submission (zero‑shot, no task‑specific training):** DER 10.56%, WER 34.47%, SER 79.88% (5th of 7 teams).[^3_5]
- **Dialectal variation:** Egyptian 3.70% DER vs Algerian 13.73% DER, showing strong sensitivity to dialect.[^3_5]
- **Key bottlenecks:** Case endings and vowel ambiguity.[^3_5]


### License \& availability

- **Base model license:** ByT5 is **Apache‑2.0**; Fine‑Tashkeel code/eval scripts are public.[^3_6][^3_5]
- **Where:** Paper + code linked from OSACT7/KSAA‑2026 proceedings; ByT5 on Hugging Face (`google/byt5-small`).[^3_6][^3_5]

______________________________________________________________________

## 5) DiacNet‑1.0 (multilingual byte‑level diacritic restoration, includes Arabic‑like scripts)

**Best for:** Understanding how to train a **self‑supervised, multilingual** diacritic restoration model; architectural blueprint more than an Arabic‑specific solution.[^3_6]

### Architecture

- **Base:** `google/byt5-small` (byte‑level seq2seq T5).[^3_6]

```
- **Type:** Single joint model for 10 languages; uses language‑tag prefixes (e.g. `<yor>`, `<vie>`).[^3_6]
```

- **Relevance to Arabic:** Shows that byte‑level T5 can restore complex diacritics (tones, accents) via self‑supervised “strip‑and‑restore” training; same pattern applies to Arabic tashkeel.[^3_6]


### Training data \& objective

- **Data:** `olaverse/qg-passages-multi` (machine‑generated passages in 10 languages), split into sentences.[^3_6]
- **Objective:** Self‑supervised: take fully diacritized text, deterministically strip diacritics to create input; train model to reconstruct original.[^3_6]
- **Arabic analog:** You can do the same with QCRI Wikipedia corpus, Tashkeela, or Shamela: strip tashkeel → train ByT5/CATT to restore.[^3_8][^3_9][^3_7]


### Performance

- **Median CER:** ~0.02 across most languages; Yoruba higher due to tonal ambiguity.[^3_6]
- **Lesson for Arabic:** Most errors come from genuine linguistic ambiguity (case endings, homographs), not model capacity.[^3_5][^3_6]


### License \& availability

- **License:** **Apache‑2.0** (base + data + model).[^3_6]
- **Where:** Hugging Face – `olaverse/diacnet-1.0`.[^3_6]

______________________________________________________________________

## 6) Arabic‑diacritizer (PyPI, BiLSTM + attention)

**Best for:** Ready‑to‑use Python package; decent accuracy with sentence caching; good reference implementation.[^3_10]

### Architecture

- **Type:** BiLSTM + Bahdanau attention; character‑ or subword‑level tagging with a **sentence cache** for speed.[^3_10]
- **Claimed performance:** ~6.6% DER on their test set; reported to beat some large LLM baselines.[^3_10]


### License \& availability

- **License:** MIT/Apache‑style (check repo).[^3_10]
- **Where:** PyPI (`arabic-diacritizer`) + GitHub.[^3_10]

______________________________________________________________________

## Comparative summary (practical view)

| Model | Architecture | Domain strength | Size | Speed | DER (typical) | License | Good use case |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **CATT‑EO** | Transformer encoder (char‑level tagger) | MSA + classical | ~tens of M params (ONNX 78 MB fp32) | Fast (non‑autoregressive) | ~4–5% on broad test; lower on narrow benchmarks | Apache‑2.0 | Best all‑round base to fine‑tune; production SOTA |
| **CATT‑ED** | Seq2seq Transformer | MSA + classical | Larger (enc+dec) | Slower (autoregressive) | Similar or slightly better on long‑range | Apache‑2.0 | When you want generative seq2seq behavior |
| **Shakkala** | 3× BiLSTM (char‑level) | MSA (Tashkeela‑style) | ~2.5M params (10 MB ONNX) | Very fast, fixed 315 chars | ~2.9–5.7% depending on benchmark | MIT | Lightweight baseline; teaching; quick experiments |
| **Turath1.0** | 6‑layer Transformer encoder | Classical (Shamela) | ~8M params | Extremely fast (~5 ms/sentence CPU) | 7.1% on Shamela test | MIT | High‑throughput classical text pipelines |
| **Fine‑Tashkeel (ByT5)** | Byte‑level seq2seq | Dialectal + MSA (ASR transcripts) | ByT5‑small (~300M params) | Moderate (seq2seq) | 10.56% DER (zero‑shot KSAA) | Apache‑2.0 (base) | Diacritizing ASR output; dialectal focus |
| **arabic‑diacritizer** | BiLSTM + attention | MSA | Small–medium | Fast with cache | ~6.6% DER (reported) | MIT/Apache‑style | Easy pip install; decent out‑of‑the‑box |


______________________________________________________________________

## How this maps to your plan (1k tokens → scaling)

- **Start small (1k tokens):**
    - Use **Shakkala** or **Turath1.0** architecture as a sandbox: implement char‑level tagging, train on 1k tokens from QCRI Wikipedia or Tashkeela, verify DER/WER computation.[^3_8][^3_2][^3_3]
- **Move to SOTA base:**
    - Fine‑tune **CATT‑EO** on your growing corpus (10k → 1M+ tokens) once your pipeline is stable.[^3_1]
- **Optional seq2seq route:**
    - If you prefer generative modeling or plan to integrate ASR, replicate **Fine‑Tashkeel**’s approach with `google/byt5-small` and your own diacritized data.[^3_5][^3_6]

If you specify your target domain (MSA news, classical Islamic texts, dialectal ASR transcripts, or mixed), I can recommend a concrete model + dataset combo and sketch a minimal training script tailored to your hardware.

<span style="display:none">[^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_17][^3_18][^3_19]</span>

<div align="center">⁂</div>

[^3_1]: https://huggingface.co/TigreGotico/catt-diacritizer

[^3_2]: https://huggingface.co/TigreGotico/shakkala-diacritizer

[^3_3]: https://huggingface.co/yuujiElfahkrany/turath1.0

[^3_4]: https://arxiv.org/abs/2510.24247v1

[^3_5]: https://lrec.elra.info/lrec2026-ws-osact-31

[^3_6]: https://huggingface.co/olaverse/diacnet-1.0

[^3_7]: https://www.emergentmind.com/topics/shakkala-neural-system

[^3_8]: https://github.com/qcri/advancing-arabic-diacritization

[^3_9]: https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB

[^3_10]: https://pypi.org/project/arabic-diacritizer/

[^3_11]: https://aclanthology.org/2025.arabicnlp-sharedtasks.104/

[^3_12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12846286/table/sensors-26-00534-t003/

[^3_13]: https://huggingface.co/TigreGotico/catt-diacritizer/commit/8ad763613cce96511c099e9943820be7cc80fbc2

[^3_14]: https://huggingface.co/toksuite/supertoken_models-llama_google-byt5-small

[^3_15]: https://www.themoonlight.io/ko/review/abjad-ai-at-nadi-2025-catt-whisper-multimodal-diacritic-restoration-using-text-and-speech-representations

[^3_16]: https://aiwiki.ai/wiki/t5

[^3_17]: https://arxiv.org/abs/2605.25928v1

[^3_18]: https://github.com/mzx93M/KSAA2026-Fine-Tashkeel

[^3_19]: https://huggingface.co/spaces/nipponjo/tts-arabic-demo/commit/4f3c00445c7ee8e18ee5eeeff21f9b53c24d3370

