# Desert Ant Labs — Model Research Report

**Purpose:** complete inventory and teardown of Desert Ant Labs' on-device model
portfolio (architecture, training inputs, goals, measured results), as the
reference for recreating these models — English first, then Arabic as the
primary target language.

**Sources (fetched live):**
- <https://huggingface.co/desert-ant-labs> (org page + all 17 model-card READMEs via raw endpoints)
- <https://desertant.com/> (site, all 12 published model pages)
- <https://desertant.com/llms.txt> and <https://desertant.com/catalog.json> (machine-readable catalog, generated 2026-09-14)

---

## 1. About the lab

Desert Ant Labs is a startup building **on-device intelligence: one small
model per task**, shipped as native SDKs (Swift / Kotlin / JavaScript) with
Core ML, LiteRT (TFLite), ONNX and MLX artifacts. Positioning: "Little brains
in every product" — fast, specialized models that beat generalist cloud models
on performance, cost, and energy. Business model: source-available license,
free below 100k monthly active devices per SDK, unlimited inference per user.

Key traits visible across every card:

- **One model = one narrow job.** Never a generalist; each model has a single
  measurable output (a language code, a span, a probability, a JSON field).
- **Extreme size discipline.** Deployable artifacts range 0.2MB–467MB; most
  are 2–90MB. They repeatedly beat much larger baselines per byte.
- **Honest, benchmark-heavy cards.** Every number names its denominator, is
  re-measured on the exact shipped quantized bytes (torch vs ONNX vs TFLite
  vs Core ML), and failure modes are published as part of the product.
- **Build on strong open bases, don't train from scratch.** Nearly every model
  is a fine-tune / distill / prune of an open pretrained model (Parakeet TDT,
  Whisper-tiny, DistilHuBERT, mmBERT, Granite 4.0 350m, XLM-R lineage,
  MobileNetV4).

---

## 2. Portfolio overview (17 HF repos)

| Model | Task | Modality | Size (deploy) | Languages | Status |
|---|---|---|---:|---|---|
| **Voz** | Speech recognition + word timestamps | audio | 467MB | 25 | Available (Swift only) |
| **Align** | Word-timestamp refinement for Apple SpeechAnalyzer | audio | 0.56–0.7MB | 9 | Available (Swift) |
| **Clear** | Speech enhancement (denoise/dereverb/loudness) | audio | 9MB (Core ML) / 24MB (ONNX) | n/a (voice) | Available (S/K/JS) |
| **Uhm** | Filler-word detection ("uh/um/hmm" spans) | audio | 45MB (CoreML) / 24MB (tflite) | 5 (EN-trained, transfer) | Available (Swift) |
| **Ear** | Spoken language identification | audio | 13MB (CoreML) / 22MB (tflite) | 99 | Available (S/K/JS) |
| **Clips** | Clip/highlight selection from transcripts | text | 284MB | 100 | Available (Swift) |
| **Title** | Title + description generation | text (gen) | ~350MB 6-bit (Granite 350m) | multilingual | Available (Swift, MLX) |
| **Tongue** | Text language identification | text | 2MB | 84 | Available (S/K/JS) |
| **Emo** | Emoji suggestion | text | 5MB | 22 (incl. Arabic) | Available (S/K/JS) |
| **Gist** | Topic tagging (36 topics, multi-label) | text | 74MB (15MB EN-only) | 101 | Available (S/K/JS) |
| **Redact** | PII detection/redaction | text | 11.6MB (4-bit) / 24.5MB (int8) | 27 | Available (S/K/JS) |
| **Schemer** | Schema-constrained structured extraction | text | 218MB int8 / 111MB int4 | 13 | Closed beta |
| **Toxic** | Hate-speech/abuse triage | text | 80–100MB (int4) | 23 EU | Closed beta |
| **Toxic-en** | English-only variant of Toxic | text | 16.5–34MB | en | Available |
| **Moderator** | NSFW image classification | vision | 18MB fp16 / 7MB 6-bit | n/a | Closed beta |
| **Shapes** | Single-stroke shape recognition | vision/strokes | 0.2MB (CoreML) / 1.3MB (tflite) | n/a | Available (S/K/JS) |
| **Who** | Speaker labeling ("who said what") | audio+vision | unpublished | — | Closed beta (no card) |

Beta models named on the site but without HF weights: **Eye** (image/video
shot scoring), **Face** (face matching), plus **Who** (empty HF repo).

---

## 3. Per-model deep dives

### 3.1 Voz — on-device ASR (flagship)

- **Goal:** transcribe 10 minutes of audio in ~2 seconds on an iPhone, with
  word-level timestamps, entirely on the Apple Neural Engine.
- **How built / what they use:** an ANE-optimized conversion of NVIDIA's
  **Parakeet TDT 0.6B v3** (CC BY 4.0). Weight values unchanged; converted to
  Core ML and compressed. Ships as three compiled Core ML stages: `mel`
  (log-mel frontend computed inside Core ML), `encoder` (conformer-style
  acoustic encoder, 15s windows, one frame per 80ms), `decoder` (token-and-
  duration transducer; 16 windows batched into one dispatch). Long audio is
  split at pauses, transcribed independently, windows joined on the longest
  run of agreeing words. 100% Neural Engine residency, no CPU/GPU fallback.
- **Architecture:** log-mel spectrogram → conformer encoder → RNN-T-style
  transducer decoder (emits token + duration per step, which yields word
  timestamps natively).
- **Data (eval):** Open ASR Leaderboard sets (LibriSpeech, GigaSpeech,
  SPGISpeech, Earnings-22, AMI) with their own normalizer; FLEURS for the
  25-language table (10 min/language, concatenated to cross window
  boundaries).
- **Results:**
  - 7.40% avg WER over six English sets vs **7.00%** for Whisper
    large-v3-turbo (1.6GB); Voz is **467MB**.
  - ~290x realtime long-form (611s in 2.1s, M3 Ultra); 30 min in 6s on
    iPhone 17 Pro; 4.7x faster than whisper.cpp large-v3-turbo on a Mac.
  - Word timestamps: 83ms (starts) / 95ms (ends) MAE vs torchaudio MMS_FA.
  - Per-language WER (FLEURS long-form): it 3.31%, pt 6.08%, uk 6.40%, ru
    6.57%, en 7.36%, de 8.12%, es 9.01% … el 39.46% (tail >20% flagged as
    "costs more to correct than to retype").

### 3.2 Align — word-timestamp refiner

- **Goal:** fix the word timings Apple's SpeechTranscriber/SpeechAnalyzer
  returns, without replacing the transcriber. Millisecond-scale refinement.
- **How built:** a tiny two-stage Core ML cascade (coarse 285KB + fine 274KB
  compiled, fp16) run on CPU+ANE, plus log-mel filterbank (40KB) and a
  **gradient-boosted-tree calibrator** (70KB). v1.1.0 corrected Apple's
  log-mel scaling. Input: Apple's recognized words + proposed times; output:
  corrected times, with a structural fallback that keeps Apple's timing when
  a correction looks unsafe.
- **Results:** LibriSpeech test-clean: Apple raw 106.4ms → **20.2ms** MAE
  (95% of words within 50ms); test-other 111.6ms → 24.8ms. Against 258
  hand-corrected boundaries: Align 45.0ms vs WhisperX 53.5ms vs raw Whisper
  100.8ms. p90 error 230.7ms → 33.0ms. 9 languages (en, es, fr, it, pt, de,
  ja, ko, zh); ja/ko/zh improved 33–55%.

### 3.3 Clear — speech enhancement

- **Goal:** "messy recording in, studio sound out" — denoise + dereverb +
  loudness normalize, on-device, no per-minute cloud bill (Adobe
  Podcast/Dolby/Auphonic alternative). Two variants: **clear-studio**
  (silence-true) and **clear-natural** (keeps room tone/breath).
- **How built:** 9.0MB Core ML fp16-compute / 6-bit weight palette model, all
  492 ops confirmed on ANE (`MLComputePlan`); 24MB ONNX for other platforms;
  provisional Core AI builds for iOS/macOS 27. Streaming file path so an
  hour-long file uses the same memory as a 10s clip. Output presets: Spotify,
  Apple Podcasts, YouTube loudness, EBU R128.
- **Results:** 302x realtime on iPhone 16 Pro, 345x on MacBook Pro M5
  (60s clip, enhance→master→re-encode, best of 3); 413x with Core AI on
  iPhone 17 Pro. **No published audio-quality score vs cloud tools** (stated
  as not yet measured). Not a source separator; speech-only.

### 3.4 Uhm — filler-word detection

- **Goal:** frame-precise (20ms) detection of "uh/um/hmm" spans without ASR
  (ASR models drop fillers from transcripts anyway).
- **How built:** **DistilHuBERT fine-tune** (base: `ntu-spml/distilhubert`,
  distilled from facebook/hubert-base-ls960, Apache-2.0). Artifacts: 45MB
  Core ML fp16, 24MB int8-weight-only tflite (99.7–100% argmax match with
  ONNX), 51MB fp16 ONNX, 98MB fp32 reference.
- **Data:** AMI Meeting Corpus (IHM split, CC BY 4.0) + internal video
  recorded by the team. Trained on English; Spanish/French/German/Dutch by
  **acoustic transfer** (unmeasured per language).
- **Architecture:** 16kHz mono input, up to 30s windows; per-frame softmax
  over 6 classes (`0=not_filler, 1=uh, 2=um, 3=hmm, 4=and, 5=other`), one
  prediction per 20ms. Core ML shape (1, 480000) → (1, 1499, 6).
- **Results:** ~296x realtime (iPhone 17 Pro), 279x (iPad Pro M4), 169x
  (iPhone 15 Pro). Limits published: degrades on music/laughter/overlap;
  subtype labels are secondary signals.

### 3.5 Ear — spoken language identification

- **Goal:** name the spoken language from ~30s of audio before transcription
  starts, so the right transcriber is chosen; 99 languages; must handle
  non-speech-heavy recordings.
- **How built:** derived from **openai/whisper-tiny** (MIT): parameters the
  task doesn't use are removed, the language-prediction subgraph is kept,
  compressed for on-device (13MB CoreML / 22MB tflite). The log-mel frontend
  (80 bins, 400-pt FFT, 160 hop, 30s windows) runs on the host (can't be
  fp16) from `ear_meta.json`.
- **Clever trick:** it doesn't listen to the whole file. It ranks candidate
  30s windows by **syllable-rate loudness variation** (speech rises/falls
  3–6x/sec) and reads only the 3 most speech-like windows — 250ms total.
  Window-by-position picks the language only 4% of the time on speech-sparse
  recordings; loudest-windows fails on music intros.
- **Results:** on 162 real recordings, 98.5% of answers above the
  reliability threshold routed correctly; zero confident answers wrong.
  Known limits: Norwegian/Swedish/Danish confused (isReliable forced false),
  speech under loud music ~60% correct.

### 3.6 Clips — clip selection

- **Goal:** from a transcript, rank non-overlapping highlight clips
  (shorts/highlights) with no context-window limit.
- **How built:** two int8 graphs sharing weights (Core ML multifunction
  package 284MB with `select` and `score` functions; two 283MB tflites
  elsewhere) + 4MB Unigram tokenizer. Selector graph 128 tokens wide, scorer
  256, fixed batch of 16 sentences; sentences >64 tokens truncated. The model
  never reads the transcript whole, so length is bounded by time, not tokens.
  Clips carry no titles — that's Title's job.
- **Results:** 25-min transcript (404 sentences) → 12 ranked clips in 9.19s
  on iPhone 17 Pro; 2.78ms per candidate (encoder only). Score is
  uncalibrated between videos — threshold on percentile (0.8 → 5.0 clips /
  17% coverage; 0.3 → 32.1 clips / 74%). Published weak spots: clip edges,
  sensitivity to quantization, Android needs 2.1GB RAM (no Android SDK yet).

### 3.7 Title — titles & descriptions (the only generative model)

- **Goal:** a plain-register factual title (3–8 words) + 1–2 sentence
  description for any passage; no emoji, no clickbait.
- **How built / architecture:** fine-tune of **ibm-granite/granite-4.0-350m**
  (hybrid Mamba/transformer "granitemoehybrid" per HF tags), shipped as MLX
  6-bit quantized safetensors, Apple-silicon only. Fine-tuned against ONE
  fixed instruction prompt with a fixed reply format (`TITLE: … / DESC: …`);
  the chat template is treated as part of the task. They explicitly rejected
  Core ML/ANE export: on short autoregressive decode the ANE is
  bandwidth-bound and lost on every metric.
- **Data:** fine-tuned on transcript clips; holds up on news, product
  descriptions, email.
- **Results:** **no published quality figures** — the card honestly reports
  "internal testing, less settled than that phrase usually implies" with a
  known defect (occasional forbidden stock opening phrase).

### 3.8 Tongue — text language ID

- **Goal:** identify the language from as few as 3 words (search boxes, chat,
  keyboard), in tens of microseconds, at 2MB total, with no tokenizer file.
- **How built:** 59 languages learned by a **hashed character n-gram lexical
  model** (feature hashing → int8 weights, no tokenizer/vocab to ship) + 25
  more decided by **script alone** (Hangul → Korean, Thai → Thai; no model
  involved) = 84 languages. Normalization/hashing/script-routing run in the
  host; the ONNX graph is only the head. Ships int8 (2.01MiB) and int4
  (1.01MiB, −0.4pp on 1–2 word input).
- **Results (3-word input unless noted):** FLORES-200 3-word **0.933** vs
  lingua (293MB) 0.887; FLORES 5-word 0.974; held-out single words 0.759;
  held-out sentences 0.971; Tweets "eld" 0.992. Latency 0.013ms/word in JS.
  Reports ties ("la casa" = it/es) instead of guessing. Failure modes
  published: ms/id inseparable, Cyrillic Mongolian reads as Russian,
  brand names/numbers have no answer.

### 3.9 Emo — emoji suggestion

- **Goal:** emoji suggestions on every keystroke for short intent-oriented
  text (tasks, messages), <2ms, 22 languages.
- **Architecture:** small text classifier over an **812-emoji vocabulary**;
  5MB Core ML / 10.2MB int8 tflite + 750KB Unigram tokenizer. Input: plain
  string; output: probability distribution over emojis, optimized for top-1
  relevance. (Card does not name the base encoder; size suggests a compact
  transformer encoder.)
- **Data/training:** not disclosed. Languages incl. Arabic, Hindi, Thai, CJK.
- **Results:** <2ms per suggestion, 5MB total. Limits: near-ties expected,
  long-form text noisier. 27k downloads — one of their most-used models.

### 3.10 Gist — topic tagging

- **Goal:** multi-label topics (fixed 36-topic taxonomy, mapped to IAB 2.2 +
  Apple Podcasts categories) for posts/titles in 101 languages; scores
  aggregate across a collection (channel-level topics).
- **How built:** explicitly **no transformer at inference**: a two-stream
  classifier — **static int8 multilingual embedding (64MB, vocab-pruned) +
  hashed n-grams → small fp16 head** (6MB CoreML / 13MB tflite) + 4MB
  Unigram tokenizer = 74MB. English-only variant: 15MB, topic-identical on
  English input.
- **Data/eval:** 572 human-labeled real posts (held-out).
- **Results:** recall@3 **91%** (the product metric), recall@1 71% — tied
  with multilingual-e5-small+head (110MB transformer) at 74MB, beaten only
  by cloud Qwen2.5-7B zero-shot (79% recall@1). Beats zero-shot NLI
  (mDeBERTa 73% r@3) and GLiClass (65%). 15-language spot check: 88% top-3,
  CJK/Arabic/Cyrillic ≥ Latin.

### 3.11 Redact — PII redaction

- **Goal:** strip 20 PII label types (names, addresses, emails, phones,
  cards, IBANs, national IDs…) before text leaves the device; 27 languages
  (all EU + no/is); avoid over-redaction as much as under-redaction.
- **How built / architecture:** a **23M-param token classifier** (BERT-
  lineage encoder per tags) + a **deterministic rules layer** for structured
  IDs (IBAN, IMEI, routing numbers). Deploy: 11.6MB 4-bit Core ML / 24.5MB
  int8 tflite. `ORG` detected but not redacted by default.
- **Data (eval):** WikiANN, MultiNERD, a format-valid structured-PII set,
  24 EU languages; plus an 11,528-row adversarial negative set across 27
  languages (sentence-initial capitals, ALL-CAPS, months, UI vocabulary…).
- **Results:** recall **88.8** / precision **99.6** at 11.6MB / 23M params,
  vs GLiNER-PII (570M, 2.3GB): 91.1/90.4; Rampart (18.5M): 61.4/97.2; OpenAI
  privacy filter (1.5B, 3GB): 60.2/93.5. 94.1% of adversarial negative rows
  come back untouched. Honest gap: AWS Comprehend beats it on English names
  (English-only, cloud, per-call).

### 3.12 Schemer — schema-constrained extraction (closed beta)

- **Goal:** given a caller JSON schema (OpenAI/Gemini `responseSchema`
  compatible), return typed JSON: **typed by construction** (labels from
  declared choices, clamped numbers, ISO-8601 datetimes with relative-date
  resolution), **absence detection** (unstated fields → null), **verbatim
  spans with character offsets** (hallucination audit = substring check).
- **How built / architecture:** **pruned mmBERT-base encoder** (22 layers,
  103k vocab kept of 256k) jointly encoding `[schema summary ||| document]`
  at two compiled shapes (256 / 1216 tokens) → one shared cross-attention
  reader → **thin per-type heads**: label entailment, 3-way boolean
  (absent/false/true), BIO span tagging with trained presence gates,
  datetime and number component decoders → deterministic harness (locale
  number parsing, ISO composition, 13-language relative-date lexicons,
  format gates) shipped as data with conformance fixtures. Nested schemas
  via deterministic segment-and-recurse in the harness. 211M params;
  218MB int8 (0.800) / 111MB int4 AWQ (0.793); 8.7ms/encoder pass on ANE.
- **Results (9,021 held-out records, 13 languages, unseen schemas):**
  overall **0.800** — above Qwen 3.5 0.8B (0.651), NuExtract-2.0-2B (0.643),
  GLiNER2-multi (0.585), LFM2-1.2B-Extract (0.489); below Gemma 4 26B A4B
  (0.834, 17.2GB) and Claude Haiku 4.5 API (0.816). Killer stat: **absence
  detection 0.911 vs 0.18–0.43 for every LLM and span extractor benched**.
  Best calendar score of any model at any size (0.946). Wins all 13
  languages vs the strongest sub-1B LLM (0.670–0.745 per language; only
  0.075 spread across languages).

### 3.13 Toxic / Toxic-en — hate-speech triage (closed beta)

- **Goal:** on-device triage of hateful/abusive/threatening text in 23 EU
  languages; three content heads (`HATEFUL`, `ABUSIVE`, `THREAT`) + 10
  protected-group target heads. Explicitly "triage, not verdict".
- **How built / architecture:** multilingual: **trimmed XLM-R-lineage
  encoder, 159.6M params** (Detoxify-class architecture), 95,552-piece
  SentencePiece-Unigram tokenizer. toxic-en: same recipe, English-only
  encoder trimmed to **31.9M params** (~1/5 the size). Artifacts int4
  (multilingual: 80–100MB) / int8 (toxic-en: 16.5–34MB; int4 measured to
  cost too much on the smaller encoder). Multi-head: 2 multi-label output
  layers on one encoder. Supervision named per head: HATEFUL scoped to EU
  Framework Decision 2008/913/JHA; ABUSIVE trained on civil_comments
  insult/identity_attack severity votes; THREAT on threat labels.
- **Data:** civil_comments, germeval2018 (90% train), synthetic
  translate-and-audit multilingual sets; evaluation on real Multilingual
  HateCheck (dev-benchmark qualification stated honestly), plus out-of-domain
  textdetox, offenseval2020, HateXplain.
- **Results:** 7-EU real MHC macro-F1 **0.847** (torch) / 0.839 (ONNX) /
  0.830 (TFLite); beats Shieldstral-1.0-3B (0.786), Qwen3Guard-0.6B (0.652),
  Llama-Guard-3-1B (0.649), Detoxify multilingual (0.487) at phone size.
  toxic-en: **0.855** English MHC at 31.9M params. Out-of-domain:
  textdetox 0.724, offenseval2020 0.658. Notably honest: published
  non-hate FPR 0.26–0.30 (0.12 EN), counter-speech false positives, and
  that prompted Gemma 4 12B (0.958) beats it at ~75x params.

### 3.14 Moderator — NSFW image detection (closed beta)

- **Goal:** score images 0–1 for nudity/sexual activity; pass swimwear/
  lingerie, flag real nudes; per-region heads for granular policies
  (allow-topless).
- **How built / architecture:** **MobileNetV4-Conv-Medium backbone (8.4M
  params, Apache-2.0)** + 2-layer MLP head (1280→512→5), L2-normalized
  features. Input 384×384, ImageNet-normalized; NSFW score = max of 5 region
  heads (`nude`, `sex_act`, `nipples_visible`, `genitals_visible`,
  `buttocks_visible`). 18MB fp16 CoreML / 7MB 6-bit / 35MB fp32 ONNX.
  Multi-crop inference for recall (1 crop: 74%/97%; 8 crops: 88%/94%).
- **Data:** **100% clean provenance** — no scraping: public-domain fine art
  and museum open-access collections, openly licensed imagery, and
  in-house generative photoreal images. Positives drawn only from
  permissive/synthetic sources.
- **Results (464-image held-out, 8-crop, thr 0.5):** recall **87.8%** /
  specificity **93.7%** vs NudeNet 82.6%/87.1%. Limits: photoreal only —
  anime and mosaic-censored content out of scope.

### 3.15 Shapes — single-stroke shape recognition

- **Goal:** snap one hand-drawn stroke to a clean shape in <10ms; replaces
  Apple's private "Snap to Shape" API.
- **How built / architecture:** tiny stroke classifier over an ordered point
  sequence: input `[1,256,3]` features + `[1,256]` mask; classes
  `line/rectangle/triangle/ellipse/star` + `none` reject class (scribbles);
  plus fitted vector geometry and snap gates (axes, 15° rotations,
  near-regular squares/circles). 200KB 4-bit CoreML / 1.3MB fp32 tflite;
  ships inside the app bundle on Apple.
- **Results:** <10ms per stroke. (No public benchmark; internal.)

### 3.16 Who — speaker labeling (no card)

- HF repo exists but empty (106-byte card). Site: "On-device speaker labeling
  that turns a recording into per-person turns with face boxes and
  timestamps, ready to cut or caption." Closed beta; no published details.

---

## 4. The Desert Ant playbook (transferable patterns)

1. **Never train from scratch.** Start from a strong open checkpoint
   (Parakeet TDT 0.6B v3, Whisper-tiny, DistilHuBERT, mmBERT-base, Granite
   4.0-350m, XLM-R, MobileNetV4), then: prune the task-irrelevant subgraph
   (Ear), trim the encoder (Toxic, Schemer), distill, quantize, and measure
   the loss per artifact.
2. **Non-neural tricks around the neural core.** Script-based routing with no
   model (Tongue), syllable-rate window selection (Ear), deterministic rules
   for structured IDs (Redact), harness post-processing as shipped data
   (Schemer), GBT calibrators (Align), percentile thresholds (Clips). These
   carry a large share of the quality per byte.
3. **Architecture follows the deployment target.** Generative = MLX on Apple
   silicon (Title); encoder-only heads for everything latency-critical;
   Core ML multi-function packages to share weights (Clips); compiled
   `.mlmodelc` shipped to avoid per-launch recompilation.
4. **Measure everything on shipped bytes.** Every quantization delta is
   published (e.g., Toxic torch 0.847 → ONNX 0.839 → TFLite 0.830). Numbers
   name their denominator; dev-benchmark contamination is disclosed
   (Toxic/HateCheck, Schemer).
5. **Benchmarks with baselines at matched operating points.** Every
   comparison re-runs the competitor through the same harness (Redact vs
   GLiNER/OpenAI; Schemer vs 15 extractors; Toxic vs 5 guardrails; Gist vs
   6 embedders; Tongue vs lingua/HeLI/Apple).
6. **Publish failure modes as product.** isReliable flags, tie detection,
   per-language WER tables with "don't use this tail" guidance, disabled
   head valves for failed gates.

---

## 5. Arabic adaptation plan (for our recreation project)

**Recommended English-first starters (easy, high-signal, feasible on one
consumer GPU):**

1. **Tongue-analogue (lang-ID)** — best first target: architecture is
   feature-hashing + small classifier, no tokenizer, ~2MB; training data
   generatable from FLORES-200 / OSCAR / Twitter (eld benchmark); eval
   methodology fully copyable.
2. **Emo-analogue (emoji suggestion)** — small text classifier; Desert Ant's
   own 22-language model already covers Arabic, giving us a direct baseline
   to compare against; short-intent text is easy to source/annotate.
3. **Gist-analogue (topic tagging)** — static embedding + head, no
   transformer at inference; the 36-topic IAB-mapped taxonomy can be reused
   as-is; a few hundred human-labeled eval posts is a cheap eval set.

**Then the audio tier (harder, needs GPU time):**

4. **Uhm-analogue (Arabic filler detection)** — DistilHuBERT recipe; Arabic
   fillers (يعني، إيه، أمم) are acoustically distinct; AMI-style meetings +
   Arabic podcast audio as supervision base.
5. **Ear-analogue (spoken Arabic dialect ID)** — Whisper-tiny language-head
   trick specialized to Arabic varieties (MSA vs Egyptian vs Gulf vs Levant
   vs Maghrebi) — a task nobody ships on-device and a natural Arabic-first
   differentiator.
6. **Voz-analogue (Arabic ASR)** — the flagship pattern: fine-tune/compress
   an open Arabic-capable recognizer; Arabic ASR WER is far worse than
   English at comparable sizes, so the value gap is large.

**Then text-heavy tiers:**

7. **Redact-analogue (Arabic PII)** — BERT-size token classifier + Arabic
   deterministic rules (national IDs, IBAN, local phone formats); WikiANN /
   MultiNERD have Arabic coverage; the adversarial negative-set method
   transfers directly.
8. **Title-analogue (Arabic titles)** — Granite-350m-class small LM
   fine-tune; our SFT pipeline (configs/sft_t1.yaml) already targets this
   scale; needs an Arabic transcript-clip dataset.
9. **Schemer-analogue (Arabic structured extraction)** — crown jewel, largest
   lift (multi-head encoder + harness with Arabic relative-date lexicons
   "بكرة الساعة ٣", Arabic-Indic numerals, Hijri/Gregorian duality).

**Data strategy note:** Desert Ant trains on a mix of public datasets
(AMI, FLEURS, WikiANN, MultiNERD, civil_comments, HateCheck-style
constructs), in-house generated content, and synthetic
translate-and-audit sets for lower-resource languages. The
translate-and-audit synthetic pattern is exactly how they covered 15 of 23
languages — the same pattern can bootstrap Arabic (and then dialect)
coverage where labeled Arabic data is thin.

---

## 6. Open items / gaps in the source material

- **who**: no card, empty HF repo — architecture unknown.
- **Emo, Gist, Shapes**: training data and base encoders undisclosed.
- **Clear**: no audio-quality benchmark vs Adobe Podcast/Dolby published yet.
- **Title**: no quality figures at all (self-declared "internal testing").
- **Schemer**: private pre-release; weights staged but SDK not released.
- Site pages **Eye** and **Face** have no weights anywhere yet.

*Report generated from live fetches of huggingface.co/desert-ant-labs and
desertant.com; catalog.json generated 2026-09-14; HF lastModified dates
2026-08-03 … 2026-09-18.*
