---
license: other
license_name: research-only
license_link: LICENSE
language:
- ar
- en
pipeline_tag: text-generation
tags:
- arabic
- diacritization
- tashkeel
- nano
- transformer
model-index:
- name: tashkeel-nano-4p5m
  results:
  - task:
      type: text-generation
      name: Arabic Diacritization
    dataset:
      name: SadeedDiac-25
      type: Misraj/SadeedDiac-25
    metrics:
    - type: der
      value: 9.91
      name: DER (CE, %), benchmark evaluator
---

# Tashkeel Nano 4.5M - Arabic Diacritization

**4,488,160 parameters. DER (CE) 9.91 and WER (CE) 31.03 on the published
**SadeedDiac-25** benchmark, scored with that benchmark's own evaluator -
**ahead of every 7B-13B system on its leaderboard.**

> **Superseded.** The smaller successor
> [`ahmadsy/tashkeel-nano-1p1m`](https://huggingface.co/ahmadsy/tashkeel-nano-1p1m)
> (1.08 M) scores **DER (CE) 3.78 / WER (CE) 12.73** on the same benchmark with
> the same evaluator, at 4.14x fewer parameters. Prefer it.

Every number in this card is either produced by a downloadable model or taken
from a published leaderboard. Nothing here is compared against an unpublished
internal checkpoint, because a reader cannot check such a claim.

- **Model:** `ahmadsy/tashkeel-nano-4p5m` (internal id `e63s_s4_12l160`)
- **Task:** predict the Arabic diacritic marks (fatha, damma, kasra, shadda,
  tanwin, sukun) for every Arabic letter in a piece of text.
- **Architecture:** bidirectional char-level Transformer encoder + 15-class
  diacritic head (see [Architecture](#architecture)).
- **Parameters:** 4,488,160 (4.49 M) - **Size:** 17.2 MB (fp32 safetensors)
- **Published checkpoint:** step 19,500 (best `val_loss` 0.1758) of a
  20,000-step run on the full v3 corpus.

---

## Results on SadeedDiac-25 (published benchmark)

[SadeedDiac-25](https://huggingface.co/datasets/Misraj/SadeedDiac-25) is a
1,200-paragraph Arabic diacritization benchmark (50% Modern Standard Arabic,
50% Classical Arabic; 600 of its paragraphs come from the Fadel test set).
It ships a gold reference and a published leaderboard, so it is a comparison
a reader can actually reproduce.

The scores below were produced by running each model on the benchmark's own
`input` field and scoring with the benchmark authors' evaluator
([`misraj-ai/Sadeed`](https://github.com/misraj-ai/Sadeed), `Eval_code.ipynb`),
`gt_missing_diacritic_is_error=False`. `CE` = case endings. Lower is better.

| System | Params | DER (CE) | WER (CE) | DER (w/o CE) | WER (w/o CE) |
|---|---|---|---|---|---|
| Claude-3.7-Sonnet | - | **1.39** | **4.67** | **0.77** | **2.31** |
| Gemini-Flash-2.0 | - | 3.19 | 7.99 | 2.38 | 5.50 |
| tashkeel-nano-1p1m | 1.08 M | 3.78 | 12.73 | 2.25 | 7.41 |
| GPT-4 | - | 3.86 | 5.27 | 3.86 | 10.93 |
| Sadeed (Kuwain 1.5B) | ~1.5 B | 7.29 | 13.74 | 5.26 | 9.92 |
| **tashkeel-nano-4p5m (this)** | **4.49 M** | **9.91** | **31.03** | **5.01** | **15.73** |
| Aya-23-8B | 8 B | 25.63 | 47.49 | 19.76 | 40.25 |
| ALLaM-7B-Instruct | 7 B | 50.36 | 70.34 | 39.41 | 67.09 |
| Yehia-7B | 7 B | 50.88 | 70.23 | 39.77 | 67.15 |
| Jais-13B | 13 B | 78.68 | 99.75 | 60.73 | 99.57 |
| Gemma-2-9B | 9 B | 78.86 | 99.79 | 60.92 | 99.59 |
| SILMA-9B-Instruct-v1.0 | 9 B | 78.66 | 99.74 | 60.71 | 99.56 |

All rows except the two `tashkeel-nano-*` rows are quoted from the benchmark's
published leaderboard. This model ranks **6th of 12** on DER (CE) - ahead of
Aya-23-8B and every 7B-13B system - and **5th** on DER (w/o CE).

### Mark order: why this model's score needs no correction

SadeedDiac-25's gold reference is **not** Unicode-NFC: it writes the shadda
*before* the vowel. The benchmark's evaluator is order-sensitive and silently
drops a shadda that follows a vowel, so a model that emits standard NFC order
is penalised on every geminate for a serialization artifact rather than a
diacritization error.

**This package emits shadda-first**, because its renderer predates the NFC
mark-order fix applied to this line on 2026-10-11. That legacy order happens
to match the benchmark's gold convention, so the numbers above need no
normalisation - but it is **not** canonical Unicode, and it disagrees with
every NFC reference on every geminate. Its successor
`ahmadsy/tashkeel-nano-1p1m` ships the corrected renderer and emits true NFC.

| Model | DER (CE) | WER (CE) | DER (w/o CE) | WER (w/o CE) |
|---|---|---|---|---|
| tashkeel-nano-1p1m, as emitted (NFC) | 8.78 | 31.14 | 6.19 | 22.73 |
| tashkeel-nano-1p1m, order normalised | **3.78** | **12.73** | **2.25** | **7.41** |
| tashkeel-nano-4p5m (this, legacy order) | **9.91** | **31.03** | **5.01** | **15.73** |

The 1.08 M successor is the stronger model on both readings.

---

## Registered tracker (this project's own metric)

Both published models under this project's own registered scorer
(`der_registered`, word-level, mean over fadel_test / sadeed25 / wikinews2014
- all three are public corpora), each at its own best decode:

| Model | Params | fadel_test | sadeed25 | wikinews2014 | 3-gate mean |
|---|---|---|---|---|---|
| **tashkeel-nano-1p1m** (centered) | **1.08 M** | **5.20** | **11.71** | **17.64** | **11.51** |
| **tashkeel-nano-4p5m (this)** (nospace+centered) | **4.49 M** | **14.30** | **22.59** | **31.81** | **22.90** |

This metric is **word-level** (a word counts as correct only if all its marks
match) and is therefore not comparable to the benchmark's diacritic-level
`DER (CE)` column above. It is reported only so the two published models can
be compared to each other under one fixed convention.

### Correction: earlier numbers in this card were inflated

If you have seen **31.31 / 43.37 / 47.19 (mean 40.62)** in an earlier revision
of this card, those numbers were **wrong** and are retracted. They were
produced by a decoder that predates the NFC mark-order fix and were then
compared against references rendered in the old order, so they carried the
full geminate-order penalty. The corrected value for the same checkpoint and
the same decoder arm is **19.70 / 30.30 / 36.31 (mean 28.77)**; with the
whitespace-matched decode used above it is **14.30 / 22.59 / 31.81 (22.90)**.

### The whitespace mismatch (this model only)

This model was trained on a bed with **no space tokens**, so its space
embedding row is never trained and feeding it raw text hands it an untrained
embedding. Stripping whitespace before encoding and reinserting marks at the
original character positions recovers about **-20.4%** relative on the
registered tracker (28.77 to 22.90). Its successor
`ahmadsy/tashkeel-nano-1p1m` was trained on a bed with real space tokens, so
raw text is the correct input for it and no workaround is needed.

---

## Held-out witness: abdou_test (self-report)

The `abdou_tashkeel` **test** split (15,091 sentences / 41,378 lines / 1.54 M
words) was never trained on by any model in this line. It is a public dataset
but it has no published leaderboard, so this is a **self-report**, not a
comparison:

| Metric | Value |
|---|---|
| DER | 39.30 |
| DER (no case) | 29.30 |
| WER | 39.65 |
| text preservation | 0.9846 |

Per-gate detail for the published checkpoint (step 19,500), registered
scorer, this project's `single` decode arm:

| gate | DER | WER | DER_nocase | text_preservation |
|---|---|---|---|---|
| fadel_test | 0.1970 | 0.1970 | 0.1371 | 1.0000 |
| sadeed25 (overlap-flagged) | 0.3029 | 0.3029 | 0.2191 | 0.9999 |
| wikinews2014 | 0.3631 | 0.3631 | 0.2575 | 1.0000 |
| wikinews2024 (in-domain, contaminated) | 0.5118 | 0.5118 | 0.4180 | 1.0000 |

The best decode arm is `nospace_centered`, tabulated in the registered tracker
above.

---

## WikiNews-2024 multi-reference (in-domain, flagged)

Multi-ref DER on the WikiNews-2024 surface (356 sentences, 10,616 words).
This surface is **in the training set** for this model (contamination stamp),
so it is reported for completeness and is **not** external evidence. Only
published models are shown:

| Model | Params | DER | DER_nocase | WER |
|---|---|---|---|---|
| mishkala (published) | 12.5 M | 1.52 % | 1.34 % | 1.52 % |
| **tashkeel-nano-4p5m (this)** | **4.49 M** | **1.77 %** | **1.45 %** | **1.77 %** |
| zmahood (published) | ~4.5 M | 1.85 % | 1.63 % | 1.85 % |

Note the asymmetry: this surface is **in-domain for this model** but
out-of-domain for mishkala and zmahood, so it is not a clean comparison in
either direction. The honest external evidence is the SadeedDiac-25 table at
the top of this card.

---

## Architecture

Bidirectional char-level Transformer encoder with a 15-class diacritic head
(Llama-style blocks with the causal mask **off**):

| component | value |
|---|---|
| char vocabulary | 171 (deterministic: Arabic block, digits, punctuation, ASCII, 4 specials) |
| embedding | 171 x 160 |
| layers | 12 |
| hidden | 160 |
| attention | 8 heads / 2 KV heads (GQA), head_dim 20, RoPE, bidirectional |
| FFN | SwiGLU, 640 |
| norm | RMSNorm (pre-norm), eps 1e-6 |
| head | Linear(160 to 15), no bias |
| max context | 128 chars |
| total params | 4,488,160 |

The 15 classes are the diacritic labels: bare, fatha, damma, kasra, sukun,
fathatan, dammatan, kasratan, shadda, shadda+fatha, shadda+damma, shadda+kasra,
shadda+fathatan, shadda+dammatan, shadda+kasratan. The head is applied only at
Arabic-base positions; every other character (digits, Latin, punctuation,
tatweel, Quranic marks, invisibles) is a passthrough context token and is
reinserted byte-exact, so non-Arabic text survives verbatim.

---

## Training data (v3 corpus)

The model was trained on the **v3** token pack = the v2b corpus +
**Sadeed_Tashkeela** (classical books), packed into ctx-128 char windows.

| source | role | windows |
|---|---|---|
| fadel train (AliOsm/arabic-text-diacritization) | train | 50,000 |
| abdou_tashkeel train | train | 1,443,972 |
| qcri wiki (QCRI diacritized Wikipedia) | train | 32,818 |
| wikinews2024 (QCRI WikiNews-2024) | train | 356 |
| sadeed_tashkeela (Misraj/Sadeed_Tashkeela) | train | 990,544 |
| abdou_tashkeel valid | val | 29,633 |
| fadel val | val | 2,500 |
| sadeed_tashkeela val | val | 52,134 |
| **total** | | **2,433,430 train / 160,214 val** |

Pack stats: 12,702 windows quarantined (invalid mark combinations), 7,581
duplicates removed. wikinews2024 is in the training set - this is the
contamination that makes the wn2024 gate in-domain.

Note: the space token is **absent** from this bed, which is why the decode
workaround in the registered tracker above is needed.

### Sources & citations

- **fadel** - AliOsm/arabic-text-diacritization (GitHub), Fadel train/val/test split.
- **wikinews / qcri wiki** - QCRI advancing-arabic-diacritization (GitHub):
  WikiNews-2024/2014 multi-ref benchmarks, Wikipedia diacritized corpus.
- **sadeed_tashkeela** - HuggingFace `Misraj/Sadeed_Tashkeela` (Tashkeela-derived).
- **sadeed_25** (gate only) - HuggingFace `Misraj/SadeedDiac-25` benchmark.
- **abdou_tashkeel** - HuggingFace `abdou_tashkeel` (compilation; see license note).

### License & use

**Research use only.** The training mix includes web-scraped franken-datasets:
abdou_tashkeel's card claims MIT for the compilation but ~45% of its content
is Tashkeela (commonly GPL-2 mirrored) plus shamela pages and
GPT-4o-mini-diacritized Wikipedia; fadel derives from Tashkeela;
sadeed_tashkeela is Tashkeela-derived (GPL-2 noted); qcri is paper-bundled
shared data (citation expected). A commercial superset would need a real
license audit. Not for commercial use without one.

---

## Strategy - how this result was achieved

1. **Stage-1 char-LM pretraining** (internal, not published): a 28L x 272
   char-LM (30.1 M) trained on raw Arabic for 12k steps,
   `val_loss` 2.4075 to 1.2806. This is the warm-start source.
2. **Warm transfer, not from-scratch:** the nano model initializes its
   embeddings + encoder stack from that pretrain (shape-filtered partial
   transfer for hidden < 272) and trains the 15-class head fresh.
3. **Smallest-model sweep:** six architectures shrinking depth/width on the
   full v3 corpus with an identical recipe, so the size effect is isolated.
   This model is the 4.5 M arm. Below it the embedding capacity becomes the
   cost driver; this arm was still improving at the initial wall and was
   auto-resumed to 20k steps.
4. **External gate probing every 500 steps** (DER, WER, DER_nocase,
   text_preservation). This matters: the in-domain `val_loss` improved
   monotonically to the wall and would have mis-selected the optimum - the
   external probes picked the real peaks.
5. **Published checkpoint:** best `val_loss` of the extended run (step 19,500).
   A separate best-external-gate checkpoint (step 16,500) is a different
   trade-off and is not the published artifact.

### Training config

| field | value |
|---|---|
| loss | vanilla cross-entropy (label_smoothing 0.0) |
| optimizer | AdamW, lr 2e-4, weight_decay 0.1 |
| batch | 32 (accum 1), ctx 128 |
| steps | 20,000 (10k + 10k auto-resume) |
| precision | fp16 autocast + GradScaler, fp32 master weights |
| seed | 777 |
| selection | best val_loss, step 19,500 |

---

## Usage

### From the HuggingFace hub (recommended)

~~~python
import sys
from huggingface_hub import snapshot_download

# download the repo and put its bundled tashkeel_nano package on the path
sys.path.insert(0, snapshot_download('ahmadsy/tashkeel-nano-4p5m'))

from tashkeel_nano import TashkeelNano
model = TashkeelNano.from_pretrained('ahmadsy/tashkeel-nano-4p5m')
print(model.diacritize('السلام عليكم ورحمة الله'))
~~~

The repo bundles the `tashkeel_nano` Python package (model, tokenizer, labels,
passthrough codec, inference). Adding the snapshot directory to `sys.path`
makes it importable; `from_pretrained` then loads the weights from the same
snapshot.

### CLI

~~~bash
python diacritize.py --text 'السلام عليكم ورحمة الله'
python diacritize.py --file input.txt
echo 'مرحبا بالعالم' | python diacritize.py
~~~

### Notes

- Runs on CPU (4.5 M params) or GPU; `device='cuda'` to force GPU.
- Input is diacritized or bare Arabic; marks are stripped internally and
  re-predicted. Non-Arabic text passes through byte-exact.
- **Raw text is not this model's best input** - see the whitespace note above.
- Output uses the legacy shadda-first mark order, not NFC.

---

## Limitations & honesty

- **The earlier results table in this card was inflated by a renderer
  defect and is retracted** - see the correction note above.
- **The package emits legacy shadda-first mark order, not Unicode NFC.**
  This happens to match the SadeedDiac-25 gold, so the benchmark score is
  unaffected, but it disagrees with every NFC reference. Use
  `ahmadsy/tashkeel-nano-1p1m` for canonical output.
- **Raw-text decoding is not this model's best arm.** It needs whitespace
  stripping to reach its fair score.
- **wikinews2024 is in the training set** (contamination stamp). The wn2024
  gate and the WikiNews-2024 multi-ref table are in-domain for this model and
  are **not** external evidence.
- **sadeed25 is overlap-flagged:** Sadeed_Tashkeela is in the training mix.
- **mishkala (12.5 M) still leads on the WikiNews-2024 multi-ref surface**
  (1.52% vs our 1.77%) - but that surface is in-domain for us and
  out-of-domain for mishkala, so it is not a clean comparison.
- **License:** research-only (see above). Not for commercial use without a
  license audit.
- **Superseded** by `ahmadsy/tashkeel-nano-1p1m` (1.08 M), which is smaller
  and better on every published comparison in this card.

---

## Reproduction

SadeedDiac-25 scores: download the benchmark
(`Misraj/SadeedDiac-25`, `input`/`output` columns), run each model's own
published `diacritize` on the `input` column, then score with the benchmark
authors' evaluator from `misraj-ai/Sadeed` (`Eval_code.ipynb`),
`gt_missing_diacritic_is_error=False`.

The registered-tracker numbers were produced with this project's scorer over
identical splits. Training and gate-probe provenance (not downloadable,
listed for auditability of the numbers above): phase `e63s_s4_12l160`,
config `configs/e63s_s4_12l160.yaml`, gate metrics
`runs/diac/e63s_s4_12l160/gate_eval.csv`, fair-decode re-score
`runs/diac/e100_published_nospace/<corpus>/score_registered.json`. Training
used one Quadro RTX 3000 (6,144 MiB, fp16-only Turing); a bigger GPU would be
faster, not different.
