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
- soup
model-index:
- name: tashkeel-nano-1p1m
  results:
  - task:
      type: text-generation
      name: Arabic Diacritization
    dataset:
      name: SadeedDiac-25
      type: Misraj/SadeedDiac-25
    metrics:
    - type: der
      value: 3.78
      name: DER (CE, %), benchmark evaluator, mark order normalized
---

# Tashkeel Nano 1.1M - Arabic Diacritization

**1,083,328 parameters. 3.78% DER (CE) and 12.73% WER (CE) on the published
**SadeedDiac-25** benchmark, scored with that benchmark's own evaluator -
**3rd of the 11 systems on its leaderboard, ahead of GPT-4 and ahead of the
**dedicated 1.5B Sadeed model.**

Every number in this card is either produced by a downloadable model
(`ahmadsy/tashkeel-nano-1p1m`, `ahmadsy/tashkeel-nano-4p5m`) or taken from a
published leaderboard. Nothing here is compared against an unpublished
internal checkpoint, because a reader cannot check such a claim.

- **Model:** `ahmadsy/tashkeel-nano-1p1m` (internal id `e98_soupB`, artifact `soupB_uniform`)
- **Task:** predict the Arabic diacritic marks (fatha, damma, kasra, shadda,
  tanwin, sukun) for every Arabic letter in a piece of text.
- **Architecture:** bidirectional char-level Transformer encoder + 15-class
  diacritic head (see [Architecture](#architecture)).
- **Parameters:** 1,083,328 (1.08 M) - **Size:** 4.15 MiB (fp32 safetensors)
- **Published checkpoint:** a **uniform soup of 12 step probes** from a
  60,000-step run (steps 32,500 to 60,000) on the `v3full` corpus.

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
| **tashkeel-nano-1p1m (this)** | **1.08 M** | **3.78** | **12.73** | **2.25** | **7.41** |
| GPT-4 | - | 3.86 | 5.27 | 3.86 | 10.93 |
| Sadeed (Kuwain 1.5B) | ~1.5 B | 7.29 | 13.74 | 5.26 | 9.92 |
| Aya-23-8B | 8 B | 25.63 | 47.49 | 19.76 | 40.25 |
| ALLaM-7B-Instruct | 7 B | 50.36 | 70.34 | 39.41 | 67.09 |
| Yehia-7B | 7 B | 50.88 | 70.23 | 39.77 | 67.15 |
| Jais-13B | 13 B | 78.68 | 99.75 | 60.73 | 99.57 |
| Gemma-2-9B | 9 B | 78.86 | 99.79 | 60.92 | 99.59 |
| SILMA-9B-Instruct-v1.0 | 9 B | 78.66 | 99.74 | 60.71 | 99.56 |

All rows except ours are quoted from the benchmark's published leaderboard.
Rank of this model: **3rd of 11** on DER (CE), **4th** on WER (CE),
**2nd** on DER (w/o CE), **3rd** on WER (w/o CE).

### Read this before quoting the number above

SadeedDiac-25's gold reference is **not** Unicode-NFC: it writes the shadda
*before* the vowel, while this model emits standard **NFC** order (vowel
before shadda). The benchmark's evaluator is order-sensitive and silently
**drops a shadda that follows a vowel**, so scoring our raw output against
that gold marks every geminate wrong - a serialization artifact, not a
diacritization error.

Both readings are reported here, because hiding either one would be
misleading:

| Evaluation of this model | DER (CE) | WER (CE) | DER (w/o CE) | WER (w/o CE) |
|---|---|---|---|---|
| As emitted (NFC), scored literally | 8.78 | 31.14 | 6.19 | 22.73 |
| Mark order normalised to the benchmark's convention | **3.78** | **12.73** | **2.25** | **7.41** |

The table at the top of this section uses the normalised row, which compares
*linguistic content* and is the like-for-like reading against the
leaderboard. The literal row is what you get if you feed this model's raw
output straight into the benchmark's evaluator. The same defect was
root-caused independently inside this project (see
[Unicode NFC](#unicode-nfc-this-model-emits-canonical-text)). Whether each
leaderboard system above was scored with or without the same normalisation is
not stated by the benchmark, so treat the ranking as approximate.

---

## Registered tracker (this project's own metric)

For continuity with earlier releases, the same two public models under this
project's own registered scorer (`der_registered`, word-level, mean over
fadel_test / sadeed25 / wikinews2014 - all three are public corpora):

| Model | Params | fadel_test | sadeed25 | wikinews2014 | 3-gate mean |
|---|---|---|---|---|---|
| **tashkeel-nano-1p1m (this)** | **1.08 M** | **5.20** | **11.71** | **17.64** | **11.51** |
| tashkeel-nano-4p5m (NFC-corrected) | 4.49 M | 14.30 | 22.59 | 31.81 | 22.90 |

This metric is **word-level** (a word counts as correct only if all its marks
match) and is therefore not comparable to the benchmark's diacritic-level
`DER (CE)` column above. It is reported only so the two published models can
be compared to each other under one fixed convention.

### The soup is the lever, not the extra training

The published weights are a **uniform average of 12 checkpoints** from a
single 60,000-step run. Continuing training from step 30,000 to 60,000 did
not produce a monotone gain on its own; the average over that same-basin set
is what lands the result. A soup is not a checkpoint: reproducing this exact
number requires the same 12 members and the same uniform recipe, in ascending
step order.

### Correction: the earlier 4.5M comparison was wrong

If you have seen a figure near **-64%** for this line against
`tashkeel-nano-4p5m`, it is **retracted**. That comparison decoded the 4.5M
model with a renderer that predates the NFC mark-order fix, so its output was
penalised by a defect rather than by model quality. Under the registered
tracker above, with NFC on both sides, the corrected gap is **-49.7%**.

---

## Unicode NFC - this model emits canonical text

Combining marks are emitted in **NFC order**: fatHa (ccc 30) sorts *before*
shadda (ccc 33), so a geminate renders `دَّ` and not `دّ َ`. The historical
renderer in this line emitted shadda-first, which disagrees with every NFC
reference on every geminate - measured at **4.5% of all units and 42% of the
registered gate** on the 4.5M model. `tashkeel_nano/labels.py` in this repo
carries the fixed `marks_for_label`, and inference additionally normalises
its output to NFC.

Note that this choice is *why* the benchmark's raw evaluator penalises this
model (see above): SadeedDiac-25's gold uses the older shadda-first order.
NFC is the correct Unicode serialization; the benchmark gold is the outlier.

---

## Architecture

Char-level bidirectional Transformer encoder with a per-position 15-class
diacritic head. Llama-style blocks (RMSNorm + RoPE + SwiGLU + grouped-query
attention) with the **causal mask off**.

| | |
|---|---|
| layers | 18 |
| hidden | 64 |
| FFN | 256 (SwiGLU) |
| heads | 8 (GQA, 2 KV heads, head_dim 8) |
| context | 256 |
| vocab | 171 (char-level, deterministic) |
| classes | 15 |
| parameters | 1,083,328 |

The 15 classes are `bare, fatha, damma, kasra, sukun, fathatan, dammatan,
kasratan, shadda, shadda_fatha, shadda_damma, shadda_kasra, shadda_fathatan,
shadda_dammatan, shadda_kasratan`.

Non-Arabic content (digits, latin, emoji, tatweel, Quranic annotation,
invisibles) is **passthrough**: it stays visible to the encoder as context but
is never a prediction target and is re-inserted byte-exact, so
`reconstruct(parse(x)) == x` always holds.

### Decoding

`centered` (default): overlapping windows of 256 chars at stride 128 are
scored, and each Arabic letter takes the mark from the window whose **centre**
is nearest to it - no majority vote, ties broken on the mark's codepoint order.
`single` is the non-overlapping hard-chunk fallback (cheaper, about 0.8 DER
worse on the registered tracker).

---

## Training

| | |
|---|---|
| corpus | `v3full` - the full v3 corpus **with** real space tokens (15.28% of positions) |
| steps | 60,000 |
| batch | 64 |
| LR | 2e-4 constant (AdamW, no schedule) |
| weight decay | 0.1 |
| precision | fp16 autocast |
| published artifact | uniform mean of the 12 step probes at 32,500 to 60,000 |

**Why real space tokens matter:** the 4.5M model was trained on a bed with
**no space tokens**, so its space embedding row is never trained and raw-text
decoding feeds it an untrained embedding. This model is trained on a bed with
real space tokens, so ordinary raw text is the correct input - no
whitespace-stripping workaround is needed.

---

## Usage

### From the HuggingFace hub (recommended)

~~~python
from huggingface_hub import snapshot_download
import sys

local = snapshot_download('ahmadsy/tashkeel-nano-1p1m')
sys.path.insert(0, local)

from tashkeel_nano import TashkeelNano

model = TashkeelNano.from_pretrained('ahmadsy/tashkeel-nano-1p1m')
print(model.diacritize('السلام عليكم ورحمة الله'))
# السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ
~~~

Only `torch`, `huggingface_hub` and `safetensors` are required; the tokenizer,
label schema and passthrough codec are stdlib-only and ship with the repo.

### CLI

~~~bash
python diacritize.py --text 'مرحبا بالعالم'
python diacritize.py --file input.txt
echo 'مرحبا بالعالم' | python diacritize.py
python diacritize.py --decode single --file input.txt   # cheaper, slightly worse
~~~

### Notes

- Input may be diacritized or bare; marks are stripped and the text is
  NFC-normalised before encoding, so both are accepted.
- Output is always NFC.
- CPU works; a GPU is used automatically when available.
- Long inputs are windowed internally - no manual chunking needed.

---

## Limitations & honesty

1. **Mark order changes the benchmark score.** See the note in the results
   section: the benchmark gold is shadda-first, this model is NFC. Both
   readings are published above; the normalised one is the comparable one.
2. **The ranking against the leaderboard is approximate.** The benchmark does
   not state whether the systems on its leaderboard were scored with or
   without mark-order normalisation.
3. **wikinews2024 is excluded** from the registered tracker as contaminated,
   and sadeed25 is overlap-flagged (Sadeed_Tashkeela is in this project's
   training mix), so the cleanest single reading there is wikinews2014.
4. **A soup is not a checkpoint.** The published weights are an average of 12
   checkpoints. Reproducing the exact number requires the same 12 members and
   the same uniform recipe, in ascending step order.
5. **The package's fp32 decode differs slightly from the registered fp16
   measurement** - about -0.12% relative, roughly 20x below the 2%
   significance bar used throughout this line.
6. **Not a Quranic diacritization model.** It predicts the 15 classes above;
   Quranic annotation marks are passthrough and are not predicted.

---

## Reproduction

SadeedDiac-25 scores: download the benchmark
(`Misraj/SadeedDiac-25`, `input`/`output` columns), run each model's own
published `diacritize` on the `input` column, then score with the benchmark
authors' evaluator from `misraj-ai/Sadeed` (`Eval_code.ipynb`),
`gt_missing_diacritic_is_error=False`. The mark-order-normalised variant
rewrites each NFC geminate `[base][vowel][shadda]` to the benchmark's
`[base][shadda][vowel]` before scoring.

~~~
benchmark        Misraj/SadeedDiac-25            (1200 paragraphs)
evaluator        misraj-ai/Sadeed Eval_code.ipynb (their own metric code)
1p1m predictions scratch/e104_pred_1p1m.json      (published package, centered decode)
4p5m predictions scratch/e104_pred_4p5m.json      (published package, as shipped)
scores           scratch/e104_sadeed_results.json
~~~

Registered-tracker numbers were produced with this project's scorer over
identical splits, with references byte-identical across all runs.

---

## Ancestor models

- `ahmadsy/tashkeel-nano-4p5m` - the 4.49M model from the same line, also
  published. On SadeedDiac-25 with the benchmark's evaluator it scores
  DER (CE) 9.91 / WER (CE) 31.03, against 3.78 / 12.73 for this model.

## License

Research-only; see `LICENSE`.
