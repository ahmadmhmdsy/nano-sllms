# Arabic OCR — training-data sourcing (real corpora + synthetic generation)

**Date:** 2026-10-08 · **Scope:** research only. No dataset/model was downloaded, no GPU job run, no repo file touched except this report.
**Model under study:** CNN-Transformer-CTC line recognizer, 12,143,271 params, 14 enc layers, d_model 256, 8 heads, ffn 1024, attention height-pooling, 165 symbols + CTC blank, **no diacritics**, input = single-line grayscale normalised to height 96.
**Measured gaps (free-running greedy CTC CER, raw / normalized):**

| set | raw | norm | source resolution |
|---|---|---|---|
| muharaf_test | 21.58 | 16.21 | 60 px |
| khatt_teklia_test | 16.46 | 14.97 | 149 px |
| khatt_bench | 25.25 | 23.90 | ? |
| isippt_bench | 53.93 | 53.34 | **27 px printed** |

Training data so far: Muharaf train 21,819 usable lines + KHATT teklia train 4,671 = 26,490 items. A clean Arabic text pool (10,400 texts / 5,263,840 chars) already exists at \`data/ocr_textpool/texts.jsonl\` (leakage-audited).
**Gap (a):** KHATT handwriting ~1.5x worse than a KHATT specialist (FixFips, 15.06M, 10.99 raw / 9.77 norm on the same set — \`research/EXPERIMENTS.md\` E-92).
**Gap (b):** low-resolution (27 px) printed text.

**Verification method.** Every row below was checked by fetching the actual page (HuggingFace dataset **API + raw README**, Zenodo **REST API**, GitHub **raw/API**, Kaggle **dataset-view API**). HuggingFace HTML pages are JS shells, so licence/size facts come from \`https://huggingface.co/api/datasets/<id>\` (\`cardData.license\`) and the raw README front-matter — i.e. the dataset's own card, not a search snippet. Anything I could not open is marked **UNVERIFIED**.

---

## PART 1 — MORE REAL DATA

### 1.1 Already in use (baseline for the licence comparison)

| Dataset | Licence | Lines / images | H/P | Resolution | Line-level GT | Location | Gated | Value vs our gaps |
|---|---|---|---|---|---|---|---|---|
| **Muharaf** (public part) | **CC BY-NC-SA 4.0**; restricted 428 pages proprietary, by e-mail | 36,311 text lines / 1,644 page images total; HF line set = **24,495 line images**; 1,216 public page images | H | line crops **height 60 px**, width 60–2400 (avg 576) | Yes (PNG lines + plain-text transcriptions) | HF \`aamijar/muharaf-public\`; Zenodo \`https://zenodo.org/records/11492215\`; code \`https://github.com/mehreenmehreen/muharaf\`; paper \`https://arxiv.org/html/2406.09630v1\` | No (public part) | Already used. Non-commercial + share-alike already binds us. |
| **KHATT v1.0** | Free **for academic/research** on request; LDC \`LDC2015T23\` for commercial | 1,000 writer forms; **9,327 lines** in the 2,000 random-text paragraphs; line images extracted | H | scanned 200/300/600 dpi; our crops 149 px tall | Yes (line + paragraph + page level, manually verified, Latin + Arabic) | request at \`http://khatt.ideas2serve.net/\`; LDC \`https://catalog.ldc.upenn.edu/LDC2015T23\`; paper \`https://faculty.kfupm.edu.sa/ics/alshayeb/publications/J20-KHATT.pdf\` | **Yes** — request/registration | The KHATT train we use is the Teklia mirror (\`Teklia/KHATT_v1.0\`). More KHATT is the *direct* lever on gap (a). |

### 1.2 New real-data candidates — verified

| # | Dataset | Licence | Lines / images | H/P | Resolution | Line-level GT | Exact location | Gated | Value vs our gaps |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **AMIDDA** (aggregator of 8 HTR sets) | card = \`other\` (aggregation). Sub-licences: BAYBARS / ISKANDAR / TARIMA = \`etalab-2.0\`; RASAM-1 = \`apache-2.0\`; KHATT + Muharaf keep **their own** licences; **RASM is "not redistributed"** | **66,779 lines** (train/val/test 53,420 / 6,675 / 6,684) | H (historical + modern) | not stated on card — **UNVERIFIED** | Yes, one line per row + \`dataset\` + \`transcription_guidelines\` columns | HF **\`calfa-ai/AMIDDA\`** | No | **Highest-value net-new handwriting source** if you drop the KHATT + Muharaf rows (already ours) and the RASM row. Script mix: Maghrebi + Oriental + modern cursive. |
| 2 | **BAYBARS** | \`etalab-2.0\` | **15,567 lines** | H (Arabic Oriental script, manuscripts) | UNVERIFIED | Yes | HF \`calfa-ai/baybars\` | No | Large, cleanly licensed manuscript hand. |
| 3 | **AraMS-28k-HTR** | **CC BY-NC-SA 4.0** | **27,971 main + 629 margin = 28,600 line annotations**; 3,043 pages, 14 books | H (Naskh, Ruq'ah, Maghrebi) + 1 lithographed **printed** volume | UNVERIFIED | Yes (line crops + text; \`AraMS-28k-HTR.zip\` = **2.42 GB**) | HF **\`Archatext/AraMS-28k-HTR\`** (page-level \`Archatext/AraMS-28k\`); repo \`https://github.com/ArchaText/AraMS-28k-Dataset\`; paper arXiv 2608.26921 | No | Second-biggest line-level handwriting set; adds Ruq'ah + Maghrebi. NC-SA licence. |
| 4 | **ISKANDAR** | \`etalab-2.0\` | **5,221 lines** | H (Oriental) | UNVERIFIED | Yes | HF \`calfa-ai/iskandar\` | No | Medium. |
| 5 | **RASAM-1** | \`apache-2.0\` | **7,331 lines** | H (Arabic Maghrebi) | UNVERIFIED | Yes | HF \`calfa-ai/RASAM-1\` | No | Medium; cleanest licence of the Calfa family. |
| 6 | **RASAM-2** | **UNVERIFIED** (not fetched; its siblings are apache-2.0 / etalab-2.0) | 3,720 lines | H (Maghrebi) | UNVERIFIED | Yes | HF \`calfa-ai/RASAM-2\` | No | Small. |
| 7 | **TARIMA** | \`apache-2.0\` | **2,285 lines** | H (Maghrebi) | UNVERIFIED | Yes | HF \`calfa-ai/tarima\` | No | Small. |
| 8 | **OpenITI MAKHZAN 2025.1.1** | **CC BY-NC-SA 4.0** (Zenodo API \`metadata.license.id\`) | **208 documents (200 manuscripts + 8 printed Urdu), ~1,500 page images**; zip **4.05 GB** | Both (manuscript + print) | UNVERIFIED | **Page-level** GT with segmentation (line boxes inside ALTO/page XML — **UNVERIFIED**) | Zenodo \`https://zenodo.org/records/16410106\`; paper \`https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.465\` | No | Covers printed + manuscript, multi-script (ar/fa/ota/ur) — but page-level, so a line-extraction pass is required. |
| 9 | **AHDQ** (Arabic Handwriting Dataset from Quran) | **MIT** | **642 binarised multi-line page images** (6 handwriting styles), **111.8 MB** | H (Quranic, Wolof community, Senegal) | UNVERIFIED | Text files per image; **line segmentation not provided** → page-level | Kaggle \`https://www.kaggle.com/datasets/abdoukhadrembacke/ahdq-arabic-handwriting-dataset-from-quran\` | No (Kaggle login) | New, licence-clean **handwriting** style; needs our own line detector. Binarised = far from our 96-px grayscale pipeline. |
| 10 | **Khattat** (\`saeidseyfi/khattat\`) | **CC BY 4.0** | 21,726 / 2,062 / 924 lines; **Arabic subset ≈ 6,627 train lines** (fa 10,533) | **SYNTHETIC** handwriting (dynamic-pen rendering) | unknown, "handwritten line crop" | Yes (\`image\`,\`text\`,\`language\`,\`style\`,\`augmented\`) | HF \`saeidseyfi/khattat\` (1.29 GB) | No | **Not real data** — it is a synthetic set (see Part 2 §2.6). Listed here because its name is easily mistaken for KHATT. |
| 11 | **medyas/arabic-ocr-printed-500k** | **CC BY-SA 4.0** | **500,000 train + 2,000 val** printed line JPEGs | **SYNTHETIC** printed | "realistic scan degradations", 15 typeface families; exact px **UNVERIFIED** | Yes | HF \`medyas/arabic-ocr-printed-500k\` | No | **Not real data** — see Part 2 §2.7. |

### 1.3 Printed-Arabic sources for gap (b)

| Dataset | Licence | Size | Level | Location | Status |
|---|---|---|---|---|---|
| **PATS-A01** | **UNVERIFIED** (page would not load; no licence text retrieved) | **2,766 printed text-line images** (2,751 from two classic books) | **line** | \`https://faculty.kfupm.edu.sa/ics/muhtaseb/ArabicOCR/PATS-A01.htm\` | Existence verified via the page's indexed text, but I could not open the page myself → treat the 2,766 figure as **partially verified**. Access procedure unknown. |
| **PATD** (Printed Arabic Text Database) | **UNVERIFIED** | 810 scanned grayscale images at several resolutions + a smartphone-captured mode | page | \`http://www.inf.u-szeged.hu/patd/\`; paper \`https://dl.acm.org/doi/10.1145/3323933.3324082\` | Existence verified via search snippet only → **UNVERIFIED**. |
| **APTI** (Arabic Printed Text Image DB) | **UNVERIFIED** | multi-font / multi-size / multi-style **word** images | word | \`https://icosys.ch/wp-content/uploads/pdfs/icdar-2009_A-New-Arabic-Printed-Text-Image-Database-and-Evaluation-Protocols.pdf\` | **UNVERIFIED**; word-level, wrong granularity for our CTC line setup. |
| **freococo/ocr_arabic_books** | **CC BY-NC-ND 4.0** | 100K<n<1M page images (per-split counts: e.g. 3,302 / 8,376 / 6,277 / 13,191 / 22,631 …) | page (fields: \`page_header\`,\`body\`,\`footnotes\`) | HF \`freococo/ocr_arabic_books\` | Verified. **NoDerivs** → cropping to lines is legally uncomfortable. |
| **HeshamHaroon/arabic-turath-ocr** | **not listed on card** | 100K<n<1M | page? | HF \`HeshamHaroon/arabic-turath-ocr\` | Exists; **licence UNVERIFIED**. |
| **mustaphaelkady/arabic-ocr-books** | **not listed** | 1K<n<10K | ? | HF \`mustaphaelkady/arabic-ocr-books\` | Exists; **licence UNVERIFIED**. |
| **TheRealOKAI/arabic-ocr-datasetv3**, **Yousefmd/arabic_ocr_dataset**, **JayanthMuthu/arabic-ocr**, **context212/context212-alhazen-ocr-khattmix** | not listed | 10K<n<100K each | mixed | same HF ids | Exist; **licence and line-level GT UNVERIFIED**. |

### 1.4 Screened and rejected for our line recogniser (wrong task shape)

| Dataset | Licence | Why not |
|---|---|---|
| **AHCD** (Arabic Handwritten Characters) | Kaggle card: **"Database: Open Database, Contents: Database Contents"** (ODbL-style), 79.3 MB | 16,800 **isolated character** images at 28×28. Our model reads a *line sequence* with CTC; a character classifier is not usable as line supervision. \`https://www.kaggle.com/datasets/mloey1/ahcd1\` |
| **Hijja / Hijja2** | Kaggle Hijja = **"Unknown"**; HF \`arbml/Hijja2\` has **no licence tag** | 47,434 **characters** (letters) from schoolchildren, per-letter folders. Wrong granularity + licence risk. \`https://www.kaggle.com/datasets/mohamedgreshamahdi/hijja-dataset\`, \`https://huggingface.co/datasets/arbml/Hijja2\`, code \`https://github.com/israksu/Hijja2\` |
| **IFHCDB** | **UNVERIFIED** (access not tested) | 70,120 **isolated Farsi/Arabic characters**. Wrong granularity; also Farsi. \`https://inria.hal.science/inria-00112676\` |
| **ADAB** | IEEE DataPort, licence text **UNVERIFIED** | **Online** handwriting, InkML/UPX pen trajectories — our model is **offline** images. \`https://ieee-dataport.org/open-access/adab-database\` |
| **IFN/ENIT** | free for non-commercial research, **registration form** | Handwritten **words** (Tunisian), not lines. \`http://www.ifnenit.com/register.htm\` |
| **QUWI** (Qatar University) | — | **Writer identification**, Arabic+English pages — not transcription. \`https://tsapps.nist.gov/BDBC/Search/Details/487\` |
| **AHDB** (Qatar University Arabic Handwriting DB) | **UNVERIFIED** — site \`http://handwriting.qu.edu.qa/dataset/\` not reachable | Existence inferred from the NIST database catalogue, not from the source → **UNVERIFIED**. |
| **VML-HD** | **CC BY-NC-ND 3.0** (TC-11 dataset page) | Historical Arabic, 668 pages / 5 books; **NoDerivs** conflicts with cropping + augmentation. \`https://tc11.cvc.uab.es/datasets/VML-HD_1\`, \`https://github.com/majeek/vml-hd\` |
| **VML-AHTE** | **UNVERIFIED** | Text-**line extraction** benchmark (labels are line boxes / transcripts — **UNVERIFIED**). \`https://github.com/beratkurar/arabic_handwritten_textline_extraction_dataset\` |

### 1.5 KHATT mirrors on the Hub — licence hazard table

There are many re-uploads of KHATT. Their declared licences **contradict the upstream "free for research on request" terms**, so none of them may be treated as the licence basis.

| HF id (verified via API) | Declared licence | Splits | Note |
|---|---|---|---|
| \`Teklia/KHATT_v1.0\` | **not declared** | 1K<n<10K | **the mirror we currently train on** |
| \`benhachem/KHATT\` | **not declared** | 1K<n<10K; repo holds \`data/train.zip\`, \`data/validation.zip\`, \`Groundtruth-Unicode.xlsx\` | no licence field → **UNVERIFIED** |
| \`johnlockejrr/KHATT_v1.0_dataset\` | card says **MIT** | 190 MB | **MIT on a research-only corpus is wrong** — do not rely on it |
| \`Nada2125/Khatt-Dataset-Unique-lines-full\` | **MIT** | 1K<n<10K | same caveat |
| \`AsmaaMahmoudSaeddd/Big_Khatt\` | **not declared** | train 11,216 / test 159 | |
| \`wahdan2003/khatt_iam\` | **not declared** | train 18,025 | stored pre-tokenised (\`pixel_values\`,\`labels\`) |
| \`ahmedheakl/arocrbench_khatt\` | MIT | n<1K | **benchmark** (ArabOCRBench), not a training corpus |
| Kaggle \`nizarcharrada/khattarabic\` | **"Unknown"** | 344.2 MB | Kaggle API verified |
| Kaggle \`iraqyomar/khatt-arabic-hand-written-lines\` | **"Other (specified in description)"** | 340.4 MB | description itself says "Read License of author's site" |

**Consequence:** if we want more KHATT lines we should either (i) re-request KHATT v1.0 from KFUPM, or (ii) use the Calfa/KHATT slice inside AMIDDA and **deduplicate against \`khatt_teklia/test\`**. Do not "fix" licensing by picking an MIT-tagged re-upload.

---

## PART 2 — MACHINE / SYNTHETIC DATA GENERATION

### 2.0 SEARCH THE REPO FIRST — what already exists (do not rebuild)

A working synthetic engine **already exists**; it is under \`arabic_ocr/src/data/\`, **not** under the root \`scripts/\`.

| Artifact | What it does |
|---|---|
| \`arabic_ocr/src/data/synthetic.py\` (443 lines) | Shaping with **HarfBuzz (\`uharfbuzz\`)**, rasterising with **FreeType (\`freetype-py\`)**, page composition, exact **PAW** boxes from ink connected components merged by HarfBuzz cluster, and \`degrade()\`. Its docstring records the key trap: *"PIL on this machine is built WITHOUT Raqm … the output was disconnected and left-to-right."* |
| \`degrade()\` in the same file | Gaussian blur (0.4–1.4, p .35), Gaussian noise (σ 4–18, p .5), contrast/gamma jitter (p .3), rotation ±1.2° (p .3), **JPEG q35–85 (p .35)**, polarity invert (p .12) |
| \`find_arabic_fonts()\` | Scans **\`C:/Windows/Fonts\`** for TTF/OTF that carry nominal glyphs for ا ل م |
| \`arabic_ocr/src/data/synthetic_pages.py\` | torch \`Dataset\` over rendered pages |
| \`arabic_ocr/scripts/build_stage3_data.py\` | Materialises synthetic pages → per-line crops + \`labels.json\`; also rebuilds the vocab |
| \`arabic_ocr/scripts/build_line_dataset.py\` | \`REAL_SOURCES\` + \`SYNTH_SOURCES = ['data/full/train','data/full/val']\`; flag \`--include-synthetic --synthetic-limit 12000\` |
| \`arabic_ocr/scripts/fetch_hf_lines.py\` | **The HF ingestion path to extend**: \`DATASETS = {"khatt_teklia": ("Teklia/KHATT_v1.0", …), "muharaf": ("aamijar/muharaf-public", …)}\` → \`data/hf/<name>/<split>/{images,labels.json}\`. Adding a candidate = adding one dict entry. |
| \`arabic_ocr/scripts/finetune_muharaf.py::light_augment\` (l.119–149) | **Real, implemented** augmentation: ±2.5° rotation, ±2.0° x-shear, 0.94–1.06 scale, ±2% translation, 50 % contrast 0.85–1.15, white fill preserved. Seeded by \`(seed, epoch, index)\` so a resumed run reproduces the same draw. Run via \`--augment light\` (default is \`none\`). |
| \`arabic_ocr/src/data/augmentation.py\` | \`RandomAugment\` is a **no-op stub** — every method is \`return img\` with a TODO. **Do not assume it works.** |

**Stage-1 gate evidence** (\`runs/stage1_gate/verify.json\`): 200 pages / 1,600 lines / **57 Arabic fonts**, **28–44 px**, with degradations — 0 coverage failures, 0 PAW-box mismatches, 0 orphan glyphs, 0 empty renders, 0 transcript mismatches, 0 uncovered / multi-covered ink px. So the *mechanics* are proven.

**But the training value was already tested and FAILED.** \`research/EXPERIMENTS.md\` **E-87**: a 7.9M \`vit_hybrid\` pretrained on **16,000 synthetic line crops** then fine-tuned on the 25,921 real lines scored **free-running CER 0.9100 vs the 0.5780 zero-shot baseline — +33.20 pp WORSE** (\`runs/stage3_finetune/gate.json\`). Our font-rendered synthetic lines did **not** transfer to real handwriting.

### 2.1 Font licensing (all four verified from their own OFL files)

| Font | Licence | Verified at |
|---|---|---|
| **Amiri** | **SIL OFL 1.1** | \`https://github.com/aliftype/amiri/blob/main/OFL.txt\`, \`https://github.com/google/fonts/blob/main/ofl/amiri/OFL.txt\` |
| **Cairo** | **SIL OFL 1.1** | \`https://github.com/google/fonts/blob/main/ofl/cairo/OFL.txt\`, \`https://github.com/Gue3bara/Cairo/blob/master/OFL.txt\` |
| **Noto Naskh Arabic** | **SIL OFL 1.1** ("All Noto fonts are licensed under the Open Font License") | \`https://notofonts.github.io/noto-docs/website/use/\`, \`https://cdn.jsdelivr.net/npm/@fontsource/noto-naskh-arabic@5.3.0/LICENSE\` |
| **Scheherazade New** | **SIL OFL 1.1** | \`https://raw.githubusercontent.com/silnrsi/font-scheherazade/master/OFL.txt\`, \`https://github.com/google/fonts/blob/main/ofl/scheherazadenew/OFL.txt\` |

OFL 1.1 permits commercial use, bundling into a product, and rendering images freely; it forbids selling the fonts by themselves and restricts reuse of Reserved Font Names. **Rendered output images are not a derivative of the font in any way that constrains us.**

**Repo caveat:** the generator currently scans **\`C:/Windows/Fonts\`** only. Windows-bundled Arabic fonts (Traditional Arabic, Simplified Arabic, Segoe UI, Tahoma, Arial…) are proprietary. Rendering with them locally is fine, but they must **not** be vendored into the repo, and the generator will not reproduce on a machine without them. Recommended change: vendor a small OFL set (Amiri + Cairo + Noto Naskh Arabic + Scheherazade New + a set of OFL calligraphic faces) and point \`FONT_DIRS\` at it.

### 2.2 What "correct shaping" requires, and what commonly goes wrong

Arabic is cursive and context-sensitive (isolated / initial / medial / final forms, plus lam-alef ligatures and mark stacking). Two implementation routes exist:

1. **A real shaper — HarfBuzz (\`uharfbuzz\`), or PIL built with libraqm.** This is the correct route and **the repo already uses it**. \`synthetic.py\` shapes the buffer with HarfBuzz, reads per-glyph advances for glyph boxes, and uses the HarfBuzz **cluster** to keep a floating haraka attached to its letter.
2. **\`arabic_reshaper\` + \`python-bidi\`.** This is a *text-level* workaround for engines with no shaper: it rewrites logical Unicode into Arabic **presentation forms** (U+FE70–FEFF) and applies bidi ordering. It is strictly weaker — no real mark positioning, no OpenType feature pipeline, ligature coverage from a hand-maintained table only — and it **actively corrupts text when applied on a shaper-enabled build** (see the Pillow docs issue "warn that pre-applying arabic_reshaper + python-bidi corrupts text on a Raqm build", \`https://github.com/python-pillow/Pillow/issues/9907\`). \`arabic_reshaper\` docs: \`https://github.com/mpcabd/python-arabic-reshaper\`.

Common failure modes to guard against:
- **No shaping at all** → isolated, left-to-right letters (exactly the bug \`synthetic.py\`'s docstring says it replaced). Test: render "السلام" and check for connected medial forms.
- **Double application** of reshaper (or reshaper on top of Raqm) → mangled, disconnected output.
- **Missing bidi inversion** → visually reversed word order.
- **Digit/Latin runs inside Arabic** → bidi re-ordering errors.
- **Missing glyph fallback** → tofu boxes; the repo's \`has_arabic()\` probe (nominal glyphs for ا ل م) is the cheap guard, but it does not guarantee full-charset coverage — extend it to every codepoint in the 165-symbol charset.
- **Diacritics** — deliberately excluded from our charset; a renderer that emits harakat produces labels the model cannot express.
- **ZWNJ/ZWJ and tatweel (kashida)** handling.

### 2.3 Degradation for domain transfer

\`synthetic.py::degrade()\` already covers blur, noise, contrast, small rotation, JPEG and invert. Missing knobs worth adding, in priority order against our actual gaps:

1. **Explicit low-resolution round-trip** — render at 60–149 px, then **downsample to ~27 px and back up to 96** before the pipeline sees it. This is the only knob that directly addresses gap (b). (A related probe already exists: \`arabic_ocr/scripts/image_size_probe.py\`.)
2. **Ink bleed / stroke-weight variation** — morphological dilate/erode on the ink mask, plus per-stroke-width jitter.
3. **Shear + elastic distortion** — \`light_augment\` has shear for real lines; \`degrade()\` does not.
4. **Paper texture / background** — compositing onto scanned-paper backgrounds. Note the Khat² authors collected 130+ paper backgrounds for exactly this (\`https://github.com/7abushahla/Khat_Squared\`).
5. **Scanner illumination gradient / vignetting / uneven contrast**.
6. **JPEG at very low quality, then re-blur** — model the "27 px source" double degradation.

### 2.4 Augmenting REAL lines vs synthesising NEW ones — the distinction that matters

- **Augmenting real lines** perturbs an image that already came from a human pen on real paper, so the *class-conditional distribution is unchanged*: real stroke topology, real pen pressure, real word spacing, real paper. It only teaches invariance (rotation, shear, scale, contrast). This is why it is cheap and safe — and it was **already measured as a large win in this project**: STATUS.md records that our recipe omitted the authors' \`augmentation_mode: "light"\` (ours was \`"augmentation": "none"\`) and \`train_blank_logit_penalty\`; \`light_augment\` is now implemented in \`finetune_muharaf.py\`. See also \`arabic_ocr/scripts/blank_penalty_tune.py\`.
- **Synthesising new lines** changes the *manifold*: the pixels are font outlines rasterised by FreeType, not ink deposited by a hand. It buys unlimited volume, deterministic labels, exact boxes, controlled vocab coverage — but every pixel is off-distribution for handwriting. Augmentation explores within the real manifold; synthesis creates a second manifold that may barely intersect it.

This distinction is exactly why E-87's 16,000 synthetic pretraining lines *hurt*: the synthetic manifold did not overlap real handwriting enough to help the subsequent fine-tune, while it did shift the initialisation.

### 2.5 Arabic HANDWRITING synthesis — what exists and whether it is usable

| System | What it is | Arabic? | Licence | Usable? |
|---|---|---|---|---|
| **Arabic-One-DM** | Adaptation of One-DM to Arabic one-shot handwritten-text generation: ResNet-18 style encoders + an HTR model + SD-1.5 VAE, three stages (Khat² synthetic pretrain → real-dataset aggregation → architecture change). Paper "One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation". | **Yes** | **No LICENSE file in the repo** (raw \`main/LICENSE\` → 404) | In principle the only real Arabic handwriting generator. **Not turnkey**: needs the Khat² pretrain corpus + 4 real datasets + GPU training. Licence is a blocker for anything redistributed. \`https://github.com/7abushahla/Arabic-One-DM\` |
| **Khat²** | Synthetic Arabic **word** images: **2,000+ calligraphic fonts scraped from multiple websites**, 130+ paper backgrounds, the HATFormer 8.2M-word Arabic corpus; generates N×N (words × fonts) images with random background + distortion/blur. | **Yes** | **No LICENSE file** (404); fonts are scraped with unvetted licences | Generation code + \`dataset/data.csv\` + \`dataset/gen_line_images.py\` are public. **Licence risk: the fonts.** \`https://github.com/7abushahla/Khat_Squared\` |
| **saeidseyfi/khattat** | Released **synthetic** handwritten-line corpus (fa/ar/en), dynamic pen with variable stroke width/pressure, scanner/camera distortion variants; 21,726/2,062/924, **Arabic ≈ 6,627 train lines**, CC BY 4.0. | Yes (mixed) | **CC BY 4.0** | **The cheapest available probe of handwriting-synthesis value** — already rendered, licence-clean, line-level. |
| **One-DM** | ECCV 2024 one-shot diffusion mimic (en/zh/ja). | No | verified only that the repo exists | Baseline the Arabic work builds on. \`https://github.com/dailenson/One-DM/\` |
| **CONSTANT** | WACV 2026 one-shot handwriting generation (patch contrastive + style-aware quantisation). | No (not Arabic-specific) | UNVERIFIED | Research only. \`https://github.com/blk31/CONSTANT\` |
| **AR-DiffHTG** | Autoregressive-conditional diffusion HTG. | UNVERIFIED | UNVERIFIED | \`https://github.com/wh807088026/AR-DiffHTG\` |
| **DiffusionHTR / SyntheticHTR** | "Synthetic handwriting dataset generation using diffusion models". | UNVERIFIED | UNVERIFIED | \`https://github.com/liamtabib/SyntheticHTR\` |
| **HATFormer** | Historic **handwritten Arabic text-line recognition** (arXiv 2410.02179) — a recogniser, not a generator; also compiled the 8.2M-word Arabic corpus Khat² reuses. | Yes | UNVERIFIED | Useful as a teacher/corpus, not as a generator. |

**Verdict:** there is **no turnkey, licence-clean Arabic handwriting synthesiser**. The two Arabic options are research code with missing licences; the practical runway is a released synthetic set (Khattat) as a cheap falsification probe.

### 2.6 Printed synthetic vs handwriting synthetic — which domain would it actually help?

This is the honest risk analysis the task asks for.

**Printed / low-resolution (gap b) — synthetic printed is plausibly worth it.**
- Our model already scores **53.34 normalised CER on ISIPPT with zero printed training** — i.e. it partially reads printed text already; the failure looks like a *rendering/resolution* mismatch, not vocabulary.
- External evidence that synthetic printed works at scale: \`medyas/arabic_PP-OCRv6_small_rec\` is recorded (STATUS.md, §"2026-10-04 (a)") as **CER 10.2 % real printed / 13.5 % ISIPPT held-out from 800k synthetic lines over 26 base + 12 display fonts**. ⚠️ This is a **model-card claim reproduced in STATUS.md, not independently verified by us — UNVERIFIED**.
- E-76 records \`PaddlePaddle/arabic_PP-OCRv5_mobile_rec\` at **8.70 % free-running CER on real printed lines** (87.4 % of lines under 20 %), versus our from-scratch control at 144.8 % — so printed Arabic is solvable in this parameter class.
- Cheapest path: **do not build a renderer**; \`medyas/arabic-ocr-printed-500k\` (CC BY-SA 4.0) is already 500k+2k rendered printed lines.
- **But the cheaper-than-that first move is §2.3 item 1**: a resolution/downscale augmentation experiment on the existing real lines, because gap (b) is measured on a 27 px source and our own engine already renders 28–44 px.

**Handwriting (gap a) — synthetic handwriting is *not* justified yet.**
- Direct in-repo evidence against it: **E-87 FAIL** (synthetic pretrain + real fine-tune = 0.9100 vs 0.5780).
- Font-rendered "handwriting" is still font rendering. The Khat² approach (2,000+ calligraphic fonts) is the same idea our 57-font Windows engine already does — just larger and with a licence problem.
- A real generator (Arabic-One-DM) needs a full training pipeline, has no licence, and its payoff is unproven for a 12M-param CTC line reader.
- Cheap falsification: train one arm that adds \`saeidseyfi/khattat\`'s **Arabic** lines (≈6.6k) to the existing 26,490 real items and score on \`khatt_teklia_test\` / \`muharaf_test\`. If that arm does not move the handwriting CER, do not invest in generation.

**Net:** synthetic **printed** = plausibly helpful and cheap to test with a released corpus; synthetic **handwriting** = low expected value until a cheap released-set probe says otherwise. **Real handwriting data (Part 1 §1.2) dominates both for gap (a).**

---

## RANKED RECOMMENDATION TABLE

Effort: S = hours, M = a day, L = multi-day. Cost = download/disk + GPU. Licence risk: 🟢 low / 🟡 needs a call / 🔴 blocking. Value is against *our* two gaps.

| # | Action | Effort | Cost | Licence risk | Expected value | Against gap |
|---|---|---|---|---|---|---|
| 1 | Add **AMIDDA** net-new rows (drop KHATT + Muharaf rows to avoid re-training on our test data; RASM is not redistributed) — ≈34k historical manuscript lines from BAYBARS/RASAM-1/2/TARIMA/ISKANDAR | S (one \`fetch_hf_lines.py\` entry) | download + one fine-tune | 🟡 aggregator card is \`other\`; sub-sets are etalab-2.0 / apache-2.0 | **High** — largest licence-tractable handwriting injection; new scripts (Maghrebi, Oriental, Ruq'ah) | (a) |
| 2 | **Deduplicate + add more KHATT lines** from AMIDDA's KHATT slice / KFUPM re-request; hard-check overlap with \`data/hf/khatt_teklia/test\` | S–M | download + fine-tune | 🟡 upstream is research-only; **do not** use MIT-tagged mirrors as the basis | **High** — directly the gap where FixFips beats us 1.5x | (a) |
| 3 | **AraMS-28k-HTR** (28,600 line crops, 2.42 GB) | S | 2.4 GB + one fine-tune | 🟡 CC BY-NC-SA 4.0 (non-commercial, share-alike) | **Med–High** — big, exact line-level, adds Ruq'ah/Maghrebi | (a) |
| 4 | **Resolution augmentation on existing real lines** (downsample→27 px→96, plus shear/bleed/texture) — no new data, extend \`light_augment\` | S | GPU time only | 🟢 none | **High per unit effort** — the only lever aimed exactly at 27 px | (b) |
| 5 | **OpenITI MAKHZAN** (4.05 GB; 200 manuscripts + printed Urdu) | M (page-level → needs line extraction) | 4 GB + extraction pass | 🟡 CC BY-NC-SA 4.0 | **Med** — both print and manuscript, but page-level | (a)+(b) |
| 6 | **medyas/arabic-ocr-printed-500k** synthetic printed (500k+2k) as a printed-pretrain arm | S | download + pretrain | 🟡 CC BY-SA 4.0 (share-alike; matters if we redistribute weights) | **Med** — plausible printed/low-res gain; external card supports it (UNVERIFIED) | (b) |
| 7 | **Khattat** Arabic subset (≈6.6k synthetic lines) as a cheap falsification of handwriting synthesis | S | download + one arm | 🟢 CC BY 4.0 | **Med as an experiment, low as a fix** | (a) |
| 8 | **AHDQ** (642 Quranic page images, MIT) | M (needs our line detector) | 112 MB + detection | 🟢 MIT | **Low–Med** — new style but binarised, page-level | (a) |
| 9 | **PATS-A01 / PATD / APTI** printed | M (licence investigation first) | unknown | 🔴/🟡 **UNVERIFIED** | **Med if licences clear** — real printed lines at varied resolutions | (b) |
| 10 | **Arabic-One-DM / Khat²** handwriting generation | L (GPU training, 2k-font corpus) | high | 🔴 no LICENSE file; scraped fonts | **Low** until #7 proves the domain helps | (a) |
| 11 | **AHCD / Hijja / IFHCDB / ADAB / IFN-ENIT / QUWI / VML-HD / freococo books** | — | — | 🔴 various | **Do not use** — see DO-NOT-DO | — |

---

## DO-NOT-DO LIST

1. **Do not download or train anything without the user's explicit go** — this was a research task; no dataset or model was fetched.
2. **Do not train on AMIDDA's KHATT and Muharaf rows** — our \`khatt_teklia_test\` / \`muharaf_test\` sets are the KHATT v1.0 / muharaf-public splits; re-ingesting them through AMIDDA is leakage. Filter \`dataset\` to the net-new sources and deduplicate.
3. **Do not rely on MIT/CC tags on KHATT re-uploads** (\`johnlockejrr/KHATT_v1.0_dataset\`, \`Nada2125/\`…). The upstream is research-only on request; a re-uploader cannot relicense it.
4. **Do not crop or augment VML-HD (CC BY-NC-ND 3.0) or \`freococo/ocr_arabic_books\` (CC BY-NC-ND 4.0)** without an explicit legal call — NoDerivs vs cropping/augmentation is unresolved.
5. **Do not use licence-Unknown data as a training basis**: Kaggle Hijja ("Unknown"), Kaggle \`nizarcharrada/khattarabic\` ("Unknown"), \`benhachem/KHATT\` (no licence field), \`HeshamHaroon/arabic-turath-ocr\`, \`mustaphaelkady/arabic-ocr-books\`, \`TheRealOKAI/arabic-ocr-datasetv3\`, \`Yousefmd/arabic_ocr_dataset\`, \`context212/…-khattmix\`. A scratch probe is the most they can be.
6. **Do not use character-level datasets for a CTC line reader** — AHCD (16,800 chars @ 28×28), Hijja/Hijja2 (47,434 letters), IFHCDB (70,120 isolated chars). Wrong label granularity; the model needs sequences.
7. **Do not use online-ink or wrong-task corpora** — ADAB (InkML/UPX), IFN/ENIT (words), QUWI (writer ID), VML-AHTE (line *extraction*, transcription UNVERIFIED).
8. **Do not build a handwriting GAN/diffusion from scratch.** No licence-clean Arabic generator exists, and the in-repo evidence (E-87) says rendered-synthetic lines do not transfer.
9. **Do not repeat a from-scratch synthetic pretrain of the E-87 shape** (7.9M vit_hybrid on 16k synthetic lines) — it is a proven FAIL (+33.20 pp worse). Any synthetic arm must be an *incremental* arm on top of the current 26,490-line real recipe, measured against it.
10. **Do not render Arabic with a Raqm-less PIL or double-apply \`arabic_reshaper\`.** The repo's \`uharfbuzz\` + \`freetype-py\` path is the correct one; keep it.
11. **Do not vendor Windows-bundled Arabic fonts** (\`C:/Windows/Fonts\`) into the repo. Vendor OFL fonts (Amiri / Cairo / Noto Naskh Arabic / Scheherazade New) instead.
12. **Do not trust \`arabic_ocr/src/data/augmentation.py\`** — \`RandomAugment\` is a no-op stub. The working implementation is \`finetune_muharaf.py::light_augment\`.
13. **Do not assume printed synthetic data fixes handwriting.** Cap: synthetic printed may move gap (b); it will not move gap (a).
14. **SECURITY — do not commit the HF token already hard-coded in \`arabic_ocr/scripts/download_data.py\` (\`os.environ['HF_TOKEN'] = ...\`).** It should be removed from the source, moved to \`.env\`, and the token **rotated** (it is in the working tree and may be in git history). Reported here rather than in the file.

---

## UNVERIFIED / OPEN ITEMS (explicit)

- **AMIDDA, BAYBARS, ISKANDAR, RASAM-1/2, TARIMA, AraMS-28k, MAKHZAN image resolutions** — not stated on the cards → **UNVERIFIED**.
- **RASAM-2 licence** — not fetched (siblings are apache-2.0 / etalab-2.0) → **UNVERIFIED**.
- **OpenITI MAKHZAN line-level segmentation** — page-level GT with segmentation is advertised; whether per-line boxes can be extracted directly → **UNVERIFIED**.
- **PATS-A01 / PATD / APTI** — existence seen only in indexed text / search snippets; \`faculty.kfupm.edu.sa\` and \`inf.u-szeged.hu\` would not load for me → **partially verified / UNVERIFIED**.
- **AHDQ line segmentation, image resolution, exact style/count semantics** → **UNVERIFIED**.
- **Arabic-One-DM and Khat² licences** — \`LICENSE\` returns 404 in both repos → **no licence = all rights reserved by default**; **UNVERIFIED/risky**.
- **CONSTANT, AR-DiffHTG, DiffusionHTR, HATFormer licences** → **UNVERIFIED**.
- **\`medyas/arabic_PP-OCRv6_small_rec\` "800k synthetic lines → 13.5 % ISIPPT"** — a model-card claim recorded in STATUS.md, not reproduced by us → **UNVERIFIED**.
- **VML-AHTE** — whether it ships line transcriptions → **UNVERIFIED**.
- **Qatar University AHDB** — site unreachable; existence only from the NIST catalogue → **UNVERIFIED**.
- **\`benhachem/KHATT\` contents** — \`data/train.zip\` + \`data/validation.zip\` + \`Groundtruth-Unicode.xlsx\` confirmed via the repo tree, but the number of lines inside them → **UNVERIFIED**.

## Suggested next steps (for the user to gate)

1. Approve one experiment: **resolution/downscale augmentation** on the current recipe vs the current baseline (cheapest possible attack on gap (b); no new data, no licence).
2. Approve a **leakage-audited AMIDDA pull** (net-new rows only) as a second arm against gap (a), reusing \`arabic_ocr/scripts/fetch_hf_lines.py\` + \`audit_textpool_leakage.py\`.
3. Only if (2) underdelivers, approve the **Khattat Arabic subset** probe before any investment in Arabic handwriting synthesis.
4. Separately and urgently: **rotate + remove the hard-coded HF token** in \`download_data.py\`.
