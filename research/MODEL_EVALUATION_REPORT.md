# Model Evaluation Report — nano_slMs
# Generated: 2026-09-28
# Updated: 2026-09-29 (all benchmark on-disk sizes measured)
# Corrected: 2026-10-02 (WikiNews 2024 multi-ref DER recomputed with the
#   word-weighted aggregation fix in eval_multi_ref.py; previous values were
#   ~29.8x too small and the ranking was wrong)
# Corrected again: 2026-10-02 (the registered DER is sensitive to shadda/vowel
#   byte order, which labels.py declares equivalent; mishkala is vowel-first
#   and loses 7.25 points to it. Under the schema-faithful canonical condition
#   mishkala ranks FIRST, ahead of zmahood. See section 3.1.)

## 1. Code LLM SOTA Benchmarks

| Benchmark | Source | Problems | On-Disk Size | Metric | Best For |
|---|---|---|---|---|---|
| HumanEval | OpenAI | 164 | 0.2 MB (208 KB) | pass@k | Code generation, function-level |
| MBPP | Google | 500 (test split) | 0.27 MB (280 KB) | pass@k | Code generation, beginner-friendly |
| BigCodeBench | BigCode | 1,140 | 6.63 MB | pass@k | Diverse code generation, real-world |
| LiveCodeBench | — | 880 (release_v5) | 4.57 MB (repo) | pass@k | Contamination-free, latest problems |
| CodeContests | — | 16,534 | 25.17 GiB (too big) | pass@k | Competitive programming, hard problems |
| APPS | — | 10,000 | N/A (scripts unsupported) | pass@k | 3 difficulty levels, interview-style |
| small-code-models | arXiv 2506.10995 | Multiple clone-detection benchmarks | 0.56 MB (586 KB) repo | Clone detection | Specifically for <220M models |

### Benchmark On-Disk Sizes (measured 2026-09-29)

| Benchmark | Problems | Size on Disk | Source |
|---|---|---|---|
| HumanEval | 164 | **0.2 MB** (208 KB) | openai/openai_humaneval |
| MBPP | 500 (test split) | **0.27 MB** (280 KB) | google-research-datasets/mbpp |
| BigCodeBench | 1,140 | **6.63 MB** | bigcode/bigcodebench |
| LiveCodeBench | 880 (release_v5) | **4.57 MB** (repo) | LiveCodeBench/LiveCodeBench |
| CodeContests | 16,534 | **25.17 GiB** (too big to download) | deepmind/code_contests |
| APPS | 10,000 | **N/A** (dataset scripts unsupported) | codeparrot/apps |
| Design2Code | 485 | **0.08 MB** (85 KB) JSONL + images | SALT-NLP/Design2Code |
| WebGen-Bench | 6,667 | **4.91 MB** | luzimu/WebGen-Bench |
| UI-Bench | 30 | **0.04 MB** (39 KB) | AfterQuery/ui-bench |
| ReactBench | 1,618 | **163.07 MB** | IDEA-AI4S/ReactBench |
| small-code-models | Multiple benchmarks | **0.56 MB** (586 KB) repo | jorge-martinez-gil/small-code-models |

Note: Design2Code images are stored separately in the HuggingFace cache. The small-code-models repo is a framework; actual datasets download separately.

## 2. UI Chain Model SOTA Benchmarks

| Benchmark | Source | Size | Metric | Best For |
|---|---|---|---|---|
| Design2Code | ACL 2025 | 484 webpages | Visual similarity, text-to-HTML/CSS | Front-end generation, design fidelity |
| WebGen-Bench | NeurIPS 2025 | 6,667 examples | Interactive website generation | Full website generation, functionality |
| UI-Bench | arXiv 2508.20410 | 30 examples | Design capabilities, text-to-app | App-level UI generation |
| ReactBench | reactbench.com | 1,618 examples | React component generation | React-specific UI generation |

## 3. Diacritizer Multi-Reference Evaluation (WikiNews 2024)

### Results (356 sentences, 10,616 words)

Recomputed 2026-10-02 with the word-weighted aggregation fix in
`diacritizer/scripts/eval_multi_ref.py`. The previous table summed per-line
fractions and divided by total words, under-reporting DER by ~29.8x AND
changing the ranking (it over-weights short lines). Cross-checked against
`eval.py compare` on the same pred/ref (s7: 0.68604 gate vs 0.68642 here).

**Scale caveat**: the WikiNews 2024 gate inputs contain "/" alternatives, so
every model evaluated here copies "/" into its predictions. DER compares
diacritic marks only ("/" is not a mark), so it is robust to this, but the
absolute values are on a contaminated-input condition and are NOT comparable
to clean-input DER figures. mishkala/zmahood are recomputed with our
eval_multi_ref on this same condition (their published single-ref numbers
are on different conditions and not comparable).

| Model | Params | DER | DER_nocase | WER | Status |
|---|---|---|---|---|---|
| zmahood (published) | ~4.5M | 47.88% | 42.06% | 48.02% | Best published |
| mishkala (published) | 12.5M | 52.45% | 45.86% | 52.57% | Published |
| **e63s_s4_12l160** | **4.49M** | **55.90%** | **46.29%** | **55.94%** | **Best e63s** |
| stage2b2500 (gold) | 30.1M | 57.66% | 47.42% | 57.71% | Previous gold |
| e63s_s3_8l272 | 8.64M | 58.36% | 46.36% | 58.41% | |
| e63s_s5_10l128 | 2.40M | 59.23% | 49.07% | 59.29% | |
| e63s_s2_10l272 | 10.78M | 59.98% | 47.31% | 60.04% | |
| e63s_s1_12l272 | 12.93M | 60.51% | 47.95% | 60.56% | |
| e63s_s6_8l96 | 1.09M | 60.68% | 51.55% | 60.73% | Smallest |
| e63s_s7_18l64 | 1.08M | 68.64% | 60.06% | 68.69% | Diverged@9171 |

### 3.1 Registered DER is byte-order sensitive (found 2026-10-02)

`labels.label_for_marks` accepts a shadda+vowel run in EITHER byte order and
maps both to one class. Its docstring records this on purpose: "Real corpora
use BOTH orders: shadda+vowel (canonical) and vowel+shadda (raw Tashkeela
order). Both are the same gemination class."

`eval_der` compares raw mark STRINGS, so it does not honour that equivalence.
A model that emits vowel+shadda is scored wrong against a shadda-first
reference even though the schema calls the two identical. mishkala emits
vowel-first throughout and loses **7.25 DER points** to this alone - enough to
change which model ranks first.

The `canonical` condition rewrites every mark run to shadda-first order before
comparing. Measured on the same 356 lines / 10,616 words:

| Model | DER (registered) | DER (canonical) | delta |
|---|---|---|---|
| **mishkala** | 52.45% | **45.20%** | 7.25% |
| zmahood | 47.88% | 47.89% | -0.01% |
| e63s_s4_12l160 | 55.90% | 51.33% | 4.57% |
| stage2b2500 | 57.66% | 52.80% | 4.86% |
| e63s_s3_8l272 | 58.36% | 53.51% | 4.84% |
| e63s_s2_10l272 | 59.98% | 54.90% | 5.09% |
| e63s_s5_10l128 | 59.23% | 54.97% | 4.26% |
| e63s_s1_12l272 | 60.51% | 56.16% | 4.35% |
| e63s_s6_8l96 | 60.68% | 56.99% | 3.69% |
| e63s_s7_18l64 | 68.64% | 67.19% | 1.45% |

Under the canonical condition **mishkala is the best model on this gate**, not
zmahood, and the paired line bootstrap + exact McNemar agree (mishkala vs
zmahood: difference -2.69 points, 95% CI [-3.89, -1.11], p = 6.8e-39).

This corroborates the independent class metrics: mishkala also beats zmahood on
position accuracy (86.82% vs 82.34%) and macro-F1 (0.7378 vs 0.6820), which
are order-insensitive by construction. Before this was found, the word-level
DER said the opposite of the per-class metrics - that disagreement was the
signal.

The registered gate is left UNCHANGED so the historical series stays
comparable; the canonical condition is reported beside it. Any future
comparison against a published model must state which of the two it uses.

Evidence: `research/wn2024_full_metrics.json` and
`research/wn2024_full_metrics.md`, regenerated by
`diacritizer/scripts/eval_report.py` (parity-locked to `eval_der` by
`diacritizer/scripts/selftest.py --phase a0d`).

### Key Findings

1. **e63s_s4_12l160 (4.49M params) is the best e63s model on this gate**:
   - DER 55.90% on WikiNews 2024 multi-ref (contaminated-input condition)
   - Beats stage2b2500 (DER 57.66%) — the previous gold standard
   - 6.8x smaller than stage2b2500 (4.49M vs 30.1M params)
   - 6.7x smaller file size (17.2 MB vs 114.9 MB)
   - But LOSES to published zmahood (47.88% registered / 47.89% canonical)
     and mishkala (52.45% registered / 45.20% canonical)
   - NOT a SOTA/PUBLISHABLE claim: it does not beat the published
     baselines on this metric under either condition.

2. **e63s_s3_8l272 (8.64M params) does NOT outperform stage2b2500**:
   - DER 58.36% vs stage2b2500 57.66% (worse, not better)

3. **No e63s model beats the published baselines on this gate**:
   - Registered: zmahood 47.88% and mishkala 52.45% both beat every e63s model
   - Canonical: zmahood 47.89% and mishkala 45.20% still beat the best e63s
     model (e63s_s4_12l160 at 51.33%)
   - The e63s family spans 55.90%-68.64% registered / 51.33%-67.19% canonical

4. **Scale caveat**: these DER values are on the contaminated-input
   condition (WikiNews 2024 gate inputs contain "/"). The registered
   external gate (fadel/sadeed25/wn2014) is consistent where it overlaps:
   e63s_s4 0.4024 < stage2b2500 0.4263 < e63s_s6 0.4645 < e63s_s7 0.5608.
   A clean-input multi-ref DER requires regenerating predictions from
   clean (first-alternative) bare input.

### e63s Model Configurations

| Model | Layers | Hidden | FFN | Params | Size | DER |
|---|---|---|---|---|---|---|
| e63s_s1_12l272 | 12 | 272 | 1088 | 12.93M | 49.4 MB | 60.51% |
| e63s_s2_10l272 | 10 | 272 | 1088 | 10.78M | 41.2 MB | 59.98% |
| e63s_s3_8l272 | 8 | 272 | 1088 | 8.64M | 33.0 MB | 58.36% |
| **e63s_s4_12l160** | **12** | **160** | **640** | **4.49M** | **17.2 MB** | **55.90%** |
| e63s_s5_10l128 | 10 | 128 | 512 | 2.40M | 9.2 MB | 59.23% |
| e63s_s6_8l96 | 8 | 96 | 384 | 1.09M | 4.2 MB | 60.68% |
| e63s_s7_18l64 | 18 | 64 | 256 | 1.08M | 4.1 MB | 68.64% |

## 4. Models Worthy of Publishing

### Code LLMs (pipeline: S → P → T)

| Model | Path | Params | val_loss | Verdict |
|---|---|---|---|---|
| Target | runs/target/final/model.safetensors | 226.5M | 1.864 | Strong |
| Pilot | runs/pilot/final/model.safetensors | ~110M | 1.161 | Strong |
| U-1 Full | runs/u1_full/final/model.safetensors | 30.3M | 0.270 | Excellent |
| SFT-T1 | runs/sft_t1/final/model.safetensors | 226.5M | 0.836 | Good |
| SFT-v2-E1 | runs/sft_v2_e1/final/model.safetensors | 226.5M | 0.280 | Strong |
| KT2-SFT-R2B | runs/kt2_sft_r2b/final/model.safetensors | 236.2M | 0.370 | Good |

### UI Chain Models (pipeline_v2)

| Model | Path | Params | Gate Results | Verdict |
|---|---|---|---|---|
| U2H-EOS-Avg | runs/v2/ui/u2h_arm_eos_avg/.../weight-average.pt | 28M | 100% greedy, 100% pass@4 | Excellent |
| U2I-EOS-Avg5 | runs/v2/ui/u2i_arm_eos_avg5/.../weight-average.pt | 28M | 100% greedy, 100% pass@4 | Excellent |
| U5-Full-Depth12 | runs/v2/ui/u5_full_depth12/.../weight-average.pt | 24M | 91.7% pass@4 family, 90% novel | Good |

### Diacritizer Models (D-line) — NOT SOTA

DER on the WikiNews 2024 multi-ref gate (contaminated-input condition; see
§3 caveat), registered condition. Sorted best-first. No D-line model beats
the published baselines under either condition: mishkala 52.45% registered /
45.20% canonical, zmahood 47.88% / 47.89%. See §3.1 for why the two
conditions disagree and which one matches the label schema.

| Model | Path | Params | DER (multi-ref) | Size | Verdict |
|---|---|---|---|---|---|
| **e63s_s4_12l160** | **runs/diac/e63s_s4_12l160/final/model.pt** | **4.49M** | **55.90%** | **17.2 MB** | **Best e63s; beats gold; loses to published** |
| stage2b2500 | runs/diac/stage2b2500/final/model.pt | 30.1M | 57.66% | 114.9 MB | Previous gold |
| e63s_s3_8l272 | runs/diac/e63s_s3_8l272/final/model.pt | 8.64M | 58.36% | 33.0 MB | |
| e63s_s5_10l128 | runs/diac/e63s_s5_10l128/final/model.pt | 2.40M | 59.23% | 9.2 MB | |
| e63s_s2_10l272 | runs/diac/e63s_s2_10l272/final/model.pt | 10.78M | 59.98% | 41.2 MB | |
| e63s_s1_12l272 | runs/diac/e63s_s1_12l272/final/model.pt | 12.93M | 60.51% | 49.4 MB | |
| e63s_s6_8l96 | runs/diac/e63s_s6_8l96/final/model.pt | 1.09M | 60.68% | 4.2 MB | Smallest |
| e63s_s7_18l64 | runs/diac/e63s_s7_18l64/final/model.pt | 1.08M | 68.64% | 4.1 MB | Diverged@9171 |

### Already Published (HuggingFace)

| Model | Size | HF Repo |
|---|---|---|
| tashkeel-350m-v2 | 649 MB | Etherll/Tashkeel-350M-v2 |
| mishkala | 153 MB | flokymind/mishkala |

## 5. Files Created

### Report
- research/MODEL_EVALUATION_REPORT.md — This report

### Benchmark Data (downloaded)
- data/benchmarks/humaneval/humaneval.jsonl — HumanEval (0.2 MB)
- data/benchmarks/mbpp/mbpp.jsonl — MBPP (0.27 MB)
- data/benchmarks/bigcodebench/bigcodebench.jsonl — BigCodeBench (6.63 MB)
- data/benchmarks/livecodebench_repo/ — LiveCodeBench repo (4.57 MB)
- data/benchmarks/design2code/design2code.jsonl — Design2Code (0.08 MB)
- data/benchmarks/webgen_bench/data.jsonl — WebGen-Bench (4.91 MB)
- data/benchmarks/ui_bench/ — UI-Bench CSVs (0.04 MB)
- data/benchmarks/reactbench/data.parquet — ReactBench (163.07 MB)
- data/benchmarks/small-code-models/ — small-code-models repo (0.56 MB)

### Diacritizer Predictions (WikiNews 2024)
- models/e19/wn2024.gold.raw.pred.txt — stage2b2500 predictions
- models/e19/wn2024.mishkala.pred.txt — mishkala predictions
- models/e19/wn2024.zmahood.pred.txt — zmahood predictions
- models/e19/wn2024.e63s_s1_12l272.pred.txt — e63s_s1 predictions
- models/e19/wn2024.e63s_s2_10l272.pred.txt — e63s_s2 predictions
- models/e19/wn2024.e63s_s3_8l272.pred.txt — e63s_s3 predictions
- models/e19/wn2024.e63s_s4_12l160.pred.txt — e63s_s4 predictions
- models/e19/wn2024.e63s_s5_10l128.pred.txt — e63s_s5 predictions
- models/e19/wn2024.e63s_s6_8l96.pred.txt — e63s_s6 predictions
- models/e19/e63s_model_info.json — e63s model configurations

### Full-Metric Harness (added 2026-10-02)
- research/wn2024_full_metrics.json — 50+ metrics for all 10 models, with CIs,
  paired significance, per-class recall, and confusion matrices
- research/wn2024_full_metrics.md — the same as a readable report
- research/wn2024_models.json — the model manifest the driver reads
- diacritizer/src/eval_metrics.py — reusable metric library (pure stdlib)
- diacritizer/scripts/eval_report.py — reusable driver (CPU only)
- pipeline_v2/slm_train/evaluation/diacritizer_suites.py — the declared metric
  suite, reusing pipeline_v2 MetricSpec/SuiteSpec/Threshold
- pipeline_v2/slm_train/evaluation/diacritizer.py — artifact to
  EvaluationReport bridge
- pipeline_v2/tests/unit/test_diacritizer_evaluation.py — 17 tests
- diacritizer/scripts/selftest.py — new phase a0d parity-locks the library
  against eval_der (25 checks; gate is now 4/4 PASS)

### Scripts
- diacritizer/scripts/eval_multi_ref.py — Multi-ref evaluation script
- diacritizer/scripts/eval_e63s_wn2024.py — E63s evaluation script
- diacritizer/scripts/eval_mishkala_wn2024.py — Mishkala inference script
- diacritizer/scripts/eval_zmahood_wn2024.py — Zmahood inference script
- scripts/measure_benchmark_sizes.py — Benchmark size measurement script
