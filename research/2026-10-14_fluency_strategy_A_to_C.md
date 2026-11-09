# Fluency strategy: approved A -> C (2026-10-14)

## The decision (user-approved)

**A -> C.** Push data quality + coverage at the current 25.7M capacity to build the instruments and the
post-training recipe, then transplant that recipe onto a stronger borrowed base for actual fluency.
Capacity scale-up (path B, from scratch at 100-200M) is explicitly deferred - not chosen.

Rationale, measured: capacity is the binding ceiling (E-121 arm B: 1/32, 19/32 invalid Python, BPB 5.29% worse at
~25.7M total / 9.8M active), imitation alone is not sufficient (E-122: 6x exposure -> legacy 32/32 but grade still
0/28), and RL cannot create capability (E-122c: 4 RFT rounds, zero positives for grade/count_even/count_positive/
lookup, aspect pool unchanged at 32/48).

Literature alignment: "New Skills or Sharper Primitives?" (arXiv 2602.08281) supplies the vocabulary that fits our
data exactly - the null set (epsilon-incapability) vs the feasible set (delta-capability), emergence through
probability amplification, the MULTIPLICATIVE BARRIER in complex reasoning, and capability EROSION under RLVR
optimization (our count_positive 14 -> 3, and the aspects-only RFT's 9/32 -> 5/32). NeurIPS 2025 ("Does RL Really
Incentivize Reasoning Capacity Beyond the Base Model?") argues RL reshapes rather than expands the base.
Chinchilla: compute-optimal is ~20 tokens/param, so 25.7M params wants ~514M tokens and E-117 saw 163.8M (~32%).
phi-1 (Textbooks Are All You Need) and SmolLM2 (data-centric, 4-stage schedule with a DECAY phase, Stack-Edu code
mixture) both say curation beats raw volume for code.

## Approved cheap high-leverage moves, in order

1. Finish the coverage measurement (Phase 1.3 rung/aspect/legacy batteries).
2. A quality filter + dedup pass on the pretrain corpus, measured in BPB - CPU-only, no GPU decision needed.
3. Adopt an explicit DECAY PHASE (SmolLM2's last-stage idea) instead of the ad-hoc reference SFT.
4. Broaden the battery: 32-108 tasks over four constructs cannot certify fluency. Add a breadth battery across
   many constructs plus held-out HumanEval-style tasks.
5. Only then revisit capacity.

## What this direction does NOT settle (open items, raised 2026-10-14)

- **Tokenizer - analysed 2026-10-14 (user question: which is cheaper in compute?).** The user's reasoning is
  correct and dominant at this size: a 40k BPE vocab at dim 192 costs 7.68M embedding params = ~30% of a 25.7M model,
  which is a bad trade for a nano model. Byte-level (vocab 260) costs 49,920 params = 0.2%.

  | Aspect | byte (260) | BPE 40k | small BPE (2-8k) |
  |---|---|---|---|
  | embedding params @ dim 192 | 49,920 (0.2%) | 7.68M (30%) | 0.38-1.5M (1.5-6%) |
  | params left for the transformer | 25.65M | 18.0M | 24.2-25.3M |
  | bytes per token | 1.0 | ~3.5 | ~2.5-3.0 |
  | FLOPs per byte of text | 1.0x | 0.29x | ~0.37x |
  | 1024-token window covers | 1 KB | 3.5 KB | ~2.7 KB |
  | steps to emit a 48-byte body | 48 | ~14 | ~18 |
  | indent (4 spaces) | 4 dependent tokens | ~1-2 | ~2-3 |
  | vocab softmax per token | 260 | 40k (154x) | 2-8k (8-30x) |
  | tokenizer artifacts | none | vocab+merges | vocab+merges |
  | already built and corpus packed | yes | no | no |

  Verdict: byte-level wins on parameters, softmax cost, implementation simplicity and readiness; BPE wins on
  sequence length / attention FLOPs (3.5x more text per step) and generation steps. For a FIXED compute budget the
  cheapest-in-all-aspects option is a SMALL-vocab BPE (2k-8k), which captures most of the sequence saving at a
  fraction of the embedding cost. Recommendation: keep byte-level for path A (already built, param argument sound),
  treat a small-vocab BPE as a cheap future experiment (train tokenizer, re-pack, one 600-step run), and note it is
  moot for path C because SmolLM2-135M is BPE. Byte-level's measured cost is the indent barrier above.

## Corpus sources - MEASURED access (2026-10-14, account ahmadsyy)

GATED (real data files return GatedRepoError 403; README/metadata/listings are readable WITHOUT a grant, which is
what fooled an earlier probe): bigcode/starcoderdata, bigcode/the-stack-v2, the-stack-v2-dedup, the-stack-dedup,
the-stack-smol, nampdn-ai/tiny-codes. USER ACTION to unlock the ~250B-token inline source: click "Access repository"
(accept terms) at https://huggingface.co/datasets/bigcode/starcoderdata and
https://huggingface.co/datasets/bigcode/the-stack-v2 - grants are automatic once accepted.

OBTAINABLE NOW (ungated; byte-range 206 verified on the starred ones):
  1. codeparrot/codeparrot-clean * - Python-only, whole-file, exact-dedup + quality filtered, ~50GB uncompressed /
     ~13.8GB gz, content inline. PRIMARY.
  2. code-search-net/code_search_net (python config, 1.56GB; already the M3 source) + Nan-Do/code-search-net-python
     (599MB) + bigcode/python-stack-v1-functions-filtered-sc2 * (248,934 functions / 138MB).
  3. HuggingFaceTB/stack-edu python * - 2025 educational-score filtered (best quality signal) but content is
     METADATA-ONLY (blob_id) and needs a Software Heritage S3 fetch pipeline.
  Also ungated: codeparrot/github-code(-clean), HuggingFaceTB/smollm-corpus python-edu (S3), OpenCoder, SWE-smith,
  codeforces-cots, OpenCodeInstruct.

EXACT BUILD COMMAND for data/reference/bytes_pg_replay (Phase 1.2b corpus - the data survey found "pg_replay" recorded
nowhere else in the repo, so it is preserved here):
  & .\.venv\Scripts\python.exe scripts\reference_sft_bytes.py --out-dir data\reference\bytes_pg_replay \
      --include-playground --playground-tier all --instances 400 --playground-instances 100 --force
  (train.bin 3,333,114 B; playground 2,700 train examples / 135 val vs 8,000 replay examples)

DECISION: build the first real-Python corpus from codeparrot-clean + the function corpora now; request the bigcode
grants in parallel.

## The number that reframes everything (2026-10-14)

For a BYTE model, 1 token = 1 byte, so tokens == bytes. Chinchilla-style compute-optimal for 25.7M params is
~20 tokens/param = ~514M tokens = **~0.5 GB of Python text**. We trained E-117 on 163.8M tokens (~32% of optimal)
and every corpus since has been synthetic and SMALLER than that. So the compute-optimal data volume for this model
is half a gigabyte - trivially obtainable (codeparrot-clean alone is ~13.8GB gz). The binding constraint was never
data availability; it was that we were training on curated synthetic corpora roughly 3x below our own optimum.
- **No real Python corpus.** Our corpora (FAMILIES, ASPECTS, PLAYGROUND) are synthetic; the bigcode datasets are
  gated. Path C needs a real fluency corpus with provenance and a licence plan.
- **Decontamination** for any real corpus (train/test overlap with the batteries).
- **The C target**: which base, and when. User ruled SmolLM2 135M/360M first; Qwen3.5-0.8B later, weights not fetched.
- **Compute plan for C**: fine-tuning a 360M base on an 8 GiB Turing GPU needs a LoRA-vs-full decision.
- **Aspect-pool budget**: it still runs at 48 bytes and may understate multi-statement constructs.
