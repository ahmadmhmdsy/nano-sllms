# DeepSeek-V4.1-Flash — Extracted Research Notes

**Source:** `C:\Users\AhmadMhmoud\Downloads\Documents\DeepSeek_V41_Tech_Report.pdf` (51 pages)
+ `deepseek_v41_video_explain_subtitle.md` (video explainer transcript)
**Raw text extract:** `research/raw/deepseek_v41_report_extract.txt` (161k chars, page-marked)
**Extracted:** 2026-09, via uv ephemeral env + pypdf (no PDF lib in project venv)

---

## 1. Headline facts

| Item | Value |
|---|---|
| Model | DeepSeek-V4.1-Flash (multimodal MoE) |
| Backbone params | 552B total (55% smaller than V4-Pro's 1.6T, ~2x V4-Flash's 284B) |
| Activated params | **8B per token during prefill, 16B during decode** |
| Context | up to 1M tokens (trained at 64K from scratch, extended to 1M at 34T) |
| Pretraining | 45T tokens, multimodal, no instability |
| Layers | 40 = 20-layer causal **encoder** + 20-layer **decoder** (CED) |
| Hidden dim | 5120 |
| MoE | 1 shared + 384 routed experts, 6 activated, expert d_ff 2304, SwiGLU clamped at 10 |
| Global KV cache | **890 bytes/token** (V4-Flash: ~3,500; V1: ~390,000) → 4x and 437x reductions |
| Persistent KV (SSD/host) | ~1/8 of V4-Flash |
| Decode FLOPs | ~flat vs context length (4K→1M costs only +1/4) |
| Optimizers | head-wise Muon (matrices) + AdamW (norms/biases) + momentum+Sinkhorn (embeddings/head/Engram) |
| Post-training | no algorithmic novelty — SFT → RL → on-policy distillation; all gains from data/environment synthesis pipelines |

## 2. Architecture components (the actual innovations)

### 2.1 CED — Causal Encoder-Decoder (Sec 2.2)
- Bottom L/2 layers = **causal encoder**; top L/2 = **decoder**.
- Decoder **does not compute its own global KV**. Global KV for decoder layer l is *projected*
  from the encoder's final hidden state H_{L/2}:  C_l = H_{L/2}·W^KV_l,  Z_l = H_{L/2}·W^Z_l (layer-dependent projections).
- Prefill only runs the encoder half → prefill cost O(N·L/2) instead of O(N·L). Nearly halves prefill compute.
- Decoder still computes **layer-local SWA KV** from its own hidden states (local depth preserved).
- Inspired by YoCo (Sun et al., 2024).

### 2.2 CSA2 — Compressed Sparse Attention 2 (Sec 2.3)
Compresses KV along 3 multiplicative dimensions: entry size (GQA/MLA-style), sequence (m-token
compression), and **layer (cross-layer sharing)** — CSA2's contribution is the layer dimension.

Every layer keeps its **own** main Q and SWA KV. Three statically assigned modes:
| Mode | Main KV | Indexer K | Top-K indices |
|---|---|---|---|
| **Full** | computes own + indexer Q, fresh Top-K | projected from own main KV | fresh |
| **Reindex** | reuses from nearest Full layer | (comes with reused KV) | fresh — own indexer Q rescores shared indexer K |
| **Reuse** | reuses | reuses | reuses from latest index-producing layer (zero indexing compute) |

Simplifications vs CSA (V4): no overlapping source entries in the compressor, no absolute
positional embedding in compression, indexer K is a *projection of main KV* (not a separate path).
Config (V4.1): encoder layers use compression ratio m=2; decoder uses m=1 (uncompressed main KV);
indexer: 32 heads × 128 dim; top-k = 512 selected entries; main attention: 64 Q heads × 512 dim,
query compression dim 1280; SWA window n_win = 128.

### 2.3 Hierarchical Sparse Indexer (Sec 2.3.2, decoder only)
- The decoder's first Full-mode layer scores the whole context, picks its Top-K, and also does
  **blockwise candidate selection**: top 2,048 blocks × 8 positions = up to **16,384 candidate positions**.
- All later (Reindex) layers score **only inside that pool** → per-query indexer cost goes from
  O(context) to O(constant). Training-aware: restriction applied identically in training & inference.

### 2.4 Single-Pass mHC (Sec 2.4.1)
- mHC: n residual streams between blocks with token-wise mixing coefficients (A,B,C).
- Single-Pass shifts input-mixing coefficients by one block (block l uses A_{l-1}) so residual
  update + coefficient prediction fuse into **one kernel** (Mega-mHC). Activation memory traffic
  halved: (4n+4)d → (2n+2)d. Negligible quality loss.

### 2.5 Engram — conditional memory (Sec 2.4.2)
- 196B params in **host RAM**, not HBM. Multi-head hashing of n-grams (orders 2,3,4), 8 heads,
  ~16M-entry tables (distinct primes), FP8 tables, embedding dim 2048/order. Modules at layers 1 & 14.
- Stores static facts (memorization) so GPU/HBM is freed for reasoning. RDMA-prefetched, overlapped.
- Trained with momentum + Sinkhorn balancing instead of Adam (huge optimizer-state saving).

### 2.6 DSpark — speculative decoding (Sec 2.4.3)
- 3-block drafter, SWA window 128, drafts 5 positions in one pass + Markov head + confidence head;
  confidence-scheduled verification length per request. Trained *after* backbone pretraining,
  backbone frozen; during post-training trained without gradients into backbone.

### 2.7 FP4 Main KV cache (Sec 2.4.4)
- QAT during post-training; OCP MXFP4-style, E2M1 + one E4M3 scale per 16 channels (NVFP4-like,
  no second-level global scale — safe because cache magnitudes ≤ ~22.6 vs 2688 format max).
- Quantize **after RoPE**. SWA KV stays FP8 (sensitive to quantization). Nearly halves main KV storage.

### 2.8 SWA Bounded Replay (Sec 3.2.2) — the deployment trick
- SWA KV is **no longer persisted at all** (was ~half the persistent cache).
- On cache miss, replay only the last **n_win tokens** (128), not L×n_win — approximate state
  reconstruction, negligible quality loss (validated in training too: "train-aware adaptation").
- Turns "catastrophic miss" into "cheap recompute": 128 tokens of recompute << SSD round-trip.

## 3. Training setup (Sec 4.2)
- LR 2.6e-4, 2k warmup, flat until 28T, cosine to 2.6e-5 by 40T, flat to 45T. Batch 100.6M tokens.
- Sparse attention from scratch at 64K — **no dense warmup stage**.
- Muon: momentum 0.95, wd 0.1, update RMS rescaled to 0.18 (Adam-LR compatible). Head-wise Muon
  for Q/K weights (separate preconditioner per head; also adopted by GLM-5, Kimi-K3).
- AdamW for non-matrix params: β=(0.9,0.95), ε=1e-20, wd 0.1.
- Sinkhorn update for embeddings/head/Engram: K=11 normalization steps, γ=0.18 LR correction.
- Auxiliary-loss-free load balancing, per-modality (text vs image) correction biases.
- Vision encoder: 32-layer ViT, 2D-RoPE, linear patch embed (Muon-compatible), RMSNorm+SwiGLU;
  SigLIP contrastive on 47B pairs at 224², then autoregressive FT at 544²–1344²; pixel-unshuffle 3×3
  → 9x fewer visual tokens.

## 4. Post-training (Sec 5) — lessons
- "The marginal return of engineering the data and environment pipeline substantially exceeds that
  of algorithmic novelty in post-training."
- Tasks = (problem, environment, verification) triplets; model-assisted task synthesis with
  difficulty+correctness rewards; RL across heterogeneous scaffolds; **model merging to
  reinitialize successive RL runs** aggregates parallel RL compute.
- **Controllable reasoning effort**: scalar b∈[1,100] in system prompt; per-effort GRPO subgroups;
  exponential length penalty k(b)=k0·exp(-(b-b_min)/τ). API tiers: low=50, high=75, max=100.
  Effort 60–80 recovers most accuracy at <half the token budget.
- Async RL: sample-level dispatch, token-level interruption, KV/routing persisted at token
  granularity across checkpoint switches, staleness loss-masking.
- Final stage: full-vocab on-policy distillation from **40+ heterogeneous teachers**.
- Multi-agent (Agent Team mode) beats single-agent at every wall-clock deadline tested.

## 5. Base-model results (Table 1, base models)
vs V4-Pro-Base (1.6T, 49B act) and V4-Flash-Base (284B, 13B act), V4.1 (552B, 8B/16B act):
- MMLU-Pro 74.1 (best), HumanEval 79.4 (best), BigCodeBench 60.6 (best), GSM8K 93.0 (best),
  MATH 61.1 (vs 64.5 Pro), BBH 86.1, HellaSwag 87.2, LongBench-V2 45.2.
- 5–10% better on held-out BPB with 1/3 total params and 1/4 activated params of V4-Pro.
- Instructed model: DeepSWE 74.2 (beats Opus-5 74.0), Terminal-Bench 2.1 90.6, Codeforces 3471,
  GPQA-D 90.9. Gap remains on expert-science agentic tasks (Terminal-Bench 4.0: 31.2 vs 51.8).

## 6. Limitations (self-reported, Sec 6)
- CSA2 sparse-selection errors and approximate SWA replay may degrade untested boundary cases.
- Parity with frontier on most tasks ≠ parity on hardest reasoning/edge cases.

## 7. Video transcript — anything not in the paper
The transcript is a faithful popularization; only soft additions: the "junior analysts / senior
executives" analogy for CED; "student notes / desk / filing cabinets" analogy for HBM/SSD KV;
explicit note that DeepSeek is compute-starved and *optimizes software so hardware works less* —
the core design philosophy. All numbers in the transcript match the paper.
