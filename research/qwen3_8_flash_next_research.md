# Qwen3.8-Flash-Next ("queen3.8 flash next") — Research Notes

**Date:** 2026-09-05 · **Method:** project Exa helper (exa_search.py via the scripts/exa_research.py driver),
23 raw Exa payloads in research/raw/ (searches s1-s9, answers a1-a3, page fetches c1-c11), extracted page
texts in research/raw/txt/.
**Question:** what exactly is the model this project calls "queen3.8 flash next" (our nano-size target),
and **how do they train it** (and their small models)?

> Primary sources: QwenLM GitHub READMEs, the Alibaba Cloud blog, the Hugging Face model card, and the
> technical report arXiv:2608.30320 (fetched in full, 116k chars). Everything below is sourced; purely
> synthesized statements are marked "(Exa answer)".

---

## 0. TL;DR

- **"queen" = Qwen.** "queen3.8 flash next" = **Qwen3.8-Flash-Next**, released **2026-08-26** by Alibaba's
  Qwen team: a multimodal sparse-MoE model that is the **early architecture preview of Qwen4** — the same
  role Qwen3-Next played for Qwen3.5 in Sept 2025.
- Headline: **125B main params + 51B n-gram embedding table (held off the GPU) + 4B MTP**, only **6B
  activated per token**, 48 layers, hybrid **3 GDN : 1 QSA** token mixing, 256K native context (1M with
  YaRN).
- vs the previous 397B-A17B flagship: **best on 8/14 base benchmarks**, at most 2.6 pts behind on the
  rest, at **~1/3 activated params, ~1/3 training tokens, ~1/9 training FLOPs**.
- Training recipe (the part this project cares about): **Muon optimizer for 2-D linear-map weights +
  AdamW for embeddings/router/low-rank/gates**, fused matrices split before orthogonalization, **no
  batch-size warmup** (starts directly at target batch), **refitted scaling laws** (larger LR + batch),
  and a two-stage **continued pretraining for QSA** (indexer-only warm-up ~2B tokens, then joint sparse
  training ~200B tokens at 256K ctx). Stress tests: the full production run finished with **zero loss
  spikes**, without qk-clip or SwiGLU-clip.
- Small models are **not** trained the flagship way: Qwen trains small models by **strong-to-weak
  distillation** from flagships (off-policy trace SFT + on-policy logit alignment, ~1/10 the GPU-hours).
- For **nano_SLMs**: the nano-feasible subset is the **3:1 GDN/full-attention hybrid + gated residual +
  sigmoid gates + zero-centered RMSNorm** — and this repo's transformers 5.16.1 already ships
  qwen3_next, qwen3_5, qwen3_5_moe and qwen4_exp implementations to port from. QSA, the 51B n-gram
  table, ultra-sparse MoE and Muon are big-scale levers; the realistic path to a *useful* nano
  Flash-Next is **distillation**, exactly as Qwen does it.

---

## 1. Naming and lineage (what "queen3.8 flash next" actually is)

The project resources (resources/*.md) spell it "queen model 3.8" / "Qwen3.8-Flash-Next"; web sources
confirm the real name and timeline:

| Date | Release | What it is |
|---|---|---|
| 2025-09-11 | **Qwen3-Next-80B-A3B** | First hybrid **GDN + Gated Attention** (3:1) ultra-sparse MoE; architecture preview for Qwen3.5 |
| 2026-02-16 | **Qwen3.5-397B-A17B** | Production validation of that hybrid; natively multimodal (early fusion), 201 languages, million-agent RL |
| 2026-02 → 03 | Qwen3.5-27B / 35B-A3B / 122B-A10B / 9B / 4B / 2B / 0.8B | Family rollout incl. small dense models |
| 2026-04-16/22 | Qwen3.6-35B-A3B / 27B | Stability + agentic-coding update; thinking preservation |
| 2026-08-02/12 | **Qwen3.8-Max (2.4T-A95B)** | First open-source Qwen-Max-class weights |
| 2026-08-14 | Qwen3.8-27B | Dense 27B of the 3.8 family |
| **2026-08-26** | **Qwen3.8-Flash-Next** | **Preview of the Qwen4 architecture** (GDN + QSA, GR, n-gram embedding, Muon recipe) |
| future | Qwen4 | Full family to be built on this architecture |

Sources: GitHub QwenLM/Qwen3.8 repo news [S9], GitHub QwenLM/Qwen3.8-Flash-Next [S1], Alibaba Cloud blog
[S2], HF model card [S3], mlabonne Qwen3.5 review [S7], arXiv:2608.30320 [S4].

Qwen3.8-Flash-Next is post-trained and served as **Qwen3.8-Flash** on QwenCloud (1M ctx default; priced
$0.16/M input and $0.47/M output tokens). The GitHub repo is Apache-2.0.

---

## 2. Model card (Qwen3.8-Flash-Next)

From the HF model card [S3]:

| Property | Value |
|---|---|
| Type | Causal LM **with vision encoder** (multimodal) |
| Training stage | Pre-training **and** Post-training |
| Parameters | **125B** total, **6B** activated/token |
| Extra capacity | **51B n-gram embedding** (20M bigram/trigram table, host-RAM offloaded) + **4B MTP** |
| Hidden dim | 2560 |
| Vocab | 248,320 (padded) |
| Layers | 48, laid out as 12 x ( 3 x (GDN -> MoE) -> 1 x (QSA -> MoE) ) |
| GDN heads | 48 V-heads, 16 QK-heads, head dim 128 |
| QSA | 24 Q / 2 KV heads, head dim 256, partial RoPE 64 dims; indexer = MQA (4 query heads + 1 shared key head), indexer head dim 128; budget 512 blocks = 2048 tokens, block size r=4 |
| MoE | 512 experts, 10 routed + 1 shared, expert intermediate dim 640 |
| Gated Residual | 4 branches, bottleneck rank 320 |
| MTP | 1 layer, trained multi-step (indices reused across speculative steps) |
| Context | 262,144 native, extensible to 1,000,000 (YaRN) |

Post-trained model scores (HF card, selection): SWE-bench Pro **62.5**, SWE-bench Multilingual **81.0**,
DeepSWE 1.1 **58.7**, IFBench **81.3**, GPQA Diamond **91.7**, LiveCodeBench v6 **91.9**, AndroidWorld
**84.5**.

---

## 3. Architecture — the four upgrades over Qwen3.5

The upgrade axes are **attention, residual, embedding, optimization** [S1][S2][S4].

### 3.1 Attention: GDN + QSA hybrid (3:1)

- Three of every four layers use **Gated DeltaNet (GDN)** — a linear-attention/recurrent layer that
  compresses history into a fixed-size state; one in four layers keeps **global attention** for precise
  token-level retrieval (periodic full attention matters most for long-context quality).
- **GDN gated delta rule** (paper §2.1.1), per head, with data-dependent decay alpha_t and write gate
  beta_t:

  ```
  S~_t = alpha_t * S_{t-1}
  e_t  = v_t - S~_t^T k_t
  S_t  = S~_t + beta_t * k_t e_t^T      (targeted erase-and-write, NOT additive)
  y_t  = S_t^T q_t
  ```

  q,k = L2Norm(SiLU(ShortConv(W x))) (short depthwise causal conv = local inductive bias);
  v = SiLU(ShortConv(W x)); beta_t = sigmoid(W_beta x);
  alpha_t = exp(-exp(A) * softplus(W_alpha x + b_alpha));
  output uses a **sigmoid** gate: o = W_o[ sigmoid(W_z x) ⊙ RMSNorm(y) ] — sigmoid beats the original
  GDN's SiLU gate. All RMSNorms are **zero-centered** (weight decay on norm weights). RoPE lives only in
  the full-attention layers; a NoPE variant looked identical in pretraining but "endlessly generates"
  after post-training.
- **QSA (Qwen Sparse Attention)** replaces the full-attention layers *at continued-pretraining time*: a
  lightweight **MQA indexer** (4 query heads + 1 shared key head) average-pools keys into micro-blocks
  (r=4), applies partial RoPE at block level, scores blocks with ReLU-activated similarities, and Top-K
  selects blocks (budget K=2048 tokens = 512 blocks + always the tail block); core attention then runs
  only over selected tokens. Compressing before indexing cuts indexer cost from O(n^2) to O(n^2/r).
- Architecture ablation (28-layer 25B-A3B probes, 400B+80B tokens): GDN hybrid > SWA hybrid > full
  attention (9-benchmark avg 53.81 / 51.15 / 49.87).
- QSA vs full attention: matches or beats it on 7/8 short-ctx benchmarks (avg 75.9 -> 76.8) and **wins at
  long ctx** (RULER >512K: 90.08 -> 93.00; MRCR 512K: 30.66 -> 40.53; 1M: 20.71 -> 26.44). Kernel-level
  at 1M tokens: **7.6x faster prefill, 4.9x faster decode**; 8.6x prefill throughput vs Qwen3.7-Plus at
  1M with 90% prefix-cache hits [S2][S4].

### 3.2 Residual: Gated Residual (GR)

- Residual stream widened to **4 branches** (Hyper-Connections idea) **with the branch-mixing operator
  dropped**; reads go through an elementwise data-dependent **sigmoid gate** (GatedNorm:
  RMSNorm(u) ⊙ sigmoid(W2 SiLU(W1 RMSNorm(u))), low-rank bottleneck r = d/8), writes through a
  per-branch scalar gate s = 2*sigmoid(...). GR **replaces the block's pre-normalization** (no extra
  norm layer added).
- Ablation (25B-A3B, 560B tokens): avg accuracy pre-norm 50.91 -> mHC-dynamic 54.47 -> **GR 54.66**;
  loss 1.617 -> 1.590.
- Mechanism (exact path decomposition, Eq. 35-37): **one branch becomes a long-range highway** (typical
  skip ~10.9 layers; e.g. layer-0 GDN -> layer-15 attention share rises 0.020 -> 0.138) while the other
  three stay local (median 1.2-3.5 skips); the readers of long-range paths are predominantly the
  **full-attention layers** — attention acts as the hub that integrates what GDN compressed away.
- Efficiency: **FP8 residual storage** halves residual-state traffic (gates bound activation ranges);
  fused read/write kernels traverse the widened stream once per direction.

### 3.3 Embedding: N-gram Embedding

- A single **n-gram embedding layer at layer 2**: multi-head-hashing lookup keyed by bigrams/trigrams of
  the local context, injected through contextual gating; **51B params with ~zero extra per-token FLOPs**
  (deterministic addressing -> host-memory offload + asynchronous prefetch overlapping compute).
- Ablations: placement is largely insensitive (shallow slightly best; **a single layer suffices**);
  **scaling the n-gram vocab lowers loss monotonically while downstream accuracy saturates** — the
  paper's cautionary example of loss-vs-benchmark disagreement.

### 3.4 Optimization: Muon (+ AdamW), refitted scaling laws

Covered in §4 — this *is* the training recipe.

---

## 4. How they train it (the core question)

### 4.1 Pretraining + continued pretraining (CPT)

- Scale claim: matches the 397B-A17B flagship class with **~1/3 the activated parameters, ~1/3 the
  training tokens, ~1/9 the training FLOPs** [S4 §1, §4].
- **Two-stage QSA CPT at 256K context** (paper §2.1.2 "Training Details"):
  - **Stage 1 — dense distillation (indexer warm-up):** the full-attention backbone's attention
    distribution (softmax summed over all heads, L1-normalized, **max-pooled** onto blocks) is distilled
    into the indexer with a KL loss. Indexer only: **1,000 steps x 8 seq x 256K tokens ≈ 2B tokens**,
    LR 1e-3.
  - **Stage 2 — joint sparse training:** backbone + indexer train together under sparse masks (KL kept
    only on selected blocks; teacher renormalized over them): **8,000 steps x 96 seq x 256K ≈ 200B
    tokens**, LR 2.5e-5. LM-loss stays within ~1e-4 of the full-attention baseline (Fig. 4).
- All full-attention layers in the backbone **and in the MTP module** become QSA; MTP reuses top-k
  indices across speculative decoding steps.

### 4.2 Optimizer: Muon for linear maps, AdamW for the rest (paper §3.1)

- **Muon** (Newton-Schulz orthogonalization of Nesterov momentum, mu=0.95; update scale
  0.2*sqrt(max(A,B)); **Polar Express** per-step coefficient schedule; **8 NS steps** for extra
  orthogonalization accuracy) is applied to weights that genuinely act as 2-D linear maps:
  attention q/k/v/o, GDN input/output projections, MoE expert fc1/fc2 (routed + shared), n-gram K/V.
- **AdamW keeps:** input embeddings, output head, **MoE router** (Muon destabilizes it early; no gain
  even if applied mid/late), GR's low-rank projections (too elongated), GDN decay/beta vectors
  (scalar-per-head -> orthogonalization meaningless), attention/GDN output gates (AdamW on par or
  better). The n-gram table runs on **Adam without weight decay**.
- **Fused matrices are split before orthogonalization** (qkv and GDN-in at per-head granularity; SwiGLU
  fc1 into gate/up halves): orthogonalizing a concatenated matrix mixes singular directions across
  unrelated sub-blocks and skews the RMS scaling.
- Engineering: **Canzona** decouples logical optimizer assignment from Megatron sharding (alpha-balanced
  static partitioner equalizing NS FLOPs per DP rank; async Micro-Group All-to-All reconstruction), and
  the whole Muon step is captured in a **CUDA graph** (splitting creates ~100 tiny kernels per layer).

### 4.3 Refitted scaling laws; batch-size warmup is dead (paper §3.2)

- The new architecture + Muon shift optima to **substantially larger batch and learning rate** (with a
  slower LR decay as scale grows). Predictions verified separately in each sensitive regime:
  - Batch, on a 20-layer 10.8B-A0.89B MoE over 4T tokens: previous B=12.6M vs predicted **B=25.2M**:
    final loss 1.5774 -> **1.5702**; 1.5x beyond the optimum is flat.
  - LR, on a 48-layer 156B-A7B MoE over 419B tokens: previous recipe eta=6.8e-4 vs predicted
    **eta=1.76e-3, B=8.4M**: -7.8e-3 loss and avg benchmark 56.41 -> **60.55**; the optimum sits in a
    flat bowl (LR divided by sqrt(2) is noise-level); gradient clipping never engages after warmup (max
    pre-clip norm 28% of threshold vs 51% under the old recipe).
- **Batch-size warmup (ramping) is no longer used**: both ramp variants end slightly *worse* and cost
  **+18.8% optimizer steps** for the same token budget. Production starts **directly at the target
  batch**.

### 4.4 Training stability (paper §3.3) — the GR/gate story

- **Stress tests** reproduce production-scale instability at moderate scale by training a 28-layer
  25B-A3B MoE at **constant 2x / 4x its optimal LR**:
  - 2x: AdamW + old structure 4.3 spikes/10k steps; both Muon variants 0.2.
  - 4x: AdamW 183 spikes/10k steps and 213/19,932 threshold crossings; **both Muon runs never cross the
    clip threshold; Muon + GR records zero loss spikes**.
- Isolating the gate at 3x LR (optimizer and structure fixed): enabling GatedNorm cuts spikes 32.0 ->
  3.2 per 10k steps and threshold crossings 256 -> 20. Interpretation: high-LR training *needs* a
  rescaling mechanism; without an explicit gate the network grows activation outliers and stays fragile.
- **Production verification:** over the first 276B tokens, GR lowers loss by 0.026 vs
  Qwen3.5-structure+Muon, and the full Flash-Next recipe by 0.058 total; Muon alone shows ~2x median and
  4.2x p99.9 gradient norm vs the gated runs (0.298 vs 0.071 / 0.066). Result: the full-scale run
  finished **without a single loss spike or anomalous gradient-norm fluctuation — no qk-clip, no
  SwiGLU-clip needed**.

### 4.5 Results (base model, Table 11 [S4])

Best on 8/14 vs Qwen3.7-Plus-Base (397B-A17B) and on all 14 vs Qwen3.8-27B-Base:
MMLU 90.36 · MMLU-Redux 90.68 · MMLU-Pro **73.23** · SuperGPQA **51.36** · BBH **90.87** · GPQA 51.42 ·
GSM8K **93.29** · MATH 72.78 · EvalPlus **78.76** · MultiPL-E 79.09 · SWEBench-Pretrain **50.99** ·
MGSM **89.33** · MMMLU **84.86** · INCLUDE 78.40.

---

## 5. How Qwen trains the *family* — and especially the small models

### 5.1 Qwen3 report (arXiv:2505.09388) — the reference pipeline (Exa answer, cross-checked vs report snippets)

- **Pretraining:** 3 stages on **36T tokens**, 119 languages — S1: 30T @ 4K ctx (general knowledge) ->
  S2: 5T @ 4K (STEM/code/reasoning upweighted) -> S3: long-context stage (32K) on curated long data.
- **Post-training:** 4 stages — (1) **long-CoT cold-start SFT**, (2) **reasoning RL** (GRPO-family with
  verifiable rewards), (3) **thinking-mode fusion SFT** (unifies thinking / non-thinking; the /think
  flags), (4) **general RL** (preference alignment). Thinking-budget control falls out of stage 3.
- **Small models (0.6B-14B, 30B-A3B) are distilled, not RL-trained:** **strong-to-weak distillation**
  from the flagships (32B / 235B-A22B): off-policy teacher-response SFT + **on-policy logit alignment**,
  costing **~1/10 the GPU-hours** of the full 4-stage pipeline.

### 5.2 Qwen3.5 / Qwen3.8 family training notes

- Qwen3.5: **early-fusion multimodal pretraining on trillions of tokens** ("near-100% training
  efficiency vs text-only"), **RL scaled to million-agent environments** with progressively complex task
  distributions, asynchronous RL infrastructure, 201 languages [S9][S7].
- Qwen3.8: flagship = 2.4T-A95B (open weights 2026-08-12); thinking depth controlled via
  reasoning_effort (xhigh/medium/low); thinking context retained across turns via preserve_thinking [S9].
- Small Qwen3.8-scale students in the wild (**third-party empero-ai org, not Qwen's own**):
  full-parameter distillation of Qwen3.8-2.4T-A95B into the **Qwen3.5-2B/4B/9B architectures** on
  ~30k/45k/70k curated dense-CoT teacher traces [S10] — the same strong-to-weak pattern, now with
  Qwen3.8 teachers.

### 5.3 What Qwen does NOT do for small models

No sparse-attention CPT, no n-gram tables, no MoE at small scale — small Qwens are **dense 3:1
GDN:gated-attention hybrids** that inherit capability through **distillation** (plus the Qwen3-era
curated SFT+RL data). The architecture trick (GDN hybrid) makes small models *cheap to run*; the
training trick (distill from big teachers) is what makes them *good*.

---

## 6. What this means for nano_SLMs ("queen3.8 flash next in nano size")

Grounded in this repo: S=12M / P=100.7M (done, eval ppl 3.19) / T=226.5M (M3 running), 6 GB GPU, fp16,
SDPA, HF Trainer, 32k tokenizer, ctx 512-1024; PLAN.md A7 already defers the "Flash-Next hybrid
(GDN/QSA)" to a later upgrade — this section is the evidence-backed version of that plan.

### 6.1 Nano-feasible parts (the real "Flash-Next-Mini" checklist)

1. **3:1 GDN : full-attention hybrid** — the most transferable idea. The paper's own evidence comes from
   small probes (28-layer 25B-A3B), so the pattern is validated well below flagship scale. Benefits for
   us: only every 4th layer stores a KV cache -> longer ctx on 6 GB VRAM and cheaper decode.
   **Reference implementation already installed:** transformers 5.16.1 ships models/qwen3_next (with
   Qwen3NextGatedDeltaNet), plus qwen3_5, qwen3_5_moe and qwen4_exp — port or instantiate rather than
   writing GDN from scratch. Pure-PyTorch/SDPA is fine at ctx <= 2-4K; recurrent/chunked GDN kernels
   (FlashQLA-class) only matter for long context.
2. **Gated Residual (GR)** — cheap, and the paper's *stability* lever (zero spikes at 4x LR with Muon;
   GatedNorm alone cuts spike rate ~10x). Even without Muon, the gated read (RMSNorm ⊙ sigmoid(low-rank
   gate), 4 branches, per-branch scalar write) is ~50 lines of PyTorch and a principled answer to any
   fp16 instability at nano scale.
3. **Sigmoid gates everywhere** (GDN output gate, gated attention output gate, GR) + **zero-centered
   RMSNorm** + RoPE only on full-attention layers — small, proven stability details.
4. **Constant batch from step 0** — already our default (batch 1 x accum 32); the paper confirms ramping
   buys nothing (+18.8% optimizer steps, no gain).
5. **Higher LR than intuition** — with gates (+ optionally Muon) the optimum LR shifts up. Our 4e-4 at
   100-250M with AdamW is sane, but a small LR sweep (e.g. 4e-4 vs 7e-4) is now evidence-backed.

### 6.2 Skip at nano (with reasons)

- **QSA**: pointless below ~128K ctx — its win is long-context prefill/indexer cost; at 512-1024 tokens
  the 1-in-4 full-attention layer is already negligible. It also needs the 2-stage CPT distillation
  machinery to train.
- **51B n-gram table**: an off-GPU capacity play for 100B+ models. A *tiny* bigram/trigram table (a few
  M params, layer-2 placement, hashed lookup) is a fun low-priority experiment — the paper's ablations
  say a single early layer is enough — but it will not move a 250M model much.
- **Ultra-sparse MoE** (512 experts, 10+1 routed): at 100-250M dense beats sparse; MoE adds
  optimizer-state and routing complexity we cannot afford on 6 GB.
- **MTP**: only pays off with speculative-decoding serving; irrelevant to our offline eval loop.
- **Muon (optional experiment, not a must)**: the paper's gains are most visible at large scale (data
  efficiency at big batch, stability margin). If tried: Muon only on 2-D linear-map weights; AdamW on
  embeddings/head/scalars/low-rank; split fused qkv/fc1 before orthogonalization; ~8 NS steps.

### 6.3 The realistic path to a *useful* nano Flash-Next: distillation

Qwen's own answer for small models is **strong-to-weak distillation**, not more pretraining tokens:
teacher traces (dense CoT) -> SFT, plus on-policy logit alignment. For this project that translates to:
pretrain the nano hybrid on code (as now), then **SFT on traces generated/filtered by an open flagship**
(e.g. a Qwen3.5/Qwen3.8 instruct model via API, or a local mid-size open model) — that is what would
turn the current "code completer" into something instruction-capable, at ~1/10 the cost of any RL
stage (a2/a3 payloads; Qwen3 report; DistilQwen papers).

### 6.4 Corrections to resources/ claims vs the real paper

| Resource claim | Reality (paper/blog) |
|---|---|
| "QSA-lite = standard attention with a **fixed block-sparse mask**" | Wrong. QSA is a **learned MQA indexer** (4+1 heads, avg-pooled micro-blocks, Top-K) distilled from the full-attention teacher in a 2-stage CPT. |
| "GDN-lite: state = state*0.9 + (k*v).mean*0.1 placeholder" | Not the delta rule. Real GDN: data-dependent decay alpha_t, write gate beta_t, rank-one **erase-and-write** update (Eqs. in §3.1); q/k short-conv + L2-norm, sigmoid output gate, zero-centered RMSNorm. |
| "Muon optimizer for 2-D weights + AdamW for the rest" | **Correct** — and now much more specific (split fused matrices, Polar Express schedule, 8 NS steps; router/gates/low-rank stay on AdamW). |
| "Gated Residual = 4-branch residual with dynamic gates" | Correct (GR = Hyper-Connections width + GatedNorm read, minus branch mixing). |
| "N-gram embedding optional but powerful" | Correct at their scale (51B params, layer-2, host-offloaded); at nano it is a marginal experiment. |
| 250M recipe: 24L / d768, GDN:GA ≈ 3:1, dense | Reasonable and consistent with the paper's hybrid probes; plain-GQA v0 first (PLAN A7) remains the right order. |

---

## 7. Sources

Primary (fetched via Exa contents; full texts archived under research/raw/txt/):

- [S1] QwenLM/**Qwen3.8-Flash-Next** GitHub README — https://github.com/qwenlm/qwen3.8-flash-next (c1)
- [S2] Alibaba Cloud blog: "Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency" (2026-08-27) — https://www.alibabacloud.com/blog/qwen3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501 (c2)
- [S3] HF model card **Qwen/Qwen3.8-Flash-Next** (full config) — https://huggingface.co/Qwen/Qwen3.8-Flash-Next (c3)
- [S4] Technical report: "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability", Qwen Team, Aug 2026 — https://arxiv.org/html/2608.30320 (c4/c9/c11; 116k chars, full §1-§5)
- [S5] Qwen blog: "Qwen3.8-Max: A New Bar for Coding and Cowork" (2026-08-02) — https://qwen.ai/blog?id=qwen3.8 (c5; JS-rendered, headline facts via search highlights)
- [S6] IntuitionLabs: "Qwen3.8-Flash-Next: Architecture, Memory & Inference Guide" (2026-09-05) — https://intuitionlabs.ai/articles/qwen3-8-flash-next-architecture-memory (c6)
- [S7] mlabonne: "Qwen3.5: Nobody Agrees on Attention Anymore" (2026-02-17) — https://huggingface.co/blog/mlabonne/qwen35 (c7)
- [S8] Reuters: "Alibaba's Qwen launches Qwen3.8-Flash AI model with lower training costs" (2026-08-26) — in s1 payload
- [S9] QwenLM/**Qwen3.8** GitHub README (series news 2025-09 -> 2026-08) — https://github.com/QwenLM/Qwen3.8 (c8)
- [S10] empero-ai/Qwen3.8-{2B,4B,9B}-Distill model cards (third-party) — https://huggingface.co/empero-ai/Qwen3.8-4B-Distill (s7)
- [S11] Qwen3 Technical Report — https://arxiv.org/abs/2505.09388 (via s6/s8 highlights + a3 answer)
- [S12] HF transformers docs: Qwen3.5 (3:1 GDN hybrid, native multimodal) — https://huggingface.co/docs/transformers/en/model_doc/qwen3_5 (s9)
- [S13] vLLM Qwen3-Next support blog (2025-09-11) — https://vllm.ai/blog/2025-09-11-qwen3-next (s3)
- [S14] HF transformers qwen3_next modeling source — https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py (s3)

Exa-answer payloads (LLM-synthesized, citations inside each file): research/raw/a1_what_is_qwen38_flash_next.json,
a2_qwen_small_model_training.json, a3_qwen3_report_training.json.

## 8. Raw files & how to re-run

- Raw Exa payloads: research/raw/ (s1-s9 searches+answers, c1-c11 page fetches) and extracted page
  texts in research/raw/txt/.
- Driver: scripts/exa_research.py — wraps exa_search.py; batch task lists, --tasks <file.json>,
  --extract (dumps saved contents payloads to research/raw/txt/).
- Round task files kept for reproducibility: research/round2.json, round3.json, round4.json, round5.json
  (the script's built-in default task list is round 1). Example:
  & .\.venv\Scripts\python.exe scripts\exa_research.py --tasks research\round4.json --extract
- Note: qwen.ai pages are client-side rendered — Exa captures only the JS bootstrap for them (c5, c10);
  use the Alibaba Cloud mirror blog [S2] for blog content.

