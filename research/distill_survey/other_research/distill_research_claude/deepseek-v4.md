# DeepSeek-V4 (Pro / Flash) — architecture + training technique

Sources:
- DeepSeek-V4 technical report, arXiv:2606.19348 ("Towards Highly Efficient Million-Token Context Intelligence")
- kili-technology.com blog, "Data Story: A Deep Dive into DeepSeek V4"
- huggingface.co/blog/deepseekv4
- Tom's Hardware / VentureBeat / Medium coverage of the Engram paper (precursor research, folded into V4)
- github.com/deepseek-ai/Engram

## Model specs
- Two variants: **V4-Pro** (1.6T total params, 49B active/token) and **V4-Flash** (284B total, 13B active/token).
- Both support **1,000,000-token context**.
- Pretrained on 32T tokens (Flash) / 33T tokens (Pro) — more than double V3's 14.8T.
- Corpus emphasizes math, code, web pages, long documents, scientific papers/technical reports — deliberately chosen to be useful at million-token context lengths (you need naturally long, structured documents to teach long-range dependencies, not just concatenated short ones).
- Notable practice: DeepSeek trained **separate domain specialists first, then merged them** — rather than training one generalist model on the full mixture from token 0.

## Architecture innovations (why it can do 1M context affordably)

1. **Hybrid attention: CSA + HCA**
   - **Compressed Sparse Attention (CSA)**: compresses every *m* tokens of the KV cache into a single consolidated entry (via softmax-gated pooling with a learned positional bias), then applies **sparse top-k selection** over those compressed entries using a cheap "lightning indexer" (FP4, ReLU-scored multi-head dot product) to pick which compressed blocks actually matter.
   - **Heavily Compressed Attention (HCA)**: uses a *much larger* compression ratio (m' >> m) but keeps attention **dense** over the resulting (far fewer) entries.
   - Layers alternate between CSA and HCA blocks. Net effect at 1M context: V4-Pro needs only ~27% of the per-token inference FLOPs and ~10% of the KV cache of V3.2. Effective KV cache footprint quoted around 2% via combination of compression + FP8 storage (BF16 only for RoPE dimensions) + FP4 indexer.
   - This directly builds on **DeepSeek Sparse Attention (DSA)**, from DeepSeek-V3.2, and on **Multi-head Latent Attention (MLA)** from DeepSeek-V2 (projecting K/V into a low-rank latent space before caching) — V4 is the third generation of the same lineage: MLA (low-rank projection) → DSA (sparse selection) → CSA/HCA (compress-then-sparse-select, at two different compression granularities).

2. **Manifold-Constrained Hyper-Connections (mHC)** — replaces the standard residual connection with a *learned mapping* that constrains hidden-state trajectories to a low-dimensional manifold. Purpose: better gradient flow, allows deeper networks without degradation, improves training stability.

3. **Muon optimizer** (instead of / alongside AdamW) — orthogonalization-based optimizer, reported to give faster convergence and improved training stability versus AdamW alone. (Also independently adopted by Qwen for Qwen3.8-Flash-Next — see sibling doc. Convergent evidence Muon is becoming the frontier-lab standard for large MoE pretraining.)

4. **Retains from V3**: DeepSeekMoE framework (fine-grained expert segmentation) and Multi-Token Prediction (MTP) — predicting more than one future token per forward pass as an auxiliary training objective, which densifies the training signal per step.

## Infrastructure / systems-level training tricks (relevant if you ever scale up)
- A single **fused kernel** for the MoE modules that overlaps computation, communication, and memory access simultaneously (avoids idle GPU time waiting on all-to-all communication).
- **TileLang** — a DSL used to balance developer productivity against raw kernel efficiency (write kernels faster without losing much speed).
- **Batch-invariant, deterministic kernels** — ensures bitwise-reproducible training/inference (huge for debugging and for verifying distillation experiments give identical outputs given identical inputs).
- Extended the **autograd framework with tensor-level checkpointing** for fine-grained activation recomputation — lets them trade compute for memory precisely at the tensor level rather than whole-layer level, which matters a lot when VRAM is the binding constraint (directly relevant to your VRAM-limited setup — recomputing activations instead of storing them is the standard trick to fit bigger batches/contexts in less memory).
- Agent-training-specific infra ("DSec" in HF blog coverage): layered 3FS storage for fast rollout container/image loading, preemption-safe trajectory replay (so an interrupted RL rollout doesn't have to re-run tool calls from scratch), and a uniform API across execution substrates (function-call sandboxes vs full VMs) so the training harness doesn't need to be rewritten per environment type.

## Engram — DeepSeek's separate "conditional memory" research (the likely inspiration/precursor for parts of V4)
(Published as standalone paper ~Jan 12, 2026, co-authored by DeepSeek founder Liang Wenfeng; widely expected to and reported to have fed into V4's design.)

- **Core idea**: Mixture-of-Experts gives "conditional *computation*" (route each token to different expert weights depending on content). Engram adds an orthogonal axis: "conditional *memory*" — a lookup-based, deterministic memory module that retrieves static knowledge without spending compute reconstructing it via attention/MLP layers every time.
- **Mechanism**: 
  1. At each token position, look at the local suffix (2-grams, 3-grams) of recent tokens.
  2. **Tokenizer compression**: normalize tokens to canonical form first (collapse case/whitespace/Unicode variants) — this shrank effective vocabulary by 23%, improving lookup density.
  3. **Multi-head hashing**: since you can't store a table entry for every possible n-gram combination, hash the compressed n-gram through multiple hash functions into a large embedding table (mitigates hash collisions via multiple independent heads).
  4. **Context-aware gating**: the retrieved memory embedding is NOT blindly added to the residual stream — it's gated by the current hidden state, so the model can down-weight retrieved memory when it doesn't fit the current context.
- **Key property — determinism**: unlike MoE routing (depends on live hidden-state activations, so routing is dynamic/opaque), Engram's lookup index depends *only* on the input token sequence. This means the (very large, e.g. 100B-parameter) memory table can be **offloaded to host/system RAM** (not GPU VRAM) and prefetched asynchronously, because you know in advance which addresses you'll need — DeepSeek reports under 3% inference overhead from this offload for a 100B-param memory table.
- **Empirical finding — optimal split**: through systematic sweeps, DeepSeek found the best allocation of sparse model capacity is roughly **75–80% dynamic computation (MoE) / 20–25% static memory (Engram)**. Pure MoE (100% compute, 0% memory) was suboptimal — too much depth gets wasted "simulating" retrieval that could've been a cheap lookup instead. Under iso-parameter/iso-FLOPs comparisons, an "Engram-27B" model beat MoE-only baselines across knowledge, reasoning, code, and math.
- **Why it matters conceptually**: it reframes "memory" as a first-class trainable architectural primitive rather than something bolted on afterward (e.g. RAG/vector DB) — the model *learns* what to store in the lookup table jointly with the rest of pretraining, end-to-end, fully differentiable.

## What this means for your own smaller-scale project
You will not reproduce a 100B-parameter Engram table or a 1M-token CSA/HCA attention stack on a VRAM-limited setup — but the *transferable ideas* are:
1. **Separate "cheap deterministic lookup" from "expensive contextual reasoning"** — even a small hash-based n-gram/embedding cache for frequent local patterns (names, recurring phrases, entities in your domain) can offload work from your limited attention context, conceptually similar in spirit even if you're doing it via a much simpler cache/dictionary rather than a trained hashed embedding table.
2. **Compress-then-sparse-select is a two-stage general pattern**: (a) pool/compress a block of context into a single summary representation, (b) attend/select sparsely over the compressed summaries rather than raw tokens. You can approximate this at inference time even without changing model weights: chunk your long input, summarize each chunk (cheaply, even with a small model or heuristic), then feed the model only the most relevant compressed chunks + the literal most recent tokens (mirrors CSA's "consolidate + sparse top-k select" behavior) — see 04-context-window-boosting/README.md for the concrete recipe.
3. **Tensor-level activation checkpointing** is a free, architecture-agnostic way to trade compute for memory on your own GPU-poor setup — worth enabling (`gradient_checkpointing=True` in HF Transformers, or granular checkpointing in your own training loop) if you're VRAM-bound during training.
