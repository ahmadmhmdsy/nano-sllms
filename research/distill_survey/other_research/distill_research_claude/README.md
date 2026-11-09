# Boosting effective context window on a VRAM-limited (~1000 token) setup

This answers: "our cap is 1000 tokens because of VRAM — can we trick the model into effectively handling more, like DeepSeek-V4's compression, and can we give it a persistent memory that survives even when we wipe the context?"

Sources:
- arXiv:2402.02244 — "Beyond the Limits: A Survey of Techniques to Extend the Context Length in LLMs"
- arXiv:2607.05708 — "Akashic: A Low-Overhead LLM Inference Service with MemAttention" (covers Mem0, MemGPT, RMM, SeCom memory-compression patterns used in production agent frameworks)
- arXiv:2507.19353 — "Smooth Reading" (sliding-window recurrent LLM long-context behavior)
- arXiv:2406.14909 — "Mixture of Attention Spans" (heterogeneous sliding-window lengths)
- Medium/"Context Kills VRAM" series (practical VRAM-vs-context-length mechanics)
- hardware-corner.net — context length / VRAM practical guide
- redis.io blog on context windows
- Cross-reference: 02-frontier-training-techniques/deepseek-v4.md (CSA/HCA, Engram) and qwen3.8-flash-next.md (N-gram table) for how frontier labs solve the same problem at huge scale

## First, understand WHY 1000 tokens costs what it costs
- VRAM during inference ≈ fixed cost (model weights) + **linear cost in context length** (the KV cache: one key+value vector pair per token per layer per head, stored for the whole prompt+generation).
- Training/self-attention compute is quadratic in context length; but for a fixed small model just doing inference, the dominant *memory* cost from context is the KV cache, which is linear — so your bottleneck is almost certainly **KV cache VRAM**, not raw compute.
- This means: you cannot literally "trick" the model into holding 10,000 tokens of full-precision KV cache in the VRAM budget for 1,000 — the physics of the cache size doesn't change. What you CAN do is reduce **what has to go into that cache**, or reduce **the per-token cost of what's in it**. That's exactly what DeepSeek-V4 and Qwen3.8-Flash-Next do at giant scale; here's the small-scale equivalent of each of their tricks.

## Trick 1 — Reduce per-token KV-cache cost (architecture-level, needs the right base model)
- **Grouped Query Attention (GQA)**: multiple query heads share one K/V projection — most modern open small models (Qwen, Llama-3, etc.) already use this. If your current model doesn't, switching to a GQA-based architecture cuts KV cache size substantially for free.
- **Quantized KV cache**: store K/V in FP8 or even INT4/INT8 instead of FP16/BF16 (llama.cpp, vLLM, and other inference engines support this). This alone can roughly halve or quarter your effective KV memory cost, letting the *same* 1000-token VRAM budget hold 2-4x as many tokens.
- **FlashAttention**: doesn't reduce KV cache size directly, but reduces the *working* SRAM/HBM traffic during attention computation, freeing some VRAM headroom and speeding things up — worth having enabled regardless.
- This is the small-scale analogue of DeepSeek-V4's "FP8 storage for most KV entries, BF16 only for RoPE dims" and the general MLA (low-rank KV projection) lineage — same idea (store less per token), simpler implementation.

## Trick 2 — Reduce the NUMBER of tokens that need dense attention (this is the one you can implement immediately, no architecture change needed)
This is the direct small-scale analogue of DeepSeek's CSA (**compress + sparse-select**) and of sliding-window architectures:

1. **Sliding window + rolling summarization** (documented production pattern used by Mem0, MemGPT, RMM, SeCom per the Akashic paper):
   - Keep only the most recent N raw tokens verbatim (e.g. last 300-500 tokens) — this is your "sliding window."
   - Everything older gets periodically compressed: every time you approach your token budget (e.g. every 800-1000 tokens), have the model (or a smaller/cheaper model) generate a **concise running summary** of everything so far.
   - Discard the raw old tokens, keep only: [running summary] + [most recent raw tokens] + [new input]. This is functionally the same shape as CSA: "consolidate old context into a compressed representation, then attend densely only to what's near/relevant."
   - Concretely for your 1000-token cap: reserve maybe 150-250 tokens for the rolling summary, and let the remaining 750-850 tokens be the sliding window of recent raw turns + current query.

2. **Retrieval instead of brute-force inclusion (RAG-lite)**:
   - Instead of trying to cram everything into the context, store past information (facts, prior turns, documents) in a simple vector index or even a keyword/BM25 index.
   - At each new query, retrieve only the top few most relevant chunks and inject *those* into the small remaining context budget, rather than the full history.
   - This is literally what "retrieval-based memory" does in the Mem0/LlamaIndex pattern cited in the Akashic paper: compress history into compact representations, retrieve only a handful of relevant items per query.

3. **MoA-style heterogeneous windows** (arXiv:2406.14909) — if you ever get to fine-tune, different attention heads can be assigned different window lengths (some short/local, some long/sparse) rather than one uniform window for all heads — reported 1.2-1.4x memory reduction and large throughput gains with minimal quality loss. This is more involved (needs model surgery/calibration) but is a proven, published recipe if you outgrow the simple sliding-window approach.

## Trick 3 — Give the model persistent memory that survives even when you wipe the context entirely
This is the part of your question about "even if we remove the entire context and provide it with another, the model can remember part of it."

There are two very different tiers of doing this, matching what the frontier labs do vs. what's practical for you right now:

**A) Frontier-lab tier (training-time, parametric)** — DeepSeek's Engram / Qwen's N-gram embedding table (see 02-frontier-training-techniques/ for full detail):
- A separate, large embedding lookup table, addressed deterministically by hashed local token n-grams, trained end-to-end with the rest of the model.
- Because addressing is deterministic (depends only on input tokens, not on live hidden states), the table can be **offloaded to system RAM** and prefetched asynchronously — it doesn't need to live in GPU VRAM at all, which is exactly your constraint.
- This requires retraining/fine-tuning the base model to actually learn to use such a table — not something you bolt on to an already-trained small model without further training.

**B) Practical tier for you right now (inference-time, external, no retraining needed)** — "OS-inspired" agent memory, exactly the MemGPT/Mem0 pattern:
- **MemGPT's core idea ("virtual context management")**: treat the model's actual context window like RAM, and an external store (disk/DB) like swap space. The model (or your orchestration code) explicitly "pages" information in and out — writing important facts out to persistent storage, and reading back only what's relevant to the current turn.
- Concretely, this looks like:
  1. Maintain a small persistent key-value or document store *outside* the model (a JSON file, SQLite DB, or vector DB) — this is your "memory" that survives across sessions/context wipes.
  2. After each interaction (or periodically), extract the durable facts worth keeping (you can literally prompt your model: "extract any facts from this conversation worth remembering long-term, as a short bullet list") and write them to that store.
  3. On a new session/context (even after totally wiping the prompt), before running the user's new query, retrieve the small number of relevant stored facts (via keyword match or vector similarity) and prepend them to the fresh 1000-token context.
  4. This gives you the *functional* effect of "the model remembers even after I removed the whole context" — the persistence lives in your external store, not in the frozen model weights, but from the end-user's point of view it behaves the same way, and it costs zero extra VRAM.

## Trick 4 — Combine 2 and 3 (this is literally what "compress + expand" means for your use case)
Your instinct that compression and expansion should "work together" is exactly right and exactly what both DeepSeek and the agent-memory frameworks do:
- **Compress**: rolling summarization + retrieval (Trick 2) keeps your active 1000-token window always full of the *highest-value* information rather than raw chronological history.
- **Expand/persist**: the external memory store (Trick 3B) is your unlimited-size "cold storage" that the compression step writes to and reads from.
- Together: effectively unbounded conversation length, bounded active VRAM, at the cost of (a) occasional summarization calls (cheap, can use a small/fast model), and (b) some information loss from compression (mitigate by keeping raw text of the most recent turns verbatim, only summarizing older material).

## Direct mapping table (frontier technique → your small-scale equivalent)

| Frontier technique | What it does | Your small-scale equivalent |
|---|---|---|
| DeepSeek CSA (compress every m tokens, then sparse-select) | Reduce KV cache + FLOPs at 1M context | Rolling summary of old turns + sliding window of recent raw turns |
| DeepSeek HCA (very aggressive compression, dense attention over survivors) | Cheap coarse global context | A single running "conversation summary" string, always included |
| DeepSeek Engram / Qwen N-gram table (deterministic, offload to host RAM) | Persistent parametric memory outside GPU VRAM | External JSON/SQLite/vector-DB memory store, retrieved and injected at inference time |
| MLA / GQA (shrink K/V per token) | Cheaper KV cache per token | Quantized KV cache (FP8/INT4) + GQA-based base model |
| Multi-Token Prediction (train-time aux objective) | Denser training signal | Not directly applicable to inference-time context tricks, but relevant if you fine-tune your own small model later |

## What NOT to expect
- None of Tricks 1-4 literally let a small un-modified model attend with full fidelity over more raw tokens than its true architectural limit and your true VRAM budget allow — you are always trading *some* fidelity (via summarization/retrieval imprecision) for *effective* reach. The frontier labs pay for higher fidelity at large scale with billions of dollars of architecture R&D + massive training runs (Engram tables, CSA/HCA, mHC, Muon at trillion-token scale). At your scale, the honest, robust path is Tricks 2+3 (summarization + retrieval + external memory store) — they require zero model retraining, work with any base model, and are exactly what production agent frameworks (MemGPT, Mem0) use today for this exact problem.
