# Qwen3.8-Flash-Next / Qwen3.5 — architecture + training technique

(Note: there is no "Qwen 3.8 Pro" as of this research; the relevant recent releases are **Qwen3.5** (e.g. Qwen3.5-35B-A3B, unified vision-language) and **Qwen3.8-Flash-Next**, released ~August 2026, which is explicitly framed by Qwen as a public "working draft" of their next architecture generation, not a finished flagship.)

Sources:
- github.com/QwenLM/Qwen3.8-Flash-Next
- huggingface.co/Qwen/Qwen3.8-Flash-Next and unsloth's mirror/GGUF pages
- IntuitionLabs guide (referencing Qwen's own arXiv:2608.30320 technical report, "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability")
- huggingface.co/Qwen/Qwen3.5-35B-A3B model card

## Qwen3.8-Flash-Next specs
- 125B total parameters, **6B activated per token** (very sparse MoE).
- Plus a **51B-parameter N-gram embedding table** (separate from the main 125B; conceptually parallel to DeepSeek's Engram idea — cheap local-pattern lookup capacity added alongside the main compute-heavy transformer).
- Plus a **4B-parameter Multi-Token-Prediction (MTP) head** for speculative decoding.
- Native context length: 262,144 tokens; extensible to 1,000,000.
- Compared to its own predecessor Qwen3.7-Plus: **~1/9th the training cost**, yet superior capability — the headline efficiency claim.

## Architecture innovations

1. **Hybrid attention: Gated DeltaNet + Qwen Sparse Attention (QSA)**
   - Alternates **3 Gated DeltaNet linear-attention layers** with **1 QSA layer** (their new sparse retrieval attention, replacing the older "Gated Attention" pairing used in the previous generation).
   - Gated DeltaNet = a linear-attention variant (constant-ish per-token cost, unlike full quadratic self-attention) used for most layers; the sparse full-attention layer is interspersed sparingly to preserve long-range precise retrieval ability. This "mostly-linear, occasionally-full-attention" pattern is the same general strategy as DeepSeek's CSA/HCA alternation — cheap approximate attention most of the time, expensive precise attention occasionally.

2. **Gated Residual (GR) stream** — widens the residual stream into **4 parallel branches**, with a dynamic gate controlling reads/writes to each branch. Purpose: strengthens cross-layer information flow and training stability (conceptually parallel to DeepSeek's Manifold-Constrained Hyper-Connections — both labs are independently converging on "the plain residual connection is a bottleneck; replace it with something richer/gated").

3. **N-gram Embedding table (51B params)** — looks up a table keyed by local context to add model capacity "for free" (very little extra compute per token). Explicitly designed to be **offloadable to host memory** with asynchronous prefetching overlapped with compute — same design philosophy as DeepSeek's Engram (deterministic addressing → can precompute/prefetch → doesn't need to sit in precious GPU VRAM).

4. **Optimizer — Muon + AdamW split by weight category**: Muon applied to some parameter groups, AdamW to others, chosen per weight-category "to maximize efficiency." Refinements mentioned: better orthogonalization accuracy in Muon, careful division of labor between Muon/AdamW, and splitting of fused parameters (i.e. some weight matrices that are normally fused for efficiency are split apart specifically so each piece can get the optimizer best suited to it).

5. **Training-schedule trick — no batch-size warmup**: guided by refitted scaling laws for this new architecture, they **skip the traditional batch-size warmup** and start directly at the target batch size. This substantially reduces total optimizer steps while still safely supporting larger learning rates for stable convergence. (Batch-size warmup is a very standard practice most training runs do "just in case" — Qwen's point is that if you've actually characterized your scaling law properly for the architecture, you can skip it and save real wall-clock/compute.)

## Practical takeaway
- Both DeepSeek-V4 and Qwen3.8-Flash-Next independently arrived at the same three ideas: (1) mostly-cheap/occasionally-expensive attention alternation, (2) a large deterministic offloadable lookup table as a memory supplement to the main network, (3) gated/hyper residual connections instead of plain ones. This convergence is a strong signal these are now "best practice," not one-off experiments — worth prioritizing if/when you scale beyond toy sizes.
- The optimizer lesson (Muon > pure AdamW for large sparse MoE training, and don't blindly warmup batch size — validate against your own scaling behavior first) is directly actionable even at small scale: it's cheap to try swapping in Muon (open-source implementations exist, e.g. github.com/KellerJordan/Muon) on your own training runs and see if convergence improves.
