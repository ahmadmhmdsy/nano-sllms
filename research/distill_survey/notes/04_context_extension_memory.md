# Context extension & memory — survey (2026-09-08)

Research note. Scope: can the 226M GQA student (16L, hidden 1024, 16Q/4KV heads, CodeLlama tokenizer, **trained ctx 1024**, RTX 3000 6 GB, fp16, SDPA) handle >>1024 context — and can a memory module double as cross-call recall?
Cross-references `research/gdn_sandbox_design.md` (GDN hybrid, fp32-state machinery) and `configs/target.yaml` (dims; seq-1024 train peak 4.24 GB).
Labels: VERIFIED = stated in a fetched payload; REPORTED = secondary/blog claim; COMPUTED = our arithmetic; UNVERIFIED = pointer not fetched.

## TL;DR
- KV cache is **not** our bottleneck: 16 KiB/token fp16 (COMPUTED) → 512 MiB even at 32k, smaller than the ~452 MB fp16 weights. The real blockers are positional extrapolation and O(n²) prefill compute.
- Positional extension splits into two families: **no-finetune** (NTK-aware, Dynamic-NTK, StreamingLLM) and **short fine-tune** (PI, YaRN — "10x less tokens, 2.5x less steps than previous methods" — https://arxiv.org/abs/2309.00071).
- YaRN is the production default ("Most production long-context models use YaRN or NTK scaling, often combined with a brief continued-pretraining stage" — https://mlmentorship.com/concepts/long-context-llms/); published Llama-2 YaRN checkpoints reach 8k/32k/64k/128k (https://github.com/jquesnelle/yarn).
- StreamingLLM: sliding window + first-token attention sinks → stable streaming to 4M tokens **with no fine-tuning**, but only continuation quality — dropped tokens are unrecoverable (https://arxiv.org/abs/2309.17453).
- KV quantization (KVQuant, 3-bit, <0.1 ppl degradation) is mature but saves us only ~0.25–0.4 GB at 32k — worth doing late, not first (https://arxiv.org/abs/2401.18079).
- Soft-token compression (ICAE 4×, AutoCompressor ~13×, 500xCompressor 6–500×) is trained 7B-scale and needs full-model fine-tune; at 226M on code it is research-project territory (§3).
- Learned memory (RMT memory tokens, Infini-attention, Titans) is the only family that trains long-context *behavior* at 1024-chunk VRAM cost AND gives persistent state you can save/restore across calls.
- Infini-attention's compressive memory is a delta-rule linear state — the same math family as the GDN sandbox; the fp32-state/fp16-IO machinery from `gdn_sandbox_design.md` §4 is directly reusable.
- Cross-call "remembering" is feasible: segment-recurrent state is a few MB tensor (COMPUTED) — checkpoint it per conversation, restore on swap. But it stores compressed gist, not verbatim recall.
- Grounded-answer ranking (ans2) put KV quantization #1 for a 226M model — our math disagrees (§5, conflict noted in §6).
- Ranked plan: (1) NTK/Dynamic-NTK inference-only eval, (2) YaRN fine-tune at ctx 2048, (3) RMT-lite memory-token + segment recurrence (reusing GDN state code), (4) compression models — park.

## Sources
| Source | URL | Date | Type |
|---|---|---|---|
| YaRN paper (ICLR 2024) | https://arxiv.org/abs/2309.00071 | 2023-09 | paper (abstract+TOC payload) |
| YaRN repo (8k–128k checkpoints, training data) | https://github.com/jquesnelle/yarn | — | code/repo |
| EleutherAI "Extending the RoPE" | https://blog.eleuther.ai/yarn/ | 2023-11-13 | blog (math of PI/NTK) |
| YaRN hyperparams guide | https://alessioborgi.github.io/blog/transformers/yarn/ | 2026-05-26 | blog (β grouping, ~400 steps) |
| YaRN config in NeMo-RL | https://docs.nvidia.com/nemo/rl/0.7.0/guides/yarn-long-context.html | — | docs (required config fields) |
| ICAE (ICLR 2024) | https://arxiv.org/html/2307.06945 | 2023-07 | paper (abstract payload) |
| AutoCompressors (EMNLP'23) | https://arxiv.org/abs/2305.14788 + https://github.com/princeton-nlp/AutoCompressors | 2023-05 | paper+repo |
| 500xCompressor (ACL 2025) | https://arxiv.org/html/2408.03094 + https://aclanthology.org/2025.acl-long.1219.pdf | 2024-08/2025 | paper (abstract payloads) |
| Infini-attention | https://arxiv.org/html/2404.07143v2 | 2024-04 | paper (abstract payload) |
| Titans (NeurIPS 2025) | https://arxiv.org/pdf/2501.00663 | 2025-01 | paper (abstract payload) |
| ∞-former (note: returned for RMT query, is *not* RMT) | https://arxiv.org/pdf/2109.00301v3 | 2021-09 | paper (abstract payload) |
| StreamingLLM (ICLR 2024) | https://arxiv.org/abs/2309.17453 + https://github.com/mit-han-lab/streaming-llm | 2023-09 | paper+repo |
| KVQuant (NeurIPS 2024) | https://arxiv.org/html/2401.18079v4 | 2024-01 | paper (abstract payload) |
| KV optimization guide (formulas, MQA→Ring) | https://www.youngju.dev/blog/llm/2026-03-07-llm-long-context-kv-cache-optimization.en | 2026-03-07 | blog (fetched) |
| Long-context LLMs: training & serving | https://mlmentorship.com/concepts/long-context-llms/ | 2025-11-15 | guide (fetched) |
| KV guide w/ GQA formula example | https://www.gpuyard.com/tutorials/howto/optimize-kv-cache/ | 2026-07-31 | blog |
| InsiderLLM KV guide (hybrid-attention cuts) | https://insiderllm.com/guides/kv-cache-optimization-guide/ | 2026-02-23 | blog |
| Kara sliding-window KV compression | https://arxiv.org/html/2607.01237v2 | 2026-07-03 | paper (abstract payload) |
| Exa grounded answer | research/raw/ans2_ctx_boost.json | 2026-09 | answer (cites mlmentorship, youngju, ICAE, Infini-attention, continuousfunction.ai) |

## 1. RoPE/YaRN family — what each does, cost at 226M, multiples

- **Position Interpolation (PI)** (Chen et al. 2023; kaiokendev): rescale positions by `g(m)=m/s`, `s=L'/L`, then short fine-tune. EleutherAI derivation: https://blog.eleuther.ai/yarn/ (VERIFIED). Typically ~2–4× with fine-tune (REPORTED, no payload number).
- **NTK-aware** (bloc97): scale the RoPE base (θ), not positions; preserves high-frequency dims. **No fine-tuning needed** to stay coherent somewhat past training length; used by Code Llama (VERIFIED: YaRN paper says "Code Llama (using NTK-aware interpolation)" — our tokenizer lineage, not our checkpoint).
- **Dynamic NTK**: NTK scaling applied at inference only when seq_len exceeds training length (Qwen 7B ships it; VERIFIED). The zero-cost experiment for us.
- **YaRN**: per-frequency-band treatment — low-freq dims → linear interpolation, high-freq dims → unchanged, mid-freq → NTK-style ramp (α=1, β=32 defaults), plus an attention-temperature correction (VERIFIED: https://alessioborgi.github.io/blog/transformers/yarn/). "10x less tokens and 2.5x less training steps than previous methods" (VERIFIED abstract). Llama-2-7b fine-tunes published at 8k/32k/64k/128k on Long-Data-Collections (VERIFIED repo). ~400 fine-tuning steps needed (REPORTED at 7B scale: Borgi blog).
- **Config knobs** (HF `rope_scaling`, required-field list from NeMo-RL docs, VERIFIED): `rope_type: yarn`, `factor = target/original`, `original_max_position_embeddings: 1024`, `beta_fast: 32`, `beta_slow: 1`, `mscale`, `mscale_all_dim`, `truncate`. Zero extra VRAM — it is a re-parametrization of existing `q/k` projection use (COMPUTED).
- **LongLoRA / LongRoPE**: NOT covered by any fetched payload — UNVERIFIED pointers (LongLoRA: fine-tune with shifted-sparse attention + LoRA, arXiv 2309.12307; LongRoPE: search-based non-uniform interpolation to 2M, arXiv 2402.13753). Listed for completeness only; do not quote numbers from this note.
- **Fine-tune cost at 226M**: YaRN's efficiency claims are 7B-scale; nothing in the payloads tests ≤1B. COMPUTED estimate for us: 300–1,000 steps × (2048 tok × bs1 × accum16 ≈ 32k tok/step) ≈ 10–32M tokens — trivial in tokens; the binding constraint is activation VRAM at long seq (§5). Extrapolating a 4k-fine-tuned model *inference-only* to 8k is the YaRN paper's own claim ("extrapolate beyond the limited context of a fine-tuning dataset", VERIFIED).

## 2. KV-cache VRAM levers — math for OUR model

- **Generic formula** (MHA form): `KV bytes = 2 (K+V) × n_layers × d_model × seq_len × batch × dtype_bytes` (VERIFIED: youngju.dev; worked example LLaMA-2-7B @ 4k = 2 GB).
- **GQA form** (VERIFIED arithmetic pattern from GPUYard's Llama-2-70B example: `2×1×32768×80×8×128×2`): replace `d_model` with `kv_heads × head_dim`.
- **COMPUTED for our model**: head_dim = 1024/16 = 64; per layer per token = 2 × 4 (kv_heads) × 64 × 2 B (fp16) = **1,024 B**; × 16 layers = **16,384 B/token ≈ 16 KiB/token** (batch 1). (The task-prompt hint "16·2·1024·2 per layer" was the MHA form — GQA divides by 4.)
  - ctx 1024: 16 MiB · ctx 4k: 64 MiB · ctx 8k: 128 MiB · ctx 32k: 512 MiB. Weights fp16 ≈ 452 MB (226M×2B) → KV surpasses weights only past ~29k tokens (COMPUTED).
- **Lever A — sliding window + attention sinks (StreamingLLM)**: keep KV of `sink` initial tokens (sinks exist because initial tokens absorb excess attention mass, VERIFIED) + last `w` tokens; positions are **cache-relative**. No fine-tuning; stable to 4M tokens; 22.2× speedup vs window-recompute baseline (VERIFIED). At w=1024+4 sinks: **~16.1 MiB fixed regardless of stream length** (COMPUTED, ≈0 savings vs our 32k full KV — the win is constant-memory streaming + compute, not saved MiBs, at our scale).
- **Lever B — KV quantization**: KVQuant: pre-RoPE key quant + per-channel + non-uniform datatypes → **3-bit with <0.1 ppl degradation**; enables Llama-7B @ 1M ctx on one A100 (VERIFIED). Generic INT8 = 2×, INT4 = 4× (VERIFIED: mlmentorship "int8 or int4 KV cache. 2–4× cache reduction"). For us: INT8 → 8 KiB/token → 256 MiB @ 32k (**saves 256 MiB**; COMPUTED). Low priority — KV is small already.
- **Lever C — cross-layer KV sharing (YOCO / MLKV)**: reuse one layer's KV across layers (inter-layer sharing). UNVERIFIED in this note (no payload fetched): YOCO arXiv 2405.05254, MLKV arXiv 2410.07999. Naive division by n_layers would take us 16 KiB → 1 KiB/token (COMPUTED upper bound if all layers share), but that changes the architecture — not a drop-in for a trained checkpoint. Closest *verified* data point: hybrid attention — Gemma-4-style "5 of 6 layers SWA (1024-window) + 1/6 full attention" cuts effective KV "closer to 1/4 of what the legacy formula predicts" (REPORTED: InsiderLLM, 2026 blog) — i.e., the GDN-hybrid direction in `gdn_sandbox_design.md` is the same lever.
- **Lever D — PagedAttention/offload**: fragmentation control + CPU offload; helps serving throughput, not a 6 GB single-request limit (VERIFIED characterization: mlmentorship; InsiderLLM `--cpu-moe` for MoE). Marginal for bs=1 decode.
- **Recent pointer** (VERIFIED payload, not applicable directly): Kara (arXiv 2607.01237, 2026) — decoding-time sliding-window KV compression with Token2Chunk, built on vLLM/PagedAttention.
- **Sanity check on the grounded answer**: ans2 ranks KV quantization #1 for a 226M model. COMPUTED reality: our KV @ 32k is 512 MiB on a 6 GB card — quant saves ~256–384 MiB but positional extrapolation, not KV, is what breaks first. Rank corrected in §5.

## 3. Context compression into soft tokens

- **ICAE** (ICLR 2024, https://arxiv.org/abs/2307.06945): a LoRA-adapted copy of the LLM *encodes* a long context into `M` memory slots (paper example: 2,572 chars / 512 words → **128 slots**, VERIFIED); the LLM conditions on the slots. **4× compression, ~1% additional params**, trained with autoencoding + LM objectives on massive text, then instruction fine-tuned (VERIFIED). Max input in released Mistral-7B models: 5,120 (VERIFIED repo).
- **AutoCompressors / gisting lineage** (https://arxiv.org/abs/2305.14788): the LM itself compresses past segments into **summary vectors = soft prompts**; unsupervised, segment-wise — "summary vectors from all previous segments are used in language modeling"; fine-tuned OPT/Llama-2 up to 30,720 seq len (VERIFIED). README example: 660 tokens → 50 summary vectors (~13×, VERIFIED). (Gist tokens, Mu et al. arXiv 2306.04667: UNVERIFIED pointer — no payload.)
- **500xCompressor** (ACL 2025, https://arxiv.org/html/2408.03094): trains a small compressor (~0.25–0.3% extra params) that maps text into **1–~80 special tokens whose KV values are spliced into the base LLM** — base LLM used *without fine-tuning*; 6×–500× ratios; retains 62.26–72.89% of LLM capability (arXiv version) / 70–74% F1 & 77–84% EM (ACL version); 27–90% compute reduction, 55–83% memory savings at 500× (all VERIFIED from the two abstract payloads). Key mechanism note: **"KV values outperform embeddings in preserving information at high compression ratios"** (VERIFIED).
- **Tiny-scale caveats** (COMPUTED/REASONED): all three were demonstrated on 7B+ LLMs; the *encoder* must be as capable as what it re-encodes, so at 226M the ceiling drops; our domain is code (high information density — worst case for 4–500× compression); training ICAE/AutoCompressor-style = full-model fine-tune (VRAM fits at ctx 1024, but quality is unproven at this scale). 500xCompressor is the most compatible in principle (base model untouched, KV-splice works with GQA), but its ArxivCorpus/ArxivQA training regime does not transfer to CodeSearchNet-style data without a re-train.

## 4. Learned memory modules — state carried across segments, and across *calls*

- **RMT (Recurrent Memory Transformer)**: memory tokens `M` prepended/appended to each fixed-length segment; segment N's output memory vectors become segment N+1's input memory — trained with TBPTT so gradients flow through the memory chain. NOT covered by the fetched payloads (the ctx3 RMT query returned the **∞-former** instead): UNVERIFIED pointer arXiv 2304.11062. Adjacent VERIFIED payload: **∞-former** (arXiv 2109.00301) — unbounded long-term memory via continuous-space attention with "sticky memories", computation independent of context length.
- **Infini-attention** (https://arxiv.org/html/2404.07143v2): one block = **masked local (windowed) attention + long-term linear-attention compressive memory**; the segment's old KV/Q states update a bounded memory matrix that is passed to the *next segment*; enables "fast streaming inference"; demonstrated 1M passkey retrieval and 500K book summarization with 1B/8B models (VERIFIED). Third-party impl: lucidrains/infini-transformer-pytorch — "linear attention scheme to compress past memories", segment API with `past_memories` in/out, `detach_mems_every_num_segments` for learnable writes, delta-rule option (`use_mem_delta_rule`) (VERIFIED repo payload).
- **Titans** (https://arxiv.org/pdf/2501.00663, NeurIPS 2025): a neural long-term memory module that **memorizes at test time** (surprise-driven updates) alongside attention ("attention = short-term, accurate; neural memory = long-term, persistent"); three integration variants (memory-as-context / as-gate / as-layer); scales ">2M context window" (VERIFIED). Training requires the memory module itself (MLP + test-time gradient descent) — heaviest of the three to retrofit.
- **Does memory persist across independent inference calls?** Within a session: yes by construction — the state tensor is threaded segment→segment. Across calls/context swaps: **yes, if we checkpoint and restore the state** — the memory is an explicit tensor we own (COMPUTED/REASONED; the papers evaluate continuous streams, not cross-session recall — no source claims or refutes cross-session persistence, so treat as an engineering hypothesis to test). It is compressed gist, not verbatim recall (COMPUTED from mechanism).
- **Integration options for an existing GQA checkpoint** (COMPUTED/REASONED):
  1. **RMT-lite memory-token prefix**: freeze architecture, add M new memory-token embeddings (16×1024 ≈ 16k params), fine-tune with segments of (1024−M) real + M memory tokens, thread memory vectors across segments. VRAM ≈ unchanged from the 4.24 GB seq-1024 run (COMPUTED). Lowest-risk retrofit.
  2. **Infini-attention layer**: reuse each layer's existing Q/K/V projections; add a per-head linear memory `S ∈ R^{64×64}` updated by delta rule between segments. State = 16L × 16heads × 64×64 × 4B (fp32) ≈ **4.2 MB** (COMPUTED) — trivially checkpointable. **Synergy**: this is the same delta-rule state machinery + fp32-state/fp16-IO discipline as `gdn_sandbox_design.md` §1.1/§4 (R1) — one code path serves both.
  3. **Titans-style module**: new memory MLP per layer trained from scratch-ish — a new pretraining-scale experiment, not a checkpoint retrofit (REASONED).

## 5. Applicability to nano_SLMs — ranked feasibility (our model: 1024-trained, 6 GB, fp16, SDPA)

VRAM baselines (COMPUTED): seq-1024 train peak = 4.24 GB (VERIFIED `configs/target.yaml` comment); fp16 weights ≈ 452 MB; optimizer/AMP fp32 states ≈ 3.6 GB of that peak; activations @1024 w/ grad_ckpt ≈ 0.6 GB → @2048 ≈ 1.2 GB (total ≈ 4.8–5 GB, likely fits) → @4096 ≈ 2.4 GB (total ≈ 6 GB+, OOM) (COMPUTED, linear extrapolation — verify with a probe, not assumption). 8-bit Adam (repo-flagged fallback, `configs/target.yaml` L59) frees ~1.8 GB if needed.

| Rank | Lever | Verdict | VRAM / cost at 4k–8k |
|---|---|---|---|
| **A1** | **Dynamic-NTK / NTK-aware inference-only eval** | CHEAPEST, do first: `rope_scaling` config change, zero training, measure passkey/NIAH at 2k/4k | 0 extra VRAM; KV @4k = 64 MiB, @8k = 128 MiB (COMPUTED). No-finetune NTK quality at 226M is UNTESTED in sources — expect degradation beyond ~2× (REASONED from YaRN narrative) |
| **A2** | **YaRN 1024→4k fine-tune** | FEASIBLE but VRAM-shaped: fine-tune **at ctx 2048** with factor 2, then extrapolate to 4k inference-only (the YaRN paper's own extrapolation claim, VERIFIED); ctx-4096 training OOMs without 8-bit Adam or chunked training. Steps: ~300–1,000 (COMPUTED estimate from ~400-step 7B REPORTED figure), ≈ 10–32M tokens, hours-scale on the throttled GPU | Weights+optimizer ~4.2 GB + activations ~1.2 GB @2048 ≈ 5.4 GB — tight; probe before launching (COMPUTED) |
| **B** | **Sliding window + sinks (StreamingLLM eval)** | TRIVIAL eval-time add (no training): keep 4 sink + last w KVs, cache-relative positions. Good for streaming continuation; **cannot retrieve beyond window** (VERIFIED mechanism) | ~16 MiB fixed at any length (COMPUTED); zero training |
| **C** | **RMT-lite memory tokens + segment recurrence** (or Infini-attention layer variant) | THE only lever that (i) trains long-context behavior at 1024-chunk VRAM (~4.24 GB, unchanged), (ii) survives context swaps via checkpointable state, (iii) builds directly on `gdn_sandbox_design.md` fp32-state code. Needs new code (memory-threading in `src/` + eval loop) + a 1k–3k-step fine-tune on long-doc streams (COMPUTED estimate) | State to persist: RMT tokens ≈ 0.5 MB; Infini-style memory ≈ 4.2 MB fp32 (COMPUTED) — per-conversation recall for ~free |
| **D** | **ICAE / AutoCompressor / 500xCompressor** | RESEARCH-PROJECT ONLY at 226M: 7B-scale results, full-model fine-tune, code-domain compression untested; revisit only after A–C. 500xCompressor is the most architecturally compatible (KV-splice + no base-model fine-tune) if ever attempted | Training fits @1024 (same 4.24 GB envelope) but quality is the risk, not VRAM (REASONED) |

Recommended sequence: A1 (today, eval-only) → B (same eval harness) → A2 YaRN @2048 fine-tune → C as the G1-successor experiment (shares GDN state code); D parked.

## 6. Pointers from the parallel-runs comparison (2026-09-08) — verification round

Verified 2026-09-08 against `research/raw/ver7_*.json` (12/12 round-7
tasks); do NOT quote numbers beyond what the payloads contain:
- **Engram**: **VERIFIED** (`ver7_engram.json`) = arXiv 2601.07372,
  "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs"
  (+ github.com/deepseek-ai/Engram; ACL 2026 long). Abstract: conditional
  memory as a complementary sparsity axis to MoE — Engram "modernizes
  classic N-gram embedding for O(1) lookup"; a "U-shaped scaling law"
  governs the MoE↔static-memory allocation; scaled to 27B parameters.
  Claude's specifics (tokenizer compression −23%, host-offload <3%
  overhead, 75–80/20–25 split) stay REPORTED — not in the abstract payload.
  Parametric-memory sibling of §4's RMT/Infini/Titans family, adjacent to
  the GDN state path; needs training (not a checkpoint retrofit).
- **Larimar (IBM)**: **VERIFIED** (`ver7_larimar.json`) = arXiv
  2403.11901, ICML 2024 (+ github.com/IBM/larimar): "distributed episodic
  memory" with "dynamic, one-shot updates of knowledge" without retraining;
  8–10× editing speed-ups reported. The most literal published match to
  the user's "add a small memory part" ask; compare against
  Infini-attention before any design work.
- **CAMELoT (IBM)**: **VERIFIED existence** (`ver7_camelot.json`) =
  arXiv 2402.13449 (ICML-W 2024): training-free associative memory coupled
  to a frozen attention-based LLM, consolidating token representations
  into a non-parametric distribution. ChatGPT's "~30% ppl reduction"
  number: UNCONFIRMED in our payload — do not quote.
- **Compression sweep**: **VERIFIED existence** (`ver7_compression_sweep.json`):
  CompLLM (arXiv 2509.19228, segmented compression into concept
  embeddings, frozen base), AdmTree (arXiv 2512.04550, NeurIPS 2025,
  semantic-tree gist compression), AttnComp (arXiv 2509.17486,
  attention-guided Top-P compression for RAG), AMS (arXiv 2605.23200,
  region-quota KV eviction — ID matches Perplexity's), TurboQuant (arXiv
  2504.19874; **correction**: 4-bit KV via random rotation + Lloyd-Max —
  Perplexity's "3.5-bit / 6× / 328 GB→55 GB" numbers are NOT confirmed;
  drop them). All still PARK under §5-D (trained compressor / serving
  stack required).
- **ARMT / InfLLM**: **NOT confirmed** this round (the sweep answer could
  not find them) — remain UNVERIFIED pointers.
- **MiniMax-M1**: **VERIFIED** (`ver7_minimax_m1.json`) = arXiv
  2506.13585, "world's first open-weight, large-scale hybrid-attention
  reasoning model" — hybrid MoE + lightning (linear) attention.
  Corroborating datapoint for the hybrid/linear direction already chosen
  in `gdn_sandbox_design.md`; record only.

## 7. Open questions
1. **Does our from-scratch model even have attention sinks?** StreamingLLM's sinks were observed in pretrained LMs; a 1024-trained-from-scratch 226M may not need sink tokens. Test: measure layer-wise first-token attention share at seq 1024 (cheap eval-only run).
2. **What is the actual activation VRAM curve vs ctx?** 4.24 GB @1024 is one point; the @2048 fit for YaRN training is a linear extrapolation (COMPUTED) — run a short ctx-2048 VRAM probe (when GPU is free) before committing the fine-tune.
3. **How much does inference-only NTK/YaRN lose at 4k at 226M scale?** All payload evidence is 7B+; the passkey-retrieval curve (no-finetune vs YaRN-finetuned) is unmeasured at our scale and decides A1 vs A2.
4. **How many memory tokens M before gist quality collapses for code?** Papers used ~50–128 slots for hundreds of tokens of prose; code is denser. Sweep M ∈ {8, 16, 32, 64} in the C experiment.
5. **Does checkpointed cross-call memory actually work?** No source tests state save/restore across independent calls; validate with a two-session recall eval (tell it a fact in session 1, restore state in session 2, ask for the fact).
