# Analysis: DeepSeek-V4.1-Flash → a nano-SLM for a 6 GB Turing GPU

**Companion of:** `research/deepseek_v41_notes.md` (raw extraction) and
`research/raw/deepseek_v41_report_extract.txt` (full paper text).
**Hardware target (ENVIRONMENT.md):** RTX 3000 6 GB / Quadro RTX 4000 8 GB, sm_75, **fp16 only**,
no bf16, no flash-attn (PyTorch SDPA), torch 2.14+cu126, transformers 5.16.1.
**Repo context:** existing ladder S (12M smoke) → P (~110M pilot) → T (~250M target), plain GQA
decoder, 32k vocab, ctx 256–1024, single GPU, auto-resume training.

---

## 1. My honest read of the paper

### What is genuinely new
1. **CED (Causal Encoder-Decoder)** — the boldest idea. Decoder global KV is a *linear projection*
   of the encoder's final hidden state. That's it: `C_l = H_{L/2} @ W_l`. Half the network never
   touches the long context during prefill. The striking part is that it works — "dumb projection"
   beats "each layer reads everything". This validates a broader trend (YoCo, cross-layer sharing):
   **deep global context understanding does not need deep global KV generation.**
2. **CSA2's three modes** (Full/Reindex/Reuse) — a clean, statically scheduled "who computes what"
   taxonomy for cross-layer attention sharing. Most layers do *no* indexing work at all.
3. **Hierarchical Sparse Indexer** — first decoder indexer narrows the world to 16K candidates;
   all deeper indexers are O(1) in context length. Cheap, training-aware, no extra state.
4. **SWA Bounded Replay** — the most counterintuitive and most *transferable* idea: don't store the
   cheap-to-recompute thing; recompute it. Recompute-vs-store is decided by bandwidth math, not
   intuition, and it halves the persistent cache.
5. **The meta-lesson** (their own words): post-training gains came almost entirely from
   **data/environment pipelines**, not algorithmic novelty. And with a compute-starved budget,
   they optimized *software so the hardware works less* — the exact philosophy a nano-SLM lab
   on one consumer GPU must adopt.

### What I'm skeptical about / caveats
- The architecture's wins are **deployment-economics wins** (HBM bytes/token, SSD traffic), not
  raw FLOP wins for small models. At 1k context on a 6 GB card, KV cache is *not* the bottleneck —
  weights, optimizer state, and activations are. Directly porting CSA2/CED buys little at nano scale.
- Approximate state reconstruction (bounded replay) and sparse selection are acknowledged
  robustness risks in their own conclusion. At small scale we'd inherit that risk without their
  evaluation infrastructure.
- Engram (196B fp8 tables in host RAM) and DSpark are serving-scale components; irrelevant for
  training a 100–250M model, except as concepts.
- fp16-only Turing means FP4 KV / FP8 tricks are largely unavailable (fp8 compute unsupported;
  we can only fake small-precision *storage* with dequant-on-read).

### What transfers to us — ranked by leverage
| # | Idea from V4.1 | Nano-scale translation | Expected benefit on 6 GB |
|---|---|---|---|
| 1 | CED: decoder KV projected from one shared hidden state | **Cross-layer KV sharing lite**: let the top half of layers reuse the bottom half's K/V via per-layer projections (or YOCO-style shared KV) | Big prefill-time savings at ctx≥512; also shrinks KV memory so we can raise context |
| 2 | Cross-layer KV reuse (layer dimension) | **2 KV layers per 4 transformer layers** (reuse K/V pairs in groups; each layer keeps its own Q) | ~50% KV cache and K/V-projection memory cut; proven direction (also YOCO, CLA papers) |
| 3 | GQA already in repo | keep, and go more aggressive: 1 KV head is fine at ≤250M with 32k vocab | halves KV again |
| 4 | SWA Bounded Replay | **Windowed attention for most layers + recompute instead of storing local KV** in long-context evals; keep 1–2 global layers | enables 2k–4k ctx experiments inside 6 GB |
| 5 | "No dense warmup — train sparse from scratch" | If we adopt any sparsity, train it **from step 0** — don't pretrain dense then convert (their data point says you don't need the dense stage) | saves an entire training stage |
| 6 | Head-wise Muon + Sinkhorn-embedding update | optional later: Muon-style momentum update is venv-implementable in pure PyTorch; ~1.3–2x sample efficiency reported across labs | faster convergence per token, which is our scarcest resource |
| 7 | Their data philosophy: filter "implicit duplication" (model-generated low-info content), invest in data curation over algorithm novelty | applies directly to our pilot/target runs: strict dedup + quality filters beat clever tricks | same tokens → lower loss |
| 8 | Controllable reasoning effort (scalar in system prompt + exponential length penalty in RL) | relevant only to the SFT/RL phase later — cheap to emulate with SFT data stratified by answer length conditioning | future, low priority |

### What NOT to port
- MoE (384 experts): at 250M params on one GPU, dense beats MoE — routing overhead + optimizer
  memory dominate. The repo's plain GQA decoder is the right call.
- Multimodal stack, Engram, DSpark, DSec, async RL infra: scale-mismatched.
- FP4/FP8 caches: unsupported compute path on sm_75.

## 2. The concrete proposal — "nano-CED" config

A minimal, honest adaptation: **half the layers carry KV; the other half borrow it.**

```
Config NC-1 (fits alongside existing T target, ~250M):
  layers        16
  d_model       1024
  heads         16 Q / 4 KV  (GQA, kept)
  context       1024 (target 2048 with KV sharing)
  SwiGLU d_ff   3072
  KV layout     layers 0–7: standard GQA layers (they own K/V)
                layers 8–15: KV-free layers — Q-only attention, K/V obtained as
                             K_l = H_7 @ Wk_l, V_l = H_7 @ Wv_l   (per-layer projections of
                             layer-7's output hidden state; the CED trick verbatim, scaled down)
  SWA           optional: layers 1..15 use window 128 + the two shared-KV groups;
                layer 0 global. (only if ctx 2048 probes OK)
  RoPE/RMSNorm/tied-emb, fp16, SDPA — unchanged from src/model.py
```

Why this is safe to try:
- It touches **only** the attention K/V source in the top half. Everything else (data pipeline,
  trainer, auto-resume) is untouched — the change is contained in `src/model.py` + config.
- Param delta is small: we *remove* K/V projection weights from 8 layers and add two 1024×1024
  projections per shared-KV layer (or one shared + per-layer bias) → net params roughly flat.
- Memory math (seq 1024, fp16, 4 KV heads, d_head 256): full KV for 16 layers ≈
  2·16·1024·(4·256)·2B ≈ 67 MB per sequence — small, but the **activation** saving during
  backward (8 layers never store K/V or recompute them) and the option to double context is
  the real win at 6 GB.

Staged plan (mirrors the repo's S→P→T ladder discipline):
1. **S-NC (smoke):** 4-layer version of the split (2 KV-bearing + 2 KV-borrowing), smoke config,
   sanity_check + short train. Gate: loss curve overlaps dense smoke within noise, VRAM ≤ smoke's.
2. **P-NC (pilot):** 12-layer pilot (6+6). Gate: matches or beats M3 pilot loss at equal steps;
   VRAM probe shows headroom → then raise ctx to 2048 and probe again.
3. **T-NC (target):** the 16-layer config above; only after both gates.
Each stage: one EXPERIMENTS.md row, verdict in WHAT_WORKS.md, and vram_probe before committing.

Risks / honest unknowns:
- CED-style sharing was proven at 552B/45T tokens; at 110M the encoder half may be too shallow to
  produce a good shared summary. Mitigation: give borrowing layers a *learned per-layer low-rank
  correction* from their own hidden state (cheap hybrid), if the pure version underperforms.
- fp16 numerics: projections of a shared hidden state could concentrate magnitude; keep the
  existing grad-scaler checks and watch grad norms (the fill-lesson from row 49).
- Extra engineering cost is small but nonzero; do not start this while SFT/eval work owns the GPU.

## 3. Cheapest first experiment (one day, no architecture change)
Before any code: quantify the actual KV/activation headroom on our ladder.
- Run `vram_probe` at ctx 512 vs 2048 for the current target config, and measure how much VRAM
  is weights+optimizer vs KV+activations. If KV is <5% of the budget (I expect it is), the
  priority order flips: **context-length wins are cheap via KV sharing only when we want ctx≥2k**;
  otherwise spend the effort on data quality (their biggest lever) and optimizer efficiency (Muon-style).
- This measurement decides whether idea #1/#2 is worth an experiment at all. Evidence before architecture.

## 4. Bottom line
DeepSeek-V4.1-Flash is a **memory-bandwidth-economics paper**, not a modeling breakthrough paper —
its intelligence gains came from data curation and scale. For a nano-SLM on a 6 GB Turing card, the
transferable gold is: (a) the CED/cross-layer KV-sharing pattern as a *contained, config-gated*
experiment once we want ctx ≥ 2k; (b) "train the efficient architecture from scratch, no warmup";
(c) obsessive data quality over architectural novelty; (d) optional Muon-family optimizer for
sample efficiency. Everything else is scale-mismatched. Recommended sequencing: measure first
(probe KV share of VRAM), then smoke-gate a 2+2 split, then pilot.
