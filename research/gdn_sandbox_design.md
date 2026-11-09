# Milestone G1 — S-Scale Hybrid-Attention Sandbox Design (Gated DeltaNet + Gated Attention)

**Status:** DESIGN — pre-implementation. User-gated per TASKS row 16 / row 6 successor.
**Date:** 2026-09-06 · **Author:** am-research subagent.
**Method:** Exa web research (root `exa_search.py` / batch `scripts/exa_research.py`);
7 raw payloads under `research/raw/milestone_g1_*.json` + 3 deep-dive contents fetches.
**Prior work extended:** `research/qwen3_8_flash_next_research.md` — architecture and
training-recipe context already documented; this doc is the *S-scale validation plan*
that turns that research into a runnable experiment. No duplication of the existing
prior research (lineage, Qwen3-Next/3.8 model card, distillation path).

> **Hard constraints carried forward** (PLAN §1 A3/A4, ENVIRONMENT, MEMORY row 1):
> single Quadro RTX 4000 (Turing sm_75); fp16 **only** (no bf16); **no triton**;
> **no flash-attn** (PyTorch SDPA only); venv = uv-managed CPython 3.12.9, torch
> 2.14.0+cu126, transformers 5.16.1. Nothing in this design touches those.

---

## 1. Verified facts (architecture + why hybrid helps)

### 1.1 Gated DeltaNet — the mechanism (chunked delta rule)

The recurrence, per head, with data-dependent decay `alpha_t` and write gate
`beta_t`, is the published delta rule (Yang et al., NeurIPS 2024, "Parallelizing
Linear Transformers with the Delta Rule over Sequence Length", arXiv:2406.06484;
ICLR 2025 Gated DeltaNet extension arXiv:2412.06464):

```
S~_t = alpha_t * S_{t-1}            # decay the previous state
v~_t = v_t - S~_t^T k_t              # erase the old contribution of k_t
S_t  = S~_t + beta_t * k_t v~_t^T    # write the new (gated, residual) value
y_t  = S_t^T q_t                     # read with the current query
```

`alpha_t = exp(-exp(A) * softplus(W_a x + b_a))` (per-head scalar); `beta_t =
sigmoid(W_b x)`. Qwen3-Next additionally puts a short depthwise causal
convolution (kernel 4) in front of q/k and adds a **sigmoid output gate**
(`o = W_o [ sigmoid(W_z x) ⊙ RMSNorm(y) ]`) plus **zero-centered RMSNorm** —
three stability details the Qwen team ships (`research/qwen3_8_flash_next_research.md` §3.1).

FLA's reference naive `delta_rule_recurrence` and `delta_rule_chunkwise` are the
textbook implementations (`research/raw/milestone_g1_megatron_gdn.json`;
`fla/ops/delta_rule/naive.py`). **Both cast q/k/v/beta to fp32 internally and
cast the output back to input dtype** — the canonical fp16 stability workaround
that we follow (§4).

### 1.2 Gated Attention — one sigmoid gate kills the attention sink

Gated Attention is a minimal change: multiply the softmax-attention output by a
per-head sigmoid gate. From lilting.ch
(`research/raw/milestone_g1_hybrid_ratio_details.json`):
*"first-token attention drops from 46.7% to 4.8%."* The gate gives the head a
"don't use this output" path, suppressing the attention-sink artefact. The HF
modeling source confirms the SiLU-gated RMSNorm and per-head structure
(`research/raw/milestone_g1_gated_attention.json`, `transformers/.../qwen3_next/modeling_qwen3_next.py`).

### 1.3 Hybrid ratio — 3:1, three independent sources agree

Raschka (hybrid gallery, `milestone_g1_qwen3_next_hybrid_ratio.json`):
*"Qwen3-Next 80B-A3B has 48 layers arranged as 12 repetitions of the 3:1 pattern.
Three Gated DeltaNet blocks followed by one Gated Attention. 36 recurrent, 12 softmax."*
Same pattern restated by lilting channel (`milestone_g1_hybrid_ratio_details.json`)
and Raschka ch04/08 (`milestone_g1_deltanet_scratch_ch04.json`):
*"three linear-attention style layers, one full-attention layer, then repeat."*

### 1.4 Why the hybrid helps at our scale

* **Memory:** the GDN state is fixed-size per head (a `[d_k, d_v]` matrix; no KV
  cache). Only every 4th attention layer keeps a KV cache. Net win on 6 GB VRAM:
  lower peak memory, cheaper decode, in exchange for one full-attention layer
  re-enabling precise token-level retrieval (`milestone_g1_qwen3_next_hybrid_ratio.json`).
* **Formal expressivity:** "Provably Shorter Scratchpads in Hybrid
  DeltaNet-Attention Decoders" (arXiv:2605.16640) proves **O(1) CoT scratchpads**
  for parity-conditioned retrieval under the Qwen-style hybrid, where pure GDN
  has no constant solution and pure attention needs polynomial scratchpad —
  hybrid is strictly more capable than either pure mechanism.
* **S-scale validity:** the GDN ablation (28-layer 25B-A3B probes) finds
  hybrid > SWA > full attention at 9-benchmark average (53.81 / 51.15 / 49.87;
  `qwen3_8_flash_next_research.md` §3.1). Relative ordering is what matters at S;
  absolute numbers will be tiny.

---

## 2. Concrete implementation plan

### 2.1 Config-driven arch flag — `model.arch: gqa | gdn_hybrid`

Add a single top-level key to `src/model.py` and `configs/smoke.yaml` siblings.
Backward compatible: default `gqa` keeps the existing pipeline unchanged.

```yaml
# configs/gdn_smoke.yaml (proposed; created at impl time)
name: gdn_smoke
tokenizer: { name: codellama/CodeLlama-7b-hf, vocab_size: 32768 }
model:
  arch: gdn_hybrid            # NEW — {gqa, gdn_hybrid}; default gqa
  layers: 4                   # 4 layers → 3 GDN + 1 GA in a single 3:1 block
  hidden: 256
  heads: 4                    # GA Q heads
  kv_heads: 2                 # GA KV heads (GQA inside GA only)
  linear_heads: 4             # GDN value heads
  head_dim_gdn: 64            # d_k = d_v at S (4 heads * 64 = 256)
  ffn: 1024
  ctx: 256
  dropout: 0.0
  tie_embeddings: true
  hybrid_step: 1              # every Nth layer is GA; Qwen default 1 (3:1)
  use_short_conv: true
  use_sigmoid_gate: true      # GDN + GA output gates (recommended)
  use_zero_centered_rmsnorm: true
  gdn_state_dtype: fp32       # required for fp16 stability (§4)
```

`src/model.py` keeps its current `build_model(cfg, vocab_size)` signature. When
`cfg["model"].get("arch") == "gdn_hybrid"`, dispatch to a new
`build_gdn_hybrid_model(cfg, vocab_size)` in the same file. The function returns
`AutoModelForCausalLM.from_config(config)` for arch=gqa (unchanged); for
arch=gdn_hybrid it instantiates a small custom `Qwen3NextStyleForCausalLM` whose
layer layout is **imported from `transformers.models.qwen3_next`** (already in
this venv — sidesteps writing GDN from scratch; the "reference impl already
installed" point from prior research §0).

### 2.2 Class layout (inside `src/model.py`)

```
src/model.py
├── build_config(cfg, vocab_size)            # unchanged (LlamaConfig branch)
├── build_gdn_hybrid_config(cfg, vocab_size) # NEW
├── build_model(cfg, vocab_size)             # dispatches on cfg["model"]["arch"]
└── build_gdn_hybrid_model(cfg, vocab_size)  # NEW
    ├── GatedDeltaNetLayer(cfg, layer_idx)
    │   ├── q_proj, k_proj, v_proj, b_proj, a_proj, g_proj, o_proj
    │   ├── q_conv1d / k_conv1d / v_conv1d    # depthwise causal conv k=4
    │   ├── o_norm (zero-centered RMSNormGated)
    │   └── chunkwise_gated_delta_rule(q,k,v,β, chunk_size=32, fp32_state=True)
    ├── GatedAttentionLayer(cfg, layer_idx)   # GQA Llama attn + per-head sigmoid gate
    ├── HybridBlockSelector(layer_idx, cfg)   # ((layer_idx+1) % hybrid_step)==0 → GA
    └── RMSNorm pre-norm + SwiGLU FFN (identical to GQA baseline)
```

For S=4 layers with `hybrid_step=1`, layer order is `[GDN, GDN, GDN, GA]`
(three GDNs then one GA — Qwen convention; HF config exposes
`decoder_sparse_step=1`, see `milestone_g1_gated_attention.json`).

### 2.3 Where the gating / decay parameters live

* `b_proj`, `a_proj` (per-head scalars; 4-d at S): `nn.Linear(d, linear_heads, bias=False)`.
  Init small — `b` biases β near 0.5; `a` gives decay ≈ 0.9 at step 0.
* `g_proj` (GDN output gate, head-dim per head): `nn.Linear(d, 2*linear_heads*head_dim_gdn, bias=False)`.
* `o_gate` (GA per-head): `sigmoid(W_gate(x))` of shape `[B, H, 1]`, applied per
  head before `o_proj` (mirrors Qwen3-Next `RMSNormGated` pattern).
* All are plain `nn.Parameter`s in their `Layer` modules — saved/loaded by HF's
  standard checkpoint machinery. No special state-dict plumbing.

### 2.4 The chunkwise delta-rule path (the only non-trivial piece)

Three options, ranked by **fidelity to our constraints** (no triton, no
flash-attn, sm_75, fp16):

| Option | Source | Constraint fit | Verdict |
|---|---|---|---|
| (a) Megatron `torch_chunk_gated_delta_rule` (`deterministic_mode=True`) | `milestone_g1_megatron_gdn.json` | **Pure PyTorch, no triton** | **Recommended** — Megatron keeps this *because* "FLA is not deterministic". |
| (b) FLA `chunk_gated_delta_rule` kernels | `fla/ops/gated_delta_rule/` | Triton-bound | Faster but kernels are Triton. Naive `delta_rule_recurrence` is O(L) sequential per token — impractical beyond seq 1k. |
| (c) Hand-rolled chunked matmul | n/a | Pure PyTorch | WY representation + causal masking is non-trivial; getting it wrong silently degrades quality. Rejected. |

Decision: port Megatron's `torch_chunk_gated_delta_rule` into `src/model.py`
(or a small new `src/gdn.py`). A few hundred lines, deterministic, no Triton,
tracks the published chunkwise algorithm (WY representation, Householder
products, `l % chunk_size == 0`).

### 2.5 Param math — S-scale (~12.3M, param-matched to GQA)

Baseline `configs/smoke.yaml` (GQA) trunk: 4 × (4·256² + 3·256·1024) ≈ 7.34M
plus tied embedding 8.39M ≈ **15.7M** (PLAN §3 quotes "~12M"; we target 12.3M
with ±15% slack to absorb GDN projection sizing).

For a GDN layer at S: `q_proj 65,536 + k_proj 65,536 + v_proj 131,072 + b_proj
1,024 + a_proj 1,024 + g_proj 131,072 + o_proj 131,072 + 3×(4·64) depthwise
conv 768 + SwiGLU 786,432` ≈ **1.31M**. For a GA layer at S:
`q/k/v/o attn 262,144 + per-head gate 1,024 + SwiGLU 786,432` ≈ **1.05M**.

4-layer hybrid: `3 × 1.31M + 1 × 1.05M = 4.98M` trunk + tied emb 8.39M ≈
**~13.4M total** — within the 10.5–14.2M S budget. The 2× v expansion is what
Qwen3-Next does (`expand_v: 2`, `fla/layers/gated_deltanet.py`,
`milestone_g1_deltanet_pytorch_impl.json`); if we hit the upper bound, drop to
`expand_v: 1.5` (saves ~32K/layer, marginal quality loss per ch04/08).

### 2.6 New / changed files at implementation time (post-M3, not now)

* `configs/gdn_smoke.yaml` — new (sibling of `smoke.yaml`).
* `configs/gdn_smoke_ab.yaml` — variant with `arch: gqa` (param-matched control).
* `src/model.py` — add the `gdn_hybrid` branch (backward-compatible dispatch).
* `src/gdn.py` *(new)* — port `torch_chunk_gated_delta_rule` + the GDN + GA sub-modules.
* `scripts/sanity_check.py`, `scripts/train.py`, `scripts/eval.py` — **no change**.

---

## 3. S-scale validation protocol (the gate to P-scale)

The protocol uses the existing pipeline unmodified — that's the value: if G1
doesn't flow through `sanity_check → train → eval` exactly like M0/M2/M3, it's
not a valid sandbox.

### 3.1 Three acceptance gates

| Gate | Script | Time | PASS criterion |
|---|---|---|---|
| **G1.a** sanity_check | `& .venv/Scripts/python.exe scripts/sanity_check.py --config configs/gdn_smoke.yaml` | <30 s GPU | All 4 sub-checks PASS (`model_build`, `fwd_bwd` finite loss, `gpu` peak <1.5 GB at ctx 256/bs 2, `tokenizer` roundtrip). Forward+backward through **mixed GDN+GA** stack must succeed in fp16 on sm_75 with the fp32-state workaround. |
| **G1.b** short train | `scripts/train.py --config configs/gdn_smoke.yaml` (200 steps, ctx 256, bs 1 × accum 32) | 3–5 min wall (matches M0) | Loss monotonically ↓; TensorBoard `train/loss` exists; first eval at step 50 returns finite `eval/loss`; val ppl ≤ 4.0 at step 200 (random-init ~12+; sanity expects real signal). |
| **G1.c** eval + control | `scripts/eval.py --config configs/gdn_smoke.yaml --ckpt runs/gdn_smoke/final` AND same on `gdn_smoke_ab.yaml` (`arch: gqa` control) | <1 min per run | val_loss + perplexity recorded; 3 generation prompts produce locally syntactic Python; **Δval_ppl(gdn_hybrid − gqa) ≤ 0** — hybrid not worse than matched GQA on this small data (qualitative at S; absolutes not comparable to P). |

### 3.2 Acceptance to earn a P-scale trial

A **PASS** in all three gates does not by itself earn a P-scale trial. The
trial is earned when:

1. G1.a/b/c all PASS on the S config; **and**
2. The hybrid logs **comparable loss trajectories** to the param-matched GQA
   control (`gdn_smoke_ab.yaml`) at S — not a win, just no regression; **and**
3. **The fp32-state memory cost** (§4) on S stays below ~10% of total VRAM at
   ctx 256, extrapolated to ≤ ~15% at P (ctx 512, ~8× larger state); **and**
4. The **kill/resume drill** on the S config still PASSes (auto-resume contract
   from PLAN §5.3 must not regress when the model class changes).

P-scale requires fresh user approval and a fresh VRAM probe — neither is in
scope here.

### 3.3 What G1 is NOT trying to prove

Not a quality win — S is too small; perplexity is noisy. Not a long-context
result — ctx 256 only. Not "Qwen3-Next at nano scale" — that is a distillation
story (`qwen3_8_flash_next_research.md` §6.3); G1 is the architecture sandbox
that has to pass **before** any P-scale experiment.

---

## 4. fp16 / sm_75 risk register

| # | Risk | Source | Mitigation |
|---|---|---|---|
| R1 | **Delta-rule state underflow/overflow in fp16** — state `S ∈ R^{B,H,d_k,d_v}` accumulates across the whole sequence; fp16 can vanish (decay) or spike (write) | `milestone_g1_fp16_state_stability.json`; triangular-inversion paper (arXiv:2605.21325): *"due to its high-sensitivity to numerical errors, [it] can significantly deteriorate end-to-end model accuracy if not carefully implemented"* | **fp32 state, fp16 IO.** All GDN ops cast q/k/v/β to fp32 internally and cast output back to input dtype (FLA naive pattern). Config flag `gdn_state_dtype: fp32` (default). Cost at S ≈ 256 KB fp32 activations ≈ 2% of 8 GB. |
| R2 | **Triangular inversion loss in chunkwise kernel** (`(I − tri(βKKᵀ))⁻¹` factor) — documented fp16 instability hot spot | arXiv:2605.21325 (`milestone_g1_fp16_state_stability.json`) — paper specifically studies fp32, fp16, bf16 | Run the inverse **inside the chunkwise kernel in fp32** (`torch.linalg.solve_triangular`); cast chunk outputs back to fp16 only at the very end. Megatron's port already does this. |
| R3 | **No triton chunked kernel ⇒ slower than FLA** | Constraint | Acceptable: S runs in 3–5 min wall even at 2–3× chunk-kernel slowdown (M0 smoke already finishes in minutes; this doc budgets for it). If S exceeds 8 min wall, fall back to chunk_size=16 (smaller inverses). |
| R4 | **Autograd through the recurrent state** keeps an extra graph edge per token | Standard PyTorch | Already true for the GA path; GDN adds another. Mitigation: keep `gradient_checkpointing=True` (already default at S per `configs/smoke.yaml` line 51). No new code. |
| R5 | **Attention sink on the GA layers** (the original motivation for Qwen's GA) — first-token attention dominance | lilting.ch (`milestone_g1_hybrid_ratio_details.json`) | The GA sub-module **must** include the per-head sigmoid output gate. Without it we lose the only stability benefit GA brings. Config flag defaults `True`. |
| R6 | **No bf16 fallback** if fp16 turns out to be unworkable for GDN | Constraint (PLAN A3) | Acceptable: fp32-state (R1) is the answer. No bf16 on this hardware. If we hit NaN losses that fp32 state does not fix, the experiment fails — that's a valid answer, not a process failure. |
| R7 | **Quantised `RMSNormGated` / fused kernels from HF Hub** auto-downloaded by `@use_kernel_forward_from_hub` | `Qwen3NextRMSNormGated` decorator in transformers source (`milestone_g1_gated_attention.json`) | **Disable the decorator** in the local port: copy the class but drop the decorator line. PyTorch fallback (the function body) is the same math (plain fp32 RMSNorm + SiLU(gate)); the Hub kernel is a fused-Triton optimisation we don't want. |
| R8 | **State carries across eval/train boundaries incorrectly** if GDN forgets `output_final_state=False` | SDPA cache analogue | `use_cache=False` (already the case during training, `src/model.py` line 28); during short-ctx eval we ignore the state and recompute. |

### 4.1 Approximate VRAM cost at S

fp16 weights (12.3M × 2 B) ≈ 24 MB; fp16 AdamW first-moment state ≈ 98 MB; fp16
grads ≈ 24 MB; fp32 GDN state ≈ 1 MB; activations (grad_ckpt, ctx 256, bs 2) ≈
200 MB. **TOTAL ≈ ~350 MB at S.** Comfortable margin below the 6 GB limit.

---

## 5. Post-M3 implementation timing (auto-resume contract)

This task is **never started while the live M3 target run is running** (M3 is
row 1 in TASKS, `in_progress` since 2026-09-06 10:56; a `python` PID at >2 GB
VRAM is the live trainer). The contract is enforced by HANDOFF §8.3's existing
convention: **no design-time code touches `src/model.py` while a run holds the
GPU**. Pre-flight uses the same gate as `scripts/c12_preflight.py`.

Sequence (chronological):

1. **Now** — this document, user-gated per TASKS row 16.
2. **When M3 finishes** (row 1 → done): post-M3 exit per HANDOFF §8.2 (eval
   final, commit metrics, free the GPU).
3. **User approves** G1 implementation start (this doc earned the row; starting
   it is a separate user call).
4. **Implementation window** opens: write `src/gdn.py`, add the `gdn_hybrid`
   branch to `src/model.py`, add `configs/gdn_smoke.yaml` + control
   `configs/gdn_smoke_ab.yaml`. Commit.
5. **Validation** — G1.a/b/c gates (§3), each a separate background job; no
   overlap; kill/resume drill rerun.
6. **Decision** — PASS → row-16 deliverable met; row-6 ("Hybrid architecture"
   upgrade path) unlocked for a separate user decision at P-scale.

**Auto-resume preserved end-to-end:** `train.py` reads the new
`Qwen3NextStyleForCausalLM` state-dict the same way it reads `LlamaForCausalLM`
today (unchanged `src/model.py` surface) — resume contract doesn't need
re-proving, only a quick drill on the new arch.

---

## 6. Sources (raw Exa payloads)

Batch files (reproducible): `research/raw/milestone_g1_tasks.json` (7 tasks) +
`research/raw/milestone_g1_deep_dive_tasks.json` (3 deep-dive fetches). All
payloads land in `research/raw/milestone_g1_*.json`:

* `deltanet_pytorch_impl.json` · NVlabs GatedDeltaNet, Megatron
  `gated_delta_net.py`, FLA `gated_deltanet.py`
* `delta_rule_paper.json` · Yang et al. NeurIPS 2024 + arXiv:2406.06484
* `qwen3_next_hybrid_ratio.json` · Raschka hybrid gallery, arXiv:2605.16640
  (parity scratchpad), Raschka ch04/08
* `gated_attention.json` · HF Qwen3-Next doc + transformers source
  (RMSNormGated, GA, decoder_sparse_step)
* `fp16_state_stability.json` · arXiv:2510.04212 (low-prec failures),
  arXiv:2605.21325 (triangular inversion fp16), FLA `naive.py`
* `mamba_turing_sm75.json` · mamba-scan-lite (PyTorch, no compile),
  mamba3-minimal (pure PyTorch)
* `hybrid_design_answer.json` · Exa-grounded recommendation
* Deep-dive contents: `megatron_gdn.json`, `hybrid_ratio_details.json`,
  `deltanet_scratch_ch04.json`

**Reproduction** (safe to re-run; total Exa cost observed ~$0.011):

```powershell
& .\.venv\Scripts\python.exe scripts\exa_research.py --tasks research\raw\milestone_g1_tasks.json
& .\.venv\Scripts\python.exe scripts\exa_research.py --tasks research\raw\milestone_g1_deep_dive_tasks.json
```
