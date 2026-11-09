# Distill + context survey — synthesis report (2026-09-08)

Answers the 2026-09-08 user research request. Raw Exa payloads live in
`research/raw/` (slugs `kd*`, `stg*`, `ctx*`, `chk*`, `ans*` from
`exa_tasks_round6.json`, 22 tasks); per-topic notes live in `notes/`.
Nothing here is committed to a run — every adoption item is user-gated.

## 1) Answers to the six questions

**Q1 — How do Qwen/others distill, and why distilled > pretrained?**
Full answer: `notes/01_distillation_methods.md`.
- Converged frontier recipe for small students: (1) offline trace SFT
  (teacher outputs as CE targets), (2) on-policy logit KD on student
  samples, (3) **RL never for small students** — stated independently by
  Qwen3 TR ("distillation significantly outperforms RL in performance and
  training efficiency") and DeepSeek-R1 (direct distill > RL on the
  student).
- R1-Distill = plain SFT on ~800k teacher traces, zero KL/RL; Qwen small
  models used only ~45–70k *curated* traces. Quality beats quantity.
- On-policy KD (GKD/MiniLLM/DistiLLM): student samples + reverse KL ≫
  forward KL for instruction tuning; Thinking Machines' on-policy
  distillation ≈ 7–10× fewer gradient steps than RL (Qwen3-8B hit 70%
  AIME'24 in ~150 steps from a 400k-SFT ckpt). Requires **shared
  tokenizer** (hard blocker for our Qwen3.5-0.8B teacher).
- Our own measured evidence: C12 Tier 3 KD (same-tokenizer P→S) beat the
  baseline at every eval point (2.4755 vs 2.6201 @2k steps) and passed the
  "≥ baseline at 1/3 steps" gate — the ~1/10 pretrain-cost claim transfers
  to this hardware (TASKS row 20).

**Q2 — What is fed at each stage, and in what order/why?**
Full answer: `notes/03_training_stages_data.md`.
- Canonical map: bulk pretrain → mid-train HQ pivot (stable II) →
  high-quality anneal/decay → long-context extension → SFT → RL →
  distillation. Bulk→HQ two-phase is now standard (OLMo 2, Phi-4).
- Verified small-budget recipes: SmolLM2 (1.7B/11T): 90/10 web-code bulk →
  75/20/5 → 74/16/10 → anneal w/ synthetic textbooks; ctx 2k→8k over 75B
  tokens. SmolLM3: mixtures chosen by 3B-scale 50–100B-token ablations —
  the directly transferable method for us.
- **Must-fix finding (conflicts with our plan):** high-quality-last under
  LR decay-to-zero is quantifiably suboptimal (arXiv 2511.18903; TREC).
  Fixes: end LR ≈ 1/3 peak, or constant LR + checkpoint averaging (CMA,
  +1.64% avg — free via our existing 3-ckpt rotation).
- MTP (multi-token prediction): densifies signal for data-constrained
  regimes; depth 1–2, λ 0.05–0.2, heads discarded at inference. No sub-1B
  ablation exists → 500-step probe only.

**Q3 — Latest models (Sept 2026): DeepSeek-V4 / "Queen 3.8"?**
Full answer: `notes/01_distillation_methods.md` §4 +
`research/qwen3_8_flash_next_research.md`.
- **DeepSeek-V4 VERIFIED**: Pro 1.6T total/49B active, Flash 284B/13B, 1M
  ctx, CSA+HCA sparse attention, mHC, Muon, >32T tokens; Pro ≈ 27% FLOPs /
  10% KV of V3.2; FP4 QAT in post-training. "Consolidation via on-policy
  distillation" = REPORTED only.
- **Qwen3.8-Flash-Next VERIFIED as a Qwen4 preview** (GDN+QSA 3:1 hybrid,
  gated residual, N-gram embedding, Muon, CPT — already detailed in the
  repo doc; even its QSA indexer is itself distilled, sparse student ←
  dense teacher).
- **"Qwen3.8 the Last" / "3.8 Next-Flash" as exact names: UNVERIFIED** —
  appear nowhere in 8+ payloads. Treat as hearsay until a primary source
  shows up.

**Q4 — Distilling from cloud models (text outputs only, no logits)?**
Full answer: `notes/02_api_only_distillation.md`.
- Yes — every text-only channel is **tokenizer-agnostic** (DSS, Orca,
  R1-Distill, Zephyr all cross-tokenizers). Taxonomy:
  - **SeqKD** (SFT on teacher outputs): the de facto standard (Alpaca,
    Vicuna, WizardLM, R1-Distill). Off-policy flaw: student never sees its
    own errors → motivates on-policy variants. We already did this (T1 +
    SFT v2; ast 0.98/0.96 e1).
  - **CoT/rationale distillation**: Distilling Step-by-Step — 770M T5
    outperforms few-shot PaLM-540B using only 80% of a benchmark's
    examples. **SCoTD kills the "226M too small" objection**: students as
    small as 125M–1.3B *train* on teacher rationales successfully.
  - **On-policy black-box**: GAD (discriminator-as-reward, Nov 2025) made
    it real but needs a comparable discriminator + adversarial training →
    NOT feasible on our single GPU. **SODA (Apr 2026): static one-time
    snapshot of student rollouts + teacher preference pairs beats/matches
    15/16 with 10× speed, 27% less GPU memory than GAD** — this is nearly
    our Tier 2 V1 judge-rerank with preference pairs instead of
    winner-only SFT.
  - **DPO from scores, not logits**: Zephyr-7B DPO'd purely on GPT-4
    numeric judgments (REPORTED).
- Cost reality: judge-rerank 2k prompts × 8 candidates ≈ ~$1.6 at
  small-model API tiers (ESTIMATE — pricing unverified); local Qwen3.5-0.8B
  judge = $0 but a 4–9 h GPU window under the never-co-run rule.

**Q5/Q6 — Context >1024 on 6 GB VRAM; in-model memory; compression.**
Full answer: `notes/04_context_extension_memory.md`.
- **KV cache is NOT our bottleneck (COMPUTED)**: 16 KiB/token fp16
  (2·16L·4kv·64·2B) → 512 MiB even at 32k ctx vs ~452 MB weights. Real
  blockers: positional extrapolation + O(n²) prefill.
- Ranked feasibility for our 226M GQA @1024:
  1. **Dynamic-NTK eval** — zero fine-tune, config/eval-only, cheap first
     experiment.
  2. **StreamingLLM** (window + attention sinks) — no fine-tune, stable to
     4M tokens, but dropped tokens unrecoverable (continuation, not
     retrieval). Caveat: sink evidence is 7B+; our from-scratch model may
     lack the artifact (cheap test).
  3. **YaRN fine-tune @ ctx 2048** (fits ~5.4 GB computed) then
     extrapolate to 4k+; published "10× less tokens, 2.5× less steps" vs
     PI; ~400 steps at 7B scale (REPORTED). Ctx-4096 training OOMs unless
     8-bit Adam (row 11 default) frees ~1.8 GB.
  4. **Learned memory (RMT-lite memory tokens / Infini-attention)** — the
     only family that *trains* long-context behavior at 1024-chunk VRAM
     AND yields a checkpointable state tensor (0.5–4.2 MB) that can be
     saved/restored across independent inference calls = the "remember
     part of it after a context swap" ask. Infini-attention's delta-rule
     memory ≈ GDN math — same fp32-state code path as
     `gdn_sandbox_design.md` §4; natural G1 successor. Honest note: no
     source tests cross-session persistence; engineering hypothesis.
  5. Soft-token compression (ICAE/gisting/500xCompressor): 7B-scale
     full-model fine-tune — research-project-only; parked.
- KV quantization saves only ~256 MiB @32k for us — low priority.

## 2) Prioritized adoption plan (all user-gated)

| # | Track | What | Cost | Repo hook |
|---|---|---|---|---|
| A | Context quick wins | Dynamic-NTK + StreamingLLM eval probes on runs/sft_v2_e1/final (eval-only, no training) | ~1–2 h GPU, config-only | new milestone; note 04 §5 |
| B | YaRN extend | YaRN fine-tune at ctx 2048 (~300–1k steps, 8-bit Adam) → eval @ 4k | ~2–4 h GPU | new milestone; note 04 §1/§5 |
| C | KD follow-up | Tier-3-style KD with **T (226M) as teacher** (same CodeLlama tokenizer, ~0.45 GB fp16) + DistiLLM skew-KL in kd.py | ~3–4 h GPU | extends TASKS row 20; note 01 §2/§5 |
| D | Tier 2 V1 → V2b | Judge-rerank with Qwen3.5-0.8B (frozen contract, row 14) then SODA-style preference-pair rerank-SFT | 4–6 h impl + ~1 h pilot GPU; +$ if cloud judge | TASKS rows 5/7; note 02 §4 |
| E | Data/stage revision | Milestone D mix split into bulk ~80M / mid-train ~27M (Evol 10–12%) / anneal ~27M HQ; **LR fix: end ≈1/3 peak or constant-LR + CMA**; optional MTP depth-1 500-step probe | next pretrain window | TASKS row 13; note 03 §4 |
| F | LoRA-SFT v2 variant | Same corpus, frozen base via the row-10 LoRA hook (0.72 GB measured); predicted ~0-2% forgetting | ~1-2 h GPU | analysis #2 (training_analysis §5.2); plan §F |
| G | SFT v3 mixed-corpus | Evol + minimax3 + generic-phrasing slice — anti-repetition-loop medicine; 1 epoch, pilot-first | CPU prep + ~1 h GPU | analysis #3 (§5.3); plan §G |

*(Cloud CoT corpus — old F — stays optional/unselected: few USD + CPU-side
prep, see note 02 §3; revive whenever a cloud teacher budget exists.)*

Integrated 2026-09-08 with `research/training_analysis_2026-09-08.md`:
standing three-surface gates (ast + CSN forgetting guard + mini_eval),
1-epoch-default rule, pilot-first measurement, knee-stop for the next
pretrain — see `adoption_plan.md` §Standing gates.

2026-09-08 addendum (post-COMPARISON): `adoption_plan.md` gained
Track H (agent-side memory, zero training) and data-prep/judge/pointer
bullets on Tracks D/E/G from the parallel-runs comparison
(`COMPARISON.md` §7). This §2 table is the original snapshot and is
left unchanged.

## 3) Honest gaps
- "Qwen3.8 the Last / Next-Flash" naming: UNVERIFIED (see Q3).
- No sub-1B MTP ablation in payloads (probe before adopting).
- No published evidence for cross-call memory persistence (our use case).
- GAD discriminator and GKD-style on-policy loops: not feasible on 6 GB.
- Exa payload `kd4` sample counts for ≤1B students absent — 27.5–52k
  anchors are chat-SFT scale, not code-rerank scale.
- LongLoRA/LongRoPE, gisting, MLKV/YOCO had no payload coverage — left as
  UNVERIFIED pointers in note 04 rather than filled from memory.
