# Distillation methods — survey (2026-09-08)

Scope: how frontier labs (Qwen, DeepSeek, DeepMind, Thinking Machines) distill large
models into small ones, and the Sept-2026 landscape of DeepSeek-V4 (Flash/Pro) and
Qwen3.8-Flash-Next training techniques — with applicability to this repo's 226M student.
Cross-references existing repo notes instead of re-explaining:
`research/c12_distillation_report.md` (strong-to-weak deep-dive), `research/c12_tier2_brief.md`
(tokenizer-mismatch blocker + Tier 2 costs), `research/qwen3_8_flash_next_research.md`
(Qwen3.8-Flash-Next architecture: GDN+QSA 3:1, GR, n-gram embedding, Muon, CPT).

Labels: **VERIFIED** = present in raw Exa payload text (`research/raw/*.json`).
**REPORTED** = single secondary source. **UNVERIFIED** = not found in any payload.

## TL;DR

- **Offline trace SFT (Kim&Rush seq-KD)** → student trains by CE on teacher-generated text; the weakest but cheapest form; this is exactly our finished C12 Tier 1 (AST 0.60→0.86).
- **DeepSeek-R1 distills** → 6 dense students (1.5B–70B, Qwen2.5/Llama bases) = plain SFT on **800k R1 traces, no KL, no RL**; R1 paper: direct distillation *outperforms* running RL on the student — **VERIFIED** ([R1 paper](https://arxiv.org/html/2501.12948v1)).
- **GKD** → on-policy KD: student samples its own sequences, teacher logits score them; on-policy + mixed data **consistently beat** fixed supervised data; reverse KL ≫ forward KL for instruction tuning; no backprop through sampling — **VERIFIED** ([GKD](https://arxiv.org/html/2306.13649v3), ICLR 2024; [HF TRL GKDTrainer](https://huggingface.co/docs/trl/main/en/gkd_trainer)).
- **DistiLLM** → skew-KL loss (stability vs plain KLD gradient explosions) + adaptive off-policy reuse of student generations; faster convergence than fwd/reverse KL — **VERIFIED** ([DistiLLM](https://arxiv.org/html/2402.03898v2), PMLR v235).
- **MiniLLM** → policy-gradient reverse-KL KD (the original "match teacher only where it matters" objective); background for GKD's divergence analysis — **UNVERIFIED** in payloads (not sampled), cite [arxiv 2306.08543](https://arxiv.org/abs/2306.08543) only as prior art.
- **Thinking-Machines on-policy distillation** → student samples, teacher grades **each token via reverse KL** (implemented as RL with per-token advantage = −KL); Qwen3-8B student hits 70% AIME'24 in **~150 steps** from a 400k-SFT-prompt checkpoint where off-policy SFT would need **~2M prompts**; **7–10× fewer gradient steps than RL**, ~1/10 RL compute — **VERIFIED** ([blog](https://thinkingmachines.ai/blog/on-policy-distillation/), [alphaXiv mirror](https://www.alphaxiv.org/abs/2605.on-policy-distillation)).
- **Qwen3 recipe (strong-to-weak)** → flagship does 4-stage SFT+RL; 0.6B–14B/30B-A3B students instead get (1) off-policy trace SFT on /think+/no_think outputs, then (2) on-policy logit-KL to Qwen3-32B/235B-A22B; report: "distillation from advanced teacher models significantly outperforms RL" — **VERIFIED** ([Qwen3 TR](https://arxiv.org/html/2505.09388v1)); recipe details cross-ref `c12_distillation_report.md` (~45k–70k traces, ~1/10 GPU-hours).
- **DeepSeek-V4 (2026-04-24)** → Pro 1.6T/49B-active, Flash 284B/13B-active, 1M ctx; CSA+HCA hybrid attention, mHC residual, Muon, >32T-token pretrain; Pro at 1M ctx = **27% FLOPs / 10% KV of V3.2**; FP4 QAT in post-training — **VERIFIED** ([V4 paper](https://arxiv.org/html/2606.19348), [DeepSeek news](https://www.deepseek.com/en/news/v4-preview/)).
- **Qwen3.8-Flash-Next (2026-08-26)** → 125B/6B-active MoE + 51B n-gram tables, Qwen4 architecture preview; all details cross-ref `qwen3_8_flash_next_research.md`. "Qwen3.8 *the Last*" / "Next-Flash" as official names: **UNVERIFIED** — payloads only say Flash-Next previews Qwen4.
- **For nano_SLMs**: only two doors are open now — (a) more trace-SFT (any teacher, any tokenizer) and (b) intra-ladder logit KD (Tier 3, `scripts/kd.py` already committed, ~3–4 h). Cross-teacher on-policy logit KD (GKD/TM-style) is **blocked by the 32k↔248,320 vocab mismatch** (`c12_tier2_brief.md`); the practical substitute is teacher-as-judge rerank (Tier 2 V1).

## Sources

| Source | URL | Date | Type |
|---|---|---|---|
| Qwen3 Technical Report | https://arxiv.org/html/2505.09388v1 | 2025-05-14 | paper (VERIFIED text) |
| "Why Qwen3 Skipped RL and Used Distillation" (temperature2) | https://temperature2.com/p/2026-08-10-did-you-know-knowledge-distillation/ | 2026-08-10 | blog analysis |
| GKD / On-Policy Distillation of LMs (Agarwal et al.) | https://arxiv.org/html/2306.13649v3 · https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf | 2023-06 / ICLR 2024 | paper (VERIFIED text) |
| HF TRL GKDTrainer docs | https://huggingface.co/docs/trl/main/en/gkd_trainer | — | docs |
| Kim & Rush, Sequence-Level KD | https://aclanthology.org/D16-1139/ | 2016 | paper |
| DistiLLM (Ko et al.) | https://arxiv.org/html/2402.03898v2 · https://proceedings.mlr.press/v235/ko24c.html | 2024-02 | paper (VERIFIED text) |
| DeepSeek-R1 paper | https://arxiv.org/html/2501.12948v1 | 2025-01-22 | paper (VERIFIED text) |
| R1-Distill family guide + HF card | https://deepseekai.guide/models/deepseek-r1-distill/ · https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 2026-04-25 | secondary (REPORTED) |
| Thinking Machines, "On-Policy Distillation" | https://thinkingmachines.ai/blog/on-policy-distillation/ · https://www.alphaxiv.org/abs/2605.on-policy-distillation | 2025-10-27 | blog (VERIFIED text) |
| Tinker distillation recipes | https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/ | — | docs |
| DeepSeek-V4 paper | https://arxiv.org/html/2606.19348 | 2026-04-26 | paper (VERIFIED text) |
| DeepSeek V4 preview announcement | https://www.deepseek.com/en/news/v4-preview/ | 2026-04-24 | announcement |
| HF blog on V4 | https://huggingface.co/blog/deepseekv4 | 2026-04-24 | blog |
| V4-Flash deep dive (local-ai-zone) | https://local-ai-zone.github.io/blog/deepseek-v4-flash-deep-dive.html | 2026-04-26 | secondary |
| Qwen3.8-Flash-Next design paper | https://arxiv.org/html/2608.30320 | 2026-08 | paper |
| Qwen3.8-Flash-Next HF + GitHub + Alibaba blog | https://huggingface.co/Qwen/Qwen3.8-Flash-Next · https://github.com/qwenlm/qwen3.8-flash-next · https://www.alibabacloud.com/blog/qwen-3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501 | 2026-08-26/27 | primary |
| 2026 landscape grounded answer | `research/raw/ans3_2026_landscape.json` | 2026-09 | Exa answer payload (model-synthesized) |
| Repo cross-refs | `research/c12_distillation_report.md`, `c12_tier2_brief.md`, `c12_tier_order_decision.md`, `c12_distillation_plan.md`, `c12_runbook.md`, `qwen3_8_flash_next_research.md` | 2026-09-06/08 | internal |

## 1. Offline (off-policy) distillation — trace SFT and sequence-level KD

- **Kim & Rush (2016)** coined sequence-level KD: train the student on *teacher-generated output sequences* (beam-searched), collapsing the ensemble's knowledge into a single small model ([ACL D16-1139](https://aclanthology.org/D16-1139/)). Modern "teacher-trace SFT" is the LLM-scale version: cross-entropy on teacher text, no logits needed, **works across tokenizers** (only decoded text passes through). **VERIFIED**.
- **DeepSeek-R1 distill recipe** ([paper](https://arxiv.org/html/2501.12948v1)): R1-Zero = pure RL, no SFT; R1 = cold-start SFT → reasoning RL → rejection-sampled new SFT data near RL convergence → more RL. For students: "direct distillation from DeepSeek-R1 outperforms applying RL on it" (tested on Qwen2.5-32B base); "reasoning patterns discovered by larger base models are crucial" — i.e. **RL on a small model can't rediscover what RL on a big model found**. **VERIFIED**.
- **The 800k corpus**: R1-Distill checkpoints (1.5B/7B/8B/14B/32B/70B; Qwen2.5 and Llama 3.x bases) are plain SFT on **800k R1 samples, no KL loss** ([HF card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B); [guide](https://deepseekai.guide/models/deepseek-r1-distill/) says "800,000 reasoning samples"). The 600k-reasoning + 200k-non-reasoning split seen in the literature is **REPORTED** (secondary source; not in our payload text).
- **Sample-count finding**: Qwen's small models needed only ~45k–70k curated traces to reach the 2B/4B/9B distill level at ~1/10 the GPU-hours of self-RL — cross-ref `c12_distillation_report.md` §1–3 (Qwen-reported; our payload confirms the two-part recipe but not those exact counts; counts were verified in the earlier C12 session).
- **Cost curve in this repo** (`c12_distillation_plan.md` §3, estimates): SFT pilot 5k×1ep ≈ 1–2 h; full 20k×2ep ≈ 6–8 h; max 75k×2ep ≈ 21–26 h. Actual Tier 1: 16,376 train + 400 val pairs (78,264 seen, 21% kept after ast-filter) ≈ 5–6 h (`c12_runbook.md` §1, §6). **PASS: AST 0.60→0.86** (user-stated, 2026-09-08).

## 2. Logit / white-box KD

Requires **shared token space** — every method below aligns teacher and student per-position logits, so it is undefined across tokenizers (our blocker: CodeLlama 32k student vs Qwen 248,320 vocab, `c12_tier2_brief.md` §3).

- **Classic KD (Hinton-style)**: forward KL, student covers the *entire support* of the teacher distribution — GKD's analysis shows this is wasteful under capacity mismatch: "forward KL requires the student to cover the entire support of the teacher token-level distribution" ([GKD §2](https://arxiv.org/html/2306.13649v3)). **VERIFIED**.
- **MiniLLM** ([arxiv 2306.08543](https://arxiv.org/abs/2306.08543)): reverse-KL policy-gradient KD — mode-seeking, so the student focuses its limited capacity on the teacher's dominant modes rather than its full distribution. Not sampled in payloads → **UNVERIFIED** details; GKD's payload independently confirms the property: reverse KL "can use student's limited capacity to focus on generating samples that are likely under the teacher." **VERIFIED** (that quote is GKD text).
- **DistiLLM** ([paper](https://arxiv.org/html/2402.03898v2), PMLR v235): two components — (1) **skew KL divergence**: mitigates gradient explosion of plain KLD, "stable gradients and minimal approximation errors, empirically leading to faster convergence and superior performance"; (2) **adaptive off-policy**: reuses student-generated outputs efficiently instead of re-sampling every step. **VERIFIED**.
- **GKD** ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)): the framework that generalizes all of the above — pick the divergence *and* the data source; λ = fraction of on-policy student-generated sequences; **no backprop through sampling**; sampling temperature γ=1. Findings from payload: "purely on-policy and mixed data distributions consistently outperform GKD variants only using a fixed supervised dataset, showing the importance of generating sequences from the student"; "in the context of instruction tuning, reverse KL performs much better than forward KL"; used JSD(0.1) on WMT, forward KL elsewhere. **VERIFIED**. Implementable today via [TRL GKDTrainer](https://huggingface.co/docs/trl/main/en/gkd_trainer).
- **Thinking Machines on-policy distillation** ([blog 2025-10-27](https://thinkingmachines.ai/blog/on-policy-distillation/)): student samples trajectories; teacher scores **each token** with reverse KL conditioned on the same prior; trained as RL with per-token advantage = −reverse KL, discount 0; teacher needs only a single forward pass while the cheap student does all sampling. Properties from payload: reverse KL is "unhackable" (low KL always means teacher-endorsed behavior), mode-seeking, zero when student ≡ teacher. Experiments (Qwen3-8B student ← Qwen3-32B teacher): 70% AIME'24 in **~150 steps** from a 400k-SFT-prompt checkpoint vs **~2M prompts** for off-policy SFT to reach the same score; **7–10× fewer gradient steps than RL**; ~1/10 the compute of RL; cumulative lifecycle compute reduction "estimated between 50x a[nd …]" (cut off in payload). They deliberately did **not** use top-k logit distillation ("could be used to further improve compute efficiency"). **VERIFIED** (numbers from alphaXiv mirror of the blog). Recipes for OpenThoughts3/DeepMath/Tulu3 in the [Tinker cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/). Shared-tokenizer requirement is *implied* by same-family setups (Qwen3-8B←Qwen3-32B), never stated — **REPORTED**.
- **Conflict with the task brief**: the "2–4× data-efficiency claim" does **not** appear in any payload; the payload numbers are 7–10× (vs RL, gradient steps), ~1/10 (vs RL, compute), and ~2M→~150 steps (vs off-policy SFT). Treated as corrected.

## 3. How Qwen3 actually trained its small models

Pipeline order ([Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)):

1. **Pretrain** the flagship (base → long-context).
2. **Flagship post-training, 4 stages**: long-CoT cold-start SFT → reasoning RL → thinking-mode RL → two stages adding non-thinking capability. **VERIFIED**.
3. **Small models skip the RL stages entirely — strong-to-weak distillation**:
   - **Phase 1, off-policy**: student trains on teacher outputs generated in **both /think and /no_think** modes — "develops basic reasoning skills and the ability to switch between different modes of thinking, laying a solid foundation for the next on-policy training phase." **VERIFIED**.
   - **Phase 2, on-policy**: student generates its own responses per prompt in either mode, "fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence." **VERIFIED**.
   - **Mode control via logit distillation**: "directly distilling the output logits from teacher models into lightweight student models can effectively enhance their performance while maintaining fine-grained control over their reasoning processes... eliminates the necessity of performing an exhaustive fine-tuning" for /think vs /no_think. The exact blending rule (max-probability blend of the two mode distributions) is documented in `c12_distillation_report.md` §3 — cross-ref, not re-derived here.
4. **Why not RL for the small ones**: "Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency" ([TR §Strong-to-Weak](https://arxiv.org/html/2505.09388v1)). Same conclusion as DeepSeek-R1 (§1) — convergent evidence across both labs. **VERIFIED**.
5. Qwen3.5/3.8 continuation: ~30k/45k/70k curated dense-CoT teacher traces per size class, same pattern (`qwen3_8_flash_next_research.md` §5.2) — **REPORTED** there.

## 4. 2026 landscape: DeepSeek-V4 (Flash/Pro) + Qwen3.8 the Last / Next-Flash

### DeepSeek-V4 — **VERIFIED** (paper + announcement, 2026-04-24/26)

- **Existence**: preview open-sourced 2026-04-24; **V4-Pro 1.6T total / 49B active**, **V4-Flash 284B total / 13B active**, both **1M-token context** ([arxiv 2606.19348](https://arxiv.org/html/2606.19348), [news](https://www.deepseek.com/en/news/v4-preview/), [HF blog](https://huggingface.co/blog/deepseekv4)). "Flash" ≠ small: it is a 284B MoE — the name denotes inference economics.
- **Architecture/training** (paper abstract + text): hybrid attention **CSA (compressed KV + DeepSeek Sparse Attention) + HCA (aggressive compression, keeps dense attention)**; **mHC** (Manifold-Constrained Hyper-Connections) replacing plain residuals; **Muon** optimizer "for faster convergence and greater training stability"; **>32T-token pretrain**; DeepSeekMoE retained with minor changes; MTP kept. At 1M ctx, V4-Pro = **27% single-token FLOPs and 10% KV cache vs V3.2** (a second payload passage gives "10% FLOPs / 7% KV" — likely the Flash numbers; model attribution ambiguous in the extract). **VERIFIED** numbers, **REPORTED** attribution for the second pair.
- **Post-training**: "comprehensive post-training pipeline"; **FP4 quantization-aware training** for MoE expert weights and the indexer QK path during post-training; heterogeneous on-disk KV cache; V4-Pro-Max = max-effort reasoning mode, claimed SOTA-open. **VERIFIED**.
- **"Independent cultivation of domain-specific experts, followed by unified model consolidation via on-policy distillation"** — **VERIFIED 2026-09-08** (`research/raw/ver7_v4_posttrain.json`; paper quote): "The post-training pipeline of DeepSeek-V4 series features a two-stage paradigm: the independent cultivation of domain-specific experts, followed by unified model consolidation via on-policy distillation"; the unified model is the student minimizing reverse-KL against the specialist teachers. Resolves open question 1.
- MIT-licensed weights — **REPORTED** ([local-ai-zone](https://local-ai-zone.github.io/blog/deepseek-v4-flash-deep-dive.html)).

### Qwen3.8 ("the Last") / Next-Flash

- **Exists as Qwen3.8-Flash-Next**, released 2026-08-26/27, "early preview of the architecture used in Qwen4" ([HF](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [GitHub](https://github.com/qwenlm/qwen3.8-flash-next), [Alibaba Cloud blog](https://www.alibabacloud.com/blog/qwen-3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501), [design paper](https://arxiv.org/html/2608.30320)). **VERIFIED**. Full architecture/training breakdown is already in `qwen3_8_flash_next_research.md` (GDN+QSA 3:1 hybrid, Gated Residual 4-branch, 51B n-gram embedding tables host-offloaded, Muon+AdamW split, MTP, 262k→1M YaRN ctx; family timeline incl. Qwen3.8-Max 2.4T-A95B 2026-08). Do not duplicate here.
- **Distillation-relevant nugget from that note**: the QSA indexer is itself a **learned MQA indexer distilled from the full-attention teacher in a 2-stage CPT** (`qwen3_8_flash_next_research.md` §6.4) — frontier labs even distill *inside the architecture* (sparse-attention student ← dense teacher), not just across model sizes.
- Training-cost claim: Flash-Next costs "roughly a ninth" of its predecessor's training ([DataCamp](https://www.datacamp.com/blog/qwen3-8-flash-next)) — **REPORTED**.
- **"Qwen3.8 'the Last'" / "Next-Flash" as official names**: not found in any payload. **UNVERIFIED** — treat as community nicknames until a primary source surfaces.

## 5. Applicability to nano_SLMs

Ground truth: student T = 226M GQA (16L, h1024, 16Q/4KV), CodeLlama-32k tokenizer, ctx 1024, fp16, 6 GB (RTX 3000, sm_75), pretrain on ~164M CSN-python tokens done, Tier-1 trace SFT done (AST 0.60→0.86). Local teacher: Qwen3.5-0.8B (248,320 vocab → **logit KL undefined**, `c12_tier2_brief.md`). Auto-resume contract: any new train script must resume crash-safe like `train.py`/`kd.py`.

| Method | Verdict | Cost / notes |
|---|---|---|
| Trace SFT (seq-KD class; R1/Qwen Phase-1 style) | **Feasible now — already done once** | Scale-up options: SFT-max 75k×2ep ≈ 21–26 h (plan est.); actual run was 16.4k pairs ≈ 5–6 h. SFT-v2 corpus (minimax3, 19,252 pairs) already staged per git log 2026-09-07. Any new teacher works — tokenizer mismatch is irrelevant for text-level traces. |
| Intra-ladder logit KD, P→S (Tier 3; classic fwd-KL, shared 32k vocab) | **Feasible now** | `scripts/kd.py` + A/B configs committed (git log), CPU rig PASS; ~3–4 h GPU, VRAM fits (teacher P 0.2 GB + student + logits ≈ 268 MB fp16 for batch 8; plan §5). **Suggested extension not in any doc: T (226M) as teacher for S/P** — 32k-tokenized, ~0.45 GB fp16, stronger teacher than P; one `kd.py` config away. |
| DistiLLM (skew-KL + adaptive off-policy) | **Feasible with work — intra-ladder only** | ≈1 day of work on `kd.py` (skew-ε KL + off-policy buffer). Payoff at 100M→12M unproven; the stability claim matters at fp16 where plain-KLD logits can overflow. |
| GKD on-policy KD with Qwen3.5-0.8B teacher | **Not feasible (blocked)** | Per-position student-logits vs teacher-logits needs identical vocab; 32k vs 248,320 (`c12_tier2_brief.md` §3). No cheap fix: no off-the-shelf CodeLlama-32k teacher of useful quality is known — verify StarCoder-family vocabs before assuming otherwise. |
| Teacher-as-judge rerank (Tier 2 V1; the practical on-policy surrogate) | **Feasible with work** | Judge = Qwen3.5-0.8B (local, downloaded; 8-bit bnb verified on sm_75). Impl 4–6 h CPU + ~1 h GPU pilot; judge ≈ 1–2 s/candidate at 1.5B fp16, cheaper at 0.8B (`c12_tier2_brief.md` §4–5; `c12_tier_order_decision.md` judge contract). This is trace-SFT data *quality* distillation — the off-policy half of Qwen's recipe with our own taste filter. |
| TM/GKD full on-policy loop (student samples → teacher per-token reverse KL → RL-style update) | **Not feasible now** | VRAM might even fit (0.8B fp16 teacher ≈ 1.6 GB + 226M student + optimizer), but the tokenizer blocker kills teacher scoring, and a sampling+update loop is a new non-auto-resuming failure surface on the single 6 GB GPU. Revisit only if a same-tokenizer teacher appears (then T→S on-policy is the smallest viable experiment). |
| Frontier architecture borrowing (CSA/HCA, mHC, GDN/QSA, Muon, FP4) | **Not applicable at 226M** | Cross-ref `qwen3_8_flash_next_research.md` §6.1–6.2: Gated-Residual and Muon are the nano-feasible picks; hybrid sparse attention and n-gram tables are not. FP8/FP4 QAT irrelevant (no fp16-successor support on Turing). |

Bottom line: the frontier convergence is **trace SFT first, logit-KL on-policy second, RL never for small students**. Our ladder already has door #1 open (Tier 1 PASS) and door #2 plumbed (Tier 3, `scripts/kd.py`); the only missing piece is a same-tokenizer teacher for the on-policy half, which T-as-teacher can supply cheaply.

## 6. Pointers from the parallel-runs comparison (2026-09-08) — verification round

Verified 2026-09-08 against `research/raw/ver7_*.json` (12/12 tasks in
`ver_tasks_round7.json`); labels per repo convention:
- **QAD — quantization-aware distillation**: **VERIFIED** (`ver7_qad.json`)
  = arXiv 2607.04244, "Quantize the Target, Quantize the Drafter" (nota-ai,
  3rd place, Efficient Qwen Competition @ ICML 2026; HF card
  nota-ai/Qwen3.5-4B-QAD-W4A16): QAD "trains the INT4 target model to follow
  the original BF16 model's distribution, recovering the accuracy that
  post-training quantization (PTQ) loses", original quantization grid
  retained; plus a two-stage block-diffusion drafter for speculative
  decoding. Claude's ~8k-step / 8-bit-AdamW details: still REPORTED (not in
  our payload). Not runnable here (no INT4 path on sm_75); the transferable
  principle stands — teacher-regenerated/fresh-target distillation data
  beats static distill sets.
- **DistilQwen2.5**: **VERIFIED** (`ver7_distilqwen.json`) = arXiv
  2504.15027, Alibaba Cloud, ACL 2025 industry track — "Industrial
  Practices of Training Distilled Open Lightweight Language Models";
  corroborates the Qwen3 recipe in §3 at production scale.
- **ToS/compliance** (Claude's black-box file; a policy rule, not a paper
  claim): cloud providers may prohibit training on their outputs — carried
  into adoption_plan Track D and note 02 §5 before any cloud-teacher
  decision.

## Open questions

1. **Is DeepSeek-V4's post-training consolidation-by-on-policy-distillation real?** ANSWERED 2026-09-08: VERIFIED via `ver7_v4_posttrain.json` (paper quote in §4).
2. **Does any ≥100M-quality teacher share a 32k-class vocab with CodeLlama?** (StarCoder/SantaCoder vocabs unverified.) If yes, true cross-model on-policy KD unblocks; if no, T-as-teacher is the ceiling.
3. **Would Qwen-style /think+/no_think logit blending help a code-only student?** Our Tier 1 trained on a single response mode; the Qwen control signal comes from blending two mode distributions — likely low-value for a non-mode-switching nano model, but it is the one Phase-1→Phase-2 refinement we never priced.
4. **Do DistiLLM's skew-KL stability benefits survive fp16 + 32k vocab at our scale?** Needs one 1–2 h A/B on the existing `kd.py` rig (kd_s_t1 vs skew-KL variant).
5. **What does the cut-off TM number say?** Payload ends "cumulative compute reduction ... between 50x a…" — pull the blog's efficiency paragraph to pin the 50–100× claim before quoting it anywhere.
