# Training stages & data — survey (2026-09-08)

Scope: what data goes into each training stage and why — pretrain mix, mid-training/continued
pretrain, high-quality anneal/decay, long-context extension, SFT, RL, distillation, MTP — with
small-model recipes as evidence. Payloads: `research/raw/stg1..stg5_*.json` (+ skim `s4`/`s8`;
Qwen3.8-specific findings live in `research/qwen3_8_flash_next_research.md`, not duplicated).
Labels: VERIFIED = in local payload text; REPORTED = secondary source inside a payload;
UNVERIFIED = my estimate or absent from payloads.

## TL;DR

- The modern pipeline is: **bulk pretrain (web-heavy) → mid-train stable phase(s) (pivot to
  knowledge-dense data) → high-quality anneal/decay → (long-context extension) → SFT → RL →
  distillation**; bulk→HQ two-phase is now standard (OLMo 2, Phi-4, LongCat-Flash) (VERIFIED,
  arxiv.org/html/2511.18903).
- **High-quality data belongs late** — but only if the LR is still high when it arrives. Aggressive
  LR decay "wastes" late HQ data; fixes: moderate decay (~end LR ≈ 1/3 peak) or constant-LR +
  checkpoint weight-averaging (CMA): +1.64% avg over random ordering, 1.5B/30B tok (VERIFIED,
  arxiv.org/html/2511.18903).
- Schedule innovations with numbers: TREC (arxiv.org/pdf/2509.25380) says place HQ data at
  **low points of the training re-evaluation curve**, not blindly at the end (predictable in-run
  from AdamW's implicit EMA coefficients); Drop-Stable-Rampup (arxiv.org/html/2605.25698v1) gives
  +1.70 acc vs WSD / +2.98 vs cosine (GSM8K +4.23, MATH +2.80) on 15B MoE over 108B midtrain
  tokens by dropping batch size at the quality transition, then ramping at the end (VERIFIED).
- SmolLM2 (1.7B, ~11T tok ≈ 2 epochs, ~$250K) and SmolLM3 (3B, 11.2T): same shape — 2 stable
  stages → HQ pivot → upsampled decay (code 10→24%, math 3→14% ramps; SmolLM2 anneal = 1T HQ:
  58/24/14/4 web/code/math/synthetic-textbook; mixtures picked via 3B ablations on 50–100B
  tokens) (VERIFIED, arxiv.org/html/2502.02737 + huggingface.co/blog/smollm3).
- Qwen3 (36T, 119 langs): S1 >30T general → S2 ~5T higher-quality STEM/code/reasoning/synthetic
  with **accelerated LR decay** → S3 long-context 4K→32K (75% long + 25% medium; ABF RoPE
  10,000→1,000,000, YaRN, DCA) (VERIFIED, arxiv.org/html/2505.09388).
- DeepSeek-V3 (14.8T): MTP loss λ=0.3 first 10T → 0.1 last 4.8T; warmup 2K steps→2.2e-4 then
  constant LR; post-pretrain YaRN 2×1000 steps 4K→32K→128K; post-train SFT+RL, then distill
  R1's CoT verification/reflection patterns into V3 (VERIFIED, arxiv.org/pdf/2412.19437v2).
- MTP densifies training signal and is explicitly recommended for **data-constrained** pretraining;
  depth 1–2, loss scale 0.05–0.2 (default 0.1), heads discarded at inference (VERIFIED,
  docs.nvidia.com/nemo/megatron-bridge/latest/training/multi-token-prediction.html).
- For small models, **distillation beats RL**: Qwen3's 0.6B–14B students are logit-distilled from
  235B/32B teachers; distilled models "significantly outperform" RL-trained ones with ~10× less
  training time (VERIFIED payload text; 10× is REPORTED via debuggercafe.com/qwen3-unified-models-for-thinking-and-non-thinking).
- "MiniCPM-style stable→decay→anneal": MiniCPM/WSD itself is **absent from local payloads**
  (UNVERIFIED here; cf. arXiv 2404.06395); the payload-verified analogues are SmolLM2/SmolLM3's
  stable-stable-decay + anneal stages and the mid-training survey (arxiv.org/html/2510.06826).

## Sources

| Source | URL | Date | Type | Payload |
|---|---|---|---|---|
| Mid-Training of LLMs: A Survey | https://arxiv.org/html/2510.06826 | n/a | survey | stg1[3] |
| How LR Decay Wastes Your Best Data (curriculum) | https://arxiv.org/html/2511.18903 | 2025-11-24 | paper | stg1[1,4] |
| DiReCT: annealing sample selection | https://arxiv.org/html/2605.31175 | n/a | paper | stg1[0] |
| Optimal Data Scheduling / Drop-Stable-Rampup | https://arxiv.org/html/2605.25698v1 | n/a | paper | stg1[2] |
| TREC data placement (111M–3.9B models) | https://arxiv.org/pdf/2509.25380 | n/a | paper | stg1[5] |
| Two-Phase Pretraining | https://arxiv.org/html/2412.15285v1 | n/a | paper | stg1[6] |
| SmolLM2 paper | https://arxiv.org/html/2502.02737 | 2025-02-04 | paper | stg2[1,7] |
| SmolLM3 blog + smollm README + DeepWiki + Playbook | https://huggingface.co/blog/smollm3 · https://github.com/huggingface/smollm/blob/main/text/pretraining/README.md · https://deepwiki.com/huggingface/smollm/2.2-smollm3-pretraining · https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook | 2025-07-08 / 2025-08-22 / 2025-10-30 | blog+docs | stg2[0,2,3,4,5] |
| Qwen3 Technical Report + blog | https://arxiv.org/html/2505.09388 · https://qwenlm.github.io/blog/qwen3/ | 2025-05-14 / 2025-04-29 | paper+blog | stg3[0,1,2] |
| Qwen3 mechanics article (sec. details) | https://www.besthub.dev/articles/how-qwen3-achieves-multi-stage-pretraining-long-context-and-thought-controlled-rl-45b82021330c | 2025-05-13 | article | stg3[3] |
| Qwen3 strong-to-weak distill blog | https://debuggercafe.com/qwen3-unified-models-for-thinking-and-non-thinking/ | n/a | blog | s8[5] |
| DeepSeek-V3 Technical Report | https://arxiv.org/pdf/2412.19437v2 | 2024-12 (arXiv id) | paper | stg4[1,2] |
| DeepSeek-V3 repo/HF card + KD DeepWiki | https://github.com/deepseek-ai/DeepSeek-V3 · https://huggingface.co/deepseek-ai/DeepSeek-V3 · https://deepwiki.com/deepseek-ai/DeepSeek-V3/3.3-knowledge-distillation | n/a | repo+docs | stg4[0,3,4,5] |
| Megatron Bridge/Core MTP docs | https://docs.nvidia.com/nemo/megatron-bridge/latest/training/multi-token-prediction.html · https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/multi_token_prediction.html | n/a | docs | stg5[0,1] |
| DeepSeek Explained pt.4 (MTP) + PyImageSearch MTP | https://pub.towardsai.net/deepseek-explained-part-4-multi-token-prediction-d905344f2c2b · https://pyimagesearch.com/2026/03/30/autoregressive-model-limits-and-multi-token-prediction-in-deepseek-v3/ | 2025-04-22 / 2026-03-30 | articles | stg5[2,3] |

## 1. The canonical stage map

**S0. Bulk pretrain (stable phase).**
- Purpose: general language/world/code foundation from massive diverse tokens.
- Data: web-crawl dominated + code + math seeds (SmolLM2 S1 90% web / 10% code over 6T tok;
  SmolLM3 S1 85/12/3; Qwen3 S1 >30T of ~36T, 119 langs — mixes in §3) (VERIFIED).
- LR: high, long plateau — SmolLM3 2e-4 constant then decay (README: batch 2.36M tok, seq 4096);
  DeepSeek-V3 warmup 2K steps to 2.2e-4 then constant (VERIFIED).
- Why: bulk cheap tokens buy general capability; the mid-training survey attributes late-stage
  noise/diminishing returns to noisy web tokens (gradient-noise-scale framing)
  (arxiv.org/html/2510.06826).

**S1. Mid-training / continued pretrain (second stable phase).**
- Purpose: pivot mix to knowledge-dense, reasoning-relevant data before any final decay.
- Data: upsampled STEM/code/reasoning/synthetic. Qwen3 S2 = ~5T higher-quality tokens
  (STEM, coding, reasoning, synthetic), seq still 4096, "accelerate the learning rate decay
  during this stage" (VERIFIED). SmolLM3 S2 adds Stack-Edu/FineMath4+/InfiWebMath4+/MegaMath
  (VERIFIED). Two-phase bulk→HQ adopted by OLMo 2, Phi-4, LongCat-Flash (VERIFIED,
  arxiv.org/html/2511.18903); Llama 3 reports in-run mix changes (REPORTED, stg1[7] payload).
- LR: still high; Qwen3 decays faster in S2. 2511.18903 argues the pivot works best while the
  LR has NOT collapsed. Why: cheap broad tokens stop paying; targeted data consolidates skills
  the base was too noisy to learn cleanly (survey framing).

**S2. Anneal / high-quality decay phase.**
- Purpose: final convergence + capability consolidation on the best data available (DiReCT:
  "two key levers: learning rate decay and targeted data selection"; standard heuristic =
  upsample reasoning-heavy math/code as in Llama 3 / Phi-4) (VERIFIED, arxiv.org/html/2605.31175).
- Data + size: highest-quality, often partially synthetic; ~7–10% of total pretrain tokens in
  both Smol recipes (1T/11T; 1.1T/11.2T — VERIFIED arithmetic). SmolLM2 anneal = 1T HQ tokens:
  58% web / 24% code / 14% math / 4% synthetic textbooks (VERIFIED via survey); SmolLM3 decay =
  63% web/24% code/13% math + OpenMathInstruct-2 + OpenMathReason… (VERIFIED blog + DeepWiki);
  SmolLM3 decay = linear over 522,000 steps starting step 4,198,001 (VERIFIED, DeepWiki).
- Why: low LR + clean data = final capability lock-in; DiReCT frames it as exploitation of the
  low-curvature subspace after stable-phase exploration.

**S3. Long-context extension.**
- Purpose: extend usable context without regressing short-context quality. Data: long-document
  corpora mixed with the base mix.
- Recipes: Qwen3 4K→32K (75% of data 16K–32K long + 25% medium; ABF RoPE 10,000→1,000,000 + YaRN
  + DCA; the "four-fold effective capacity" phrasing is REPORTED via besthub.dev); DeepSeek-V3
  YaRN 2 phases × 1000 steps 4K→32K→128K; SmolLM2 2k→8k over 75B tokens (40% long docs + 60%
  stage-4 mix); SmolLM3 4K→64K progressive, NoPE+YaRN to 128k (all VERIFIED).
- Cost signal: SmolLM2 spent 75B/11T ≈ 0.7% of budget on ctx extension (VERIFIED arithmetic).

**S4. SFT.**
- Qwen3 post-train 4 stages: (1) long-CoT cold-start SFT — data with verified reference answers/
  test cases, 2-stage filtering (drop unverifiable queries; drop queries answerable without CoT,
  judged by Qwen2.5-72B-Instruct), N candidates then manual quality filter; (3) thinking-mode
  fusion — unified SFT over long-CoT + instruction data generated by the stage-2 (RL) model
  (VERIFIED report + blog). DeepSeek-V3: SFT then RL, post-training = 0.1M GPU-hours (VERIFIED).
  Small-model SFT data: SmolTalk, built by HF because existing instruct sets were "too small
  and/or low-quality" (VERIFIED, arxiv.org/html/2502.02737).

**S5. RL (optional).**
- Qwen3: stage-2 reasoning RL with rule-based rewards (~3,995 query-verifier pairs REPORTED via
  besthub.dev; entropy control), then stage-4 general RL (VERIFIED report). For small models RL
  is the wrong tool: the Qwen3 report states distilled small models "significantly outperforms
  reinforcement learning in performance and training efficiency" (VERIFIED); ~10× training-time
  reduction is REPORTED (debuggercafe.com).

**S6. Distillation.**
- Strong-to-weak: Qwen3 0.6B/1.7B/4B/8B/14B students distilled from 235B-A22B/32B teacher
  logits, off-policy → on-policy progression (VERIFIED payload; s8[5]). DeepSeek-V3: distill
  R1-series long-CoT (verification + reflection patterns) into V3 after SFT+RL, balancing
  accuracy vs generation length (VERIFIED report + deepwiki KD page).
- This matches the repo's own cheap-ladder KD result (research/training_analysis_2026-09-08.md
  §3.5: "KD beat from-scratch at every budget point").

**S+ (training-signal booster). MTP (multi-token prediction).**
- What: auxiliary heads predict tokens 2..D ahead; each depth = shared embedding + projection +
  one Transformer block + shared output head; sequential causal chain (keeps coherence vs
  parallel variants); trained on ground-truth intermediates → no error accumulation; heads
  discarded at inference (VERIFIED, docs.nvidia.com/megatron-core/0.15.0 + stg5[2,3]).
- Why: "densifies the training signals and may improve data efficiency"; explicitly recommended
  for "data-constrained scenarios where maximizing learning from limited data is critical";
  also a speculative-decoding foundation (VERIFIED, docs.nvidia.com/nemo/megatron-bridge/...).
- Knobs: depth typical 1–2; loss scaling default 0.1, range 0.05–0.2 (VERIFIED, Megatron docs);
  DeepSeek-V3 used λ=0.3 for the first 10T tokens and 0.1 for the last 4.8T (VERIFIED).

## 2. Evidence per lab

**Qwen3** (arxiv.org/html/2505.09388; qwenlm.github.io/blog/qwen3):
- ~36T tokens (Qwen2.5: 18T), 119 languages; corpus expanded by OCR-ing PDFs with Qwen2.5-VL,
  refined by Qwen2.5, plus synthetic from Qwen2.5-Math/Qwen2.5-Coder (VERIFIED).
- Stage structure, post-train 4-stage pipeline, and distill-vs-RL verdict as in §1 (VERIFIED).

**DeepSeek-V3** (arxiv.org/pdf/2412.19437v2):
- 671B MoE / 37B active; 14.8T tokens; "no irrecoverable loss spikes"; 2.664M H800 GPU-hours
  pretrain + 0.1M for all post-training stages (VERIFIED).
- MTP λ 0.3→0.1 at the 10T-token mark; LR warmup 2K steps to 2.2e-4 then constant; YaRN context
  extension 2×1000 steps 4K→32K→128K; post-train SFT+RL then R1 distillation (VERIFIED).

**SmolLM2** (arxiv.org/html/2502.02737, 1.7B):
- ~11T tokens ≈ **2 epochs** on the collected corpus; multi-stage **manual rebalancing**
  (mixing rates updated between stages from observed evals); ~1e23 FLOPs ≈ $250K (VERIFIED).
- Four principles: (1) performance-driven interventions (mix adapted to eval bottlenecks);
  (2) upsample HQ math/code during annealing, reserving FineMath/Stack-Edu for final stages;
  (3) introduce medium datasets (OWM, InfiMM-WebMath, Stack-Edu) mid-training;
  (4) avoid excessive data repetition (citing Muennighoff et al.) (VERIFIED).
- Stage mixes in §1/§3. Built FineMath, Stack-Edu, SmolTalk where existing data was too
  small/low-quality. Tokenizer (49,152) trained on 70% FineWeb-Edu / 15% Cosmopedia-v2 /
  8% OpenWebMath / 5% StarCoderData / 2% StackOverflow (VERIFIED).

**SmolLM3** (huggingface.co/blog/smollm3, 3B):
- 11.2T tokens, 3 stages, 384×H100 × 24 days; mixture ratios set via **3B-model ablations on
  50B–100B tokens** (VERIFIED) — the directly transferable method for us.
- Stage mixes, decay window (linear, 522k steps from step 4,198,001), batch 2.36M tok, LR 2e-4,
  AdamW (beta1 0.9…) as in §1 (VERIFIED). Ctx 4K→64K progressive (NoPE + YaRN to 128k)
  (VERIFIED README).

**MiniCPM-style anneal findings:**
- MiniCPM's WSD 3-phase (warmup→stable→decay) is NOT in the local payloads — treat as
  UNVERIFIED-here (cf. arXiv 2404.06395). Payload-verified equivalents: SmolLM3's explicit
  "Stable/Stable/Decay" phases, SmolLM2's anneal stage, DiReCT's stable+annealing two-stage
  framing, and the mid-training survey's "iterative LR annealing passes" practice.
- The anneal-phase design space is being formalized: DiReCT selects anneal samples via
  validation-Hessian spectral geometry instead of content heuristics (VERIFIED abstract);
  TREC shows HQ-data placement should follow AdamW-EMA-predictable low points, and even tests
  CPT on a 3.9B model trained to 234 tokens/param (900B tokens) then continued +18B tokens
  (~5 TPP) (VERIFIED).

## 3. Curriculum principles with numbers

1. **Quality-vs-quantity over training time.** Ascending-quality curricula only pay off while
   the LR is high: under constant LR curriculum > random shuffling; under standard decay the
   advantage vanishes; fixes give +1.64% avg over random (1.5B/30B-tok validation), and under
   multi-phase pretraining reordering alone gives +1.2% avg / >2% on core benchmarks
   (VERIFIED, arxiv.org/html/2511.18903).
2. **LR-decay coupling.** The more aggressive the decay (longer decay phase, lower end LR), the
   more the curriculum benefit disappears; a moderate end LR (~1/3 of peak, in their setting)
   restores it; the stronger fix is constant LR + weighted checkpoint averaging (CMA)
   (VERIFIED, same paper). TREC independently flags "HQ data at the very end under decay-to-zero"
   as suboptimal and recommends TREC-low-point placement, predictable from AdamW's implicit EMA
   coefficients (VERIFIED, arxiv.org/pdf/2509.25380). Drop-Stable-Rampup adds the batch-size
   lever: drop batch at the quality transition, hold, ramp up at the end (+1.70 vs WSD, +2.98
   vs cosine; GSM8K +4.23 / MATH +2.80 on 15B MoE / 108B tok) (VERIFIED,
   arxiv.org/html/2605.25698v1).
3. **Repetition vs diversity.** SmolLM2's 11T ≈ 2 epochs, with principle (4) "avoiding excessive
   data repetition" (Muennighoff et al.) — repetition handled by re-mixing between stages rather
   than re-passing the same data (VERIFIED); matches the repo lesson "first epoch buys nearly
   everything; later epochs buy forgetting" (training_analysis_2026-09-08.md §4.2).
4. **Code+math ratios ramp with stage.** Verified small-model ramps: SmolLM2 code 10%→20%→16%→24%,
   math 0→5%→10%→14% (+4% synthetic textbooks in anneal; S1 web = 60% FineWeb-Edu + 40% DCLM);
   SmolLM3 code 12%→15%→24%, math 3%→10%→13% (VERIFIED). Web share never drops below ~58% in
   these general models — our code-specialized mix has no web leg at all, which is the biggest
   recipe divergence from the evidence (see §4).
5. **Specialized datasets are created, not found.** SmolLM2 built FineMath/Stack-Edu/SmolTalk
   because existing options were too small/low-quality; SmolLM3 ablated mixtures at 3B on
   50B–100B tokens before committing 11.2T (VERIFIED) — the pilot-then-scale method.
6. **Scale caveat.** One curriculum-dynamics analysis (stg1[7]) notes differences narrow "by 410M"
   at matched compute (REPORTED, fragment context); TREC covers 111M–3.9B (VERIFIED), so
   mid-training/anneal benefits are attested down to our scale.

## 4. Applicability to nano_SLMs

State of play: M3 base = 226.5M params (16 layers, hidden 1024, 16H/4KV GQA, RoPE θ=10k, tied
embeddings, vocab 32,768), trained **164M tokens @ ctx 1024** (0.73 tok/param — far below
Chinchilla 20×; deep in Megatron's "data-constrained" regime). Next run = Milestone D: ~134M
tokens, all-code (30% the-stack-smol python / 55% starcoderdata python / 10% CSN / 5%
Evol-Instruct), ctx 1024, cosine LR 4e-4→~0, ~4,100–5,400 steps ≈ 15–20 h
(research/pretrain_mix_proposal.md §2; configs/target.yaml).

**A revised 3-stage plan on the same D budget (~134M, no new tokens, same wall-clock):**

| Stage | Tokens | Mix (mapped from pretrain_mix_proposal.md) | LR |
|---|---|---|---|
| 1. Bulk stable | ~80M (60%) | D mix as proposed: starcoderdata 55% + stack-smol 30% + CSN 10% + Evol 5% | 4e-4 constant |
| 2. Mid-train stable II | ~27M (20%) | starcoderdata continues; **Evol 5%→10–12%** (SFT pre-bake upsample); CSN 10%; optional +2–3% math sliver (OpenWebMath — *new* source, proposal delta) | 4e-4 constant |
| 3. Anneal/decay | ~27M (20%) | curated HQ upsample: stack-smol (10k curated rows) + Evol 10–15% + CSN 10–15%, HQ code ≈ 70–75% | decay here only |- Notes: stage sizes follow SmolLM3's shape (bulk ≈ 70%, mid ≈ 18%, decay ≈ 10%) but shift more
  mass into decay because 134M tokens is only ~1.2 epochs over the D corpus, and the anneal
  stage is where the SFT-format pre-bake (Evol) does its work.
- **LR coupling is the critical change.** Our cosine→~0 conflicts with placing HQ data late
  (2511.18903: curriculum benefit dies under aggressive decay; end LR ≈ 1/3 peak "in their
  setting"). Two cheap fixes: (a) WSD-style — hold 4e-4 through stages 1–2, decay only in stage
  3 to a *moderate* end (~1.3e-4); (b) CMA — constant LR throughout + weighted average of the
  final ~3 checkpoints; our `save_steps: 100` / `save_total_limit: 3` rotation already produces
  the averaging candidates at zero extra disk. (b) is the more payload-faithful fix and also
  smooths the thermal-pace noise our 6GB card injects.
- **Long-context extension:** optional final mini-stage 1024→2048 over ~1–2M tokens (SmolLM2
  spent ~0.7% of budget on ctx extension). VRAM at seq 1024 = 4.24 GB alloc / 4.4 GB reserved
  (target.yaml note) — a 2048 probe must run alone per the never-co-run rule. Low priority:
  our eval surfaces (CSN val, mini_eval) are 1024-tiled. Skip unless repo-context tasks become
  a goal.
- **MTP feasibility (226M, fp16, ctx 1024, 6 GB):** depth-1 MTP = 1 extra Transformer block
  (hidden 1024 → ~12–13M params ≈ ~25 MB fp16) + projection + shared head; VRAM headroom
  (4.24/6 GB) absorbs it; fp16 has no bf16 dependency in the MTP loss path (VERIFIED
  architecture; **our step-time/VRAM numbers UNVERIFIED — estimate +10–15% step time per
  depth**). We are the textbook "maximize learning from limited data" case (Megatron docs).
  Probe: Milestone-B-style 500-step A/B, depth 1, λ=0.1 (Megatron default; DeepSeek used 0.3
  then 0.1), heads dropped at eval/inference — zero inference-cost change. Evidence at 226M
  is thin: TREC is curriculum-only; no small-model MTP ablation in payloads. Treat as
  candidate, not default.
- **SFT lane unchanged, reinforced:** evidence says for small models distill > RL (Qwen3
  strong-to-weak; repo lesson 5 "distill, don't pretrain"). SmolTalk is a concrete candidate for
  the generic-instruct slice of mixed-corpus SFT v3 (analysis §5.3). Keep 1-epoch SFT + CSN
  forgetting guard + mini_eval gate.
- Costs: 3-stage split = same 134M tokens ⇒ same ~15–20 h; overhead = 3 tokenize passes (CPU,
  minutes) + config plumbing; math sliver ≈ 3M tokens ≈ small download vs 34.7 GB free; MTP
  probe ≈ 500 steps ≈ ~1–2 h. Budget check: 134M @ 32,768 tok/step = ~4,100 steps
  (VERIFIED arithmetic, proposal §2.3).
- **What we are NOT doing (evidence-backed):** no web-crawl leg (code-specialized by design —
  but note we then deviate from every verified recipe, which keeps ≥58% web: a deliberate,
  flagged choice), no RL stage (Qwen3: distill beats RL for small models), no decay-to-zero
  cosine with HQ data late unless we adopt fix (a) or (b) above.

## 5. Pointer from the parallel-runs comparison (2026-09-08) — VERIFIED 2026-09-08

- **MiniPLM** (`research/raw/ver7_miniplm.json`): **VERIFIED** = arXiv
  2410.17215, ICLR 2025 (Gu, Zhou, Meng, Zhou, Huang; code
  github.com/thu-coai/miniplm). Distinct from MiniLLM (note 01). Abstract
  (payload text): "a KD framework for pre-training LMs by refining the
  training data distribution with the teacher LM's knowledge"; offline
  teacher inference (KD for multiple students "without adding training
  costs"); "operates solely on the training corpus, enabling KD across
  model families"; effectiveness lever = teacher-enhanced "training data
  difficulty and diversity".
- **Why it matters here**: the corpus-only, cross-model-family property
  means data-side KD is NOT blocked by our 32k↔248k tokenizer wall (unlike
  logit KD) — a Track E candidate: teacher-refined curation of the
  mid-train/anneal streams. Keep exact benchmark deltas unquoted until the
  paper body is fetched (abstract payload only).

## 6. Open questions

1. **Does 3-stage beat uniform at 134M tokens / 226M params?** Uniform D-mix vs staged mix is
   a 2-arm question (~15–20 h each). Cheaper first probe: staged recipe on the S-scale 12.3M
   model (~2–3 h) vs the M0/M2 pattern — does SmolLM3's "ablate at 3B on 50–100B tok" method
   survive at 12M?
2. **Moderate-decay vs constant-LR+CMA:** is end-LR ~1.3e-4 vs ~0 separable under our 3-surface
   gate (ast / CSN guard / mini_eval credit), or is the 1.64%-scale effect below our eval noise?
3. **TREC/AdamW-EMA logging:** log EMA coefficients per eval step (every 100 steps, cheap) — can
   we detect the anneal low-point in-run and make stage-3 onset a measured trigger instead of a
   fixed step count (arxiv.org/pdf/2509.25380 claims advance predictability)?
4. **MTP at our scale:** does depth-1 λ=0.1 improve eval loss + mini_eval credit within the
   +10–15% step-time estimate, and does the fp16 MTP head stay stable at ctx 1024? (500-step
   A/B answers both; small-model MTP evidence is absent from payloads.)
5. **Math sliver:** is +2–3% OpenWebMath worth a new download and tokenizer pass, or does
   HQ-code upsampling alone carry the gain at 0.73 tok/param? (SmolLM evidence is for web+code
   models; code-only anneal evidence is missing — the largest recipe-shaped unknown for us.)
