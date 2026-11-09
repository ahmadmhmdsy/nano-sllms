# COMPARISON — our distill_survey vs other agents' research (2026-09-08)

Compares four independent research runs answering the **same user prompt** (the
2026-09-08 distillation/context research request, quoted in the survey README):

| Producer | Location | Size | Files | Tooling evidence |
|---|---|---|---|---|
| **Ours (this repo)** | `research/distill_survey/` (REPORT + adoption_plan + 4 notes) | ~105 KB | 8 + 22 task defs | 22 Exa tasks; raw payloads saved in `research/raw/` (69 files; all kd*/stg*/ctx*/chk*/ans* slugs present — existence verified 2026-09-08) |
| **Claude** | `other_research/distill_research_claude/` | ~42 KB | 6 | Named arXiv IDs + vendor blogs per file; no raw payloads |
| **Perplexity** | `other_research/distill_research_perplexity/` | ~80 KB | 6 | 74 numbered web citations per doc; URL list included |
| **ChatGPT** | `other_research/deep-research-report_chatgpt.md` | ~12 KB | 1 | Prose; one sources paragraph, no inline links |

Purpose: judge where the other runs are **right, wrong, or ahead of us**, extract
anything worth merging, and screen their recommendations against **this repo's
hardware and contract** (226M student, 6 GB Turing sm_75, fp16-only, single GPU,
never-co-run, auto-resume). Nothing here changes a plan item — all adoption stays
user-gated.

---

## 1. Verdict table

Scale 1–5. "Fit" = usable on OUR hardware/repo without translation errors.

| Dimension | Ours | Claude | Perplexity | ChatGPT |
|---|---|---|---|---|
| Prompt coverage (all 5 asks) | **5** | 4 (misses RoPE/YaRN family) | 5 (breadth) | 3 (all named, none deep) |
| Source verifiability | **5** (payload trail + VERIFIED/REPORTED/UNVERIFIED/COMPUTED labels) | 4 (named sources, verbatim quotes, no payloads) | 2 (74 cites but unverified mix; placeholder IDs; one hard error) | 1 (named papers, no links) |
| Depth on the two frontier models | **5** | 4 (adds Engram detail) | 3 (V4 good, Qwen3.8 section wrong) | 2 |
| Fit to this repo | **5** | 3 | 2 | 2 |
| Actionability (concrete next steps) | **5** (tracks A–G w/ costs+gates) | 3 | 3 (generic code; wrong stack for us) | 1 |
| Honesty about uncertainty | **5** (gaps section; conflicts flagged) | 4 ("what NOT to expect") | 2 (confident tone over thin evidence) | 2 |
| Unique value we lack | — | **4** (Engram, QAD, agent-memory pattern) | **3.5** (compression-paper sweep, black-box zoo, production ops) | 2 (MiniPLM, Larimar/CAMELoT) |

**Bottom line:** ours is the strongest *base of record* (only run with a raw
evidence trail, uncertainty labels, and repo-integrated gating). Claude is the
strongest *single-file explainer* and contributes 2 items we genuinely lack
(Engram; the zero-training agent-side memory pattern). Perplexity is the widest
*paper sweep* but the least trustworthy per-claim (its Qwen3.8 section is wrong
and its hardware advice does not survive our fp16-only Turing constraint).
ChatGPT is a readable essay with 3 unique pointers (MiniPLM, Larimar, CAMELoT)
and little else.

---

## 2. Coverage matrix — the prompt's five asks

Legend: ● deep + verified, ◐ covered, ○ named only, ✗ missing/wrong.

| Ask | Ours | Claude | Perplexity | ChatGPT |
|---|---|---|---|---|
| 1. How Qwen/labs distill; why distilled > pretrain | ● (Qwen3 TR + R1 + GKD/DistiLLM/MiniLLM/TM-on-policy, verbatim quotes; own C12 evidence AST 0.60→0.86) | ◐ (Qwen 4-stage + strong-to-weak verbatim; best *conceptual* why; DistilQwen2.5 + QAD extras) | ◐ (taxonomy + OPD + multi-teacher; unsourced numbers like "85–95% of teacher") | ○ (narrative; MiniPLM mention) |
| 2. Stage-by-stage data & why this order | ● (canonical S0–S6+MTP map; SmolLM2/3, Qwen3, V3 mixtures; LR-decay×curriculum conflict 2511.18903; TREC; Drop-Stable-Rampup; concrete 134M-token 3-stage revision) | ◐ (Qwen S1/S2/S3 + 4-stage post-train; V4 corpus emphasis; no LR/anneal mechanics) | ◐ (V4 5-phase pipeline w/ data-mix %; expert-cultivation table w/ datasets+rewards; thin on "why this order") | ○ (generic pretrain→SFT→RLHF→distill) |
| 3. DeepSeek-V4 Flash/Pro + Qwen3.8 | ● (V4 VERIFIED vs paper+announcement; name-attribution ambiguity flagged; "the Last/Next-Flash" correctly UNVERIFIED; cross-ref repo Flash-Next doc) | ◐ (V4 specs + CSA/HCA mechanism + Engram; Flash-Next 125B/6B + 51B n-gram + no-batch-warmup; honest "no Qwen 3.8 Pro") | ◐ (V4 5-phase good; **Qwen3.8 section wrong**: reads "3.8" as ~3.8B params + FlashAttention confusion) | ○ (V4 specs OK; Qwen3.8-Max 2.4T claim; "community distilled Qwen3.8 into 9B models" uncited) |
| 4. API-only/cloud distillation | ● (taxonomy a–g: SeqKD, DSS/SCoTD, GAD, **SODA**, judge-rerank, DPO-from-scores, data-centric, GrayKD; cost arithmetic; Tier-2 V2a–V2d mapping) | ◐ (SeqKD, DSS, **GAD mechanics** — Bradley-Terry on last-token hidden state, warmup, reward hacking, **ToS/legal warning**) | ◐ (GAD + **Proxy-KD, ROPD, KED** + distillation-resistance (CMI) + production data ops) | ○ (GAD + generalities) |
| 5. Context >1024 + in-model memory | ● (COMPUTED KV math for our GQA; PI/NTK/Dynamic-NTK/YaRN; StreamingLLM; KV quant; YOCO/MLKV; ICAE/AutoCompressor/500xCompressor; RMT/Infini-attention/Titans; cross-call-persistence hypothesis; ranked feasibility) | ◐ (correct KV mechanics + **rolling-summary + RAG-lite + MemGPT/Mem0 external memory** recipe w/ token budgets; Engram-vs-external tiering; **no RoPE/YaRN/NTK at all**) | ◐ (widest compression sweep: CompLLM, AMS, AdmTree, AttnComp, ARMT, Infini-attention, InfLLM, MemInsight; **but FP8/vLLM advice inapplicable on our GPU**) | ◐ (pos-encoding tricks; **Larimar + CAMELoT** memory modules — direct hit on the "small memory part" ask) |

---

## 3. Cross-validation — where independent runs agree

Claims stated by ≥2 independent producers (confidence ↑):

| Claim | Ours | Claude | Perplexity | ChatGPT | Status |
|---|---|---|---|---|---|
| V4: Pro 1.6T/49B, Flash 284B/13B, 1M ctx, CSA+HCA, Muon, 32T+ tok | VERIFIED | ✓ | ✓ | ✓ | Solid (multi-source) |
| V4-Pro @1M ctx ≈ 27% FLOPs / 10% KV of V3.2 | VERIFIED (attribution ambiguity flagged) | ✓ (+ "effective KV ~2% via FP8" — REPORTED) | ✓ | — | Multi-source; exact KV framing varies by source |
| Qwen small models = off-policy trace SFT → on-policy logit KD; "distillation significantly outperforms RL" | VERIFIED (TR quote) | ✓ (verbatim quotes) | — | ○ | Solid |
| V4 = separate domain experts merged via on-policy distillation | REPORTED only (ans3 citation) | ✓ ("trained separate domain specialists first, then merged") | ✓ (detailed 5-phase w/ web:19/22/23/24/25/27) | ✓ (explicit two-stage) | Multi-source agreement — worth upgrading ours from REPORTED to REPORTED×3 (still not paper-verified) |
| GAD: Qwen2.5-14B student ≈ GPT-5-Chat on LMSYS (black-box) | ✓ (note 02) | ✓ | ✓ (web:32 = arXiv 2511.10643) | ✓ | Solid |
| Flash-Next ≈ 1/9 predecessor training cost | REPORTED (DataCamp) | ✓ (same) | ✓ | — | Single-origin chain — keep REPORTED |
| On-policy > off-policy for students (exposure bias) | VERIFIED (GKD/TM/SODA) | ✓ | ✓ | ✓ | Solid |

## 4. Contradictions and errors found (the other side of cross-validation)

1. **Perplexity — "Qwen 3.8 Next-Flash: ~3.8B parameters"** (`state_of_art_architectures.md` §3.1). Wrong: reads the *version number* as a size. Ours + Claude independently give Flash-Next = 125B total / 6B active + 51B n-gram tables (HF card, GitHub, arXiv 2608.30320). Perplexity also confuses "Flash" (economics) with FlashAttention. → **Its Qwen3.8 section is unusable.**
2. **Perplexity — FP8 KV cache as the Day-1 win** ("1K → 4K–8K tokens"). Inapplicable here twice over: (a) ENVIRONMENT hard fact — Turing sm_75 is fp16-only, no FP8; (b) its "Day 1" stack is `vllm serve`, which our custom from-scratch model + repo contract does not use. The 50%-KV-savings *idea* maps to our INT8 KV option, which our own math already demotes (saves ~256 MiB @32k — not the bottleneck).
3. **Claude — "your bottleneck is almost certainly KV cache VRAM"** (context README). Contradicted by our COMPUTED math (note 04 §2): 16 KiB/token → 512 MiB even @32k vs ~452 MB weights; our real blockers are positional extrapolation + O(n²) prefill/activation VRAM. Claude's statement is right for generic chat models, wrong for our geometry. This matters: it pushes Claude's whole recommendation stack toward serving-side tricks and away from the RoPE/YaRN family we actually need.
4. **Claude — no RoPE/PI/NTK/YaRN coverage at all** for ask 5 — the one family our note 04 ranks #1–2 for feasibility. Biggest single coverage hole among the three outside runs.
5. **Perplexity — placeholder citations** in `state_of_art_architectures.md` §8: "arXiv:2509.xxxxx (search 'CompLLM compression')" — unverified IDs presented as sources. Treat its compression numbers (4×, 17×, 50% KV, "85–95% of teacher") as leads, not facts, until individually verified.
6. **Perplexity — quality gates mismatched to our objective**: ROUGE-L > 0.75, hallucination-rate gates for chat products; our gate set is AST pass + CSN forgetting guard + mini_eval (three-surface standing gates). Do not import its gate list.
7. **Perplexity — timeline promises** ("Week 2: 85–90% of teacher; Month 1: 90–95%") unsourced and contradicted by the literature all four runs otherwise cite (SODA/DSS/Qwen numbers are task-dependent). Marketing tone; ignore as forecasts.
8. **ChatGPT — "community efforts have already distilled Qwen3.8 into 9B-scale models"** — uncited; consistent with nothing else in any folder. Ignore until sourced.
9. **Minor reconciliation:** ours CORRECTED the brief's assumed "2–4× data-efficiency" claim for TM on-policy distillation to the payload numbers (7–10× fewer gradient steps vs RL; ~150 steps vs ~2M prompts vs off-policy SFT). Claude/Perplexity/ChatGPT carry no contradicting number — our correction stands.

---

## 5. What the other runs have that we lack (merge candidates)

### From Claude — highest-value unique items
1. **Engram deep-dive** (deepseek-v4.md): DeepSeek's conditional-memory research — hashed n-gram lookup table, tokenizer compression (−23% effective vocab), multi-head hashing, context-gated insertion, deterministic addressing → host-RAM offload with <3% overhead, optimal 75–80% compute / 20–25% memory capacity split. Directly relevant to our "small memory part in the model" ask AND to the GDN sandbox direction (state-based, not attention-based recall). Ours never mentions Engram. **Action: add a pointer + summary to note 04 §4 as a parametric-memory sibling of RMT/Infini-attention/Titans (REPORTED — not in our payloads).**
2. **Zero-training agent-side memory pattern** (context README, Trick 2+3): rolling-summary window (150–250 summary tokens + 750–850 raw) + external JSON/SQLite/vector store with fact extraction + retrieval injection. This is the *only* proposal across all four runs that answers the user's "remember after we wipe the context" ask **today, with zero model changes** — ours only covers in-model memory (checkpointable state tensors), which needs new training code. **Action: candidate new cheap track (H) in adoption_plan — inference-harness experiment, no GPU training, complements Track A (eval-only).** Honest caveat from our side applies: it is orchestration memory, not model memory; fidelity is summarization-limited.
3. **Quantization-Aware Distillation (QAD)** (arXiv 2607.04244, Qwen3.5-4B): distill BF16 teacher → INT4-initialized student, frozen quant scales, 8-bit AdamW, ~8k steps. Not runnable now (no INT4 pipeline on sm_75, needs same-arch teacher), but the *teacher-regenerated fresh data > static data* principle it embodies corroborates our trace-SFT plans.
4. **Flash-Next training details** we hadn't recorded: **no batch-size warmup** (justified via refitted scaling laws) and **Muon+AdamW split by weight category**. Both are cheap-to-think-about levers for the next pretrain config (Track E). REPORTED (IntuitionLabs/Qwen sources).
5. **GAD implementation caveats** (README1): discriminator on last-token hidden state + Bradley-Terry loss, warmup phases required, reward-hacking risk, **API ToS legality of distillation** — the ToS point is absent from our note 02 and belongs there (user asked about cloud teachers; compliance is part of the answer).
6. V4 infra notes: batch-invariant deterministic kernels; tensor-level activation checkpointing. We already run gradient checkpointing; record as "already have the transferable piece."

### From Perplexity — verify-then-merge leads
1. **Compression-paper sweep beyond our ICAE family** (all need verification before any number is quoted): CompLLM (segmented compression, frozen base, ~20-token segments → Concept Embeddings), AdmTree (semantic-tree gist allocation), AttnComp (attention-guided RAG compression), AMS (region-quota KV eviction), TurboQuant (3.5-bit KV), ARMT (associative recurrent memory, ~65k ctx), InfLLM (training-free external memory), Compressive Transformer. Our note 04 §3/§4 covers ICAE/AutoCompressor/500x/RMT/Infini/Titans; this list roughly doubles the compression design space. **Action: add an UNVERIFIED-pointers block to note 04 (repo convention — list, don't quote numbers).**
2. **Black-box methods we don't have**: Proxy-KD (train a white-box proxy from text+scores, then soft-label KD from the proxy — could break our tokenizer wall *if* the proxy shares the student vocab: worth a design note), ROPD (rubric-scored distillation — maps onto our judge-rerank V1 as a scoring-schema upgrade), KED (teacher explanations as auxiliary targets — we already do rationale traces; KED adds explanation-of-explanation framing), IOA (pedagogical identify→organize→adapt curriculum).
3. **MiniMax-M1**: lightning (linear) attention at frontier scale — corroborates the GDN/hybrid direction already chosen in `gdn_sandbox_design.md`; record as corroborating datapoint only.
4. **Production data ops** (absent from our notes as explicit checklist): MinHash + semantic dedup, mix 10–50% human/real data to prevent model collapse, progressive rollout with quality gates, decontamination vs eval sets. Most map onto gates we already run, but the **dedup + human-data-mix items are missing from our SFT v3 / Tier-2 data plans**. **Action: two bullet adds to adoption_plan Track D/G data prep (CPU-only, free).**

### From ChatGPT — three pointers worth one grep each
1. **MiniPLM** (Gu et al. 2024 — *distinct from MiniLLM*, which we do cite): KD into the *pretraining data distribution*; claims better downstream accuracy at lower compute than vanilla pretraining. Directly on-topic for our Track E next-pretrain design. **Verify via arXiv, then add to note 03 §1 as a data-side distillation option.**
2. **Larimar (IBM)**: episodic memory module with one-shot, gradient-free memory *writes* at inference — the most literal match found anywhere for the user's "add a small part in the model as memory… even if we remove the entire context" ask. Our note 04 §4 covers RMT/Infini/Titans but not Larimar. **Verify, then add to note 04 §4.**
3. **CAMELoT (IBM)**: consolidation-style compressed memory slots (reported ~30% perplexity reduction on a 7B Llama — REPORTED, verify). Same bucket as #2.

### What others got that we already knew we lacked
Our REPORT §3 "Honest gaps" said: no sub-1B MTP ablation, no cross-call persistence evidence, GAD infeasible, naming unverified. The other runs do not fill **any** of those gaps with evidence (Perplexity/Claude/ChatGPT all cite GAD only at 14B scale; nobody tests cross-call memory persistence). They do fill different gaps (§5 above). Our gaps section remains accurate.

---

## 6. Feasibility screen — their concrete recommendations vs our constraints

Constraint set: 6 GB RTX (sm_75, fp16-only, no flash-attn, PyTorch SDPA), custom 226M GQA w/ CodeLlama-32k tokenizer, single GPU never-co-run, auto-resume contract, venv-only Python.

| Their recommendation | Source | Verdict for us |
|---|---|---|
| FP8 KV cache (vLLM flag) | Perplexity (Day 1), ChatGPT (partial) | **REJECT** — no FP8 on sm_75; INT8 KV exists but saves little (our math) |
| `vllm serve` + prefix caching | Perplexity, Claude (implicitly) | **REJECT as stated** — wrong serving stack for a custom in-repo model; prefix-KV-reuse idea worth remembering for infer.py only |
| LoRA fine-tune Qwen2.5-1.5B student on 10–100K GPT-4 pairs | Perplexity guide | **REPLACE** — our student is our own 226M; corpus scale unproven at our size; our Tier-1 did AST 0.60→0.86 with 16.4k curated pairs; rationale-heavy traces (DSS/SCoTD) are the evidence-backed direction |
| GAD adversarial training | Claude, Perplexity, ChatGPT | **NOT FEASIBLE at 226M/6 GB** (comparable discriminator + RL loop + instability) — our SODA static-snapshot substitute (Track D V2b) is the right scale-down; all three sources actually support this (SODA: 10× faster, 27% less memory than GAD) |
| Rolling summary + external memory store | Claude (Trick 2+3), Perplexity (Option A) | **FEASIBLE NOW, zero training** — the one context answer none of ours covers; candidate Track H (CPU/inference-side) |
| Muon optimizer trial | Claude, Perplexity (implied) | **POINTER** — REPORTED at MoE scale; zero evidence ≤1B; only alongside the Track E probe discipline (500-step A/B rule) |
| No batch-size warmup (scaling-law-justified) | Claude (Flash-Next) | **POINTER** — our runs are tiny; batch warmup is already ~free at our step counts; note only |
| CompLLM/AdmTree-style trained compressor | Perplexity | **PARK** — matches our own note-04 verdict that soft-token compression is research-project-only at 226M; two more sources now agree it needs a trained compressor + frozen base, which we cannot afford to validate |
| Multi-teacher OPD (V4-style) | Perplexity, Claude, ChatGPT | **NOT FEASIBLE as-is** (needs logits from 10+ teachers); the *pattern* (independent specialists → distill) maps onto our P→S/T→S intra-ladder KD (Track C) — same shape, our scale |
| Rubric-based scoring (ROPD) | Perplexity | **ADAPTABLE** — our judge already returns strict-JSON verdicts; a structured rubric schema is a cheap upgrade to V1's judge prompt (Track D) |

---

## 7. Scoring summary and recommendation

**Per-topic winners:** distillation methods + stages/data + frontier models + repo fit → **ours**. Conceptual "why" + Engram/agent-memory → **Claude**. Compression-paper breadth + black-box method zoo + production ops → **Perplexity** (verify before use). Memory-module pointers → **ChatGPT** (Larimar/CAMELoT).

**Recommended actions (all user-gated, none executed):**
1. Keep `distill_survey/` as the base of record; treat the three `other_research/` folders as *lead generators*, not as sources of quotable numbers.
2. Merge (after 1-exa-call verification each, per repo convention):
   - note 01: QAD + DistilQwen2.5 pointers; ToS/legal bullet for cloud teachers (from Claude).
   - note 02: Proxy-KD + ROPD + KED pointers; reward-hacking + ToS caveats.
   - note 03: MiniPLM pointer (pretraining-data distillation).
   - note 04: Engram, Larimar, CAMELoT, ARMT, CompLLM, AdmTree, AttnComp, AMS, TurboQuant, InfLLM as UNVERIFIED pointers; MiniMax-M1 as GDN corroboration.
   - adoption_plan: candidate Track H (zero-training rolling-summary + external-memory harness experiment, extends Track A); dedup + real-data-mix bullets for Track D/G data prep; ROPD-style rubric schema as V1 judge-prompt option.
3. Upgrade (annotate, don't promote to VERIFIED): V4 "specialists → on-policy-distill merge" now has 3 independent secondary sources (ours REPORTED + Claude + Perplexity + ChatGPT) — still worth the single arxiv fetch of the V4 post-training section to settle it.
4. Do **not** import: Perplexity's FP8/vLLM stack, its quality gates, its timelines; Claude's "KV is your bottleneck" framing; ChatGPT's uncited claims.
5. Process lesson (goes to MEMORY.md if user approves): the parallel-runs experiment worked — 3 outside engines produced ~2× unique leads at ~zero risk because ours is the only run with a verification layer; the correct division of labor is *theirs = candidate discovery, ours = verification + gating*.

---

*Comparison produced 2026-09-08 by the repo agent from a full read of all 21 files
in both trees (ours ~105 KB; other_research ~114 KB). Raw-payload existence for
our side verified against `research/raw/` (69 files). No files under
`other_research/` were modified. Labels follow repo convention
(VERIFIED/REPORTED/UNVERIFIED/COMPUTED); "✓" in §3 means "stated by that run",
not independently verified by us unless labeled.*

**2026-09-08 update (post-approval):** §7 actions executed — Track H added
to `adoption_plan.md` (D/E/G bullets included), pointer blocks merged
into notes 01–04, bookkeeping lines added to `REPORT.md` and this
folder's README map.

**2026-09-08 verification round (user-approved; 12/12 tasks OK via the Exa
helper):** tasks in `ver_tasks_round7.json`, payloads in
`research/raw/ver7_*.json`. Promotions: the V4 two-stage post-training
claim is **VERIFIED** with the paper quote ("independent cultivation of
domain-specific experts, followed by unified model consolidation via
on-policy distillation") — note 01 §4 REPORTED→VERIFIED. VERIFIED
existence/claims: MiniPLM (arXiv 2410.17215, ICLR 2025), Larimar (2403.11901,
ICML 2024), CAMELoT (2402.13449), Engram (2601.07372 + deepseek-ai/Engram),
QAD (2607.04244 + nota-ai W4A16 card), DistilQwen2.5 (2504.15027), Proxy-KD
(2401.07013), ROPD (2605.07396), KED (OpenReview 3Y6gc7sesn), MemGPT
(2310.08560) + Mem0 (2504.19413) — Track H's anchors, MiniMax-M1
(2506.13585), Flash-Next no-batch-warmup + Muon/AdamW split (HF model card).
Corrections: TurboQuant is **4-bit** (rotation + Lloyd-Max) — Perplexity's
"3.5-bit / 6× / 328 GB→55 GB" dropped; CAMELoT's "~30% ppl" number
UNCONFIRMED; **ARMT and InfLLM NOT confirmed** this round (remain
UNVERIFIED pointers in note 04 §6). Strategic finding: MiniPLM's
corpus-only KD works "across model families" — a data-side route around
our tokenizer wall for Track E (noted in note 03 §5).
