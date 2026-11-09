# API-only (black-box) distillation — survey (2026-09-08)

Scope: what can a 226M student (CodeLlama tokenizer, ctx 1024, Python) learn from a
cloud teacher it can only *talk to* (text in / text out, no logits, no weights)?
Extends [research/c12_tier2_brief.md](../c12_tier2_brief.md) (V1 judge-rerank tier);
does not duplicate its VRAM gates or teacher table.

## TL;DR

- Black-box = teacher gives text only; white-box logits/hidden-state matching is "entirely inapplicable" there (SODA intro, VERIFIED).
- The de facto standard in the black-box regime is **SeqKD**: SFT the student on teacher outputs — this is what Alpaca/Vicuna/UltraChat/WizardLM all did (SODA intro citing Taori/Chiang/Peng/Zhou, VERIFIED). Our Tier-1 trace SFT already IS SeqKD + rationale data.
- SeqKD's documented flaw: purely off-policy — the student never sees its own generative distribution, limiting OOD generalization (SODA, VERIFIED). This is exactly the Tier-2 rationale in c12_tier2_brief §1.
- **CoT/rationale distillation** is cheap black-box supervision: Distilling Step-by-Step got a **770M T5 to outperform few-shot PaLM-540B using only 80% of a benchmark's examples** — ">700x model size reduction" (Google blog, VERIFIED). Symbolic CoT Distillation shows 125M–1.3B students benefit from CoT *training* even though CoT *prompting* emerges only ~50B+ (SCoTD, VERIFIED) — directly relevant at 226M.
- **On-policy black-box** is now real: GAD (arXiv 2511.10643) trains a discriminator to tell student samples from teacher samples; the discriminator acts as an on-policy reward model — beats SeqKD but needs adversarial training + a comparably-sized discriminator (VERIFIED).
- **SODA** (arXiv 2604.03873) shows the adversarial machinery is unnecessary: pair teacher responses against a **one-time static snapshot of student rollouts** → preference optimization. Matches/beats SOTA in 15/16 settings, **10× faster training, 27% less peak GPU memory** (VERIFIED). This is the closest published template for our judge-rerank V2 upgrade.
- **DPO-from-teacher preferences** is a proven cloud-teacher channel: Zephyr-7B was DPO'd on UltraFeedback GPT-4 *scores* (REPORTED) — teacher gives numbers, not logits.
- Frontier practice with cloud teachers = SFT on outputs (R1-Distill = pure SFT of Qwen/Llama bases on ~800K R1 samples, REPORTED), self-instruct data gen (Alpaca: 52K instr. from text-davinci-003, ~<$600 data cost, REPORTED), judge/reward-scored reranking (LLaMA-2 SFT: 27,540 samples via rejection sampling, REPORTED).
- White-box on-policy (GKD; Thinking Machines per-token reverse-KL: Qwen3-8B → 70% AIME'24 at ~1/10 RL compute, VERIFIED) is NOT available cross-tokenizer — logit KL is undefined (repo finding, VERIFIED) — but its *text-only* cousins (judge-graded rerank-then-SFT, preference pairs) are.
- Every text-only channel (trace SFT, judge scores, rankings, preference pairs) is tokenizer-agnostic — DSS, Orca, R1-Distill, Zephyr all cross tokenizer boundaries (VERIFIED by construction / REPORTED).
- Cloud cost is output-token-dominated; judge-rerank of 2K×8 candidates is a few USD at small-model API tiers, while full CoT regeneration is 10–50× the output tokens (arithmetic ESTIMATE below; prices not source-stated).

## Sources

| Source | URL | Date | Type |
|---|---|---|---|
| GAD — Black-Box On-Policy Distillation of LLMs | https://arxiv.org/abs/2511.10643 | 2025-11-13 | paper |
| SODA — Semi On-Policy Black-Box Distillation | https://arxiv.org/abs/2604.03873 | 2026-04-04 | paper |
| GrayKD (AAAI-40) | https://doi.org/10.1609/aaai.v40i38.40470 | 2026-03-14 | paper |
| Bounded Behavioral Indistinguishability for Black-Box LLM Distillation | https://arxiv.org/html/2605.30448 | 2026-05 (inferred from ID) | paper |
| Knowledge Distillation of Black-Box LLMs | https://arxiv.org/abs/2401.07013 | 2024-01-13 | paper (survey-style) |
| Distilling Step-by-Step (ACL Findings 2023) | https://aclanthology.org/2023.findings-acl.507.pdf | 2023 | paper |
| Distilling Step-by-Step (Google blog, numbers) | https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/ | 2023-09-21 | blog |
| Symbolic CoT Distillation (SCoTD) | https://aclanthology.org/2023.acl-long.150.pdf | 2023 | paper |
| Key Factors for CoT Distillation into SLMs | https://arxiv.org/abs/2502.18001 | 2025-02 | paper |
| GKD — On-Policy Distillation (white-box) | https://arxiv.org/abs/2306.13649 | 2023-06 / ICLR 2024 | paper |
| GKD TRL trainer | https://huggingface.co/docs/trl/main/en/gkd_trainer | current | docs |
| On-Policy Distillation (Thinking Machines) | https://thinkingmachines.ai/blog/on-policy-distillation/ | 2025-10-27 | blog |
| Tinker distillation recipes | https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/ | current | docs |
| Exa grounded answer (ans1_api_distill.json) | repo research/raw/ | 2026-09-08 | answer |
| C12 Tier 2 brief (this repo) | [research/c12_tier2_brief.md](../c12_tier2_brief.md) | 2026-09-06 | repo design note |

REPORTED-from-memory, not re-verified this session: R1-Distill (https://arxiv.org/abs/2501.12948), Alpaca (https://crfm.stanford.edu/2023/03/13/alpaca.html), Orca (https://arxiv.org/abs/2306.02707), LLaMA-2 SFT (https://arxiv.org/abs/2307.09288), Zephyr/UltraFeedback (https://arxiv.org/abs/2310.01376, https://huggingface.co/blog/zephyr).

## 1. The black-box taxonomy

**a) Sequence-level KD on teacher outputs (SeqKD / offline SFT).** SFT the student on
(instruction, teacher-output) pairs; teacher access = text output only. SODA calls it
"the de facto standard" in the discrete regime (VERIFIED). Data budget: Alpaca 52K
(REPORTED), LLaMA-2 SFT 27,540 (REPORTED); can be as low as a few K for narrow domains.
Known result: this is how every Alpaca-lineage small model exists. Weakness: off-policy
— no exposure to the student's own failure modes (SODA, VERIFIED).

**b) CoT/rationale distillation.** Teacher generates rationales (CoT-prompted); student
is multi-task-trained on label + rationale tasks (DSS mechanism, VERIFIED). Teacher
access: text only. DSS result: 770M T5 outperforms few-shot PaLM-540B using only 80% of
a benchmark's examples, ">700x model size reduction" (VERIFIED). SCoTD: 125M–1.3B
students trained on teacher rationalizations still benefit from CoT (VERIFIED) —
removes the "too small to reason" objection at our scale. Key-Factors paper (2502.18001)
systematically varies granularity/format/teacher across 4 teachers × 7 students × 7
datasets; payload highlights cut off its findings mid-sentence (SLMs behave differently
from LLMs in CoT distillation — detail UNVERIFIED).

**c) On-policy black-box (student samples → teacher signal).** Two flavors:
- *Fully on-policy / adversarial — GAD.* Student is a generator; a discriminator
  distinguishes its responses from the teacher's; discriminator = on-policy reward model
  that "co-evolves with the student"; GAD "consistently surpasses" SeqKD (VERIFIED).
  Cost: extra discriminator of comparable size + alternating updates + instability
  (SODA's characterization, VERIFIED).
- *Semi on-policy — SODA.* One-time static snapshot of the student's zero-shot rollouts;
  pair each with the (near-always better) teacher response; translate the capability gap
  into preference optimization. "No complex filtering, no costly dynamic rollouts, no
  adversarial training"; 15/16 benchmark settings match/beat SOTA, 10× faster, 27% less
  peak GPU memory, validated on four compact Qwen2.5/Llama-3 students (VERIFIED).
- *Judge/reward-rerank then SFT* (our brief V1): sample N from student, teacher scores
  as text (LLM-judge score or mean logprob under the judge's own tokenizer), SFT on
  winners. Same family as rejection-sampling SFT; LLaMA-2 built its SFT set this way
  with a reward-model reranker (REPORTED). The gray-zone answer (ans1) also lists
  CoTD-PO (RL with teacher preference scores), SimCT (cross-tokenizer supervision via
  aligned units), daDPO (teacher distribution folded into DPO) — **UNVERIFIED**: the
  payload stored 0 result records; do not plan against those names without a source.
- *Teacher-grades-every-token (white-box only).* GKD and the Thinking Machines recipe
  need per-token teacher logits on the student's own trajectories (70% AIME'24 at ~1/10
  RL compute; Tinker: 76.7% with rank-128 LoRA, VERIFIED). Not available to us.

**d) DPO/preference pairs from teacher.** Teacher text (or scores) provides chosen vs
rejected; student runs DPO. SODA is exactly this with teacher-vs-student pairs
(VERIFIED). Zephyr-7B was DPO'd purely on GPT-4 numeric scores over 1–10 dimensions
(REPORTED) — proof that *scores*, not logits, suffice. Data budget: UltraFeedback-scale
~60K pairs (REPORTED) but SODA-style student-contrast pairs are per-prompt cheap.

**e) LLM-as-judge reranking.** Teacher ranks N candidates (scalar or pairwise);
winner per prompt feeds SFT (rejection sampling), or rankings feed DPO. Teacher access:
text + optionally scores. This is the V1 design in c12_tier2_brief §3.

**f) Data-centric distillation.** Teacher *generates the dataset* (self-instruct,
Evol-Instruct, persona-driven) rather than answers queries — what Tier 1 already did
with Evol-Instruct traces. SODA cites Zhou et al. (Evol-Instruct) as a SeqKD instance
(VERIFIED).

**g) GrayKD** (AAAI 2026): "multi-rationale injection" from a text-only teacher to
bridge black-box↔white-box gaps (abstract-level, VERIFIED scope only; mechanism not
extracted).

## 2. What frontier labs actually do with cloud teachers

- **R1-Distill** (REPORTED): Qwen2.5/Llama bases SFT'd purely on DeepSeek-R1 *outputs*
  (~800K samples: 600K reasoning + 200K non-reasoning) — no logits, no RL for the
  released students. Text-only transfer across model families.
- **Alpaca self-instruct** (REPORTED): 175 seed instructions expanded by
  text-davinci-003 → 52K; data generation "<$600". Template for our Tier-1-style corpus.
- **Judge/rerank pipelines** (REPORTED): LLaMA-2 SFT = 27,540 top-rejection-sampled
  outputs; UltraFeedback = GPT-4-scored prompts feeding Zephyr's DPO. Frontier-standard
  when you own no teacher weights.
- **Thinking Machines/Tinker** (VERIFIED as white-box): on-policy per-token reverse-KL
  on OpenThoughts3/DeepMath/Tulu3 recipes — the *learned-from* frontier method, but it
  requires teacher logits; their API surface exposes grading, ours (text-only cloud or
  local Qwen3.5-0.8B) does not.

## 3. Cost model

Structure (no 2026 price source in payloads — arithmetic only, ESTIMATE; verify current
API pricing before any spend):

- **Judge-rerank** (per 2K prompts, N=8 candidates, ~150-token instruction + ~400-token
  candidate, ~30-token verdict): input ≈ 2K×8×550 ≈ 8.8M tokens, output ≈ 0.5M.
  At small-model-tier illustrative pricing $0.15/$0.60 per 1M (UNVERIFIED): **~$1.6**.
  Even an order of magnitude more tokens stays single-digit USD.
- **CoT/trace generation** (teacher writes full traces; 2K prompts × ~600 out-tokens):
  ~0.3M in / ~1.2M out → still <$1 at tier pricing, but 10–50× the output tokens of
  judging, and quality of *every* token must be trusted (no gate).
- **Local Qwen3.5-0.8B teacher** (exists on disk): c12_tier2_brief §5 measured/estimated
  ~1–2 s per candidate at 1.5B fp16 on this GPU; 0.8B faster → 16K candidates ≈ 4–9 h of
  GPU window time, $0. GPU windows must not overlap train runs (AGENTS.md §4.6).
- **Sample-count guidance for 226M** (no source states a 226M-specific number — honest
  gap): published anchors are 27.5K–52K for general chat SFT (REPORTED); domain-narrow
  rerank recipes work at far fewer unique prompts because selection, not volume, is the
  lever (SODA uses compact students, count not stated, VERIFIED-scope). Start from the
  Tier-1 instruction pool scale and gate on AST pass-rate (brief §6), not on absolute
  data volume.

## 4. Applicability to nano_SLMs — V2 recipe options (extends c12_tier2_brief V1)

Build on brief V1 (student samples → teacher scores as TEXT → best per instruction →
pairs.jsonl → sft.py). What the survey adds per option:

- **V2a — Feasible-now: judge-rerank with code-native gates (upgrade of brief V1).**
  Student sampling: temp 0.8, top_p 0.95, N=4–8, ctx 512 (brief §5); add top_k ~50.
  Judge prompt shape for CODE (Python): system prompt = strict code reviewer; return
  JSON `{score: 0-10, syntax_ok, issues: [], fixed_snippet?}`; parse ladder JSON →
  fenced-block regex → reject; *programmatic gates outrank the judge*: fenced-code
  extraction + ast.parse (brief V1) + optionally exec/doctest sandbox for verifiable
  prompts. Acceptance gates: brief §6 (AST pass on 50 held-out STRICTLY above Tier-1
  0.86; CSN val regression ≤10–15%; grad_norm ≤0.6×clip; zero NaNs). Cost: local judge
  $0 + one GPU window (4–9 h est.); cloud judge ~$2 (ESTIMATE).
- **V2b — Feasible-now: SODA-style preference pairs (static snapshot), plain pair file.**
  chosen = teacher/best candidate, rejected = *typical* student rollout from a one-time
  snapshot (not cherry-picked worst — SODA uses raw snapshot, VERIFIED). Two paths:
  (i) margin-gated rerank SFT (winner-only, i.e., V1 again) or (ii) true DPO — needs a
  DPO trainer; **trl/peft not installed** (brief §2, VERIFIED) → needs-work, and DPO
  stability at 226M is unproven. V2b(i) costs nothing extra over V2a.
- **V2c — Feasible-now (user-gated): cloud CoT/trace generation for a Tier-1.5 corpus.**
  Frontier teacher writes reasoning-then-code traces; same AST gates. Upside per DSS:
  rationale supervision beat the teacher itself at small scale (VERIFIED). Few-USD class
  (ESTIMATE); requires explicit user approval (network + spend rules).
- **V2d — Needs-work: semi-on-policy hard-subset mining.** Snapshot the student's
  *failing* prompts (AST-fail candidates), resample those with teacher targets — SODA's
  "expose the student to its own inferior modes" (VERIFIED) operationalized as data
  selection. Small diff on distill_onpolicy.py.
- **Not-feasible: GAD** — discriminator of comparable size + adversarial instability on
  a 6/8 GB single GPU defeats the purpose (SODA's own cost argument, VERIFIED).
- **Not-feasible: GKD / per-token reverse-KL (GKD-TRL, TML blog)** — white-box; requires
  teacher logits on student tokenization; cross-tokenizer KL undefined (brief §3).
- GPU discipline: judge phase = teacher inference only (brief §4 profile ~4.1 GB peak at
  1.5B; 0.8B lighter); never overlap with an active train run (AGENTS.md §4.6).

## 5. The tokenizer-mismatch elephant

- **What breaks:** any loss computed *position-wise on tokens* across teacher and
  student — logit KL, per-token reverse-KL, hidden-state matching. Confirmed by the
  repo finding (c12_tier2_brief §3: student CodeLlama 32k vs Qwen ~151k BPE → KL
  mathematically undefined, VERIFIED) and by SODA's framing that KL-based white-box
  alignment is "entirely inapplicable" in the black-box regime (VERIFIED).
- **What still works (all text-only, hence tokenizer-agnostic):**
  - Trace/CoT SFT — Tier 1 already proved it here (AST 0.60→0.86, repo REPORTED);
    DSS is the canonical cross-tokenizer proof (T5 student, PaLM teacher, VERIFIED).
  - Judge scores & rerank — the judge only reads strings; our V1 even scores via mean
    logprob under the *judge's own tokenizer* (brief §3), which is self-consistent.
  - Preference pairs / DPO — chosen/rejected are text pairs; SODA pairs Qwen2.5 and
    Llama-3 students with frontier teachers across families (VERIFIED); Zephyr took
    GPT-4 scores into a Mistral tokenizer (REPORTED).
  - Data generation — self-instruct/Evol corpora are plain text (Tier 1 precedent).
- Frontier corroboration: R1-Distill (DeepSeek tokenizer → Qwen/Llama tokenizers, pure
  text SFT, REPORTED); Orca (GPT-4 traces → LLaMA student, REPORTED).
- ans1 names SimCT as a method built for cross-tokenizer alignment — UNVERIFIED
  (payload stored 0 records); interesting only if SODA-style preference transfer
  underperforms.
- Conclusion: the tokenizer blocker kills logit KD, *not* the API-only program.
  Every V2a–V2d option above is tokenizer-safe by construction.

## 5. Pointers from the parallel-runs comparison (2026-09-08) — verification round

Verified 2026-09-08 against `research/raw/ver7_blackbox_methods.json`
(12/12 round-7 tasks succeeded):
- **Proxy-KD**: **VERIFIED** = arXiv 2401.07013, "Knowledge Distillation of
  Black-Box Large Language Models" (+ OpenReview MYb7hT6D9e): a white-box
  proxy sits between the black-box teacher and the student so the student
  trains on the proxy's soft labels/KL. Relevant to us ONLY if the proxy
  shares the student's 32k CodeLlama vocab (the tokenizer wall, brief §3) —
  otherwise it inherits the same blocker. Design-note only.
- **ROPD — rubric-based on-policy distillation**: **VERIFIED** = arXiv
  2605.07396 (+ github.com/Peregrine123/ROPD_official): induces
  prompt-specific rubrics by contrasting teacher vs student outputs, then
  scores student rollouts with them for on-policy optimization. Maps onto
  the V1 strict-JSON judge schema as an upgrade option (adoption_plan
  Track D).
- **KED — knowledge-explaining distillation**: **VERIFIED existence** =
  OpenReview 3Y6gc7sesn, "Improving Knowledge Distillation with Teacher's
  Explanation" (superfeature-explaining teachers). We already train on
  rationale traces (Tier 1), so we approximate this; recorded for
  completeness.
- **Operational caveats (Claude's GAD file)**: adversarial/judge loops
  invite reward hacking — monitor with a held-out scorer; and check
  provider ToS on training-from-outputs before ANY cloud-teacher spend
  (now also in adoption_plan Track D). GAD itself stays not-feasible at
  our scale (§1c).

## Open questions

1. **Judge validity at 0.8B:** how well does local Qwen3.5-0.8B-as-judge agree with
   ast.parse/exec outcomes on a ~100-candidate calibration set? Agreement decides
   local-judge V2a vs cloud-judge; currently assumed, never measured.
2. **Preference at 226M:** is SODA-style DPO stable at 226M params, and what is the
   minimal pair count before collapse/overfit? (Needs trl/peft install + tiny pilot;
   nothing in the sources covers <1B students.)
3. **Cloud CoT marginal value:** do frontier-teacher Python traces beat the Tier-1
   Qwen trace corpus at equal instruction count (AST pass > 0.86), and at what actual
   USD under current pricing?
4. **Static-snapshot staleness (SODA):** after one rerank round, the snapshot is stale —
   how many snapshot-refresh rounds before gains plateau, and does hard-subset mining
   (V2d) beat uniform resampling?
5. **Teacher-completes-student-context (DAgger-style text-only OPD):** feeding the
   student's sampled prefix to the teacher and SFT-ing the teacher's continuation is
   the one remaining text-only route to true per-state on-policy correction — is the
   extra engineering (per-state API/local calls, prefix stitching) worth it vs V2a
   rerank, given the brief's exposure-bias framing?
