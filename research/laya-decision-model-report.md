# Laya — Deep Research Report: what it is, how it was trained, and whether we can recreate it at 50–100M params

**Status:** research note (no experiment run). **Date:** 2026-09-22 session.
**Trigger:** user request — deep-research convaiinnovations/laya + Luni/laya-jev-benchmark
(data → strategy → architecture), then assess: can we train a ~50M model with same/near
scores, recreate it as the author describes (hardware-fit), learn from it, and build a
50–100M param version.
**Raw payloads:** `research/raw/laya/txt/` (model cards ×3, configs, eval files, dataset
README/RESULTS, bench scripts, sibling APIs). Subagent reports integrated: training-pipeline
(code-verified from GitHub + HF `rl_common.py`), feasibility (small-encoder evidence).

---

## 1) TL;DR

- **Laya is NOT a generative LM.** It is a *non-autoregressive "System 1" decision model*:
  one forward pass maps (state, typed questions) → typed answers with calibrated
  probabilities. Question types: `choice` (softmax over options), `score` (ordinal
  distribution), `noul` (yes/no probability). It never generates text.
- **Family (all Apache 2.0, one HF repo with subfolders):** English root = ModernBERT-large
  (395M) fully fine-tuned + small decision head = **421M**; `laya-multilingual` = mmBERT-base
  + head = **322M**, 100+ languages; `laya-typed-decisions` = 421M specialist for four
  synthetic workflows.
- **Training recipe (code-verified):** RLCD — REINFORCE with group-mean baseline
  (GRPO-style), Gaussian logit noise for exploration, reward = strictly proper scoring rules
  (log + spherical − RPS), TD(λ=1.0) for multi-turn episodes, per-(type,option-count)
  temperature calibration fitted post-hoc. Root run: pure policy gradient (no CE), G=8,
  σ 1.0→0.3, 7,313 updates / 1 epoch / **1.96 h**. The public fine-tune notebook mixes RL + CE.
- **Training data:** author claims 100% human-labeled public datasets ("Zero Synthetic
  Shortcuts", 6 source categories; 13 task families in the eval). Root training/data-generation
  code is **NOT published** (negative finding). The typed-decisions fine-tune data IS public:
  `LocalLLaMA/typed-decisions` (1,200 train cases / 6,000 decisions, synthetic, teacher
  ≈ 4B-class LLM, 3 samples @ temp 0.7, gold = mean distribution).
- **Third-party reality check (Luni):** Laya's "+16.0%" claim vs Jev compares two different
  benchmarks. On shared sets: phishing raw accuracy 0.505 (≈ chance; 0.611 after Platt), and
  **base Laya is near-chance zero-shot on typed-decisions (0.36)** — the 0.766 capability
  comes entirely from fine-tuning. Luni fine-tuned Laya on a 180k public mixture in **55 min**
  (3 epochs) → macro held-out acc 0.84.
- **Feasibility for us (single RTX 3000 6 GB, Turing sm_75, fp16-only, SDPA):** **YES with
  conditions.** Recommended path = existing small pretrained encoder (MiniLM-L12-H384, 33M or
  BERT-Medium) + Laya's decision head unchanged + same RLCD recipe ≈ 38–50M total.
  Expected: within −2 to −5 points macro on robust tasks; hard low-data families drop more.
  Time: fine-tune ~2–4 h, RLCD pass ~10–40 h (dominant, least certain) — probe first.
  From-scratch pretrain is possible (ELECTRA precedent) but ~1.5–2.5 weeks and weaker.

---

## 2) What Laya is

Pipeline tag `text-classification`. A **decision model**: input = a *state* (text, email,
ticket, or JSON) + typed questions (schema defined at request time); output = one calibrated
probability distribution per question, in a single forward pass (~33–40 ms on T4). No text
generation → nothing to parse, nothing to hallucinate. Sits in the "System 1" niche: reflex
decisions (routing, triage, guardrails, moderation, scoring) that an LLM call (236–710 ms)
handles slowly.

| Checkpoint | Backbone | Params | Ctx | Trained |
|---|---|---|---|---|
| `laya` (root) | ModernBERT-large, 28L/H1024, vocab 50368 | 421M | 512 (head 192) | from earlier ckpt, 1 epoch RLCD |
| `laya-multilingual` | mmBERT-base, 22L/H768, vocab 256000 | 322M | 1024 (head 256) | from scratch, 4 epochs RLCD |
| `laya-typed-decisions` | ModernBERT-large (same as root) | 421M | 1024 (head 256) | fine-tuned from `laya` |

A pure-Python `Router` dispatches per-request by script/language (<0.5 ms) because the
English checkpoint **collapses on non-Latin scripts while staying confident** (Khmer 0.000
accuracy at 0.952 confidence — confidence gating cannot catch it).

## 3) Architecture (code-verified: `laya/common.py` + `rl_common.py`)

**Sequence packing** (`build_sequence`): one flat sequence per call —
`[CLS] <type> instructions [SEP] <marker> opt0 <marker> opt1 … [SEP] state [SEP]`.
Options rendered as `label: criterion`; every option gets its own `[MASK]`-style marker
token; budget split: options consume `head_max_len` (192 en / 256 ml), state gets the rest
(`max_len − head`), state truncated from the LEFT for multi-turn prefixes (keep most recent).

**Decision head** (trained from scratch, ~27M on the 421M model):
1. encoder hidden states + a learned **per-question-type embedding** (choice/score/noul),
   broadcast over all tokens;
2. `head_layers=2` standard `nn.TransformerEncoderLayer` (norm_first, dropout 0.1);
3. `torch.gather` at the option-marker positions → scorer `LayerNorm → Linear(d,d) → GELU
   → Linear(d,1)` → one logit per option → **masked softmax within that question's options**.
   All questions of a call are collated into ONE forward pass.
4. **Act/escalate head**: pooled [CLS] + 4 detached distribution stats (top-1 prob, top1−top2
   margin, normalized entropy, k/255) → `Linear(d+4,256) → GELU → Linear(256,2)`.
   Cost matrix: act +1.0 / act-wrong −3.0 / escalate −0.5 → policy acts only above P(correct)
   > 0.625. (Caveat: the public notebook trains it with a zeroed loss term; the shipped eval
   shows automation_rate 1.0.)
5. **Confidence** = 1 − H(p)/log(k); **temperature buffers** (3 per-type +
   per-option-count buckets) divide logits, runtime-clamped [0.5, 5.0].

Answer space defined at request time — new schemas need no retraining. Known hard limit:
options share the fixed `head_max_len` budget, so >20-option questions starve (Banking77
77 options → ~3–4 tokens/label → 0.425 vs Jev 0.870).

## 4) Training strategy (RLCD, code-verified)

- **Reward** (`proper_reward`): `r = log_score + w_sph·spherical − w_rps·RPS` (RPS only for
  ordinal `score`). All three are strictly proper → the unique reward-maximizing report is
  the honest distribution. Defaults w_sph=0.5, w_rps=1.0 (notebook uses 0.75).
- **Exploration:** z = logits + ε, ε ~ N(0, σ²I), zero-mean projected across options;
  σ 1.0→0.3 (root, G=8) / σ 0.4→0.1 (notebook, G=4).
- **Update:** REINFORCE, advantage = (r − mean(group)) / (std(group) + 1e-6) — GRPO-style
  group baseline; Gaussian log-prob of the sampled noise as the policy log-likelihood.
- **Multi-turn:** episode sliced into up to 6 prefixes; TD(λ=1.0) targets from the terminal
  outcome (Monte-Carlo) grouped per record.
- **Two published regimes:** root checkpoint = **pure policy gradient, zero CE** (dev.to);
  public fine-tune = **RL + soft-CE mix** (`loss = (loss_rl + 1.0·loss_ce)/GRAD_ACCUM`) on
  the benchmark's soft gold distributions.
- **Notebook hyperparameters (exact):** EPOCHS=4, MICRO_BATCH=8, GRAD_ACCUM=4 (2×GPU ⇒
  effective 64), GROUP_SIZE=4, LR_ENCODER=2.5e-5, LR_HEAD=1e-4, AdamW wd=0.01,
  CosineAnnealingLR (no warmup), fp16 autocast + GradScaler, clip 1.0, max 4096 tokens/batch,
  encoder gradient checkpointing, `max_len=1024` `head_max_len=256`.
- **Temperature calibration (post-hoc):** one temperature per (type × option-count bucket:
  2 / 3-5 / 6-10 / 11+), LBFGS on held-out logits minimizing NLL, clamped [0.1,10]. Moves ECE
  0.466→0.081 (root). The multilingual checkpoint ships UNCALIBRATED (all temps 1.0).
- **Root run record:** 7,313 updates, 1 epoch, **1.96 h**, world_size=1,
  `fine_tuned_from_checkpoint: true` (earlier checkpoint unpublished). Multilingual: 15,987
  updates, 4 epochs, **4.97 h**, from scratch. **No SFT stage found anywhere** (UNKNOWN if a
  private one exists).
- **Root fine-tune cost reference:** typed-decisions fine-tune ≈ 4–6 min on 2×T4 (6,000
  items); "4–5 hours for 4 epochs over ~30k questions" for the bigger mix.

## 5) Training data

**Root checkpoint — 13 eval task families** (eval/results.md, 23,024 in-task + 2,400
zero-shot questions): conversation outcomes (3600, acc 0.482), email triage (2691, 0.732),
emotion and tone (1825, 0.906), inference & fact checking (3022, 0.883), instruction-following
(600, 0.878), intent and routing (1475, **0.991**), moderation and safety (2708, 0.967),
reading comprehension (770, 0.847), response quality scoring (3146, 0.581), robustness checks
(744, 0.851), search relevance (733, 0.628), sentiment and rating (961, **0.442** — weak),
topic classification (749, 0.939). Overall: acc 0.753, ECE 0.030, acc@50%-coverage 0.947.

**Author's provenance claim (dev.to, "Zero Synthetic Shortcuts"):** 100% human-labeled
real-world public datasets in 6 categories: (1) support triage & intents (customer-service
conversations, banking intents, ticket queues), (2) inference & fact checking (NLI/fact
verification), (3) content safety (human-consensus toxicity labels), (4) security & guardrails
(real jailbreak/injection prompts), (5) rubrics & quality (multi-axis human rubric ratings),
(6) multi-turn SaaS sales/support trajectories with verified outcomes. Augmentation: option
order shuffling, question paraphrase, raw-text/JSON state alternation, distractor-question
injection. The author explicitly argues AGAINST teacher-LLM synthetic labels for calibration
("you calibrate to the LLM's own mistakes"). **No dataset list, no teacher name, no
data-generation code published** — the root pipeline is a black box; BENCHMARKS.md only marks
themes "in training mix" vs "held out".

**Typed-decisions fine-tune data (public):** `LocalLLaMA/typed-decisions` — 4 workflows
(agent-trace observability, customer service, invoice processing, security incidents),
1,600 cases total (1,200 train / 400 test), 5 typed questions per case over shared state.
Built: latent-factor skeleton → state rendered (text by a model where textual) → labeled by a
**teacher endpoint of roughly 4B-class capability, 3 samples @ temp 0.7** → gold = mean of the
3 sampled distributions (soft where ambiguous). Reference points: majority 0.520 (1600-case
set), factor ceiling 0.704, teacher self-agreement 0.735.

**Luni's generalist fine-tune mixture (180k items):** mostly BoolQ, SQuAD v2, SNLI, MultiNLI,
ANLI, SciTail (+ bitext support, civil comments, clinc_oos, enron spam, jailbreak hub,
phishing email, chaos_mnli, unli, yelp_stars, support tickets…), 3 epochs, **55 minutes** on a
single fast GPU, length-bucketed batching (2.66–3.34× fewer token slots than random order).
Result: macro held-out acc 0.838–0.840, grounding probes 2/5 → 5/5. Checkpoint:
`Luni/laya-grounded` (CC-BY-NC-4.0 because mixture includes ANLI + customer-support-tickets).

## 6) Benchmarks + the third-party reality check

**Author-reported (T4):** typed-decisions 0.766 (beats Jev 0.727 and the 0.735 teacher
ceiling); AG News 0.950; DAIR Emotion 0.595; XNLI-en 0.860; MASSIVE-en 0.783; ECE 0.081
(post-calibration); p50 38.4 ms (1 q) → 6–8× faster than Jev's 236–276 ms. 103–332 q/s.

**Luni's critique (one RTX 5090, 2026-09-19, dataset repo we were given):**
- The "+16.0% vs Jev" claim compares **two different benchmarks** — not evidence.
- Phishing (`AreLit/PhishNChips`, 2,000 emails): Laya raw **0.505 ≈ chance**, recall 1.2%,
  AUROC 0.678 (≈ Jev's 0.689 — ranking fine, threshold wrong); Platt calibration → 0.611 vs
  Jev 0.626; Claude Haiku 4.5 0.813 (both ~20 points behind).
- **Probe suite (11 hand-written assertions):** base Laya fails 7/11 — contradiction handling
  (P(a)+P(¬a) = 1.73 and 0.09), grounding ("has contacted before" = 0.20 despite the text
  stating it), option-rename instability. Fine-tune fixes most (data > loss: the no-
  consistency-loss arm scored best on probes).
- typed-decisions: base 0.360 (below the 0.461 majority baseline!) → fine-tuned 0.767. Above
  the 0.735 teacher ceiling = partially memorizing teacher noise.
- WiSE-FT sweep interpolates base↔fine-tuned (0.0→0.767 acc, ECE minimized at α 0.2).
- Renaming options still flips verdicts after fine-tuning — runtime schemas not yet robust.

**Specialist baselines on the same typed-decisions test set (dataset card, Adaptive
Classifier, encoder frozen, soft-target apportioning):** **MiniLM-L6 (22M) 0.587**;
**ModernBERT-base (149M) 0.646**; Jev (generalist, zero-shot) 0.727; teacher ceiling 0.735;
Laya-typed-decisions (421M, RLCD+CE) 0.766. → This is the ONLY directly-relevant scaling
evidence on this exact benchmark: 22M→58.7%, 149M→64.6%, 421M+recipe→76.6%.

## 7) Ecosystem & prior work

**Jev (TypeSafe AI, typesafe.ai — NOT typesafe.com, which redirects to Lightbend).** Closed
hosted 'System One' decision model, released 2026-09-15, version jev-1.13.0; weights and
architecture UNKNOWN. Third-party measurements: phishing 0.626 acc / ECE 0.154 / AUROC 0.689
/ p50 239 ms; typed-decisions 0.727 / ECE 0.144 / 710 ms per case (saturated, at the 0.735
teacher ceiling); nibzard DMB: banking77 76.3%, spam 93.0%, hard cap at 255 options (rejects
256+), only model that 'bluffs' (admits uncertainty 49.7%) yet worst ECE 0.246; verified
pricing is per-decision ~$0.07/1k decisions, ~$0.024/1k calls (the '$0.042/1M tokens' figure
on Laya's card: NOT found — treat as unverified). Jev Decision Index (132,422-request suite,
31 open repros): **Laya itself scores 16.39 — near the bottom** (top repro 'Jevfire' 55.74;
mayafree AUC leaderboard: Jev 0.7350 vs Laya-Typed-Decisions 0.5144). Lesson: Laya's real
strengths are speed + calibration + license, not generalist breadth.

**Backbone lineage.** ModernBERT (arXiv 2412.13663): bidirectional encoder, 2T tokens
English+code, native 8192 ctx, Local-Global Alternating Attention (full attention every 3rd
layer; 128-token sliding window elsewhere; dual RoPE thetas 160k/10k), unpadding + flash
attention, StableAdamW + trapezoidal LR, trained on 8x H100; base 149M/22L, large 395M/28L;
GLUE 88.4/90.4. mmBERT (arXiv 2509.06888): ModernBERT architecture, 3T tokens, 1800+
languages (low-resource languages added only in the decay phase), sans-pos positions,
256k vocab, MIT.

**Recipe lineage.** RLCD (real paper: arXiv 2307.12950, Yang et al. — the commonly-cited
2309.16367 is a mimetic-gravity paper) = principle-guided contrastive preference labels +
PPO for alignment; Laya's 'RLCD' is a downstream re-interpretation: RL against strictly
proper scoring rules (log/Brier/RPS) so honest probabilities are the unique reward
maximizers. Group-mean-baseline REINFORCE = GRPO (arXiv 2402.03300, DeepSeekMath — PPO
variant, critic-free, memory-lean).

**Small-model prior art (community Jev repros at our scale):** people ARE building small
typed-decision models — pngwn/nanodiff-350m-typed-decisions (masked diffusion, MIT),
abidlabs/jev-typed-decisions-causal-0.6b (Qwen3-0.6B LoRA), samatv256/mini-Jev (Qwen3-0.6B),
Mapika/decider-0.8b, heman10x/rlcd-modernbert-151m (RLCD ModernBERT, 151M), Tiny-Jev,
open-jev-4B — mostly causal-LM or diffusion LMs; none published as encoder+decision-head at
~50M. Luni/laya-grounded (421M, CC-BY-NC): 180k public mixture (BoolQ, SQuAD v2
unanswerable, SNLI/MultiNLI/ANLI/SciTail + 51.6% inverse-pairs), RLCD + semantic-consistency
penalty (Xu et al. 2018 form, lambda 0.05→0.30), 3 epochs / 55 min on one RTX 5090 → probe
failures 7→2, grounding 2/5→5/5, macro 0.840, ECE 0.156; documented regressions: phishing
0.611→0.512, option-rename stability worse. Its rl_agent_config inherits the base's
1-epoch/1.96h training block (unexplained vs README's '3 epochs, 55 min' — likely copied).

## 8) Feasibility: a 50–100M recreation on our hardware

**Verdict: YES with conditions.** Capability lives in the recipe + fine-tuning, not in 400M
parameters. But "same/near score" holds on robust mid-data tasks, NOT uniformly.

**Evidence (fetched, primary sources):** the 22–33M class beats 2018 BERT-base on MNLI
(DeBERTa-v3-xsmall 88.1 vs 84.5; MiniLM-L12-H384 85.7) and sits ~2–4 points under 300–400M
encoders on large-data classification; gaps widen on hard low-data tasks (RTE 66.7 vs 88.0).
On Laya's own benchmark: MiniLM-22M 0.587 / MBERT-base-149M 0.646 / Laya-421M 0.766.

**Embedding-budget trap:** ModernBERT-large's vocab alone (50368×1024 = 51.6M) exceeds a 50M
total budget. DeBERTa-v3-xsmall hides the same trap (22M backbone + 48M embeddings = 70M).
Feasible design: 30522-vocab + H384 ≈ 11.7M emb + ~22M backbone + Laya head ≈ **38–45M**.

**Recommended recipe (faithful where it matters):**
1. Encoder: `microsoft/MiniLM-L12-H384-uncased` (33M, MIT, MNLI 85.7) or BERT-Medium
   (8L/H512, ~41M). Vanilla architectures → zero ModernBERT/SDPA/fp16 risk on sm_75.
2. Head: Laya's head UNCHANGED (type embedding, 2-layer transformer, option-marker scorer,
   act head), implemented from the verified `rl_common.py`/`common.py` logic.
3. Data: start with the public `LocalLLaMA/typed-decisions` train split (1,200 cases) with
   soft-CE + RLCD (notebook regime); then scale with a Luni-style public mixture (BoolQ, SNLI,
   MultiNLI, ANLI, SciTail, SQuAD v2…) for the generalist arm.
4. RLCD exactly as verified: log + w_sph·spherical − RPS reward, group-mean-baseline REINFORCE,
   Gaussian logit noise (σ 0.4→0.1 first), per-(type,bucket) LBFGS temperature calibration.
5. Eval: Luni's bench scripts (saved in research/raw/laya/txt/) + the probe suite.

**Expected deltas vs Laya published** (estimates; no direct small-model numbers exist on
Laya's suites): XNLI-en 0.860 → ~0.82–0.85; MASSIVE-en 0.783 → ~0.73–0.77; AG News 0.950 →
~0.93–0.95; in-task macro 0.753 → ~0.70–0.74; ECE stays ~0.03–0.05 (post-hoc calibration,
head unchanged). typed-decisions: plausibly 0.60–0.68 vs the 421M's 0.766 (scaling evidence
above), with RLCD+CE closing part of the gap. Hard low-data families may drop 10–20 points.

**Costs on our RTX 3000 6 GB (estimates — probe before committing):**
- Fine-tune (Luni recipe, 33M model): ~2–4 h GPU. VRAM: ~0.7 GB optimizer+grads + activations
  → comfortable.
- RLCD pass: ~10–40 h (group sampling × step cost — dominant, least certain).
- From-scratch pretrain alternative: ELECTRA-small precedent = 4 days V100 → ~6–12 days
  Turing; usable weak checkpoint ~1–2 days; +1–3 points worse than pretrained-init.
  → Only worth it as a learning exercise, not for scores.
- Running the REAL Laya (421M, ~843 MB fp16) for inference/eval on our card: fits (~1.5 GB);
  full AdamW fine-tune of it does NOT fit 6 GB (≈6.7 GB state alone) — 8-bit-optimizer or
  head-only/LoRA would be needed. "Learn from it" is practical via: eval it locally with the
  SDK on our benchmarks, and study its calibrated errors (probe suite).

## 9) Proposed staged plan (nothing started — user-gated)

- **L0 (CPU, no gate):** this report + raw payloads (done). Optionally clone `laya` SDK
  eval-only on CPU/GPU and run Luni's probe suite against the real checkpoints to build
  first-hand error intuition.
- **L1 (smoke, user-gated):** MiniLM-L12-H384 + Laya head, soft-CE only, typed-decisions
  train split (1,200 cases) — reproduce the notebook regime at 33M; target ≥ 0.587 (MiniLM
  specialist baseline) on the official test split; measure VRAM/pace with a probe config first.
- **L2 (pilot, user-gated):** add RLCD (mixed RL+CE) + temperature calibration; target ≥
  0.65 (ModernBERT-base level) at 33–50M; add Luni-style mixture for a generalist arm.
- **L3 (optional, user-gated):** scale to 50–100M (BERT-Medium/H512 or 12L/H512), from-scratch
  pretrain only if the user wants the full learning path (~1.5–2.5 weeks GPU).

## 10) Honest caveats & unknowns

1. Root-checkpoint training + data-generation code are unpublished; the "100% human-labeled"
   claim is unverifiable (no dataset list, no teacher name).
2. dev.to's "83.8% macro" covers 11 families; the HF eval file's 13-family overall is 0.753
   (two weak families excluded from the article table) — quote carefully.
3. RLCD group size / total training tokens for the root run: UNKNOWN.
4. No small-encoder scores exist on Laya's exact benchmark suites (estimates are GLUE/MNLI-
   anchored + the one typed-decisions scaling table).
5. Our GPU's fp16 throughput is unprobed for this workload; all time estimates are ranges.
6. Laya is Apache 2.0; Luni's fine-tuned checkpoint is CC-BY-NC-4.0 (mixture licensing).
7. The probe suite is 11 hand-written assertions over 2 examples — indicative, not a benchmark.

## 11) Primary sources

- Model family + cards: huggingface.co/convaiinnovations/laya (root, multilingual,
  typed-decisions subfolders; rl_agent_config.json, encoder configs, eval/results.{md,json})
- Code: github.com/NandhaKishorM/laya (SDK: laya/common.py, router.py; BENCHMARKS.md;
  notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) + HF rl_common.py
- Author write-up: dev.to (Nandakishor M) — "I Built Non-Autoregressive Decision Models…"
- Third-party: huggingface.co/datasets/Luni/laya-jev-benchmark (README, RESULTS.md, bench/)
  + huggingface.co/Luni/laya-grounded
- Benchmark data: huggingface.co/datasets/LocalLLaMA/typed-decisions (README: build process,
  baselines, splits)
- Small-encoder evidence: google/bert_uncased_* cards, google/electra-small-discriminator,
  microsoft/MiniLM-L12-H384-uncased, microsoft/deberta-v3-xsmall, arXiv 1909.10351 (TinyBERT),
  arXiv 2004.02984 (MobileBERT), google-research/electra README, ModernBERT card + issues
  (#163, #172; PR #35, #40; T4 design-target issues #6/#16/#17), arXiv 2204.08582 (MASSIVE)
- Raw copies: research/raw/laya/txt/ (17 files)
