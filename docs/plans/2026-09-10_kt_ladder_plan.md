# Knowledge-Transfer Ladder — reusing SmolLM2 for our architecture (KT ladder)

Date: 2026-09-10 · Status: **USER-APPROVED 2026-09-10** · Step handoff: [2026-09-10_kt_ladder_handoff.md](./2026-09-10_kt_ladder_handoff.md) · TASKS rows 43-46

## 1) Problem this solves

The M3 base (runs/target/final, 226.5M) is ~30-100x undertrained and single-domain:
CodeSearchNet Python only, 47.9M unique tokens consumed as ~164M token-steps
(5000 steps x 32,768 tok/step), vs a Chinchilla-class budget of ~4.5B tokens for
226.5M params. SFT/e1/LoRA are behavior layers on that ceiling: ast 0.98 measures
SYNTAX validity, SFT val 0.28 measures fit to teacher traces, the forgetting guard
measures preservation — none measure capability. The user diagnosis (2026-09-10):
"a poor base stays poor through SFT" — confirmed against the artifacts.

## 2) User decisions (record, 2026-09-10)

- Q1 objective: **BOTH** capability and research value.
- Q2 architecture: **decide after seeing KT-1 (Route 3) A/B numbers** — user may later
  put SmolLM2 itself as the base, but first wants evidence on whether OUR
  architecture is good.
- Q3 GPU budget: **BOTH** evening windows and multi-day unattended pretrain windows.
- Q4 teacher size: **SmolLM2-360M first**, upgrade to 1.7B only if gates say so.

## 3) Physics — what cannot transfer

Teacher = HuggingFaceTB/SmolLM2-360M: LlamaForCausalLM, 32L/d960/15Q/5KV, vocab
49,152 (Llama-3 BPE), tied embeddings, Apache-2.0. Ours: 16L/d1024/16Q/4KV, vocab
32,768 (CodeLlama sentencepiece), tied, ctx 1024 (YaRN 4096 lineage exists).

- **No weight merging/souping across tokenizers** — the row-41 soup trick is
  same-vocab only.
- **No naive logit KD across tokenizers** — this is the TASKS row-14 blocker that
  froze Tier-2 V1 ("logit KL undefined cross-tokenizer").
- What CAN transfer: **through data** (always), **through aligned distributions**
  (Phases 2/4), **through aligned weights** (Phase 1).

## 4) Phase 1 (KT-1) — embedding transplant A/B — the evidence run

Goal: measure whether SmolLM2's input-space knowledge accelerates OUR architecture
from scratch, and produce the numbers that decide Q2.

Teacher asset: DOWNLOADED 2026-09-10 (exit 0) ->
C:\Users\AhmadMhmoud\.cache\huggingface\hub\models--HuggingFaceTB--SmolLM2-360M\snapshots\f8027fd0eaeea54caa13c31d31b9fdc459c38b49

Method (scripts/embed_transplant.py — NEW, CPU-only, co-run-safe):

1. Load teacher embed_tokens.weight [49152, 960] as fp32 (tied = same storage as
   its lm_head; SmolLM2 is bf16-trained — cast at extraction).
2. Token-string alignment our 32,768 CodeLlama SP pieces <-> SmolLM2 49,152
   Llama-3 BPE pieces: normalize SP "▁" vs BPE "Ġ" space conventions, exact
   normalized-piece match, collect (our_id, teacher_id) pairs; report match rate
   (investigate if < ~60%).
3. Fallback for unmatched pieces: re-tokenize OUR piece with the SmolLM2 tokenizer
   (add_special_tokens=False) and average the matched sub-embeddings (OMP-lite;
   cf. training-free tokenizer transplantation, arXiv 2506.06607).
4. Lift d960 -> d1024 with a fixed seeded ROW-ORTHONORMAL matrix W (1024x960):
   preserves pairwise geometry/norms up to rotation (JL-style; FOCUS-style
   embedding warm start, EMNLP 2023).
5. Save data/kt/embed_init_smol360.pt = {embed_tokens: [32768,1024] fp32, meta:
   match stats, seed, teacher snapshot hash}.

A/B design (arms identical EXCEPT init; single-variable experiment):

- configs/kt_ab_control.yaml and configs/kt_ab_transplant.yaml — copies of
  target.yaml: 226.5M dims, data/target/tokens (CSN, on disk), max_steps 1000,
  batch 1 / accum 32 (32,768 tok/step), lr 4e-4 cosine, warmup 150, seed 42,
  adamw_torch, fp16, grad_ckpt, eval/save every 100, save_total_limit 3,
  output runs/kt_ab_control vs runs/kt_ab_transplant.
- scripts/train.py patch: config-gated train.init_embeddings (path; DEFAULT OFF —
  same additive contract style as Track B's init_from). Copy the saved matrix
  into model.embed_tokens.weight after build_model, BEFORE maybe_wrap_peft; the
  tied head follows via _tied_weights_keys. Auto-resume contract untouched.
- Harness cross-check: the control arm must reproduce T's own early tfevents
  curve (@500 = 2.8567, @1000 = 2.2650) within noise — proves the A/B rig.
- VRAM: M1 probe says 4.24 GB alloc @1024 for these dims — fits the 6 GB card.
  Pace ~2214 tok/s -> ~4-5 h per arm; run arms STRICTLY SEQUENTIALLY (single GPU).

Deliverable/gate: eval-loss curves at matched steps (runs/kt_ab_*/logs tfevents).
A >=3-5% loss reduction at step 1000 = meaningful warm start (any positive delta
informs Q2); instability or regression = transplant fails, Q2 leans warm-start.

Decision hooked here (Q2): keep our architecture (continue ladder on it) vs
warm-start SmolLM2-135M weights outright (Route 0). USER DECIDES AFTER NUMBERS.

## 5) Phase 2 (KT-2) — SmolLM2-360M as on-policy judge — Route 2

Student samples -> SmolLM2-360M scores/ranks -> winners become SFT pairs. This is
the Track D V1 judge-rerank contract (frozen row 14/32) with a code-capable
teacher; knowledge flows through SCORES so tokenizer mismatch is a non-issue.
360M fits the 6 GB card beside a 226M student; upgrade to 1.7B on the 8 GB card
or via 8-bit (bitsandbytes on sm_75 verified in the Tier-2 brief). sft_data.py
AST/filter gates mandatory — a 360M teacher still hallucinates code. Runs after
KT-1 (uses the winning base); pairs with TASKS row 42 corpus quality.

## 6) Phase 3 (KT-3) — pretrain OUR arch on the public mix — Route 1

next_pretrain.yaml is already wired (Milestone D mix mode: starcoder 62.5% /
the-stack-smol 20.3% / CSN 11.5% / Evol 5.7% = ~133.65M kept tokens at pilot
dims). Extend candidates with fineweb-edu / cosmopedia-class corpora and re-size
for 226.5M dims. Multi-day unattended window (approved), standby guard
(powercfg /change standby-timeout-ac 0) mandatory, disk-watch row-2 rules apply.
Honest note: the 134M-token class is still small-data; a real ceiling lift wants
>=1B tokens — the SmolLM2 paper's core finding is that DATA QUALITY dominates at
small scale, which is what makes this route worth it even token-lean.

## 7) Phase 4 (KT-4) — ULD-style cross-tokenizer logit KD — Route 4 (research spike)

Universal Logit Distillation (arXiv 2402.12030) makes logits comparable across
vocabularies via embedding-space alignment; newer refinements: contextual
dynamical mapping (ACL 2025 Findings), DWA-KD (EACL 2026). scripts/kd.py already
has the frozen-teacher + custom compute_loss scaffolding — add an alignment
projection in front of the KL. Payoff: ANY open model (SmolLM2 360M/1.7B, the
on-disk Qwen3.5-0.8B) can supervise ours token-by-token at ~1/10 GPU-hours, and
Tier-2 V2 un-freezes. Risk: alignment quality decides everything.

## 8) Parked / rejected

- Route 5 model stitching (30L/d576 vs 16L/d1024, different norms/rope — low ROI
  at our scale): parked.
- Route 0 warm-start SmolLM2-135M outright: NOT rejected — it is exactly the Q2
  fallback if KT-1 numbers favor it.
- Cross-tokenizer weight souping: impossible (§3 physics).

## 9) Citations

- ULD — Towards Cross-Tokenizer Distillation: https://arxiv.org/abs/2402.12030
- Training-Free Tokenizer Transplantation via OMP: https://arxiv.org/abs/2506.06607
- On-Policy Distillation of Language Models (GKD): https://arxiv.org/abs/2306.13649
- SmolLM2 — data-centric training: https://arxiv.org/abs/2502.02737
- Model stitching (bridging representation gaps): https://proceedings.mlr.press/v322/traft26a.html
- FOCUS — embedding initialization across vocabularies: https://aclanthology.org/2023.emnlp-main.829.pdf
- DWA-KD (EACL 2026 Findings): https://aclanthology.org/2026.findings-eacl.181.pdf
- Cross-tokenizer KD via contextual dynamical mapping (ACL 2025): https://aclanthology.org/2025.findings-acl.419/

## 10) Risks / honest caveats

- 360M teacher noise: AST/filter machinery is mandatory on any teacher-derived data.
- Embedding warm start is not magic: geometry survives to first order, the
  transformer still has to adapt — the A/B measures exactly this, honestly.
- fp16-only Turing card; teacher weights are bf16 — fp32 at extraction, fp16 ours.
- Thermal throttle pace swings are normal — never kill a run for pace; sysmem
  fallback masks OOM on the 6 GB card (MEMORY 32).
- Single GPU: KT-1 arms strictly sequential; only the transplant script is co-run-safe.
- Session gotcha (2026-09-10, DSH harness): run_code->pwsh binding intermittently
  rejected valid calls ("missing required property description" / "binding
  arguments must be lossless JSON") — retry or fall back to direct tools; did not
  block any file work.
