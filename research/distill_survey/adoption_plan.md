# Adoption plan — tracks A–H (2026-09-08)

Actionable successor of `REPORT.md` §2, integrated with
`research/training_analysis_2026-09-08.md` (the whole-history run analysis)
on the same day. Planning only: every GPU step below waits for an explicit
user go; implementation must never touch the auto-resume contract
(config-driven flags, defaults off; `sanity_check` gate after any `src/`/
script change; never co-run with a live train).

2026-09-08 addendum (COMPARISON round): Track H added; data-prep/judge
bullets added to Tracks D/G; pointer block added to Track E — all sourced
from the parallel-runs comparison (`COMPARISON.md` §7). Items from
`other_research/` are UNVERIFIED pointers until individually verified
per repo convention.

## Standing gates (from the training analysis — apply to EVERY track)

- **Three-surface eval, always**: in-dist quality (ast / own val) + OOD
  forgetting guard (CSN val @1024 vs the branch base) + execution credit
  (mini_eval). e2 looked best on eval loss and failed the guard (+19.7%) —
  never ship on one surface (analysis §3.4, §4.1).
- **1 epoch is the default SFT epoch count**; the first epoch buys nearly
  everything, later epochs buy forgetting (analysis §4.2). Re-derive LR/
  epochs per corpus (density lesson, §4.3) — never copy between corpora.
- **Pilot-first measurement**: S-scale/pilot runs are minutes-cheap
  (Tier 3 S run ≈ 3 min GPU); estimates were off 10-100× across the
  project (analysis §4.9). Measure, then scale.
- **Disk ≥ 5 GB free before any run with checkpoints** (~2.7 GB each at
  T-scale); never co-run GPU jobs; never kill for pace (analysis §6).

## A — Context eval probes (no training) — ~1-2 h GPU

> **RESULT 2026-09-09 (row 29 done, user go):** A-gate FIRED — NTK@2048
> = -4.17% (target) / -3.43% (e1) vs the 1024 baselines; NTK@4096 = +1.75%
> (base) / +3.9% (e1); raw noop extension +12-58%; StreamingLLM pos_shift
> remap collapses at every width (sink 4 ~= sink 0 — no sink
> specialization in a from-scratch 1024 model); window+sink mask with
> ABSOLUTE positions = near-baseline @2048 (-0.65%/+0.26%). **Decision:
> B targets ctx 4096 (YaRN factor 4); 2048/factor-2 fallback if the
> 8-bit-Adam vram_probe @4096 OOMs.** Full tables + provenance:
> track_a_result.md; runs/ctx_probes/20260909T184503Z/. Knobs shipped in
> src/model.py + eval.py flags (defaults off; sanity 4/4, selftest 5/5).

- **A1 Dynamic-NTK**: inference-only RoPE base scaling when seq > 1024.
  Eval-only knobs (`eval.ctx_override`, `rope.scaling: dynamic-ntk`); run
  on BOTH `runs/target/final` (pure-LM surface — cleanest CSN signal) and
  `runs/sft_v2_e1/final` (instruct surface); measure CSN val loss @ 2048/4096
  vs the 1024 baseline (fair eval = same tokens, packed at the longer ctx).
- **A2 StreamingLLM**: eval-only windowed attention mask (last-W tokens +
  first ~4 sink tokens), W ∈ {512, 1024}; same CSN protocol. Cheap test
  whether a from-scratch 226M even has usable attention sinks (open
  question in note 04).
- Files touched: `src/model.py` (rope-scaling + mask options, eval path
  only), `scripts/eval.py` (ctx override), one config. VRAM safe: no
  grads/optimizer; KV 512 MiB @32k (COMPUTED, note 04 §2).
- Gate → decision: if NTK @2048 already holds val loss within ~+2%, skip
  straight to a 4096 target in B; else B runs at 2048 as planned.

## B — YaRN fine-tune 1024→2048 → extrapolate 4k — ~2-4 h GPU

- Prereqs: A results + `vram_probe` at ctx 2048 with the 8-bit Adam default
  (row 11: frees ~1.8 GB vs fp32 AdamW; ctx-4096 training OOMs otherwise,
  note 04 §1).
- Design: config `rope.yarn {factor: 2, original_max: 1024}` (+ attention
  temperature per YaRN), train ~500-1000 steps on existing pilot/target
  tokens at ctx 2048, then eval @ 4096 (no further training). YaRN ≈
  NTK-by-parts at scale 2; published cost anchor ~400 steps @ 7B (REPORTED)
  → we budget ≤1k.
- **Base discipline**: fine-tune `runs/target/final` (the sacred pure-LM
  apex) in a NEW run dir; the artifact is a context-extended BASE — the
  e1/SFT-v3 instruct recipe re-applies on top afterwards if it wins. The
  base itself is never touched in place (analysis §1.3, §6.4).
- Files: `src/model.py` (YaRN rope re-param, config-gated),
  `configs/yarn_2048.yaml`, reuse auto-resume as-is.
- Gates: val loss @2048 ≤ 1024-baseline +2%; forgetting guard on
  held-out @1024; 4k extrapolation reported honestly (expected worse than
  2048; memory-module track is the long-term answer, note 04 §4).

## C — T-as-teacher KD + skew-KL — ~3-4 h GPU

- Unblocks the tokenizer wall (note 01 §5): teacher = 226M `runs/target/final`
  (the best pure-LM artifact — KD here runs on packed CSN tokens, a pure-LM
  surface; `runs/sft_v2_e1/final` = optional instruct-KD arm C1b on Evol
  pairs, later), student = P-arch (110M), SAME CodeLlama tokenizer → logit
  KD defined. Teacher fp16 ≈ 0.45 GB.
- C1 arms (identical data/steps/seed, 1k-2k steps @ batch 8, target tokens):
  baseline P pretrain vs KL(student‖T, τ=1) 0.5*CE blend (row 20 recipe,
  scaled up). C2: same with **DistiLLM skew-KL** (skew λ=0.1) replacing
  plain KLD — drop-in loss upgrade in `scripts/kd.py`, A/B'd separately.
- Success gate mirrors row 20: distilled ≥ baseline at ≤1/3 steps →
  P-scale rungs become distilled instead of pretrained (~1/10 claim at
  scale). Opens later: true on-policy KD (GKD student-sampling reverse-KL
  vs T logits) as C3 — design doc only, not in this plan.
- Files: `scripts/kd.py` (skew-KL option), `configs/kd_t2p*.yaml`.
- **RESULT 2026-09-09 (Track C executed — TASKS row 31; GATE PASS, plain-KD
  adopted, skew-KL rejected)**: teacher = frozen 226.5M `runs/target/final`,
  student = P-arch 12L/768 **100.68M** (not the ~110M estimate above), data =
  `data/target/tokens` CSN ctx 1024, 2000 steps, batch 4×accum 2 (effective 8),
  adamw_bnb_8bit (row 11 default; deviation from row 20's adamw_torch is
  deliberate and shared by all arms → internally fair), seed 42, eval+save 100.
  **Baseline 2.9453 → plain-KD 2.7476 @2000 (−6.71%)**; KD ahead at EVERY
  matched step (100: −2.0%, 500: −2.5%, 600: **−3.05%**, 1000: −4.0%, 1500:
  −6.4%) → the §5 gate "distilled ≥ baseline at ≤ 1/3 steps" **PASSES at P
  scale**; future P rungs can be distilled. The gap persists at only 2.25×
  compression (row 20's was 8.2×) — the advantage is not a compression artifact.
  **C2 skew-KL (DistiLLM α-SKL, λ=0.1, CPU-validated exact vs brute-force
  mixture-KL to ≤2.4e-7): 2.8104 @2000 = +2.29% vs plain-KD — REJECTED.** It
  led only in the 800–1000 window (−0.6/−1.0%) and lost late; both KD arms'
  end-of-run grad norms healthy (1.0–1.6) → no fp16 instability for the skew
  floor to fix; `skew_lambda=0` (byte-identical legacy path) stays the default.
  VRAM ladder (fp32 KD logits dominate): micro-b8 7897/8192 (96%) → b4/a2
  full-cycle 7010 (86%) chosen; skew chunked-per-sample 7447 (91%). 3 full
  arms, ~3.5 h GPU total incl probes, 0 OOM, 0 interruptions. Evidence:
  `runs/kd-t2p-{baseline,kd,kd-skew}/final/{train_summary.json,eval_curves.json}`.

## D — Tier 2 V1 → V2b (SODA-style preference rerank) — 4-6 h impl + ~1 h pilot GPU

- V1 contract stays frozen (row 14): student samples K=8, temp 0.8;
  Qwen3.5-0.8B judge text-tower, strict-JSON verdict schema, 5-step parse
  ladder, verdict jsonl + agreement-rate gate. Teacher inference always
  STANDALONE (never co-run).
- V2b delta: from the SAME one-time sampling snapshot (SODA's trick — no
  adversarial loop), build preference pairs per prompt from judge scores:
  winner trace → rerank-SFT (as V1) PLUS store pairs. Train arms:
  (a) winner-only SFT (V1 baseline), (b) pair-weighted SFT (winner ↑,
  loser down-weighted; true DPO blocked on missing `trl` — revisit later).
  Hard AST/code gates before any pair enters the corpus (existing ast
  filter + parse ladder).
- Data: 2k code prompts × 8 candidates from a MIXED prompt pool (held-out
  Evol instructions + generic-phrasing prompts) — the analysis diagnosed
  e1's disease as narrow distribution (repetition loops off-corpus), so the
  judge/rerank data itself must widen the phrasing distribution, mirroring
  the SFT v3 idea (analysis §1.2, §5.1, §5.3). Hard AST/code gates before
  any pair enters the corpus (existing ast filter + parse ladder).
- Training recipe: rerank-SFT at **1 epoch, pilot-first** (minimax3 e2
  proved 2 epochs overwrite the base on dense corpora; analysis §4.2/§4.3);
  three-surface gate per Standing gates.
- Data prep additions (2026-09-08 COMPARISON round; CPU-only): MinHash +
  semantic dedup and decontamination against CSN val / held-out AST
  prompts before any pair enters the corpus; keep a small real/human slice
  in the mix (production checklists REPORTED 10–50% — unverified, treat as
  direction, not number). Widens the phrasing distribution the analysis
  diagnosed as e1's disease and guards repetition loops; shared with
  Track G's pass.
- Judge-prompt option (UNVERIFIED pointer, from Perplexity): ROPD-style
  structured rubric — score dimensions (correctness / syntax /
  completeness) instead of a single verdict, mapped onto the V1
  strict-JSON schema. Adopt only if it improves the parse ladder /
  agreement-rate calibration (note 02 open question 1), not for the name.
- Cloud-teacher compliance (from Claude; was missing everywhere): check
  the provider's ToS on training-from-outputs before ANY cloud
  judge/teacher spend (V2c or a cloud judge); cost figures stay ESTIMATE.
- Files: `scripts/tier2_judge.py` (judge + snapshot + pair builder),
  `data/sft/tier2/`, `configs/sft_t2b.yaml`.

## E — Stage/LR revision for the NEXT pretrain (row 13 successor) — planning now, runs in the next pretrain window

- Budget split of the ~134M-token D-mix: bulk ~80M (current 30/55/10/5)
  → mid-train ~27M (Evol-instruct share ↑ to 10-12%) → anneal ~27M (HQ
  code + Evol + CSN upsample). Evidence: SmolLM2/3 stage mixes, note 03 §2.
- **LR fix (the must-do)**: replace cosine→0 with
  `cosine_with_min_lr` (min_lr_rate ≈ 1/3) — verifies the LR-decay-kills-
  curriculum finding (2511.18903/TREC) — OR constant-LR + CMA
  (checkpoint-average of last 3 via existing rotation; +1.64% avg,
  REPORTED). Decide at implementation; CMA script is ~30 lines, CPU.
- **Knee-stop**: stop ~1 epoch past the eval knee instead of running to
  budget — M3 spent 2,000 extra steps (~40% of the run) for +1.7% eval
  (analysis §3.1, §5.5); `load_best_model_at_end` already catches the
  floor, the budget just shrinks.
- Optional MTP probe arm (separate, after main arms): depth-1 head,
  λ=0.1, 500 steps, config-gated in `src/model.py`+`train.py`; discarded
  at inference. No sub-1B evidence exists → probe only (note 03 §4).
- Files: `configs/pretrain_v2.yaml` (multi-source wiring already planned
  in row 13), optional `scripts/ckpt_average.py`, optional MTP flag.
- Recorded pointers, NOT planned (2026-09-08 COMPARISON round; verification
  update: the Flash-Next facts are VERIFIED on its HF card,
  `ver7_flashnext_training.json` — "The Muon and AdamW optimizers are
  applied to specific weight categories… eliminate traditional batch-size
  warmups and start directly at the target batch size"; QAD VERIFIED, see
  note 01 §6): Muon optimizer; no-batch-size-warmup; QAD's
  teacher-regenerated fresh-data principle. All REPORTED at MoE/frontier
  scale with zero ≤1B evidence; none enters a config without the 500-step
  A/B discipline first; Muon additionally has no sm_75/fp16 track record
  in any payload.

## F — LoRA-SFT variant of SFT v2 (from the analysis' ranked list) — ~1-2 h GPU

- The analysis' #2 recommendation and the cheapest medicine on the list:
  same minimax3 corpus, frozen base via the live row-10 LoRA hook
  (measured: 0.72 GB peak @ 2.52% trainable). Predicted: forgetting
  ~0-2% (frozen base ⇒ guard can barely move), instruct quality somewhat
  below e1 (full-FT). If it lands near e1's ast, it becomes the default
  SFT recipe (cheap + safe + mergeable).
- Files: configs only (`configs/sft_v2_lora.yaml` — the row-10 hook +
  `build_sft_config` peft block already exist; no src changes).
- Gates: three-surface standing gate; compare against e1 (ast 0.98/0.96,
  forgetting +9.8%).

## G — SFT v3 mixed-corpus (from the analysis' ranked list) — CPU prep + ~1 h GPU

- The analysis' #3 recommendation, the direct anti-repetition-loop move:
  Evol + minimax3 + a slice of generic-phrasing instructions → widen the
  instruction distribution e1 degenerates on. 1 epoch, LR re-derived per
  the density lesson (pilot-first).
- Shares the mixed prompt pool with Track D's judge data (one curation
  pass feeds both).
- Prep additions (2026-09-08 COMPARISON round; CPU-only): MinHash/semantic
  dedup + decontamination on the mixed corpus — ONE shared pass with
  Track D (bullet above), then the existing sft_data.py generic path;
  production-ops checklist imported from `other_research/`
  (COMPARISON §5).
- Files: data prep only (sft_data.py generic-instruction path exists) +
  `configs/sft_v3.yaml`.
- Gates: three-surface standing gate; specifically the config's
  qualitative prompts (the surface where e1 loops) must improve.

## H — Agent-side memory experiment (zero training) — added 2026-09-08 (COMPARISON round)

- What: the only proposal across all four research runs that answers the
  user's "remember after we wipe the context" ask with ZERO model changes
  (Claude context README Tricks 2+3; Perplexity guide §7.2 Option A):
  rolling summary + external store. Keep the last ~300–500 raw tokens plus
  a 150–250-token running summary inside the 1024 budget; periodically
  extract durable facts to an external JSON/SQLite store (prompted
  extraction); on a fresh session, retrieve top-k stored facts and inject
  them into the prompt before generation.
- Honest framing: orchestration memory, NOT model memory — fidelity is
  summarization-limited and injected facts cost real context budget. The
  trained in-model path stays Track C; H is the cheap complement that also
  builds C's eval.
- Why it earns a track: no VRAM change, no training run, no `src/`
  change (prompt assembly lives in the infer path), and its eval IS the
  harness Track C needs — the two-session recall eval (note 04 §7 item 5:
  tell a fact in session 1, wipe context, restore in session 2, ask for
  the fact) runs here first and is reused unchanged if C ever trains.
- Sources: **VERIFIED 2026-09-08** (`research/raw/ver7_memgpt_mem0.json`):
  MemGPT (arXiv 2310.08560 — virtual-context paging, the "illusion of an
  infinite context" on fixed-context models) and Mem0 (arXiv 2504.19413 —
  production agent memory with extract/update/search modules); Perplexity's
  external-memory pseudocode is the implementation sketch. The token
  budgets are pattern defaults, not measured for our model.
- Cost: CPU prep/store code + short eval-class GPU inference windows
  (minutes-scale, same class as Track A probes); never co-run with a live
  train (AGENTS.md §4).
- Files: infer-path prompt assembly (new helper script or a
  `scripts/infer.py` option) + a JSON/SQLite store under `data/`
  (gitignore decision at implementation); no `src/model.py` change, no
  config-gated training flag; auto-resume contract untouched.
- Gate: two-session recall passes on ≥2 facts across a context swap; no
  regression on single-session mini_eval prompts; report the
  summary+facts token overhead honestly (150–250 of the 1024 tokens are
  taken from the working window).
- **RESULT 2026-09-08 (executed, 3 GPU iterations on runs/sft_v2_e1/final)**:
  mechanism validated — store 6/6, retrieval 6/6, model-mode summary 7/7
  folds (61 tok), regression PASS (ast 4/4 preamble vs 2/4 plain),
  overhead ~7.4% of ctx — but gate_recall FAIL: copy-out of injected
  facts 1/6 (f3 only, deterministic across runs); needle-style completion
  probe 0/6; a "no code" format cue made it worse (0/6). The wall is the
  student's copy-out (MEMORY lesson 18), not the memory layer — the
  trained path (Track C) or a stronger student is the fix. Evidence:
  runs/agent_memory_h/report.json (+ iter2_qa_cue/ snapshot),
  scripts/agent_memory_eval.py, data/agent_memory/store.json (gitignored).
- **ADDENDUM 2026-09-09 (Track H2 Phase 1 fix executed - gate_recall PASS)**:
  copy-behavior LoRA-SFT (TASKS row 37 Phase 1: 585 subagent-authored copy
  pairs, LoRA r=16, 70 steps) flipped the gate - store arm 3/6 (probe 3/6,
  qa 2/6, controls 0/6; required >= 2; baseline 1/6), retrieval 6/6. Code
  guards: CSN +0.13% (near-zero forgetting), ast greedy 0.90. Honest deltas:
  the Track H regression PROBE collapsed to 0/4 ast in BOTH arms (gate =
  preamble >= plain still PASSes) - the QA-style template now elicits
  copy-tautologies while the SFT-template ast stays 0.90; the model-mode
  summarizer stopped folding (7/7 -> 0), so the mechanism runs
  extract/store/retrieve/copy only (overhead 1.3% of ctx); 2 of 3 recall
  misses came from the prompted extractor corrupting stored facts ("March
  15" -> "ISO 15", wifi value echoed) - deterministic-only extraction is
  the obvious next mechanism patch. Evidence:
  runs/agent_memory_h2_rerun/report.json.
- **ADDENDUM 2 2026-09-09 (Track H2 Phase 2 executed — all gates PASS,
  recall at threshold)**: mixed corpus = 585 copy + 947 general-QA (371
  dolly open_qa + 223 no_robots prose cats + 353 OpenHermes-2.5 GPT-4
  slice, user-sanctioned "more powerful datasets"; NO local teacher; NOT
  Qwen3.5-0.8B). LoRA r=16 from e1, 184 steps, eval 4.9461. Gates: CSN
  2.0077 vs e1 2.0466 = **-1.9% (improved)**; ast greedy 0.86 >= 0.85
  (sampled 0.78); Track H rerun on the new final = **2/6 BOTH arms**
  (prompted 2/6, deterministic 2/6; control 0/6; Phase 1 was 3/6 — the
  copy share fell 100% -> 38%). The deterministic arm stored all 6
  scripted facts VERBATIM (incl "March 15" + "Pineapple42", the exact two
  the prompted extractor corrupted in Phase 1; f6 wifi qa flipped to
  PASS) — extractor corruption FIXED; remaining wall = pure copy-out.
  Honest: summarizer still 0 folds; regression gate PASSes trivially
  (0/4 = 0/4); canned-prompt tautologies persist (43/50 greedy) — the QA
  mix did NOT visibly cure the collapse at this LoRA scale. Follow-up
  user-gated (TASKS row 38): accept, higher-copy-share variant, stronger
  student, or Track C. Evidence: runs/agent_memory_h2p2_prompted/,
  runs/agent_memory_h2p2_deterministic/, runs/h2p2_mixed_lora/final.

## Order & independence

A → B are sequenced (A's result picks B's target ctx). C, D, E-prep are
independent and can interleave in free GPU windows (one at a time,
never-co-run rule). E's training waits for the user's next-pretrain go.
F and G are the cheapest wins on the board (F = analysis' #2; G = #3) and
slot into any free window; G's data prep is CPU-only (co-run safe).
H is independent of A–G: it needs only short eval-class GPU inference
windows (never co-run) with all prep/store code CPU-side, and its recall
eval is the same harness Track C would reuse.

## Suggested first window

Two equally defensible openings — pick by goal:
- **Fastest medicine**: G (CPU prep + ~1 h) or F (~1-2 h) — both attack
  the diagnosed narrow-distribution/forgetting problems directly
  (analysis §5.2/§5.3).
- **Strategic**: A (1-2 h, eval-only) then C1 (3-4 h) — A is the lowest
  risk, and C's same-tokenizer T→P KD is the cleanest strategic proof
  (distill-vs-pretrain at P scale) that unblocks future rungs.
- **Cheapest new experiment**: H — no training at all, minutes-scale
  inference windows (schedule like A); opens the memory question without
  touching the model.
