# Cross-check: Qwen3.8-Flash-Next research findings vs the current nano_SLMs pipeline

**Date:** 2026-09-06. **Inputs:** research/qwen3_8_flash_next_research.md (sources S1-S14) checked against
src/model.py, src/data.py, scripts/train.py, configs/{pilot,target}.yaml, PLAN.md, HANDOFF.md, and live
telemetry from runs/target (M3) and runs/pilot (M2). The M3 run was only ever read, never touched; the
auto-resume contract (PLAN.md 5.3) is unaffected. Complementary to the research doc: this file judges the
findings against OUR code, not against the web sources.

## 0. Verdict table

| # | Research finding (research doc section) | Current pipeline (code evidence) | Verdict |
|---|---|---|---|
| C1 | v0 = plain GQA decoder; Flash-Next hybrid deliberately deferred (research doc 6; PLAN.md A7) | src/model.py builds LlamaForCausalLM (GQA 16Q/4KV at M3) with SDPA - exactly that v0 | CONSISTENT (deliberate) |
| C2 | resources/ GDN/QSA skeletons are wrong pseudo-code (research doc 6.4) | model.py docstring already rejects them ("those are pseudo-code, this is battle-tested") | CONFIRMED - research now proves why |
| C3 | Muon on 2-D linear maps + AdamW on the rest; split fused qkv/fc1 before Newton-Schulz (research doc 4.2) | single AdamW (beta 0.9/0.95, wd 0.1, optim=adamw_torch) for everything | CONSISTENT at nano (Muon = optional M4 experiment). Note: HF Llama keeps q/k/v/o and gate/up as SEPARATE Linears, so the "split before NS" rule is automatically satisfied if Muon is ever added |
| C4 | batch-size warmup is dead; start directly at target batch (research doc 4.3) | batch 1 x accum 32 constant from step 0 (pilot and M3 both) | MATCH |
| C5 | refitted scaling laws push optimum LR UP under gates+Muon (research doc 4.3) | lr 4e-4, cosine (pilot converged cleanly; M3 running) | PARTIAL - 4e-4 is sane today; the evidence-backed 4e-4 vs 7e-4 sweep is an M4 action |
| C6 | LR warmup still used by Qwen (only BATCH ramping was abolished) | warmup_steps 150 (3% of 5000) | MATCH - do not confuse the two warmups |
| C7 | stability: zero spikes in production; max pre-clip grad norm ~28% of threshold post-warmup (research doc 4.4) | pilot (full 3000 steps): grad_norm p50 0.532, mean 0.578, max 1.149 vs clip 1.0 - clip effectively idle post-warmup, zero spikes. M3 so far (still in LR warmup): 1.27-1.39, clip engaged - normal while LR ramps | CONSISTENT (pilot evidence); re-check M3 norms at step 500+ |
| C8 | zero-centered RMSNorm, which RELIES on weight decay applied to norm weights (research doc 3.1) | HF Trainer 5.16.1 excludes any param whose name matches rmsnorm/norm/bias from decay (trainer.py:1308-1318), and LlamaRMSNorm initializes weight at 1.0 | PORT DETAIL - if zero-centered RMSNorm is ever ported, it must ALSO be un-excluded from weight decay, or the zero-init has nothing anchoring it |
| C9 | 3:1 GDN : full-attention hybrid = the top nano-feasible port; port from local reference (research doc 6.1.1) | not implemented; transformers 5.16.1 ships models/qwen3_next (with Qwen3NextGatedDeltaNet) + qwen3_5, qwen3_5_moe, qwen4_exp locally - verified on disk | ACTIONABLE post-M3 (the A7 upgrade path) |
| C10 | Gated Residual (GR) = the cheap stability lever (research doc 6.1.2) | not implemented (plain residual + pre-norm) | OPTIONAL post-M3 |
| C11 | skip QSA / n-gram table / ultra-sparse MoE / MTP at nano (research doc 6.2) | all absent from the pipeline | MATCH |
| C12 | small models become useful via strong-to-weak DISTILLATION, not more pretraining (research doc 5.1, 6.3) | pipeline ends at pretraining: eval = val ppl + 3 code prompts; no SFT/distillation stage exists anywhere in scripts/ | GAP - the biggest missing piece if the goal is an instruction-capable nano model |
| C13 | flagships train ~1 pass over trillions of unique tokens (research doc 5.1) | M3 = 3.4 epochs over 47.9M tokens (deliberate, HANDOFF 3b: M2's eval curve was still falling at 2.35 epochs; load_best_model_at_end guards the downside) | ACCEPTED DEVIATION (nano data-bound) |
| C14 | research doc 6 numbers: S 12.3M / P 100.7M / T 226.5M params; 47.9M train tok; 164M-token budget | configs + vram_probe (226.53M) + pilot train_summary (100.68M) + HANDOFF all agree | CONFIRMED |

## 1. Live M3 snapshot (read-only, 2026-09-06 00:07)

- Progress: step 106/5000 (~2%), sustained 23.7 s/it. ETA = ~33 h total -> completion ~2026-09-07 morning.
  HANDOFF's launch estimate (13-14 h from ~9.6-12.7 s/it) was optimistic; the 2,198 tok/s probe number did
  not hold. HANDOFF 2 + 3b corrected this turn.
- train/loss 8.06 @50 -> 5.60 @100 (healthy early descent; pilot had 8.22 @50).
- LR still ramping (1e-4 @50 -> 3e-4 @100 -> 4e-4 @150); grad_norm 1.27 -> 1.39 with clip 1.0 engaged -
  expected during warmup. The meaningful check is the settled post-warmup norm (pilot settled at ~0.53x
  threshold); first checkpoint + first eval land at step 500 (also the eval_batch-2 VRAM watch point).
- GPU at check time: 77 %, 4958 MiB, 75 (no OOM risk regime; eval VRAM watch stays at step 500).

## 2. Already aligned - needs NO action

C1 (v0-first order), C2 (pseudo-code rejection), C4 (constant batch), C6 (LR warmup), C11 (skip-list),
C14 (doc numbers). Nothing in the running M3 should change: the research doc's findings are either
already matched or are post-M3 upgrades.

## 3. Real deltas -> ordered action list (all POST-M3; gated on user approval per HANDOFF 8.3)

1. LR sweep 4e-4 vs 7e-4 - cheapest, directly evidence-backed (research doc 4.3); two short pilot-config runs.
2. GDN 3:1 hybrid port (research doc 6.1.1) from the locally installed qwen3_next; plain PyTorch/SDPA is fine
   at ctx 1024 (custom GDN kernels only matter for long context).
3. GR + zero-centered RMSNorm together with the port (research doc 6.1.2-6.1.3), including the C8 fix
   (un-exclude norm weights from decay) - bundle in one M4 config so stability effects are attributable.
4. Muon experiment (optional; research doc 6.2): only on 2-D linear maps, AdamW on embeddings/head/gates/scalars;
   HF's unfused projections make the split rule free.
5. Distillation SFT stage (research doc 6.3): teacher-trace SFT (+ on-policy logit alignment later) - the step
   that turns the code-completer into something instruction-capable at ~1/10 the cost of any RL stage.

Hard rule while M3 is live: no edits to src/, configs/, or the training process. M4 work starts only after
M3 completes (eval.py + metric commit per HANDOFF 8.2).

## 4. Method / how to re-run

All read-only: git status; EventAccumulator over runs/target/logs and runs/pilot/logs (tags train/loss,
train/grad_norm, train/learning_rate); the file reads listed in the header; nvidia-smi. The decay-exclusion
behavior is verified at trainer.py:1308-1318 - re-check after any transformers upgrade
(grep get_decay_parameter_names in .venv/Lib/site-packages/transformers/trainer.py).