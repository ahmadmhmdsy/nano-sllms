# STEP HANDOFF — KT-1 embedding-transplant A/B (knowledge-transfer ladder, Phase 1)

Written 2026-09-10. Scope: ONLY this step. Global narrative lives in HANDOFF.md;
ladder plan: [2026-09-10_kt_ladder_plan.md](./2026-09-10_kt_ladder_plan.md); live
statuses: TASKS rows 43-46 (row 43 = this step).

## State snapshot when written (2026-09-10, DSH session)

- Repo main = e08ec9a (fast-forwarded, tracked-clean); row41 resume pack restored
  from checkpoint_backup (sha256-verified; yarn_4096/final, yarn_lora_sft_v1
  final+final_a60, checkpoint-{750,1000,1172} all on disk).
- USER APPROVED the ladder: Q1 both (capability+research) / Q2 decide-after-KT-1
  numbers / Q3 GPU budget both (evenings + multi-day) / Q4 teacher 360M first.
- SmolLM2-360M DOWNLOADED (exit 0): C:\Users\AhmadMhmoud\.cache\huggingface\hub\models--HuggingFaceTB--SmolLM2-360M\snapshots\f8027fd0eaeea54caa13c31d31b9fdc459c38b49
- NOT STARTED YET: scripts/embed_transplant.py, train.py init_embeddings hook,
  configs/kt_ab_*.yaml, any GPU run. GPU idle at last check (RTX 3000, 6 GB).
- Disk ~19.8 GB free (fine: rotation-capped ckpts + finals ~4 GB total for both arms).

## Exact next steps, in order

1. **Write scripts/embed_transplant.py** per plan §4 (CPU-only, co-run-safe; prints
   alignment stats; saves data/kt/embed_init_smol360.pt). Gate: exact-match rate
   reported — investigate if < ~60%.
2. **Run it** (no GPU): & .\.venv\Scripts\python.exe scripts\embed_transplant.py
3. **Patch scripts/train.py**: config-gated train.init_embeddings (DEFAULT OFF,
   additive only — auto-resume contract untouched). Copy matrix into
   model.embed_tokens.weight after build_model, before maybe_wrap_peft; tied head
   follows automatically.
4. **Write configs/kt_ab_control.yaml + configs/kt_ab_transplant.yaml** per plan §4
   — identical except output dirs and the single train.init_embeddings key.
5. **GPU gate**: nvidia-smi idle; single GPU — arms run STRICTLY SEQUENTIALLY, never
   co-run. Launch CONTROL first (harness cross-check vs T's own early curve), then
   transplant:
   & .\.venv\Scripts\python.exe scripts\train.py --config configs\kt_ab_control.yaml
   & .\.venv\Scripts\python.exe scripts\train.py --config configs\kt_ab_transplant.yaml
6. **Any crash/sleep**: re-run the EXACT command, zero flags (auto-resume contract,
   PLAN §5.3). For unattended stretches: powercfg /change standby-timeout-ac 0
   first (restore after) — the 2026-09-10 sleep incidents are in HANDOFF.
7. **Compare + report**: eval_loss at matched steps from runs/kt_ab_*/logs
   tfevents; control must match T's own curve (@500 2.8567 / @1000 2.2650) as rig
   validation. Record delta in TASKS row 43, then bring the numbers to the user
   for the Q2 decision (keep our arch vs warm-start SmolLM2). Do NOT decide Q2
   unilaterally — it is user-gated by design.

## Gotchas

- DSH harness (this session): run_code->pwsh intermittently rejected valid calls
  ("missing required property description" / "binding arguments must be lossless
  JSON") — retry or use direct tools; did not block file work.
- transformers 5.16.1: no console loss lines (read tfevents), processing_class=,
  eval_strategy=.
- fp16 only (sm_75); tied embeddings — copying embed_tokens.weight updates the
  head via _tied_weights_keys.
- SmolLM2 weights are bf16: extract embeddings as fp32; our runs stay fp16.
- Teacher noise: any teacher-derived data later goes through sft_data.py AST gates.
