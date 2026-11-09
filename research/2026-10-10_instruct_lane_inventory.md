# The instruct lane: what "e2" is, what survived, and what it means for the plan

Date: 2026-10-10. Trigger: user asked to check "the other model, I think its name is e2, more
powerful, not a base, near instruct mode" and whether the plan should shift.

## 1. What e2 actually is

e2 = `runs/sft_v2/final` - the **2-epoch** SFT v2 run of the 226.5M M3 target on the minimax3
distillation corpus (25,146 raw pairs -> 19,252 kept at min_chars 30, prompt-masked, ast_filter).
2,344 steps.

- instruct surface: ast greedy 0.98 / sampled 1.00; eval 0.2421 (best_metric 0.2421 @2250,
  0.2433 @2000); checkpoint val_loss 2.2308 @2000 / 2.2323 @2250 (ppl 9.31 / 9.32).
- general ability: CSN forgetting **+19.7% = gate FAIL**.
- TASKS row 19 kept it deliberately as overfit evidence; the project analysis calls it "the single
  most instructive result" - not the deliverable.

The sibling **e1** = `runs/sft_v2_e1/final` was the DELIVERABLE: 1 epoch, 1,172 steps, ast greedy
0.98 / sampled 0.96, CSN 2.0466 = **+9.8% WITHIN gate**. So "e2" is not a stronger model than the
base - it is the *more overfit* one: it wins on the instruct surface and loses on general ability.

## 2. Neither is on disk today

- `runs/sft_v2` (e2) and `runs/sft_v2_e1` (e1): both absent (Test-Path False).
- The 2026-09-09 user-approved deletion of `checkpoint-sft_v2_e1-final.zip` (868 MB) was justified
  as "CRC-verified redundant vs disk + pack". The pack it was redundant with - the 2026-09-09
  machine-move resume pack, sha256 `ceda94adbce10e7af5bcd85a7b58fbbe550072bd1f02bf0c91d68437f705bd4e`,
  which held all ten weights-bearing finals including `sft_v2_e1` - is **also absent** from this
  machine. `checkpoint_backup/` now holds only the glm53-flash E116 pack (1,351 MB).
- Surviving evidence (numbers reproducible, weights not):
  `runs/evidence/2026-09-08_cleanup/sft_v2/` (final config/eval_report/train_summary, checkpoint-2000
  and -2250 eval_reports, checkpoint-2000/2250/2344 trainer_states) and
  `runs/evidence/2026-09-16-tierA-cleanup/sft_v2/` (train_out.log, train_err.log, tfevents).
- Recovery is cheap either way: a copy from the other machine if the pack still exists, otherwise a
  retrain from `configs/sft_v2_e1.yaml` (1,172 steps) on the corpus that is still on disk
  (`data/distillation_data/minimax3/combined/all_train.jsonl`, `data/sft/minimax3/ds`).

## 3. What actually survives in the instruct lane - and it is the better model

| model | params | what it is | on disk |
|---|---|---|---|
| runs/target/final | 226.5M | M3 base (CSN pretrain, ctx 1024) | yes |
| runs/yarn_4096/final | 226.5M | YaRN ctx-4096 continued-tune of the target | yes |
| runs/yarn_lora_sft_v1/final | 226.5M | merged LoRA-SFT r=32/alpha=64 instruct re-SFT on the yarn base | yes |
| runs/yarn_lora_sft_v1/final_a60 | 226.5M | alpha=0.6 soup, the E-07 promoted deliverable | yes |
| runs/pilot/final | 100.7M | M2 base | yes |
| runs/sft_v2/final (e2), runs/sft_v2_e1/final (e1) | 226.5M | the SFT v2 pair | **no - weights gone** |

Recorded gates: yarn_4096 base-surface @4096 **1.7035** (the Track B win); yarn_lora_sft_v1 guard
@1024-yarn 1.9471 = +3.99% PASS, ast greedy 0.94 / sampled 0.92, base @4096 1.7825 = **+4.64% FAIL**
against the strict <= +2% bar (hence the alpha=0.6 soup).

So the model the user remembers as "more powerful, near instruct mode" is best represented today by
**runs/yarn_lora_sft_v1/final_a60**: 226.5M, instruct-tuned, and **ctx 4096 - four times the target's
1024**. It is the only model in the repo that is both promptable in instruct mode and long-context.

## 4. Why this matters for the plan

1. E-117 is a **base-level** comparison (val loss / bits-per-byte). The parent ladder's apex is
   instruct-level. A base model is not fairly measured against an instruct model, and the user's
   "can our strategy beat the target" question really means the instruct surface.
2. The parent instruct recipe is fully reproducible from disk: corpus, prepared ds, configs
   (sft_v2_e1.yaml, yarn_lora_sft_v1.yaml) and the eval harness (eval.py CSN guard, mini_eval, ast).
   e1 is ~1,172 steps at 226.5M - a few GPU-hours; the LoRA variant trains 4.1% of parameters.
3. Context: the parent line already reached 4096 (Track B + the LoRA-SFT). The glm line has not. Any
   fair long-context comparison must give the glm arm the same treatment.

## 5. Proposed next steps (S2-S4 are user-gated; the GPU is owned by E-117 until it finishes)

- **S0 (now, CPU only)**: run the E-119 bits-per-byte baseline for the four models that exist
  (target, yarn_4096, yarn_lora_sft_v1/final, final_a60) across the HumanEval / MBPP / DS-1000 /
  BigCodeBench text sets. Real cross-model numbers today, no GPU, no waiting.
- **S1 (now, docs)**: record the loss (done), and ask the user whether the 2026-09-09 resume pack
  still exists on the other machine - a copy beats a retrain.
- **S2 (after E-117, GPU)**: restore e1 by retrain with the recorded recipe so the parent instruct
  bar exists again with fresh numbers on the same harness.
- **S3 (after S2, GPU)**: instruct-parity arm - SFT the E-117 winner on the exact same corpus
  (needs a small SFT script in the glm repo; its trainer forward pass and the E-118 evaluator already
  exist) and compare instruct-vs-instruct against e1 on the same battery.
- **S4 (path 2, GPU)**: use `yarn_lora_sft_v1/final_a60` as the local teacher - generate aspect-task
  solutions, validate them with the E-118 sandbox, and feed accepted pairs to the RL/SFT line. This is
  the "learn by try" loop with a real teacher instead of a self-generated one.

## 6. Open question for the user

Does the 2026-09-09 machine-move resume pack (or `checkpoint-sft_v2_e1-final.zip`) still exist on
the other machine or another drive? If yes, e1/e2 return by copy; if no, they return by retrain.
