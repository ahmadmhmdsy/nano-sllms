# C12 Tier 1 launch runbook - execute when M3 completes

Status: LAUNCH-READY (2026-09-06). Everything below is committed; at launch time
nothing needs to be written, only executed. Live status owner: TASKS.md row 4.
Spec: research/c12_distillation_plan.md (Tier 1). Decision log at the bottom.

## 0. What is already prepared (committed)

| Piece | Where | State |
|---|---|---|
| Tokenized SFT data | data/sft/evol/ds | 16,376 train + 400 val prompt-masked pairs (meta.json: 78,264 seen, 21% kept, ast-filter dropped 38%) |
| Held-out instructions | data/sft/evol/val_instructions.jsonl | 400 rows (eval uses the first 50) |
| SFT trainer | scripts/sft.py | auto-resume per PLAN §5.3; --pilot writes to runs/sft_t1_pilot (isolated from the full run) |
| Eval (instructions + forgetting guard) | scripts/eval.py | instruction eval + ast.parse pass-rate; sft_t1.yaml now carries data.tokens_dir so the CSN regression check runs |
| Launch gate | scripts/c12_preflight.py | run first; dry-runnable pre-M3 (see §2) |
| Config | configs/sft_t1.yaml | base runs/target/final; ctx 512; batch 1 x accum 16; lr 3e-5 cosine; warmup 100; eval+save 250; limit 3; fp16; grad-ckpt; seed 42 |
| Weights policy | .gitignore | sft_t1* model.safetensors + checkpoints stay LOCAL (same LFS decision as M3) |

## 1. Confirm M3 is done (the gate)

- runs/target/final/model.safetensors AND runs/target/final/train_summary.json exist.
- No live trainer on the GPU (nvidia-smi). NEVER co-run: SFT needs ~3.5-4.5 GB and
  M3 holds ~5.5 GB of the 8 GB card.
- If M3 is still training or was interrupted: rerun the M3 train command, not this
  runbook. This runbook starts only after M3 completes.

## 2. Preflight gate (blocks launch on any hard FAIL)

    & .\.venv\Scripts\python.exe scripts\c12_preflight.py --pilot   # before the pilot
    & .\.venv\Scripts\python.exe scripts\c12_preflight.py           # before the full run

Dry-run it NOW (while M3 trains) to see every check work:

    & .\.venv\Scripts\python.exe scripts\c12_preflight.py --base-model runs/target/checkpoint-500

Expected while M3 is live: "base model" + "gpu free" FAIL - correct behavior, not a bug.

Disk bars: pilot >= 4.5 GB free; full >= 10.0 GB free (3 rotating ~2.7 GB checkpoints
+ ~0.9 GB final). M3 completion leaves runs/target holding ~9 GB of checkpoints, so
full mode MAY require deleting runs/target/checkpoint-* — the preflight disk check
decides (2026-09-06 probe: 16.3 GB free mid-M3; re-measure at completion). If deletion
is needed: ASK THE USER first (ENVIRONMENT.md: report before deleting anything; M2
precedent, TASKS row 2 / HANDOFF §3). runs/target/final supersedes every rotated checkpoint.

## 3. Baseline forgetting-guard number (HANDOFF §8.2 - do this first)

    & .\.venv\Scripts\python.exe scripts\eval.py --config configs\target.yaml

Writes runs/target/final/eval_report.json: CSN val_loss @ ctx 1024 = the BASELINE
that §7 compares the SFT model against. Commit it (HANDOFF §8.2 pattern).

## 4. Pilot SFT (plumbing proof, ~45-75 min)

    & .\.venv\Scripts\python.exe scripts\sft.py --config configs\sft_t1.yaml --pilot

- 5,000 pairs, 1 epoch = ~312 optimizer steps (eval+save at 250).
- Output is ISOLATED: runs/sft_t1_pilot (checkpoints + final) - the full run
  always starts clean in runs/sft_t1, no manual archiving.
- First 100 steps: measure s/it and re-ETA (M3 lesson: pace swings are thermal -
  never kill a run for pace alone).
- Expect grad_norm to settle <= ~0.6 x clip (the M2/M3 signature); zero NaNs.
- ANY interruption (crash, standby, kill): rerun the SAME command - auto-resume,
  zero flags (PLAN §5.3).

## 5. Pilot gate, then archive the pilot dir

    & .\.venv\Scripts\python.exe scripts\eval.py --config configs\sft_t1.yaml --ckpt runs\sft_t1_pilot\final

Read runs/sft_t1_pilot/final/train_summary.json + eval_report.json:
- eval_loss fell across the run; some ast pass-rate > 0; generations eyeball-sane.

FAIL -> stop, debug, do NOT start the full run.
PASS -> nothing to archive: sft.py --pilot isolates the pilot under
runs/sft_t1_pilot, so the full run always starts CLEAN in runs/sft_t1
(CPU e2e-verified 2026-09-06 on a tiny model).

## 6. Full SFT (~5-6 h)

    & .\.venv\Scripts\python.exe scripts\sft.py --config configs\sft_t1.yaml

- 16,376 pairs x 2 epochs = ~2,047 steps; eval+save every 250.
- First 100 steps: measure s/it, re-ETA. Monitor via TensorBoard (transformers 5.16.1
  prints no console loss lines):

    & .\.venv\Scripts\python.exe -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as A; ea = A('runs/sft_t1/logs'); ea.Reload(); print([(s.step, round(s.value, 4)) for s in ea.Scalars('train/loss')][-10:])"

- Any interruption: rerun the SAME command.

## 7. Validation gates (plan §1 / §3.3)

    & .\.venv\Scripts\python.exe scripts\eval.py --config configs\sft_t1.yaml

- Instruction eval: 50 held-out instructions, greedy + temp 0.8 -> ast.parse
  pass-rate in runs/sft_t1/final/eval_report.json (Tier-1 success criterion).
- Forgetting guard: eval_report.json val_loss vs the §3 baseline; regression
  <= ~10-15% = PASS (catastrophic-forgetting check).
- Qualitative side-by-side vs base T (same prompts, both checkpoints):

    & .\.venv\Scripts\python.exe scripts\infer.py --config configs\sft_t1.yaml --ckpt runs/target/final
    & .\.venv\Scripts\python.exe scripts\infer.py --config configs\sft_t1.yaml --ckpt runs/sft_t1/final

## 8. Commit + close out

- Commit small final-dir files only: train_summary.json, eval_report.json, config.json,
  generation_config.json, tokenizer.json, tokenizer_config.json (+ tfevents). Model
  safetensors stay LOCAL (LFS decision).
- Update TASKS.md row 4 (status truth) and HANDOFF §8.4 (narrative).
- Pre-push: git lfs status; no >100 MB file outside LFS (HANDOFF §7).

## Decision log

- 2026-09-06 - user asked "apply C12 with M3 checkpoint-1000?": mechanically possible
  (a Trainer checkpoint dir loads via from_pretrained), but declined per the plan gate:
  the base is 20%-trained (eval 2.265 @1000 vs the final T target ~1.2-1.3 by ladder
  trend), SFT cannot co-run with the live M3 (~2.6 GB free of 8), and the full SFT
  would be redone on final T anyway (double GPU cost). User chose: WAIT for
  runs/target/final; prepare everything meanwhile (this runbook + preflight + the
  configs/sft_t1.yaml fix).
- 2026-09-06 - user approved prep-during-M3 (CPU-only work): configs/sft_t1.yaml
  data.tokens_dir addition, .gitignore sft_t1 weight rules, docs, preflight script.
  The live M3 process reads only configs/target.yaml at startup - untouched.
