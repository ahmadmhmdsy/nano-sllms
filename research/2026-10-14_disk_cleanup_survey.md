# Disk-cleanup survey - model artifacts (READ-ONLY; no deletions)

- **Date:** 2026-10-14 (Asia/Baghdad). Survey ran ~11:31-11:35; one nested run directory is flagged LIVE below.
> **EXECUTED 2026-10-14 11:5x (user approved "clean them").** Lead deleted the CLOSED + SUPERSEDED set:
> **65 paths deleted, 0 failures, 7.58 GiB freed; E: free 7.10 -> 14.69 GiB.** All 13 referenced keepers
> re-verified present after deletion (e117_byte/checkpoint-5000, e121_ref/checkpoint-0100, e121b_rl_all/checkpoint-0096,
> e122_ref600/checkpoint-0330 + 0600, e122d_ref_pg/checkpoint-0510 + 0600, pretrain/checkpoint-0100,
> pretrain-dense/checkpoint-0400, rl/checkpoint-0096, rl-regime-a1/a4/checkpoint-0096, rl-regime-a6/checkpoint-0192).
> **Deliberately EXCLUDED** (still present): runs/trocr_finetune_v2 (2.00 GiB - its own row says verify the file set
> against the surviving trainer_state copy first), runs/ocr_bakeoff_smoke, runs/stage05_pilot, runs/trocr_finetune,
> and rl-regime-a6/checkpoint-0096 (98 MB; the handoff names 0192 as that run's keeper, so 0096 is superseded, but
> 98 MB is not worth any risk). The ~8 GiB UNKNOWN set was NOT touched - it still needs an owner's word.
> CONTRADICTION FOUND IN THIS SURVEY: e121b_rl_all is listed as CLOSED (whole run deletable) AND its
> checkpoint-0096 is named a keeper. The Lead kept the run and deleted only its superseded 0030/0095.
> The 1363 pre-existing unstaged tracked deletions were NOT touched.

- **Scope:** `runs\` and `checkpoint_backup\` in the parent repo and in `glm-5.3-flash-from-scratch`, plus dangling `BUNDLE_MANIFEST_*.json`.
- **Rule honoured:** nothing was deleted or moved. This is a report; the user approves every deletion.
- **Measurement command (exact, per directory):** `$m = (Get-ChildItem -LiteralPath <dir> -Recurse -File | Measure-Object -Sum Length).Sum`
  Newest file date = max `LastWriteTime` over the recursive file set. Step count for a checkpoint = the integer in `checkpoint-NNNN`; for a run = number of `checkpoint-*` children. Contents detected from the recursive file inventory (`weights.pt`->`model.pt`, `state.pt`, `progress.jsonl`, `training-receipt.json`, `events.out.tfevents*`). This fork names weights `model.pt` (not `weights.pt`) and optimizer state `state.pt` (there is no separate `optimizer.pt`).

## Totals (measured)

| Scope | Files | Size |
|---|---|---|
| Parent `runs\` | 19,713 | **23.40 GiB** (25,126,634,318 B) |
| Parent `checkpoint_backup\` | 4 | **1.32 GiB** (1,416,790,287 B) |
| Nested `runs\` | 380 | **9.07 GiB** (9,742,224,665 B) |
| Nested `checkpoint_backup\` | 0 | does not exist |

**Reclaimable if CLOSED + SUPERSEDED are approved (measured): 10,446,175,835 B = 9.73 GiB** - CLOSED = 6,839,540,770 B (6.37 GiB), SUPERSEDED = 3,606,635,065 B (3.36 GiB). All other paths are REFERENCED or UNKNOWN and are **excluded**.

## Headline answer - does the pruning protect the BEST checkpoint's state.pt? **No.**

`glm-5.3-flash-from-scratch/glm53_flash/runtime.py`, `prune_states` (L167-177):
```python
def prune_states(output: Path, keep: int = 1) -> list[str]:
    """Keep only the newest `keep` optimizer-state files (disk is tight)."""
    steps = list_checkpoints(output)
    stale = steps if keep <= 0 else steps[:-keep]
    removed = []
    for step in stale:
        state = checkpoint_dir(output, step) / STATE_FILENAME
        if state.exists():
            state.unlink()
            removed.append(str(state))
    return removed
```

`prune_checkpoints` (L189-232) protects the **directory** of the best step:
```python
    protected = set(keep_steps) | set(steps[-keep_last:]) | {steps[-1]}
    if best_step is not None:
        protected.add(best_step)
```
but it never touches `state.pt`, and the trainers call `prune_states` **before** `prune_checkpoints` - e.g. `scripts/train_pretrain.py` L507-508:
```python
            prune_states(args.output, args.keep_state)
            retained = prune_checkpoints(args.output, keep_last=args.keep_checkpoints, metric_key="val_loss")
```

**Consequence (the +196 MiB issue):** with the default `--keep-state 1` only the **newest** checkpoint keeps its `state.pt` (206,107,851 B = 196.6 MiB); the best checkpoint survives only as weights (`model.pt`), so its optimizer/RNG/scaler state is gone and it cannot be resumed. MEMORY.md L1563-1565 states this explicitly. Measured confirmation: `runs/e122_ref600/checkpoint-0330` (the current baseline init) has **no `state.pt`**, while only `checkpoint-0600` (newest) carries the 196.6 MiB `state.pt`. So pruning protects the best **directory**, not the best **state**.

## Table 1 - nested repo `glm-5.3-flash-from-scratch\runs\` (primary scope)

| Path | Bytes | Size | Newest | Contents | Class | What frees if deleted | Risk / why |
|---|---|---|---|---|---|---|---|
| `glm-5.3-flash-from-scratch/runs/_pytest_tmp/` | 43,873,744 | 41.8 MiB | 2026-10-10 | model.pt/logs only | CLOSED | whole run directory and its files (41.8 MiB) | pytest temp dirs; disposable test scaffolding |
| `glm-5.3-flash-from-scratch/runs/_smoke/` | 35,099,733 | 33.5 MiB | 2026-10-08 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (33.5 MiB) | training smoke scratch |
| `glm-5.3-flash-from-scratch/runs/_smoke2/` | 35,099,734 | 33.5 MiB | 2026-10-08 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (33.5 MiB) | training smoke scratch |
| `glm-5.3-flash-from-scratch/runs/e117_byte/` | 622,458,316 | 593.6 MiB | 2026-10-11 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | E-117 closed (FAIL) but checkpoint-5000 is the arm-A donor/init cited in HANDOFF L4/L328-329 |
| `glm-5.3-flash-from-scratch/runs/e117_byte_batch2_probe/` | 103,058,457 | 98.3 MiB | 2026-10-10 | progress.jsonl | CLOSED | whole run directory and its files (98.3 MiB) | one-off batch-2 probe |
| `glm-5.3-flash-from-scratch/runs/e117_byte_failed_attempts/` | 83,719 | 0.1 MiB | 2026-10-10 | progress.jsonl | REFERENCED | n/a - retain | TASKS L580: the log is kept at runs/e117_byte_failed_attempts |
| `glm-5.3-flash-from-scratch/runs/e117_byte_failed_oom/` | 412,283,479 | 393.2 MiB | 2026-10-10 | progress.jsonl | CLOSED | whole run directory and its files (393.2 MiB) | dead OOM attempt (TASKS L620 cites it historically) |
| `glm-5.3-flash-from-scratch/runs/e121_ref/` | 412,260,714 | 393.2 MiB | 2026-10-11 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | arm A reference-SFT base; HANDOFF L180-181 the only viable base |
| `glm-5.3-flash-from-scratch/runs/e121_whole/` | 415,311,505 | 396.1 MiB | 2026-10-13 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (396.1 MiB) | E-121b FAIL, verdict recorded (EXPERIMENTS L1218) |
| `glm-5.3-flash-from-scratch/runs/e121b_ref/` | 412,260,778 | 393.2 MiB | 2026-10-13 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (393.2 MiB) | E-121c FAIL arm B (EXPERIMENTS L1217) |
| `glm-5.3-flash-from-scratch/runs/e121b_rft/` | 223,304,183 | 213.0 MiB | 2026-10-13 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (213.0 MiB) | E-121d FAIL (EXPERIMENTS L1210) |
| `glm-5.3-flash-from-scratch/runs/e121b_rft_all/` | 223,338,802 | 213.0 MiB | 2026-10-13 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (213.0 MiB) | E-121e PASS (EXPERIMENTS L1211); ladder provenance |
| `glm-5.3-flash-from-scratch/runs/e121b_rl_all/` | 326,930,459 | 311.8 MiB | 2026-10-13 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (311.8 MiB) | E-121f PASS (EXPERIMENTS L1216) |
| `glm-5.3-flash-from-scratch/runs/e122_ref600/` | 515,603,240 | 491.7 MiB | 2026-10-13 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | checkpoint-0330 = CURRENT baseline init (task note); E-122 PASS |
| `glm-5.3-flash-from-scratch/runs/e122d_ref_pg/` | 515,603,316 | 491.7 MiB | 2026-10-14 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | checkpoint-0510 = E-122d best (task note); verdict recorded |
| `glm-5.3-flash-from-scratch/runs/e122e_replay_lr1e4/` | 412,223,685 | 393.1 MiB | 2026-10-14 | progress.jsonl | UNKNOWN | whole run directory and its files (393.1 MiB) | NO ledger row; newest file 2026-10-14T11:32:50 (during survey) => likely LIVE mixed-replay run. DO NOT TOUCH. |
| `glm-5.3-flash-from-scratch/runs/pretrain/` | 618,482,994 | 589.8 MiB | 2026-10-08 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | checkpoint-0100 is the A2/A4/A6 RL init (HANDOFF L643, TASKS L384) |
| `glm-5.3-flash-from-scratch/runs/pretrain-dense/` | 1,442,855,041 | 1.34 GiB | 2026-10-08 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | checkpoint-0400 is the determinism pair / E-113 dense-repeat evidence |
| `glm-5.3-flash-from-scratch/runs/rl/` | 534,745,498 | 510.0 MiB | 2026-10-08 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | checkpoint-0096 kept in the E-116 resume pack (HANDOFF L645) |
| `glm-5.3-flash-from-scratch/runs/rl-regime-a1/` | 534,719,408 | 509.9 MiB | 2026-10-08 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | checkpoint-0096 kept in the E-116 pack (HANDOFF L645) |
| `glm-5.3-flash-from-scratch/runs/rl-regime-a3/` | 517,045,729 | 493.1 MiB | 2026-10-08 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (493.1 MiB) | E-113; proven bit-identical to pretrain/checkpoint-0200, dropped from pack |
| `glm-5.3-flash-from-scratch/runs/rl-regime-a4/` | 534,745,599 | 510.0 MiB | 2026-10-10 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | checkpoint-0096 is the E-116 G1 equivalence proof |
| `glm-5.3-flash-from-scratch/runs/rl-regime-a6/` | 433,672,238 | 413.6 MiB | 2026-10-10 | progress.jsonl,training-receipt.json | REFERENCED | n/a - retain | checkpoint-0192 = E-116 192-group endpoint, in pack (HANDOFF L646) |
| `glm-5.3-flash-from-scratch/runs/smoke_rl_init/` | 412,185,447 | 393.1 MiB | 2026-10-10 | progress.jsonl,training-receipt.json | CLOSED | whole run directory and its files (393.1 MiB) | RL smoke scratch |
| `glm-5.3-flash-from-scratch/runs/eval/` | 4,782,979 | 4.6 MiB | 2026-10-10 | model.pt/logs only | REFERENCED | n/a - retain | eval JSON artifacts |
| `glm-5.3-flash-from-scratch/runs/regime/` | 192,491 | 0.2 MiB | 2026-10-10 | model.pt/logs only | UNKNOWN | whole run directory and its files (0.2 MiB) | 192 KB of regime JSON; no doc reference found |

### SUPERSEDED checkpoints (earlier step; later checkpoint kept) - safe but name the keeper

| Checkpoint | Bytes | Newest | Contents |
|---|---|---|---|
| `runs/e117_byte\checkpoint-4750/` | 309,154,268 | 2026-10-11 | state.pt |
| `runs/e121_ref\checkpoint-0075/` | 103,046,378 | 2026-10-11 | model.pt only |
| `runs/e121b_rl_all\checkpoint-0030/` | 103,046,250 | 2026-10-13 | model.pt only |
| `runs/e121b_rl_all\checkpoint-0095/` | 103,046,254 | 2026-10-13 | model.pt only |
| `runs/e122_ref600\checkpoint-0570/` | 103,046,379 | 2026-10-13 | model.pt only |
| `runs/e122d_ref_pg\checkpoint-0570/` | 103,046,379 | 2026-10-14 | model.pt only |
| `runs/pretrain\checkpoint-0000/` | 103,046,280 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0000/` | 103,046,280 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0010/` | 103,046,317 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0020/` | 103,046,317 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0030/` | 103,046,318 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0040/` | 103,046,318 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0050/` | 103,046,318 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0060/` | 103,046,318 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0070/` | 103,046,318 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0080/` | 103,046,318 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0090/` | 103,046,318 | 2026-10-08 | model.pt only |
| `runs/pretrain-dense\checkpoint-0100/` | 103,046,319 | 2026-10-08 | model.pt only |
| `runs/rl\checkpoint-0000/` | 103,046,300 | 2026-10-08 | model.pt only |
| `runs/rl\checkpoint-0024/` | 103,046,201 | 2026-10-08 | model.pt only |
| `runs/rl\checkpoint-0048/` | 103,046,201 | 2026-10-08 | model.pt only |
| `runs/rl\checkpoint-0072/` | 103,046,201 | 2026-10-08 | model.pt only |
| `runs/rl-regime-a1\checkpoint-0000/` | 103,046,306 | 2026-10-08 | model.pt only |
| `runs/rl-regime-a1\checkpoint-0024/` | 103,046,201 | 2026-10-08 | model.pt only |
| `runs/rl-regime-a1\checkpoint-0048/` | 103,046,201 | 2026-10-08 | model.pt only |
| `runs/rl-regime-a1\checkpoint-0072/` | 103,046,201 | 2026-10-08 | model.pt only |
| `runs/rl-regime-a4\checkpoint-0000/` | 103,046,300 | 2026-10-10 | model.pt only |
| `runs/rl-regime-a4\checkpoint-0024/` | 103,046,201 | 2026-10-10 | model.pt only |
| `runs/rl-regime-a4\checkpoint-0048/` | 103,046,201 | 2026-10-10 | model.pt only |
| `runs/rl-regime-a4\checkpoint-0072/` | 103,046,201 | 2026-10-10 | model.pt only |
| `runs/rl-regime-a6\checkpoint-0000/` | 103,046,300 | 2026-10-10 | model.pt only |
| `runs/rl-regime-a6\checkpoint-0096/` | 103,046,201 | 2026-10-10 | model.pt only |
| `runs/rl-regime-a6\checkpoint-0144/` | 103,046,202 | 2026-10-10 | model.pt only |

Keepers: `e117_byte/checkpoint-5000` (REFERENCED), `e121_ref/checkpoint-0100` (REFERENCED), `e121b_rl_all/checkpoint-0096`, `e122_ref600/checkpoint-0600` (0330 stays as baseline), `e122d_ref_pg/checkpoint-0600` (0510 stays as best), `pretrain/checkpoint-0100` and `0400`, `pretrain-dense/checkpoint-0400`, `rl|a1|a4 checkpoint-0096`, `rl-regime-a6/checkpoint-0192`.

## Table 2 - parent `runs\` (summary)

| Run | Size | Newest | Class | Note |
|---|---|---|---|---|
| `laya/` | 6.23 GiB | 2026-09-24 | REFERENCED | TASKS 224 / HANDOFF cite `runs/laya/l3_e74*` |
| `s10_adapters/` | 2.33 GiB | 2026-10-13 | UNKNOWN | per-aspect adapters; no doc row found |
| `trocr_finetune_v2/` | 2.00 GiB | 2026-10-03 | **CLOSED** | TrOCR line abandoned (MEMORY L140/L158); verify its file set against the surviving trainer_state copy first |
| `yarn_lora_sft_v1/` | 1.69 GiB | 2026-09-10 | REFERENCED | `final_a60` promoted deliverable (HANDOFF L545-547) |
| `stage05_pilot_gpu/` | 1.36 GiB | 2026-10-04 | UNKNOWN | OCR Stage 0.5 PaddleOCR arms |
| `s10_adapters_big/` | 1.07 GiB | 2026-10-13 | UNKNOWN | as s10_adapters |
| `hybrid_c2_smoke/`, `hybrid_c2_tiny/` | 1.01 GiB each | 2026-10-01 | UNKNOWN | byte-duplicate pair; no ledger row |
| `target/` | 868 MiB | 2026-09-09 | REFERENCED | SFT-v2 source; restored bit-exact |
| `yarn_4096/` | 864 MiB | 2026-09-10 | REFERENCED | YaRN base for the 4096 lineage |
| `stage05b/` | 531 MiB | 2026-10-06 | UNKNOWN | OCR Stage 0.5b (E-88 closed) |
| `e122c_rft_cap/` | 386 MiB | 2026-10-14 | **CLOSED** | E-122c (perfect NULL, EXPERIMENTS L1213); has state.pt + receipt |
| `e122c_rft_all` / `_sb` / `_tierall` | 98 MiB each | 2026-10-14 | **CLOSED** | E-122c variants, identical scores |
| `kd-t2p-baseline` / `kd-t2p-kd-skew` | 388 MiB each | 2026-09-09 | **CLOSED** | Track C done (TASKS 31); HANDOFF keeps only `kd-t2p-kd/final` |
| `kd-t2p-kd/` | 388 MiB | 2026-09-09 | REFERENCED | distilled-P rung (HANDOFF L3318) |
| `pilot/` | 388 MiB | 2026-09-09 | REFERENCED | historical pilot evidence |
| `u1_full/`, `u1_fused/` | 119 MiB each | 2026-09-26 | REFERENCED | U-line deliverables |
| `runs/evidence/` | ~191 MiB | 2026-09-22 | REFERENCED | cleanup provenance snapshots |
| `s10_merged_average/`, `s10_merged_ties/` | 98 MiB each | 2026-10-13 | UNKNOWN | merges of the s10 adapters |
| `hflines/`, `hflines_v3/`, `vit_real/`, `stage2_*`, `stage3_*`, `crnn_*`, `detector_viz/` | varies (largest 274 MiB) | 2026-10-05 | UNKNOWN/CLOSED | OCR detector stages; `ocr_bakeoff_smoke` and `stage05_pilot` are CLOSED |

## Table 3 - `checkpoint_backup\` and dangling manifests

| Path | Bytes | Class | Note |
|---|---|---|---|
| `resume_pack_2026-10-08_glm53flash_E116.zip` | 1,416,597,303 | REFERENCED | the live machine-move resume pack (only zip on disk) |
| `BUNDLE_MANIFEST_2026-10-08_glm53flash_E116.json` | 70,165 | REFERENCED | manifest of that zip |
| `BUNDLE_MANIFEST_2026-10-08_glm53flash_E113.json` | 59,259 | **CLOSED / DANGLING** | declares `resume_pack_..._E113.zip` (zip_bytes 1,072,078,195) - **zip absent** |
| `BUNDLE_MANIFEST_2026-10-08_glm53flash_E114.json` | 63,560 | **CLOSED / DANGLING** | declares `resume_pack_..._E114.zip` (zip_bytes 1,295,814,897) - **zip absent** |

A recursive `Get-ChildItem -Recurse -Filter resume_pack*.zip` over the whole parent tree returns **only** the E116 zip, so the E113/E114 manifests are dangling. Deleting the two manifests frees only ~120 KB - it does not recover the 1.0/1.3 GiB they record. Nested `checkpoint_backup\` does not exist.

## UNKNOWN items and what would classify them

- `glm-5.3-flash-from-scratch/runs/e122e_replay_lr1e4` - **newest file 2026-10-14T11:32:50, during the survey; has progress.jsonl but no training-receipt.json; almost certainly the LIVE E-122d mixed-replay run.** No EXPERIMENTS row yet. Do-not-touch.
- `runs/regime/` (192 KB) - no document reference; need the run's own report.
- Parent `s10_adapters/`, `s10_adapters_big/`, `hybrid_c2_smoke|tiny/`, `s10_merged_*`, `stage05_pilot_gpu/`, `stage05b/` - large and unmatched by any TASKS/HANDOFF/MEMORY row I found. Need the owning task row, the trainer receipt/verdict, or the user's word that the line is finished.
- `e117_byte_failed_oom` and `e117_byte_batch2_probe` - classified CLOSED as dead attempts; if forensic logs are wanted they become REFERENCED.

## Risks / cautions before any deletion

1. `runs/e122e_replay_lr1e4` appears live - do not delete or prune under it.
2. There is no `optimizer.pt` in this fork; optimizer state is `state.pt` (196.6 MiB) and only the newest checkpoint has one. Deleting a checkpoint directory also deletes its receipt/progress evidence.
3. The destructive-op rule from the Tier-A incident (MEMORY L627): cleanup scripts must normalise KEEP paths to the same absolute form as `Get-ChildItem` and must verify every KEEP path after the delete.
4. `runs/evidence/` snapshots are the only remaining record of several already-deleted runs - keep them.
5. `laya/`, `yarn_lora_sft_v1/`, `yarn_4096/`, `target/`, `u1_full|fused/`, `kd-t2p-kd/` are referenced deliverables and are excluded from the reclaimable figure.
