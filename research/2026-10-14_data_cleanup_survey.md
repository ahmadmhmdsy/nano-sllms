# Data and scratch cleanup survey - 2026-10-14

**READ-ONLY survey. Nothing was deleted, moved or modified.** Every deletion below needs explicit user approval.

> **EXECUTED 2026-10-14 12:0x.** Lead deleted the 9 UNKNOWN runs/ artifacts (user approved "clean them"):
> runs/s10_adapters, runs/s10_adapters_big, runs/hybrid_c2_smoke, runs/hybrid_c2_tiny, runs/stage05_pilot_gpu,
> runs/stage05b, runs/s10_merged_average, runs/s10_merged_ties, glm-5.3-flash-from-scratch/runs/regime.
> **9 deleted, 0 failures, 7.50 GiB freed; E: free 13.48 -> 20.98 GiB.** Combined with the earlier CLOSED+SUPERSEDED
> pass (7.58 GiB), the session freed ~15 GiB: E: went 7.10 -> 20.98 GiB.
> **HELD BACK deliberately:** data/distillation_data (51.5 MiB) - this survey called it "LLM-teacher output, likely
> IRREPLACEABLE", and the dossier that would confirm or refute that (research/2026-10-14_unknown_set_dossier.md)
> had not landed when the deletion ran. 51.5 MiB is 0.7% of the set; it stays until that evidence arrives.

- Scope: data\ (all subdirs), scratch\, and every .pytest_tmp* / .pytest_cache in the repo.
- Sizes measured with Get-ChildItem -Recurse -File | Measure-Object -Sum Length (real on-disk bytes); date = newest file mtime in the subtree.
- Host disk E: **7.63 GB free** of 256 GB - headroom is tight, so the reclaimable figure matters.

## 0. Headline

| Metric | Measured |
|---|---|
| data\ total | **15,757,236,921 B = 14.68 GiB** (220,076 files) |
| scratch\ total | **336,813,341 B = 321.2 MiB** (487 files) |
| repo pytest debris (scratch/pytest_tmp*, .pytest_cache) | 129,742 B = 0.12 MiB |
| glm-5.3-flash-from-scratch/.pytest_tmp* + .pytest_cache | 105,311,464 B = 100.4 MiB |
| **Reclaimable, CLOSED only (zero rebuild cost, on disk now)** | **approx 265.6 MiB (278.5 MB)** |
| **Reclaimable, CLOSED + REGENERABLE** | **approx 15.0 GiB** (dominated by re-downloadable corpora) |

> WARNING: the working tree ALREADY has **1,363 tracked files deleted** (unstaged): 1,116 under runs/mex, 53 runs/ctx_probes, 25 data/legacy_check, 15 data/snapshot_row13_row16_wip, 9 data/catalog, plus 195 untracked (??) and 20 modified. This is a pre-existing state, not caused by this survey. Do not add to it until the user says whether it is an intended in-progress purge or an interrupted checkout.

---

## A. data\ - significant items

| Path | Size (MiB) | Newest | Class | How to regenerate | Risk |
|---|---|---|---|---|---|
| data\teacher\ | 1687.99 | 2026-09-06 | REGENERABLE | venv python scripts\download_teacher.py (network, HF) | low; gitignored, needs network |
| data\hf\ | 7196.51 | 2026-10-08 | REGENERABLE | arabic_ocr dataset downloads (arabic_ocr/scripts/fetch_hf_lines.py, build_hf_lines.py) - network | med; 61,590 files, long re-download |
| data\hflines\ | 1270.55 | 2026-10-01 | REGENERABLE | arabic_ocr\scripts\build_hf_lines.py (from data\hf) | med; big image corpus |
| data\linecorpus\ | 991.30 | 2026-10-01 | REGENERABLE | arabic_ocr\scripts\build_line_corpus.py | med |
| data\qari_10k\ | 971.72 | 2026-09-30 | REGENERABLE | arabic_ocr\scripts\build_line_dataset.py | med |
| data\laya\ | 724.43 | 2026-09-24 | REGENERABLE | laya\scripts\build_data_l2.py (+ pretrain_encoder.py) | low; gitignored, regenerable |
| data\u1\ | 642.69 | 2026-09-26 | REGENERABLE | scripts\prepare_data.py + tokenize_data.py --config configs/u1_full.yaml / u1_fused.yaml / u1_smoke.yaml | low; tokens tracked in git |
| data\target\ | 504.66 | 2026-10-10 | REGENERABLE | bytes: scripts\target_bytes.py (needs runs\target\final tokenizer); tokens: prepare_data.py + tokenize_data.py --config configs/target.yaml | med; bytes dir gitignored |
| data\wholefile\ | 318.06 | 2026-10-11 | REGENERABLE | scripts\wholefile_bytes.py (network: codeparrot/codeparrot-clean) | med; network |
| data\pilot\ | 235.79 | 2026-09-22 | REGENERABLE | prepare_data.py + tokenize_data.py --config configs/pilot.yaml | low |
| data\sft\ | 96.41 (101,093,011 B) | 2026-09-22 | REGENERABLE | scripts\sft_data.py --config configs/sft_t1.yaml; sft_v2; scripts\h2p2_build_corpus.py; scripts\kt2_judge.py | low; */raw/ gitignored, pairs.jsonl tracked |
| data\custom_example\ | 96.08 | 2026-09-06 | REGENERABLE | prepare_data.py + tokenize_data.py --config configs/custom_example.yaml | low |
| data\ocr_textpool\ | 76.81 (raw\ = 47.14 / 49,431,026 B) | 2026-10-04 | **raw\ CLOSED**, rest tracked | raw\ is an explicit duplicate of the committed texts.jsonl per .gitignore; arabic_ocr\scripts\harvest_text_pool.py can rebuild | **low - clearest single reclaimable data item** |
| data\distillation_data\ | 51.50 | 2026-09-09 | **UNKNOWN / likely IRREPLACEABLE** | data\distillation_data\*\meta\gen_all.py exists, but re-generation needs the same LLM/teacher outputs (not deterministic) | high - do not delete without user |
| data\md_measure\ | 44.33 | 2026-09-22 | REGENERABLE (network) | data\md_measure\*.py + upstream dataset | med; *\raw/ gitignored |
| data\reallines\ | 33.57 | 2026-10-01 | REGENERABLE (source fetch) | arabic_ocr\scripts\build_hf_lines.py (images gitignored; label manifest tracked) | med; needs source images |
| data\diac\ | 22.73 | 2026-09-22 | REGENERABLE | diacritizer download_data.py + prepare_data.py (seed 20260911) | low; gitignored/regenerable |
| data\benchmark\ | 21.02 | 2026-09-30 | REGENERABLE (source) | source images + tracked manifests | med |
| data\bench\ | 18.36 | 2026-10-10 | REGENERABLE | scripts\bench_inventory.py, scripts\eval_bench_bpb.py | low |
| data\reference\ | 12.34 | 2026-10-14 11:23 | see section B | see section B | see section B |
| data\smoke\ | 6.36 | 2026-09-28 | REGENERABLE | prepare_data.py + tokenize_data.py --config configs/smoke.yaml | low |
| data\linecorpus_smoke\ | 3.50 | 2026-10-01 | REGENERABLE | build_line_corpus.py smoke path | low |
| data\mex\ | 0.52 | 2026-09-22 | REGENERABLE | mex\scripts\build_mu2_g1.py, build_mu2_v3q.py | low; gitignored |
| data\linecorpus_tiny\ | 0.02 | 2026-10-01 | REGENERABLE | build_line_corpus.py tiny path | low |
| data\catalog\ | **0 (absent from disk)** | - | CLOSED (already deleted; 9 files tracked at HEAD) | git checkout HEAD -- data/catalog | none; 0 bytes on disk |
| data\legacy_check\ | **0 (absent from disk)** | - | CLOSED (already deleted; 25 files tracked at HEAD) | git checkout HEAD -- data/legacy_check | none; 0 bytes on disk |
| data\snapshot_row13_row16_wip\ | **0 (absent from disk)** | - | CLOSED (15 files tracked at HEAD) | git checkout HEAD -- data/snapshot_row13_row16_wip | none; 0 bytes on disk |

---

## B. data\reference\ - the three E-122 corpora (hash-verified, measured just now)

| File | Bytes | Newest | Class | SHA256 |
|---|---|---|---|---|
| bytes\train.bin (E-122) | 2,173,454 | 2026-10-11 10:32 | **REGENERABLE** | 572EED7F...E41840 |
| bytes\val.bin | 108,798 | 2026-10-11 10:32 | REGENERABLE | 9BDF6424...E756B9 |
| bytes\meta.json | 744 | 2026-10-11 10:32 | REGENERABLE | FDA104D3...3E746D |
| bytes_playground\train.bin (E-122d) | 6,812,468 | 2026-10-14 08:55 | **REGENERABLE** | BB48C570...6BFC227 |
| bytes_playground\val.bin | 340,872 | 2026-10-14 08:55 | REGENERABLE | 9EE952D8...4FDB6D8 |
| bytes_playground\meta.json | 1632 | 2026-10-14 08:55 | REGENERABLE | F13749EA...21DCC87 |
| bytes_pg_replay\train.bin (Phase 1.2b) | 3,333,114 | 2026-10-14 11:23 | **REFERENCED (LIVE)** | 24C2CC94...F65151E9 |
| bytes_pg_replay\val.bin | 166,612 | 2026-10-14 11:23 | **REFERENCED (LIVE)** | E92D2EC9...8CBF2511CE |
| bytes_pg_replay\meta.json | 1629 | 2026-10-14 11:23 | REFERENCED (LIVE) | 1AD0022F...4D1001D15 |

**Exact regenerate commands (venv only):**

- E-122 bytes:  & .\.venv\Scripts\python.exe scripts\reference_sft_bytes.py   (defaults: --out-dir data/reference/bytes, seed 9319, 400 instances/source)
- E-122d bytes_playground:  & .\.venv\Scripts\python.exe scripts\reference_sft_bytes.py --include-playground --playground-tier all --out-dir data\reference\bytes_playground
- bytes_pg_replay: rebuilt by the same builder with a replay-dominant mixture, but the **exact flags are UNKNOWN** (see section G, item 2).

**LIVE READER - do not touch:** PID **37804** (venv shim) / **33232** (real interpreter, 1.75 GB WS) is running scripts\train_pretrain.py with --corpus ..\data\reference\bytes_pg_replay\train.bin --val-corpus ..\data\reference\bytes_pg_replay\val.bin --init-from runs\e122_ref600\checkpoint-0330 ... --output runs\e122e_replay_lr1e4 . The parent background job pwsh-107 is this process. The file is memmapped; deleting/moving it will corrupt or kill the run.

Provenance from the meta files:
- bytes: 20 canonical sources x 400 = 8,000 train / 400 val examples; split pretrain, seed 9319; ~150.8 epochs at 5000 steps.
- bytes_playground: 20 canonical + 27 playground rungs x 400 = 18,800 train / 940 val; playground tier all; ~48.1 epochs at 5000 steps.
- bytes_pg_replay: 20 canonical (8,000) + **2,700 playground** = 10,700 train / 535 val (replay-dominant, about 100 per rung, NOT the 400/rung default); ~98.3 epochs at 5000 steps. This is the rebalanced mixture the E-122d remedy called for.

E-122d verdict is recorded (research/EXPERIMENTS.md E-122d row; TASKS.md row 171): rung battery 31/108 -> 4/108, grade 0/28 -> 1/28, aspect pool 32/48 -> 23/48 = CATASTROPHIC FORGETTING from imitation, not from RL. bytes_playground is a closed-result input (verdict on record) but is only 6.8 MiB and regenerable - keep unless the user wants the space.

---

## C. scratch\ - significant items

| Path | Size (MiB) | Newest | Class | Disposition | Risk |
|---|---|---|---|---|---|
| scratch\s10_real_smoke\ | 198.73 (208,378,990 B) | 2026-10-13 | REGENERABLE | re-run the S10 real-training smoke (glm-5.3-flash-from-scratch trainer); holds two ~103 MB model.pt | low-med; GPU/time to rebuild |
| scratch\postproc_ab\ | 53.40 | 2026-09-22 | **CLOSED** | old diac/OCR post-proc prediction dumps; no script references it; re-derivable by re-running eval | low |
| scratch\stage2b2500\ | 22.46 | 2026-09-22 | **CLOSED** | diac stage-2 predictions | low |
| scratch\stage2final\ | 22.33 | 2026-09-22 | **CLOSED** | diac stage-2 predictions | low |
| scratch\asas_summ_train.parquet | 8.64 (9,057,814 B) | 2026-09-22 | **CLOSED** | data blob, no producer script in repo | low |
| scratch\stage2a\ | 6.02 | 2026-09-22 | **CLOSED** | diac stage-2 predictions | low |
| scratch\gate_overlap\ | 4.90 | 2026-09-22 | **CLOSED** | produced by scratch\g_overlap_sample.py (itself an old probe) | low |
| scratch\s10_smoke\ | 3.62 | 2026-10-13 | REGENERABLE | re-run S10 smoke | low |
| scratch\render_demo\ + scratch\ui_demo\ | 0.16 | 2026-09-26 | **CLOSED** | static HTML demos | none |
| scratch\pytest_tmp\ + scratch\pytest_tmp_v\ | 0.12 | 2026-10-13 | **CLOSED** | pytest debris (--basetemp) | none |
| scratch\corpus_pipeline_fixture\ | 0.05 | 2026-10-14 11:32 | **REFERENCED / UNKNOWN** | created minutes before this survey (11:28-11:32); looks like an active test fixture | do not touch |
| loose fix_*.py, expdbg*.py, probe_*.py, hf_*_search.py, tead_*.py, dl_*.py, close_e4x/e5x.py | small | 2026-09-22 | CLOSED (true throwaways) | one-off historical debug/probe scripts | low |

---

## D. Stray pytest temp dirs

| Path | Bytes/MiB | Newest | Class | Note |
|---|---|---|---|---|
| glm-5.3-flash-from-scratch/.pytest_tmp\ | 33.48 MiB | 2026-10-12 12:35 | **CLOSED** | test debris (adapter .pt fixtures) |
| glm-5.3-flash-from-scratch/.pytest_tmp2\ | ~0 | 2026-10-12 12:35 | **CLOSED** | test debris |
| glm-5.3-flash-from-scratch/.pytest_tmp3\ | 33.48 MiB | 2026-10-12 12:36 | **CLOSED** | test debris |
| glm-5.3-flash-from-scratch/.pytest_tmp4\ | 33.48 MiB | 2026-10-12 12:37 | **CLOSED** | test debris |
| glm-5.3-flash-from-scratch/.pytest_cache\ | 0.01 MiB | 2026-10-08 | **CLOSED** | cache |
| nano_SLMs/.pytest_cache\ | <0.01 MiB | 2026-10-13 | **CLOSED** | cache |

Subtotal (glm tmp + caches) = **100.44 MiB**. (.pytest_tmp_corpus was listed earlier but is no longer on disk.)

**A pytest is running RIGHT NOW**: PID 10668/36756 = python -m pytest tests/test_aspects_evaluator.py -v --basetemp=.pytest_tmp_aspects, writing its own basetemp under the glm project. Only the five dated 10/12 debris dirs above are safe; do NOT remove any .pytest_tmp_aspects.

---

## E. Total reclaimable figure

**CLOSED - safe now, zero rebuild cost (on disk today):**

| Item | MiB |
|---|---|
| glm-5.3-flash-from-scratch/.pytest_tmp{2,3,4} + both .pytest_cache | 100.45 |
| data\ocr_textpool\raw\ (explicit .gitignore duplicate) | 47.14 |
| scratch\postproc_ab\ | 53.40 |
| scratch\stage2b2500\ | 22.46 |
| scratch\stage2final\ | 22.33 |
| scratch\asas_summ_train.parquet | 8.64 |
| scratch\stage2a\ | 6.02 |
| scratch\gate_overlap\ | 4.90 |
| scratch\render_demo\ + scratch\ui_demo\ | 0.16 |
| scratch\pytest_tmp\ + scratch\pytest_tmp_v\ | 0.12 |
| **TOTAL CLOSED (on disk)** | **approx 265.6 MiB (278.5 MB)** |
| data\catalog, data\legacy_check, data\snapshot_row13_row16_wip (already absent) | 0 |

**REGENERABLE (excluding the CLOSED items above):**

data\ total (15,028.7 MiB) minus bytes_pg_replay (3.34, REFERENCED) minus distillation_data (51.50, UNKNOWN) minus ocr_textpool/raw already counted CLOSED (47.14) = **approx 14,927 MiB**.
Plus scratch s10_real_smoke + s10_smoke = 202.35 MiB.
**REGENERABLE total = approx 15,129 MiB = 14.77 GiB.**

> **CLOSED + REGENERABLE combined = approx 15.04 GiB.** But 14.77 GiB of that is only regenerable AT A COST - re-downloading data\hf (7.0 GiB, 61,590 files), data\teacher (1.65 GiB), wholefile (network) and rebuilding the OCR corpora. Recommend reclaiming **only the CLOSED 265.6 MiB** now; treat the rest as a separate, user-approved, network/time-budgeted decision.

---

## F. Scratch scripts to PROMOTE (not delete)

scratch\ is fully gitignored (.gitignore: scratch/ ... NOT durable), so wiping it loses these instruments permanently. These three are LOAD-BEARING and cited across the durable docs - promote to e.g. scripts/bench/ (or the glm package) before any scratch purge:

1. **scratch\bench_playground.py** - the Phase-0 per-rung playground battery; 'the rung-level baseline every later change is measured against' (research/2026-10-14_per_part_playground_design.md section 7). Cited: TASKS.md:324, MEMORY.md:1795.
2. **scratch\bench_aspect_pool.py** - the aspect-pool (48) battery behind the E-119/E-122/E-122d verdicts. Cited: TASKS.md:323, MEMORY.md:1740, HANDOFF.md:31, research/WHAT_WORKS.md:441.
3. **scratch\bench_aspects.py** - the frozen 48-byte coding battery (external vs our arms, E-119). Cited: research/EXPERIMENTS.md:1209, TASKS.md:323 and 706, HANDOFF.md:39 and 158, MEMORY.md:1646 and 1674.

Secondary candidates worth promoting (evidence instruments, not confirmed load-bearing): val_bpb_byte.py, leak_check_e122.py, passk_preflight.py, compare_e121b.py, compare_rft.py, curve_e115.py, curve_e116.py.

True throwaways (do not promote): the ~25 fix_*.py, expdbg*.py, probe_*.py, hf_*_search.py, tead_*.py, dl_*.py, close_e4x/e5x.py, render_test.py.

---

## G. Unclassified / needs user decision

1. **data\distillation_data\ (51.5 MiB)** - LLM-teacher outputs; meta/gen_all.py exists but re-running it would not reproduce the same outputs. Class UNKNOWN, treat as IRREPLACEABLE unless the user confirms it is reproducible.
2. **data\reference\bytes_pg_replay exact build flags** - the string pg_replay appears NOWHERE in the repo (no script, log, md or meta key). The mixture is recoverable from meta.json (8,000 canonical + 2,700 playground, about 100/rung) but the precise invocation that produced it is not recorded. Record the command before this corpus is ever rebuilt or deleted, otherwise it is not faithfully regenerable.
3. **scratch\corpus_pipeline_fixture\ (0.05 MiB, created 11:28-11:32 today)** - active-looking test fixture; class REFERENCED/UNKNOWN.
4. **data\md_measure\, data\reallines\, data\benchmark\** - 'regenerable' only if the upstream datasets/source images are still reachable; classify REGENERABLE (network-dependent) and verify source availability before deleting.
5. **scratch\s10_real_smoke\ (198.7 MiB)** - re-runnable only via the glm-5.3-flash-from-scratch trainer; confirm the exact smoke config before reclaiming.

---

## H. Safety notes for whoever acts on this

- **Do not touch** data\reference\bytes_pg_replay while PID 37804 (job pwsh-107) is alive; it is memmapped by the live trainer runs\e122e_replay_lr1e4.
- **Do not touch** the in-flight .pytest_tmp_aspects (pytest PID 10668/36756).
- The **1,363 pre-existing unstaged deletions** (mostly runs/mex) are outside this survey's scope but should be explained before any further cleanup.
- Many 'REGENERABLE' corpora are also git-tracked (2,619 files under data\): deleting them locally is recoverable via git checkout, but dirties the working tree further.
- Measured free space: **7.63 GB** on E:. Only the CLOSED approx 265.6 MiB is risk-free to reclaim.

---

*Survey method: read-only PowerShell Measure-Object sums, Get-FileHash SHA256 on the reference corpora, grep/glob for producer scripts and config mappings, live process inspection (Get-CimInstance Win32_Process), and git status --porcelain. No writes outside this report.*
