# Unknown-set disk dossier - the ~7.4 GiB the two 2026-10-14 surveys could not classify

Date: 2026-10-14. Read-only investigation; nothing deleted. Companion surveys:
research/2026-10-14_disk_cleanup_survey.md and research/2026-10-14_data_cleanup_survey.md (both left this set UNKNOWN).

Method: Get-ChildItem -Recurse -File | Measure-Object; extension histograms; top-3 by size; ripgrep over TASKS.md,
HANDOFF.md, MEMORY.md, research/*.md, docs/, scripts/, configs/, and the nested glm-5.3-flash-from-scratch repo;
tools.read on small config/metadata files; git ls-files for tracking; full-file SHA-256 for the hybrid pair.

## Headline

- The claimed hybrid_c2_smoke == hybrid_c2_tiny byte-duplicate pair is FALSE: the 3 files have identical sizes but
  all three differ by SHA-256 (see Q1).
- The s10_* set is the S10 per-aspect LoRA line (nested glm-5.3-flash-from-scratch trainer), measured 2026-10-13;
  its verdict is already recorded in HANDOFF/MEMORY (see Q2).
- Runs we can now name: hybrid_c2_* -> arabic_ocr C2 (vit_hybrid) configs; stage05_pilot_gpu -> E-77/E-78;
  stage05b -> E-88; regime -> E-113; distillation_data -> C12/minimax3/omen_alpha SFT corpus.
- Git tracking (git ls-files count / file count): stage05_pilot_gpu 32/76, stage05b 35/52, distillation_data 2383/2383;
  s10_*, hybrid_c2_*, merged_*, regime all 0 (untracked).

## Measured table

| # | target | bytes / files | newest -> oldest (LastWriteTime) | what it contains | ownership (exact file:line) | regenerability | verdict |
|---|---|---|---|---|---|---|---|
| 1 | runs/s10_adapters/ | 2,506,060,810 B (2.33 GiB) / 156 | 2026-10-13T15:09:25 -> 2026-10-13T15:06:41 | 12 aspect dirs (clamp, contains_zero, count_even, count_positive, first_or_empty, grade, has_key, is_palindrome, key_count, lookup, repeat_text, sign), each checkpoint-NNNN/ with model.pt + metadata.json; ext .pt=72 .json=72 .jsonl=12. Biggest: grade/checkpoint-0003/model.pt 103,500,619; count_even/checkpoint-0002/model.pt 103,500,619; count_even/checkpoint-0003/model.pt 103,500,619 | S10 line: HANDOFF.md:28 and :33; MEMORY.md:1641 and :1756; research/WHAT_WORKS.md:424; research/2026-10-10_glm53_flash_rl_playground_v2_design.md:185; trainer glm-5.3-flash-from-scratch/scripts/train_aspect_lora.py:1; merge glm-5.3-flash-from-scratch/scripts/merge_loras.py:1. NO config references the run dir. metadata.json stage=aspect_lora, lora_r=8, lora_alpha=16, seed=31415 | No committed runner hardcodes runs/s10_adapters (grep found none); re-run train_aspect_lora.py once per aspect (GPU). Regenerable in principle, not by one command | NEEDS-USER - newest S10 evidence, but the S10 verdict is already recorded and re-creating the 2.33 GiB costs GPU/time |
| 2 | runs/s10_adapters_big/ | 1,147,763,476 B (1.07 GiB) / 69 | 2026-10-13T16:21:15 -> 2026-10-13T15:56:39 | 6 aspect dirs (clamp, first_or_empty, grade, is_palindrome, repeat_text, sign); ext .pt=31 .json=33 .jsonl=5. Biggest: sign/checkpoint-0005/model.pt 103,500,619; grade/checkpoint-0005/model.pt 103,500,619; is_palindrome/checkpoint-0004/model.pt 103,500,619 | Same S10 documents as #1 (no separate row). No file:line names runs/s10_adapters_big specifically. metadata stage=aspect_lora, r=8, alpha=16 | Same trainer as #1 (more epochs); no committed runner | NEEDS-USER - same S10 line, no ledger row at all, only on-disk copy; lowest confidence of the S10 set |
| 3 | runs/hybrid_c2_smoke/ | 1,087,065,925 B (1.01 GiB) / 3 | 2026-10-01T15:00:17 -> 2026-10-01T15:00:16 | best.pt 543,532,829; last.pt 543,532,957; history.json 139 (1 epoch: train 2.9068 / val 3.1977 / CER 1.0) | arabic_ocr/configs/hybrid_c2_smoke.yaml:38 (save_dir: runs/hybrid_c2_smoke); produced by the arabic_ocr vit_hybrid trainer (arabic_ocr/scripts/distill.py:175 writes best.pt, :180 history.json); runs/logs/hybrid_c2_smoke.log:1 'Config: hybrid_c2_smoke'. NO TASKS/HANDOFF/MEMORY row | Config + trainer exist: arabic_ocr/configs/hybrid_c2_smoke.yaml (save_dir = this dir) run through the arabic_ocr trainer; exact entrypoint not pinned (train.py vs distill.py) | NEEDS-USER - dead 2026-10-01 C2 attempt (CER 1.0), regenerable but not referenced by any ledger and NOT a duplicate |
| 4 | runs/hybrid_c2_tiny/ | 1,087,065,939 B (1.01 GiB) / 3 | 2026-10-01T14:39:59 -> 2026-10-01T14:39:58 | best.pt 543,532,829; last.pt 543,532,957; history.json 153 (1 epoch: train 6.9049 / val 3.1284 / CER 4.589) | arabic_ocr/configs/hybrid_c2_tiny.yaml:38 (save_dir: runs/hybrid_c2_tiny); same trainer as #3. NO ledger row | Same as #3 with hybrid_c2_tiny.yaml | NEEDS-USER - same as #3; explicitly NOT a byte-duplicate of smoke |
| 5 | runs/stage05_pilot_gpu/ | 1,465,373,929 B (1.36 GiB) / 76 | 2026-10-04T12:07:35 -> 2026-10-04T10:49:45 | dirs dyn/, lr1e5/, output/; ext .pdparams=24 .states=20 .json=9 .jsonl=9 .yml=8 .log=6; root eval_control*.json, eval_best.json, eval_epoch*.json, pilot_gpu.yml, train.log. Biggest: output/best_model/model.pdparams 61,043,023; dyn/lr1e5/output/iter_epoch_3.pdparams 61,043,023; dyn/lr1e5/output/iter_epoch_2.pdparams 61,043,023 | research/EXPERIMENTS.md:16 (E-77) and :15 (E-78); TASKS.md:250 (row 127 done); HANDOFF.md:2100 (E-77) and :2066 (E-78); arabic_ocr/STATUS.md:934 and :942 | Training used an external PaddleOCR stack (not found as a committed command); eval reproducible via arabic_ocr/scripts/stage05_eval.py --repo E:\PaddleOCR. 32 of 76 files are git-tracked (the JSON/yml/log); 24 .pdparams weights untracked | NEEDS-USER - recorded E-77/E-78 FAIL evidence; 1.4 GiB is untracked weights rebuildable only with the external PaddleOCR recipe |
| 6 | runs/stage05b/ | 557,339,132 B (531 MiB) / 52 | 2026-10-06T14:04:11 -> 2026-10-06T11:53:15 | dir output/ (best_model, best_accuracy, iter_epoch_1..); root eval_*/diag_* json+jsonl, _e88_row.md, _e88_ww.md, _make_cfg.py, train_full_lr1e5.yml. ext .json=12 .jsonl=12 .pdparams=9 .states=8 .py=3 .log=2 .md=2 .txt=2 .yml=2. Biggest: output/best_model/model.pdparams 61,043,023; output/best_accuracy.pdparams 61,043,023; output/iter_epoch_1.pdparams 61,043,023 | HANDOFF.md:1377,1382,1391,1426,1429,1482,1491,1498,1503; TASKS.md:253 (row 130 CLOSED FAIL E-88); MEMORY.md:19; arabic_ocr/scripts/build_stage05b_lists.py:45 (out_dir = REPO/'runs'/'stage05b') | build_stage05b_lists.py:45 rebuilds the lists; training needs the external PaddleOCR stack. 35 of 52 files git-tracked; 9 .pdparams untracked | NEEDS-USER (lean SAFE-TO-DELETE) - E-88 is closed FAIL and the verdict/JSON are tracked; only the untracked .pdparams weights require the external stack to rebuild |
| 7 | runs/s10_merged_average/ and runs/s10_merged_ties/ | 103,042,773 B (98 MiB) each / 3 each | avg 2026-10-13T15:10:12; ties 2026-10-13T15:10:17 | model.pt 103,041,675; metadata.json 870; config.json 228 (dim 192, 12 layers, 8 experts, vocab 260) | S10 line: HANDOFF.md:28 and :33 (s10avg/s10ties results); MEMORY.md:1756. metadata.json (average) records the exact 12 input adapters runs/s10_adapters/*/adapter.pt, method=average, baked_from glm-5.3-flash-from-scratch/runs/e121_ref/checkpoint-0100, loaded 48 / merged 24, max_abs_diff 1.9e-6; ties dir is the TIES merge | merge_loras.py --method average|ties over runs/s10_adapters (CPU, seconds); metadata.json already records the inputs | KEEP - tiny (196 MiB total) and cited by name as the measured S10 merges |
| 8 | glm-5.3-flash-from-scratch/runs/regime/ | 192,491 B (0.2 MiB) / 7 | 2026-10-10T09:49:27 -> 2026-10-08T14:34:50 | children.log 165,025; regime.log 17,023; a5-curve.json 3,646; a6-curve.json 3,276; selection.json 871; analyze_receipts.py 1,212; apply_e113.py 1,438 | Owned by glm-5.3-flash-from-scratch/scripts/regime_match.py:29,43 (LOG_DIR = ROOT/'runs'/'regime'); research/2026-10-08_glm53_flash_regime_match.md:62 and E-113; README.md:386; HANDOFF.md:648,:729 | glm-5.3-flash-from-scratch/scripts/regime_match.py --stage all (re-runs pretrain + RL; expensive) | KEEP - E-113 evidence cited line-by-line, 0.2 MiB |
| 9 | data/distillation_data/ | 53,997,983 B (51.5 MiB) / 2383 | 2026-09-09T07:56:28 -> 2026-09-09T07:56:27 | dirs minimax3/ and omen_alpha/ (four shapes A/B/C/D jsonl batches + combined/); ext .jsonl=2326 .py=35 .md=11 .txt=8 .json=3; generators included (omen_alpha/meta/gen_all.py 261,723; gen_lib.py; aggregate.py; minimax3/meta/gen_shape_*.py, gen_topup*.py). Biggest: omen_alpha/combined/all_train.jsonl 8,407,505; minimax3/combined/all_train.jsonl 8,044,925; minimax3/shape_a_instruction_code/train.jsonl 5,728,508 | configs/distill_custom.yaml:23 and configs/distill_custom_b.yaml:23; configs/sft_v2.yaml:24 and configs/sft_v2_e1.yaml:24; configs/kt2_judge.yaml:34 and kt2_judge_r2.yaml:27; configs/yarn_lora_sft_v1.yaml:44; docs/plans/2026-09-10_kt_resume_handoff.md:197-199; TASKS.md:34 (row 19); data/distillation_data/minimax3/README.md:1, HANDOFF.md:1, omen_alpha/HANDOFF.md:1 | omen_alpha/meta/gen_all.py is deterministic ('re-running gen_all reproduces identical bytes', omen_alpha/HANDOFF.md); minimax3 generators + aggregate.py/finalize.py present. AND fully git-tracked: git ls-files data/distillation_data = 2383 = file count | KEEP - only 51.5 MiB, git-tracked, referenced by 7 configs, generator code alongside it; the 'IRREPLACEABLE' label overstates the risk (working copy is recoverable from git; the templates rebuild omen_alpha byte-for-byte) |

## Q1 - Is hybrid_c2_smoke a byte-duplicate of hybrid_c2_tiny?

NO. Full recursive SHA-256 inventory (relative path | size | sha256) was computed for both directories; the manifest
hashes differ and every corresponding file differs:

| file | size | smoke sha256 | tiny sha256 |
|---|---|---|---|
| best.pt | 543,532,829 | db7cb9b715359837048c8cebebdaf9c086282cb9ce565da0f4423423114bb8e7 | 428d10c57856e7f0c84050f224c1f16738c45751349a531cbc82809443b050d6 |
| last.pt | 543,532,957 | a1c1de866feffd68766c3a89158c15678a5155e09b5dfb188cce2f6d36f9f39a | 49fdea29234aae004dcc5177e41470cb22d72dac9897f49c1b712df857147431 |
| history.json | 139 / 153 | 6c0e02d9583ba28f5a49211ea8e2356e692ce4def146d128289bd1ba8ec14b46 | fe8008231cf8d7a11f08b6e2ce4e917ff61ff1687de06d5446452df8b8a0fce8 |

Manifest SHA-256: smoke 9D933E49E380EE3D4AF0B6B5BE51C13983752F44EED5AEDD2839228FDBAD3904;
tiny 049A822DE7738C9089854646CC59920AE07A2309F4541244EFADF983305DDA45. The models are the same architecture and
dtype shape (hence identical byte sizes) trained on different data (linecorpus_smoke vs linecorpus_tiny) with
different history (CER 1.0 vs 4.589). Deleting 'the duplicate' would delete a distinct run.

## Q2 - What are s10_adapters / s10_adapters_big / s10_merged_* and is S10 live?

1. They ARE per-aspect LoRA artifacts of the S10 line. Each aspect checkpoint metadata.json says stage=aspect_lora,
   lora_r=8, lora_alpha=16, seed=31415 (e.g. runs/s10_adapters/clamp/checkpoint-0002/metadata.json). The 12 aspect
   names are exactly the S10 aspect families. The merged metadata.json lists the 12 inputs as
   runs/s10_adapters/<aspect>/adapter.pt and records method=average / TIES with max_abs_diff ~1.9e-6.
2. Docs DO reference S10 (by line name, not by run path): HANDOFF.md:28 and :33; MEMORY.md:1641 and :1756;
   research/WHAT_WORKS.md:424; research/2026-10-10_glm53_flash_rl_playground_v2_design.md:185 (the S10 spec);
   trainers glm-5.3-flash-from-scratch/scripts/train_aspect_lora.py:1 and merge_loras.py:1. NO configs reference
   the run directories, and TASKS.md has no S10 row (its nearest rows are 170/E-121b and 171, a different ladder).
3. Is the S10 line still active? No - it is a measured probe, not an open task. HANDOFF.md 2026-10-13 records the
   S10 measurement: all 12 adapters trained from arm A (110,592 LoRA params each), average and TIES merges gave
   base 9/32 -> 10/32 and scored 26/48 on the aspect pool - IDENTICAL to the base and to RL. Verdict: structurally
   SAFE but not yet EFFECTIVE (only 12 training examples per aspect). The active ladder is E-121b/E-122 (TASKS 170, in_progress),
   which does not consume these adapters. No process was seen writing under runs/s10_* (newest write 2026-10-13 16:21).

## Verdict summary (one line each)

| target | verdict | strongest reason |
|---|---|---|
| runs/s10_adapters/ | NEEDS-USER | Newest S10 evidence, but its verdict is already recorded and the 2.33 GiB is GPU-expensive to rebuild |
| runs/s10_adapters_big/ | NEEDS-USER | Same S10 line with no ledger row and no other copy |
| runs/hybrid_c2_smoke/ | NEEDS-USER | Dead 2026-10-01 C2 attempt (CER 1.0); regenerable from its config, not referenced |
| runs/hybrid_c2_tiny/ | NEEDS-USER | Dead C2 attempt (CER 4.59); NOT a duplicate of smoke |
| runs/stage05_pilot_gpu/ | NEEDS-USER | Recorded E-77/E-78 evidence; 1.4 GiB of untracked weights need the external PaddleOCR stack |
| runs/stage05b/ | NEEDS-USER (lean SAFE-TO-DELETE) | E-88 closed FAIL, verdict + JSON tracked; only untracked .pdparams need the external stack |
| runs/s10_merged_average/ + ties/ | KEEP | 196 MiB total, cited by name as the measured S10 merges |
| glm-5.3-flash-from-scratch/runs/regime/ | KEEP | 0.2 MiB E-113 evidence cited line-by-line |
| data/distillation_data/ | KEEP | 51.5 MiB, fully git-tracked, referenced by 7 configs, generators included |

## Caveats

- PowerShell Get-ChildItem -Recurse -File silently returned empty for some runs/ subpaths in later probes (likely a
  >260-char checkpoint path aborting enumeration); counts/bytes here are from the first clean pass and the turn-1
  measurement, and were cross-checked by extension histograms.
- runs/stage05b/_e88_row.md, _e88_ww.md and glm-5.3-flash-from-scratch/runs/regime/selection.json were listed by
  Get-ChildItem but could not be read back through the read tool or Get-Content; ownership for E-88/E-113 therefore
  rests on TASKS.md/HANDOFF.md/trainer-script hits, which are unambiguous.
- Minimax3/HANDOFF.md section 5 claims the corpus is not in git; measurement contradicts it for data/distillation_data
  (2383/2383 files in the index). Treat the handoff sentence as stale or as referring to the nested minimax3/ tree.
