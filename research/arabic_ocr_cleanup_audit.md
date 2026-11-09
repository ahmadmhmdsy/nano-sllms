# Arabic OCR disk-usage cleanup audit (READ-ONLY)

**Scope.** Inventory only. Nothing was deleted, moved, renamed, truncated or committed. No GPU job, training run or inference was executed. All measurements come from read-only Get-ChildItem / Measure-Object / Get-FileHash / git status / git check-ignore / git ls-files.

**Audit target.** E:\python projects\nano_SLMs on drive E:. At audit time Get-PSDrive E reported **Free = 5,576,601,600 B (5.19 GiB)**, Used = 250,678,591,488 B. This matches the 5.19 GB free stated in the task.

**Totals measured.** arabic_ocr/runs = **9,508,192,387 B (8.86 GiB)**, 230 files, of which *.pt = 3,498,960,949 B and logits caches = 5,997,060,328 B. The remaining ~12.2 MB is tracked JSON and log text.

---

## 1. Checkpoints - arabic_ocr/runs/**/*.pt

.pt files total: **3,498,960,949 B** across 108 files in 7 run dirs.

Grouped rows cover the two --keep-eval-ckpts populations; every individual file in a group has the same size, classification, evidence and risk.

| path | bytes | mtime | classification | evidence | delete risk |
|---|---:|---|---|---|---|
| arabic_ocr/runs/grow_test/planA_final_L14.pt | 48,654,787 | 2026-10-08 08:29 | LIVE | HANDOFF.md:86 - "Init = arabic_ocr/runs/grow_test/planA_final_L14.pt" (Plan B / stage06c init) | high - run init |
| arabic_ocr/runs/grow_test/planA14000_L14.pt | 48,654,575 | 2026-10-07 13:30 | STALE | no reference found (E-106 growth work, closed) | low |
| arabic_ocr/runs/grow_test/d384_l8.pt | 61,401,203 | 2026-10-07 12:47 | STALE | no reference found (E-106 widen+deepen arm, closed) | low |
| arabic_ocr/runs/grow_test/d256_l14.pt | 48,653,303 | 2026-10-07 12:47 | STALE | no reference found (E-106 depth 4->14 arm, closed) | low |
| arabic_ocr/runs/grow_test/d256_l8.pt | 29,673,075 | 2026-10-07 12:47 | STALE | no reference found (E-106 depth 4->8 arm, closed) | low |
| arabic_ocr/runs/grow_test/d384_l4.pt | 26,693,345 | 2026-10-07 12:48 | STALE | no reference found (E-106 widen-only arm, closed) | low |
| arabic_ocr/runs/grow_test/d8_l8.pt | 29,665,115 | 2026-10-07 12:46 | ORPHAN | no reference found anywhere (not even in E-106) | low |
| arabic_ocr/runs/stage05d/ckpt_last.pt | 17,021,657 | 2026-10-06 16:20 | LIVE | STATUS.md:437 "arabic_ocr/runs/stage05d/ckpt_last.pt (weights local-only)"; STATUS.md:425 scorecard artefact | high |
| arabic_ocr/runs/stage05e/char_lm.pt | 3,505,373 | 2026-10-07 08:03 | LIVE | stage05e/scorecard_teacher.json:7 and scorecard_adapt11700.json:7 carry "lm": "arabic_ocr\\runs\\stage05e\\char_lm.pt" | medium - only 3.3 MiB |
| arabic_ocr/runs/stage05f/ckpt_last.pt | 17,021,657 | 2026-10-07 09:12 | LIVE | STATUS.md:317 "Artifacts: arabic_ocr/runs/stage05f/{ckpt_last.pt, ckpt_final.pt, ...}" (B2 model) | high |
| arabic_ocr/runs/stage05f/ckpt_final.pt | 17,021,749 | 2026-10-07 09:13 | LIVE | STATUS.md:317 (same line) | high |
| arabic_ocr/runs/stage06a/ckpt_last.pt | 184,201,483 | 2026-10-07 12:01 | STALE (closed E-104, FAIL) | STATUS.md:36 and STATUS.md:204 document --resume from it; E-104 in research/EXPERIMENTS.md records the from-scratch run FAIL | medium - documented resume path, but the experiment is closed and superseded by Plan A/B |
| arabic_ocr/runs/stage06a/probe/ckpt_latest.pt | 184,201,483 | 2026-10-07 12:01 | DUPLICATE | Get-FileHash SHA256 = C689ADCEB0742DF83D540F8204FA4D8F9AD9D392B096B17401C71386F2FC8BFE, identical to stage06a/ckpt_last.pt (same hash, same size) | none - content preserved in the kept copy |
| arabic_ocr/runs/stage06b/ckpt_last.pt | 51,050,699 | 2026-10-07 16:04 | LIVE | HANDOFF.md:61 "planA_final (66,000)" and HANDOFF.md:68 "planA_final = ckpt_last.pt"; key planA_final in stage06b/blank_penalty_verdict.json | high |
| arabic_ocr/runs/stage06b/ckpt_final.pt | 51,050,949 | 2026-10-07 16:06 | ORPHAN | no reference found in STATUS.md / HANDOFF.md / TASKS.md / MEMORY.md / research/EXPERIMENTS.md / any scorecard*.json / blank_penalty_verdict.json | medium - final full-state (model+optimizer) checkpoint of a PASS-verdict run |
| arabic_ocr/runs/stage06b/ckpt_step059000.pt | 17,022,209 | 2026-10-07 15:43 | LIVE | HANDOFF.md:63 shortlist tag planA_s59000; key in blank_penalty_verdict.json | high |
| arabic_ocr/runs/stage06b/ckpt_step061000.pt | 17,022,209 | 2026-10-07 15:49 | LIVE | HANDOFF.md:62 shortlist tag planA_s61000; key in blank_penalty_verdict.json | high |
| arabic_ocr/runs/stage06b/ckpt_step063000.pt | 17,022,209 | 2026-10-07 15:55 | LIVE | HANDOFF.md:64 shortlist tag planA_s63000; key in blank_penalty_verdict.json | high |
| arabic_ocr/runs/stage06b/ckpt_step066000.pt | 17,022,209 | 2026-10-07 16:04 | STALE (weight-twin of ckpt_last) | no doc names this file; E-110 and HANDOFF.md:68 define planA_final as ckpt_last.pt, step 66,000. File differs from ckpt_last only by the optimizer block (17.0 MB vs 51.1 MB) | low |
| arabic_ocr/runs/stage06b/ckpt_step001000.pt ... step058000, step060000, step062000, step064000, step065000 (**62 files**) | 1,055,376,958 | 2026-10-07 12:44-16:01 | STALE | no reference found; each is a --keep-eval-ckpts periodic-eval artefact. Card values are preserved in the tracked stage06b/scorecard.json and train_log.jsonl | low |
| arabic_ocr/runs/stage06c/ckpt_last.pt | 145,967,307 | 2026-10-08 10:44 | LIVE | HANDOFF.md:12 "planB step 26,000"; E-111; key planB_final in stage06c/blank_penalty_verdict.json | high |
| arabic_ocr/runs/stage06c/ckpt_final.pt | 145,968,037 | 2026-10-08 10:47 | ORPHAN | no reference found in the six doc sources or any scorecard/verdict JSON | medium - final full-state checkpoint of a PASS-verdict run |
| arabic_ocr/runs/stage06c/ckpt_step023000.pt | 48,657,283 | 2026-10-08 10:31 | LIVE | HANDOFF.md:14; key planB_s23000 in blank_penalty_verdict.json | high |
| arabic_ocr/runs/stage06c/ckpt_step024000.pt | 48,657,283 | 2026-10-08 10:35 | LIVE | HANDOFF.md:13; key planB_s24000 in blank_penalty_verdict.json | high |
| arabic_ocr/runs/stage06c/ckpt_step026000.pt | 48,657,283 | 2026-10-08 10:44 | LIVE | HANDOFF.md:12; key planB_s26000; E-111 records its weights are identical to ckpt_last (0 of 206 tensors differ) | high |
| arabic_ocr/runs/stage06c/ckpt_step001000.pt ... step022000, step025000 (**23 files**) | 1,119,117,509 | 2026-10-08 08:38-10:40 | STALE | no reference found; --keep-eval-ckpts artefacts, cards preserved in tracked stage06c/scorecard.json | low |

### 1a. Checkpoint sub-totals

| run dir | files | bytes | classification summary |
|---|---:|---:|---|
| arabic_ocr/runs/grow_test | 7 | 293,395,403 | 1 LIVE init + 6 STALE/ORPHAN growth arms |
| arabic_ocr/runs/stage05d | 1 | 17,021,657 | LIVE |
| arabic_ocr/runs/stage05e | 1 | 3,505,373 | LIVE (char LM) |
| arabic_ocr/runs/stage05f | 2 | 34,043,406 | LIVE |
| arabic_ocr/runs/stage06a | 2 | 368,402,966 | 1 STALE + 1 DUPLICATE |
| arabic_ocr/runs/stage06b | 68 | 1,225,567,442 | 4 LIVE (ckpt_last + 3 steps) + 1 ORPHAN ckpt_final + 62 STALE steps + 1 weight-twin |
| arabic_ocr/runs/stage06c | 28 | 1,557,024,702 | 4 LIVE (ckpt_last + 3 steps) + 1 ORPHAN ckpt_final + 23 STALE steps |
| **total** | **108** | **3,498,960,949** |  |

---

## 2. Logits caches - arabic_ocr/runs/**/logits/**

All caches are .npz, 5 files per tag (isippt_bench, khatt_bench, khatt_teklia_test, muharaf_test, muharaf_val). A h96 tag is **375,115,500 B = 357.7 MiB**, confirming the task figure. All cache dirs are covered by .gitignore:212 (arabic_ocr/runs/stage*/logits/).

Total logits bytes = **5,997,060,328 B (5.59 GiB)** across 14 tags / 70 files.

| tag path | bytes | mtime | classification | evidence | delete risk |
|---|---:|---|---|---|---|
| stage05e/logits/teacher | 375,114,240 | 2026-10-07 07:59-08:00 | LIVE | E-109 in research/EXPERIMENTS.md; STATUS.md:376 "Cached logits: arabic_ocr/runs/stage05e/logits/teacher/" | high |
| stage05e/logits/adapt11700 | 375,114,240 | 2026-10-07 08:03-08:04 | LIVE | E-98 / E-109 (ckpt-11700 + blank penalty reference row) | high |
| stage05e/logits/stage05f | 375,114,240 | 2026-10-07 10:27 | LIVE | E-101 / E-109; STATUS.md:269 "cache stage05f logits" | high |
| stage05e/logits/teacher_h128 | 499,352,956 | 2026-10-07 10:29-10:30 | STALE | E-102 (research/EXPERIMENTS.md:1193) closed/REFUTED height sweep; TASKS.md:285 closed row 143; TASKS.md:289 "The stale teacher_h128/h160/h192 caches (1.78 GB) are from the closed E-102 height sweep and are the first reclaim candidate" | low |
| stage05e/logits/teacher_h160 | 623,596,984 | 2026-10-07 10:30-10:31 | STALE | same E-102 evidence | low |
| stage05e/logits/teacher_h192 | 747,843,668 | 2026-10-07 10:32-10:34 | STALE | same E-102 evidence; TASKS.md:285 records the h192 numbers already in the ledger | low |
| stage06b/logits/planA_final | 375,115,500 | 2026-10-07 16:13 | LIVE | key planA_final in stage06b/blank_penalty_verdict.json (E-110) | high |
| stage06b/logits/planA_s59000 | 375,115,500 | 2026-10-07 16:14 | LIVE | key planA_s59000 in stage06b/blank_penalty_verdict.json | high |
| stage06b/logits/planA_s61000 | 375,115,500 | 2026-10-07 16:14 | LIVE | key planA_s61000 in stage06b/blank_penalty_verdict.json | high |
| stage06b/logits/planA_s63000 | 375,115,500 | 2026-10-07 16:15-16:16 | LIVE | key planA_s63000 in stage06b/blank_penalty_verdict.json | high |
| stage06c/logits/planB_final | 375,115,500 | 2026-10-08 10:49-10:50 | DUPLICATE | **verified**: all 5 npz files SHA256-match the corresponding files in stage06c/logits/planB_s26000 (e.g. muharaf_val.npz = 0AAB3C0F8AFAFE72BB18656A8F9C22AC3CC748CEC000131AE21AEE1C23B7FF29). planB_final is ckpt_last = step 26,000 = planB_s26000 (HANDOFF.md:31-33, E-111) | none - byte-identical to the kept planB_s26000 tag |
| stage06c/logits/planB_s23000 | 375,115,500 | 2026-10-08 10:52-10:53 | LIVE | key planB_s23000 in stage06c/blank_penalty_verdict.json (E-111) | high |
| stage06c/logits/planB_s24000 | 375,115,500 | 2026-10-08 10:53-10:54 | LIVE | key planB_s24000 | high |
| stage06c/logits/planB_s26000 | 375,115,500 | 2026-10-08 10:50-10:51 | LIVE | key planB_s26000 (keep this one of the duplicate pair) | high |

No logits/ directory exists at arabic_ocr/runs/stage06a/logits even though STATUS.md:115-128 describes caching there - see UNVERIFIED items.

---

## 3. Other large directories requested

| path | bytes | files | mtime | git status | doc reference | risk |
|---|---:|---:|---|---|---|---|
| arabic_ocr/runs/mex | **MISSING** | - | - | - | TASKS.md:137 and TASKS.md:171 reference runs/mex/..., but the path does not exist in this working tree | n/a - nothing to reclaim |
| arabic_ocr/runs/sft_t1 | **MISSING** | - | - | - | not present under arabic_ocr/runs | n/a |
| arabic_ocr/runs/sft_v2_e1 | **MISSING** | - | - | - | not present under arabic_ocr/runs | n/a |
| arabic_ocr/runs/smoke | **MISSING** | - | - | - | absent | n/a |
| arabic_ocr/runs/soup_p0 | **MISSING** | - | - | - | absent | n/a |
| runs/mex (repo root) | **MISSING** | - | - | - | same as above | n/a |
| runs/sft_t1, runs/sft_v2_e1, runs/smoke, runs/soup_p0 (repo root) | **MISSING** | - | - | - | HANDOFF.md:1492 (2026-09-16) listed runs/smoke + runs/soup_p0 as "STILL ON DISK ~1.43 GB"; they are no longer present | n/a - already gone |
| runs/ (repo root, whole tree) | 25,529,696,080 | 21,999 | - | mixed; 1,915 files tracked (git ls-files runs) | out of the arabic_ocr scope; largest children: runs/laya 6.69 GB, runs/stage05_pilot_gpu 4.02 GB, runs/trocr_finetune_v2 2.15 GB, runs/yarn_lora_sft_v1 1.82 GB | not assessed here - out of scope |
| checkpoint_backup/ | 136,810,450 | 3 | 2026-09-24 | **ignored** by .gitignore:59 (checkpoint_backup/); git ls-files checkpoint_backup = 0 entries | HANDOFF section 3b / AGENTS.md: "user-staged machine-move checkpoint zips". Contents: U1_u1_full_2026-09-24.zip 113,171,237 B, uiline_alpha_resume_20260924.zip 23,637,773 B, RESUME_uiline_alpha_20260924.md 1,440 B | **high** - user-staged backups, not agent-managed |
| data/catalog/ | **MISSING on disk** | 0 | - | **9 files tracked and deleted in the working tree** (git ls-files --deleted data/catalog = 9; git status shows " D data/catalog/...") | deletion is staged-but-uncommitted; nothing to reclaim (already gone) | n/a - recoverable only via git checkout |
| data/hf (incl. data/hf/muharaf) | 1,404,797,140 | 31,173 | - | **NOT matched by any .gitignore rule** (git check-ignore -v data/hf data/hf/muharaf returned nothing); git ls-files data/hf = 0 | the training corpus | **NEVER DELETE** |
| arabic_ocr/models/ | 168,193,656 | 126 | - | **ignored** by .gitignore:178 (arabic_ocr/models/) | local-only model weights | **NEVER DELETE** |
| data/ocr_textpool/raw/ | 49,431,026 | 21 | - | **ignored** by .gitignore:193 (data/ocr_textpool/raw/) | explicit repo rule | **NEVER DELETE** |
| C:\\Users\\Ahmad Mahmoud\\.cache\\huggingface | 56,161,628,615 | 2,776 | - | OUTSIDE the repo | hub 47,585,703,395 B, datasets 8,544,993,971 B, lerobot 7,688,704 B, xet 23,236,663 B | **out of scope / do not touch** - reported for size only |

Root runs/ and arabic_ocr/runs/ are different trees. The five run dirs named in the task brief do not exist in either tree; the arabic OCR runs live only under arabic_ocr/runs/{grow_test,stage05d,stage05e,stage05f,stage06a,stage06b,stage06c} plus tiny smoke/card dirs.

---

## 4. NEVER-DELETE list

1. data/hf/muharaf/** (and all of data/hf, 1,404,797,140 B) - training corpus, explicit user rule.
2. data/ocr_textpool/raw/*.json (49,431,026 B) - explicit user rule.
3. arabic_ocr/models/** (168,193,656 B) and every other local model-weight file - LOCAL-ONLY policy.
4. arabic_ocr/runs/stage06b/ckpt_last.pt (51,050,699 B) - planA_final, the Plan A verdict checkpoint.
5. arabic_ocr/runs/stage06b/ckpt_step059000.pt, ckpt_step061000.pt, ckpt_step063000.pt (51,066,627 B) - the three shortlisted Plan A checkpoints behind live verdict keys.
6. arabic_ocr/runs/stage06c/ckpt_last.pt (145,967,307 B) and ckpt_step023000/024000/026000.pt (145,971,849 B) - the Plan B verdict checkpoints.
7. arabic_ocr/runs/grow_test/planA_final_L14.pt (48,654,787 B) - stage06c run init.
8. All LIVE logits tags: stage05e/logits/{teacher,adapt11700,stage05f} (1,125,342,720 B); stage06b/logits/{planA_final,planA_s59000,planA_s61000,planA_s63000} (1,500,462,000 B); stage06c/logits/{planB_s23000,planB_s24000,planB_s26000} (1,125,346,500 B).
9. arabic_ocr/runs/stage05d/ckpt_last.pt, stage05f/ckpt_last.pt, stage05f/ckpt_final.pt, stage05e/char_lm.pt.
10. checkpoint_backup/** (136,810,450 B) - user-staged machine-move architecture.
11. C:\\Users\\Ahmad Mahmoud\\.cache\\huggingface/** (56.16 GB) - outside the repo; not ours to reclaim.
12. Every tracked file under arabic_ocr/runs/** (the 27 git ls-files arabic_ocr/runs entries: scorecards, verdicts, logs, per-line JSONL) - the evidence for the verdicts.

---

## 5. Reclaimable totals grouped by risk tier

| tier | bytes | GiB | items |
|---|---:|---:|---|
| **none** (byte-identical copy retained elsewhere) | 559,316,983 | 0.52 GiB | stage06a/probe/ckpt_latest.pt; stage06c/logits/planB_final |
| **low** (closed experiment / unreferenced eval checkpoint) | 4,307,050,900 | 4.01 GiB | teacher_h128/h160/h192; 63 stage06b step ckpts; 23 stage06c step ckpts; 6 grow_test arms |
| **medium** (no doc reference, but a final/resumable full-state checkpoint) | 381,220,469 | 0.36 GiB | stage06a/ckpt_last.pt; stage06b/ckpt_final.pt; stage06c/ckpt_final.pt |
| **high** | 0 (never delete) | 0.00 | all LIVE artifacts |
| **TOTAL none+low (safe first pass)** | **4,866,367,883** | **4.53 GiB** |  |
| TOTAL none+low+medium | 5,247,588,352 | 4.89 GiB |  |

The safe-first-pass figure is **4,866,367,883 B, about 4.53 GiB**, i.e. roughly 87% of the 5.19 GiB currently free on E:.

---

## 6. Recommended order of reclamation (bytes per unit of risk)

1. **arabic_ocr/runs/stage05e/logits/teacher_h128, teacher_h160, teacher_h192 - 1,870,793,608 B (1.74 GiB), risk low.** Closed E-102 height sweep; TASKS.md:289 already names these as the first candidate. No live verdict reads them (the live stage05e verdicts use teacher/adapt11700/stage05f only). Best bytes-per-risk item.
2. **23 unreferenced arabic_ocr/runs/stage06c/ckpt_step*.pt - 1,119,117,509 B (1.04 GiB), risk low.** All --keep-eval-ckpts artefacts except 023000/024000/026000. Per-checkpoint card values remain in the tracked stage06c/scorecard.json; all three live verdict checkpoints stay.
3. **63 unreferenced arabic_ocr/runs/stage06b/ckpt_step*.pt - 1,072,399,167 B (1.00 GiB), risk low.** Includes ckpt_step066000.pt (weight-twin of the kept ckpt_last.pt). Card trajectory remains in tracked stage06b/scorecard.json + train_log.jsonl.
4. **arabic_ocr/runs/stage06c/logits/planB_final - 375,115,500 B (0.35 GiB), risk none.** Byte-identical to the kept planB_s26000 tag (SHA256 verified on all 5 npz). The recorded verdict is unchanged.
5. **6 stale arabic_ocr/runs/grow_test growth arms - 244,740,616 B (0.23 GiB), risk low.** E-106 is closed and its verdict is in the ledger; keep planA_final_L14.pt (run init).
6. **arabic_ocr/runs/stage06a/probe/ckpt_latest.pt - 184,201,483 B (0.17 GiB), risk none.** Exact SHA256 duplicate of the kept stage06a/ckpt_last.pt.
7. *(Optional, needs a user call)* stage06c/ckpt_final.pt + stage06b/ckpt_final.pt - 197,018,986 B (0.18 GiB), risk medium. No doc reference; they are the trainer-written final full-state files (model + optimizer) of two PASS-verdict runs. Reclaim only if the user confirms the optimizer state is not needed.
8. stage06a/ckpt_last.pt - 184,201,483 B, risk medium. STATUS.md documents --resume; E-104 closed as FAIL. Leave unless the user decides the from-scratch run is dead.

---

## 7. UNVERIFIED items and corrections to the brief

1. **CORRECTION - the stated byte-identical duplicate is not byte-identical.** The brief (and HANDOFF.md:32 wording "byte-identical") says stage06c/ckpt_last.pt equals ckpt_step026000.pt. Measured: ckpt_last.pt = 145,967,307 B, SHA256 16DEFA007D7A52FF5A5973E7F2D5CF288802DEA050B214D37F096691840DE98A; ckpt_step026000.pt = 48,657,283 B, SHA256 04038CB9D1780AA645B11E8432A16BD7859DE7CA80A77E3E7BC1D79C37A441B0. The FILES are not identical because ckpt_last.pt / ckpt_final.pt carry the optimizer state (~3x model size). What IS true (and what HANDOFF / E-111 mean) is that their **model weights** are identical: "0 of 206 tensors differ, max|delta| 0.000e+00". Note also that the brief says "stage06b/ckpt_last.pt", but step 026000 exists only in stage06c.
2. **CORRECTION - the five named run dirs do not exist.** arabic_ocr/runs/{mex,sft_t1,sft_v2_e1,smoke,soup_p0} and runs/{mex,sft_t1,sft_v2_e1,smoke,soup_p0} are all MISSING (Test-Path false; absent from the 47-dir root runs/ listing and the 11-dir arabic_ocr/runs/ listing). TASKS.md:137 and TASKS.md:171 still reference runs/mex/..., and HANDOFF.md:1492 still lists runs/smoke and runs/soup_p0 as on-disk (~1.43 GB), so those docs are STALE.
3. **UNVERIFIED - arabic_ocr/runs/stage06a/ckpt_final.pt is referenced by STATUS.md:115/122/127 but does not exist on disk.** Only ckpt_last.pt and probe/ckpt_latest.pt are present. Those STATUS.md command blocks appear stale/aspirational.
4. **UNVERIFIED - data/hf git tracking.** data/hf and data/hf/muharaf are not matched by any .gitignore rule, and git ls-files data/hf = 0. This is irrelevant to the decision (NEVER DELETE per rule) but the tracking status is not fully characterised.
5. **Due diligence on "unreferenced" step files.** Classification is by exact-path / step-number search across arabic_ocr/STATUS.md, HANDOFF.md, TASKS.md, MEMORY.md, research/EXPERIMENTS.md, every scorecard*.json, both blank_penalty_verdict.json, plus a ckpt_step grep across *.py and *.md. Absence in those sources is reported as "no reference found". It is possible a scratch script outside those sources reads a step file, but no such reference was found and none of the 66/26 step files is named in the pre-registered shortlist rules (TASKS.md:289, E-110, E-111) other than 059000/061000/063000 and 023000/024000/026000.
6. **UNVERIFIED - step checkpoints were not hash-checked against each other.** They are all equal-sized and their mtimes imply a monotone training sequence; only the same-size clusters were enumerated. Reclaiming the 62+23 unreferenced files assumes each holds its own step, which the trainer naming and mtimes support but was not hash-confirmed (that would require hashing ~2.2 GB read-only).

---

## 8. Method (read-only)

- File sizes/mtimes: Get-ChildItem -Recurse -File and Measure-Object -Property Length -Sum.
- Duplicate proofs: Get-FileHash -Algorithm SHA256 (stage06a pair, stage06c logits pair, stage05f pair, grow_test pair, stage06b key files).
- Tracking: git status --porcelain, git ls-files, git ls-files --deleted, git check-ignore -v.
- References: grep over the six durable docs, all scorecard*.json, both blank_penalty_verdict.json, *.py, *.md.
- Disk: Get-PSDrive E.
- Written file: this report only. No other file was created, modified, deleted or committed.

## EXECUTED 2026-10-07 (user-approved)

The risk-**none** + risk-**low** tier was deleted in one gated pass: **113 files / 4,866,367,883 B (4.53 GiB)**,
exactly matching this report's subtotal. Gates asserted before the first `Remove-Item`: the byte total equalled
4,866,367,883; stage06b/stage06c/grow_test counts were 63/23/6; all 13 named never-delete `.pt` files were present;
and no never-delete path appeared in the delete list. Drive E: free 5.19 -> **9.73 GiB**.

Deliberately NOT deleted (still on disk): the risk-**medium** tier (`stage06a/ckpt_last.pt` 184,201,483;
`stage06b/ckpt_final.pt` 51,050,949; `stage06c/ckpt_final.pt` 145,968,037 = 381,220,469 B total), all LIVE
checkpoints, all LIVE logits tags, `data/hf`, `data/ocr_textpool/raw`, `arabic_ocr/models`, and `checkpoint_backup/`.
