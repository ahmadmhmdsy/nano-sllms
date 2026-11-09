# EOS Re-baseline Verdicts - C1, C2, C3

- **Date:** 2026-10-15
- **Agent:** delegated analysis subagent (CPU only; no training; nothing under runs/ or data/ modified).
- **Status:** analysis DONE. C1 mechanism re-derived; C3 numbers re-derived; C2 and all exact 188-task totals BLOCKED on GPU.
- **Companion docs:** research/2026-10-15_eos_rebaseline.md (corpus-by-corpus EOS inventory); research/2026-10-14_eos_example_sft.md (E-122g recipe); MEMORY.md lines 1815-1835 and 1879-2030 (E-122g/E-122h/E-122i); research/EXPERIMENTS.md rows E-122, E-122b, E-122c, E-122d.
- **Ready-to-run GPU list:** research/2026-10-15_rebaseline_gpu_commands.ps1 (written, NOT executed).

> Established fact, not re-derived here: ByteTokenizer maps text bytes to 4..259 and reserves pad=0 / bos=1 / eos=2 / sep=3 (glm-5.3-flash-from-scratch/glm53_flash/tokenizer.py). Because eos_id (2) sits below byte_offset (4), any packed corpus without an explicit EOS contains only ids >= 4. The model then never sees its stop token, and decode() (which stops only at eos_id) runs every completion to the generation budget.

---

## 0. Method

Numbers below were re-derived by parsing the arm rows / by_construct objects in the raw JSON, not quoted from memory. Per-task strict/extracted in these files are objects {status, passed, tests_passed, tests_total, pass_fraction}, so the aggregate fields and .passed were used. Raw completion lengths were computed from the "completion" field of each row (the "kept" field is a post-trimmed prefix and is NOT the raw generation).

Files read for this verdict: research/raw/e122h_eos_control.json, e122g_eos_ex_vs_base.json, e122e_ab_256.json, e122d_rung_cuda.json, e122d_aspectpool_cuda.json, e122d_legacy_cuda.json, e122f_eos_vs_base.json, e122e_smoke_cpu.json; scratch/e122h_bench.log, e122e_ab_256.log, e122f_eos_vs_base.log, e122h_eos_train.log, scratche122h_noeos_train.log, scratch/bytes_pg_replay_eos.log; glm-5.3-flash-from-scratch/runs/e122*/training-receipt.json; data/reference/*/meta.json; MEMORY.md, research/EXPERIMENTS.md, TASKS.md, research/2026-10-15_eos_rebaseline.md, research/2026-10-14_eos_example_sft.md, research/2026-10-15_special_tokens.md.

The run dirs are under glm-5.3-flash-from-scratch/runs/, not runs/.

---

## 1. Confounded-run -> corpus-EOS-class map

This map does not exist in research/2026-10-15_eos_rebaseline.md (that doc has the corpus inventory in its section 2 and per-claim corpus classes in section 1, but no single run-level table). It is reproduced here.

### 1a. Packed training corpora, EOS class (from each meta.json)

| Corpus (data/reference/) | Unit | EOS flag in meta | EOS class | IDs / examples |
|---|---|---|---|---|
| bytes | reference-solutions | (no append_eos key) | **NO-EOS** | 1,086,727 / 8,000 |
| bytes_playground | reference-solutions | (no append_eos key) | **NO-EOS** | 3,406,234 / 18,800 |
| bytes_playground_noeos | reference-solutions | append_eos=false, eos_id=null | **NO-EOS** | 8,181,784 / 43,200 |
| bytes_playground_eos | reference-solutions | append_eos=true, eos_id=2 | **EXAMPLE-LEVEL EOS** | 8,224,984 / 43,200 |
| bytes_pg_replay | reference-solutions | (no append_eos key) | **NO-EOS** | 1,666,557 / 10,700 |
| bytes_pg_replay_eos | reference-solutions | append_eos=true, eos_id=2 | **EXAMPLE-LEVEL EOS** | 2,876,493 / 16,800 |
| bytes_pg_replay_fim | reference-solutions + FIM sentinels | append_eos=true, eos_id=2 | **EXAMPLE-LEVEL EOS + FIM** | 3,231,813 / (reported in e122i) |
| corpus_codeparrot_v1/pack | whole-file raw Python | append_eos=true, eos_id=2 | **FILE-LEVEL EOS** (~1 per 8.9 KB) | 350,641,560 / 39,345 docs |

Lineage note (important for single-variable claims): bytes_playground_noeos and bytes_playground_eos have identical source lists (108) and identical example counts (43,200) and differ only by the append_eos flag, so that PAIR is a clean single-variable EOS control. The same is NOT true across the older runs: bytes_playground (47 sources, 18,800 ex) and bytes_pg_replay (47 sources, 10,700 ex) predate the playground expansion to 88 rungs, while bytes_playground_eos / bytes_pg_replay_eos are 108-source supersets. No on-disk EOS corpus reproduces the ORIGINAL 47-source corpus exactly.

### 1b. Confounded runs

| Run (checkpoint) | Training corpus | EOS class | Steps / lr / seed | Init | Feeds | Recorded score |
|---|---|---|---|---|---|---|
| e122_ref600/checkpoint-0330 | bytes | NO-EOS | 600 / 1e-4 / 42 | arm A (runs/e121_ref/checkpoint-0100) | base for C1/C2/C3 and E-122g/h | 3/71 strict (E-122h); 95/188 (C1 base) |
| e122d_ref_pg/checkpoint-0510 | bytes_playground | NO-EOS | 600 / 1e-4 / 42 | e122_ref600/checkpoint-0330 | C1, C3 | 41/188 |
| e122d_ref_pg/checkpoint-0600 | bytes_playground | NO-EOS | 600 / 1e-4 / 42 | e122_ref600/checkpoint-0330 | C1, C3 | 42/188 |
| e122e_replay_lr1e4/checkpoint-0600 | bytes_pg_replay | NO-EOS | 600 / 1e-4 / 42 | e122_ref600/checkpoint-0330 | C2 | 1/71 strict, 5/71 extracted |
| e122e_replay_lr3e5/checkpoint-0600 | bytes_pg_replay | NO-EOS | 600 / 3e-5 / 42 | e122_ref600/checkpoint-0330 | C2 | 4/71 strict, 7/71 extracted |
| e122f_eos_cc/checkpoint-0600 | corpus_codeparrot_v1/pack | FILE-LEVEL EOS | 600 / 3e-5 / 42 | e122_ref600/checkpoint-0330 | failed EOS attempt | 0/71 (collapsed) |
| e122g_eos_ex/checkpoint-0800 | bytes_pg_replay_eos | EXAMPLE-LEVEL EOS | 800 / 3e-5 / 42 | e122_ref600/checkpoint-0330 | EOS breakthrough control | 59/71 |
| e122h_noeos/checkpoint-0800 | bytes_playground_noeos | NO-EOS | 800 / 3e-5 / 42 | e122_ref600/checkpoint-0330 | E-122h control | 0/71 |
| e122h_eos/checkpoint-0800 | bytes_playground_eos | EXAMPLE-LEVEL EOS | 800 / 3e-5 / 42 | e122_ref600/checkpoint-0330 | E-122h treatment | 64/71 |
| e122i_fim/checkpoint-0400 | bytes_pg_replay_fim | EOS + FIM | 400 / 3e-5 / 42 | e122h_eos/checkpoint-0800 | E-122i | 60/71 |
| e122i_nofim/checkpoint-0400 | bytes_pg_replay_eos | EXAMPLE-LEVEL EOS | 400 / 3e-5 / 42 | e122h_eos/checkpoint-0800 | E-122i control | 68/71 |

Steps/lr/seed above are from the training-receipt.json scalars. The init lines are from the E-122g recipe doc, the E-122h start events in scratch/e122h_eos_train.log and scratche122h_noeos_train.log (both "initialized_from": "glm-5.3-flash-from-scratch\runs\e122_ref600\checkpoint-0330"), and EXPERIMENTS.md row E-122d ("Same init (E-122 best)") plus E-122b ("E-122 best (330)" = e122_ref600/checkpoint-0330).

Reproducibility gap worth fixing: none of e122d_rung_cuda.json / e122d_aspectpool_cuda.json / e122d_legacy_cuda.json stores the arm paths, so the base = checkpoint-0330 mapping is derived from EXPERIMENTS.md + MEMORY.md, not from the JSON itself.

---

## 2. Claim C1 - Phase 1.3 "catastrophic forgetting" (95/188 -> 41/188 -> 42/188)

**Status: ALREADY-ANSWERED (mechanism, via E-122h ANALOGOUS control). Exact 188-task totals: NEEDS-GPU (BLOCKED on disk).**

### What was measured

Three greedy dev batteries, summed to 188 tasks. Re-derived from the raw JSON:

| Battery | Evidence file | Instrument | base | pg510 | pg600 |
|---|---|---|---|---|---|
| rung | research/raw/e122d_rung_cuda.json | bench_playground_batched-style, split dev, tier all, per-rung 4, budget_bytes 160, max_tokens 64 | 31/108 | 4/108 | 4/108 |
| aspect pool | research/raw/e122d_aspectpool_cuda.json | bench_aspect_pool.py, split dev, tier all, per-aspect 4, budget_bytes 48 | 32/48 | 23/48 | 24/48 |
| legacy | research/raw/e122d_legacy_cuda.json | bench_aspects.py-style, split dev, per-family 4, budget_bytes 48 | 32/32 | 14/32 | 14/32 |
| **TOTAL** | | | **95/188** | **41/188** | **42/188** |

Checkpoints: base = e122_ref600/checkpoint-0330; pg510 = e122d_ref_pg/checkpoint-0510; pg600 = e122d_ref_pg/checkpoint-0600. Training corpus for pg510/pg600 = data/reference/bytes_playground. **EOS present: NO** (no append_eos key in the meta; byte_offset 4). The base was trained on data/reference/bytes, also NO-EOS, so no arm in this comparison could terminate.

The 188 arithmetic is confirmed exactly: 31+32+32 = 95; 4+23+14 = 41; 4+24+14 = 42. Nothing was re-run.

### What is on disk to re-baseline it

The exact Phase 1.3 arm checkpoint (e122d_ref_pg) is on disk but was trained on a NO-EOS corpus, so re-scoring it cannot recover termination - the model has no eos_id to emit. There is no EOS-trained arm evaluated on the 188-task combination anywhere on disk. Therefore the exact corrected totals cannot be produced from existing data.

### Is E-122h the same experiment? NO - it is merely ANALOGOUS.

The task hypothesis was that the E-122h noeos arm is the Phase 1.3 experiment redone. Verified point by point:

| Dimension | E-122d (C1) | E-122h | Same? |
|---|---|---|---|
| Init | e122_ref600/checkpoint-0330 | e122_ref600/checkpoint-0330 | **YES** |
| Single variable | corpus change vs E-122 | EOS on/off | related but not the same variable |
| Corpus | data/reference/bytes_playground, 47 sources, 18,800 examples, 3,406,234 ids, NO-EOS | bytes_playground_noeos vs bytes_playground_eos, 108 sources, 43,200 examples, 8.18M / 8.22M ids | **NO** |
| Steps / lr | 600 / 1e-4 | 800 / 3e-5 | **NO** |
| Task set | 188 tasks (rung 108 + aspect 48 + legacy 32) | 71 tasks (split dev, tier core, pool playground, per-rung 1) | **NO** |
| Instrument | 3 scripts, budgets 160/48/48, max_tokens 64 (rung) | bench_playground_batched.py, budget 160, max_tokens 256 | **NO** |

Conclusion: **analogous, not the same experiment.** E-122h is a clean single-variable EOS control (its two arms differ only by append_eos) at the same init, but it changes the corpus size, the step/lr schedule and the evaluation battery at once relative to Phase 1.3. It therefore answers the MECHANISM question, not the magnitude question.

### Corrected numbers that ARE available on disk (re-derived)

From research/raw/e122h_eos_control.json (raw "completion" lengths, budget 160):

| Arm | Corpus EOS | strict | extracted | completion min/median/max | at budget (>=160 B) |
|---|---|---|---|---|---|
| base (e122_ref600/checkpoint-0330) | NO-EOS | 3/71 | 11/71 | 160 / 160 / 160 | 71/71 |
| noeos (e122h_noeos/checkpoint-0800) | NO-EOS | 0/71 | 0/71 | 160 / 160 / 160 | 71/71 |
| eos (e122h_eos/checkpoint-0800) | EXAMPLE-LEVEL EOS | 64/71 | 64/71 | 21 / 56 / 139 | 0/71 |

Supporting, from research/raw/e122g_eos_ex_vs_base.json: base 3/71 (71/71 at budget, all 160 B) vs eos_ex e122g_eos_ex/checkpoint-0800 59/71 (0/71 at budget, min 21 / median 57 / max 141).

### What this changes

The direction of the Phase 1.3 verdict is contradicted at the same init: an EOS-less task-SFT arm DEGRADES the model (3/71 -> 0/71), while the EOS version of the same task-SFT IMPROVES it (3/71 -> 64/71). The old "95 -> 41" is therefore not safe to report as genuine catastrophic forgetting; it is at minimum confounded by non-termination, and the 48-byte budget on the aspect and legacy batteries adds a second confound. However, do NOT substitute 64/71 (a 71-task core battery at a different schedule) for the 188-task totals. If the 188-task number is needed, the exact re-run is in the GPU list, Section C1.

---

## 3. Claim C2 - Phase 1.2b replay A/B (base 3/11, lr1e4 1/5, lr3e5 4/7)

**Status: NEEDS-GPU.**

### What was measured, and the confound proof

From research/raw/e122e_ab_256.json (split dev, tier core, pool playground, per-rung 1, budget_bytes 160, max_tokens 256, 71 tasks). Re-derived, with raw completion lengths computed from each row:

| Arm | Checkpoint | Corpus | EOS | strict | extracted | completion min/median/max | at budget (>=160 B) |
|---|---|---|---|---|---|---|---|
| base | e122_ref600/checkpoint-0330 | bytes | NO-EOS | 3/71 | 11/71 | 160 / 160 / 160 | **71/71** |
| lr1e4 | e122e_replay_lr1e4/checkpoint-0600 | bytes_pg_replay | NO-EOS | 1/71 | 5/71 | 160 / 160 / 160 | **71/71** |
| lr3e5 | e122e_replay_lr3e5/checkpoint-0600 | bytes_pg_replay | NO-EOS | 4/71 | 7/71 | 160 / 160 / 160 | **71/71** |

The task brief states this claim as base 3/11, lr1e4 1/5, lr3e5 4/7; those are exactly the strict/extracted pairs above (3 & 11, 1 & 5, 4 & 7). The confound is now proven directly from the data: **all three arms produced completions of exactly 160 bytes for all 71 tasks** - none ever terminated. The differences among 3, 1 and 4 strict passes are therefore differences in what garbage follows a valid body, not differences in capability.

### Can it be re-baselined from disk?

**No.** Both trained arms used data/reference/bytes_pg_replay (NO-EOS). No EOS-trained lr1e4 or lr3e5 replay arm exists. The nearest EOS arms (e122g_eos_ex 59/71 and e122h_eos 64/71) used different corpora and a different schedule (800 steps, lr 3e-5), so they are controls from which a direction can be inferred, not the replay A/B re-measured. The e122e checkpoints cannot be re-scored into validity because the model has no stop token.

### What this changes

The "replay hurts" conclusion must not be used. A valid replay A/B requires both trained arms AND the control to be able to terminate. The E-122h noeos arm (0/71, an EOS-less task corpus, strictly worse than the NO-EOS base 3/71) is consistent with the old replay arms looking bad, and the E-122h eos arm (64/71) is the corrected direction, but neither is the replay A/B. The re-run is in the GPU list, Section C2.

---

## 4. Claim C3 - "the corpus bought +1 grade rung and cost 54 task-passes"

**Status: RE-MEASURED (both raw numbers re-derived and confirmed from disk). Corrected EOS verdict: NEEDS-GPU.**

### +1 grade rung - confirmed

From research/raw/e122d_rung_cuda.json, by_construct.grade (28 grade tasks per arm, the rung battery at budget 160 / max_tokens 64):

| Arm | grade strict |
|---|---|
| base (e122_ref600/checkpoint-0330) | 0/28 |
| pg510 (e122d_ref_pg/checkpoint-0510) | **1/28** |
| pg600 (e122d_ref_pg/checkpoint-0600) | **1/28** |

So the corpus did move grade off zero (0 -> 1 of 28). Corroborated by research/EXPERIMENTS.md row E-122d: "grade 0/28 -> 1/28 (the one designed target DID move)".

### cost 54 task-passes - confirmed as arithmetic

95/188 (base) - 41/188 (pg510) = **54**. For pg600 it is 95 - 42 = 53. So the "54" specifically compares base against the pg510 checkpoint. The per-battery losses are: rung -27 (31 -> 4), aspect pool -9 (32 -> 23), legacy -18 (32 -> 14); total -54.

### Confounds

Both arms were NO-EOS (see Section 1b), and the aspect and legacy batteries were additionally run at a 48-byte budget that a separate MEMORY.md entry (2026-10-15, "A completion budget can erase 87% of real capability") shows can erase many real passes. So the magnitude of the "cost" is confounded twice. The +1 grade rung is at least directionally real (it moved within a single instrument), but 1/28 is 3.6 percent and is itself not validated against an EOS-capable control.

### What this changes

Report the arithmetic as-is only if labelled confounded. The sentence "corpus bought +1 grade rung and cost 54 task-passes" must become "on NO-EOS corpora, the corpus moved grade 0/28 -> 1/28 and the 188-task total fell 95 -> 41 (pg510) / 95 -> 42 (pg600); both numbers are confounded by non-termination and, on 80 of 188 tasks, by a 48-byte budget". The corrected version requires the C1 GPU re-run and the fixed instrument (budget 160 / max-tokens 1000) for all three batteries.

---

## 5. Summary of statuses

| Claim | Recorded numbers (re-derived) | Status | Evidence |
|---|---|---|---|
| C1 Phase 1.3 forgetting | 95/188 -> 41/188 -> 42/188; base 3/71, noeos 0/71, eos 64/71 | **ALREADY-ANSWERED (mechanism, ANALOGOUS) / NEEDS-GPU (exact 188 totals)** | research/raw/e122d_rung_cuda.json + e122d_aspectpool_cuda.json + e122d_legacy_cuda.json; research/raw/e122h_eos_control.json |
| C2 replay A/B | base 3/11, lr1e4 1/5, lr3e5 4/7 (all 71/71 at 160 B) | **NEEDS-GPU** | research/raw/e122e_ab_256.json |
| C3 +1 grade / cost 54 | grade 0/28 -> 1/28; 95 - 41 = 54 (95 - 42 = 53) | **RE-MEASURED (numbers) / NEEDS-GPU (corrected verdict)** | research/raw/e122d_rung_cuda.json |

Honest labels for the whole task: evidence location PASS; corpus-EOS map PASS; C1 mechanism PASS (re-derived), exact totals BLOCKED (needs GPU); C2 BLOCKED (needs GPU); C3 numbers PASS (re-derived), corrected verdict BLOCKED (needs GPU); new measurements SKIPPED (CPU-only constraint, GPU busy); command list PASS (written, not run); runs/ and data/ unmodified PASS.

---

## Appendix A - Ready-to-run GPU command list

The exact commands are in research/2026-10-15_rebaseline_gpu_commands.ps1. They were written but NOT executed. That file contains: a Test-Path preflight with forward slashes, the C1 EOS re-run (E-122d recipe on bytes_playground_eos), the C2 replay EOS re-runs (lr 1e-4 and lr 3e-5 on bytes_pg_replay_eos), and the scoring batteries (rung / aspect / legacy for C1, core-playground for C2) at both the original instrument and the repaired instrument (budget 160, max-tokens 1000).

Caveats recorded in that file:

1. `--arm` specs require the form `name=glm:<path>` (scratch/bench_playground_batched.py splits on "=" then ":" and chooses GlmArm only when kind == "glm"). The template in research/2026-10-15_eos_rebaseline.md section 4 writes `--arm "base=<ckpt>"` without the `glm:` prefix and would fail to load the arm.
2. bytes_playground_eos and bytes_pg_replay_eos are 108-source supersets of the original 47-source corpora, so the re-run isolates EOS imperfectly. A clean pair can be rebuilt on CPU with scripts/reference_sft_bytes.py --append-eos into a new --out-dir, but it will not reproduce the original 47-source corpus (the playground pool has since grown to 88 rungs).
3. scratch/bench_aspects.py (legacy battery) hardcodes its byte budget, so the repaired instrument may need a source edit or a parameter addition; verify before scoring.
4. Single GPU: do not launch while any repo train.py / sft.py run is active (AGENTS.md section 4.6).
5. AGENTS.md section 3 asks for the laya preflight gate before unattended long launches; that gate is config-based and does not directly cover train_pretrain.py, whose flags below are taken from the E-122g recipe doc and confirmed by the e122h training receipts. Recommended, cheaper gate: a 2-step smoke plus reading the trainer --help before the full launch.

---

## Appendix B - Evidence index

| Path | What it proves |
|---|---|
| research/raw/e122h_eos_control.json | E-122h base 3/71, noeos 0/71, eos 64/71; raw lengths 71/71, 71/71, 0/71 at 160 B |
| research/raw/e122g_eos_ex_vs_base.json | base 3/71 vs eos_ex 59/71; lengths 71/71 vs 0/71 at budget |
| research/raw/e122e_ab_256.json | C2: base 3/11, lr1e4 1/5, lr3e5 4/7 strict/extracted; all three 71/71 at 160 B |
| research/raw/e122d_rung_cuda.json | C1/C3 rung: 31/108 -> 4/108 -> 4/108; grade 0/28 -> 1/28 -> 1/28 |
| research/raw/e122d_aspectpool_cuda.json | C1 aspect: 32/48 -> 23/48 -> 24/48 |
| research/raw/e122d_legacy_cuda.json | C1 legacy: 32/32 -> 14/32 -> 14/32 |
| research/raw/e122f_eos_vs_base.json | failed file-level EOS: base 3/71, eos 0/71 |
| scratch/e122h_bench.log | per-task E-122h rows backing the JSON |
| scratch/e122e_ab_256.log | summary lines [base] 3/71, [lr3e5] 4/71 |
| data/reference/*/meta.json | append_eos / eos_id / source counts / example counts |
| glm-5.3-flash-from-scratch/runs/e122*/training-receipt.json | steps, lr, batch, seed, corpus paths and digests |
| scratch/e122h_eos_train.log, scratche122h_noeos_train.log | init = e122_ref600/checkpoint-0330 for both E-122h arms |
| research/EXPERIMENTS.md rows E-122b/E-122d | E-122 best = checkpoint 0330; E-122d grade 0/28 -> 1/28; catastrophic-forgetting verdict |
| research/2026-10-15_eos_rebaseline.md | corpus inventory by EOS class; the falsification rule |

## Appendix C - Validation label summary

- PASS: corpus-EOS map built and cross-checked against meta.json and receipts.
- PASS: C1 raw numbers re-derived (95/41/42) and E-122h status adjudicated as analogous.
- PASS: C2 confound proven directly (all three arms 71/71 at 160 B).
- PASS: C3 raw numbers re-derived (grade 0/28 -> 1/28; 54 = 95 - 41).
- BLOCKED: exact corrected 188-task totals (need GPU; no EOS Phase 1.3 arm on disk).
- BLOCKED: corrected replay A/B (need GPU; no EOS replay arms on disk).
- SKIPPED: all new training/eval (CPU-only constraint; GPU busy).
- PASS: commands written to research/2026-10-15_rebaseline_gpu_commands.ps1, not executed.
- PASS: no file under runs/ or data/ modified.
