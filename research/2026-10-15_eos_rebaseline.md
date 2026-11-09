# EOS Re-baseline Investigation (E-122g fallout)

**Date:** 2026-10-15
**Agent:** delegated investigation subagent (read-only outside this file)
**Status:** DONE (investigation). No GPU job launched, no training, nothing under runs/ or data/ modified.
**Scope:** locate the confounded measurements, classify every packed corpus by EOS class, name the
largest task-format EOS-capable corpus, and give an ordered re-baseline plan.

> Established fact (not re-derived here): ByteTokenizer maps text bytes to 4..259 and reserves
> pad=0 / bos=1 / eos=2 / sep=3 (`glm-5.3-flash-from-scratch/glm53_flash/tokenizer.py:8`).
> Because eos_id (2) < byte_offset (4), every corpus packed without an explicit EOS contains only
> ids >= 4. The model had never seen its own stop token, so `decode()` (stops only at eos_id) ran
> every completion to the generation budget. Example-level EOS fixed this: base
> `runs/e122_ref600/checkpoint-0330` = strict 3/71, `runs/e122g_eos_ex/checkpoint-0800` = strict
> 59/71, base completions all exactly 160 B, eos_ex median 57 B / 0-of-71 at budget.

---

## 1. Exact commands / artifacts / reports behind the affected numbers

Bench harness for everything below: `scratch/bench_playground_batched.py` (greedy, prompt regime;
`strict` = whole completion is one valid function; `extracted` = longest valid line-prefix, so
non-termination is not scored as wrong Python). Defaults: `--budget-bytes 48`, `--max-tokens 1000`
(raised from 64 in E-122e; 64 silently truncated every completion). The measurements below used
`--budget-bytes 160 --max-tokens 256 --batch-size 8 --tier core --pool playground --split dev
--per-rung 1`. Per-task rows are at `arms.<name>.rows`; `strict`/`extracted` are objects
`{status,passed,...}`, not booleans.

### A. The E-122g breakthrough (base vs example-level EOS)
- **Base:** `runs/e122_ref600/checkpoint-0330` — corpus `data/reference/bytes` (**NO-EOS**),
  600 steps, E-122. Score strict **3/71**, extracted 11/71; all 71 completions exactly 160 B.
- **eos_ex:** `runs/e122g_eos_ex/checkpoint-0800` — corpus `data/reference/bytes_pg_replay_eos`
  (**EXAMPLE-LEVEL EOS**), 800 steps, lr 3e-5 cosine, warmup 10, batch 4 x accum 8, seq 1024,
  seed 42, vram cap 5.0, init = e122_ref600/checkpoint-0330. Score strict **59/71**, extracted
  59/71; min 21 / median 57 / max 141 B, 0-of-71 at budget.
- **Train command (verbatim):** `research/2026-10-14_eos_example_sft.md` §"Exact command":
  `train_pretrain.py --corpus data\reference\bytes_pg_replay_eos\train.bin --val-corpus ...\val.bin
  --init-from glm-5.3-flash-from-scratch\runs\e122_ref600\checkpoint-0330 --output ...\runs\e122g_eos_ex
  --steps 800 --batch-size 4 --grad-accum 8 --sequence-length 1024 --learning-rate 3e-5
  --lr-schedule cosine --warmup-steps 10 --val-windows 32 --checkpoint-fraction 0.05
  --vram-cap-gib 5.0 --amp-dtype fp16 --device cuda --seed 42`.
  Receipt `runs/e122g_eos_ex/training-receipt.json`: corpus digest `462523ed...`, total_ids
  2,876,493, windows 2809, steps_completed 834 (2 resume events), peak 3.03 GiB, no spill.
- **Report:** `research/raw/e122g_eos_ex_vs_base.json` (218,433 B; `arms.base` + the eos arm;
  rows at `arms.<name>.rows`). Battery log: `scratch/e122f_eos_vs_base.log`
  (arm "eos" there is the failed E-122f, see below).
- Narration: `MEMORY.md` 2026-10-15 "BREAKTHROUGH" entry and 2026-10-14 "NEVER TAUGHT TO STOP"
  entry. Command receipt reportedly in session `7e105ad5` turn 105.

### B. E-122f — the FAILED file-level EOS attempt
- `runs/e122f_eos_cc` = 600 steps on `data/reference/corpus_codeparrot_v1` (**FILE-LEVEL EOS**;
  one EOS per ~8.9 KB whole source file), init = e122_ref600/checkpoint-0330.
  Result extracted **11/71 -> 0/71**, degenerate repetition. Mechanism: EOS was an alien token
  (step-1 loss 2.6719 vs 1.2257 for the on-distribution replay-EOS corpus, identical init) and
  raw-Python imitation does not teach task-following.
- Artifacts: `scratch/e122f_eos_cc.log`, `scratch/e122f_eos_vs_base.log`,
  `research/raw/e122f_eos_vs_base.json`; doc `research/2026-10-14_eos_example_sft.md` §"Prior
  failed attempt".

### C. Phase 1.2b replay A/B (the confounded comparison)
- Arms: base `runs/e122_ref600/checkpoint-0330`; `runs/e122e_replay_lr1e4`;
  `runs/e122e_replay_lr3e5`. Training corpus for both trained arms:
  `data/reference/bytes_pg_replay` (**NO-EOS**). Recorded scores at max-tokens 256:
  base strict 3/71 extracted 11/71; lr1e4 1/71 / 5/71; lr3e5 4/71 / 7/71.
- Artifacts: `scratch/e122e_ab_256.log` (summary lines `[base] strict 3/71 extracted 11/71`,
  `[lr3e5] strict 4/71 extracted 7/71`), `research/raw/e122e_ab_256.json`,
  `scratch/e122e_smoke_cpu.log` + `research/raw/e122e_smoke_cpu.json`. Each run dir has a
  `training-receipt.json`. Verdict "replay hurts" is **suspect**: both trained arms and the base
  ran on NO-EOS data, so all three could only ever hit the budget.

### D. Phase 1.3 corpus-SFT "catastrophic forgetting" (95/188 -> 41/188 -> 42/188)
- The corpus-SFT arm is `runs/e122d_ref_pg` = 600 steps on `data/reference/bytes_playground`
  (**NO-EOS**), init = E-122 best, lr 1e-4 cosine (TASKS 171); retained `checkpoint-0510` (best)
  + 0570/0600. The 188-task total is the Phase 1.3 rung + aspect + legacy battery combination.
- Direct evidence of the artifact: `MEMORY.md` 2026-10-14 — **all 216 completions in the Phase
  1.3 run were exactly 64 bytes = the generation cap**, never fewer. So the "forgetting" totals
  were measured on a model that could not terminate; the budget, not the model, was ending output.
- Artifacts: TASKS.md rows 171-174; `research/EXPERIMENTS.md` rows E-122/E-122b/E-122d;
  `runs/e122d_ref_pg/training-receipt.json`, `progress.jsonl`.
- The verdict "corpus bought +1 grade rung and cost 54 task-passes" is the same E-122d/Phase 1.3
  comparison and inherits the same confound (recorded in `MEMORY.md` 2026-10-15 as one of the
  three verdicts requiring re-baseline).

---

## 2. Packed-corpus inventory by EOS class (the primary output)

Method: read each `meta.json`; `append_eos`/`eos_id` are the packer flags; `byte_offset == 4` +
`shape == "reference-solutions"` identifies ByteTokenizer example-level corpora. "tokens" below =
uint16 id count = `.bin bytes / 2`.

| Corpus | Shape / unit | append_eos | EOS class | Train ids (examples) | Built by |
|---|---|---|---|---|---|
| `data/reference/bytes` | reference-solutions / function | absent | **NO-EOS** | 1,086,727 (8,000; 20 sources) | `scripts/reference_sft_bytes.py` |
| `data/reference/bytes_pg_replay` | reference-solutions / function | absent | **NO-EOS** | 1,666,557 (10,700; 47 sources) | `reference_sft_bytes.py --include-playground` |
| `data/reference/bytes_playground` | reference-solutions / function | absent | **NO-EOS** | 3,406,234 (18,800; 47 sources) | `reference_sft_bytes.py --include-playground --playground-tier all` |
| `data/reference/bytes_pg_replay_eos` | reference-solutions / function | **true, id 2** | **EXAMPLE-LEVEL EOS** | 2,876,493 (16,800; 108 sources) | `reference_sft_bytes.py --append-eos` (log `scratch/bytes_pg_replay_eos.log`) |
| `data/reference/corpus_codeparrot_v1/pack` | whole-file / raw Python file | **true, id 2** | **FILE-LEVEL EOS** (1 per 39,345 docs, ~8.9 KB each) | 350,641,560 (39,345 docs + 500 val) | `scripts/corpus_pipeline.py --stage all ... --append-eos --val-docs 500` |
| `data/wholefile/bytes` | whole-file / codeparrot-clean | absent | **NO-EOS** | 163,301,250 "bytes" (= ids) | wholefile packer |
| `data/target/bytes` | byte arm, tokenizer `runs\target\final` | absent | **NO-EOS** | 47,937,963 tokens | E-117 target-corpus byte arm |
| `data/sft/*`, `data/custom_example` | CodeLlama BPE instruction pairs | n/a (`eos_token_id=2` in template) | **EOS-present but NOT the byte lane** | — | `sft_data.py` |

Notes:
- `corpus_codeparrot_v1` was originally built without EOS (TASKS 178); the pack was later re-run
  with `--append-eos --val-docs 500` (TASKS 179: `train.bin 690,409,630 B + val.bin 10,873,490 B =
  350,641,560 ids = 350,602,215 text bytes + 39,345 EOS`). That EOS is **per whole file**, so it is
  file-level, not example-level.
- `data/reference/bytes` also has `append_eos` simply missing (written before the flag landed).
- **There is exactly one example-level EOS corpus in the repo: `bytes_pg_replay_eos`.**

---

## 3. Largest task-format corpus for scaling the recipe

- **Largest example-level (function-sized) corpus overall: `data/reference/bytes_playground` —
  3,406,234 uint16 ids / 18,800 examples / 47 sources. It has NO EOS**, so it cannot be used as-is
  for termination training.
- **Largest example-level corpus that already has (example-level) EOS: `bytes_pg_replay_eos` —
  2,876,493 ids / 16,800 examples / 108 sources** (88 playground rungs + legacy families +
  aspects, 400 instances/source, held-out `pretrain` split, seed 9319; built by
  `reference_sft_bytes.py --append-eos`, per `scratch/bytes_pg_replay_eos.log`). This is the
  corpus the 59/71 recipe used.
- **Largest packed corpus of any kind: `corpus_codeparrot_v1` (350.6 M ids)** — but it is
  whole-file raw Python with **file-level** EOS; E-122f already showed this unit fails (0/71 +
  degenerate repetition). Do not scale the recipe on it.
- **Recommended scaling move (CPU, no GPU):** re-pack `bytes_playground` with
  `--append-eos` into a NEW out-dir (the builder hard-refuses to write into
  `data/reference/bytes`), e.g.
  `reference_sft_bytes.py --include-playground --playground-tier all --append-eos --instances 400
  --out-dir data/reference/bytes_playground_eos`
  -> ~3,406,234 + 18,800 = ~3.43 M ids (18,800 EOS). Raising `--instances` (e.g. 800) scales it
  further. This is the first genuinely larger on-distribution, example-level EOS corpus available.

---

## 4. Ordered re-baseline plan

**Exact bench command for every arm below** (from parent root; `--device cuda` unless GPU is
busy, cpu gives identical deterministic greedy rows per WHAT_WORKS):
```powershell
& .\.venv\Scripts\python.exe scratch\bench_playground_batched.py `
  --arm "base=<ckpt>" --arm "<arm>=<ckpt>" `
  --split dev --tier core --pool playground --per-rung 1 `
  --budget-bytes 160 --max-tokens 256 --batch-size 8 --device cuda `
  --out research\raw\<name>.json
```
Report per arm: strict, extracted, and the completion-length distribution (min/median/max,
fraction at budget) — length is the direct termination evidence.

Falsification rule (applies to every step): the claim under test is **"the no-termination artifact,
not genuine forgetting, produced the old verdict."** It is falsified if, **with EOS present in
every arm's training corpus**, the trained arm still fails to beat its EOS-trained control (or
still loses ~54 task-passes / still lands near 41/188). A valid comparison requires the control arm
to be able to terminate too; comparing a NO-EOS base (100% at budget) against an EOS-trained arm is
exactly the invalid comparison that produced the old numbers.

- **Step 0 (CPU, no GPU).** Rebuild `bytes_playground_eos` (§3) and, if needed, a larger
  `bytes_pg_replay_eos` (`--instances 800`). Verify each `meta.json` has
  `append_eos: true, eos_id: 2` and the EOS count equals the example count.
- **Step 1 (missing control — highest priority).** The EOS-trained `e122g_eos_ex` bundles
  (a) learning to emit EOS with (b) 1.5 epochs of task-distribution rehearsal. Add a
  **termination-only control**: train briefly (e.g. 100-200 steps) at the SAME init on
  `bytes_pg_replay_eos` with a small lr, and score it. Then:
  - `e122g_eos_ex` >> termination-only control on strict => the capability jump came from the
    readmitted task distribution, not EOS per se.
  - termination-only control already ~59/71 => EOS *is* the whole effect; re-baseline every prior
    arm against it.
  Also score a re-trained EOS base (e.g. run base `bytes` corpus with `--append-eos`), because
  `e122_ref600/checkpoint-0330` itself can never terminate and is therefore not a valid control.
- **Step 2.** Re-run the Phase 1.2b replay A/B on `bytes_pg_replay_eos`: init
  `e122_ref600/checkpoint-0330`, 600 steps, lr **1e-4** and **3e-5**, same batch/accum/seq/schedule
  as E-122e. Compare against the **EOS-trained control** from Step 1, not against the NO-EOS base.
  Falsified if lr1e4/lr3e5 still tie or worsen the control; explained if they now improve it
  (the old 1/71 and 4/71 read the cap, not the model).
- **Step 3.** Re-run Phase 1.3 corpus-SFT forgetting on `bytes_playground_eos`: init
  `e122_ref600/checkpoint-0330`, 600 steps, lr 1e-4 (E-122d recipe). Re-run all three Phase 1.3
  batteries (rung + aspect + legacy) at `--max-tokens 1000 --budget-bytes 160`, and record the
  length distribution. Compare the SFT arm against the Step 1 EOS control on the same totals.
  Falsified if the SFT arm still loses ~54 task-passes (or still ~41/188) when both arms terminate.
- **Step 4.** Recompute the grade-rung delta ("corpus bought +1 grade rung") directly from the
  Step 3 EOS re-baseline: per-construct grade rungs solved, SFT arm vs EOS control.

**Hard sequencing constraints:** Steps 2-4 need GPU and are the Lead's call — this investigation
launched nothing. Each trained arm needs its own 800-step-lr-3e-5 ex `e122g`-style run
(~70 min each on the single GPU) plus the batteries; budget accordingly. Do not overwrite any
existing corpus (write `*_eos` out-dirs) and do not retrain on `corpus_codeparrot_v1`
(file-level EOS is the known-failed unit).

---

## 5. Deliverable status
- Document: **PASS** (this file).
- Corpus classification: **PASS** (every packed byte corpus under `data/` classified from its
  `meta.json`; the EOS class of each affected run's training corpus recorded).
- Re-baseline plan: **PASS** (ordered, with exact bench command and a falsification rule).
- GPU work: **SKIPPED** (hard constraint — read-only outside this doc; Lead owns all GPU work).
- Source files, `runs/`, `data/`: **unmodified**.
