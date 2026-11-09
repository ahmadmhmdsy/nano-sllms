# RESUME HANDOFF — KT-2 round 2 → SFT r2 → KT-3 (fresh-context, zero conversation)

Written 2026-09-10 evening, mid-round-2. This file is SELF-CONTAINED for an
agent resuming with fresh context: state, todos in execution order, exact
commands, gates, gotchas. The ladder spec is
docs/plans/2026-09-10_kt_ladder_plan.md (read it for WHY); this file owns WHAT NEXT.

## 0. Read order (before touching anything)

1. CLAUDE.md (operating system) → AGENTS.md (repo map)
2. PLAN.md → HANDOFF.md (top blockquote = latest state) → TASKS.md rows 43-48
   → MEMORY.md (esp. lessons 1, 22, 25b, 26, 29, 31, 32, 33, 36, 37)
   → ENVIRONMENT.md
3. This file top-to-bottom.

## 1. State snapshot (as written)

- **RUNNING NOW**: harness background job "pwsh-8" = KT-2 judge ROUND-2 chain
  (sample → score → select; configs/kt2_judge_r2.yaml) writing
  runs/kt2_judge_r2/ and data/sft/kt2_judge_r2/pairs.jsonl.
  Launched ~1 h before this file was written; ETA ~5-5.5 h total.
  Check it: harness job_output("pwsh-8") — NOTE the job_list tool itself may
  fail with "binding arguments must be lossless JSON" (known DSH bug, seen
  twice today); use job_output with the known id, or just look for
  runs/kt2_judge_r2/score_report.json + select_report.json + pairs.jsonl.
- **DONE and user-accepted** (numbers you can trust):
  - KT-1 transplant A/B (row 43, done): transplant init BEAT control by
    -6.24% eval @1000 (2.2902 vs 2.4426; gate >=3-5% MET; gap still growing).
    USER DECISION (Q2): KEEP OUR ARCHITECTURE; the transplant embedding init
    is the standard warm-start lever. Route 0 test queued as row 47.
  - KT-2 judge round 1 + SFT (row 44, done): 250 prompts x 4 -> 248 winners
    -> LoRA-SFT from runs/yarn_lora_sft_v1/final_a60 (30 steps) -> runs/kt2_sft/final.
    Gates PASS: CSN guard 1.8987 vs a60 1.8917 (+0.37%, gate <=5%); ast
    greedy 0.90 (= base), sampled 0.92 (base 0.96, -4 pts = 50-sample noise).
- **Baselines for gates**: a60 CSN @1024 = 1.8917 (runs/yarn_lora_sft_v1/gates/soup_a60_1024.json);
  kt2_sft r1 = 1.8987 (runs/kt2_sft/gates/gate_1024.json); ast greedy 0.90.
- **Git**: main = e08ec9a; ALL session work is UNCOMMITTED (user-gated):
  scripts/{embed_transplant,kt2_judge}.py, scripts/train.py (init_embeddings hook),
  configs/kt2_*.yaml + kt_ab_*.yaml, docs/plans/2026-09-10_kt_ladder_*.md,
  TASKS/HANDOFF/MEMORY edits, runs/kt_* + runs/kt2_* artifacts.
  **data/kt/ is NOT gitignored** - the 134 MB embed_init_smol360.pt is a
  weights-like file; if the user approves a commit, gitignore data/kt/*.pt
  first (weights stay local per repo policy).
- **Disk**: ~22 GB free. **GPU**: RTX 3000 6 GB, fp16-only (sm_75), idle at
  last check; standby AC timer already 0 (no guard needed).

## 2. Hard operating rules (details in CLAUDE/AGENTS/MEMORY)

- venv only: & .\.venv\Scripts\python.exe ... — never bare python/pip.
- SINGLE GPU: idle gate (nvidia-smi memory.used < ~600 MiB) before EVERY GPU
  command; never co-run GPU jobs; never kill a run for pace.
- Auto-resume (train.py/sft.py only): after any crash re-run the EXACT
  command with zero flags. NOT applicable to kt2_judge stages (append-files:
  to re-run a stage, first wipe that stage's outputs in the run dir).
- Batch conventions (MEMORY 37): SFT batch 4/accum 4 (effective 16); batched
  inference patterns are in scripts/kt2_judge.py (packed LL, left-padded
  verdict gen, num_return_sequences).
- PYTHONIOENCODING=utf-8 in every pwsh call that prints python output
  (cp1252 UnicodeEncodeError otherwise).
- Weights stay local; no deletions/pushes/commits without user approval;
  report before deleting anything.
- Honest labels PASS/FAIL/SKIPPED/BLOCKED; never claim untested success.

## 3. TODO — execute in order

### STEP 0 — assess round-2 state
- [ ] Is pwsh-8 still running? (job_output "pwsh-8"; wait:true if you need it)
- [ ] If finished: read runs/kt2_judge_r2/score_report.json — GATE
      unparseable_rate < 0.10 (r1 was 0.003); select_report.json — winners
      expected ~1400-1480 of 1500 prompts (drops: no_ast mostly); pairs.jsonl
      line count.
- [ ] If the chain failed mid-stage: read the job tail, fix the script, wipe
      THAT stage's outputs in runs/kt2_judge_r2 (candidates.jsonl for sample;
      verdicts.jsonl + score_report for score; select_report + pairs for
      select) and re-run that one stage:
      & .\.venv\Scripts\python.exe scripts\kt2_judge.py --stage <stage> --config configs\kt2_judge_r2.yaml

### STEP 1 — combined SFT corpus (CPU)
- [ ] Concatenate round-1 + round-2 winner pairs into
      data/sft/kt2_sft_r2/pairs.jsonl (raw concat of
      data/sft/kt2_judge/pairs.jsonl [248] + data/sft/kt2_judge_r2/pairs.jsonl;
      expected ~1728-1728 lines; cross-round dupes should be 0 by the pool
      exclusion - verify with a sha1-count if you want certainty).
- [ ] Count lines exactly -> that number is "rows" in STEP 2 (MEMORY 22:
      sft_data derives n_val from the rows TARGET, so rows must ~= corpus size).

### STEP 2 — configs/kt2_sft_r2.yaml (NEW file; copy configs/kt2_sft.yaml, change)
- base_model: runs/yarn_lora_sft_v1/final_a60
  (FRESH from a60, NOT chained onto runs/kt2_sft/final — the combined corpus
  already contains round-1's pairs; one clean LoRA pass over all data, no
  double-counting, single guard baseline. This is the agreed plan-A design.)
- data: dataset=data/sft/kt2_sft_r2/pairs.jsonl; raw_dir=data/sft/kt2_sft_r2;
  dataset_dir=data/sft/kt2_sft_r2/ds; rows=<counted in STEP 1>;
  val_fraction=0.06; min_chars=30; min_instruction_chars=10; dedupe=true;
  ast_filter=true; template unchanged.
- train: epochs=2; batch=4; accum=4 (effective 16 — unchanged math);
  lr=1.0e-4 cosine; warmup_steps=20 (~200 steps expected); logging_steps=10;
  eval_steps=50; save_steps=50 (save must be a multiple of eval);
  save_total_limit=3; fp16=true; grad_ckpt=true; optim=adamw_torch; seed=42.
- name/output/final: kt2_sft_r2 / runs/kt2_sft_r2 / runs/kt2_sft_r2/final.
- model block mirrors a60 dims (16L/1024/16Q/4KV/3072FFN, tie) for sanity_check.

### STEP 3 — validate + train
- [ ] & .\.venv\Scripts\python.exe scripts\sft_data.py --config configs\kt2_sft_r2.yaml
      (CPU; expect kept ~= corpus, split ~ (1-0.06*N) train / ~104 val)
- [ ] GPU idle gate, then background job:
      & .\.venv\Scripts\python.exe scripts\sft.py --config configs\kt2_sft_r2.yaml
      (~200 steps; batched pace ~2-4 min per 50 steps + load/save overhead)
- [ ] Collect runs/kt2_sft_r2/final/train_summary.json (best_eval_loss is on
      the small judge-val — NOT comparable to CSN; the CSN gate is STEP 4).

### STEP 4 — gates (same instruments as r1)
- [ ] & .\.venv\Scripts\python.exe scripts\eval.py --config configs\kt2_sft_r2.yaml
      --ckpt runs\kt2_sft_r2\final --ctx 1024 --label kt2r2_guard_1024
      --report-out runs\kt2_sft_r2\gates\gate_1024.json
- [ ] GATES: CSN val_loss <= 1.9863 (a60 1.8917 + 5%); ast greedy >= 0.90;
      sampled >= 0.92 (r1 level; r1 base was 0.96 — flag if below 0.92).
- [ ] PASS -> TASKS row 48 -> done with numbers; HANDOFF top line; MEMORY if
      a new lesson emerged. FAIL -> read failed_examples in the report; do
      NOT iterate SFT variants without the user (repo rule).

### STEP 5 — report to user (always user-gated after this)
- [ ] Round-2 report: winners, unparseable rate, agreement rate, combined
      corpus size, guard delta, ast rates.
- [ ] OPTION B (KT-3, row 45) needs the user's explicit GO for the multi-day
      window — do NOT self-launch the pretrain.
- [ ] Row 47 (Route 0 warm-start SmolLM2-135M test) queued, user-gated timing.
- [ ] Row 46 (KT-4 ULD spike) parked.

### STEP 6 — KT-3 prep (CPU/network, co-run-safe with any GPU job)
- [ ] Read configs/next_pretrain.yaml + research/milestone_d_measurement.md.
      The existing mix is sized for PILOT dims (133.65M tokens) — re-size for
      226.5M dims and extend candidates (fineweb-edu / cosmopedia-class).
      MEASURE tok/row before sizing (MEMORY 29: whole-file sources measured
      ~4x their estimates).
- [ ] Write configs/kt3_pretrain.yaml:
      model = 226.5M dims (mirror configs/kt_ab_control.yaml model block,
      ctx 1024); data.tokens_dir = NEW prepared tokens dir;
      train.init_embeddings = data/kt/embed_init_smol360.pt (the PROVEN
      -6.24% warm start; hook already in scripts/train.py, default OFF);
      batch=4, accum=8 (= 32,768 tok/step, same budget class as target);
      optim=adamw_bnb_8bit (MEMORY decision: default for the NEXT pretrain);
      lr=4.0e-4 cosine, warmup_steps=150, seed=42; eval/save 100, limit 3.
- [ ] Data prep (CPU/network; co-run-safe): prepare_data.py + tokenize_data.py
      with the new config. Gotchas: bigcode repos are gated (terms accepted
      2026-09-09, HF_TOKEN works) and need the data_dir load form (MEMORY
      27/28); TinyCode-style corrupt shards kill full downloads - stream
      with row caps below the bad shard (MEMORY 17).
- [ ] LAUNCH = USER-GATED (multi-day window). Before launch: disk >5 GB rule
      (MEMORY 15), standby AC 0 (already), GPU idle, then the exact command
      zero-flags for auto-resume.

## 4. Gotchas that bit THIS session (fresh-agent traps)

- DSH harness: probe run_code->pwsh with a trivial call first — the binding
  bug ("missing required property description" / "binding arguments must be
  lossless JSON") resurfaced twice today including on job_list with NO args;
  retry usually works, direct file tools (read/write/edit/glob/grep) always
  work.
- edit tool refuses files not read THIS session — read the target region
  first; after ANY failed multi-edit batch, audit which edits landed before
  retrying (MEMORY 8).
- Writing files via run_code String.raw templates: a literal backtick inside
  the content TERMINATES the template ("Expression expected") — this bit the
  kt2_judge.py write (code-fence regex). Use chr(96) assembly for fence chars.
- scripts run from scripts/ need sys.path.insert(0, repo_root) before
  "import src.model" (train.py pattern); kt2_judge.py already has it.
- Our checkpoints load natively ONLY after "import src.model" (auto-register,
  MEMORY 26); transformers 5.16.1 prints NO console loss lines — read
  tfevents via EventAccumulator (MEMORY 2) with runs/<run>/logs.
- The 360M judge is LENIENT (score-2 bias): selection = verdict score then
  LL tiebreak; watch judge-vs-LL agreement (r1: 99.6%) + unparseable gate.
- cp1252 console: set PYTHONIOENCODING=utf-8 in the pwsh call or python
  prints crash on unicode (data is fine; only the console encode fails).

## 5. Artifact map (what lives where)

- Ladder spec: docs/plans/2026-09-10_kt_ladder_plan.md (§4 KT-1, §5 KT-2,
  §6 KT-3, §7 KT-4, §8 parked/rejected incl. Route 0) · prior step handoff:
  docs/plans/2026-09-10_kt_ladder_handoff.md · THIS file = the live one.
- Transplant init: data/kt/embed_init_smol360.pt ([32768x1024] fp32 + meta;
  built by scripts/embed_transplant.py; alignment 58.7% exact / 41.1% OMP-lite
  fallback = 99.98% teacher-derived, investigated + justified, see row 43).
- Judge: scripts/kt2_judge.py (stages prompts/sample/score/select);
  configs/kt2_judge.yaml (r1) + kt2_judge_r2.yaml (r2, exclusion + probe
  reuse logic in stage_prompts); runs/kt2_judge{,_pilot,_r2}/.
- Pairs: data/sft/kt2_judge/pairs.jsonl (r1, 248) · data/sft/kt2_judge_r2/
  (r2, pending) · combined target: data/sft/kt2_sft_r2/pairs.jsonl (STEP 1).
- SFT: configs/kt2_sft.yaml (r1) · runs/kt2_sft/final (+ gates/gate_1024.json)
  · kt2_sft_r2 (STEP 2-4) · LoRA recipe origin: configs/yarn_lora_sft_v1.yaml.
- Gate baselines: runs/yarn_lora_sft_v1/gates/soup_a60_1024.json (CSN 1.8917,
  ast 0.90/0.96) · runs/kt2_sft/gates/gate_1024.json (1.8987, 0.90/0.92).
- KT-1 evidence: runs/kt_ab_control/logs vs runs/kt_ab_transplant/logs
  (tfevents; -6.24% @1000); train_summary.json in each final dir.
- Teacher: C:/Users/AhmadMhmoud/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/f8027fd0eaeea54caa13c31d31b9fdc459c38b49
  (embed [49152x960] bf16; sha256-stamped in the .pt meta).
- Prompt pool sources: data/distillation_data/omen_alpha/combined/all_train.jsonl
  (25k pairs, exec-verified, 0 overlap w/ minimax3) +
  data/distillation_data/minimax3/combined/all_train.jsonl.
