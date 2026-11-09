# nano_SLMs — Full Pipeline Plan (Data → Train → Eval)

**Goal:** stand up an end-to-end SLM training pipeline on this machine (Quadro RTX 4000, 8 GB VRAM, CUDA 12.6),
validate it with the **smallest possible model + dataset**, then scale to a **≤250M** model.
**Non-negotiable requirement:** the training process auto-checkpoints and can auto-resume after any crash/interrupt.

---

## 1) Assumptions (state now, correct me if wrong)

| # | Assumption | Basis |
|---|---|---|
| A1 | Domain = **Python code** model | all 3 files in resources/ |
| A2 | Single GPU, no multi-GPU/DeepSpeed | Quadro RTX 4000, 8 GB |
| A3 | **fp16, not bf16** | Turing (sm_75) has no fast bf16; HF fp16=True + GradScaler |
| A4 | flash-attn **not used** (Ampere+ only) | PyTorch SDPA attention instead — fine on Turing |
| A5 | Network is slow (~230 KB/s observed) | torch/lib install timing → download **small subsets only**, cache locally |
| A6 | venv ready: torch 2.14.0+cu126, transformers 5.16.1, accelerate, datasets, bitsandbytes, tensorboard | installed & verified |
| A7 | v0 architecture = plain GQA decoder (RMSNorm + RoPE + SwiGLU + GQA + tied embeddings); Flash-Next hybrid (GDN/QSA) is a **later upgrade**, not the smoke test | resources' skeletons are pseudo-code; plain transformer first is what they themselves recommend |

---

## 2) Success criteria

1. A training run can be started, **killed at any moment**, and restarted with **zero manual fiddling** — it resumes from the last checkpoint automatically.
2. The pipeline produces: tokenized data shards → loss curves (TensorBoard) → periodic val loss/perplexity → final model + tokenizer → sample generations.
3. The ~110M "pilot" model trains stably for hours without OOM, with VRAM ≤ ~6.5 GB.
4. A ~250M config is **proven to fit** (VRAM probe) before any long run is attempted.

---

## 3) Model ladder (start small, confirm, scale)

All: decoder-only, RoPE, pre-norm RMSNorm, SwiGLU, GQA (Q heads > KV heads), tied embeddings, fp16.

| Config | Layers | d_model | Q/KV heads | d_ff | Vocab | Ctx | ~Params | Purpose |
|---|---|---|---|---|---|---|---|---|
| **S — smoke** | 4 | 256 | 4/2 | 1024 | 32k | 256 | ~12M | prove pipeline end-to-end in minutes |
| **P — pilot** | 12 | 768 | 12/4 | 2048 | 32k | 512 | ~110M | real multi-hour run (note: resources' "~250M" row is ~110M with tied emb + 32k vocab) |
| **T — target** | 16 | 1024 | 16/4 | 3072 | 32k | 512–1024 | ~250M | only after P is stable; VRAM-probed first |

Params verified by formula: emb = V·d (tied); per layer = 4·d² (attn) + 3·d·d_ff (SwiGLU).

---

## 4) Data pipeline

### 4.1 Sources
- **Smoke (S):** TheGamingMahi/TinyCode (HF, small synthetic Python) — take 2,000–10,000 rows. ~10s of MB download.
- **Pilot (P):** bigcode/the-stack-v2 **streamed**, filter lang=python + permissive licenses, cap file length ≤ 4k tokens, dedupe (exact-hash on normalized text), target **50–150 MB raw ≈ 15–50M tokens**. Streamed with take(N) so we never download the full TB-scale dataset.
- **Target (T):** same source, more shards (only if M2 succeeds).

### 4.2 Steps (scripts/prepare_data.py)
1. Stream → filter (language, license, length) → dedupe → write data/raw/python_*.jsonl (one {"text": ...} per line).
2. Tokenize (scripts/tokenize_data.py): **Phase S/P reuse an existing 32k tokenizer** (CodeLlama per resources, or Qwen2.5-Coder's — decide at implementation; CodeLlama matches resources) → pack into fixed-length uint32 binary shards (data/tokens/train_XXXX.bin, val_XXXX.bin), memmap-loaded at train time (no runtime tokenization bottleneck).
3. Val split: hold out ~1–2% of shards, never mixed into train.
4. (Later milestone, optional) train our own 32k BPE on the corpus with tokenizers — needed only when we outgrow borrowed tokenizers.

---

## 5) Training pipeline (scripts/train.py)

### 5.1 Stack
HF Trainer + TrainingArguments (matches resources' pattern), custom model class in src/model.py (registered or passed directly).

### 5.2 Fixed settings (8 GB VRAM fit strategy)
- fp16=True (GradScaler auto) — **not** bf16 (Turing)
- gradient_checkpointing=True
- per_device_train_batch_size=1, gradient_accumulation_steps=32 (effective batch 32 seqs)
- S: seq 256 · P/T: seq 512 first, 1024 only if VRAM headroom confirmed
- Optimizer: AdamW (beta2=0.95, wd=0.1, grad clip 1.0, cosine LR, warmup ~1–3% of steps); LR 4e-4 (S/P)
- If OOM at any point → switch optim="adamw_bnb_8bit" (bitsandbytes installed) before touching batch size
- Dataset returns pre-packed fixed-length blocks → default_data_collator, no padding waste

### 5.3 AUTO-CHECKPOINT spec (the hard requirement)

1. **Rolling checkpoints:** save_steps = 50 (S) / 500 (P), save_total_limit=3 → always keeps the 3 most recent checkpoint-* dirs (model + optimizer + scheduler + RNG + trainer state). Disk cost ~3× model+optim size (P ≈ ~2.5 GB total — fine).
2. **Best checkpoint:** load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, save_strategy="steps" aligned with eval_strategy="steps" → the best-val checkpoint is never rotated away.
3. **Auto-resume on start:** train.py always scans --output_dir for checkpoint-*; if any exist it calls trainer.train(resume_from_checkpoint=True) (HF picks the newest). Launching the same command twice continues the run — no flags, no manual paths. This is the single behavior that makes the run "unattended-safe".
4. **Crash surface covered:** optimizer state, LR schedule, grad-scaler, dataloader position, RNG state all live inside HF checkpoints → resume continues at the exact saved step.
5. **Keep-final-forever:** after training completes, train.py copies the final + best checkpoints to runs/<name>/final/ so save_total_limit can never delete the finished artifact.
6. Checkpoints in **safetensors** (save_safetensors=True, default).

### 5.4 Logging
TensorBoard (logging_steps=10 S / 50 P) → runs/<name>/logs/: loss, LR, grad-norm, tokens/s.

---

## 6) Eval pipeline

| When | What | How |
|---|---|---|
| During training (every eval_steps) | val loss + perplexity | eval_strategy="steps" on held-out shards; ppl = exp(eval_loss) |
| End of run | load best checkpoint | load_best_model_at_end |
| Post-run (scripts/eval.py) | perplexity on full val set; **generation sanity**: 10 prompts (function stubs, e.g. "def is_prime(n):") → greedy samples; eyeball syntax/coherence | scripts |
| Post-run (optional, later) | tiny held-out task battery; HumanEval only if we get that far | out of scope for M0–M2 |

For S/P the honest bar is *mechanical*: loss ↓ smoothly, val ppl decreases, generated Python is locally syntactic. A 50M-token pilot will **not** produce a useful coding assistant — the deliverable is a **validated, resumable pipeline**, not a good model.

---

## 7) Repo layout (to be created)

```
nano_SLMs/
├─ PLAN.md                  ← this file
├─ resources/               (existing)
├─ configs/
│  ├─ smoke.yaml            (S config + train args)
│  ├─ pilot.yaml
│  └─ target.yaml
├─ scripts/
│  ├─ prepare_data.py       stream→filter→dedupe→jsonl
│  ├─ tokenize_data.py      jsonl→packed .bin shards
│  ├─ train.py              train + auto-resume + eval loop
│  ├─ eval.py               ppl + generation samples
│  └─ vram_probe.py         250M fit test (M1)
├─ src/
│  ├─ model.py              GQA decoder (S/P/T sizes)
│  └─ data.py               memmap shard dataset
├─ data/                    raw/, tokens/   (gitignored)
└─ runs/                    checkpoints+logs (gitignored)
```

---

## 8) Milestones

### M0 — Smoke test (S, ~12M params, TinyCode 2k rows)
- Build repo skeleton, model, data prep, tokenizer packer, train.py with §5.3 auto-resume.
- Run ~200 steps (~minutes). **Then deliberately kill mid-run and relaunch → must resume from last checkpoint with no manual action.** This is the acceptance test for the checkpoint requirement.
- Verify: loss decreases; val eval runs; checkpoint rotation works; eval.py generates text; TensorBoard shows curves.
- **Exit criteria:** kill/resume drill passed + all artifacts produced. Est. total: 1–2 h (mostly implementation).

### M1 — VRAM capability probe (T config, no real training)
- vram_probe.py: build 250M model, fp16 + checkpointing, batch 1, seq 512, run forward+backward+optimizer.step for ~20 steps on synthetic tokens; report peak VRAM + tokens/s.
- Decision gate: peak ≤ 6.5 GB → T approved for M3; 6.5–7.8 GB → switch to 8-bit Adam, re-probe; OOM → stay at seq 512 / lock P as max size.
- Also measures P config throughput → honest time estimate for M2/M3. Est.: 30 min.

### M2 — Pilot run (P, ~110M, 15–50M tokens from The Stack v2 python subset)
- Prepare data (streamed subsets — expect slow download; run as background job).
- Train ~1–2 epochs; checkpoints every 500 steps; eval every 500.
- Unattended: start via background job; monitor via TensorBoard/log; auto-resume handles any interruption.
- Exit: stable loss curve over ≥4 h, no OOM, val ppl ↓, samples locally syntactic, final+best artifacts saved. Est.: download 1–3 h (slow network) + training 4–8 h.

### M3 — Target run (T, ~250M) — *optional continuation*
- Only after M2 exit criteria. Longer context (1024) if probe allows. Same auto-checkpoint machinery, more data. Days-scale.

---

## 9) Risks & mitigations

| Risk | Mitigation |
|---|---|
| OOM at 250M | M1 probe gates M3; 8-bit Adam; seq 512; batch 1; grad checkpointing already on |
| bf16 temptation on Turing | fp16 only (A3) |
| Slow network stalls data prep | stream + take(), background jobs, TinyCode first (tiny download) |
| transformers 5.x API drift (eval_strategy, processing_class, …) | verify arg names against installed 5.16.1 at implementation time |
| Windows num_proc in datasets.map | guard with if __name__ == "__main__"; default num_proc=1 if flaky |
| Disk creep from checkpoints | save_total_limit + final-artifact copy (§5.3.5); data/ + runs/ gitignored |
| Long-run interruptions (this machine) | the entire §5.3 spec exists for this |

## 10) Out of scope (for now)
MoE, Flash-Next GDN/QSA hybrid blocks (upgrade path after M3), custom tokenizer training, distributed training, HumanEval-style benchmarking, inference server.
(HumanEval/MBPP benchmarking came into scope for the SmolLM2 line — see §11.)

---

## 11) SmolLM2-135M — fluent Python sub-agent (user-approved 2026-10-18)

**Goal (user):** a *fluent Python sub-agent* on SmolLM2-135M, with no switch to a bigger base.

**Definition (what "fluent" means here, so it can be measured):**
- The model is a *worker*, not a planner.
- **Input:** a scoped request, such as a signature + docstring + examples, a traceback to fix, or a tool request.
- **Output:** correct Python, or ONE strict tool-call format.
- **Loop:** inside a runtime loop (`run_python` / `run_tests` / `read_file`) it uses tool output to self-correct.
- **Context:** short, ≤ ~2k tokens.
- Planning stays with an orchestrator: a bigger model or plain code.

**Scope (SA-B, user-adopted 2026-10-27; research/2026-10-27_sa_b_scope_proposal.md):**
- **Product statement:** writes short single-function Python (list, dict, math, string) with tests. It is not a general-purpose coder.
- **In scope:** tasks whose solution is ≤ 48 AST nodes, needs no imports and is one function.
- **Out of scope until the model grows:** imports / stdlib APIs (itertools, re, heapq), multi-helper programs, nested-loop algorithms, tasks above ~76 nodes.
- Size comes from the reference solution, so routing at inference needs a proxy; the evidence is one checkpoint (sa4_scale075).

### 11.1 Gates (targets to measure against, not promises)

| Metric | Baseline (SA-0) | Target |
|---|---|---|
| HumanEval / MBPP-sanitized pass@1, greedy (`scripts/bench_passk.py`) | SA-0 re-run | 10–15 % |
| pass@10 with test filtering | SA-0 | 25–35 % |
| Playground ladder (`scripts/bench_ladder.py`, dev) | L0 71/71 (CPT+LoRA) | L4 ≥ 90 % of 1,056 |
| Breadth ladder (SA-B, new sub-skills, held-out `final` split) | — | ≥ 70 % on ≥ 1,000 tasks / ≥ 200 sub-skills |
| Tool-call format validity | — | ≥ 98 % (100 % with constrained decoding) |
| Fix-from-traceback success (own held-out set) | — | ≥ 40 % |
| English forgetting (BPB on `val_en.bin`) | base | ≤ +3 % vs base |

### 11.2 Battery ladder (≥ 1,000 tasks, stepped to save compute)

There are two axes. Both are stepped, so a weak checkpoint never pays for 1,000 tasks.

**Robustness ladder** (`scripts/bench_ladder.py`, built 2026-10-18):
- Same 88 sub-skills, more instances: fresh entry points, descriptions and generated cases.
- Levels: L0 core×1 = 71 → L1 all×1 = 88 → L2 all×3 = 264 → L3 all×6 = 528 → L4 all×12 = 1,056.
- An arm climbs to the next level only while its strict pass rate is ≥ `--gate` (default 0.90).
- `--reuse` skips levels already on disk.

**Breadth ladder** (SA-B):
- New construct ladders in `glm53_flash/playground.py`:
  - classes, exceptions, f-strings/formatting
  - dict/set comprehensions, generators, sorting/bisect
  - stdlib `collections`/`itertools`/`re`, string parsing, multi-function tasks
- Two agent tiers: *fix-from-traceback* and *tool-call*.
- Target: ≥ 200 sub-skills × 5 instances = ≥ 1,000 tasks.
- **Built 2026-10-18:**
  - 33 `BREADTH` constructs / 192 new sub-skills (150 write + 22 fix + 20 tool). Pools: `breadth`, `wide` (280 = 88 + 192), `fix`, `tool`.
  - `bench_ladder.py` levels: B1 breadth×1 = 192, W2 wide×2 = 560, W4 wide×4 = 1,120. W4 is the ≥ 1,000-task / ≥ 200-sub-skill battery, at 4 instances rather than 5.
  - The original 88 sub-skills (L0-L4) stay byte-identical.

**Compute policy:**
- During a run, checkpoints get L0 + L1 only (≤ 159 tasks).
- L2–L4, HumanEval/MBPP and the breadth ladder run only at stage ends.
- The `final` split is touched ONLY for stage gates, never for tuning.
- Training draws from the `rl` / `pretrain` splits and MBPP-train.

**SA dev gate (SA-V.1, 2026-10-27):**
- Primary for code: the clean gate, 370 he/mbpp val tasks (he 180, mbpp 190) in research/raw/sa_v_clean_gate.json. Run it with `sa5_rft.py --split val --task-ids research/raw/sa_v_clean_gate.json --code 370 --fix 0 --tool 0`.
  - Rule: whole duplicate cluster in val, 13-gram coverage < 0.5 against sa2 + sa4 train. Built by scratch/sa_v_clean_gate.py; audit: research/raw/sa_v_clean_gate_audit.json.
- **Scoped gate (SA-B, 2026-10-27), reported next to the clean gate, never instead of it:** the 99 clean-gate tasks whose reference is ≤ 48 nodes, import-free and single-function, in research/raw/sa_b_scoped_gate.json (`ids`, plus `he_mbpp_ids`: HE 55 + MBPP 223). Built by scratch/sa_b_scoped_gate.py; run it with the same sa5_rft command and `--task-ids research/raw/sa_b_scoped_gate.json --code 99`. Baselines (sa4_scale075): E k=10 pass@1 16.1 %, solved@10 43.4 %; SA-7 runtime 39.4 % vs greedy 20.2 % (oracle 45.5 %).
- The old 567-task gate (code 300 / fix 267) is still reported, for continuity only: 424 of its tasks have a train twin.
- Fix signal: the tier-fix eval only, since no fix/repair val task passes the clean rule.
- Grader: a timeout or RSS kill is retried once at 4× the timeout. A retry that passes marks the sample NOT_EVALUATED: it is left out of the gate mean and the RLOO group. A retry that fails scores 0.

### 11.3 Stages (TASKS.md rows SA-0 … SA-7)

| Stage | What | Reuses | Exit gate |
|---|---|---|---|
| SA-0 | Baselines: clean pass@1 (idea D) + ladder L0–L4 on base / CPT / LoRA / CPT+LoRA | bench_passk, bench_ladder | numbers recorded |
| SA-1 | CPT + 15 % English replay (idea C), so the model keeps reading instructions | smollm2_mix_english, sft.py | code BPB ≤ CPT; English BPB ≤ +3 % (2026-10-19: English PASS +0.3 %; code FAIL +4.0 % vs CPT; B1 22/192 = CPT) |
| SA-B | Breadth battery (≥ 1,000 tasks, agent tiers) | playground.py, evaluator, sandbox | battery tests pass; base and checkpoints scored |
| SA-2 | Data: textbook-quality Python (each download needs user OK) + own generator (bug injection → traceback → fix; tool-call traces); execution-verified, decontaminated | corpus_pipeline, bench_decontam | 0 benchmark leakage; every sample's tests pass |
| SA-3 | Instruction SFT on verified instruction→code | sft.py (LoRA) | pass@1 > SA-0 best |
| SA-4 | Agent-format SFT (strict tool-call JSON, multi-turn fix traces) | sft.py | format validity ≥ 98 %; fix ≥ 25 % |
| SA-5 | Expert iteration / RFT: sample k, keep the passing ones, retrain; 2–3 rounds | train_rft.py idea, bench_passk executor | pass@1 up each round; no L0 regression |
| SA-6 | **RL (mandatory, user 2026-10-18).** Details below. | train_rl.py, sandbox.py | pass@1 and ladder ≥ SA-5 on `final` split; forgetting within gate |
| SA-7 | Runtime: constrained tool-call decoding, sample-k + test filter, retry-with-traceback; agent-loop eval | sandbox | tool validity 100 %; agent-loop success reported |

**Many-ways principle (user-approved 2026-10-25, TASKS SA-C):** grade the result, not the style.
- The sandbox has two layers. The safety layer is always on: no dunders, the call and import allowlists, the size cap, and the out-of-process timeout. The style layer is lifted by the named preset `evaluator.OUTCOME_TIER`; evaluator defaults never change.
- Diversity is measured, not only pass@1: `bench_passk --samples N` reports pass@k and AST-distinct correct solutions.
- Training data carries several verified solutions per task, starting with the next teacher batch.
- RL can use an outcome-only reward (`rl_smollm2 --policy outcome`).
- SA-4 adds repair loops with real sandbox errors as feedback.
- SA-7 samples N answers and returns the first one that passes the tests.

**SA-6 RL details:**
- **Port:** the RL playground (`glm-5.3-flash-from-scratch/scripts/train_rl.py`, RLOO with a sandboxed unit-test reward) to HF SmolLM2 + LoRA.
- **Reward:** case-fraction / binary from `glm53_flash.evaluator`.
- **Tasks:** playground `rl` split, MBPP-train, generated fix tasks.
- **Reporting:** the model result on every gate.

**Honest limits:**
- A 135M model stays a worker: no long-horizon planning, no large codebases, weak recall of rarely-seen library APIs.
- The ceiling is set by data quality and the test/retry loop, not by the model's own reasoning.

**References:** phi-1 (data quality at small scale), replay vs forgetting (Ibrahim et al. 2024), RLOO (Ahmadian et al. 2024).

### 11.4 SA-M — usable scoped product (user-approved 2026-10-28)

Ideas A (more samples + better selection) and D (scope router / abstain), on the current models. TASKS.md row SA-M.

- **Step 1 (GPU):** coverage curve. k=64 samples per task, no retry, all samples stored with visible and full results. Score oracle and visible-filter runtime at N = 8 / 16 / 32 / 64 from sample prefixes. Scoped 99 on sa4_scale075 and sa_b_scale075, then the clean 370 on the better one. This also picks the shipped model.
- **Step 2 (CPU, offline):** selection methods on the stored samples: visible filter; + output-agreement vote on auto-generated extra inputs; + tie-break (most common / shortest). Gate: close ≥ half of the runtime-to-oracle gap.
- **Step 3 (CPU):** scope router + abstain. Post-hoc abstain (no visible passer, or a small vote group) and a prompt classifier (imports named, arg count, docstring length, example count) trained on reference-size labels. Report coverage vs accuracy-on-answered over the clean 370. Proposed gate: ≥ 75 % accuracy on answered tasks, fixed after step 1.
- **Step 4:** product eval on the clean 370, scoped 99 and HE/MBPP arms; GPU and CPU latency; frozen runtime config.
- **Milestone gate:** scoped runtime ≥ 45 % (now 39.4 %) and the accuracy-on-answered gate.
- **Out of scope:** new data (B), RL (C), model growth (E). Growth stays the last resort; a few cheap layers are acceptable if VRAM and compute hours allow. When B runs later, the teacher is the Gemini API free tier only (user decision 2026-10-28, via scripts/teacher_client.py), execution-verified and decontaminated.

## 12) Never-seen bench (NS), user-approved 2026-11

**Purpose:** a fixed test the model never trained on, used to show whether each new model improves or gets worse against plain base SmolLM2 (`models/hf/SmolLM2-135M`, nothing added). Each focused or general task is its own set and is scored separately, never mixed.

**Sets:** A_cruxeval, A_heval_pro, A_lcb, A_mbpp_pro, A_rwpb (from the net); B_body, B_fix, B_general, B_predict, B_tool_call, B_write_fn (ours, via Gemini). Bench data is eval-only.

**Scoring:** greedy plus runtime, with oracle shown as the ceiling.

| Tier | Items | k | max-new | Use | Report |
|---|---|---|---|---|---|
| quick | 50 per set, fixed by sha256 of id | 8 | 256 | every new model, about 20 min at batch 128 | `research/raw/bench_ns_<name>_quick.json` |
| full | all | 16 | default | final numbers for a model worth keeping | `research/raw/bench_ns_<name>.json` |

A tier is compared only with itself; the compare refuses mixed k or max_new. Baselines: `bench_ns_base_quick.json`, `bench_ns_base.json`. Results live in HANDOFF, 2026-11-02 entry.
