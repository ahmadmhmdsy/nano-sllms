# C12 application plan - distillation stage for nano_SLMs

**STATUS: Tier 1 IMPLEMENTED + LAUNCH-READY (2026-09-06).** User decision: prepare everything
now, execute only after M3 completes (SFT from checkpoint-1000 was considered and DECLINED —
20%-trained base, no co-run VRAM, double GPU cost; see research/c12_runbook.md decision log).
Gate unchanged: (1) M3 completion (runs/target/final) + eval.py + metric commit (HANDOFF §8.2);
(2) launch only per research/c12_runbook.md after scripts/c12_preflight.py passes. Prep-time
deviations (user-approved, CPU-only, live M3 reads only configs/target.yaml): sft_t1.yaml gained
data.tokens_dir (eval.py forgetting-guard fix), .gitignore gained sft_t1 weight rules,
scripts/c12_preflight.py + research/c12_runbook.md created.

Rationale: `research/c12_distillation_report.md`. Crosscheck context: row C12, action item 5.

## 1. Objective

Turn the M3 base model (T, 226.5M, code-completer) into an instruction-capable code assistant via
strong-to-weak distillation (C12) - and where cheap, validate the ~1/10 GPU-hour claim on our own ladder.

Success criteria (Tier 1):
- SFT T answers held-out Evol instructions with locally-syntactic Python; ast.parse pass-rate reported on generations.
- CSN val loss regression <= ~10-15% vs base T (no catastrophic forgetting).
- Zero NaNs / loss spikes; grad_norm settles <= ~0.6x clip (the M2/M3 signature).
- Auto-resume works (M0 drill: kill -> relaunch -> continues, zero flags).
- Metrics + summary committed after completion (HANDOFF §8.2 pattern); weights stay LOCAL (same LFS decision as M3).

## 2. Prerequisites (check at execution)

- runs/target/final/ exists (M3 done).
- Disk >= ~3 GB free for the SFT run (E: had ~16 GB at M3 launch; M3 consumes ~10 GB).
- data/target/raw/source.txt says code-search-net (not Evol) -> confirms Evol is contamination-free SFT data for T.
- tokenizer.pad_token on CodeLlama-32k: if None, set pad = eos and record it in the config.

## 3. Tier 1 - teacher-trace SFT of T on Evol-Instruct (recommended first move)

Why this is C12 almost for free: nickrosh/Evol-Instruct-Code-80k-v1 **is** teacher trace data
(evolved instructions + model-written responses, ~430 tok/row - HANDOFF §6), and T never saw it.
That is Qwen off-policy trace SFT with a public trace corpus instead of generating our own.

### 3.1 New files (created only at execution)

- `scripts/sft_data.py` - stream Evol -> filter (min_chars 200, dedupe; reuse prepare_data patterns)
  -> optional ast.parse quality pre-filter of responses -> split 98/2 (hold-out val) -> template:

  ```
  ### Instruction:
  {instruction}
  ### Response:
  {response}<eos>
  ```

  Tokenize with **prompt masking** (labels = -100 through the "### Response:" line; train only on
  response + eos), truncate to ctx 512 and drop longer rows -> save HF dataset to data/sft/evol/.
- `configs/sft_t1.yaml` - model: load runs/target/final; data: data/sft/evol + the CSN val shard path
  (regression check); train: output runs/sft_t1, ctx 512, batch 1, accum 16, **lr 3e-5 cosine**
  (SFT LR << pretraining 4e-4; final value decided at execution), warmup 100, eval+save every 250,
  save_total_limit 3, fp16, grad_ckpt, seed 42; eval: held-out instruction prompts + generation args.
- `scripts/sft.py` - HF Trainer + a padding collator that masks prompt labels
  (DataCollatorForSeq2Seq-style; **no new dependencies**). It MUST implement the §5.3 auto-resume
  contract (checkpoint scan -> trainer.train(resume_from_checkpoint=True), final+best copy to
  runs/sft_t1/final/) - the hard requirement applies to every new training script.

### 3.2 Schedule (ladder philosophy: smoke -> full)

| Run | Pairs | ~Steps (16/step) | Est. time @ 8-10 s/it (ctx 512) |
|---|---|---|---|
| sft pilot (plumbing proof) | 5k, 1 epoch | ~310-600 | ~1-2 h |
| sft full | 20k subsample, 2 epochs | ~2.1-2.5k | ~6-8 h |
| (optional) sft max | all ~75k, 2 epochs | ~9.4k | ~21-26 h |

Estimates carry the thermal caveat (HANDOFF §5.6: M2 swung 5.0-7.4 s/it). Measure the first 100
steps and re-ETA (the M3 lesson).

### 3.3 Validation

- Training curves via EventAccumulator (train/loss, train/grad_norm - §5.2 gotcha: no console loss lines).
- Instruction eval: N=50-100 held-out instructions, greedy + temp 0.8; report ast.parse pass-rate +
  eyeball; write runs/sft_t1/final/eval_report.json (existing convention).
- Forgetting guard: eval loss on the data/target val shards before vs after SFT (<= ~10-15% worse = pass).
- Side-by-side: base T vs SFT T on identical instructions (scripts/infer.py --ckpt runs/sft_t1/final).

### 3.4 VRAM (fits 6 GB)

T fp16 0.45 GB + grads 0.45 + AdamW ~1.8 + activations (grad_ckpt, ctx 512, batch 1) - strictly
lighter than the M3 regime (4.24 GB peak at ctx 1024). Padded logits: batch x 512 x 32768 x 2 B
~= 33 MB/seq fp16 - safe.

## 4. Tier 2 - on-policy logit alignment (deferred until Tier 1 passes)

- Student T samples continuations; a frozen teacher logits score them (KL): Qwen on-policy logit
  alignment (GKD/MiniLLM-style).
- Teacher on 6 GB: Qwen2.5-Coder-1.5B fp16 ~= 3.1 GB -> over budget with student + optimizer.
  Options at execution: (a) teacher in 8-bit via installed bitsandbytes 0.50.2 (~1.6 GB; VERIFY
  Turing sm_75 support first), (b) teacher on CPU (slow, small batches), (c) smaller student.
- Requires a teacher model download (~3 GB) - slow-network caveat (PLAN A5).
- Decision point: only start if Tier 1 shows instruction quality is limited by the trace data, not by plumbing.

## 5. Tier 3 - intra-ladder pretraining KD (optional; tests the ~1/10 claim)

- Design: teacher P (runs/pilot/final, 100.68M, ppl 3.19) -> S-sized student (12.3M), loss =
  KL(student || teacher logits, tau=1) on data/pilot packed shards (optional 0.5*CE blend).
- Fair A/B (report §5): S and P both pretrain on Evol at different row counts, so the baseline is a
  **from-scratch S-sized run on data/pilot shards with identical steps** - not the existing smoke checkpoint.
- Cost: S-sized runs are cheap (est. 2-4 s/it at ctx 512): baseline ~2-3 h; the distilled run adds
  a teacher forward (~+30-50% step time) -> ~3-4 h.
- Read-out: distilled >= baseline at <= 1/3 the steps => the ~1/10 claim transfers to our hardware
  -> future rungs can be distilled instead of pretrained from scratch.
- VRAM: teacher 0.2 GB + student 0.03 + optimizer ~0.15 + logits (batch 8 ~= 268 MB fp16 /
  ~536 MB fp32 in the KL) - fits with room.

## 6. Sequencing & decision points

- Tier 1 is orthogonal to the M4 architecture items (crosscheck §3 list) and consumes the M3 artifact
  as-is. Promote it to first post-M3 action if the goal is a useful instruct model; otherwise keep
  the crosscheck order. **USER DECISION (§8.3).**
- Tier 2 starts only after Tier 1 passes its success criteria.
- Tier 3 uses P (available now) but runs only when the GPU is free - never co-run with a live training run.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Catastrophic forgetting of code completion | low SFT LR; eval+save every 250 with load_best_model_at_end; CSN val regression gate |
| CodeLlama ships no chat template | fixed plain template (§3.1) stored in config; inference wrapper uses the same |
| Evol response quality varies | min_chars + dedupe + optional ast.parse pre-filter |
| Padding waste / long rows | truncate to ctx 512, drop > ctx; small eval batch |
| Thermal pace swings break ETAs | measure first 100 steps, re-ETA (M3 lesson) |
| Disk creep | save_total_limit 3 + final copy (§5.3 convention); check free space at execution |

## 8. Commands at execution time (template)

```powershell
# data prep
& .\.venv\Scripts\python.exe scripts\sft_data.py --config configs\sft_t1.yaml
# pilot SFT (plumbing proof)
& .\.venv\Scripts\python.exe scripts\sft.py --config configs\sft_t1.yaml --pilot
# full SFT - rerun the SAME command to auto-resume after ANY interruption
& .\.venv\Scripts\python.exe scripts\sft.py --config configs\sft_t1.yaml
# eval (instructions + ppl regression)
& .\.venv\Scripts\python.exe scripts\eval.py --config configs\sft_t1.yaml
```

## 9. What NOT to do

- Do not touch runs/target/ or its process while M3 is live; no src/ or configs/ edits before the gate passes.
- venv python only (.\.venv\Scripts\python.exe; uv-managed, never pip).
- Tier 1/3 need **no new packages** (HF Trainer + datasets suffice); Tier 2 needs a model download, not a package.
