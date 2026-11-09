# SmolLM2-135M LoRA SFT path — access, tokenizer/EOS audit, corpus, trainer feasibility, VRAM

**Date:** 2026-10-15 · **Status:** PREPARED / NOT RUN · **GPU:** untouched (foreign job held 4734/8192 MiB)
**Scope:** CPU + network only. Nothing under runs/ was created or modified. No trainer was written or edited.

## 0. TL;DR

| Task | Verdict | Evidence |
|---|---|---|
| 1 Access | **PASS** | HuggingFaceTB/SmolLM2-135M @ 93efa2f0…, gated=False, staged locally (272,447,893 B) |
| 2 Tokenizer + EOS | **PASS** | vocab 49152, eos/bos/unk=`<|endoftext|>`=0, pad None; eos written once/example; reread count == 43,200 / 2,160 |
| 3 Corpus re-pack | **PASS** | data/reference/smollm2_pg_eos/ — 8,224,984→2,386,306 ids (train), 411,120→119,273 (val) |
| 4 Trainer feasibility | **YES via scripts/sft.py** | model+dataset+peft surface validated on CPU; 2 gaps left (VRAM ceiling, eval harness) |
| 5 VRAM | **FITS at batch 4 with grad-ckpt** | analytic peak ≈ 2.5–3.0 GiB vs ~4.3–4.5 GiB cap when idle |

The training half is launch-ready with **no trainer code change**. The two remaining gaps are (a) `scripts/sft.py` has **no §20 allocator ceiling**, and (b) **no eval harness can score an HF SmolLM2 model** on the playground battery — see §4/§6.

---

## 1. Access (task 1) — PASS

Verified with `huggingface_hub.HfApi` using `HF_TOKEN` read from the project `.env` (present, len 37; never printed). Probe: `scratch/smollm2_probe.py` (exit 0).

- **Model id:** `HuggingFaceTB/SmolLM2-135M`
- **Revision:** `main` = `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (only branch; **no tags exist**). lastModified 2025-02-06.
- **Gated:** `False` (private False, license apache-2.0, downloads 1,623,820, library_name transformers, 10 repo files).
- **Repo files:** config.json, generation_config.json, merges.txt, model.safetensors, special_tokens_map.json, tokenizer.json, tokenizer_config.json, vocab.json, README.md, .gitattributes.
- **Staged locally, pinned revision:** `models/hf/SmolLM2-135M/` — `model.safetensors` 269,060,552 B (256.6 MiB); whole tree 272,447,893 B. Config `torch_dtype` is bfloat16, but the model is fp16-safe to load on Turing.
- **Disk:** E: 41.98 GiB free at probe time. No 2T-token download; nothing from the pretraining corpus was fetched.

## 2. Tokenizer + EOS audit (task 2) — PASS

Loaded `AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")` into the default HF cache (probe `scratch/smollm2_tok_audit.py`, exit 0).

| Field | Value |
|---|---|
| class / fast | GPT2Tokenizer / True |
| vocab_size = len(tok) | **49152** |
| model_max_length | 8192 |
| eos_token / **eos_token_id** | `<|endoftext|>` / **0** |
| bos_token / bos_token_id | `<|endoftext|>` / 0 |
| unk_token / unk_token_id | `<|endoftext|>` / 0 |
| **pad_token / pad_token_id** | **None / None** |
| add_bos_token / add_eos_token | false / false |
| additional_special_tokens | [] |
| all_special_ids | 0..16 |

Encoding probes: `encode("hello world", add_special_tokens=False) = [28120, 905]`; `encode("<|endoftext|>", add_special_tokens=False) = [0]`.

**How EOS is written into the packing (explicit):** each recovered example is encoded with `add_special_tokens=False` (so no implicit BOS/EOS), then **exactly one** `eos_token_id = 0` is appended to that example's ids, and the per-example arrays are concatenated into one stream. No padding token is ever written into the `.bin`.

**How it was VERIFIED (counted, not assumed):**
1. `eos_in_text_only_encoding = 0` for both splits — ordinary text **never** produces id 0, so counting id 0 in the packed stream is unambiguous (no collision with a real text token).
2. The written streams were re-read from disk and id 0 counted: **train 43,200 == 43,200 examples**, **val 2,160 == 2,160 examples**.
3. The source byte corpus independently carried exactly 43,200 / 2,160 id-2 EOS (bin probe: only special id present is 2).

This is the direct antidote to the E-122 "EOS that was never in the corpus" failure: the token is written deterministically and its presence is asserted by count, with a zero-count control on the raw text.

## 3. Corpus re-pack (task 3) — PASS

Source: `data/reference/bytes_playground_eos/{train,val}.bin` (uint16 little-endian; `id = 4 + utf8_byte`, byte-scheme EOS = **2**).
Recover: split on id 2, subtract 4, decode utf-8 (bad_utf8_examples = 0). Re-encode each example with SmolLM2's **own** tokenizer; append one id 0.
Script: `scratch/smollm2_repack_pg.py` (job exit 0). Output: `data/reference/smollm2_pg_eos/`.

| split | byte ids before | SmolLM2 ids after | ratio | ids/example before → after | examples | EOS before | EOS after (reread) | .bin bytes |
|---|---|---|---|---|---|---|---|---|
| train | 8,224,984 | 2,386,306 | 0.2902 | 190.39 → 55.24 | 43,200 | 43,200 | 43,200 ✅ | 4,772,612 |
| val | 411,120 | 119,273 | 0.2901 | 190.33 → 55.22 | 2,160 | 2,160 | 2,160 ✅ | 238,546 |

Example token lengths (SmolLM2): train min/median/max = 28/53/100; val 29/52/100.
sha256 — train.bin `3f268891604499ced2c70dea92dfc8d99aacc3e1919f7cf6cb206c436da63132`; val.bin `eee162c9f6ee6545ba9e54ff77e8f08d1e1caca5a2a9930b9ee1b0917f3a3b8d`; train.txt `5cdb66ac91639754e0d1da250d2489daeec37c26397c2205d30cc061a1ea6b9e`; val.txt `93cc247d2f98b9e0f8da34e18aba4bf6b41a46ffd553cc813061bf97b31e5b59`.

**Strict val split kept:** the val tail is the corpus's own held-out val.bin (2,160 examples / 119,273 ids); train never sees it.

**Dataset materialization for the trainer** (CPU; `scratch/smollm2_make_dataset.py`, job exit 0) → `data/reference/smollm2_pg_eos/ds/{train,val,manifest.json}`:
- ctx=1024 causal-LM blocks, `labels == input_ids` (full CLM; the corpus is bare reference bodies + example-level EOS, exactly the E-122h analogue).
- train 2,331 rows / 2,386,306 tokens / **EOS 43,200** (final row 386 tokens); val 117 rows / 119,273 tokens / **EOS 2,160** (final row 489).
- Honest note: a first pass sliced only whole 1024-blocks and silently dropped the tail (8 train + 9 val EOS). Fixed by keeping the partial final row; the collator masks pad positions with -100, so a short row is safe. The numbers above are the corrected ones.

## 4. LoRA trainer feasibility (task 4) — verdict + exact gap

**Files inspected:** `glm-5.3-flash-from-scratch/scripts/train_aspect_lora.py`, `glm-5.3-flash-from-scratch/glm53_flash/lora.py`, `scripts/sft.py`, `src/model.py` (`maybe_wrap_peft`, `save_final`), `mex/scripts/train_mu2_lora*.py`, `configs/sft_t1.yaml`, `scratch/bench_playground_batched.py`.

- **`train_aspect_lora.py` is NOT reusable for SmolLM2.** It is hardwired to the custom byte model: `from glm53_flash import ByteTokenizer`, `load_checkpoint()` (our `.pt` format), `model.config.max_sequence_length`, `select_tasks(..., pool="aspects")`, and it calls `logits, _ = model(inputs)` (a 2-tuple). It also uses the repo's own LoRA (`glm53_flash.lora.apply_lora`, targets alias to `qkv`/`out`) rather than PEFT. Pointing it at `LlamaForCausalLM` would be a rewrite, not a retarget.
- **`glm53_flash/lora.py`** is a custom `LoRALinear` for this decoder (fused qkv / out leaves) — incompatible with HF Llama module names and not PEFT.
- **`mex/scripts/train_mu2_lora.py`, `train_mu2_lora_muon.py`, `train_e58_head.py`** do use PEFT but are bound to the mu2 model + its own data/config format.
- **`scripts/sft.py` is the viable base and needs no trainer edit.** It is model-agnostic: `AutoTokenizer.from_pretrained(name)`, `AutoModelForCausalLM.from_pretrained(base, attn_implementation="sdpa")`, `maybe_wrap_peft(model, cfg)`, then HF `Trainer`. It already carries the transformers 5.16.1 adaptations (`processing_class=`, `eval_strategy=`; no `logging_dir`/`save_safetensors`/`fp16`-as-removed-arg), plus `fp16=`, `gradient_checkpointing_kwargs={"use_reentrant": False}`, `adamw_torch`.

**CPU validation of the exact trainer surface** (`scratch/smollm2_validate.py`, job exit 0):
```
dataset: train_rows=2330->2331, columns=[input_ids, labels], labels==input_ids
AutoTokenizer.from_pretrained("models/hf/SmolLM2-135M") -> pad_token_id None
AutoModelForCausalLM.from_pretrained("models/hf/SmolLM2-135M", dtype=fp16, attn_implementation="sdpa")
src.model.maybe_wrap_peft(model, cfg) -> PeftModelForCausalLM
  base_total = 134,515,008 (134.5M) · trainable = 1,843,200 (1.35%) · max_position_embeddings 8192
```
PEFT 0.20.0 wraps `LlamaForCausalLM` cleanly (probe `scratch/smollm2_peft_test.py`): r=16 on q/k/v/o → 1,843,200 trainable; r=8 → 921,600; all-linear r=16 → 4,884,480.

**Exact gap (what is missing / was changed):**
1. **base_model must be a local path under ROOT** — `scripts/sft.py` gates on `ROOT/cfg["base_model"]` existing and names an HF id would fail. **Closed:** staged `models/hf/SmolLM2-135M` at the pinned revision.
2. **Data must be `datasets.load_from_disk(dataset_dir/{train,val})` with `{input_ids, labels}`** — the `.bin` is not directly consumable. **Closed:** `data/reference/smollm2_pg_eos/ds/` via the scratch materializer (**promote to scripts/ if adopted**).
3. **pad_token None** → `scripts/sft.py` already sets pad = eos (id 0). No change.
4. **OPEN — §20 allocator ceiling.** `scripts/sft.py` never calls `set_per_process_memory_fraction` (unlike the custom runtime's `apply_vram_cap`). To honor the 60% cap it needs a ~3-line insert before `Trainer(...)` (or a wrapper); **not made** because the brief forbids a new/edited trainer.
5. **OPEN — eval harness (the larger gap).** `scratch/bench_playground_batched.py` (the 64/71 instrument) imports `glm53_flash.ByteTokenizer` and `glm53_flash.runtime.load_checkpoint`; it cannot load or generate from an HF `PeftModel`. There is **no runnable SmolLM2 → executable-Python scoring path yet**, so the 64/71 comparison cannot be produced without a new generation+scoring harness. This is a separate deliverable.

Config prepared (not run): `configs/sft_smollm2_pg.yaml` (r=16, alpha=32, targets q/k/v/o, ctx 1024, batch 4, accum 8, lr 1e-4, 3 epochs, fp16, grad_ckpt true).

## 5. VRAM feasibility (task 5) — ANALYTIC ESTIMATE (GPU busy; not measured)

GPU (live): Quadro RTX 4000, total 8192 MiB, used 4734 MiB, free 3261 MiB, util 23%; sm_75, fp16 only, no bf16, no flash-attn → SDPA (mem-efficient/math). **60% of free = 1957 MiB right now → do NOT launch.**

Dims: H=576, L=30, nh=9, nkv=3, hd=64, I=1536, V=49152, ctx=1024, B=4.

| Component | Batch 4 @1024, fp16 | Notes |
|---|---|---|
| Base weights (frozen) | 256.6 MiB | 269,060,552 B measured |
| LoRA r16 q/k/v/o + AdamW m,v/grad (fp32) | ~33 MiB | 1.84M trainable params |
| CUDA context / workspaces | ~300–500 MiB | typical for this driver/venv |
| Saved activations, grad-ckpt | ~135 MiB | 30 × 4×1024×576×2 B |
| Backward recompute (1 layer) + attn | ~125 MiB | score matrix 75 MiB if math fallback |
| Logits fp16 + CE | ~400–1170 MiB | 4×1024×49152×2 = 402 MiB; fp32 upcast adds ~768 MiB |
| **Peak with grad-ckpt** | **~2.5–3.0 GiB** | fits the ~4.3–4.5 GiB idle cap |
| Peak **without** grad-ckpt | ~4.0–4.5 GiB | saved activations ~1.5 GiB → at/over cap |

**Cap comparison:**
- **Now** (foreign job active): cap 1957 MiB < projected 2.5 GiB → **do not launch** (this is why the run is prepared, not started).
- **When idle** (assumed free ≈ 7.2–7.5 GiB after WDDM desktop baseline): cap ≈ 4.3–4.5 GiB → batch 4 fits with ~1.3–1.8 GiB headroom.
- **Safe batch size: 4, with `grad_ckpt: true`.** Batch 2 is ultra-safe; batch 8 (~3.5–4.5 GiB) likely violates the cap — ramp only after a measured pilot. Enforce the cap in code (§4 gap 4) and confirm no shared-memory spill (§20.3).

## 6. Ready-to-run command list (for when the GPU frees)

Forward slashes + a Test-Path preflight, per the E-122h transport lesson. The preflight and static gate are CPU-only and can run now; the trainer lines are for later.

```powershell
# 0) preflight — fail LOUDLY before anything else
foreach ($p in @(
  'scripts/sft.py',
  'configs/sft_smollm2_pg.yaml',
  'models/hf/SmolLM2-135M/model.safetensors',
  'models/hf/SmolLM2-135M/tokenizer.json',
  'data/reference/smollm2_pg_eos/ds/train',
  'data/reference/smollm2_pg_eos/ds/val'
)) { if (-not (Test-Path $p)) { throw "missing $p" } }

# 1) static gate (dup keys, pyflakes, --help, data, GPU smoke) — CPU-only
& .\.venv\Scripts\python.exe laya/scripts/preflight_check.py --script scripts/sft.py --config configs/sft_smollm2_pg.yaml

# 2) re-check free VRAM immediately before launch; require >= 5000 MiB free so 60% >= 3000 MiB
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

# 3) pilot (writes runs/sft_smollm2_pg_pilot only)
& .\.venv\Scripts\python.exe scripts/sft.py --config configs/sft_smollm2_pg.yaml --pilot

# 4) full run (writes runs/sft_smollm2_pg)
& .\.venv\Scripts\python.exe scripts/sft.py --config configs/sft_smollm2_pg.yaml
```

Notes: (a) step 2 must show ≥ ~5 GiB free or wait — at 3261 MiB free today it does not. (b) `scripts/sft.py` has no allocator ceiling; until gap 4 is closed, treat the free-VRAM gate + post-hoc spill check as the guard. (c) TensorBoard events are the loss source of truth; `runs/sft_smollm2_pg/final/train_summary.json` holds the final metrics.

## 7. Falsifiable prediction — "bottleneck is data" vs "bottleneck is pretraining"

Pre-registered before any run (protocol: `scratch/bench_playground_batched.py`'s playground battery, deterministic greedy, strict = whole completion is one valid function; report strict/71 **and EOS-emission rate**, per E-122i). Comparison target = E-122h's byte model, **strict 64/71** same corpus discipline.

- **H_data (our bottleneck is SFT-data/scale).** SmolLM2-135M LoRA reaches **strict ≥ 65/71** (strictly beats 64/71) **and** fixes ≥ 2 of the 4 constructs the byte model never solved — {grade, count_even, count_positive, lookup} (all 0 for E-122h). Interpretation: with a 76,000× larger pretraining budget and a 5.25× larger model, the same 43,200-example EOS corpus unlocks capability our 26.2M-token from-scratch model cannot reach → the binding constraint is pretraining data scale.
- **H_pretraining (the limit is pretraining/architecture, not SFT data).** SmolLM2-135M LoRA scores **strict ≤ 64/71**, or shows no gain on those 4 constructs. Interpretation: the residual failures are not fixed by adding a stronger base + the same SFT data, so they are a property of the task/construct coverage or the eval, not of our pretraining budget.
- **Confound to state with either result:** the tokenizer differs from the byte model, so byte budgets and losses are **not comparable** — decide only on strict executable pass counts (+ EOS-emission rate). A "≥ 2×" style bar is banned when the control is 0 (E-122i lesson).

## 8. Artifacts & status

| Artifact | Path | Status |
|---|---|---|
| Re-packed corpus | data/reference/smollm2_pg_eos/{train,val}.bin (+ .txt, meta.json) | PASS |
| Trainer dataset | data/reference/smollm2_pg_eos/ds/{train,val,manifest.json} | PASS |
| Staged base model | models/hf/SmolLM2-135M/ (272,447,893 B, pinned 93efa2f0…) | PASS |
| Prepared config | configs/sft_smollm2_pg.yaml | ready, not run |
| Probes/helpers | scratch/smollm2_{probe,tok_audit,repack_pg,make_dataset,peft_test,validate}.py | CPU, exit 0 |
| GPU work | — | SKIPPED (foreign job active) |
| runs/ | — | UNTOUCHED |
| Trainer code | — | UNCHANGED (per brief) |
| SmolLM2 eval harness | scratch/bench_hf.py | **READY** — CPU smoke passed 2026-10-15 |
| sft.py section-20 VRAM cap | scripts/sft.py --vram-cap-gib | **CLOSED** — tests/test_sft_vram_cap.py 6/6 |

---

## 9. BLOCKERS CLOSED (2026-10-15, CPU smoke)

Both section-4 gaps are closed. scripts/sft.py now honors --vram-cap-gib (default 0
= unchanged behaviour) and scratch/bench_hf.py adds a corrected, EOS-instrumented
HF arm on top of the (unmodified) protected harness. Exact commands, exit codes,
the VRAM-cap test result and the raw HF completions are in
research/2026-10-15_smollm2_eval_harness.md. The full 71-task SmolLM2-vs-byte A/B
is GPU-ready but NOT run (GPU busy).
