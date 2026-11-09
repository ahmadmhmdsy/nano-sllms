# SmolLM2-135M eval harness + sft.py VRAM cap — both blockers closed (CPU)

**Date:** 2026-10-15 · **Status:** CODE + CPU SMOKE DONE · **GPU:** untouched (foreign job active)
**Scope:** CPU only. Nothing under runs/ was created or modified. The protected
`scratch/bench_playground_batched.py` was NOT edited (hash re-verified below).
**Companion:** research/2026-10-15_smollm2_path.md (the prepared path this closes the gaps in).

## 0. TL;DR

| Blocker | Verdict | Evidence |
|---|---|---|
| A — `scripts/sft.py` had no section-20 allocator ceiling | **CLOSED** | `--vram-cap-gib` added, mirrors `glm53_flash.runtime.apply_vram_cap`; `tests/test_sft_vram_cap.py` 6/6 PASS (exit 0) |
| B — no eval harness can score an HF model | **CLOSED on CPU** | `scratch/bench_hf.py` corrected + EOS-instrumented HF arm; pad-check PASS; 8-task CPU smoke exit 0; two-arm A/B plumbing exit 0 |

Two real defects were found in the protected `HfArm` (it had never been run) and
fixed in the wrapper. The full 71-task SmolLM2-vs-byte A/B is **GPU-ready but
NOT run** (GPU busy) — command in section 6.

## 1. Preflight (exit 0)

```powershell
foreach ($p in @(
  'scripts/sft.py','configs/sft_smollm2_pg.yaml',
  'models/hf/SmolLM2-135M/model.safetensors','models/hf/SmolLM2-135M/tokenizer.json',
  'data/reference/smollm2_pg_eos/ds/train','data/reference/smollm2_pg_eos/ds/val',
  'scratch/bench_playground_batched.py','glm-5.3-flash-from-scratch/glm53_flash/playground.py'
)) { if (-not (Test-Path $p)) { throw "missing $p" } }
& .\.venv\Scripts\python.exe -c "import sys,torch,transformers;print(sys.version.split()[0],'torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"
(Get-FileHash scratch/bench_playground_batched.py -Algorithm SHA256).Hash
```

All 8 inputs OK; venv `3.12.9 torch 2.14.0+cu126 tf 5.16.1 cuda True`.
Protected-file hash **79956C236085CF8F5B84AE712B2240AFBC6975AD30C19D819728AA38C6116DCA**
— byte-identical to the MEMORY.md-protected hash, so the wrapper imports it, never edits it.

Task-set check (the "71 core tasks"): len(playground_pool('core')) == 71
(`--tier core --pool playground --per-rung 1 --split dev`), exactly the E-122h protocol
(`--budget-bytes 160 --max-tokens 256`; byte model = 64/71).

## 2. BLOCKER A — VRAM cap in scripts/sft.py

**Change (merged into the existing file; default path unchanged):**

- `apply_vram_cap(gib)` — same semantics as `glm-5.3-flash-from-scratch/glm53_flash/runtime.py:101`
  and its caller `train_pretrain.py` (`--vram-cap-gib`, default 0.0):
  fraction = min(1.0, gib*1024**3 / total_memory), then
  `torch.cuda.set_per_process_memory_fraction(fraction, device)`.
  No-op on CPU or when gib <= 0, so **unset == current behaviour**.
- CLI: `--vram-cap-gib FLOAT` (default 0.0).
- Call site: `cap = apply_vram_cap(args.vram_cap_gib)` immediately **before**
  `AutoModelForCausalLM.from_pretrained(...)`, i.e. before HF Trainer moves the
  model to the GPU.
- `train_summary.json` now carries `vram_cap_gib` / `vram_cap_fraction` (via **cap).

**Test:** `tests/test_sft_vram_cap.py` (repo dependency-free runner).

```powershell
& .\.venv\Scripts\python.exe tests/test_sft_vram_cap.py        # exit 0
```

```
PASS unset cap is a no-op
PASS negative cap is a no-op
PASS cap sets the fraction on the CUDA device
PASS cap larger than the device clamps to 1.0
PASS cap is applied before the model reaches the GPU
PASS CLI exposes --vram-cap-gib

6/6 checks passed
```

The test monkeypatches torch.cuda (8 GiB fake device) and asserts: 0.0 / negative /
None are no-ops; 2.0 GiB -> fraction 0.25 and exactly one
`set_per_process_memory_fraction(0.25, cuda:0)`; 100.0 clamps to 1.0; the call
site precedes both `from_pretrained` and `trainer = Trainer(`; and `--help`
exposes the flag.

**Lint:** `python -m py_compile scripts/sft.py ...` exit 0; `python -m pyflakes scripts/sft.py ...` exit 0.

**Honest limit:** the actual CUDA enforcement of the fraction cannot be exercised
on CPU. See section 7.

## 3. BLOCKER B — HF arm

### 3.1 Two defects in the protected HfArm (found by running it, not by reading)

Run of the **protected** file directly, 2 tasks, batch 1 (exit 0):

```powershell
& .\.venv\Scripts\python.exe scratch/bench_playground_batched.py --arm "smol=hf:models/hf/SmolLM2-135M" --split dev --tier core --per-rung 1 --budget-bytes 160 --max-tokens 256 --batch-size 1 --limit 2 --device cpu --out scratch/bench_hf_protected_probe.json   # exit 0
```

Raw rows (from the JSON):

- dev-grade-if_else-000 completion ended ...print(grade_if_else_beidoea(90))\n<|endoftext|>
  -> **EOS leaked into the scored text**.
- dev-grade-single_if-000 completion was 158 B of garbage, no EOS.

**Defect 1 (EOS leak / budget poisoning).** transformers.generate RETURNS the eos
token (and then pads with pad_token_id; for SmolLM2 pad == eos == 0). The protected
code decodes those tokens into the completion and their bytes count against the
160-byte budget — '<|endoftext|>' is 13 bytes, so termination both corrupts scoring
and steals budget from the body.

**Defect 2 (dtype).** The protected __init__ calls from_pretrained(path) with no
dtype; the staged config's torch_dtype is bfloat16, which does not exist on this
sm_75 Turing GPU (not triggered on CPU; it would be on the GPU A/B).

### 3.2 Fix — scratch/bench_hf.py (new file; does not touch the protected file)

Imports the protected harness, supplies corrected HfArm + an EOS-instrumented GlmArm,
delegates to the protected main() so **task construction, batching, scoring and the
report schema are exactly the protected path**, then augments the written report
with per-row eos_emitted and an arm-level eos_rate (MEMORY.md E-122i lesson 3).

- HfArm: dtype = torch.float16 on cuda, float32 on cpu; attn_implementation='sdpa';
  **cuts the continuation at the FIRST eos** before the byte trim.
- Left padding + attention_mask + model.generate (HF derives per-row positions from
  the mask); no manual last-position gather is needed, which the pad-check proves.
- GlmArm EOS path is copied verbatim from the protected complete/complete_batch so
  the byte arm is bit-for-bit the protected byte arm.

### 3.3 Left-pad correctness (the padding trap) — PASS

```powershell
& .\.venv\Scripts\python.exe scratch/bench_hf_padcheck.py        # exit 0, "PAD-CHECK PASS"
```

3 prompts of different lengths are generated sequentially (batch 1) and as one
left-padded batch; token-for-token identical in all 3 rows. Uses ngram=0 (the
benchmark default); see section 7 for the --no-repeat-ngram > 0 caveat.

### 3.4 CPU smoke — base SmolLM2, 8 core tasks, batch 2 (exit 0)

```powershell
& .\.venv\Scripts\python.exe scratch/bench_hf.py --arm "smol=hf:models/hf/SmolLM2-135M" --split dev --tier core --per-rung 1 --budget-bytes 160 --max-tokens 256 --batch-size 2 --limit 8 --device cpu --out scratch/bench_hf_smoke.json   # exit 0
```

Aggregate: **strict 0/8, extracted 4/8, EOS emitted 2/8 (0.25)**. No completion
contains '<|endoftext|>' (leak fixed).

```text
task                          eos strict ext bytes  raw completion (JSON-escaped)
dev-grade-single_if-000       no  F      T   158    "\n    if score >= 90:\n        return 'A'\n    else:\n        return 'F'\n\n# Complete this function.\n# Return 'A' when the score is at least 90, otherwise 'F'.\ndef"
dev-grade-if_else-000         yes F      T   103    "\n    if score >= 90:\n        return 'A'\n    else:\n        return 'F'\n\nprint(grade_if_else_beidoea(90))\n"
dev-grade-if_elif-000         no  F      F   158    "\n    if score >= 90:\n        return 'A'\n    elif score >= 80:\n        return 'B'\n    elif score >= 70:\n        return 'F'\n    else:\n        return 'A'\n\nprint("
dev-grade-if_elif_else-000    no  F      T   160    "\n    if score >= 90:\n        return 'A'\n    elif score >= 80:\n        return 'B'\n    else:\n        return 'F'\n\nprint(grade_if_elif_else_huymyhq(90))\nprint(grade"
dev-grade-chained_comparison-000 no F   F   158    "\n    if score >= 70:\n        return 70\n    elif score >= 71:\n        return 71\n    elif score >= 72:\n        return 72\n    elif score >= 73:\n        return 73"
dev-grade-and_or-000          no  F      F   160    "\n    if score < 0.0:\n        return 'A'\n    elif score > 100.0:\n        return 'F'\n    else:\n        return 'X'\n\ndef grade_and_or_naehatl_2(score):\n    if score"
dev-grade-full-000            no  F      T   158    "\n    if score >= 90:\n        return 'A'\n    elif score >= 80:\n        return 'B'\n    elif score >= 70:\n        return 'C'\n    else:\n        return 'F'\n\nprint("
dev-count_even-comprehension-000 yes F    F   159    "\n    # Create a list comprehension to collect the even values\n    even_values = []\n    for value in values:\n        if value % 2 == 0:\n            even_values."
```

**Read this as plumbing evidence only.** Base SmolLM2 cannot follow the
"complete this function" instruction, so 0/8 is expected and is NOT a capability
verdict; the A/B needs the LoRA-tuned arm.

### 3.5 Two-arm A/B plumbing — byte vs SmolLM2, 4 tasks (exit 0)

```powershell
& .\.venv\Scripts\python.exe scratch/bench_hf.py --arm "byte=glm:glm-5.3-flash-from-scratch/runs/e122h_eos/checkpoint-0800" --arm "smol=hf:models/hf/SmolLM2-135M" --split dev --tier core --per-rung 1 --budget-bytes 160 --max-tokens 256 --batch-size 2 --limit 4 --device cpu --out scratch/bench_hf_ab_probe.json   # exit 0
```

```
[byte] strict 4/4  extracted 4/4   eos 4/4 (1.0)
[smol] strict 0/4  extracted 3/4   eos 1/4 (0.25)
```

This proves the A/B plumbing end-to-end: same entries, both arms load in one
process, per-row eos_emitted is recorded for both, and the byte arm reproduces its
known strong performance (4/4 on these 4 grade rungs; the full battery gives 64/71).

### 3.6 Why EOS is reported directly, not as at_budget

For the byte arm, "completion < budget" is a usable proxy for "terminated" (it stops
only at eos or the byte budget; at 160 B / 256 tokens the token cap is never
reached). For the **HF arm it is invalid**: HF trims to <= budget and a row can hit
the 256-token cap while producing only ~158 bytes (2 of the 8 smoke rows did). So
the wrapper records EOS emission at the generation boundary, and eos_rate is the
number to read. completions_at_budget remains only as a byte-model sanity check.

## 4. Files changed / added

| File | Kind | Note |
|---|---|---|
| scripts/sft.py | edited | apply_vram_cap, --vram-cap-gib, call before model load, **cap in summary |
| tests/test_sft_vram_cap.py | new | 6 checks, exit 0 |
| scratch/bench_hf.py | new | corrected HF arm + EOS instrumentation; imports the protected harness |
| scratch/bench_hf_padcheck.py | new | left-pad == sequential proof |
| scratch/bench_hf_protected_probe.json | new | evidence of the protected-HfArm EOS leak |
| scratch/bench_hf_smoke.json | new | 8-task HF smoke (raw completions) |
| scratch/bench_hf_ab_probe.json | new | 4-task two-arm A/B plumbing |
| scratch/bench_playground_batched.py | **UNCHANGED** | hash re-verified byte-identical |

## 5. Validation labels

- PASS: preflight (8/8 inputs), venv/versions, protected hash.
- PASS: --vram-cap-gib implementation + 6/6 regression checks (exit 0), py_compile, pyflakes.
- PASS: left-pad batch == sequential (pad-check).
- PASS: HF-arm CPU smoke exit 0, raw completions captured, no EOS leak.
- PASS: two-arm A/B plumbing exit 0 (byte 4/4 vs smol 0/4 with EOS rates).
- NOT RUN: full 71-task A/B (GPU busy) — command ready in section 6.
- UNVERIFIED on CPU: real CUDA cap enforcement; full-scale batch-8 GPU generation; LoRA arm.

## 6. Ready-to-run GPU command (full A/B)

The GPU is currently busy; do NOT launch while any repo train.py / sft.py is active
(AGENTS section 4.6). scratch/bench_hf.py is the runner for both arms; the byte arm
is the E-122h 64/71 checkpoint.

```powershell
# ---- full A/B: base SmolLM2 vs our byte model on the 71 core tasks ----
foreach ($p in @(
  'scratch/bench_hf.py',
  'models/hf/SmolLM2-135M/model.safetensors',
  'glm-5.3-flash-from-scratch/runs/e122h_eos/checkpoint-0800/model.pt'
)) { if (-not (Test-Path $p)) { throw "missing $p" } }

# re-check free VRAM immediately before launch (TOCTOU); need headroom for both models
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

& .\.venv\Scripts\python.exe scratch/bench_hf.py --arm "byte=glm:glm-5.3-flash-from-scratch/runs/e122h_eos/checkpoint-0800" --arm "smol=hf:models/hf/SmolLM2-135M" --split dev --tier core --per-rung 1 --budget-bytes 160 --max-tokens 256 --batch-size 8 --device cuda --out research/raw/smollm2_vs_byte_core71.json

# ---- after LoRA SFT: the pre-registered H_data/H_pretraining arm (path doc section 7) ----
& .\.venv\Scripts\python.exe scratch/bench_hf.py --arm "byte=glm:glm-5.3-flash-from-scratch/runs/e122h_eos/checkpoint-0800" --arm "smollm2_lora=hf:runs/sft_smollm2_pg/final" --split dev --tier core --per-rung 1 --budget-bytes 160 --max-tokens 256 --batch-size 8 --device cuda --out research/raw/smollm2_lora_vs_byte_core71.json

# ---- the trainer, now with the section-20 ceiling (3.0 GiB mirrors 60% of ~5 GiB free) ----
& .\.venv\Scripts\python.exe scripts/sft.py --config configs/sft_smollm2_pg.yaml --pilot --vram-cap-gib 3.0
& .\.venv\Scripts\python.exe scripts/sft.py --config configs/sft_smollm2_pg.yaml --vram-cap-gib 3.0
```

Read strict/71, extracted/71 **and** eos_rate per arm from the JSON; compare against
the byte model's 64/71 strict. The 71-task comparison is only valid because both
arms run the same playground_pool('core') entries, split, budget and max-tokens.

## 7. What is still UNVERIFIED on CPU (honest)

1. **Real CUDA cap enforcement.** set_per_process_memory_fraction was never called
   against a physical device; the unit test monkeypatches torch.cuda. The fraction
   math and the ordering (before the model reaches the GPU) are tested; actual
   enforcement and the no-shared-memory-spill guarantee (AGENTS 20.3) must be
   confirmed on the GPU with torch.cuda.max_memory_allocated.
2. **Full 71-task A/B.** Only 8 tasks (HF base) and 4 tasks (two arms) ran on CPU.
   No 64/71 vs SmolLM2-LoRA comparison exists.
3. **The LoRA arm does not exist yet** — runs/sft_smollm2_pg is not created (GPU
   busy). The H_data/H_pretraining test in the path doc section 7 is unrun.
4. **Batch-8 / fp16 GPU generation.** The pad-check used 3 short prompts and
   budget=64; full-scale batch 8 at 160/256 with fp16 weights is unverified. The
   fp16 dtype load path is also unverified on hardware (sft.py already uses fp16,
   so the risk is low).
5. **--no-repeat-ngram > 0 with left padding.** The benchmark default is 0 and the
   pad-check used 0. With ngram > 0, HF's ngram processor sees the left-pad tokens
   (pad == eos) in the window; that path was not tested.
6. **Byte-budget trim asymmetry.** The byte arm stops at >= budget (may keep a token
   past it); the HF arm trims to <= budget. At 160 B vs a max reference body of
   139 B this cannot gate pass/fail, but it is a real difference.
7. **Base SmolLM2 numbers are not a capability result** (0/8, 0/4) — the base model
   is out-of-distribution for the instruction format.
8. **research/raw/ was not written** (the GPU runs were not started); the CPU smoke
   JSONs live under scratch/.
