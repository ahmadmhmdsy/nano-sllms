# SmolLM2 CPT corpus: re-tokenized corpus_codeparrot_v1_ex (example-level EOS)

**Date:** 2026-10-17 · **Status:** DATA PASS (all artifacts verified) · **Training:** NOT RUN (CPU-only brief; GPU held by a training job)
**Scope:** CPU only. Nothing under `runs/` was created or modified. No GPU command was issued.

Re-tokenizes `data/reference/corpus_codeparrot_v1_ex` (byte-packed, one EOS per example) into
SmolLM2-135M BPE ids at `data/reference/smollm2_corpus_ex/`, using the **exact method** of
`research/2026-10-15_smollm2_path.md` §3 (the playground corpus `data/reference/smollm2_pg_eos/`),
so the two corpora are tokenizer- and EOS-consistent.

## 0. TL;DR

| Task | Verdict | Evidence |
|---|---|---|
| 1 Re-tokenize train+val | **PASS** | 227,401 + 3,474 examples -> 99,194,717 + 1,560,281 BPE ids |
| 2 EOS proof by count | **PASS** | id-0 reread = 227,401 / 3,474 == examples; zero EOS in raw-text encoding (0/230,875 examples) |
| 3 Compression + sizing | **PASS** | 350,793,745 byte ids -> 100,754,998 BPE ids = **0.2872** (3.48 byte-ids/BPE id); 1 epoch = **3,028 steps** = 4.2-7.6 h at 5-9 s/step |
| 4 ctx-1024 block dataset | **PASS** | `ds/{train,val}` 96,870 / 1,524 rows, labels == input_ids, partial final row kept (861 / 729) |
| 5 Held-out val + integrity | **PASS** | strict held-out val preserved; bad_utf8 = **0** both splits; sha256 in §6 |
| 6 This document + CPT command | **PASS** | §5, §7; `scripts/sft.py` (no `peft` block) is the correct trainer |
| GPU preflight (`laya/scripts/preflight_check.py`) | **SKIPPED** | it includes a CUDA smoke; AGENTS §4.6 forbids GPU work while a train job is live |
| Training run | **SKIPPED** | CPU-only brief; command is ready in §7 |

## 1. Method (explicit)

Source byte scheme (from `corpus_codeparrot_v1_ex/verification.json` manifest): `id = 4 + utf8_byte`,
byte-scheme EOS `= 2` (below the byte offset), one id-2 per example, no interior EOS, stream ends with EOS.
The verifier reports `non_eos_ids_below_byte_offset = 0` per split, i.e. the only sub-offset id in the
stream is the EOS 2 itself (230,875 of them == 230,875 examples) - which is why splitting on id 2 recovers
exactly one example per EOS with nothing else consumed.

Per split, exactly the playground method (`scratch/smollm2_repack_pg.py`), promoted to
`scratch/smollm2_repack_corpus_ex.py` and made streaming/resumable at 100M-token scale:

1. `np.memmap(source, dtype='<u2')`; `positions = flatnonzero(a == 2)` -> example boundaries.
   Guard: stream must end with id 2 (**asserted**); no other id < 4 exists.
2. Decode each example `(a[s:e] - 4).astype(uint8).decode('utf-8')`, counting `bad_utf8_examples`
   (`errors='replace'` on failure - none occurred).
3. `tok(texts, add_special_tokens=False)` in batches of 500. **No `add_special_tokens=True`** anywhere,
   so the tokenizer adds no implicit BOS/EOS (SmolLM2 sets `add_bos_token=false`, `add_eos_token=false`;
   `encode("<|endoftext|>") = [0]`).
4. Append **exactly one** `eos_token_id = 0` per example, of that example's end.
5. Concatenate the whole split into a uint16 little-endian `.bin` (vocab 49,152 < 65,536), flushed in
   4M-token chunks with a `_progress_<split>.json` checkpoint (restart truncates the partial tail and
   continues), so a long encode can never lose work.

Tokenized control measured during encoding (every example, not a sample):
`eos_in_text_only_encoding = 0` for **both** splits - ordinary Python text never emits id 0 for this
tokenizer, which is what makes counting id 0 in the packed stream an unambiguous EOS count.

## 2. Counts (all re-read from disk, not writer logs)

| split | examples | byte ids before | BPE ids after | ids/example | .bin bytes | EOS written | **EOS reread from .bin** | EOS from Arrow ds | EOS == examples |
|---|---|---|---|---|---|---|---|---|---|
| train | 227,401 | 345,354,526 | 99,194,717 | 436.21 incl. EOS (content-only min/mean/max 1 / 435.211 / 509,702) | 198,389,434 | 227,401 | **227,401** | **227,401** | **true** |
| val | 3,474 | 5,439,219 | 1,560,281 | 449.13 incl. EOS (content-only mean 448.131) | 3,120,562 | 3,474 | **3,474** | **3,474** | **true** |
| **total** | **230,875** | **350,793,745** | **100,754,998** | 436.4 | 201,509,996 | 230,875 | **230,875** | **230,875** | **true** |

Source EOS (for cross-check): `corpus_codeparrot_v1_ex/verification.json` reports
`eos_count = 227,401 / 3,474` and `eos_equals_derived_examples = true`, and `no_interior_eos = true`.
230,875 examples x 1 EOS -> 230,875. The manifest's split mix (94,092 function / 76,036 class / 60,747 module)
is preserved untouched (this stage only re-encodes; it never re-splits).

## 3. EOS proof (task 2) - counted, never assumed

This is the exact failure class of E-122h (`MEMORY.md` line 1879: an example-level EOS that was never
actually in the training corpus; the no-EOS arm scored 0/71 vs the EOS arm 64/71, while final losses were
indistinguishable at 0.174935 vs 0.174486). Three independent counts, plus a zero-count control:

1. **Control (raw text):** summing id 0 over the tokenization of every example's text *without* the
   appended EOS gives **0 for train and 0 for val**. Ordinary text produces no id-0 token.
2. **From the written .bin:** `np.fromfile(...).count_nonzero(a == 0)` = **227,401 (train) / 3,474 (val)**.
3. **From the materialized Arrow dataset:** `pc.sum(pc.equal(flatten(input_ids), 0))` = **227,401 / 3,474**.
4. **Equals the example count:** 227,401 == 227,401 and 3,474 == 3,474 (and 230,875 == 230,875 total),
   which also equals the source corpus's own EOS count.

Train+val EOS total = **230,875**, exactly one per example at the end of each example. Nothing was
truncated or re-padded after tokenization.

## 4. Compression ratio and CPT sizing (task 3)

- **Ratio (byte ids -> BPE ids):** 100,754,998 / 350,793,745 = **0.287220** overall
  (train 0.287226, val 0.286858) = **3.482 byte-level ids per BPE id**.
  Against raw UTF-8 source bytes: 100,754,998 / 350,062,679 = 0.287820.
  (Consistent with the playground conversion's 0.2902 - this corpus is bulkier Python, slightly more
  compressible per available byte since the same vocabulary merges multi-byte identifiers.)
- **Total BPE tokens:** 99,194,717 (train) + 1,560,281 (val) = **100,754,998**.

**One epoch (ctx 1024, batch 4, accum 8 -> 32,768 tokens/step):**

- train rows 96,870 -> steps = ceil(96,870 / 32) = **3,028 optimizer steps**.
- at the observed 5-9 s/step: 3,028 x 5 s = **15,140 s = 4.21 h**; x 7 s = **5.89 h**; x 9 s = **7.57 h**
  -> **~4.2-7.6 h per epoch** (5-9 s/step pace given in the brief).
- eval: val = 1,524 rows -> 381 forward batches of 4 per evaluation; with `eval_steps: 500` that is
  **6 evaluations per epoch**.

## 5. Block dataset (task 4) and trainer verdict (task 6)

**Materialization** (`scratch/smollm2_make_dataset_ex.py`, CPU, memory-safe at this scale): the id stream
is exposed as a zero-copy pyarrow `ListArray` (int32 values + int32 offsets), `labels == input_ids`
(the same Arrow array), ctx = 1024, and the **partial final row is kept**:

| split | rows | tokens used | final row length | EOS in used tokens |
|---|---|---|---|---|
| train | 96,870 | 99,194,717 | **861** | 227,401 |
| val | 1,524 | 1,560,281 | **729** | 3,474 |

The partial final row was kept deliberately: a whole-1024-block slice would drop the last 861/729 tokens,
including the tail EOS (the playground pass lost 8 train + 9 val EOS this way, `research/2026-10-15_smollm2_path.md` §3).
Verifier confirms `partial_final_row_kept = true` and the final row's last token is 0 in both splits.
Feature type is `List(Value('int32'))`; `SftCollator` casts to `torch.int64` (CPU dry check: batch shapes
(4, 1024) for input_ids/labels/attention_mask).

**Strict held-out val:** the val split is the source corpus's own held-out `val.bin` (3474 examples from
500 held-out docs, built by the corpus pipeline) and is never mixed into train; eval uses it unchanged.

**Which trainer is correct for CPT?** `scripts/sft.py`, configured with **no `peft` block**.

- `scripts/train.py` is NOT usable here: it trains the repo's own byte-tokenizer decoder from `src/model.py`
  over memmap packed shards (`src/data.py`), with `ByteTokenizer`; it cannot load an HF `LlamaForCausalLM`.
  See `research/2026-10-15_smollm2_path.md` §4 for the full feasibility analysis.
- `scripts/sft.py` is model-agnostic (`AutoModelForCausalLM.from_pretrained`, HF `Trainer`) and, despite its
  name, is a generic HF causal-LM trainer: with `labels == input_ids` and `SftCollator`'s positional
  padding mask (-100), the loss is the plain next-token CLM loss over raw code - i.e. continued pretraining.
- Full-parameter CPT (the standard mode for raw-code CPT) is selected by **omitting the `peft` block**:
  `src/model.py:175` `maybe_wrap_peft()` returns the model unchanged when `cfg["peft"]` is absent
  (verified on CPU: `maybe_wrap_peft(dummy, cfg) is dummy -> True`). Adding a `peft` block would switch to
  LoRA - the OOM fallback, not the default.
- Config: `configs/cpt_smollm2_ex.yaml` (name `cpt_smollm2_ex`, lr 3e-5, 1 epoch, ctx 1024, batch 4,
  accum 8, warmup 100, fp16, grad ckpt, eval/save every 500, `save_total_limit: 3`).
- Honest caveats for the launch operator:
  1. `scripts/sft.py` estimates full-FT memory at ~2.0-3.0 GiB for 135M @ 1024x4 fp16 with grad-ckpt
     (weights 0.27 GiB + fp16 grads 0.27 GiB + AdamW fp32 states 1.08 GiB + activations/logits).
     Pass `--vram-cap-gib 3.5` (§20 allocator ceiling) so a spill can never happen; if it OOMs, add a
     `peft` block (r=16 q/k/v/o) rather than raising batch size.
  2. `load_best_model_at_end=True` is hardcoded in `scripts/sft.py`, so eval must run; `eval_steps: 500`
     bounds that to ~6 evals/epoch.
  3. `--pilot` writes `runs/cpt_smollm2_ex_pilot` (isolated from the full-run tree, so auto-resume cannot
     pick up pilot checkpoints).

## 6. Paths, sha256, integrity (task 5)

```
data/reference/smollm2_corpus_ex/
  train.bin            198,389,434 B  sha256 7af7110ca15918651dc842bea16025f98e5d2930fc7b7dc1e7e5f8fd55336259
  val.bin                3,120,562 B  sha256 288e3610e519befdd80b4ce027d6e944b1b64d57564edcee338fd4b94aa99e93
  meta.json                  1,577 B  sha256 36e878d7d4c589623032b7471250a204f35adb5bd4c1cb3d6745f2536738865c
  verification.json          2,586 B  (pass = true)
  ds/manifest.json             786 B  sha256 c0e8114e6b4f5fe601cd47c33842986fb3e239b0f37aeeda85e2574b6b9d0caf
  ds/train/data-00000-of-00002.arrow  397,182,416 B  sha256 11ec44a8bc3b3171ba15033ee9f72fea211e38c43c169f7265818b17a73da969
  ds/train/data-00001-of-00002.arrow  397,181,120 B  sha256 0d81de5e6d3d2d276cee522d608d7ca4b2f1d7ea104d39cc8072955c590e638e
  ds/val/data-00000-of-00001.arrow     12,495,584 B  sha256 a4ae87c6f81eca5b03ee129d7dd31313cd2414c1af4562b7baa4adc4eaf6e292
```

- `bad_utf8_examples = 0` (train) and **0** (val) - decoded bytes re-encode losslessly to UTF-8.
- All numbers above are re-read by `scratch/smollm2_verify_corpus_ex.py` (`verification.json`, `pass = true`):
  bin <-> Arrow token counts equal, EOS counts equal, `labels_equal_input_ids_full = true`, partial final
  row lengths match the arithmetic (861 / 729), zero raw-text EOS.
- Disk: E: free went 43,368,030,208 B -> 42,359,603,200 B across the run; the new corpus is **1,008,375,774 B** (14 files: 201,509,996 B of .bin + 806,859,120 B of Arrow + metadata/hashes).

## 7. Ready-to-run CPT command (forward slashes + Test-Path preflight)

CPU preflight is done; the GPU half is deliberately not run (a training job holds the GPU). Run this when
the GPU frees, in this exact order:

```powershell
# 0) input preflight - fail loudly before anything else
foreach ($p in @(
  'scripts/sft.py',
  'configs/cpt_smollm2_ex.yaml',
  'models/hf/SmolLM2-135M/model.safetensors',
  'models/hf/SmolLM2-135M/tokenizer.json',
  'data/reference/smollm2_corpus_ex/train.bin',
  'data/reference/smollm2_corpus_ex/val.bin',
  'data/reference/smollm2_corpus_ex/ds/train',
  'data/reference/smollm2_corpus_ex/ds/val'
)) { if (-not (Test-Path $p)) { throw "missing $p" } }

# 1) static gate (dup keys, pyflakes, --help, data, GPU smoke) - run only when no train job is live
& .\.venv\Scripts\python.exe laya/scripts/preflight_check.py --script scripts/sft.py --config configs/cpt_smollm2_ex.yaml

# 2) re-check free VRAM immediately before launch (single shared GPU)
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

# 3) pilot first (writes runs/cpt_smollm2_ex_pilot; 200 rows, 1 epoch)
& .\.venv\Scripts\python.exe scripts/sft.py --config configs/cpt_smollm2_ex.yaml --pilot --vram-cap-gib 3.5

# 4) full run (writes runs/cpt_smollm2_ex; auto-resumes from checkpoint-* on any restart)
& .\.venv\Scripts\python.exe scripts/sft.py --config configs/cpt_smollm2_ex.yaml --vram-cap-gib 3.5
```

Expected full-run shape: 3,028 steps, 6 evals, ~4.2-7.6 h at 5-9 s/step. TensorBoard events are the loss
source of truth; `runs/cpt_smollm2_ex/final/train_summary.json` holds the final metrics.

## 8. Commands run (exact, with exit codes)

```powershell
& .\.venv\Scripts\python.exe scratch/smollm2_repack_corpus_ex.py --splits val      # exit 0
& .\.venv\Scripts\python.exe scratch/smollm2_repack_corpus_ex.py --splits train    # background job pwsh-18, exit 0
& .\.venv\Scripts\python.exe scratch/smollm2_make_dataset_ex.py --splits val       # exit 0
& .\.venv\Scripts\python.exe scratch/smollm2_make_dataset_ex.py --splits train     # exit 0
& .\.venv\Scripts\python.exe scratch/smollm2_verify_corpus_ex.py                  # exit 0, PASS = True
& .\.venv\Scripts\python.exe scratch/smollm2_cpt_cpu_check.py                      # exit 0
```

Helpers: `scratch/smollm2_repack_corpus_ex.py` (resumable re-tokenizer, promoted from the playground
`smollm2_repack_pg.py`), `scratch/smollm2_make_dataset_ex.py`, `scratch/smollm2_verify_corpus_ex.py`,
`scratch/smollm2_cpt_cpu_check.py`. `scratch/` is non-durable by repo convention; promote to `scripts/`
if this preprocessing becomes a recurring pipeline stage.

## 9. Limitations / honest notes

- **SKIPPED:** `laya/scripts/preflight_check.py` - its GPU smoke would touch the shared GPU while a train
  job is live (AGENTS §4.6). Run it as step 1 of §7 when the GPU is free.
- **SKIPPED:** the CPT run itself (CPU-only brief), and therefore no loss/curve claims here.
- No `research/EXPERIMENTS.md` row: this task produced a verified data asset, not a training experiment.
- The tokenizer's `model_max_length` warning (e.g. 14206 > 8192) appears during batch encoding; it is
  informational only - the long examples are split into ctx-1024 blocks downstream.
- `data/reference/smollm2_corpus_ex/_progress_*.json` are resume checkpoints kept for restartability; they
  are not inputs (delete only after confirming the final `verification.json pass = true`).
- LoRA vs full-FT is left as a launch decision driven by the §7 step-3 pilot; both paths use this exact
  dataset, so the data asset does not need re-tokenizing either way.
