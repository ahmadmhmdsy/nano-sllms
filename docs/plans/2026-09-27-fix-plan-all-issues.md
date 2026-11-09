# nano_SLMs Fix Plan — All Issues from Project Review

**Created:** 2026-09-27
**Source:** Full project review (18 issues: 2 critical, 5 high, 8 medium, 3 low)
**Execution model:** 6 parallel sub-agent groups, then integration validation

---

## Execution Overview

| Group | Name | Issues | Files | Dependencies |
|-------|------|--------|-------|--------------|
| G1 | Train Pipeline Hardening | #1, #3, #4, #7, #12, #13, #14, #15, #17 | `scripts/train.py` | None |
| G2 | Eval Decoupling | #2, #18 | `scripts/eval.py` | None |
| G3 | Data Pipeline Quality | #5, #6, #8, #11 | `scripts/prepare_data.py`, `scripts/tokenize_data.py` | None |
| G4 | Config Modernization | #9 | `configs/*.yaml` (all) | None |
| G5 | GDN Performance | #10 | `src/gdn.py` | None |
| G6 | Import Robustness | #16 | `src/model.py` | None |

**All 6 groups are independent and can run in parallel.**

---

## G1 — Train Pipeline Hardening

**File:** `scripts/train.py`
**Issues:** #1 (critical), #3, #4, #7, #12, #13, #14, #15, #17

### Task 1.1 — Fix RemoveColumnsCollator bug (CRITICAL #1)

Add to `TrainingArguments` constructor:
```python
remove_unused_columns=False,
label_names=['input_ids'],
```

**Why:** MEMORY.md lesson #40 documents that RemoveColumnsCollator strips `labels` from plain torch Datasets, causing `loss=None` during eval, which breaks `load_best_model_at_end` and `metric_for_best_model="eval_loss"`. The PackedDataset returns `{"input_ids": ..., "labels": ...}` where labels == input_ids.

**Validation:** Run `sanity_check.py --config configs/smoke.yaml` — must show PASS on all 4 checks. Then run a 50-step training and verify `eval_loss` appears in TensorBoard events.

### Task 1.2 — Add disk space check (HIGH #3)

Before training starts, check available disk space:
```python
import shutil
free_gb = shutil.disk_usage(ROOT).free / (1024**3)
estimated_need_gb = int(t["save_total_limit"]) * (n_params * 6 / 1e9) + 2.0
if free_gb < estimated_need_gb:
    raise SystemExit(f"[train] LOW DISK: {free_gb:.1f} GB free, need ~{estimated_need_gb:.1f} GB")
```

**Validation:** Test with a config that has a very high `save_total_limit` to trigger the check.

### Task 1.3 — Add config compatibility check on resume (HIGH #4)

When `resume_from` is not None, load the checkpoint's `config.json` and compare key fields:
```python
if resume_from is not None:
    ckpt_config = json.loads((resume_from / "config.json").read_text())
    current_config = model.config.to_dict()
    for key in ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]:
        if ckpt_config.get(key) != current_config.get(key):
            raise SystemExit(f"[train] CONFIG MISMATCH on resume: {key} differs")
```

**Validation:** Change a model dim in the config, try to resume — should fail with clear error.

### Task 1.4 — Add early stopping (HIGH #7)

Add `EarlyStoppingCallback` to the Trainer:
```python
from transformers import EarlyStoppingCallback
# In Trainer constructor:
callbacks=[CoopStopCallback(), EarlyStoppingCallback(early_stopping_patience=3)],
```

Add to config:
```yaml
train:
  early_stopping_patience: 3
```

**Validation:** Run a short training with a very high patience to verify it does not trigger, then a low patience to verify it does.

### Task 1.5 — Add model summary at training start (MEDIUM #12)

After building the model, print a summary:
```python
print(f"[train] model summary:")
print(f"  layers={model.config.num_hidden_layers} hidden={model.config.hidden_size}")
print(f"  heads={model.config.num_attention_heads} kv_heads={model.config.num_key_value_heads}")
print(f"  ffn={model.config.intermediate_size} ctx={model.config.max_position_embeddings}")
print(f"  vocab={model.config.vocab_size} tie={model.config.tie_word_embeddings}")
print(f"  params={n_params/1e6:.1f}M")
```

**Validation:** Run `sanity_check.py` — summary should appear in output.

### Task 1.6 — Add DataLoader seed (MEDIUM #13)

Set a seed for the DataLoader in `TrainingArguments`:
```python
data_seed=int(t.get("seed", 42)),
```

**Validation:** Run two short trainings with the same seed — data order should be identical.

### Task 1.7 — Add checkpoint verification after save (MEDIUM #14)

After `trainer.train()`, verify the final checkpoint can be loaded:
```python
from safetensors.torch import load_file
final_ckpt = trainer.state.best_model_checkpoint
if final_ckpt:
    shards = list(Path(final_ckpt).glob("*.safetensors"))
    for s in shards:
        _ = load_file(str(s))
    print(f"[train] checkpoint verification: {len(shards)} shards OK")
```

**Validation:** Run a short training — verification message should appear.

### Task 1.8 — Add training metrics export (MEDIUM #15)

At the end of training, export metrics to JSON:
```python
metrics = trainer.state.log_history
(final_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
```

**Validation:** Run a short training — `training_metrics.json` should exist in the final dir.

### Task 1.9 — Add OOM recovery suggestion (LOW #17)

Wrap the training loop in a try/except for OOM:
```python
try:
    trainer.train(...)
except torch.cuda.OutOfMemoryError:
    print("[train] OOM ERROR: Try reducing batch size, or set optim=adamw_bnb_8bit in config")
    raise
```

**Validation:** Code review — the try/except should be present.

---

## G2 — Eval Decoupling

**File:** `scripts/eval.py`
**Issues:** #2 (critical), #18

### Task 2.1 — Remove hard dependency on sft_data (CRITICAL #2)

Move the import from module level to inside the conditional block:

**Remove from top of file:**
```python
from sft_data import extract_code, parses_as_python
```

**Add inside the `if not args.skip_gen and instr_rel and template:` block:**
```python
from sft_data import extract_code, parses_as_python
```

**Why:** The import is only needed for SFT instruction eval. Having it at module level means eval.py cannot run without sft_data.py being importable.

**Validation:** Run `eval.py --config configs/smoke.yaml --skip-gen` — should work without sft_data.py.

### Task 2.2 — Fix perplexity clamp (LOW #18)

Change the perplexity calculation to report both clamped and raw values:
```python
ppl_raw = math.exp(avg_nll) if avg_nll < 20 else float('inf')
ppl = math.exp(min(avg_nll, 20))
print(f"[eval] val_loss={avg_nll:.4f} perplexity={ppl:.2f} (raw={ppl_raw})")
```

**Validation:** Run eval on an untrained model — should show `raw=inf` instead of a clamped number.

---

## G3 — Data Pipeline Quality

**Files:** `scripts/prepare_data.py`, `scripts/tokenize_data.py`
**Issues:** #5, #6, #8, #11

### Task 3.1 — Randomize val split (MEDIUM #8)

In `prepare_data.py`, change the val split logic. Instead of "last N rows", use a random sample.

**Why:** The current "last N rows" approach can produce unrepresentative val sets if the data has any ordering.

**Validation:** Run `prepare_data.py` on a small dataset — val rows should be randomly distributed, not the last N.

### Task 3.2 — Add shuffle before token packing (HIGH #6)

In `tokenize_data.py`, add an option to shuffle the token buffer before writing shards.

**Why:** If the raw data has any ordering, consecutive blocks will be correlated.

**Validation:** Run `tokenize_data.py` with and without the flag — block contents should differ.

### Task 3.3 — Add data quality filtering (HIGH #5)

In `prepare_data.py`, add optional quality filters: `max_chars`, `require_code_keywords`.

**Why:** Training on low-quality or non-code data wastes compute.

**Validation:** Run `prepare_data.py` with `require_code_keywords: true` — should filter out non-code rows.

### Task 3.4 — Add progress bars (MEDIUM #11)

Add `tqdm` progress bars to `prepare_data.py` and `tokenize_data.py`.

**Why:** For large datasets (pilot: 50k rows, target: 150k rows), the user has no visibility into progress.

**Validation:** Run `prepare_data.py` on a small dataset — progress bar should appear.

---

## G4 — Config Modernization

**Files:** `configs/smoke.yaml`, `configs/pilot.yaml`, `configs/target.yaml`, `configs/sft_t1.yaml`
**Issue:** #9

### Task 4.1 — Increase dataloader_num_workers (MEDIUM #9)

Change `dataloader_num_workers: 0` to `dataloader_num_workers: 2` in all configs.

**Why:** With memmap shards, data loading should be fast, but multiple workers can still help.

**Validation:** Run `sanity_check.py` with the new config — should still pass. Run a short training and compare tokens/s.

---

## G5 — GDN Performance

**File:** `src/gdn.py`
**Issue:** #10

### Task 5.1 — Investigate batching the chunk loop (MEDIUM #10)

The current implementation uses a Python for loop over chunks. This is a performance weakness for long sequences.

**Approach:** This is a complex change that requires careful validation. The sub-agent should:
1. Profile the current implementation to confirm the bottleneck
2. Investigate whether the chunk loop can be batched (e.g., using `torch.vmap` or manual batching)
3. If batching is not possible without changing the math, document the limitation and suggest alternatives (e.g., `torch.compile`)

**Why:** For ctx 1024 = 32 chunks, the Python loop is a bottleneck.

**Validation:** Run `scripts/test_gdn_math.py` — all tests must still pass. Benchmark before/after to measure improvement.

**Note:** This task may result in "cannot fix without changing the math" — that is an acceptable outcome. The goal is to investigate and document.

---

## G6 — Import Robustness

**File:** `src/model.py`
**Issue:** #16

### Task 6.1 — Make strict import more robust (LOW #16)

Change the import to try multiple locations:
```python
try:
    from transformers.configuration_utils import strict
except ImportError:
    from huggingface_hub.dataclasses import strict
```

**Why:** The current import from `huggingface_hub.dataclasses` works but is fragile. `transformers.configuration_utils` is the more stable location.

**Validation:** Run `sanity_check.py` — should still pass.

---

## Integration Validation

After all 6 groups complete, run the following integration tests:

### Test 1: Sanity Check
```powershell
& .\.venv\Scripts\python.exe scripts\sanity_check.py --config configs\smoke.yaml
```
**Expected:** All 4 checks PASS.

### Test 2: Short Training Run
```powershell
& .\.venv\Scripts\python.exe scripts\train.py --config configs\smoke.yaml
```
**Expected:**
- Model summary printed at start
- Training completes without errors
- `eval_loss` appears in TensorBoard events (verifies Fix #1)
- `training_metrics.json` exists in final dir (verifies Fix #15)
- Checkpoint verification message appears (verifies Fix #14)

### Test 3: Eval
```powershell
& .\.venv\Scripts\python.exe scripts\eval.py --config configs\smoke.yaml --skip-gen
```
**Expected:** Eval runs without sft_data.py (verifies Fix #2).

### Test 4: Data Pipeline
```powershell
& .\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\smoke.yaml
& .\.venv\Scripts\python.exe scripts\tokenize_data.py --config configs\smoke.yaml
```
**Expected:** Progress bars appear (verifies Fix #11), val split is randomized (verifies Fix #8).

### Test 5: Auto-Resume
```powershell
# Start training, kill after 50 steps, restart
& .\.venv\Scripts\python.exe scripts\train.py --config configs\smoke.yaml
# Kill the process, then restart
& .\.venv\Scripts\python.exe scripts\train.py --config configs\smoke.yaml
```
**Expected:** Training resumes from checkpoint-50 without errors.

---

## Risk Assessment

| Group | Risk | Mitigation |
|-------|------|------------|
| G1 | High — touches core training path | Run full validation after each task |
| G2 | Low — isolated to eval.py | Test with existing checkpoints |
| G3 | Medium — changes data pipeline | Test on small dataset first |
| G4 | Low — config-only change | Verify with sanity_check |
| G5 | Medium — complex math change | Run test_gdn_math.py after |
| G6 | Low — import-only change | Verify with sanity_check |

---

## File Ownership Matrix

To avoid conflicts, each group owns exclusive write access to its files:

| Group | Files |
|-------|-------|
| G1 | `scripts/train.py` |
| G2 | `scripts/eval.py` |
| G3 | `scripts/prepare_data.py`, `scripts/tokenize_data.py` |
| G4 | `configs/smoke.yaml`, `configs/pilot.yaml`, `configs/target.yaml`, `configs/sft_t1.yaml` |
| G5 | `src/gdn.py` |
| G6 | `src/model.py` |

**No file is owned by more than one group.**

---

## Success Criteria

The plan is complete when:
1. All 6 groups have executed their tasks
2. All 5 integration tests pass
3. No regressions in existing functionality
4. All fixes are documented in code comments
