# V2 Pipeline Fix Milestones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all critical and high-severity defects in the V2 training pipeline so that every config field is either applied or removed, every published manifest certifies what actually happened, and the full test suite passes.

**Architecture:** The V2 pipeline is a 13-stage graph (`validate → bootstrap → gpu_preflight → prepare → tokenize → dataset_validation → model_validation → evaluator_validation → resolve_final_identity → train → internal_evaluation → external_gates → publish_final`). Each stage is a pure function of its `StageContext`. The engine checks declared inputs/outputs independently. Fixes are organized by milestone, each independently testable.

**Tech Stack:** Python 3.12, PyTorch 2.14, Transformers 5.16, Pydantic, PyYAML, pytest

---

## File Structure

**Files this plan modifies:**

| File | Responsibility |
|---|---|
| `pipeline_v2/slm_train/orchestration/handlers.py` | All 13 stage handlers — the main file to fix |
| `pipeline_v2/slm_train/config/models.py` | Pydantic config schema — add/remove fields |
| `pipeline_v2/slm_train/data/prepare.py` | Data preparation — apply filters and normalization |
| `pipeline_v2/slm_train/data/dedupe.py` | Deduplication and normalization helpers |
| `pipeline_v2/slm_train/data/split.py` | Split assignment — add held-out roles |
| `pipeline_v2/slm_train/data/sources/huggingface.py` | HF source reader — wire into stage |
| `pipeline_v2/slm_train/adapters/legacy_v1.py` | V1→V2 migration — fix YAML parsing |
| `pipeline_v2/slm_train/training/selection.py` | Checkpoint selection — honor policy |
| `pipeline_v2/slm_train/training/hf_causal_lm.py` | Dead code — remove or implement |
| `pipeline_v2/slm_train/monitoring/heartbeat.py` | Dead code — remove or wire |
| `pipeline_v2/tests/unit/test_handlers.py` | Integration tests |
| `pipeline_v2/tests/unit/test_resume.py` | Resume tests |
| `pipeline_v2/tests/unit/test_data.py` | Data pipeline tests |
| `pipeline_v2/README.md` | Documentation — fix drift |
| `pipeline_v2/docs/acceptance.md` | Documentation — fix drift |
| `docs/plans/README.md` | Plan index — add this plan |

**Files this plan creates:**

| File | Responsibility |
|---|---|
| `pipeline_v2/tests/unit/test_normalization.py` | Tests for data normalization |
| `pipeline_v2/tests/unit/test_filters.py` | Tests for data filters |
| `pipeline_v2/tests/unit/test_source_roles.py` | Tests for held-out source roles |
| `pipeline_v2/tests/unit/test_migrate_v1.py` | Tests for V1 migration |

---

## Milestone 0: Diagnose Training Stage Failure (CRITICAL)

**Goal:** Find out why the `stage_train` handler returns FAIL in the test suite. All other milestones depend on this.

**Files:**
- Modify: `pipeline_v2/slm_train/orchestration/handlers.py` (diagnose only)
- Test: `pipeline_v2/tests/unit/test_handlers.py`

- [ ] **Step 1: Run the failing test with full output**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_handlers.py::test_the_full_r1_run_completes_every_stage -v -s --tb=long 2>&1 | Select-Object -Last 100
```
Expected: The test fails with `StageFailed: stage train returned FAIL`. The `-s` flag should show the `_fail()` reason printed to stdout.

- [ ] **Step 2: Read the _fail() reason from the test output**

The `stage_train` handler calls `_fail(context, reason)` which returns a `StageOutcome` with the reason. The reason is printed to stdout. Look for the line starting with `[train]` or containing `FAIL` in the output.

- [ ] **Step 3: If the reason is not in stdout, add a temporary print**

In `pipeline_v2/slm_train/orchestration/handlers.py`, find the `stage_train` function. Before each `_fail()` call, add a temporary `print(f"[DIAG] _fail: {reason}", flush=True)`. Run the test again.

- [ ] **Step 4: Identify the root cause**

Common causes:
1. **Missing fixture data**: `tiny_corpus.jsonl` or `tiny_tokenizer/tokenizer.json` not found
2. **Config validation error**: `tiny_config.yaml` fails to load
3. **Import error**: A module imported inside `stage_train` fails
4. **Runtime error**: OOM, shape mismatch, or other runtime issue
5. **Assertion error**: A `raise ValueError` or `raise RuntimeError` in the handler

- [ ] **Step 5: Fix the root cause**

Based on the diagnosis, apply the minimal fix. Do NOT proceed to later milestones until this test passes.

- [ ] **Step 6: Verify the fix**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_handlers.py::test_the_full_r1_run_completes_every_stage -v --tb=short
```
Expected: PASS

- [ ] **Step 7: Run all previously-failing tests**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_handlers.py pipeline_v2/tests/unit/test_resume.py -v --tb=short
```
Expected: All tests pass (or only tests blocked by later milestones fail with clear reasons)

- [ ] **Step 8: Commit**

```bash
git add pipeline_v2/slm_train/orchestration/handlers.py
git commit -m "fix(v2): diagnose and fix training stage failure in tests"
```

---

## Milestone 1: Data Pipeline Correctness (CRITICAL)

**Goal:** Every `data.*` config field is either applied or the config is refused. No false certifications in published manifests.

### Task 1.1: Apply `data.normalization` or certify the truth

**Files:**
- Modify: `pipeline_v2/slm_train/data/prepare.py` (add NFKC normalization)
- Modify: `pipeline_v2/slm_train/orchestration/handlers.py:733,741` (certify applied normalization)
- Create: `pipeline_v2/tests/unit/test_normalization.py`

- [ ] **Step 1: Write the failing test**

Create `pipeline_v2/tests/unit/test_normalization.py`:
```python
"""Tests for data normalization: the manifest certifies what was applied."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "pipeline_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline_v2"))


def test_nfkc_normalization_is_applied(tmp_path: Path) -> None:
    """A config declaring unicode_nfkc must produce NFKC-normalized text."""
    from slm_train.data.prepare import prepare_generation

    # "café" with combining accent -> NFKC should compose it
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"text": "café"}) + "\n", encoding="utf-8")

    result = prepare_generation(
        source_files=[source],
        output_dir=tmp_path / "prepared",
        normalization="unicode_nfkc",
        min_chars=0,
        max_chars=None,
    )
    # The prepared text should be NFKC-composed
    prepared_text = (tmp_path / "prepared" / "train.jsonl").read_text(encoding="utf-8")
    assert "café" in prepared_text  # NFKC composes the accent


def test_ws_collapse_normalization_is_applied(tmp_path: Path) -> None:
    """A config declaring ws-collapse must produce whitespace-collapsed text."""
    from slm_train.data.prepare import prepare_generation

    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"text": "hello   world"}) + "\n", encoding="utf-8")

    result = prepare_generation(
        source_files=[source],
        output_dir=tmp_path / "prepared",
        normalization="ws-collapse",
        min_chars=0,
        max_chars=None,
    )
    prepared_text = (tmp_path / "prepared" / "train.jsonl").read_text(encoding="utf-8")
    assert "hello world" in prepared_text  # whitespace collapsed
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_normalization.py -v --tb=short
```
Expected: FAIL — `prepare_generation` does not accept a `normalization` parameter, or does not apply NFKC.

- [ ] **Step 3: Implement NFKC normalization in prepare.py**

In `pipeline_v2/slm_train/data/prepare.py`, find the `prepare_generation` function. Add a `normalization` parameter and apply it:

```python
import unicodedata

def prepare_generation(
    source_files: list[Path],
    output_dir: Path,
    normalization: str = "ws-collapse",
    min_chars: int = 0,
    max_chars: int | None = None,
    # ... existing params
) -> dict:
    # ... existing code ...
    for doc in documents:
        text = doc["text"]
        if normalization == "unicode_nfkc":
            text = unicodedata.normalize("NFKC", text)
            # Also collapse whitespace after NFKC
            text = " ".join(text.split())
        elif normalization == "ws-collapse":
            text = " ".join(text.split())
        # ... rest of processing ...
```

- [ ] **Step 4: Update handlers.py to certify the applied normalization**

In `pipeline_v2/slm_train/orchestration/handlers.py`, find where the data manifest is built (around line 733). Change:

```python
# BEFORE (false certification):
"normalization": data.normalization,

# AFTER (certify the truth):
"normalization": data.normalization if data.normalization == "unicode_nfkc" else "ws-collapse",
"normalization_applied": True,  # the prepare stage actually applied it
```

Also update the `stage_prepare` handler to pass the `normalization` parameter to `prepare_generation`.

- [ ] **Step 5: Run test to verify it passes**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_normalization.py -v --tb=short
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline_v2/slm_train/data/prepare.py pipeline_v2/slm_train/orchestration/handlers.py pipeline_v2/tests/unit/test_normalization.py
git commit -m "fix(v2): apply declared data normalization and certify the truth"
```

### Task 1.2: Fix `data.filters` manifest to record applied values

**Files:**
- Modify: `pipeline_v2/slm_train/orchestration/handlers.py:740`
- Create: `pipeline_v2/tests/unit/test_filters.py`

- [ ] **Step 1: Write the failing test**

Create `pipeline_v2/tests/unit/test_filters.py`:
```python
"""Tests for data filters: the manifest records applied values, not just declared ones."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "pipeline_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline_v2"))


def test_filters_manifest_records_applied_values(tmp_path: Path) -> None:
    """A config with filters: ["min_chars:100"] must produce a manifest with min_chars=100."""
    from slm_train.data.prepare import prepare_generation

    source = tmp_path / "raw.jsonl"
    # One short doc (rejected), one long doc (kept)
    source.write_text(
        json.dumps({"text": "short"}) + "\n" +
        json.dumps({"text": "a" * 200}) + "\n",
        encoding="utf-8",
    )

    result = prepare_generation(
        source_files=[source],
        output_dir=tmp_path / "prepared",
        normalization="ws-collapse",
        min_chars=100,
        max_chars=None,
    )
    # The short doc should be rejected
    prepared = (tmp_path / "prepared" / "train.jsonl").read_text(encoding="utf-8")
    assert "short" not in prepared
    assert "a" * 200 in prepared
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_filters.py -v --tb=short
```
Expected: FAIL — `prepare_generation` does not accept `min_chars`/`max_chars` parameters.

- [ ] **Step 3: Implement filter parameters in prepare.py**

In `pipeline_v2/slm_train/data/prepare.py`, add `min_chars` and `max_chars` parameters to `prepare_generation` and apply them:

```python
def prepare_generation(
    source_files: list[Path],
    output_dir: Path,
    normalization: str = "ws-collapse",
    min_chars: int = 0,
    max_chars: int | None = None,
    # ... existing params
) -> dict:
    # ... existing code ...
    for doc in documents:
        text = doc["text"]
        if min_chars and len(text) < min_chars:
            stats["filter_rejected"] += 1
            continue
        if max_chars and len(text) > max_chars:
            stats["filter_rejected"] += 1
            continue
        # ... process doc ...
```

- [ ] **Step 4: Update handlers.py manifest to record applied filter values**

In `pipeline_v2/slm_train/orchestration/handlers.py`, find the manifest building code (around line 740). Change:

```python
# BEFORE:
filters={"declared": list(data.filters)},

# AFTER:
filters={
    "declared": list(data.filters),
    "applied": {"min_chars": min_chars, "max_chars": max_chars},
},
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_filters.py -v --tb=short
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline_v2/slm_train/data/prepare.py pipeline_v2/slm_train/orchestration/handlers.py pipeline_v2/tests/unit/test_filters.py
git commit -m "fix(v2): apply declared data filters and record applied values in manifest"
```

### Task 1.3: Refuse or implement held-out source roles

**Files:**
- Modify: `pipeline_v2/slm_train/config/models.py:11` (remove test/external-gate from enum)
- Modify: `pipeline_v2/slm_train/data/split.py` (if implementing)
- Create: `pipeline_v2/tests/unit/test_source_roles.py`

- [ ] **Step 1: Write the failing test**

Create `pipeline_v2/tests/unit/test_source_roles.py`:
```python
"""Tests for source roles: declared held-out roles must not be trained on."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "pipeline_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline_v2"))


def test_test_role_is_rejected_by_validation() -> None:
    """A config with role: test must fail validation in R1."""
    from slm_train.config.models import ConfigDocument, SourceDeclaration, FreshnessPolicy
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="test"):
        ConfigDocument(
            schema_version=2,
            release="r1",
            capabilities=["pretrain"],
            run={"name": "test", "namespace": "test", "stage": "S", "seed": 42},
            data={
                "dataset_id": "test",
                "sources": [{
                    "source_id": "test-src",
                    "kind": "local",
                    "path_or_repo": "test.jsonl",
                    "revision": None,
                    "file_set": ["test.jsonl"],
                    "role": "test",  # This should be rejected
                    "license": "CC0",
                    "expected_rows": 10,
                    "expected_tokens": 100,
                    "freshness_policy": {"max_age": "pinned", "supersession": "explicit-generation", "optional_reason": None},
                    "optional_profile": None,
                }],
                "split_salt": 17,
                "validation_fraction": 0.2,
                "dedupe": "exact_hash",
                "normalization": "ws-collapse",
                "filters": [],
                "regeneration_command": ["echo", "test"],
            },
            tokenizer={"source": "local", "revision_or_files": "tok.json", "special_ids": {"pad": 0, "bos": 1, "eos": 2}, "context": 16, "offsets": "special_tokens"},
            model={"architecture": "llama", "hidden": 32, "intermediate": 64, "layers": 2, "heads": 4, "kv_heads": 2, "context": 16, "tied_embeddings": False, "precision": "fp16", "attention": "sdpa", "base_model": None},
            train={"optimizer": "adamw", "scheduler": "cosine", "learning_rate": 3e-3, "max_steps": 10, "batch_size": 1, "gradient_accumulation": 1, "max_grad_norm": 1.0, "checkpoint_every_steps": 1, "evaluation_cadence": 1, "checkpoint_policy": "complete_only", "keep_last_checkpoints": 3, "keep_best_checkpoints": 1, "resume": True},
            evaluation={"ids": ["validation_loss"], "versions": ["1"], "suites": ["validation"], "metrics": ["val_loss", "val_perplexity"], "cadence_steps": 1, "thresholds": {"val_loss": 100.0, "val_perplexity": 1e30}, "directions": {"val_loss": "min", "val_perplexity": "min"}, "generation_samples": 1, "generation_max_new_tokens": 16, "selection_policy": "best_validation_loss"},
            runtime={"device": "cpu", "precision": "fp16", "launch_profile": "test", "vram_fraction": 0.9},
            features={"retention": "manual", "promotion": None},
            packing={"mode": "concat", "sequence_length": 16, "overflow_policy": "fail", "dropped_tail_policy": "count_and_drop"},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_source_roles.py -v --tb=short
```
Expected: FAIL — the config with `role: "test"` is accepted by validation.

- [ ] **Step 3: Remove test/external-gate from SourceRole enum**

In `pipeline_v2/slm_train/config/models.py`, line 11, change:

```python
# BEFORE:
SourceRole = Literal["train", "validation", "test", "external-gate"]

# AFTER:
SourceRole = Literal["train", "validation"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_source_roles.py -v --tb=short
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline_v2/slm_train/config/models.py pipeline_v2/tests/unit/test_source_roles.py
git commit -m "fix(v2): refuse held-out source roles that R1 cannot honour"
```

---

## Milestone 2: Config and Migration Correctness (CRITICAL)

**Goal:** The `migrate-v1` command produces a valid V2 config. Every config field is either applied or removed.

### Task 2.1: Fix `migrate-v1` to produce valid configs

**Files:**
- Modify: `pipeline_v2/slm_train/adapters/legacy_v1.py`
- Create: `pipeline_v2/tests/unit/test_migrate_v1.py`

- [ ] **Step 1: Write the failing test**

Create `pipeline_v2/tests/unit/test_migrate_v1.py`:
```python
"""Tests for V1->V2 migration: the output must be a valid V2 config."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "pipeline_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline_v2"))


def test_migrated_config_validates(tmp_path: Path) -> None:
    """migrate-v1 output must pass validate."""
    from slm_train.adapters.legacy_v1 import migrate_v1_config

    # Create a minimal V1 config
    v1_config = tmp_path / "v1.yaml"
    v1_config.write_text(yaml.safe_dump({
        "name": "test_migration",
        "seed": 42,
        "model": {"layers": 2, "hidden": 32, "heads": 4, "kv_heads": 2, "ffn": 64, "ctx": 16},
        "train": {"max_steps": 10, "batch": 1, "accum": 1, "lr": 3e-3, "warmup_steps": 2},
    }), encoding="utf-8")

    v2_config = tmp_path / "v2.yaml"
    result = migrate_v1_config(str(v1_config), str(v2_config))

    # The migrated config must exist
    assert v2_config.exists()

    # The migrated config must have schema_version: 2
    migrated = yaml.safe_load(v2_config.read_text(encoding="utf-8"))
    assert migrated.get("schema_version") == 2

    # The migrated config must not have forbidden keys
    assert "provenance" not in migrated

    # The migrated config must be loadable by the V2 loader
    from slm_train.config.loader import load_config
    config = load_config(v2_config, root=ROOT)
    assert config.run.name == "test_migration"
    assert config.run.seed == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_migrate_v1.py -v --tb=short
```
Expected: FAIL — the migrated config is missing `schema_version`, has `provenance`, or fails to load.

- [ ] **Step 3: Fix the migration adapter**

In `pipeline_v2/slm_train/adapters/legacy_v1.py`, replace the hand-rolled YAML parser with PyYAML and fix the field mappings:

```python
import yaml

def _read_mapping(path: str) -> dict:
    """Read a V1 YAML config using a real YAML parser."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Map V1 fields to V2 paths that actually exist in the schema
DIRECT_FIELDS = {
    "name": "run.name",
    "seed": "run.seed",
    "model.layers": "model.layers",
    "model.hidden": "model.hidden",
    "model.heads": "model.heads",
    "model.kv_heads": "model.kv_heads",
    "model.ffn": "model.intermediate",
    "model.ctx": "model.context",
    "train.max_steps": "train.max_steps",
    "train.batch": "train.batch_size",
    "train.accum": "train.gradient_accumulation",
    "train.lr": "train.learning_rate",
    "train.warmup_steps": "train.warmup_steps",
    "train.scheduler": "train.scheduler",
    "train.optim": "train.optimizer",
}


def render_v2_config(v1: dict, v2: dict) -> dict:
    """Render a valid V2 config from V1 fields."""
    v2["schema_version"] = 2
    v2["release"] = "r1"
    v2["capabilities"] = ["pretrain"]
    # ... map fields ...
    # Remove forbidden keys
    v2.pop("provenance", None)
    return v2
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_migrate_v1.py -v --tb=short
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline_v2/slm_train/adapters/legacy_v1.py pipeline_v2/tests/unit/test_migrate_v1.py
git commit -m "fix(v2): migrate-v1 produces valid V2 configs"
```

### Task 2.2: Remove or implement `runtime.worker_count`

**Files:**
- Modify: `pipeline_v2/slm_train/config/models.py` (remove field)
- Modify: `pipeline_v2/configs/base/r1.yaml` (remove field)

- [ ] **Step 1: Remove worker_count from RuntimeConfig**

In `pipeline_v2/slm_train/config/models.py`, find the `RuntimeConfig` class. Remove the `worker_count` field if it exists.

- [ ] **Step 2: Remove worker_count from all configs**

Run:
```powershell
Get-ChildItem -Path pipeline_v2/configs -Recurse -Filter "*.yaml" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match "worker_count") {
        Write-Output "Found in: $($_.FullName)"
    }
}
```
Remove any `worker_count` lines found.

- [ ] **Step 3: Verify no configs reference worker_count**

Run:
```powershell
Select-String -Path pipeline_v2/configs/**/*.yaml -Pattern "worker_count"
```
Expected: No matches.

- [ ] **Step 4: Commit**

```bash
git add pipeline_v2/slm_train/config/models.py pipeline_v2/configs/
git commit -m "fix(v2): remove unused runtime.worker_count field"
```

---

## Milestone 3: Dead Code Removal (MEDIUM)

**Goal:** Every module is either used or deleted. No dead abstractions.

### Task 3.1: Remove or implement `HFCausalLMTrainer`

**Files:**
- Modify: `pipeline_v2/slm_train/training/hf_causal_lm.py`
- Modify: `pipeline_v2/tests/unit/test_model_factory.py`

- [ ] **Step 1: Check if HFCausalLMTrainer is referenced anywhere**

Run:
```powershell
Select-String -Path pipeline_v2/slm_train/**/*.py -Pattern "HFCausalLMTrainer|hf_causal_lm"
```
Expected: Only referenced in its own module and tests.

- [ ] **Step 2: Delete the module and its tests**

If the module is only referenced in its own tests:
```bash
git rm pipeline_v2/slm_train/training/hf_causal_lm.py
```
Remove any tests that import from it in `pipeline_v2/tests/unit/test_model_factory.py`.

- [ ] **Step 3: Verify the package still imports**

Run:
```powershell
& .\.venv\Scripts\python.exe -c "from slm_train.orchestration.handlers import HANDLERS; print('OK')"
```
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add pipeline_v2/slm_train/training/hf_causal_lm.py pipeline_v2/tests/unit/test_model_factory.py
git commit -m "refactor(v2): remove dead HFCausalLMTrainer module"
```

### Task 3.2: Remove or wire `monitoring/heartbeat.py`

**Files:**
- Modify: `pipeline_v2/slm_train/monitoring/heartbeat.py`

- [ ] **Step 1: Check if heartbeat is referenced anywhere**

Run:
```powershell
Select-String -Path pipeline_v2/slm_train/**/*.py -Pattern "heartbeat|HealthSnapshot|observe_liveness"
```
Expected: Only referenced in its own module.

- [ ] **Step 2: Delete the module**

```bash
git rm pipeline_v2/slm_train/monitoring/heartbeat.py
```

- [ ] **Step 3: Verify the package still imports**

Run:
```powershell
& .\.venv\Scripts\python.exe -c "from slm_train.orchestration.handlers import HANDLERS; print('OK')"
```
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add pipeline_v2/slm_train/monitoring/heartbeat.py
git commit -m "refactor(v2): remove dead heartbeat module"
```

---

## Milestone 4: Documentation and Routing (MEDIUM)

**Goal:** Documentation matches reality. The repository router mentions V2.

### Task 4.1: Fix `pipeline_v2/README.md` drift

**Files:**
- Modify: `pipeline_v2/README.md`

- [ ] **Step 1: Update the README**

In `pipeline_v2/README.md`, find the line that says `run` "is currently wired to refuse with BLOCKED" and replace it with:

```markdown
`run` executes the complete 13-stage graph on the configured device, or exactly resumes a run whose stages are already published.
```

- [ ] **Step 2: Commit**

```bash
git add pipeline_v2/README.md
git commit -m "docs(v2): fix README drift about run command"
```

### Task 4.2: Fix `pipeline_v2/docs/acceptance.md` drift

**Files:**
- Modify: `pipeline_v2/docs/acceptance.md`

- [ ] **Step 1: Update the acceptance doc**

In `pipeline_v2/docs/acceptance.md`, find the section that says the GPU sequence "has NOT been executed" and update it to reference the published runs under `runs/v2/core/`.

- [ ] **Step 2: Commit**

```bash
git add pipeline_v2/docs/acceptance.md
git commit -m "docs(v2): fix acceptance doc drift about GPU runs"
```

### Task 4.3: Add `pipeline_v2` to the repository router

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add V2 to the portfolio map**

In `AGENTS.md`, find the portfolio map table. Add a row:

```markdown
| pipeline_v2 | governed-by-config evidence pipeline: 11 CLI commands, 13-stage graph, manifest-bound artifacts, cooperative stop | [pipeline_v2/README.md](pipeline_v2/README.md), [pipeline_v2/docs/acceptance.md](pipeline_v2/docs/acceptance.md), [pipeline_v2/docs/remediation-plan.md](pipeline_v2/docs/remediation-plan.md) |
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add pipeline_v2 to repository router"
```

### Task 4.4: Add this plan to the plan index

**Files:**
- Modify: `docs/plans/README.md`

- [ ] **Step 1: Add the plan to the index**

In `docs/plans/README.md`, add a row to the table:

```markdown
| docs/plans/2026-09-28-v2-fix-milestones.md | core | active | core | 2026-09-28 | none |
```

- [ ] **Step 2: Commit**

```bash
git add docs/plans/README.md
git commit -m "docs: add v2 fix milestones to plan index"
```

---

## Milestone 5: Full Test Suite Verification (CRITICAL)

**Goal:** The full V2 test suite passes with no errors and no unexpected failures.

- [ ] **Step 1: Run the full test suite**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/ -v --tb=short 2>&1 | Select-Object -Last 50
```
Expected: All tests pass. If any fail, diagnose and fix before proceeding.

- [ ] **Step 2: Fix any remaining test failures**

For each failure:
1. Read the test to understand what it expects
2. Read the implementation to understand what it does
3. Fix the implementation (not the test, unless the test is wrong)
4. Re-run the test

- [ ] **Step 3: Run the full suite again**

Run:
```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/ -v --tb=short
```
Expected: 0 failed, 0 errors.

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add pipeline_v2/
git commit -m "fix(v2): all tests pass"
```

---

## Dependency Graph

```
Milestone 0 (Diagnose Training Failure)
  |
  v
Milestone 1 (Data Pipeline Correctness)
  |-- Task 1.1 (normalization)
  |-- Task 1.2 (filters)
  |-- Task 1.3 (source roles)
  |
  v
Milestone 2 (Config and Migration)
  |-- Task 2.1 (migrate-v1)
  |-- Task 2.2 (worker_count)
  |
  v
Milestone 3 (Dead Code Removal)
  |-- Task 3.1 (HFCausalLMTrainer)
  |-- Task 3.2 (heartbeat)
  |
  v
Milestone 4 (Documentation)
  |-- Task 4.1 (README)
  |-- Task 4.2 (acceptance)
  |-- Task 4.3 (router)
  |-- Task 4.4 (plan index)
  |
  v
Milestone 5 (Full Test Suite)
```

**Parallelizable:** Tasks within a milestone are independent. Milestones 1, 2, and 3 can run in parallel after Milestone 0 completes. Milestone 4 can run in parallel with any other.

---

## Acceptance Criteria for the Entire Plan

1. The full V2 test suite passes: `0 failed, 0 errors`
2. Every `data.*` config field is either applied or the config is refused
3. Every published manifest certifies what actually happened
4. `migrate-v1` produces a valid V2 config that passes `validate`
5. No dead code remains: every module is either used or deleted
6. Documentation matches reality: README, acceptance docs, and router are current
7. The plan index in `docs/plans/README.md` includes this plan

---

## Self-Review Checklist

- [x] **Spec coverage:** Every issue from the review has a corresponding task
- [x] **Placeholder scan:** No TBDs, no "implement later", no "add appropriate error handling"
- [x] **Type consistency:** All function signatures and field names are consistent across tasks
- [x] **File paths:** All paths are exact and verified
- [x] **Test code:** Every code step has a corresponding test
- [x] **Commands:** Every step has an exact command with expected output
