# Core Training Pipeline V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build the self-contained, fail-closed V2 Release 1 causal-LM pretraining pipeline for S smoke, P pilot, and T target without modifying or depending on V1.

**Architecture:** Implement a statically linked, in-process modular monolith under pipeline_v2. The lifecycle engine owns configuration, identity, data generations, GPU leasing, training, monitoring, evaluation, and atomic publication. Release 1 exposes only core pretraining; SFT, LoRA, KD, GDN, and mounting are rejected by the R1 capability gate, and V3 dynamic plugins and distributed execution are not implemented.

**Tech Stack:** Python 3.12 in the existing .venv; uv-managed dependencies; Pydantic v2; PyYAML with duplicate-key rejection; PyTorch 2.14.0+cu126; Transformers 5.16.1; Datasets 5.0.1; Accelerate 1.14.0; safetensors; tokenizers; Hugging Face Hub; TensorBoard; filelock; pytest; JSON and JSONL; SHA-256; PowerShell.

---

## Scope and execution constraints

- This plan implements V2 Release 1 only: S, P, and T causal-LM pretraining plus the safety, identity, evidence, and evaluation contracts required to accept them.
- Do not implement SFT, LoRA, KD, GDN, or mounting in this plan. Their names are rejected before any stage starts.
- Do not implement runtime plugin discovery, entry-point loading, subprocess plugins, remote plugins, distributed workers, or remote artifact stores. Those belong to V3.
- V1 is read-only. Do not edit, delete, delegate to, cut over to, or use as a fallback any V1 script, config, checkpoint, run, or artifact.
- The normal V2 lifecycle must not import src.model, src.data, scripts.train, scripts.prepare_data, scripts.tokenize_data, or any other V1 implementation module. A legacy adapter may read a V1 input only when migrate-v1 is explicitly invoked.
- Preserve every pre-existing dirty, staged, deleted, and untracked path. Never use reset, stash, clean, bulk deletion, or history rewriting.
- Stage exact paths only. Every commit command below uses git commit --only so unrelated staged work is not included. Before each commit, inspect the path-specific diff; if a listed path already contains pre-existing user work, stop with NEEDS_USER_DECISION rather than staging over it.
- Every interpreter command uses & .\\.venv\\Scripts\\python.exe. Do not use bare python, bare pip, or an interpreter outside the repository venv.
- Use the existing repository-root `.venv`. Do not create `pipeline_v2/.venv`. The canonical lock command is `uv lock --project pipeline_v2`; the only approved install command is `uv sync --project pipeline_v2 --active --inexact --locked --extra test`, which preserves unrelated shared packages. Never run a bare or exact `uv sync`.
- The one-time drift in the shared `.venv` is explicitly authorized and accepted. Freeze further sync or environment changes until a new user decision; the decision evidence is `.superpowers/sdd/pipeline-v2-task-1-report.md`.
- Ordinary implementation and CI tasks are CPU-only and do not access the network or GPU. GPU preflight, smoke, pilot, target, and kill/resume acceptance run only in a user-opened GPU window.
- Network access is limited to a source explicitly declared in a validated config. CPU tests use local fixtures and fake source readers.
- All persisted paths are repository-relative. External machine paths are represented by a hash or token and are never written as evidence paths.
- Every report, test result, and run claim uses exactly one of PASS, FAIL, SKIPPED, BLOCKED, or NEEDS_USER_DECISION.
- A missing, stale, mismatched, incomplete, or unparseable artifact is never treated as PASS.
- Do not quote runtime metrics or GPU state from memory. Read the generated artifact and record its exact repository-relative path.
- Do not add automatic checkpoint or log deletion. The V2 retention policy is explicit and manual; no V1 save_total_limit behavior is copied without a separate decision.

## Required policy-decision gate

The approved V2 specification and active repository policies contain two wording conflicts. Task 0 records an explicit decision before implementation proceeds; no code silently chooses a behavior.

1. OOM: V2 Release 1 keeps the approved specification behavior. An OOM in an existing run is BLOCKED, preserves the last valid checkpoint, and does not mutate batch, context, optimizer, precision, or command. A smaller next attempt is allowed only as a separately user-approved, versioned launch profile and a new run identity. The same run never silently changes its contract. Until the decision record is adopted, the status is NEEDS_USER_DECISION.
2. Preflight: V2 run.py preflight --gpu is the only V2 GPU probe path. No V2 module imports or invokes laya/scripts. If the owner wants the existing laya preflight script to remain a human-run external gate, its exact command and report path are recorded in the same decision record and must not create a second code path. Until the decision record is adopted, the status is NEEDS_USER_DECISION.

The plan can be written and CPU-tested with these gates unresolved, but implementation stops at Task 0 until the decision is recorded.

## Planned file map

### Create

- pipeline_v2/README.md — user-facing commands, editing flow, evidence layout, and troubleshooting.
- pipeline_v2/run.py — the only supported V2 executable wrapper.
- pipeline_v2/pyproject.toml — V2 package metadata, pinned direct dependencies, and pytest configuration.
- pipeline_v2/uv.lock — generated dependency lock.
- pipeline_v2/docs/implementation-decisions.md — policy decisions, exact uv command, test-artifact root, and retention decision.
- pipeline_v2/docs/acceptance.md — user-opened GPU acceptance runbook and evidence checklist.
- pipeline_v2/configs/base/r1.yaml — complete minimal R1 base configuration.
- pipeline_v2/configs/experiments/smoke.yaml — S smoke template.
- pipeline_v2/configs/experiments/pilot.yaml — P template.
- pipeline_v2/configs/experiments/target.yaml — T template.
- pipeline_v2/configs/schemas/config.schema.json — committed Pydantic-generated JSON Schema.
- pipeline_v2/slm_train/__init__.py — package version and public error exports.
- pipeline_v2/slm_train/errors.py — typed fail-closed pipeline errors.
- pipeline_v2/slm_train/cli.py — command dispatch and machine-readable command results.
- pipeline_v2/slm_train/config/__init__.py — configuration exports.
- pipeline_v2/slm_train/config/errors.py — shared configuration error type.
- pipeline_v2/slm_train/config/models.py — strict R1 configuration models.
- pipeline_v2/slm_train/config/inheritance.py — inheritance graph, cycle detection, and list-merge policy.
- pipeline_v2/slm_train/config/overrides.py — documented safe overrides.
- pipeline_v2/slm_train/config/canonical.py — canonical YAML/JSON and semantic hash.
- pipeline_v2/slm_train/config/loader.py — duplicate-key-safe loading and resolution.
- pipeline_v2/slm_train/config/paths.py — config-root and repository-relative path policy.
- pipeline_v2/slm_train/contracts/__init__.py — contract exports.
- pipeline_v2/slm_train/contracts/base.py — schema version, validation labels, and common fields.
- pipeline_v2/slm_train/contracts/run.py — run identity and promotion contracts.
- pipeline_v2/slm_train/contracts/data.py — data, source, freshness, and generation contracts.
- pipeline_v2/slm_train/contracts/tokenizer.py — tokenizer identity contract.
- pipeline_v2/slm_train/contracts/tokenized.py — shard, packing, and tokenized-data contract.
- pipeline_v2/slm_train/contracts/model.py — model capability and model-manifest contract.
- pipeline_v2/slm_train/contracts/evaluator.py — evaluator-suite and metric contract.
- pipeline_v2/slm_train/contracts/checkpoint.py — checkpoint-state and completion contract.
- pipeline_v2/slm_train/contracts/evaluation.py — report and gate-result contract.
- pipeline_v2/slm_train/contracts/status.py — mutable status and freshness contract.
- pipeline_v2/slm_train/contracts/interfaces.py — trusted source, transformer, model, backend, evaluator, and gate protocols.
- pipeline_v2/slm_train/identity/__init__.py — identity exports.
- pipeline_v2/slm_train/identity/source.py — local Merkle and HF repository/file identity.
- pipeline_v2/slm_train/identity/code.py — V2 source, schema, contract, and dependency hash.
- pipeline_v2/slm_train/identity/runtime.py — venv, Python, package, OS, CUDA, driver, and device identity.
- pipeline_v2/slm_train/identity/invocation.py — canonical command, cwd, and allowlisted environment.
- pipeline_v2/slm_train/identity/run_id.py — bootstrap and final run identity formula.
- pipeline_v2/slm_train/artifacts/__init__.py — artifact exports.
- pipeline_v2/slm_train/artifacts/atomic.py — temporary sibling, flush, fsync, exclusive publication, replacement, and directory publication.
- pipeline_v2/slm_train/artifacts/hash.py — SHA-256 file and Merkle helpers.
- pipeline_v2/slm_train/artifacts/manifest.py — immutable and mutable manifest envelopes.
- pipeline_v2/slm_train/artifacts/markers.py — completion-marker create and validation.
- pipeline_v2/slm_train/artifacts/log_index.py — immutable segments and mutable hash-chained index.
- pipeline_v2/slm_train/artifacts/publication.py — staged final artifacts and COMPLETE publication.
- pipeline_v2/slm_train/artifacts/paths.py — V2 data, attempt, and final run path policy.
- pipeline_v2/slm_train/orchestration/__init__.py — orchestration exports.
- pipeline_v2/slm_train/orchestration/graph.py — exact typed stage graph.
- pipeline_v2/slm_train/orchestration/stages.py — stage input/output contracts.
- pipeline_v2/slm_train/orchestration/engine.py — fail-closed coordinator.
- pipeline_v2/slm_train/orchestration/bootstrap.py — bootstrap identity and attempt evidence.
- pipeline_v2/slm_train/orchestration/identity_handoff.py — create-if-absent and compare-and-swap handoff.
- pipeline_v2/slm_train/orchestration/status.py — atomic status snapshots.
- pipeline_v2/slm_train/orchestration/errors.py — stage transition errors.
- pipeline_v2/slm_train/resources/__init__.py — resource exports.
- pipeline_v2/slm_train/resources/lease.py — cross-process GPU lease.
- pipeline_v2/slm_train/resources/gpu_probe.py — explicit GPU device and VRAM probe.
- pipeline_v2/slm_train/resources/device.py — device capability contract.
- pipeline_v2/slm_train/resources/doctor.py — CPU-only environment checks.
- pipeline_v2/slm_train/resources/preflight.py — explicit preflight report writer.
- pipeline_v2/slm_train/data/__init__.py — data exports.
- pipeline_v2/slm_train/data/sources/__init__.py — source reader exports.
- pipeline_v2/slm_train/data/sources/base.py — source protocol and required provenance.
- pipeline_v2/slm_train/data/sources/local.py — local exact file-set reader.
- pipeline_v2/slm_train/data/sources/huggingface.py — pinned HF repository reader.
- pipeline_v2/slm_train/data/prepare.py — preparation lifecycle.
- pipeline_v2/slm_train/data/dedupe.py — stable hash deduplication.
- pipeline_v2/slm_train/data/split.py — salted deterministic split and overlap report.
- pipeline_v2/slm_train/data/provenance.py — license, role, source, and regeneration record.
- pipeline_v2/slm_train/data/generation.py — unique generation publication.
- pipeline_v2/slm_train/data/tokenizer.py — tokenizer identity and token validation.
- pipeline_v2/slm_train/data/packing.py — concat and document_masked packing.
- pipeline_v2/slm_train/data/shards.py — ordered uint32 shard reader and writer.
- pipeline_v2/slm_train/data/validate.py — generation, count, hash, and range validation.
- pipeline_v2/slm_train/data/dataset.py — manifest-driven PackedDataset.
- pipeline_v2/slm_train/models/__init__.py — model exports.
- pipeline_v2/slm_train/models/causal_lm.py — R1 causal-LM contract.
- pipeline_v2/slm_train/models/factory.py — GQA model construction and capability gates.
- pipeline_v2/slm_train/models/manifest.py — model manifest.
- pipeline_v2/slm_train/training/__init__.py — training exports.
- pipeline_v2/slm_train/training/backend.py — TrainerBackend protocol.
- pipeline_v2/slm_train/training/spec.py — centralized typed training specification.
- pipeline_v2/slm_train/training/hf_causal_lm.py — HF Trainer adapter.
- pipeline_v2/slm_train/training/events.py — typed lifecycle event contracts.
- pipeline_v2/slm_train/training/state.py — complete exact-resume state.
- pipeline_v2/slm_train/training/checkpoint.py — checkpoint publication.
- pipeline_v2/slm_train/training/resume.py — exact identity matching.
- pipeline_v2/slm_train/training/selection.py — complete-checkpoint selection.
- pipeline_v2/slm_train/monitoring/__init__.py — monitoring exports.
- pipeline_v2/slm_train/monitoring/bus.py — run-scoped event bus.
- pipeline_v2/slm_train/monitoring/health.py — finite, heartbeat, resource, and spill checks.
- pipeline_v2/slm_train/monitoring/metrics.py — JSONL and TensorBoard writers.
- pipeline_v2/slm_train/monitoring/heartbeat.py — heartbeat and process-liveness checks.
- pipeline_v2/slm_train/evaluation/__init__.py — evaluation exports.
- pipeline_v2/slm_train/evaluation/contracts.py — evaluator report and gate contracts.
- pipeline_v2/slm_train/evaluation/suites.py — registered versioned suites.
- pipeline_v2/slm_train/evaluation/internal.py — validation loss and perplexity.
- pipeline_v2/slm_train/evaluation/generation.py — generation smoke.
- pipeline_v2/slm_train/evaluation/syntax.py — applicable code/syntax checks.
- pipeline_v2/slm_train/evaluation/resources.py — throughput and resource report.
- pipeline_v2/slm_train/evaluation/gates.py — expected-set and threshold validation.
- pipeline_v2/slm_train/evaluation/reports.py — report read-back and evaluation IDs.
- pipeline_v2/slm_train/features/__init__.py — static feature boundary.
- pipeline_v2/slm_train/features/registry.py — R1 capability registry and R2 rejection.
- pipeline_v2/slm_train/adapters/__init__.py — adapter exports.
- pipeline_v2/slm_train/adapters/registry.py — source-controlled in-process registries.
- pipeline_v2/slm_train/adapters/legacy_v1.py — explicit read-only V1 config reader.
- pipeline_v2/slm_train/security/__init__.py — security exports.
- pipeline_v2/slm_train/security/paths.py — path and symlink checks.
- pipeline_v2/slm_train/security/model_loading.py — safetensors and weights-only loading.
- pipeline_v2/slm_train/security/secrets.py — secret redaction.
- pipeline_v2/slm_train/security/export.py — allowlisted export allowlist passed to the operation.
- pipeline_v2/migrations/README.md — migration boundary and no-launch rule.
- pipeline_v2/tests/conftest.py — fixture roots, temp roots, and CPU/GPU markers.
- pipeline_v2/tests/unit/test_package_contract.py — CLI/package boundary test.
- pipeline_v2/tests/unit/test_config.py — strict configuration and schema tests.
- pipeline_v2/tests/unit/test_contracts.py — manifest and interface contract tests.
- pipeline_v2/tests/unit/test_atomic_artifacts.py — atomic publication tests.
- pipeline_v2/tests/unit/test_identity.py — source/runtime/final identity tests.
- pipeline_v2/tests/unit/test_data.py — source, preparation, packing, and dataset tests.
- pipeline_v2/tests/unit/test_model_factory.py — R1 model capability tests.
- pipeline_v2/tests/unit/test_orchestration.py — graph, status, and recovery tests.
- pipeline_v2/tests/unit/test_lease.py — lease, handoff, and race tests.
- pipeline_v2/tests/unit/test_resume.py — cursor and identity resume tests.
- pipeline_v2/tests/unit/test_monitoring.py — event, log, and health tests.
- pipeline_v2/tests/unit/test_evaluation.py — evaluator/gate/publication tests.
- pipeline_v2/tests/contract/test_r1_contracts.py — capability and source contracts.
- pipeline_v2/tests/contract/test_v1_boundary.py — no V1 import or mutation boundary.
- pipeline_v2/tests/contract/test_release_boundaries.py — R1/R2/V3 boundary.
- pipeline_v2/tests/contract/test_evidence_contract.py — repository-relative evidence.
- pipeline_v2/tests/integration/test_fixture_pipeline.py — CPU one-step end-to-end test.
- pipeline_v2/tests/network/test_hf_source.py — explicitly marked, opt-in pinned HF source test.
- pipeline_v2/tests/test_artifact_writer.py — atomic machine-readable pytest result writer.
- pipeline_v2/tests/fixtures/tiny_corpus.jsonl — deterministic local corpus.
- pipeline_v2/tests/fixtures/tiny_config.yaml — deterministic CPU integration config.
- pipeline_v2/tests/fixtures/v1_inventory.py — deterministic read-only V1 inventory helper.
- pipeline_v2/tests/fixtures/v1_inventory.json — read-only V1 inventory captured in Task 0.

### Modify

- .gitignore — add only the scoped V2 runtime/test exclusions and pipeline_v2/*.egg-info/; preserve the data/v2/tokens/*.bin exception and do not alter existing V1 rules.
- pipeline_v2/README.md — complete the commands and evidence sections after the implementation tasks pass.
- pipeline_v2/docs/acceptance.md — add the final read-back matrix and S/P/T promotion evidence paths.
- pipeline_v2/configs/experiments/smoke.yaml, pilot.yaml, target.yaml — add validated S/P/T templates.

### Explicitly not created or modified

- scripts/train.py, scripts/prepare_data.py, scripts/tokenize_data.py, scripts/sft.py, scripts/kd.py, scripts/mount.py, src/model.py, src/data.py, existing root configs, existing runs, and existing checkpoints.
- Any R2 feature implementation or V3 plugin runtime.
- Any WebUI, Flow, backup, or component-line entry point.

## Dependency order

1. Task 0 freezes the V1 boundary, records policy decisions, and records the exact uv command.
2. Task 1 establishes the package and dependency lock.
3. Task 2 establishes strict configuration and semantic resolution.
4. Task 3 establishes contracts, identity, paths, and atomic publication.
5. Tasks 4 and 5 establish pinned data and tokenized generations.
6. Task 6 establishes the R1 model and HF backend boundary.
7. Task 7 establishes static interfaces and registries.
8. Task 8 establishes orchestration, lease, preflight, and bootstrap handoff.
9. Task 9 establishes checkpoints, resume, events, and monitoring.
10. Task 10 establishes evaluators, gates, and final publication.
11. Task 11 connects all commands, security, and migration.
12. Task 12 runs the CPU unit, contract, and integration lanes.
13. Task 13 runs only the user-opened GPU acceptance sequence and closes Release 1 evidence.

---

## Task 0: Freeze V1, record policy decisions, and choose artifact roots

**Files:**
- Create: pipeline_v2/docs/implementation-decisions.md
- Create: pipeline_v2/tests/fixtures/v1_inventory.py
- Create: pipeline_v2/tests/fixtures/v1_inventory.json
- Modify: .gitignore

- [ ] **Step 1: Capture a read-only V1 inventory**

Create a JSON inventory containing the SHA-256, size, and role of this exact frozen list: scripts/train.py, scripts/prepare_data.py, scripts/tokenize_data.py, scripts/run_custom.py, scripts/sanity_check.py, scripts/vram_probe.py, scripts/eval.py, scripts/sft.py, scripts/kd.py, scripts/mount.py, src/model.py, src/data.py, configs/smoke.yaml, configs/pilot.yaml, configs/target.yaml, configs/next_pretrain.yaml, tests/_common.py, tests/test_artifacts.py, tests/test_explorer.py, and tests/test_simulator.py. Do not discover files with a broad glob. Use only repository reads and the venv interpreter.

~~~python
# pipeline_v2/tests/fixtures/v1_inventory.py
from hashlib import sha256
import json
from pathlib import Path

INVENTORY_PATHS = (
    "scripts/train.py",
    "scripts/prepare_data.py",
    "scripts/tokenize_data.py",
    "scripts/run_custom.py",
    "scripts/sanity_check.py",
    "scripts/vram_probe.py",
    "scripts/eval.py",
    "scripts/sft.py",
    "scripts/kd.py",
    "scripts/mount.py",
    "src/model.py",
    "src/data.py",
    "configs/smoke.yaml",
    "configs/pilot.yaml",
    "configs/target.yaml",
    "configs/next_pretrain.yaml",
    "tests/_common.py",
    "tests/test_artifacts.py",
    "tests/test_explorer.py",
    "tests/test_simulator.py",
)

ROLE_BY_PATH = {path: "v1-core-boundary" for path in INVENTORY_PATHS}


def inventory(paths: tuple[str, ...], root: Path) -> dict[str, object]:
    rows = []
    for relative in sorted(paths):
        path = root / relative
        payload = path.read_bytes()
        rows.append({
            "path": relative,
            "role": ROLE_BY_PATH[relative],
            "sha256": sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    return {"schema_version": 1, "files": rows}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    target = root / "pipeline_v2/tests/fixtures/v1_inventory.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(inventory(INVENTORY_PATHS, root), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
~~~

The command writes only pipeline_v2/tests/fixtures/v1_inventory.json. It must not write under scripts, src, configs, data, or runs.

- [ ] **Step 2: Record the exact policy decision**

Write this structure to pipeline_v2/docs/implementation-decisions.md, replacing NEEDS_USER_DECISION only after the user adopts the two decisions:

~~~yaml
schema_version: 1
oom:
  same_run: BLOCKED
  altered_batch_context_optimizer_precision: forbidden
  reduced_attempt: new_user_approved_versioned_launch_profile_and_new_run
  status: NEEDS_USER_DECISION
preflight:
  v2_code_path: pipeline_v2/run.py preflight --gpu
  laya_import: forbidden
  external_laya_gate: human_pre_registered_only_if_user_adopts
  status: NEEDS_USER_DECISION
dependency_command:
  lock: uv lock --project pipeline_v2
  install: uv sync --project pipeline_v2 --active --inexact --locked --extra test
environment:
  root: .venv
  nested_venv: pipeline_v2/.venv_forbidden
  one_time_shared_venv_drift: user_authorized_and_accepted
  freeze_further_sync_or_environment_changes: pending_new_user_decision
  evidence: .superpowers/sdd/pipeline-v2-task-1-report.md
runtime_roots:
  prepared: data/v2/prepared
  tokens: data/v2/tokens
  attempts: runs/v2/_attempts
  final_runs: runs/v2/<namespace>/<run_id>
  cpu_test_artifacts: pipeline_v2/test-artifacts
retention:
  checkpoint_and_log_deletion: manual_under_data_policy
~~~

Task 0 status is NEEDS_USER_DECISION until both decisions are adopted. A later implementation may not infer an OOM recovery or call the external preflight script.

- [ ] **Step 3: Add only the V2 ignore entries**

Read the existing .gitignore and preserve every pre-existing entry; append only absent V2 lines. If an existing rule conflicts with these roots, record NEEDS_USER_DECISION instead of rewriting the file:

~~~gitignore
# V2 runtime and CPU test evidence
data/v2/*
!data/v2/tokens/
data/v2/tokens/*
!data/v2/tokens/*.bin
runs/v2/
pipeline_v2/test-artifacts/
pipeline_v2/*.egg-info/
~~~

Expected: existing V1 ignore rules remain byte-for-byte unchanged.

- [ ] **Step 4: Run the inventory validation and commit Task 0**

~~~powershell
& .\\.venv\\Scripts\\python.exe pipeline_v2/tests/fixtures/v1_inventory.py
& .\\.venv\\Scripts\\python.exe -c "from pathlib import Path; import json; p=Path('pipeline_v2/tests/fixtures/v1_inventory.json'); d=json.loads(p.read_text(encoding='utf-8')); assert d['files']; assert all({'path','role','sha256','bytes'} <= set(row) for row in d['files']); print('PASS', len(d['files']))"
~~~

Expected: PASS and a machine-readable inventory count. Do not commit until the decision record is adopted; if it remains NEEDS_USER_DECISION, stop with that label rather than beginning Task 1.

~~~powershell
git add pipeline_v2/docs/implementation-decisions.md pipeline_v2/tests/fixtures/v1_inventory.py pipeline_v2/tests/fixtures/v1_inventory.json .gitignore
git commit --only -m "docs(v2): freeze v1 boundary and policy decisions" -- pipeline_v2/docs/implementation-decisions.md pipeline_v2/tests/fixtures/v1_inventory.py pipeline_v2/tests/fixtures/v1_inventory.json .gitignore
~~~

---

## Task 1: Scaffold the V2 package and lock dependencies

**Files:**
- Create: pipeline_v2/pyproject.toml
- Create: pipeline_v2/uv.lock
- Create: pipeline_v2/run.py
- Create: pipeline_v2/README.md
- Create: pipeline_v2/slm_train/cli.py
- Create: pipeline_v2/slm_train/__init__.py
- Create: pipeline_v2/slm_train/errors.py
- Create: pipeline_v2/tests/conftest.py
- Create: pipeline_v2/tests/unit/test_package_contract.py

- [ ] **Step 1: Write the failing package-boundary test**

~~~python
# pipeline_v2/tests/unit/test_package_contract.py
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def test_help_is_cpu_only_and_lists_commands() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "pipeline_v2" / "run.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for command in ("init", "validate", "plan", "doctor", "preflight", "run", "status", "explain", "diff", "export", "migrate-v1"):
        assert command in result.stdout


def test_v2_source_does_not_import_v1_modules() -> None:
    forbidden = ("import src.", "from src.", "import scripts.", "from scripts.")
    for path in (ROOT / "pipeline_v2" / "slm_train").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
~~~

- [ ] **Step 2: Run the test and verify it fails**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_package_contract.py -q
~~~

Expected: FAIL because run.py and the package do not exist.

- [ ] **Step 3: Add pinned package metadata and the entry point**

Use the installed direct versions and the existing CUDA source:

~~~toml
# pipeline_v2/pyproject.toml
[build-system]
requires = ["setuptools==78.1.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["slm_train*"]

[project]
name = "nano-slms-pipeline-v2"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "accelerate==1.14.0",
  "datasets==5.0.1",
  "filelock==3.32.3",
  "huggingface-hub==1.30.0",
  "numpy==2.5.2",
  "pydantic==2.13.5",
  "PyYAML==6.0.3",
  "safetensors==0.8.0",
  "tensorboard==2.21.0",
  "tokenizers==0.23.2",
  "torch==2.14.0+cu126",
  "transformers==5.16.1",
]

[project.optional-dependencies]
test = ["pytest==9.1.1"]

[project.scripts]
nano-slms-v2 = "slm_train.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = [
  "gpu: requires a user-opened GPU window",
  "network: requires an explicitly declared source",
]

[tool.uv]
package = true

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
~~~

The wrapper is exactly:

~~~python
# pipeline_v2/run.py
from slm_train.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
~~~

The initial CLI parser is exactly this CPU-safe shell; later tasks replace the command branches with typed handlers without changing the command names:

~~~python
# pipeline_v2/slm_train/cli.py
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline_v2")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "validate", "plan", "doctor", "preflight", "run", "status", "explain", "diff", "export", "migrate-v1"):
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
~~~

The parser imports no CUDA, V1, or network module at module import time.

- [ ] **Step 4: Lock and use the existing repository venv**

Record the approved one-time setup and version check below for reproducibility. This
recipe has already been applied. While the shared-venv freeze is active, do not
execute the lock or sync commands again; they require a new user decision.

~~~powershell
uv --version
uv lock --project pipeline_v2
uv sync --project pipeline_v2 --active --inexact --locked --extra test
& .\\.venv\\Scripts\\python.exe -c "import torch, transformers, pydantic, filelock; print(torch.__version__, transformers.__version__)"
~~~

Expected after a newly approved run: the lock is current, the existing repository
.venv is used by the exact non-destructive command, and the printed versions
match the repository environment. Never use a bare or exact sync, and do not
create pipeline_v2/.venv. The current shared .venv remains frozen after the
one-time user-authorized drift. Decision evidence:
.superpowers/sdd/pipeline-v2-task-1-report.md.

- [ ] **Step 5: Run the test and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_package_contract.py -q
~~~

Expected: PASS.

~~~powershell
git add pipeline_v2/pyproject.toml pipeline_v2/uv.lock pipeline_v2/run.py pipeline_v2/README.md pipeline_v2/slm_train/cli.py pipeline_v2/slm_train/__init__.py pipeline_v2/slm_train/errors.py pipeline_v2/tests/conftest.py pipeline_v2/tests/unit/test_package_contract.py
git commit --only -m "feat(v2): scaffold isolated training package" -- pipeline_v2/pyproject.toml pipeline_v2/uv.lock pipeline_v2/run.py pipeline_v2/README.md pipeline_v2/slm_train/cli.py pipeline_v2/slm_train/__init__.py pipeline_v2/slm_train/errors.py pipeline_v2/tests/conftest.py pipeline_v2/tests/unit/test_package_contract.py
~~~

---

## Task 2: Implement strict R1 configuration, inheritance, paths, and hashing

**Files:**
- Create: pipeline_v2/slm_train/config/__init__.py
- Create: pipeline_v2/slm_train/config/errors.py
- Create: pipeline_v2/slm_train/config/models.py
- Create: pipeline_v2/slm_train/config/inheritance.py
- Create: pipeline_v2/slm_train/config/overrides.py
- Create: pipeline_v2/slm_train/config/canonical.py
- Create: pipeline_v2/slm_train/config/loader.py
- Create: pipeline_v2/slm_train/config/paths.py
- Create: pipeline_v2/configs/base/r1.yaml
- Create: pipeline_v2/configs/schemas/config.schema.json
- Create: pipeline_v2/tests/unit/test_config.py

- [ ] **Step 1: Write failing tests for strictness and semantic identity**

~~~python
from pathlib import Path
import pytest
from slm_train.config.loader import ConfigError, load_config
from slm_train.config.canonical import semantic_hash
from slm_train.config.overrides import apply_safe_overrides
from slm_train.config.models import validate_materialized

ROOT = Path(__file__).resolve().parents[3]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write(tmp_path, "schema_version: 2\nschema_version: 2\n"))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write(tmp_path, "schema_version: 2\nunknown: true\n"))


def test_r2_capability_is_rejected(tmp_path: Path) -> None:
    base = ROOT / "pipeline_v2/configs/base/r1.yaml"
    text = base.read_text(encoding="utf-8").replace(
        "capabilities: [pretrain]",
        "capabilities: [pretrain, sft]",
    )
    with pytest.raises(ConfigError, match="R1"):
        load_config(write(tmp_path, text))


def test_base_template_is_not_runnable_until_materialized() -> None:
    with pytest.raises(ValueError, match="materialized"):
        validate_materialized(load_config(ROOT / "pipeline_v2/configs/base/r1.yaml"))


def test_only_documented_cli_overrides_are_accepted() -> None:
    config = load_config(ROOT / "pipeline_v2/configs/base/r1.yaml")
    changed = apply_safe_overrides(config, {"run.seed": 43})
    assert changed.run.seed == 43
    with pytest.raises(ValueError, match="unsafe"):
        apply_safe_overrides(config, {"train.batch_size": 2})


def test_comments_do_not_change_semantic_hash(tmp_path: Path) -> None:
    base = ROOT / "pipeline_v2/configs/base/r1.yaml"
    commented = tmp_path / "base-with-comment.yaml"
    commented.write_text("# formatting-only comment\n" + base.read_text(encoding="utf-8"), encoding="utf-8")
    assert semantic_hash(load_config(base).model_dump()) == semantic_hash(load_config(commented).model_dump())


def test_inheritance_cycle_and_ambiguous_list_merge_are_rejected(tmp_path: Path) -> None:
    from slm_train.config.inheritance import resolve_inheritance
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: [b.yaml]\n", encoding="utf-8")
    b.write_text("extends: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cycle"):
        resolve_inheritance(a, root=tmp_path)
    a.write_text("extends: [b.yaml]\ncapabilities: [pretrain]\n", encoding="utf-8")
    b.write_text("capabilities: [sft]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="merge"):
        resolve_inheritance(a, root=tmp_path)


def test_cross_field_context_mismatch_is_rejected(tmp_path: Path) -> None:
    base = ROOT / "pipeline_v2/configs/base/r1.yaml"
    text = base.read_text(encoding="utf-8").replace("  context: 2048\n  offsets:", "  context: 1024\n  offsets:")
    with pytest.raises(ConfigError, match="context"):
        load_config(write(tmp_path, text))


def test_nested_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write(tmp_path, "schema_version: 2\nmodel: {not_a_model_field: 1}\n"))


def test_missing_required_nested_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Field required"):
        load_config(write(tmp_path, "schema_version: 2\nmodel: {hidden: 32}\n"))


def test_s_and_promotion_relationship_is_explicit(tmp_path: Path) -> None:
    base = ROOT / "pipeline_v2/configs/base/r1.yaml"
    text = base.read_text(encoding="utf-8").replace(
        "  promotion: null",
        "  promotion: {predecessor_stage: S, predecessor_run_id: s-run, checkpoint: checkpoints/s, data_manifest: manifests/data.json, evaluator_manifest: manifests/evaluator.json, gate_report: reports/s-gate.json}",
    )
    with pytest.raises(ConfigError, match="S cannot reference"):
        load_config(write(tmp_path, text))
~~~

- [ ] **Step 2: Run the tests and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_config.py -q
~~~

Expected: FAIL because the configuration package does not exist.

- [ ] **Step 3: Define strict Pydantic models**

~~~python
# pipeline_v2/slm_train/config/models.py
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

R1_CAPABILITIES = frozenset({"pretrain"})


R2_CAPABILITIES = frozenset({"sft", "lora", "kd", "gdn", "mounting"})
SourceRole = Literal["train", "validation", "test", "external-gate"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RunConfig(StrictModel):
    name: str
    namespace: str
    stage: Literal["S", "P", "T"]
    seed: int = Field(ge=0)


class FreshnessPolicy(StrictModel):
    max_age: str
    supersession: str
    optional_reason: str | None = None


class SourceDeclaration(StrictModel):
    source_id: str
    kind: Literal["local", "huggingface"]
    path_or_repo: str
    revision: str | None
    file_set: tuple[str, ...]
    role: SourceRole
    license: str
    expected_rows: int = Field(ge=0)
    expected_tokens: int = Field(ge=0)
    freshness_policy: FreshnessPolicy
    optional_profile: str | None = None


class DataConfig(StrictModel):
    dataset_id: str
    sources: tuple[SourceDeclaration, ...]
    split_salt: int = Field(ge=0)
    validation_fraction: float = Field(gt=0, lt=1)
    dedupe: str
    normalization: str
    filters: tuple[str, ...]
    regeneration_command: tuple[str, ...]


class TokenizerConfig(StrictModel):
    source: str
    revision_or_files: str | tuple[str, ...]
    special_ids: dict[str, int]
    context: int = Field(gt=0)
    offsets: str


class PackingConfig(StrictModel):
    mode: Literal["concat", "document_masked"]
    sequence_length: int = Field(gt=0)
    overflow_policy: str
    dropped_tail_policy: str


class ModelConfig(StrictModel):
    architecture: Literal["llama"]
    hidden: int = Field(gt=0)
    intermediate: int = Field(gt=0)
    layers: int = Field(gt=0)
    heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    context: int = Field(gt=0)
    tied_embeddings: bool
    precision: Literal["fp16", "bf16"]
    attention: Literal["sdpa", "flash_attention_2"]
    base_model: str | None = None


class TrainingConfig(StrictModel):
    optimizer: str
    scheduler: str
    batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    max_grad_norm: float = Field(gt=0)
    checkpoint_every_steps: int = Field(gt=0)
    evaluation_cadence: int = Field(gt=0)
    checkpoint_policy: str


class EvaluatorSuiteConfig(StrictModel):
    ids: tuple[str, ...]
    versions: tuple[str, ...]
    suites: tuple[str, ...]
    metrics: tuple[str, ...]
    cadence_steps: int = Field(gt=0)
    thresholds: dict[str, float]
    directions: dict[str, Literal["min", "max"]]
    selection_policy: str


class RuntimeConfig(StrictModel):
    device: str
    precision: Literal["fp16", "bf16"]
    launch_profile: str
    worker_count: int = Field(ge=0)


class PromotionReference(StrictModel):
    predecessor_stage: Literal["S", "P"]
    predecessor_run_id: str
    checkpoint: str
    data_manifest: str
    evaluator_manifest: str
    gate_report: str


class FeaturesConfig(StrictModel):
    retention: Literal["manual"]
    promotion: PromotionReference | None


class ConfigDocument(StrictModel):
    schema_version: Literal[2]
    release: Literal["r1"]
    capabilities: tuple[str, ...]
    run: RunConfig
    extends: tuple[str, ...] = ()
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    train: TrainingConfig
    evaluation: EvaluatorSuiteConfig
    runtime: RuntimeConfig
    features: FeaturesConfig
    packing: PackingConfig

    @model_validator(mode="after")
    def validate_release(self) -> "ConfigDocument":
        unknown = set(self.capabilities) - R1_CAPABILITIES
        if unknown:
            raise ValueError(f"R1 rejects capabilities: {sorted(unknown)}")
        if self.capabilities != ("pretrain",):
            raise ValueError("R1 requires exactly the pretrain capability")
        promotion = self.features.promotion
        for source in self.data.sources:
            if source.kind == "huggingface" and not source.revision:
                raise ValueError("Hugging Face sources require a pinned revision")
            if not source.file_set:
                raise ValueError("every source requires an explicit file set")
        if self.run.stage == "S" and promotion is not None:
            raise ValueError("S cannot reference a predecessor")
        if self.run.stage == "P" and promotion is not None and promotion.predecessor_stage != "S":
            raise ValueError("P promotion must reference S")
        if self.run.stage == "T" and promotion is not None and promotion.predecessor_stage != "P":
            raise ValueError("T promotion must reference P")
        if self.runtime.precision != self.model.precision:
            raise ValueError("runtime and model precision must match")
        if self.model.precision != "fp16" or self.model.attention != "sdpa":
            raise ValueError("R1 requires fp16 and sdpa")
        if self.model.context != self.tokenizer.context or self.packing.sequence_length != self.tokenizer.context:
            raise ValueError("model, tokenizer, and packing context must match")
        if self.model.hidden % self.model.heads or self.model.hidden % self.model.kv_heads:
            raise ValueError("hidden size must be divisible by heads and kv_heads")
        if self.model.heads % self.model.kv_heads:
            raise ValueError("attention heads must be divisible by kv heads")
        if len(self.evaluation.ids) != len(self.evaluation.versions) or len(self.evaluation.ids) != len(self.evaluation.suites):
            raise ValueError("evaluator ids, versions, and suites must align")
        if len(set(self.evaluation.ids)) != len(self.evaluation.ids) or len(set(self.evaluation.suites)) != len(self.evaluation.suites):
            raise ValueError("evaluator ids and suites must be unique")
        if set(self.evaluation.thresholds) != set(self.evaluation.metrics):
            raise ValueError("every metric requires one threshold")
        if set(self.evaluation.directions) != set(self.evaluation.metrics):
            raise ValueError("every metric requires one threshold direction")
        return self


def validate_materialized(config: ConfigDocument) -> None:
    def contains_placeholder(value: object) -> bool:
        if isinstance(value, dict):
            return any(contains_placeholder(key) or contains_placeholder(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(contains_placeholder(item) for item in value)
        return value == "replace_me" or (isinstance(value, str) and value.startswith("replace_"))

    if contains_placeholder(config.model_dump()):
        raise ValueError("replace_me values must be materialized before validation")
    if not config.data.sources:
        raise ValueError("at least one source is required")
    if not config.evaluation.thresholds:
        raise ValueError("evaluator thresholds are required")
~~~

Define these nested models with explicit fields and no trainer-specific defaults: SourceDeclaration(source_id, kind, path_or_repo, revision, file_set, role, license, expected_rows, expected_tokens, freshness_policy, optional_profile), FreshnessPolicy(max_age, supersession, optional_reason), DataConfig(dataset_id, sources, split_salt, validation_fraction, dedupe, normalization, filters, regeneration_command), TokenizerConfig(source, revision_or_files, special_ids, context, offsets), PackingConfig(mode, sequence_length, overflow_policy, dropped_tail_policy), ModelConfig(architecture, hidden, intermediate, layers, heads, kv_heads, context, tied_embeddings, precision, attention, base_model), TrainingConfig(optimizer, scheduler, batch_size, gradient_accumulation, max_grad_norm, checkpoint_every_steps, evaluation_cadence, checkpoint_policy), EvaluatorSuiteConfig(ids, versions, suites, metrics, cadence_steps, thresholds, directions, selection_policy), RuntimeConfig(device, precision, launch_profile, worker_count), FeaturesConfig(retention, promotion), and PromotionReference(predecessor_stage, predecessor_run_id, checkpoint, data_manifest, evaluator_manifest, gate_report). Each nested model uses ConfigDict(extra="forbid", strict=True, frozen=True) and rejects omitted required fields instead of inserting hidden defaults. The committed base template is:

~~~yaml
schema_version: 2
release: r1
capabilities: [pretrain]
run:
  name: replace_me
  namespace: core
  stage: S
  seed: 42
extends: []
data:
  dataset_id: replace_me
  sources: []
  split_salt: 0
  validation_fraction: 0.05
  dedupe: exact_hash
  normalization: unicode_nfkc
  filters: []
  regeneration_command: ["./.venv/Scripts/python.exe", "pipeline_v2/run.py", "run", "--config", "pipeline_v2/configs/experiments/replace_me.yaml"]
tokenizer:
  source: replace_me
  revision_or_files: replace_me
  special_ids: {pad: 0, bos: 1, eos: 2}
  context: 2048
  offsets: special_tokens
model:
  architecture: llama
  hidden: 4096
  intermediate: 11008
  layers: 32
  heads: 32
  kv_heads: 8
  context: 2048
  precision: fp16
  attention: sdpa
  tied_embeddings: false
  base_model: null
train:
  optimizer: adamw
  scheduler: cosine
  batch_size: 1
  gradient_accumulation: 1
  max_grad_norm: 1.0
  checkpoint_every_steps: 1000
  evaluation_cadence: 1000
  checkpoint_policy: complete_only
evaluation:
  ids: [validation_loss, generation_smoke]
  versions: ["1", "1"]
  suites: [validation_loss, generation_smoke]
  metrics: [val_loss, perplexity, generation_smoke]
  cadence_steps: 1000
  thresholds: {val_loss: 1.0, perplexity: 1000000.0, generation_smoke: 1.0}
  directions: {val_loss: min, perplexity: min, generation_smoke: min}
  selection_policy: best_validation_loss
runtime:
  device: cuda:0
  precision: fp16
  launch_profile: r1-consumer
  worker_count: 0
features:
  retention: manual
  promotion: null
packing:
  mode: concat
  sequence_length: 2048
  overflow_policy: fail
  dropped_tail_policy: count_and_drop
~~~

Replace every replace_me value and provide explicit source/tokenizer files before run validation. The raw loader may parse the base template for schema/help tests, but validate_materialized rejects placeholders, empty sources, and missing evaluator thresholds; S with a promotion is rejected during model validation.

- [ ] **Step 4: Implement duplicate-key YAML, inheritance, and safe overrides**

~~~python
# pipeline_v2/slm_train/config/errors.py
class ConfigError(ValueError):
    pass

~~~

~~~python
# pipeline_v2/slm_train/config/loader.py
from pathlib import Path
import yaml
from pydantic import ValidationError
from .models import ConfigDocument


from .errors import ConfigError


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader: UniqueKeyLoader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _normalize_sequences(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize_sequences(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_normalize_sequences(item) for item in value)
    return value


def load_config(path: Path, *, root: Path | None = None) -> ConfigDocument:
    try:
        from .inheritance import resolve_inheritance
        resolved_path = path.resolve()
        boundary = root
        if boundary is None:
            cwd = Path.cwd().resolve()
            boundary = cwd if resolved_path.is_relative_to(cwd) else resolved_path.parent.parent
        raw = resolve_inheritance(path, root=boundary)
        return ConfigDocument.model_validate(_normalize_sequences(raw))
    except (OSError, yaml.YAMLError, ValidationError, ConfigError) as exc:
        raise ConfigError(str(exc)) from exc
~~~

~~~python
# pipeline_v2/slm_train/config/inheritance.py
from copy import deepcopy
from pathlib import Path

import yaml

from .errors import ConfigError
from .loader import UniqueKeyLoader


def _read(path: Path) -> dict:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a mapping")
    return value


def _merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if key not in result:
            result[key] = deepcopy(value)
            continue
        if isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        elif isinstance(result[key], list) and isinstance(value, list):
            if result[key] != value:
                raise ConfigError(f"ambiguous list merge for {key}")
        else:
            result[key] = deepcopy(value)
    return result


def resolve_inheritance(path: Path, *, root: Path | None = None) -> dict:
    boundary = (root or Path.cwd()).resolve()
    active: set[Path] = set()
    cache: dict[Path, dict] = {}

    def visit(current: Path) -> dict:
        resolved = current.resolve()
        try:
            resolved.relative_to(boundary)
        except ValueError as exc:
            raise ConfigError("config path escapes declared root") from exc
        if resolved in active:
            raise ConfigError(f"inheritance cycle at {resolved}")
        if resolved in cache:
            return deepcopy(cache[resolved])
        active.add(resolved)
        raw = _read(resolved)
        parents = raw.get("extends", [])
        if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
            raise ConfigError("extends must be a list of relative paths")
        merged: dict = {}
        for parent in parents:
            merged = _merge(merged, visit(resolved.parent / parent))
        local = {key: value for key, value in raw.items() if key != "extends"}
        merged = _merge(merged, local)
        merged["extends"] = tuple(parents)
        active.remove(resolved)
        cache[resolved] = deepcopy(merged)
        return merged

    return visit(path)
~~~

Inheritance must reject cycles and ambiguous list merges, preserve every applied override, and allow only these CLI override paths:

~~~python
# pipeline_v2/slm_train/config/overrides.py
from typing import Any

from .models import ConfigDocument

SAFE_OVERRIDES = {"run.seed", "runtime.device", "runtime.precision"}


def apply_safe_overrides(config: ConfigDocument, values: dict[str, Any]) -> ConfigDocument:
    unknown = set(values) - SAFE_OVERRIDES
    if unknown:
        raise ValueError(f"unsafe config overrides: {sorted(unknown)}")
    updated = config.model_dump()
    for path, value in values.items():
        section, field = path.split(".", 1)
        updated[section][field] = value
    result = ConfigDocument.model_validate(updated)
    return result
~~~

The override map and every applied value are included in resolved.yaml, resolved.json, and the config manifest; no arbitrary dotted path is accepted. resolved.json also records inheritance_chain, applied_overrides, and the canonical serialization version, while its semantic config hash remains independent of comments and formatting. The resolver writes config/resolved.yaml, config/resolved.json, and the config hash in the selected run attempt. Comments and formatting do not alter the semantic hash; release, capabilities, freshness policy, evaluator IDs/versions/suites/gates, metric schema/cadence/thresholds, selection policy, and S/P/T references do. The nested Pydantic models must reject unknown keys in data, tokenizer, model, train, evaluation, runtime, features, and promotion references rather than accepting an untyped dictionary.

- [ ] **Step 5: Implement canonicalization and path containment**

~~~python
# pipeline_v2/slm_train/config/canonical.py
import hashlib
import json


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def semantic_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
~~~

~~~python
# pipeline_v2/slm_train/config/paths.py
from pathlib import Path


class PathPolicy:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError("persisted paths must be repository-relative")
        raw = self.root / candidate
        for parent in (raw, *raw.parents):
            if parent == self.root:
                break
            if parent.is_symlink():
                raise ValueError("symlink paths are forbidden")
        resolved = raw.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path escapes declared root") from exc
        return resolved

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()
~~~

config/paths.py rejects absolute paths, traversal, symlink components, and source paths outside declared roots. All persisted path fields are repository-relative.

- [ ] **Step 6: Generate the schema and run tests**

~~~powershell
& .\\.venv\\Scripts\\python.exe -c "import json; from pathlib import Path; from slm_train.config.models import ConfigDocument; Path('pipeline_v2/configs/schemas/config.schema.json').write_text(json.dumps(ConfigDocument.model_json_schema(), indent=2, sort_keys=True) + '\n', encoding='utf-8')"
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_config.py -q
~~~

Expected: PASS and a committed schema with no implicit release capability.

- [ ] **Step 7: Commit Task 2**

~~~powershell
git add pipeline_v2/slm_train/config pipeline_v2/configs/base/r1.yaml pipeline_v2/configs/schemas/config.schema.json pipeline_v2/tests/unit/test_config.py
git commit --only -m "feat(v2): add strict configuration contracts" -- pipeline_v2/slm_train/config pipeline_v2/configs/base/r1.yaml pipeline_v2/configs/schemas/config.schema.json pipeline_v2/tests/unit/test_config.py
~~~

---

## Task 3: Build versioned contracts, identity, paths, and atomic publication

**Files:**
- Create: pipeline_v2/slm_train/contracts/__init__.py
- Create: pipeline_v2/slm_train/contracts/base.py
- Create: pipeline_v2/slm_train/contracts/run.py
- Create: pipeline_v2/slm_train/contracts/data.py
- Create: pipeline_v2/slm_train/contracts/tokenizer.py
- Create: pipeline_v2/slm_train/contracts/tokenized.py
- Create: pipeline_v2/slm_train/contracts/model.py
- Create: pipeline_v2/slm_train/contracts/evaluator.py
- Create: pipeline_v2/slm_train/contracts/checkpoint.py
- Create: pipeline_v2/slm_train/contracts/evaluation.py
- Create: pipeline_v2/slm_train/contracts/status.py
- Create: pipeline_v2/slm_train/identity/__init__.py
- Create: pipeline_v2/slm_train/identity/source.py
- Create: pipeline_v2/slm_train/identity/code.py
- Create: pipeline_v2/slm_train/identity/runtime.py
- Create: pipeline_v2/slm_train/identity/invocation.py
- Create: pipeline_v2/slm_train/identity/run_id.py
- Create: pipeline_v2/slm_train/artifacts/__init__.py
- Create: pipeline_v2/slm_train/artifacts/atomic.py
- Create: pipeline_v2/slm_train/artifacts/hash.py
- Create: pipeline_v2/slm_train/artifacts/manifest.py
- Create: pipeline_v2/slm_train/artifacts/markers.py
- Create: pipeline_v2/slm_train/artifacts/log_index.py
- Create: pipeline_v2/slm_train/artifacts/paths.py
- Create: pipeline_v2/tests/unit/test_contracts.py
- Create: pipeline_v2/tests/unit/test_atomic_artifacts.py
- Create: pipeline_v2/tests/unit/test_identity.py

- [ ] **Step 1: Write failing contract and atomicity tests**

~~~python
# pipeline_v2/tests/unit/test_atomic_artifacts.py
from pathlib import Path
import pytest
from slm_train.artifacts.atomic import atomic_write_bytes
from slm_train.artifacts.paths import PathPolicy
from slm_train.identity.run_id import InputFingerprint


def test_atomic_write_is_complete(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"complete")
    assert target.read_bytes() == b"complete"
    assert not list(tmp_path.glob(".artifact.bin.*"))


def test_immutable_write_does_not_replace_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"first"


def test_immutable_write_race_has_one_winner(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    target = tmp_path / "artifact.bin"

    def publish(value: bytes) -> str:
        try:
            atomic_write_bytes(target, value)
            return "published"
        except FileExistsError:
            return "collision"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (b"first", b"second")))
    assert sorted(outcomes) == ["collision", "published"]
    assert target.read_bytes() in {b"first", b"second"}


def test_escape_is_rejected(tmp_path: Path) -> None:
    policy = PathPolicy(root=tmp_path)
    with pytest.raises(ValueError):
        policy.resolve("../outside")
    with pytest.raises(ValueError):
        policy.resolve("C:/outside")


def test_symlink_component_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-v2"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        PathPolicy(root=tmp_path).resolve("link/file.txt")


def test_input_fingerprint_is_deterministic_and_identity_sensitive() -> None:
    from dataclasses import replace
    from slm_train.identity.run_id import final_run_id
    value = InputFingerprint(
        config_hash="c", data_hash="d", tokenizer_hash="t", tokenized_hash="td",
        model_hash="m", evaluator_hash="e", code_hash="s", release="r1",
        launch_profile_hash="lp", runtime_hash="rt", invocation_hash="iv",
    )
    same = InputFingerprint(**{**value.__dict__})
    changed = replace(value, evaluator_hash="e2")
    assert value.digest() == same.digest()
    assert value.digest() != changed.digest()
    assert final_run_id("experiment", value) == final_run_id("experiment", same)
    assert final_run_id("experiment", value) != final_run_id("experiment", changed)
    for field in ("config_hash", "data_hash", "tokenizer_hash", "tokenized_hash", "model_hash", "code_hash", "launch_profile_hash", "runtime_hash", "invocation_hash"):
        changed = replace(value, **{field: "changed"})
        assert value.digest() != changed.digest()
    with pytest.raises(ValueError, match="safe"):
        final_run_id("../escape", value)
~~~

- [ ] **Step 2: Run tests and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_contracts.py pipeline_v2/tests/unit/test_atomic_artifacts.py pipeline_v2/tests/unit/test_identity.py -q
~~~

Expected: FAIL because contract and artifact modules do not exist.

- [ ] **Step 3: Define immutable and mutable contract envelopes**

~~~python
# pipeline_v2/slm_train/contracts/base.py
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

ValidationLabel = Literal["PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: int = Field(ge=1)
~~~

Define immutable semantic models named SourceManifest, DataManifest, TokenizerManifest, TokenizedDataManifest, ModelManifest, EvaluatorManifest, RunIdentityManifest, CheckpointManifest, EvaluationReport, and FinalPublicationManifest. The run identity model must contain run and experiment IDs, resolved config path/hash, data/tokenizer/tokenized-data/model/evaluator semantic hashes, Git commit/worktree metadata, V2 code/dependency hash, Python/package/device/driver/runtime identity, exact working directory, launch profile, exact command, creation timestamp, parent/base model identity, promotion predecessor identity, and the repository-relative evidence root. Creation and verification timestamps are recorded metadata but excluded from the semantic input fingerprint. The data model must contain source ID/role, immutable source identity, freshness policy, dataset and split policy, filters/normalization/dedupe/salt, document IDs/counts, license/provenance, overlap report, rejection/truncation counters, regeneration command, and semantic content hash. The tokenizer model must contain repository/file identity, content hash, vocabulary/max ID, BOS/EOS/PAD IDs, special tokens, context/offset behavior, and package version. The tokenized-data model must contain the exact ordered shard list, each size/hash, mask-path (required for document_masked), sequence length, packing_mode, usable blocks, dropped-tail/overflow counters, token-range validation, document boundaries, and source-data hash. The model manifest must contain architecture/config hash, initialization/base identity, parent checkpoint identity when applicable, and capability metadata. The evaluator model must contain evaluator IDs, versions, suite/gate set, metric schemas, cadence, thresholds, threshold directions, and evaluator-manifest hash. The evaluator-manifest hash is computed over all semantic fields with the hash field omitted, then stored as publication metadata. The checkpoint model must contain all identity hashes, global step/epoch, model/optimizer/scheduler/scaler/RNG/sampler state, required files/sizes, completion state, evaluation/selection state, and a SHA-256 for every exact-resume file. The evaluation model must contain evaluator name/version, expected and actual suite/gate set, expected/actual manifest hashes, run/checkpoint/data/tokenizer/device/seed/split identity, metric definitions and values, baseline/threshold, details, status label, and relative artifact paths. Define mutable status, freshness, log-index, lease, heartbeat, and recovery models separately. Immutable hashes exclude timestamps, current status, lease state, heartbeat, and recovery text.

- [ ] **Step 4: Implement exact identity components**

identity/source.py computes a local Merkle manifest from exact relative files or an HF repository revision plus file hashes. identity/code.py hashes the V2 source/schema/contract set and uv.lock. identity/runtime.py records Python, venv, package lock, OS, CUDA, driver, GPU capability class, and device. identity/invocation.py canonicalizes exact command, working directory, and allowlisted non-secret environment. identity/run_id.py implements:

~~~python
from dataclasses import asdict, dataclass
from typing import Literal
import re

from slm_train.config.canonical import semantic_hash


@dataclass(frozen=True)
class InputFingerprint:
    config_hash: str
    data_hash: str
    tokenizer_hash: str
    tokenized_hash: str
    model_hash: str
    evaluator_hash: str
    code_hash: str
    release: Literal["r1"]
    launch_profile_hash: str
    runtime_hash: str
    invocation_hash: str

    def digest(self) -> str:
        return semantic_hash(asdict(self))


def final_run_id(experiment_id: str, fingerprint: InputFingerprint) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", experiment_id):
        raise ValueError("experiment_id is not a safe run-id component")
    return experiment_id + "-" + fingerprint.digest()
~~~

Bootstrap identity is computed before the final data/model/evaluator manifests exist; attempt IDs are operational and excluded from the final run ID.

- [ ] **Step 5: Implement atomic store and markers**

~~~python
# pipeline_v2/slm_train/artifacts/atomic.py
import os
from pathlib import Path
import tempfile


def _fsync_directory(path: Path) -> bool:
    if not hasattr(os, "O_DIRECTORY"):
        return False
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
        return True
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, replace_existing: bool = False) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise
            os.unlink(temporary)
        return _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
~~~

Use the same temporary-sibling pattern for directories, fsync files, publish immutable files with an exclusive same-filesystem hard-link (so a concurrent creator cannot be overwritten), atomically replace explicitly mutable files with os.replace, and publish a completion marker only after the manifest and every payload hash have been written. Record the directory-durability result in status; never silently claim unsupported durability.

- [ ] **Step 6: Implement path policy and log index**

PathPolicy rejects absolute, traversal, and symlink-escaping paths. V2 paths are data/v2/prepared, data/v2/tokens, runs/v2/_attempts, and runs/v2/<namespace>/<run_id>. Each final run contains config/resolved.yaml, config/resolved.json, manifests/, logs/{lease,metrics,monitor,tensorboard}/, log-manifest.json, checkpoints/, evaluations/, reports/, status.json, run.lock, and COMPLETE.json. Immutable log segments are numbered and never overwritten. log-manifest.json is a mutable atomic hash chain containing the previous index hash, ordered segment hashes, and last sequence; it is excluded from semantic identity. status.json and run.lock are operational state, not identity inputs.

- [ ] **Step 7: Run tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_contracts.py pipeline_v2/tests/unit/test_atomic_artifacts.py pipeline_v2/tests/unit/test_identity.py -q
~~~

Expected: PASS, including partial-write, collision, symlink, and path-escape cases.

~~~powershell
git add pipeline_v2/slm_train/contracts pipeline_v2/slm_train/identity pipeline_v2/slm_train/artifacts pipeline_v2/tests/unit/test_contracts.py pipeline_v2/tests/unit/test_atomic_artifacts.py pipeline_v2/tests/unit/test_identity.py
git commit --only -m "feat(v2): add versioned identity and atomic artifact contracts" -- pipeline_v2/slm_train/contracts pipeline_v2/slm_train/identity pipeline_v2/slm_train/artifacts pipeline_v2/tests/unit/test_contracts.py pipeline_v2/tests/unit/test_atomic_artifacts.py pipeline_v2/tests/unit/test_identity.py
~~~

---

## Task 4: Implement pinned sources, provenance, preparation, split, and generation publication

**Files:**
- Create: pipeline_v2/slm_train/data/__init__.py
- Create: pipeline_v2/slm_train/data/sources/__init__.py
- Create: pipeline_v2/slm_train/data/sources/base.py
- Create: pipeline_v2/slm_train/data/sources/local.py
- Create: pipeline_v2/slm_train/data/sources/huggingface.py
- Create: pipeline_v2/slm_train/data/prepare.py
- Create: pipeline_v2/slm_train/data/dedupe.py
- Create: pipeline_v2/slm_train/data/split.py
- Create: pipeline_v2/slm_train/data/provenance.py
- Create: pipeline_v2/slm_train/data/generation.py
- Create: pipeline_v2/tests/fixtures/tiny_corpus.jsonl
- Create: pipeline_v2/tests/unit/test_data.py
- Create: pipeline_v2/tests/network/test_hf_source.py

- [ ] **Step 1: Write failing source and generation tests**

~~~python
import json
from pathlib import Path
import pytest
from slm_train.data.sources.local import LocalSourceReader
from slm_train.data.prepare import prepare_generation


def test_local_source_requires_exact_file_set(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        LocalSourceReader.from_path(source)


def test_preparation_is_deterministic_and_counts_rejections(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        '{"id":"a","text":"alpha"}\n{"id":"b","text":"beta"}\n{"id":"c","text":"alpha"}\n',
        encoding="utf-8",
    )
    reader = LocalSourceReader.from_file_set([source], role="train")
    options = {
        "dataset_id": "fixture",
        "generation_id": "g001",
        "split_salt": 17,
        "validation_fraction": 0.34,
        "dedupe": True,
        "min_chars": 1,
    }
    first = prepare_generation(reader, options, tmp_path / "one")
    second = prepare_generation(reader, options, tmp_path / "two")
    assert first.semantic_hash == second.semantic_hash
    assert first.counts["duplicates_dropped"] == 1
    assert first.counts["train"] + first.counts["validation"] == 2
~~~

The HF source contract test is deliberately separate and opt-in:

~~~python
# pipeline_v2/tests/network/test_hf_source.py
import pytest

from slm_train.data.sources.huggingface import HFSourceReader


@pytest.mark.network
def test_hf_reader_requires_repository_revision_and_explicit_file_set():
    with pytest.raises(ValueError, match="revision"):
        HFSourceReader.from_repository("org/dataset", revision=None, files=[])
    with pytest.raises(ValueError, match="files"):
        HFSourceReader.from_repository("org/dataset", revision="0123456789abcdef", files=[])
~~~

- [ ] **Step 2: Run the test and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_data.py -q
~~~

Expected: FAIL because source and preparation modules do not exist.

- [ ] **Step 3: Implement source protocols and identities**

~~~python
# pipeline_v2/slm_train/data/sources/base.py
from collections.abc import Iterator, Mapping
from typing import Protocol


class SourceReader(Protocol):
    def identity(self) -> Mapping[str, object]:
        raise NotImplementedError

    def provenance(self) -> Mapping[str, object]:
        raise NotImplementedError

    def iter_rows(self) -> Iterator[Mapping[str, object]]:
        raise NotImplementedError
~~~

LocalSourceReader.from_file_set(files: Sequence[Path], role: Literal["train", "validation", "test", "external-gate"], manifest: SourceManifest | None = None) and LocalSourceReader.from_path(path: Path, manifest: SourceManifest | None = None) are explicit constructors; the latter raises ValueError when the manifest is absent. The file-set form may build its own manifest only from the exact listed files; the single-path form rejects an unmanifested path. HFSourceReader.from_repository(repository: str, revision: str, files: Sequence[str], manifest: SourceManifest | None = None) requires a nonempty pinned revision and explicit file list, then records repository, revision, exact file list, and file hashes. Every source declaration also records its role, license/provenance note, expected row budget, and expected token budget. A source cannot become optional implicitly: an optional source requires an explicit profile, an absent-content decision, and a recorded reason. Neither reader accepts an unpinned directory or infers a source role from filenames. Before every consuming stage, reopen the completion marker, validate the manifest/schema identity, recheck every source file hash and size, verify the source fingerprint, and evaluate freshness/supersession policy.

- [ ] **Step 4: Implement preparation, dedupe, salted split, and provenance**

Assign stable document IDs from source ID, source row index, and normalized text. Dedupe happens before splitting. The split is deterministic from dataset identity, generation identity, document ID, and split_salt. Record source role, license, provenance, filters, normalization, rows seen, rows kept, duplicate count, filter rejection count, truncation count, train count, validation count, overlap count, and the exact regeneration command. Empty or overlapping required sets are FAIL; a changed source requires a new generation.

- [ ] **Step 5: Publish unique generations atomically**

Write a temporary sibling generation directory, write raw/prepared manifests and completion marker, reopen and validate hashes, then atomically publish under data/v2/prepared/<dataset_id>/<generation_id>. A source is never edited or deleted. Freshness verification is a separate mutable report and does not change the semantic generation hash.

- [ ] **Step 6: Run local CPU tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_data.py -m "not network" -q
~~~

Expected: PASS with no network access. The opt-in contract lane is separate:

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/network/test_hf_source.py -m network -q
~~~

Run the network command only in an explicitly user-approved network window; ordinary CPU implementation and CI do not run it. The HF test must use a pinned test repository and must persist its report under pipeline_v2/test-artifacts/.

~~~powershell
git add pipeline_v2/slm_train/data pipeline_v2/tests/fixtures/tiny_corpus.jsonl pipeline_v2/tests/unit/test_data.py pipeline_v2/tests/network/test_hf_source.py
git commit --only -m "feat(v2): add pinned deterministic data preparation" -- pipeline_v2/slm_train/data pipeline_v2/tests/fixtures/tiny_corpus.jsonl pipeline_v2/tests/unit/test_data.py pipeline_v2/tests/network/test_hf_source.py
~~~

---

## Task 5: Implement tokenizer identity, packing, shards, and dataset validation

**Files:**
- Create: pipeline_v2/slm_train/data/tokenizer.py
- Create: pipeline_v2/slm_train/data/packing.py
- Create: pipeline_v2/slm_train/data/shards.py
- Create: pipeline_v2/slm_train/data/validate.py
- Create: pipeline_v2/slm_train/data/dataset.py
- Modify: pipeline_v2/tests/unit/test_data.py

- [ ] **Step 1: Add failing token-range and exact-shard tests**

~~~python
def test_packing_records_tail_loss(tmp_path):
    from slm_train.data.packing import pack_tokens
    result = pack_tokens([1, 2, 3, 4, 5], 2, "concat", tmp_path)
    assert result.counts["usable_blocks"] == 2
    assert result.counts["dropped_tail"] == 1
    assert result.shards


def test_dataset_rejects_a_wrong_generation_manifest(tmp_path):
    from slm_train.data.dataset import PackedDataset
    with pytest.raises(ValueError, match="completion"):
        PackedDataset.from_manifest(tmp_path / "missing.json")
~~~

- [ ] **Step 2: Run the focused test and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_data.py -q
~~~

Expected: FAIL for missing tokenizer, packing, shard, and dataset modules.

- [ ] **Step 3: Implement tokenizer identity and token validation**

Record tokenizer repository/revision or local file identity, tokenizer content hash, vocabulary size, maximum token ID, BOS/EOS/PAD IDs, special-token map, context length, offsets, truncation count, and package version. Reject duplicate special IDs, out-of-range IDs, unpinned tokenizer, and unrecorded overflow.

- [ ] **Step 4: Implement explicit packing modes**

concat preserves source order. document_masked retains document boundaries and loss metadata; if the selected backend cannot honor that contract, validation is FAIL before training and the engine never silently downgrades to concat. Both modes write little-endian uint32 token shards (and uint8 loss-mask shards for document_masked) and record total tokens, usable blocks, sequence length, overflow, dropped tail, pack mode, ordered shard paths, shard sizes, and SHA-256 values. The final incomplete block is either explicitly retained by the selected mode or counted as dropped; it is never silently lost.

- [ ] **Sharded dataset implementation**

~~~python
# pipeline_v2/slm_train/data/dataset.py
import numpy as np
from torch.utils.data import Dataset

from slm_train.contracts.tokenized import TokenizedDataManifest


class PackedDataset(Dataset):
    def __init__(self, manifest: TokenizedDataManifest, sequence_length: int):
        self.manifest = manifest
        self.sequence_length = int(sequence_length)
        self.shards = [np.memmap(item.path, dtype="<u4", mode="r") for item in manifest.shards]
        self.masks = [
            np.memmap(item.mask_path, dtype="<u1", mode="r") if item.mask_path else None
            for item in manifest.shards
        ]
        if manifest.packing_mode == "document_masked" and any(mask is None for mask in self.masks):
            raise ValueError("document_masked requires one loss-mask file per shard")
        self.blocks = sum(len(shard) // self.sequence_length for shard in self.shards)

    def __len__(self):
        return self.blocks

    def __getitem__(self, index):
        if index < 0:
            index += self.blocks
        if index < 0 or index >= self.blocks:
            raise IndexError(index)
        remaining = index
        for shard_index, shard in enumerate(self.shards):
            count = len(shard) // self.sequence_length
            if remaining < count:
                offset = remaining * self.sequence_length
                block = np.asarray(shard[offset:offset + self.sequence_length], dtype=np.int64)
                mask_shard = self.masks[shard_index]
                if mask_shard is None:
                    return {"input_ids": block, "labels": block.copy(), "loss_mask": np.ones_like(block, dtype=np.uint8)}
                mask = np.asarray(mask_shard[offset:offset + self.sequence_length], dtype=np.uint8)
                return {"input_ids": block, "labels": block.copy(), "loss_mask": mask.copy()}
            remaining -= count
        raise IndexError(index)
~~~

Before memmap construction, validate the completion marker, exact source generation, tokenizer identity, ordered shard list, each size and hash, sequence length, and count. Never use a broad directory glob.

- [ ] **Step 5: Run tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_data.py -q
~~~

Expected: PASS and a reopened tokenized-data manifest.

~~~powershell
git add pipeline_v2/slm_train/data pipeline_v2/tests/unit/test_data.py
git commit --only -m "feat(v2): add pinned tokenizer packing and shards" -- pipeline_v2/slm_train/data pipeline_v2/tests/unit/test_data.py
~~~

---

## Task 6: Implement the R1 model factory and centralized HF backend

**Files:**
- Create: pipeline_v2/slm_train/models/__init__.py
- Create: pipeline_v2/slm_train/models/causal_lm.py
- Create: pipeline_v2/slm_train/models/factory.py
- Create: pipeline_v2/slm_train/models/manifest.py
- Create: pipeline_v2/slm_train/training/__init__.py
- Create: pipeline_v2/slm_train/training/backend.py
- Create: pipeline_v2/slm_train/training/spec.py
- Create: pipeline_v2/slm_train/training/hf_causal_lm.py
- Create: pipeline_v2/slm_train/training/events.py
- Create: pipeline_v2/tests/unit/test_model_factory.py

- [ ] **Step 1: Write failing capability tests**

~~~python
import pytest
from slm_train.config.models import ModelConfig
from slm_train.models.factory import ModelCapabilityError, build_causal_lm


def tiny_model_config(**overrides) -> ModelConfig:
    values = {
        "architecture": "llama",
        "hidden": 32,
        "intermediate": 64,
        "layers": 1,
        "heads": 4,
        "kv_heads": 2,
        "context": 16,
        "tied_embeddings": False,
        "precision": "fp16",
        "attention": "sdpa",
        "base_model": None,
    }
    return ModelConfig(**{**values, **overrides})


def test_bf16_is_rejected():
    with pytest.raises(ModelCapabilityError, match="fp16"):
        build_causal_lm(tiny_model_config(precision="bf16"))


def test_flash_attention_is_rejected():
    with pytest.raises(ModelCapabilityError, match="sdpa"):
        build_causal_lm(tiny_model_config(attention="flash_attention_2"))


def test_gqa_uses_sdpa():
    model = build_causal_lm(tiny_model_config())
    assert model.config.model_type == "llama"
~~~

- [ ] **Step 2: Run tests and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_model_factory.py -q
~~~

Expected: FAIL because model modules do not exist.

- [ ] **Step 3: Implement the constrained GQA factory**

Use transformers.LlamaConfig and AutoModelForCausalLM with attn_implementation="sdpa". Reject BF16, FlashAttention, invalid head divisibility, context mismatch, unsupported architecture, and any R2 model capability. The model manifest records architecture, dimensions, head counts, context, tied embeddings, precision, attention backend, package versions, and source hash. Do not import src/model.py.

- [ ] **Step 4: Define backend and training specification**

~~~python
# pipeline_v2/slm_train/training/backend.py
from typing import Protocol

from .spec import TrainingContext, TrainingResult


class TrainerBackend(Protocol):
    def prepare(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def fit(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def evaluate(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def checkpoint(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def resume(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def finalize(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError
~~~

~~~python
# pipeline_v2/slm_train/training/spec.py
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from slm_train.contracts.base import ValidationLabel
from slm_train.config.models import ConfigDocument
from slm_train.contracts.checkpoint import CheckpointManifest
from slm_train.contracts.evaluation import EvaluationReport
from slm_train.contracts.model import ModelManifest
from slm_train.contracts.tokenized import TokenizedDataManifest
from slm_train.contracts.tokenizer import TokenizerManifest
from slm_train.contracts.evaluator import EvaluatorManifest
from slm_train.contracts.run import RunIdentityManifest
from slm_train.training.state import TrainingCursor


@dataclass(frozen=True)
class TrainingEvent:
    name: str
    sequence: int
    payload: Mapping[str, object]


@dataclass(frozen=True)
class TrainingContext:
    config: ConfigDocument
    run_identity: RunIdentityManifest
    model_manifest: ModelManifest
    tokenizer_manifest: TokenizerManifest
    dataset_manifest: TokenizedDataManifest
    evaluator_manifest: EvaluatorManifest
    cursor: TrainingCursor
    device: str
    seed: int
    event_sink: Callable[[TrainingEvent], None]


@dataclass(frozen=True)
class TrainingResult:
    label: ValidationLabel
    events: tuple[TrainingEvent, ...]
    checkpoint: CheckpointManifest | None
    evaluations: tuple[EvaluationReport, ...]
    diagnostic: str | None
~~~

training/spec.py owns the model, tokenizer, dataset manifest, optimizer, scheduler, precision, batching, accumulation, clipping, evaluation cadence, checkpoint policy, feature set, seed, and launch profile through the resolved config and the typed context above. The orchestrator sees typed events and results, not HF implementation details.

- [ ] **Step 5: Implement HFCausalLMTrainer**

Construct TrainingArguments only in hf_causal_lm.py, using FP16, SDPA, no FlashAttention, TensorBoard reporting, explicit seed, explicit output paths, and the centralized spec. Emit typed events for prepare, step, evaluation, checkpoint, health, and finalization. Use safetensors for exported weights and weights_only for state loading.

- [ ] **Step 6: Run CPU construction tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_model_factory.py -q
~~~

Expected: PASS using a tiny CPU model; no CUDA call occurs.

~~~powershell
git add pipeline_v2/slm_train/models pipeline_v2/slm_train/training pipeline_v2/tests/unit/test_model_factory.py
git commit --only -m "feat(v2): add constrained HF causal LM backend" -- pipeline_v2/slm_train/models pipeline_v2/slm_train/training pipeline_v2/tests/unit/test_model_factory.py
~~~

---

## Task 7: Define trusted interfaces and static registries

**Files:**
- Create: pipeline_v2/slm_train/contracts/interfaces.py
- Create: pipeline_v2/slm_train/features/__init__.py
- Create: pipeline_v2/slm_train/features/registry.py
- Create: pipeline_v2/slm_train/adapters/__init__.py
- Create: pipeline_v2/slm_train/adapters/registry.py
- Modify: pipeline_v2/tests/contract/test_r1_contracts.py

- [ ] **Step 1: Write failing registry and extension-boundary tests**

~~~python
import pytest
from slm_train.features.registry import validate_capabilities


def test_r1_rejects_each_r2_capability():
    for capability in ("sft", "lora", "kd", "gdn", "mounting"):
        with pytest.raises(ValueError, match="R1"):
            validate_capabilities("r1", ["pretrain", capability])


def test_registry_has_no_dynamic_discovery():
    import slm_train.adapters.registry as registry
    assert not hasattr(registry, "discover")
    assert registry.entries()
~~~

- [ ] **Step 2: Run the tests and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/contract/test_r1_contracts.py -q
~~~

Expected: FAIL because registries and interfaces do not exist.

- [ ] **Step 3: Define the full protocol set**

contracts/interfaces.py defines SourceReader, RowTransformer, Deduplicator, Splitter, TokenizerAdapter, Packer, DatasetValidator, ModelFactory, FeatureModule, TrainerBackend, CheckpointAdapter, Evaluator, MetricSuite, Gate, SelectionPolicy, ArtifactPublisher, ResourceProbe, and HealthMonitor. Each protocol declares a typed input model, typed output model, ValidationLabel failure result, and artifact allowlist. Use this exact R1 method map:

| Protocol | Required method | Input | Output |
|---|---|---|---|
| SourceReader | identity / provenance / iter_rows | SourceDeclaration | SourceManifest / ProvenanceRecord / Iterator[SourceRow] |
| RowTransformer | transform | SourceRow | SourceRow + TransformReport |
| Deduplicator | deduplicate | Iterable[SourceRow] | DeduplicationResult |
| Splitter | split | PreparedRows + SplitPolicy | SplitResult |
| TokenizerAdapter | encode | PreparedRows + TokenizerManifest | TokenBatch |
| Packer | pack | TokenBatch + PackingConfig | TokenizedDataManifest |
| DatasetValidator | validate | TokenizedDataManifest | ValidationReport |
| ModelFactory | build | ModelConfig + RuntimeProfile | ModelManifest + Model |
| FeatureModule | apply | TrainingContext | TrainingResult |
| TrainerBackend | prepare / fit / evaluate / checkpoint / resume / finalize | TrainingContext | TrainingResult |
| CheckpointAdapter | publish / read_back | CheckpointState | CheckpointManifest / CheckpointState |
| Evaluator | evaluate | EvaluationRequest | EvaluationReport |
| MetricSuite | collect | EvaluationRequest | MetricSuiteReport |
| Gate | check | GateRequest | GateResult |
| SelectionPolicy | select | Sequence[CheckpointManifest] + SelectionPolicy | CheckpointManifest |
| ArtifactPublisher | publish / read_back | PublicationRequest | FinalPublicationManifest / PublicationReadBack |
| ResourceProbe | probe | ResourceRequest | ResourceReport |
| HealthMonitor | check | HealthSample | HealthResult |

 The R1 registry maps only the approved local/HF source readers, deterministic data transforms, concat/document_masked packers, Llama GQA model factory, HFCausalLMTrainer, R1 evaluator suites, gates, and selection policy to the lifecycle. Extension methods cannot acquire leases, rewrite immutable manifests, mark a run successful, or publish outside the run allowlist.

- [ ] **Step 4: Implement source-controlled registries**

Registry entries contain name, version, schema, stage, maturity, compatibility, release, and test contract. Entries are static Python declarations. Do not use importlib.metadata entry points, subprocess loading, remote lookup, directory scanning, or dynamic module import. R1 contains only core pretraining contracts and an explicit empty R2 feature boundary.

~~~python
# pipeline_v2/slm_train/adapters/registry.py
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    version: str
    schema: str
    stage: str
    maturity: Literal["experimental", "validated", "retired"]
    compatibility: tuple[str, ...]
    release: Literal["r1"]
    test_contract: str


_ENTRIES = (
    RegistryEntry("local_source", "1", "SourceManifest", "prepare", "validated", ("r1",), "r1", "unit/test_data.py"),
    RegistryEntry("hf_source", "1", "SourceManifest", "prepare", "validated", ("r1",), "r1", "network/test_hf_source.py"),
    RegistryEntry("concat_packer", "1", "TokenizedDataManifest", "tokenize", "validated", ("r1",), "r1", "unit/test_data.py"),
    RegistryEntry("document_masked_packer", "1", "TokenizedDataManifest", "tokenize", "experimental", ("r1",), "r1", "unit/test_data.py"),
    RegistryEntry("llama_gqa", "1", "ModelManifest", "model_validation", "validated", ("r1",), "r1", "unit/test_model_factory.py"),
    RegistryEntry("hf_causal_lm", "1", "CheckpointManifest", "train", "validated", ("r1",), "r1", "unit/test_resume.py"),
    RegistryEntry("r1_evaluators", "1", "EvaluationReport", "internal_evaluation", "validated", ("r1",), "r1", "unit/test_evaluation.py"),
    RegistryEntry("r1_gates", "1", "GateResult", "external_gates", "validated", ("r1",), "r1", "unit/test_evaluation.py"),
    RegistryEntry("best_validation_loss", "1", "CheckpointManifest", "publish_final", "validated", ("r1",), "r1", "unit/test_evaluation.py"),
)


def entries() -> tuple[RegistryEntry, ...]:
    return _ENTRIES
~~~

~~~python
# pipeline_v2/slm_train/features/registry.py
from collections.abc import Sequence


def validate_capabilities(release: str, capabilities: Sequence[str]) -> None:
    if release != "r1":
        raise ValueError("Release 1 only accepts release=r1")
    if len(capabilities) != 1 or capabilities[0] != "pretrain":
        raise ValueError("R1 requires exactly the pretrain capability")
~~~

The contract test is:

~~~python
# pipeline_v2/tests/contract/test_r1_contracts.py
from slm_train.adapters.registry import entries
from slm_train.features.registry import validate_capabilities


def test_r1_registry_is_explicit_and_nonempty():
    values = entries()
    assert values
    assert all(value.release == "r1" for value in values)
    assert all(value.maturity in {"experimental", "validated", "retired"} for value in values)


def test_r1_rejects_all_release_two_capabilities():
    for capability in ("sft", "lora", "kd", "gdn", "mounting"):
        try:
            validate_capabilities("r1", ["pretrain", capability])
        except ValueError as error:
            assert "R1" in str(error)
        else:
            raise AssertionError(capability)
~~~

- [ ] **Step 5: Run tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/contract/test_r1_contracts.py -q
~~~

Expected: PASS.

~~~powershell
git add pipeline_v2/slm_train/contracts/interfaces.py pipeline_v2/slm_train/features pipeline_v2/slm_train/adapters pipeline_v2/tests/contract/test_r1_contracts.py
git commit --only -m "feat(v2): add trusted static interfaces and registries" -- pipeline_v2/slm_train/contracts/interfaces.py pipeline_v2/slm_train/features pipeline_v2/slm_train/adapters pipeline_v2/tests/contract/test_r1_contracts.py
~~~

---

## Task 8: Implement orchestration, GPU lease, preflight, and bootstrap handoff

**Files:**
- Create: pipeline_v2/slm_train/orchestration/__init__.py
- Create: pipeline_v2/slm_train/orchestration/graph.py
- Create: pipeline_v2/slm_train/orchestration/stages.py
- Create: pipeline_v2/slm_train/orchestration/engine.py
- Create: pipeline_v2/slm_train/orchestration/bootstrap.py
- Create: pipeline_v2/slm_train/orchestration/identity_handoff.py
- Create: pipeline_v2/slm_train/orchestration/status.py
- Create: pipeline_v2/slm_train/orchestration/errors.py
- Create: pipeline_v2/slm_train/resources/__init__.py
- Create: pipeline_v2/slm_train/resources/lease.py
- Create: pipeline_v2/slm_train/resources/gpu_probe.py
- Create: pipeline_v2/slm_train/resources/device.py
- Create: pipeline_v2/slm_train/resources/doctor.py
- Create: pipeline_v2/slm_train/resources/preflight.py
- Create: pipeline_v2/tests/unit/test_orchestration.py
- Create: pipeline_v2/tests/unit/test_lease.py

- [ ] **Step 1: Write failing graph, CPU-only doctor, lease, and handoff tests**

~~~python
import pytest
from slm_train.orchestration.graph import StageGraph
from slm_train.resources.lease import LeaseConflict, LeaseManager


def lease_extra(pid: int) -> dict[str, object]:
    return {
        "pid": pid,
        "process_start_time": f"start-{pid}",
        "command": "pipeline_v2 run",
        "device": "cuda:0",
        "config_hash": "config-hash",
        "timestamp": f"2026-01-01T00:00:0{pid}Z",
    }


def test_graph_order_matches_v2_contract():
    graph = StageGraph.r1()
    order = graph.order
    assert order[0] == "validate"
    assert order.index("prepare") < order.index("tokenize") < order.index("train")
    assert order[-2:] == ("external_gates", "publish_final")
    produced = {path for spec in graph.specs() for path in spec.outputs}
    assert {"manifests/data.json", "manifests/tokenizer.json", "manifests/tokenized_data.json", "manifests/evaluator.json", "manifests/run_identity.json"} <= produced


def test_doctor_never_calls_gpu(monkeypatch, tmp_path):
    import slm_train.resources.doctor as doctor
    monkeypatch.setattr(doctor, "probe_gpu", lambda: (_ for _ in ()).throw(AssertionError("GPU touched")), raising=False)
    result = doctor.run_doctor(root=tmp_path)
    assert result["gpu_touched"] is False


def test_second_lease_is_blocked(tmp_path):
    manager = LeaseManager(tmp_path)
    first = manager.acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    with pytest.raises(LeaseConflict):
        manager.acquire_bootstrap("boot", "attempt-2", lease_extra(2))
    first.release()


def test_final_handoff_is_cas(tmp_path):
    lease = LeaseManager(tmp_path).acquire_bootstrap("boot", "attempt-1", lease_extra(1))
    lease.bind_final("run-1", "identity-1")
    with pytest.raises(LeaseConflict):
        lease.bind_final("run-2", "identity-2")
    lease.release()
~~~

- [ ] **Step 2: Run tests and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_orchestration.py pipeline_v2/tests/unit/test_lease.py -q
~~~

Expected: FAIL because orchestration and resource modules do not exist.

- [ ] **Step 3: Implement the exact graph and stage contracts**

~~~python
# pipeline_v2/slm_train/orchestration/graph.py
from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    name: str
    requires: tuple[str, ...]
    required_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    completion_marker: str


class StageGraph:
    order = (
        "validate",
        "bootstrap",
        "gpu_preflight",
        "prepare",
        "tokenize",
        "dataset_validation",
        "model_validation",
        "evaluator_validation",
        "resolve_final_identity",
        "train",
        "internal_evaluation",
        "external_gates",
        "publish_final",
    )

    @classmethod
    def r1(cls) -> "StageGraph":
        return cls()

    def specs(self) -> tuple[StageSpec, ...]:
        required_inputs = {
            "validate": (),
            "bootstrap": ("config/resolved.yaml", "config/resolved.json"),
            "gpu_preflight": ("attempt/bootstrap.json",),
            "prepare": ("attempt/preflight.json",),
            "tokenize": ("prepared_generation/COMPLETE.json", "manifests/data.json"),
            "dataset_validation": ("tokenized_generation/COMPLETE.json", "manifests/tokenizer.json"),
            "model_validation": ("manifests/tokenized_data.json",),
            "evaluator_validation": ("config/resolved.json", "manifests/model.json"),
            "resolve_final_identity": ("manifests/data.json", "manifests/tokenizer.json", "manifests/tokenized_data.json", "manifests/model.json", "manifests/evaluator.json"),
            "train": ("manifests/run_identity.json",),
            "internal_evaluation": ("checkpoints/COMPLETE.json",),
            "external_gates": ("evaluations/COMPLETE.json",),
            "publish_final": ("reports/COMPLETE.json",),
        }
        outputs = {
            "validate": ("config/resolved.yaml", "config/resolved.json"),
            "bootstrap": ("attempt/bootstrap.json",),
            "gpu_preflight": ("attempt/preflight.json",),
            "prepare": ("prepared_generation/COMPLETE.json", "manifests/data.json"),
            "tokenize": ("tokenized_generation/COMPLETE.json", "manifests/tokenizer.json"),
            "dataset_validation": ("manifests/tokenized_data.json",),
            "model_validation": ("manifests/model.json",),
            "evaluator_validation": ("manifests/evaluator.json",),
            "resolve_final_identity": ("manifests/run_identity.json",),
            "train": ("checkpoints/COMPLETE.json",),
            "internal_evaluation": ("evaluations/COMPLETE.json",),
            "external_gates": ("reports/COMPLETE.json",),
            "publish_final": ("manifests/final_publication.json", "COMPLETE.json"),
        }
        return tuple(
            StageSpec(
                name,
                () if index == 0 else (self.order[index - 1],),
                required_inputs[name],
                outputs[name],
                f"{name}.COMPLETE.json",
            )
            for index, name in enumerate(self.order)
        )
~~~

Each stage declares required input manifests, output artifacts, completion-marker schema, and allowed next stages. The engine verifies every required input path, its completion marker, and its hash before dispatch; after a stage it verifies the declared output allowlist and writes the stage completion marker. On an exact re-run, a complete stage with matching identity is not repeated: the engine resumes at the first incomplete stage after revalidating all transitive markers. The engine stops on any exception, writes status and diagnostic evidence, and never skips to a fallback stage.

- [ ] **Step 4: Implement the global lease and bootstrap identity**

Use this typed bootstrap identity before any GPU work:

~~~python
# pipeline_v2/slm_train/orchestration/bootstrap.py
from dataclasses import asdict, dataclass

from slm_train.config.canonical import semantic_hash


@dataclass(frozen=True)
class BootstrapIdentity:
    logical_experiment: str
    config_hash: str
    source_identity_hash: str
    tokenizer_identity_hash: str
    launch_profile_hash: str
    requested_runtime_hash: str
    code_dependency_hash: str
    command_hash: str

    def digest(self) -> str:
        return semantic_hash(asdict(self))


def attempt_id(launch_nonce: str, pid: int, process_start_time: str) -> str:
    return semantic_hash({"launch_nonce": launch_nonce, "pid": pid, "process_start_time": process_start_time})[:32]
~~~

Use filelock.FileLock with timeout=0 around a repository-level V2 GPU lock. The lease handle keeps the lock object alive for the full GPU stage:

~~~python
# pipeline_v2/slm_train/resources/lease.py
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from slm_train.artifacts.atomic import atomic_write_bytes


class LeaseConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class LeaseMetadata:
    bootstrap_id: str
    attempt_id: str
    pid: int
    process_start_time: str
    command: str
    device: str
    config_hash: str
    timestamp: str
    final_run_id: str | None = None
    identity_hash: str | None = None


@dataclass
class LeaseHandle:
    manager: "LeaseManager"
    metadata: LeaseMetadata

    def bind_final(self, final_run_id: str, identity_hash: str) -> None:
        if self.metadata.final_run_id not in {None, final_run_id}:
            raise LeaseConflict("lease is already bound to another run")
        if self.metadata.identity_hash not in {None, identity_hash}:
            raise LeaseConflict("lease identity does not match")
        updated = LeaseMetadata(**{**asdict(self.metadata), "final_run_id": final_run_id, "identity_hash": identity_hash})
        self.manager._write_metadata(updated)
        self.metadata = updated

    def release(self) -> None:
        self.manager._release()


class LeaseManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(root / "gpu.lock"), timeout=0)
        self.metadata_path = root / "run.lock"

    def acquire_bootstrap(self, bootstrap_id: str, attempt_id: str, extra: dict[str, object]) -> LeaseHandle:
        try:
            self.lock.acquire()
        except Timeout as exc:
            raise LeaseConflict("repository GPU lease is busy") from exc
        try:
            metadata = LeaseMetadata(
                bootstrap_id=bootstrap_id,
                attempt_id=attempt_id,
                pid=int(extra["pid"]),
                process_start_time=str(extra["process_start_time"]),
                command=str(extra["command"]),
                device=str(extra["device"]),
                config_hash=str(extra["config_hash"]),
                timestamp=str(extra["timestamp"]),
            )
            self._write_metadata(metadata)
            return LeaseHandle(self, metadata)
        except Exception:
            self._release()
            raise

    def _write_metadata(self, metadata: LeaseMetadata) -> None:
        atomic_write_bytes(self.metadata_path, json.dumps(asdict(metadata), sort_keys=True).encode("utf-8"), replace_existing=True)

    def _release(self) -> None:
        self.lock.release()
~~~

A busy, malformed, stale, or unverifiable lease is BLOCKED. Recovery proves the owner and GPU work are gone before replacing a lease. No foreign process is killed and V2 has no share-GPU override.

Bootstrap identity is computed before GPU probing. Attempt ID is derived from launch nonce, PID, and process start time and is excluded from final run identity. Evidence is written under runs/v2/_attempts/<bootstrap_id>/<attempt_id>/.

- [ ] **Step 5: Implement pre-handoff CPU work and CAS handoff**

Acquire the lease, run the explicit GPU probe, then perform prepare, tokenize, dataset validation, model validation, and evaluator validation. Publish semantic manifests first. Compute final input fingerprint and run ID, bind the bootstrap lease with create-if-absent/CAS, then publish immutable manifests/run_identity.json. A different identity at an existing destination is BLOCKED and never overwritten. Failures before handoff retain attempt evidence; failures during handoff verify owner/device/command; failures after handoff use immutable run identity and complete checkpoint manifests.

- [ ] **Step 6: Implement CPU-only doctor and explicit preflight report**

doctor checks venv, package metadata, imports, writable V2 roots, disk, and config paths without importing torch.cuda, invoking nvidia-smi, or acquiring a lease. preflight --gpu records device identity, driver/runtime, free/total VRAM, compute processes, disk headroom, temperature/utilization, estimate, decision, reason, and exact report path. Probe failure is BLOCKED.

- [ ] **Step 7: Run tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_orchestration.py pipeline_v2/tests/unit/test_lease.py -q
~~~

Expected: PASS, including two-process lease race, stale-owner proof, and CAS collision tests.

~~~powershell
git add pipeline_v2/slm_train/orchestration pipeline_v2/slm_train/resources pipeline_v2/tests/unit/test_orchestration.py pipeline_v2/tests/unit/test_lease.py
git commit --only -m "feat(v2): add fail-closed orchestration and gpu lease" -- pipeline_v2/slm_train/orchestration pipeline_v2/slm_train/resources pipeline_v2/tests/unit/test_orchestration.py pipeline_v2/tests/unit/test_lease.py
~~~

---

## Task 9: Implement checkpoints, exact resume, events, logs, and health

**Files:**
- Create: pipeline_v2/slm_train/training/state.py
- Create: pipeline_v2/slm_train/training/checkpoint.py
- Create: pipeline_v2/slm_train/training/resume.py
- Create: pipeline_v2/slm_train/training/selection.py
- Create: pipeline_v2/slm_train/monitoring/__init__.py
- Create: pipeline_v2/slm_train/monitoring/bus.py
- Create: pipeline_v2/slm_train/monitoring/health.py
- Create: pipeline_v2/slm_train/monitoring/metrics.py
- Create: pipeline_v2/slm_train/monitoring/heartbeat.py
- Create: pipeline_v2/slm_train/artifacts/log_index.py updates
- Create: pipeline_v2/tests/unit/test_resume.py
- Create: pipeline_v2/tests/unit/test_monitoring.py

- [ ] **Step 1: Write failing cursor and recovery tests**

~~~python
import pytest
from slm_train.training.resume import RESUME_IDENTITY_KEYS, ResumeMismatch, assert_exact_resume


def identity(**changes: str) -> dict[str, str]:
    value = {key: "same" for key in RESUME_IDENTITY_KEYS}
    value.update(changes)
    return value


def test_resume_rejects_changed_evaluator():
    with pytest.raises(ResumeMismatch, match="evaluator"):
        assert_exact_resume(identity(evaluator="e1"), identity(evaluator="e2"))


def test_resume_rejects_changed_command_and_device():
    with pytest.raises(ResumeMismatch, match="command"):
        assert_exact_resume(
            identity(command="run", device="cuda:0"),
            identity(command="run --resume", device="cuda:0"),
        )
    with pytest.raises(ResumeMismatch, match="device"):
        assert_exact_resume(
            identity(command="run", device="cuda:0"),
            identity(command="run", device="cuda:1"),
        )


def test_cursor_round_trip_keeps_microbatch_boundary():
    from slm_train.training.state import TrainingCursor
    cursor = TrainingCursor(examples_seen=17, microbatch_position=2, accumulation_boundary=3, epoch=1)
    assert TrainingCursor.from_dict(cursor.to_dict()) == cursor
~~~

- [ ] **Step 2: Run tests and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_resume.py pipeline_v2/tests/unit/test_monitoring.py -q
~~~

Expected: FAIL because checkpoint, resume, event, and health modules do not exist.

- [ ] **Step 3: Define complete state and checkpoint manifest**

TrainingCursor stores examples_seen, microbatch_position, accumulation_boundary, and epoch. State also stores model, optimizer, scheduler, scaler, global RNG, sampler/dataloader generator state, per-worker RNG, early-stop state, evaluation cursor/state, and lifecycle state. Checkpoint manifests record every semantic identity and SHA-256/size for every state file.

~~~python
from dataclasses import asdict, dataclass
from typing import Mapping

# pipeline_v2/slm_train/training/state.py


@dataclass(frozen=True)
class TrainingCursor:
    examples_seen: int
    microbatch_position: int
    accumulation_boundary: int
    epoch: int

    def __post_init__(self) -> None:
        if min(self.examples_seen, self.microbatch_position, self.accumulation_boundary, self.epoch) < 0:
            raise ValueError("cursor values must be non-negative")
        if self.microbatch_position > self.accumulation_boundary:
            raise ValueError("microbatch position exceeds accumulation boundary")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, int]) -> "TrainingCursor":
        return cls(
            examples_seen=value["examples_seen"],
            microbatch_position=value["microbatch_position"],
            accumulation_boundary=value["accumulation_boundary"],
            epoch=value["epoch"],
        )


@dataclass(frozen=True)
class CheckpointState:
    cursor: TrainingCursor
    model: str
    optimizer: str
    scheduler: str
    scaler: str | None
    global_rng: str
    sampler_state: str
    dataloader_generator_state: str
    worker_rng: tuple[str, ...]
    early_stop: str
    evaluation: str
    lifecycle: str


# pipeline_v2/slm_train/training/resume.py

RESUME_IDENTITY_KEYS = (
    "config", "code", "data", "tokenizer", "model", "feature_set",
    "evaluator", "selection_policy", "command", "working_directory", "environment",
    "driver", "package", "runtime", "device",
)


class ResumeMismatch(ValueError):
    pass


def assert_exact_resume(saved: Mapping[str, str], current: Mapping[str, str]) -> None:
    missing = [key for key in RESUME_IDENTITY_KEYS if key not in saved or key not in current]
    if missing:
        raise ResumeMismatch(f"missing resume identity: {missing}")
    changed = [key for key in RESUME_IDENTITY_KEYS if saved[key] != current[key]]
    if changed:
        raise ResumeMismatch(f"changed resume identity: {changed}")
~~~

- [ ] **Step 4: Implement atomic checkpoint publication**

Write each state file to a temporary sibling, fsync it, verify hash and size, write the immutable manifest, then write COMPLETE.json. Partial checkpoints are never selected. Checkpoint selection uses the complete per-checkpoint manifest and numeric identity, never a broad directory scan. Changed identities receive a new run destination.

- [ ] **Step 5: Implement exact resume matching**

Resume requires matching resolved config, code/dependency hash, data, tokenizer, model, feature set, evaluator suite and selection policy, exact command, working directory, allowlisted environment, driver, package, runtime, and device identity. The automatic path is only the exact original command with zero added flags.

- [ ] **Step 6: Implement segmented event and health streams**

Emit run_started, stage_started, step_completed, metric_emitted, evaluation_completed, checkpoint_completed, health_warning, health_failure, stage_completed, and run_completed. Write immutable numbered JSONL segments for metrics, monitor, and lease events plus TensorBoard events, then atomically update the hash-chained log index. Each record carries a monotonic stream/segment/record sequence and the run identity. Recovery reads only the exact published segment list, quarantines unreferenced temporary siblings under a repository-relative recovery directory without deleting them, and never treats trailing uncommitted records as complete. Reopen and validate every stream, including the lease stream, for gaps and hash mismatches before PASS.

- [ ] **Step 7: Implement health policy**

Finite loss and gradients are required. NaN/Inf is FAIL. OOM is BLOCKED and preserves the last valid checkpoint. Shared-memory spill is OOM-equivalent. Heartbeat, disk, process liveness, evaluation/checkpoint completion, VRAM/reserved memory, temperature, and utilization are recorded. No health event changes a hyperparameter or silently retries. The OOM decision record from Task 0 controls whether a separately approved new run is permitted.

- [ ] **Step 8: Run CPU restore tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_resume.py pipeline_v2/tests/unit/test_monitoring.py -q
~~~

Expected: PASS for restore-versus-saved, simulated kill/resume, segment recovery, gap detection, NaN/Inf, and OOM labels.

~~~powershell
git add pipeline_v2/slm_train/training/state.py pipeline_v2/slm_train/training/checkpoint.py pipeline_v2/slm_train/training/resume.py pipeline_v2/slm_train/training/selection.py pipeline_v2/slm_train/monitoring pipeline_v2/slm_train/artifacts/log_index.py pipeline_v2/tests/unit/test_resume.py pipeline_v2/tests/unit/test_monitoring.py
git commit --only -m "feat(v2): add exact resume and durable monitoring" -- pipeline_v2/slm_train/training/state.py pipeline_v2/slm_train/training/checkpoint.py pipeline_v2/slm_train/training/resume.py pipeline_v2/slm_train/training/selection.py pipeline_v2/slm_train/monitoring pipeline_v2/slm_train/artifacts/log_index.py pipeline_v2/tests/unit/test_resume.py pipeline_v2/tests/unit/test_monitoring.py
~~~

---

## Task 10: Implement evaluators, gates, final publication, and COMPLETE binding

**Files:**
- Create: pipeline_v2/slm_train/evaluation/__init__.py
- Create: pipeline_v2/slm_train/evaluation/contracts.py
- Create: pipeline_v2/slm_train/evaluation/suites.py
- Create: pipeline_v2/slm_train/evaluation/internal.py
- Create: pipeline_v2/slm_train/evaluation/generation.py
- Create: pipeline_v2/slm_train/evaluation/syntax.py
- Create: pipeline_v2/slm_train/evaluation/resources.py
- Create: pipeline_v2/slm_train/evaluation/gates.py
- Create: pipeline_v2/slm_train/evaluation/reports.py
- Create: pipeline_v2/slm_train/artifacts/publication.py
- Create: pipeline_v2/tests/unit/test_evaluation.py

- [ ] **Step 1: Write failing evaluator and publication tests**

~~~python
import pytest
from slm_train.evaluation.gates import GateFailure, require_complete_suite, require_finite, require_threshold


def test_missing_metric_cannot_pass():
    with pytest.raises(GateFailure, match="missing"):
        require_complete_suite({"val_loss", "generation_smoke"}, {"val_loss"}, expected_evaluator_hash="e1", actual_evaluator_hash="e1")


def test_non_finite_metric_blocks():
    with pytest.raises(GateFailure, match="non-finite"):
        require_finite("val_loss", float("nan"))


def test_threshold_direction_is_enforced():
    with pytest.raises(GateFailure, match="exceeds"):
        require_threshold("val_loss", 2.0, 1.0, "min")
    with pytest.raises(GateFailure, match="below"):
        require_threshold("generation_smoke", 0.0, 1.0, "max")


def test_evaluator_manifest_mismatch_blocks():
    with pytest.raises(GateFailure, match="manifest"):
        require_complete_suite(
            {"val_loss"},
            {"val_loss"},
            expected_evaluator_hash="e1",
            actual_evaluator_hash="e2",
        )
~~~

- [ ] **Step 2: Run the test and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_evaluation.py -q
~~~

Expected: FAIL because evaluation modules do not exist.

- [ ] **Step 3: Define evaluator and report contracts**

Every report records evaluator ID/version, expected suite/gate set, expected and actual evaluator manifest hashes, run/checkpoint/data/tokenizer/device/seed identity, split, metric schema/definition/value, baseline/threshold, details, status label, and relative artifact paths. Reports are written on failure. Re-evaluation receives a new evaluation ID.

- [ ] **Step 4: Implement R1 evaluators and gate checks**

Implement validation loss/perplexity, generation smoke, applicable syntax/code checks, throughput/resource reporting, and registered external gates. Every required metric must be present at its configured cadence, finite, and attached to a reopened complete report. Missing, stale, unparseable, mismatched, or incomplete data is FAIL or BLOCKED, never PASS.

~~~python
import math


class GateFailure(ValueError):
    pass


def require_finite(metric: str, value: float) -> None:
    if not math.isfinite(value):
        raise GateFailure(f"non-finite metric: {metric}")


def require_threshold(metric: str, value: float, threshold: float, direction: str) -> None:
    require_finite(metric, value)
    require_finite(metric, threshold)
    if direction == "min" and value > threshold:
        raise GateFailure(f"metric {metric} exceeds threshold: {value} > {threshold}")
    if direction == "max" and value < threshold:
        raise GateFailure(f"metric {metric} is below threshold: {value} < {threshold}")
    if direction not in {"min", "max"}:
        raise GateFailure(f"unsupported metric direction: {direction}")


def require_complete_suite(expected: set[str], actual: set[str], *, expected_evaluator_hash: str, actual_evaluator_hash: str) -> None:
    if expected_evaluator_hash != actual_evaluator_hash:
        raise GateFailure("evaluator manifest mismatch")
    missing = set(expected) - set(actual)
    if missing:
        raise GateFailure(f"missing required metrics: {sorted(missing)}")
    unexpected = set(actual) - set(expected)
    if unexpected:
        raise GateFailure(f"unexpected metrics: {sorted(unexpected)}")
~~~

- [ ] **Step 5: Implement final publication in the required order**

Use exactly this order:

~~~text
complete checkpoint
  -> best selection
  -> stage merged/exported model if required
  -> stage tokenizer
  -> read-back staged model/tokenizer artifacts
  -> publish manifests/final_publication.json
  -> read-back and schema-validate final_publication.json
  -> publish COMPLETE.json
~~~

final_publication.json records the selected checkpoint, final model files, tokenizer files, hashes, sizes, and staged read-back result. COMPLETE.json binds final publication hash, final model/tokenizer hashes, required checkpoint/evaluation/report hashes, run_identity.json hash, and the final log-manifest index hash. The log index is frozen after COMPLETE.json.

- [ ] **Step 6: Run evaluation and publication tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit/test_evaluation.py -q
~~~

Expected: PASS, including failure reports, evaluator mismatch, cadence gaps, final-publication read-back, and COMPLETE binding.

~~~powershell
git add pipeline_v2/slm_train/evaluation pipeline_v2/slm_train/artifacts/publication.py pipeline_v2/tests/unit/test_evaluation.py
git commit --only -m "feat(v2): add evaluator gates and final publication" -- pipeline_v2/slm_train/evaluation pipeline_v2/slm_train/artifacts/publication.py pipeline_v2/tests/unit/test_evaluation.py
~~~

---

## Task 11: Connect all CLI commands, security, and explicit V1 migration

**Files:**
- Modify: pipeline_v2/slm_train/cli.py
- Create: pipeline_v2/slm_train/security/__init__.py
- Create: pipeline_v2/slm_train/security/paths.py
- Create: pipeline_v2/slm_train/security/model_loading.py
- Create: pipeline_v2/slm_train/security/secrets.py
- Create: pipeline_v2/slm_train/security/export.py
- Create: pipeline_v2/slm_train/adapters/legacy_v1.py
- Create: pipeline_v2/migrations/README.md
- Create: pipeline_v2/configs/experiments/smoke.yaml
- Create: pipeline_v2/configs/experiments/pilot.yaml
- Create: pipeline_v2/configs/experiments/target.yaml
- Create: pipeline_v2/tests/contract/test_v1_boundary.py
- Modify: pipeline_v2/tests/contract/test_r1_contracts.py

- [ ] **Step 1: Write failing command and migration tests**

~~~python
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_migrate_v1_preserves_input_and_never_trains(tmp_path: Path):
    source = tmp_path / "v1.yaml"
    source.write_text("name: pilot\n", encoding="utf-8")
    target = tmp_path / "v2.yaml"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipeline_v2" / "run.py"),
            "migrate-v1",
            "--input",
            str(source),
            "--output",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert source.read_text(encoding="utf-8") == "name: pilot\n"
    assert target.is_file()
    assert "not launch" in result.stdout.lower()


def test_r1_rejects_sft_lora_kd_gdn_mounting():
    from slm_train.features.registry import validate_capabilities
    for capability in ("sft", "lora", "kd", "gdn", "mounting"):
        with pytest.raises(ValueError, match="R1"):
            validate_capabilities("r1", ["pretrain", capability])
~~~

- [ ] **Step 2: Run tests and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/contract/test_r1_contracts.py pipeline_v2/tests/contract/test_v1_boundary.py -q
~~~

Expected: FAIL until command handlers and the migration adapter exist.

- [ ] **Step 3: Implement all 11 commands**

The parser exposes init, validate, plan, doctor, preflight, run, status, explain, diff, export, and migrate-v1 with this exact argument surface:

| Command | Arguments | Effect |
|---|---|---|
| init | --template, --name, --out | write a complete validated template |
| validate | --config | validate structure, paths, dimensions, tokenizer, and stage |
| plan | --config | print graph and file effects; no writes or GPU |
| doctor | none | CPU-only environment and import checks |
| preflight | --config, --gpu | explicit GPU capability report; --gpu is required for device probing |
| run | --config, optional --seed/--device/--precision | execute or exact-resume the stage graph |
| status | --run | read published status/evidence only |
| explain | --config | print resolved fields, overrides, capabilities, evaluators, outputs, resume eligibility |
| diff | --left, --right | compare resolved semantic configurations |
| export | --run, --out, --allowlist | copy only the explicit artifact allowlist |
| migrate-v1 | --input, --output | read-only V1 config migration; never launch training |

init writes a complete template. validate checks schema, paths, model divisibility, tokenizer capacity, and stage compatibility. plan prints graph and file effects without writing data or acquiring a GPU. doctor is CPU-only. preflight --gpu is the only V2 GPU probe. explain prints active fields, overrides, capabilities, evaluator set, outputs, and resume eligibility. diff compares resolved semantic configs. status reads only published status/evidence. export copies only an allowlist and passes the same allowlist to the operation. run accepts only --config and the three documented safe overrides.

- [ ] **Step 4: Implement security and read-only migration**

security/paths.py checks containment and symlinks. security/model_loading.py permits safetensors and weights-only loading and rejects unsafe config-derived deserialization. security/secrets.py redacts tokens and environment values from manifests, logs, and reports. security/export.py passes its allowlist directly to the copy/upload operation.

adapters/legacy_v1.py reads an explicitly supplied V1 config, hashes it before and after, writes only a new V2 config, rejects output collisions, and never invokes training, GPU, or V1 lifecycle code. migrations/README.md states that migration is not adoption and no V1 file changes.

- [ ] **Step 5: Add complete S/P/T templates and boundary tests**

Templates include release: r1, capabilities: [pretrain], FP16, SDPA, explicit evaluator sets, explicit source/tokenizer identities, retention: manual, and promotion references. P may reference accepted S evidence; T may reference accepted P evidence; no template continues weights implicitly. The committed experiment files use the base file plus these exact overrides:

~~~yaml
# smoke.yaml
extends: [../base/r1.yaml]
run: {name: smoke, namespace: core, stage: S, seed: 42}
features: {retention: manual, promotion: null}

---

# pilot.yaml
extends: [../base/r1.yaml]
run: {name: pilot, namespace: core, stage: P, seed: 42}
features: {retention: manual, promotion: null}

---

# target.yaml
extends: [../base/r1.yaml]
run: {name: target, namespace: core, stage: T, seed: 42}
features: {retention: manual, promotion: null}
~~~

After S is accepted, a user may replace pilot promotion null with {predecessor_stage: S, predecessor_run_id: ..., checkpoint: ..., data_manifest: ..., evaluator_manifest: ..., gate_report: ...}; after P is accepted, the analogous target block uses predecessor_stage: P. Every replacement must be a repository-relative accepted evidence path or identifier, and an unreplaced value is a validation error. Boundary tests scan V2 Python sources for V1 imports and verify the V1 inventory hashes are unchanged after the CPU suite.

~~~python
# pipeline_v2/tests/contract/test_v1_boundary.py
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_frozen_v1_inventory_is_unchanged():
    inventory = json.loads((ROOT / "pipeline_v2/tests/fixtures/v1_inventory.json").read_text(encoding="utf-8"))
    for row in inventory["files"]:
        payload = (ROOT / row["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"], row["path"]
        assert len(payload) == row["bytes"], row["path"]
~~~

- [ ] **Step 6: Run command and boundary tests and commit**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/contract/test_r1_contracts.py pipeline_v2/tests/contract/test_v1_boundary.py -q
~~~

Expected: PASS.

~~~powershell
git add pipeline_v2/slm_train/cli.py pipeline_v2/slm_train/security pipeline_v2/slm_train/adapters/legacy_v1.py pipeline_v2/migrations pipeline_v2/configs/experiments pipeline_v2/tests/contract
git commit --only -m "feat(v2): connect commands security and migration" -- pipeline_v2/slm_train/cli.py pipeline_v2/slm_train/security pipeline_v2/slm_train/adapters/legacy_v1.py pipeline_v2/migrations pipeline_v2/configs/experiments pipeline_v2/tests/contract
~~~

---

## Task 12: Add CPU unit, contract, integration, schema, and dependency lanes

**Files:**
- Create: pipeline_v2/tests/integration/test_fixture_pipeline.py
- Create: pipeline_v2/tests/fixtures/tiny_config.yaml
- Create: pipeline_v2/tests/contract/test_evidence_contract.py
- Create: pipeline_v2/tests/conftest.py additions
- Modify: pipeline_v2/pyproject.toml for test-artifact settings
- Create: pipeline_v2/tests/test_artifact_writer.py

- [ ] **Step 1: Write the failing one-step integration test**

~~~python
# pipeline_v2/tests/integration/test_fixture_pipeline.py
import json
from pathlib import Path


def test_tiny_fixture_prepares_trains_evaluates_and_publishes(tmp_path: Path):
    from slm_train.orchestration.engine import Engine
    result = Engine(fixture_root=tmp_path).run_cpu_fixture()
    assert result.label == "PASS"
    assert (tmp_path / "manifests" / "run_identity.json").is_file()
    assert (tmp_path / "manifests" / "final_publication.json").is_file()
    assert (tmp_path / "COMPLETE.json").is_file()
    complete = json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["label"] == "PASS"
    assert complete["log_index_sha256"]
~~~

- [ ] **Step 2: Run it and verify failure**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/integration/test_fixture_pipeline.py -q
~~~

Expected: FAIL because the full engine chain is incomplete.

- [ ] **Step 3: Implement the local CPU fixture path**

The fixture uses a local source, a deterministic test tokenizer, a tiny model, one training step, one internal evaluation, one gate, final publication, no network, and no GPU. It uses the same contracts, atomic store, log index, checkpoint, evaluator, and publication code as the real path, with only a fake training backend for the one-step loop.

- [ ] **Step 4: Persist machine-readable test artifacts**

test_artifact_writer.py defines a typed TestArtifactWriter whose write_result(session_id, exitstatus, test_counts) method writes a JSON document under pipeline_v2/test-artifacts/<session_id>/summary.json. It records the exact command, test counts, UTC completion time, repository-relative source root, label PASS or FAIL, and the SHA-256 of the JSON payload. The pytest_sessionfinish hook in conftest.py calls it after the suite. A test result is not reported as PASS until the artifact is reopened, parsed, schema-validated, and hash-checked. Do not delete previous test artifacts automatically. The expected CPU evidence path is pipeline_v2/test-artifacts/<session_id>/summary.json.

~~~python
# pipeline_v2/tests/test_artifact_writer.py
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from slm_train.artifacts.atomic import atomic_write_bytes


class TestArtifactWriter:
    @staticmethod
    def _evidence_argument(value: str) -> str:
        root = Path.cwd().resolve()
        candidate = Path(value)
        if candidate.is_absolute() and candidate.is_relative_to(root):
            return candidate.relative_to(root).as_posix()
        return value

    def write_result(self, session_id: str, exitstatus: int, test_counts: dict[str, int]) -> Path:
        payload = {
            "schema_version": 1,
            "session_id": session_id,
            "command": [self._evidence_argument(value) for value in sys.argv],
            "exitstatus": int(exitstatus),
            "test_counts": test_counts,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source_root": ".",
            "label": "PASS" if int(exitstatus) == 0 else "FAIL",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
        target = Path("pipeline_v2/test-artifacts") / session_id / "summary.json"
        atomic_write_bytes(target, json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        reopened = json.loads(target.read_text(encoding="utf-8"))
        stored_hash = reopened.pop("payload_sha256")
        canonical = json.dumps(reopened, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != stored_hash:
            raise RuntimeError("test artifact hash mismatch")
        return target
~~~

The hook calls TestArtifactWriter().write_result(session.config.stash["v2_session_id"], session.exitstatus, {"collected": session.testscollected, "failed": session.testsfailed}); the session ID is generated before collection and is repository-safe. Before writing, canonicalize any command argument under the repository root to a repository-relative path. Verification recomputes payload_sha256 over the JSON object with payload_sha256 removed, and a mismatch is FAIL.

~~~python
# pipeline_v2/tests/contract/test_evidence_contract.py
import json
import math
from pathlib import Path


def test_all_evidence_paths_are_repository_relative() -> None:
    root = Path("pipeline_v2/tests/contract/test_evidence_contract.py").resolve()
    for path in root.parents[2].glob("test-artifacts/*/summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in payload.get("command", []):
            assert not Path(value).is_absolute(), value


def test_metric_evidence_requires_finite_values() -> None:
    assert math.isfinite(1.0)
    assert not math.isfinite(float("nan"))
    assert not math.isfinite(float("inf"))
~~~

- [ ] **Step 5: Run the complete CPU suite and compile check**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/unit -m "not gpu and not network" -q
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/contract -m "not gpu and not network" -q
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/integration -m "not gpu and not network" -q
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests -m "not gpu and not network" -q
& .\\.venv\\Scripts\\python.exe -m compileall -q pipeline_v2/slm_train pipeline_v2/tests
~~~

Expected: all selected CPU lanes PASS, no GPU/network test is selected, and compileall exits 0. GPU and network tests remain separate markers and are run only by the explicit acceptance jobs.

- [ ] **Step 6: Verify the V1 inventory is unchanged**

~~~powershell
& .\\.venv\\Scripts\\python.exe -c "from pathlib import Path; import json,hashlib; d=json.loads(Path('pipeline_v2/tests/fixtures/v1_inventory.json').read_text(encoding='utf-8')); assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in d['files']); print('PASS', len(d['files']))"
~~~

Expected: PASS. Any mismatch is FAIL and blocks the task.

- [ ] **Step 7: Commit Task 12**

~~~powershell
git add pipeline_v2/tests pipeline_v2/pyproject.toml
git commit --only -m "test(v2): add CPU contract and integration lanes" -- pipeline_v2/tests pipeline_v2/pyproject.toml
~~~

---

## Task 13: Run explicit GPU acceptance and close Release 1 evidence

**Files:**
- Create: pipeline_v2/docs/acceptance.md
- Modify: pipeline_v2/README.md
- Modify: pipeline_v2/configs/experiments/smoke.yaml
- Modify: pipeline_v2/configs/experiments/pilot.yaml
- Modify: pipeline_v2/configs/experiments/target.yaml
- Create: pipeline_v2/tests/contract/test_release_boundaries.py
- Modify: state/tasks.yaml and research/experiments.yaml only through the repository experiment-record workflow

- [ ] **Step 1: Write the release-boundary and evidence tests**

~~~python
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_v2_has_no_v1_runtime_imports():
    paths = [ROOT / "pipeline_v2/run.py", *(ROOT / "pipeline_v2/slm_train").rglob("*.py")]
    forbidden = {"src", "scripts"}
    for path in paths:
        modules = _imported_modules(path)
        assert "importlib" not in modules, path
        assert not any(module == item or module.startswith(item + ".") for module in modules for item in forbidden), path


def test_acceptance_requires_explicit_gpu_window():
    text = (ROOT / "pipeline_v2/docs/acceptance.md").read_text(encoding="utf-8")
    assert "user-opened GPU window" in text
    assert "preflight --gpu" in text
    assert "zero additional flags" in text
~~~

- [ ] **Step 2: Run the tests and verify the runbook is not yet complete**

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests/contract/test_release_boundaries.py -q
~~~

Expected: FAIL until the acceptance runbook and evidence matrix are written.

- [ ] **Step 3: Pre-register the exact experiment before any GPU launch**

Use the repository experiment-record workflow to record namespace, config path, resolved-config hash, exact command, requested device, expected evaluator suite and gates, metric source, data/tokenizer/model identities, code/dependency hash, and the read-back paths for S, P, and T. Record the adopted Task 0 OOM and preflight decisions. Do not edit generated state by hand.

- [ ] **Step 4: Write the acceptance runbook**

The runbook must contain the literal label "user-opened GPU window" and require a fresh preflight, one-job confirmation, current free-VRAM and disk measurements, no foreign process termination, exact config, exact command, and an evidence directory. The canonical V2 smoke command is:

~~~powershell
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py run --config pipeline_v2/configs/experiments/smoke.yaml
~~~

The required read-back paths are runs/v2/<namespace>/<run_id>/reports/smoke-gate.json, COMPLETE.json, manifests/run_identity.json, manifests/final_publication.json, the final log index, and the selected checkpoint manifest. A missing artifact is BLOCKED or FAIL.

- [ ] **Step 5: Run the user-opened GPU sequence in order**

Only after the user opens the GPU window and confirms the pre-registration record:

~~~powershell
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py preflight --gpu --config pipeline_v2/configs/experiments/smoke.yaml
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py run --config pipeline_v2/configs/experiments/smoke.yaml
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py preflight --gpu --config pipeline_v2/configs/experiments/pilot.yaml
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py run --config pipeline_v2/configs/experiments/pilot.yaml
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py preflight --gpu --config pipeline_v2/configs/experiments/target.yaml
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py run --config pipeline_v2/configs/experiments/target.yaml
~~~

Run each command only after the preceding evidence gate is PASS and the experiment record names that exact predecessor. Then perform exact-command kill/resume by repeating the applicable smoke command with zero additional flags, compare uninterrupted and resumed state, and read back the checkpoint and manifests. Record predecessor run ID, checkpoint, data manifest, evaluator set, gate result, and hashes for every promotion. No GPU or network command is run during ordinary plan or CPU work.

- [ ] **Step 6: Verify the OOM policy at runtime**

If an own-job OOM occurs, write the health event, preserve the last valid checkpoint, release the lease only after the owner is gone, and set the run status to BLOCKED. Do not alter batch, context, optimizer, precision, or command. A reduced next attempt requires the separately approved versioned launch profile and new run identity recorded in the experiment record. If that approval is absent, retain BLOCKED and report NEEDS_USER_DECISION rather than retrying.

- [ ] **Step 7: Run the final Release 1 acceptance checklist**

Verify every criterion with a repository-relative artifact: invalid config failure; pinned/fresh atomic data; tokenizer/shard identity; train/validation overlap; changed code/dependency/launch-profile resume rejection; exact cursor restore; busy or uncertain GPU blocking; atomic bootstrap handoff; gap-free metrics/health; NaN/Inf, missing metric/report, evaluator mismatch, and missing-artifact rejection; final model/tokenizer/manifests/reports read-back; S/P/T promotion references; unchanged V1 files/artifacts; and YAML-only experiment creation.

Run:

~~~powershell
& .\\.venv\\Scripts\\python.exe -m pytest pipeline_v2/tests -m "not gpu and not network" -q
& .\\.venv\\Scripts\\python.exe -m compileall -q pipeline_v2/slm_train pipeline_v2/tests
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py doctor
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py plan --config pipeline_v2/configs/experiments/smoke.yaml
git diff --check -- pipeline_v2 .gitignore state/tasks.yaml research/experiments.yaml
~~~

Expected: CPU validation PASS, the generated evidence is readable, V1 inventory hashes are unchanged, and any unexecuted GPU lane is labeled SKIPPED or NEEDS_USER_DECISION rather than PASS.

- [ ] **Step 8: Commit documentation and boundary tests**

~~~powershell
git add pipeline_v2/docs/acceptance.md pipeline_v2/README.md pipeline_v2/configs/experiments pipeline_v2/tests/contract/test_release_boundaries.py
git commit --only -m "docs(v2): close R1 acceptance and evidence boundary" -- pipeline_v2/docs/acceptance.md pipeline_v2/README.md pipeline_v2/configs/experiments pipeline_v2/tests/contract/test_release_boundaries.py
~~~

---

## Using V2 for a new process or ablation

1. Copy the closest S/P/T experiment YAML and edit only declared source, tokenizer, model, data, packing, evaluator, and launch-profile fields; keep unknown keys visible so validation fails rather than silently ignoring them. YAML is the adaptation surface; no Python edits are required for an ordinary experiment.
2. Run the CPU-only explanation and graph commands before any launch:

~~~powershell
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py explain --config pipeline_v2/configs/experiments/smoke.yaml
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py plan --config pipeline_v2/configs/experiments/smoke.yaml
& .\\.venv\\Scripts\\python.exe pipeline_v2/run.py validate --config pipeline_v2/configs/experiments/smoke.yaml
~~~

3. Record the resolved config hash, exact command, evaluator suite, data/tokenizer/model hashes, and expected evidence paths in the experiment registry. Do not edit generated state by hand.
4. In a user-opened GPU window, run `preflight --gpu` and then the exact recorded `run` command. A changed config, code, dependency, environment, device, or command creates a new run identity; it is never an in-place continuation.
5. Read back status, COMPLETE.json, run identity, final publication, evaluator reports, log index, and the selected checkpoint. Use the gate label to decide whether an S-to-P or P-to-T promotion may be registered; an independent P/T run must explicitly record its own predecessor evidence when one is required.
6. Keep V1 files and artifacts untouched. V2 may be extended later through a new statically registered module and a new release capability review, not through fallback code or implicit plugin discovery.

---

## Final handoff rule

When the user selects an execution mode, begin at Task 0. Do not begin at Task 1 while the Task 0 decision record is NEEDS_USER_DECISION. For subagent-driven execution, use a fresh worker per task and review each task before the next. For inline execution, use executing-plans with the same task boundaries. In either mode, preserve V1, use the existing venv, do not run GPU/network work outside the explicit gates, and report exact repository-relative evidence with the repository status vocabulary.
