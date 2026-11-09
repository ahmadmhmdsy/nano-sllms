# Core Training Pipeline V2 Design Specification

**Date:** 2026-09-25
**Status:** Approved design — implementation plan pending policy decisions and user execution choice
**Scope:** Core causal-LM pretraining pipeline, V2 Release 1
**Compatibility rule:** V1 remains untouched; no automatic fallback or cutover

## 1. Executive decision

Build a new, self-contained V2 under `pipeline_v2/` rather than modifying the current V1 implementation. V2 is a contract-first modular monolith for the single consumer-GPU project:

- strict typed configuration;
- immutable, hashed data and run-identity manifests;
- a fail-closed stage/orchestration engine;
- a centralized training lifecycle;
- explicit trusted extension registries;
- durable monitoring and evaluation reports;
- exact identity-checked resume;
- CPU-first validation and explicit GPU gates.

V2 Release 1 proves the core pretraining path S (smoke) → P (pilot) → T (target). Only after Release 1 is accepted does Release 2 add exactly SFT, LoRA, KD, GDN, and mounting. Any additional extension requires a separately approved release. A distributed/plugin platform is reserved for V3 and is not part of this design.

The design is intentionally fail-closed: an unknown config key, identity mismatch, stale or mixed data, incomplete artifact, unverifiable GPU, NaN/Inf loss, OOM, missing metric/report, evaluator-contract mismatch, or failed required gate stops the stage. It never silently falls back to V1, changes a training hyperparameter, accepts a nearby checkpoint, continues from partial state, or treats console output as evidence. Every failure preserves the valid artifacts and diagnostic evidence that existed before the failure.

## 2. Audit basis and current limitations

This design follows a read-only audit of the current core and related component pipelines. The most relevant verified core findings are:

1. The documented flow is sanity → prepare → tokenize → train → eval, with a separate SFT path (`README.md:46-54`; `scripts/run_custom.py:27-33`).
2. Root resume selection checks a numeric checkpoint and `trainer_state.json` but does not prove config, tokenizer, shard, code, or data identity (`scripts/train.py:25-38`, `scripts/train.py:95-101`).
3. Raw JSONL, packed shards, and metadata are written directly to canonical locations; shard numbering and broad globs can mix generations (`scripts/prepare_data.py:237-245`, `scripts/prepare_data.py:319-336`, `scripts/tokenize_data.py:44-56`, `src/data.py:19-39`).
4. GPU coordination is not centralized; a probe error in the launcher can be treated as no busy process, and direct entry points do not all enforce the one-job rule (`scripts/run_custom.py:56-75`; active policy: `rules/gpu.md:18-35`).
5. Configuration is raw YAML and script-specific, with no central strict schema or semantic capability validator.
6. Sanity and probe behavior can describe a different path from the named trainer, and some success signals are console-only (`scripts/sanity_check.py:29-48`, `scripts/vram_probe.py:1-6`).
7. Pretrain, SFT, and KD duplicate checkpoint, argument, and finalization logic, allowing lifecycle drift.
8. Existing reports can contain machine-specific absolute paths; evidence must be relative and self-describing (`rules/evidence.md:17-44`).

Related component audits show why the shared layer should be a lifecycle envelope, not a forced common model or tokenizer: UI, Laya, MEX, Langid, and diacritizer have materially different data, head, evaluator, and evidence contracts. V2 Release 1 therefore targets the core causal-LM path and exposes adapter interfaces for later component adoption.

Static inspection is not a runtime or metric claim. No GPU, network, training, evaluation, or current-job state was inferred during this design phase.

## 3. Goals and non-goals

### 3.1 Goals

- Make a new core experiment possible by editing validated YAML, not Python.
- Make config, data, tokenizer, model, code, and checkpoint identity explicit.
- Make every derived artifact atomic, complete, hashed, and readable.
- Make exact automatic resume safe and understandable.
- Make GPU ownership and resource uncertainty fail closed.
- Make required evaluation and gate evidence durable and machine-readable.
- Centralize lifecycle behavior while allowing model/data/evaluator specialization.
- Keep the current V1 code, configs, runs, and checkpoints intact.
- Provide a future path to Release 2 and V3 without building V3 now.

### 3.2 Non-goals for Release 1

- No dynamic remote plugin installation.
- No multi-node scheduler, remote workers, or distributed artifact store.
- No SFT, LoRA, KD, GDN, or mounting acceptance in Release 1.
- No component-line migration.
- No modification or deletion of V1 artifacts.
- No automatic replacement of the current `scripts/train.py` or other V1 entry points.
- No silent conversion of a changed experiment into a resumed run.

## 4. Release boundaries

### 4.1 V2 Release 1

Release 1 covers:

- S smoke, P pilot, and T target pretraining;
- local and declared Hugging Face data sources;
- strict data preparation, splitting, deduplication, tokenization, and packing;
- causal-LM model construction and training through an HF backend;
- validation loss, perplexity, generation smoke, applicable syntax/code checks, and registered gates;
- monitoring, health checks, preflight, exact resume, and final publication.

Release 1 targets the current consumer NVIDIA GPU contract: FP16 is supported, BF16 is rejected unless capability is explicitly verified, and FlashAttention is not assumed; the backend must use the declared safe attention path. The Release 1 capability registry contains only the core pretraining model, data, trainer, and evaluator contracts. A Release 1 config naming any Release 2 capability is rejected before any stage starts.

S, P, and T are independent training runs by default; the arrow denotes a promotion/gate relationship, not implicit weight continuation. P may reference an accepted S report or checkpoint as a baseline, and T may reference an accepted P report or checkpoint as a baseline, but each reference must record the predecessor run ID, checkpoint, data manifest, evaluator set, and gate result. Unaccepted or mismatched predecessor artifacts are rejected. Any weight-continuation mode is a separate capability and is not part of Release 1.

### 4.2 V2 Release 2

After Release 1 acceptance, add exactly these five existing advanced paths as registered feature modules:

- SFT and prompt-masked instruction data;
- LoRA adapters;
- KD;
- GDN;
- mounting.

Each feature receives its own schema, identity rules, tests, smoke run, and acceptance gate. Additional optimizers, losses, augmentation strategies, and other extensions are deferred to a separately approved release.

### 4.3 V3

V3 may later add distributed workers, dynamic workflows, remote storage, and a broader plugin runtime. V2 interfaces must not depend on V3 concepts.

## 5. Package and dependency direction

V2 is rooted at:

~~~text
pipeline_v2/
  README.md
  run.py
  pyproject.toml
  configs/
    base/
    experiments/
    schemas/
  slm_train/
    cli.py
    config/
    contracts/
    orchestration/
    resources/
    data/
    models/
    training/
    features/
    evaluation/
    monitoring/
    artifacts/
    adapters/
  tests/
    unit/
    contract/
    integration/
    fixtures/
  migrations/
~~~

Runtime data and reports are outside the source package:

~~~text
data/v2/
  prepared/<dataset_id>/<generation_id>/
  tokens/<dataset_id>/<generation_id>/

runs/v2/
  _attempts/<bootstrap_id>/<attempt_id>/
  <namespace>/<run_id>/
~~~

V2 code must not import or mutate V1 implementation modules as its normal lifecycle path. A legacy adapter may read a V1 checkpoint or config only as an explicit migration input. Release 1 and Release 2 modules are statically linked, in-process, source-controlled components of one deployable. There is no runtime discovery, entry-point loading, subprocess plugin, or remote plugin; registries compose approved in-process modules only. A broader plugin runtime belongs to V3.

All interpreter commands continue to use the repository venv. V2 declares and locks its direct dependencies; the implementation plan will record the exact uv installation/rebuild command before any package is added.

## 6. Configuration architecture

### 6.1 Canonical shape

A V2 config is a typed document with this top-level shape:

~~~yaml
schema_version: 2
release: r1
capabilities:
  - pretrain
run:
  name: target
  namespace: core
  seed: 42
extends: []
data: {}
tokenizer: {}
model: {}
train: {}
evaluation: {}
runtime: {}
features: {}
~~~

The final schema will define required and optional fields. The `release` field is mandatory and is part of the run fingerprint. Release 1 permits only the core pretraining capability; SFT, LoRA, KD, GDN, and mounting are rejected during validation in Release 1. The following rules are mandatory:

- duplicate YAML keys are errors;
- unknown keys are errors;
- inheritance cycles are errors;
- ambiguous list merges are errors;
- omitted values never acquire hidden trainer-specific defaults;
- CLI overrides are limited to documented safe fields;
- every override is recorded;
- absolute paths and paths escaping declared roots are errors;
- all persisted paths are repository-relative;
- V1 YAML requires explicit migration.

Pydantic v2 provides strict parsing, typed validation, and JSON Schema generation. Committed schemas live under `pipeline_v2/configs/schemas/`.

### 6.2 Resolution and hashing

The loader produces:

~~~text
resolved_config.yaml
resolved_config.json
config_hash
~~~

The hash covers the fully resolved semantic configuration, including release, capability declarations, source freshness/supersession policy, expected evaluator set, metric schemas/cadence, thresholds, and S/P/T promotion references. A changed model dimension, data source, tokenizer, seed, precision, feature set, release, or evaluation policy changes the hash. Comments and formatting do not.

### 6.3 Run identity

~~~text
experiment_id = stable logical experiment name
input_fingerprint = hash(
  resolved_config,
  data_manifest_semantic_hash,
  tokenizer_manifest_semantic_hash,
  tokenized_data_manifest_semantic_hash,
  model_manifest_semantic_hash,
  evaluator_manifest_semantic_hash,
  V2_code_and_dependency_hash,
  release,
  launch_profile,
  runtime_identity,
  invocation_contract
)
run_id           = experiment_id + input_fingerprint
~~~

A changed experiment cannot reuse an existing run directory. Re-running the same resolved config and all bound input, runtime, and invocation identities uses the same run ID and is the only case eligible for exact resume. The runtime identity includes the venv/Python identity, package lock, OS/CUDA/driver, GPU capability class and device identity, and the invocation contract includes the canonical exact command, working directory, and allowlisted non-secret environment values. A different machine, device, driver, environment, or command creates a new run identity. Persisted path fields remain repository-relative; machine-specific external paths are represented by a hash/token rather than emitted as absolute evidence paths. Any compatibility exception must be an explicit versioned contract whose hash is part of the identity; default matching is exact.

## 7. Contract and manifest layer

All contracts carry a schema version and required fields. Unless a field is explicitly labeled as mutable evidence, manifest hashes cover immutable semantic identity and exclude verification timestamps, transient status, and lease state. The initial contracts are:

### Run identity manifest (immutable)

- run and experiment IDs;
- resolved config path/hash;
- semantic data, tokenizer, tokenized-data, model, and evaluator manifest hashes;
- Git commit and worktree state as metadata;
- deterministic V2 code/dependency content hash covering source, schemas, contracts, and lockfiles;
- Python/package/device/driver identities and runtime fingerprint;
- exact working directory and launch profile;
- exact command;
- creation timestamp;
- parent/base model and promotion-reference identity, including predecessor run/checkpoint/data/evaluator/gate identities;
- immutable evidence-root path.

The run identity manifest never changes after publication. Mutable stage state, timestamps, current GPU lease, heartbeat, and recovery status live in atomically replaced `status.json` snapshots. Lease acquisition/release events are preserved as numbered `logs/lease/segment-*.jsonl` segments. Exact resume reads the immutable run identity and checkpoint manifests, not the current status snapshot.

### Data manifest

- source IDs and roles;
- immutable source identity and source fingerprint;
- configured freshness/supersession policy;
- semantic generation/content hash, excluding current verification time and status;
- dataset config and split;
- filters, normalization, dedupe, and split salt;
- document IDs and counts;
- license/provenance notes;
- overlap/leakage report;
- truncation/rejection counters;
- regeneration command;
- content hash of the immutable semantic fields.

Verification time, observed file hashes/sizes, freshness status, and policy evaluation are mutable evidence written to `status.json` and a new immutable freshness report for each revalidation. The status snapshot points to the latest report. They are not fields in the immutable data manifest and do not change the semantic manifest hash used by the run fingerprint. If the policy has expired, the consuming stage is BLOCKED. If source content or pinned identity changed, a new generation and new run identity are required; refresh or regeneration is never implicit.

### Tokenizer manifest

- immutable repository/file identity;
- tokenizer content hash;
- vocabulary and maximum token ID;
- BOS/EOS/PAD IDs;
- special tokens;
- context and offset behavior;
- package version.

Tokenizer freshness verification time and status are mutable evidence in the same way as data freshness; the tokenizer semantic hash used for run identity excludes them.

### Tokenized-data manifest

- exact ordered shard list;
- size and SHA-256 hash for each shard;
- sequence length and packing mode;
- usable block count;
- dropped-tail and overflow counters;
- token-range validation;
- document-boundary information;
- source data manifest hash.

### Model and evaluator manifests

- model architecture/config hash and initialization/base-model identity;
- exact parent/base checkpoint identity and content hash when applicable;
- expected evaluator IDs, versions, suite/gate set, metric schemas, cadence, and thresholds;
- immutable evaluator-contract manifest hash.

These manifests are immutable semantic inputs to the run fingerprint. The evaluator report binds its actual evaluator set back to `evaluator.json`.

### Checkpoint manifest

- run/config/data/tokenizer/tokenized-data/model/evaluator/code/dependency hashes;
- global step and epoch;
- model, optimizer, scheduler, scaler, RNG, and sampler state;
- required files and sizes;
- complete/partial state;
- evaluation and selection state;
- SHA-256 hashes for every exact-resume state file;
- a completion marker that binds the checkpoint directory to its exact manifest hash.

### Evaluation report

- evaluator name/version and expected evaluator-suite/gate set;
- expected-versus-actual evaluator manifest hash;
- run/checkpoint/data identity;
- split identity;
- tokenizer/device/seed identity;
- metric schema identity, definitions, and values;
- baseline and gate threshold;
- per-class/per-example details where applicable;
- status label;
- relative artifact paths.

A report is complete only after it is written, reopened, parsed, and schema-validated. The resolved config binds the expected evaluator IDs, versions, suite/gate set, metric IDs, metric schemas, cadence, and thresholds; the report records the actual set and a hash of the expected manifest. Before PASS, every required metric must be present at its required cadence, all JSONL and TensorBoard streams must be flushed, reopened, parsed, and gap-free, and required values must be finite. Any absent, stale, unparseable, mismatched, or incomplete metric/report is FAIL or BLOCKED. All status fields use the repository vocabulary exactly: PASS, FAIL, SKIPPED, BLOCKED, and NEEDS_USER_DECISION.

## 8. Filesystem and atomic publication

A run has this structure:

~~~text
runs/v2/<namespace>/<run_id>/
  config/
    resolved.yaml
    resolved.json
  manifests/
    run_identity.json
    data.json
    tokenizer.json
    tokenized_data.json
    model.json
    evaluator.json
    final_publication.json
  logs/
    lease/
      segment-000001.jsonl
    metrics/
      segment-000001.jsonl
    monitor/
      segment-000001.jsonl
    tensorboard/
      events.tfevents.000001
    log-manifest.json
  checkpoints/
    checkpoint-000100/
      manifest.json
      COMPLETE.json
      model/ optimizer/ scheduler/ scaler/ rng/ sampler/
  evaluations/
    <suite>/<evaluation_id>.json
  reports/
    <report_id>.json
  status.json
  run.lock
  COMPLETE.json
~~~

Streaming logs are published as immutable numbered segments. Each metrics, monitor, lease-event, and TensorBoard segment is written to a temporary file on the same filesystem, framed with a monotonic segment/record sequence, flushed and fsynced, then atomically renamed. Partial temporary segments and trailing uncommitted records are ignored or quarantined, never silently treated as complete. Final validation requires the expected sequence with no gaps and reparses every published record/event. `status.json` is an atomically replaced mutable snapshot; `run.lock` and lease events are operational state, not part of the immutable run identity.

`log-manifest.json` is an explicit mutable atomic index, not an immutable identity manifest. It contains an index generation, the previous index hash, the exact published segment list, and the last record sequence; each replacement is a same-filesystem temporary write plus fsync and rename. A crash leaves the previous valid index intact. The run `COMPLETE.json` binds the final log-index hash, and no segment is deleted when the index advances.

All writers use a temporary sibling path, flush and fsync the file, perform the filesystem directory-durability step when the platform exposes it, record that result, and atomically publish the completed file or directory. Canonical names are never partially written.

A generation is published only after:

1. all expected files exist;
2. SHA-256 hashes, sizes, and counts pass;
3. the manifest validates;
4. a completion marker is written last.

Consumers use the exact manifest file list. Broad globs are forbidden for V2 training and evaluation. Every publication uses a temporary path on the same filesystem, flushes and fsyncs the file or directory, and atomically renames it into place. A published contract/data/checkpoint manifest is immutable; it is never overwritten in place. The explicitly mutable `status.json` and `log-manifest.json` are versioned operational indexes replaced atomically and are excluded from identity hashes. A completion marker is written last and contains the hash of the exact manifest it certifies.

## 9. Data lifecycle

### 9.1 Source declaration

Every source declares:

- stable source ID;
- local path or HF repository;
- immutable source identity: an HF repository commit plus file hashes, or a local file-set Merkle manifest;
- config and split;
- role: train, validation, test, or external-gate;
- license/provenance note;
- expected row/token budget.

A source cannot change role because a file happens to exist in a directory. A source without a pinned immutable identity is invalid. Before every consuming stage, V2 revalidates the completion marker, manifest/schema identity, every source file hash and size, the source fingerprint, and the configured freshness/supersession policy. A stale, changed, unpinned, or incomplete generation is BLOCKED or FAIL; reuse and refresh are never implicit.

### 9.2 Preparation

Preparation:

- preserves source data;
- creates a unique generation directory;
- assigns document and source IDs;
- applies explicit filters and normalization;
- deduplicates according to the declared policy;
- creates a deterministic salted split before packing;
- checks content/document overlap;
- counts all rejections and truncations;
- publishes a manifest and completion marker atomically.

A failed required source fails the stage. Optional sources require an explicit profile and recorded reason.

### 9.3 Tokenization and packing

Tokenization:

- pins tokenizer revision and identity;
- validates special-token IDs;
- validates every token ID against model vocabulary;
- validates context length;
- records all overflow and dropped-tail counts;
- writes shards to a temporary generation;
- publishes the exact ordered shard set atomically.

Packing mode is explicit:

- `concat`: legacy-compatible concatenation for parity experiments;
- `document_masked`: document boundaries and loss/attention semantics are retained where the backend supports them.

V2 never silently discards a suffix or mixes shards from different generations. A mismatch between aggregate token counts and usable blocks is a hard failure.

## 10. Runtime and resource safety

### 10.1 One engine

The only user-facing execution path is:

~~~powershell
.\.venv\Scripts\python.exe pipeline_v2\run.py <command>
~~~

The engine owns the stage graph:

~~~text
validate
  -> bootstrap identity and acquire bootstrap GPU lease (for GPU work)
  -> GPU/VRAM probe
  -> prepare
  -> tokenize
  -> dataset validation
  -> model validation
  -> resolve final input fingerprint and bind final run ID
  -> train/resume
  -> internal evaluation
  -> external evaluation/gates
  -> publish final
~~~

### 10.2 GPU lease

GPU stages require a cross-process lease containing bootstrap ID, attempt ID, PID, process start time, exact command, device, config hash, and timestamp. After the handoff below, the same lease also contains the final run ID.

Rules:

- one protected repository GPU job at a time;
- probe failure is BLOCKED, not idle;
- no foreign process is killed;
- stale leases are not silently removed;
- recovery requires proof that the owner and its GPU work are gone;
- V2 has no share-GPU override.

### 10.3 Bootstrap identity and lease handoff

Before any GPU probe or other GPU work, V2 computes a deterministic bootstrap identity from the logical experiment name, resolved config hash, pinned source/tokenizer identities and declarations, requested launch/runtime profile, V2 code/dependency hash, and canonical exact command. Each invocation receives a unique attempt ID derived from its launch nonce, PID, and process start time; the attempt ID is operational evidence and is not part of the final run identity. The bootstrap identity is not a resumable training run and does not claim a final data or device identity. Evidence for each attempt is stored under `runs/v2/_attempts/<bootstrap_id>/<attempt_id>/`.

The bootstrap lease is acquired before the GPU probe. After the probe and CPU-side prepare/tokenize/dataset/model/evaluator validation succeed, V2 has published the semantic data, tokenizer, tokenized-data, model, and evaluator manifests, computes the final input fingerprint and run ID, atomically binds the bootstrap lease to that run ID, and publishes the immutable `run_identity.json`. The handoff uses create-if-absent/compare-and-swap semantics: an existing destination with a different identity is BLOCKED and is never overwritten.

If the process fails before handoff, the attempt evidence is retained and no resumable final run is published. If it fails during handoff, recovery verifies the bootstrap owner, device, command, and existing identity before deciding whether to resume. If it fails after handoff, the final run identity and its checkpoint manifests are authoritative; a second GPU job cannot acquire the lease. A changed bootstrap or final identity always requires a new attempt/run, and no recovery path silently falls back to V1 or deletes the failed attempt.

### 10.4 Preflight report

Every GPU preflight writes device identity, driver/runtime versions, free/total VRAM, compute processes, disk headroom, temperature/utilization, estimate, decision, and reason.

CPU is the default for structural validation. `doctor` is CPU-only by default and does not acquire or probe a GPU. GPU is requested only through the explicit `preflight --gpu` command or another declared GPU stage.

## 11. Training lifecycle

### 11.1 Backend boundary

V2 defines a backend interface:

~~~text
TrainerBackend
  prepare()
  fit()
  evaluate()
  checkpoint()
  resume()
  finalize()
~~~

Release 1 implements `HFCausalLMTrainer`. The orchestrator consumes typed events and results, not HF implementation details.

The centralized training specification owns model, tokenizer, dataset manifest, optimizer, scheduler, precision, batching, accumulation, clipping, evaluation, checkpoint policy, feature set, and seed.

### 11.2 Exact resume

Resume is allowed only when all identities match:

- resolved config hash;
- deterministic V2 code/dependency hash;
- data generation;
- tokenizer identity;
- model contract;
- feature set;
- exact evaluator suite, evaluator implementation, selection policy, and expected metric/gate manifest;
- exact command, working directory, allowlisted environment, driver, package, runtime, and device identity.

Automatic resume is triggered only by re-running the exact original command with zero additional flags. Required checkpoint state includes model, optimizer, scheduler, scaler, global RNG, the exact training cursor (examples seen, microbatch position, and gradient-accumulation boundary), epoch, sampler/dataloader generator state, per-worker dataloader RNG state, early-stop state, evaluation cursor/state, and lifecycle state. Restore-versus-saved-state tests and uninterrupted-versus-kill/resume comparisons are required before Release 1 acceptance.

A complete checkpoint is selected through its immutable per-checkpoint manifest and numeric identity, not a broad directory scan. Every exact-resume state file has a recorded SHA-256 hash, and the checkpoint completion marker binds the directory to that manifest.

### 11.3 Final publication

The final sequence is:

~~~text
complete checkpoint
  -> best selection
  -> stage merged/exported model if required
  -> stage tokenizer
  -> read-back validation of staged artifacts
  -> manifests/final_publication.json
  -> read-back validation of final publication manifest
  -> COMPLETE.json
~~~

`manifests/final_publication.json` is created only after the staged artifacts pass read-back, then atomically published and reopened/validated. It is immutable and records the exact selected checkpoint, final selected/exported model files, final tokenizer files, artifact hashes and sizes, and the read-back result. Existing final content is never partially overwritten. A changed run gets a new destination.

### 11.4 Events and health

A run-scoped event bus emits:

~~~text
run_started
stage_started
step_completed
metric_emitted
evaluation_completed
checkpoint_completed
health_warning
health_failure
stage_completed
run_completed
~~~

Required outputs are JSONL metrics, JSONL monitor events, and TensorBoard events. The monitor checks finite loss/gradients, heartbeat, evaluation/checkpoint completion, VRAM/reserved memory, shared-memory spill indicators, temperature, disk headroom, and process liveness.

OOM is BLOCKED, not silently retried with altered batch, context, optimizer, or precision. NaN/Inf is FAIL, with the last valid checkpoint preserved.

## 12. Evaluation and gates

Evaluators are registered suites with independent versions and contracts. Release 1 includes:

- internal validation loss/perplexity;
- generation smoke;
- applicable syntax/code checks;
- throughput/resource report;
- registered external gates defined by the experiment.

Every evaluator receives exact run, checkpoint, data, tokenizer, device, seed, and metric-schema identity. A mismatch in any of these identities or in the evaluator contract stops the run. Every evaluation writes a report even on failure. A missing, empty, stale, or unparseable required report cannot produce PASS. Re-evaluation creates a new evaluation ID.

A run is complete only when required internal evaluation, required gates, final model/tokenizer load, manifests, reports, and read-back checks all pass. The immutable run `COMPLETE.json` binds the final publication manifest hash, final selected/exported model and tokenizer hashes, required checkpoint/evaluation/report hashes, the final `run_identity.json` hash, and the final `log-manifest.json` index hash. The log index is frozen after `COMPLETE.json` is published.

## 13. Extension model

V2 defines narrow interfaces:

~~~text
SourceReader
RowTransformer
Deduplicator
Splitter
TokenizerAdapter
Packer
DatasetValidator
ModelFactory
FeatureModule
TrainerBackend
CheckpointAdapter
Evaluator
MetricSuite
Gate
SelectionPolicy
~~~

Registry entries declare name, version, schema, stage, maturity, compatibility, and test contract.

Maturity rules:

- experimental: explicit opt-in;
- validated: default for its declared capability;
- retired: rejected except for reading an existing run.

Registries are explicit and source-controlled. V2 does not dynamically install or import arbitrary packages during training.

The lifecycle core owns safety and artifact publication. Extensions may alter model behavior or return metrics, but cannot bypass the GPU lease, rewrite manifests, mark a run successful, or publish outside the run allowlist.

## 14. CLI and developer experience

Release 1 commands:

~~~text
init
validate
plan
doctor
preflight
run
status
explain
diff
export
migrate-v1
~~~

Examples:

~~~powershell
.\.venv\Scripts\python.exe pipeline_v2\run.py init --template pretrain --name my_model --out pipeline_v2/configs/experiments/my_model.yaml
.\.venv\Scripts\python.exe pipeline_v2\run.py validate --config pipeline_v2/configs/experiments/my_model.yaml
.\.venv\Scripts\python.exe pipeline_v2\run.py plan --config pipeline_v2/configs/experiments/my_model.yaml
.\.venv\Scripts\python.exe pipeline_v2\run.py doctor
.\.venv\Scripts\python.exe pipeline_v2\run.py run --config pipeline_v2/configs/experiments/my_model.yaml
~~~

`init` creates a complete template. `validate` checks schema, paths, model divisibility, tokenizer capacity, and stage compatibility. `plan` prints the graph and file effects without writing data or acquiring the GPU. `doctor` checks the venv, direct dependencies, imports, disk, and writable roots without touching the GPU. `preflight --gpu` is the explicit device-capability gate. `explain` shows active fields, overrides, features, evaluators, outputs, and resume eligibility. `diff` compares resolved configurations semantically.

Migration is explicit:

~~~powershell
.\.venv\Scripts\python.exe pipeline_v2\run.py migrate-v1 --input configs/pilot.yaml --output pipeline_v2/configs/experiments/pilot_v2.yaml
~~~

The original V1 file is preserved. Migration never launches training.

## 15. Security and compatibility boundaries

Release 1 must use safetensors or weights-only loading for model artifacts. It must not use config-derived unsafe deserialization. Path containment and symlink checks apply to every user-provided path. Backup/export code must use an allowlist that is actually passed to the upload operation. Secrets remain in environment variables and never enter manifests or logs.

Network access is limited to explicitly declared data sources. GPU, training, and network tests are never part of ordinary CPU CI.

Existing WebUI, Flow, backup, and component entry points are not connected to V2 until their independent security, path, job-state, and evidence contracts are addressed.

## 16. Testing and CI

### CPU unit tests

Cover config, inheritance, duplicate/unknown keys, release/capability rejection, path containment, hashing, pinned/stale source rejection, manifests, atomic writes, split determinism, dedupe, token ranges, packing, model factories, checkpoint completeness, exact cursor restore, code/dependency identity, evaluator-set binding, metric gaps, segmented log/index recovery, event/monitor transitions, fake GPU lease behavior, bootstrap/final identity handoff and race recovery, CPU-only doctor behavior, evaluator reports, and evidence paths.

### CPU contract tests

Every source, transformer, model, backend, evaluator, and gate adapter must reject incomplete/mismatched input and publish valid artifacts.

### CPU integration test

A tiny local fixture runs:

~~~text
prepare -> tokenize -> one-step train -> evaluate -> publish
~~~

with no network or GPU.

### Explicit GPU acceptance

Only in a user-opened GPU window. The canonical Release 1 smoke command is:

~~~powershell
.\.venv\Scripts\python.exe pipeline_v2\run.py run --config pipeline_v2/configs/experiments/smoke.yaml
~~~

Its required read-back evidence is `runs/v2/<namespace>/<run_id>/reports/smoke-gate.json` together with `COMPLETE.json` and `manifests/run_identity.json`. The acceptance sequence is:

1. V2 smoke;
2. exact-command kill/resume, including uninterrupted-versus-resume state comparison;
3. checkpoint and manifest read-back;
4. pilot;
5. target probe;
6. target only after pilot gates pass.

### CI

A CPU CI lane runs V2 unit/contract/integration tests and schema/dependency checks. GPU and network tests are separate, explicit jobs. Every test result is persisted as a machine-readable artifact before being reported as PASS.

## 17. Migration and cutover

### Stage 0 — Freeze and inventory

Record V1 config/script/artifact behavior. Do not edit V1. Create V2 fixtures for smoke, pilot, and target.

### Stage 1 — Foundation

Implement package, schema, contracts, atomic store, CLI, and CPU tests. No training.

**Gate:** all foundation contracts pass.

### Stage 2 — Data and pretraining

Implement source/data adapters, tokenizer/packing, model factory, HF backend, and one-step integration.

**Gate:** local fixture produces complete verified artifacts.

### Stage 3 — Safety and recovery

Implement GPU lease, preflight, monitor, checkpoint state, exact resume, and OOM/divergence handling.

**Gate:** user-approved smoke and kill/resume pass.

### Stage 4 — S/P/T

Run Release 1 smoke, pilot, and target with full evidence read-back.

**Gate:** all Release 1 success criteria pass.

### Stage 5 — Release 2

Add exactly SFT, LoRA, KD, GDN, and mounting one feature at a time. Any additional capability requires a new release decision.

### Stage 6 — Adoption boundary

V1 remains read-only and unchanged for all releases covered by this design. V2 never modifies or installs delegation into V1; any future integration requires a separate approved design and explicit user decision. V2 can operate side-by-side with V1, but there is no automatic cutover.

## 18. Release 1 acceptance criteria

V2 Release 1 is accepted only when:

- invalid configs fail before expensive work;
- data generations are atomic, pinned, fresh, and self-describing;
- tokenizer and shard identity are verified;
- train/validation overlap is checked;
- changed experiments, code, dependencies, or launch profiles cannot resume old checkpoints;
- exact re-run resumes the same run and restores the complete training cursor;
- uncertain or busy GPU state blocks execution;
- bootstrap-to-final run identity handoff is atomic, crash-safe, and race-tested;
- metrics and health events are durable, complete, gap-free, and read back;
- NaN/Inf, missing metrics, evaluator mismatches, and missing artifacts cannot produce PASS;
- final model, tokenizer, manifests, and reports are read back;
- smoke, pilot, and target gates pass with the recorded S/P/T promotion references;
- V1 files and artifacts remain intact and no V1 entry point is modified;
- a researcher can create a new experiment by editing validated YAML.

## 19. Validation posture for this specification

This specification records static architecture findings and approved design decisions. It does not claim fresh model quality, GPU memory, training speed, current process state, or network availability. Any future implementation or performance claim must cite a repository-relative artifact produced by the corresponding validation command.
