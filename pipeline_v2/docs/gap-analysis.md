# pipeline_v2 gap analysis - what the other pipelines have that V2 does not

Status: **comparison report, not a change record.** Nothing here was implemented.

Compared: the frozen V1 core (`src/`, `scripts/`, `configs/`) and the component
lines (`diacritizer/`, `laya/`, `ui/`, `mex/`, `langid/`, `webui/`, `flow/`, `viz/`)
against `pipeline_v2/` (package `slm_train`). Every V2 claim was read first-hand;
defects that could be proved by running a command were run CPU-only with output
confined to `scratch/`, then deleted.

## 0. The one-paragraph answer

`pipeline_v2` is not a superset of the other pipelines. It is a **correctness and
evidence layer** wrapped around a **much thinner training core** than anything it
replaces. It is stronger on proof (atomic manifests, completion markers, exact-
resume identity, a real GPU lease, external gates, path containment, value-based
secret redaction) and weaker on **training itself**: it ignores part of its own
config, it cannot be stopped, it has no warmup, no scheduler, no gradient
accumulation, no multi-source corpora, no Hugging Face source, and a migration
command that emits a config `validate` always rejects.

The five headline capabilities V1 has and V2 lacks - SFT, LoRA, KD, GDN,
mounting - are **not** defects. The approved design defers exactly those to
Release 2 and refuses them by name today. Treating that list as the gap is the
main way to get this analysis wrong.

## 1. Baseline: what pipeline_v2 actually is

- 11 CLI commands (`slm_train/cli.py:22-29`): `init`, `validate`, `plan`, `doctor`,
  `preflight`, `run`, `status`, `explain`, `diff`, `export`, `migrate-v1`.
- 13-stage static graph (`orchestration/graph.py:15-29`): validate, bootstrap,
  gpu_preflight, prepare, tokenize, dataset_validation, model_validation,
  evaluator_validation, resolve_final_identity, train, internal_evaluation,
  external_gates, publish_final.
- One capability, one architecture, one precision path: `R1_CAPABILITIES =
  {pretrain}`, `architecture=llama`, `precision=fp16`, `attention=sdpa`
  (`config/models.py:7,72-82,196`).
- 6 experiment configs, against roughly 150 V1 configs in `configs/` (27 `diac*`,
  25 `e63*`, 15 `e63s*`, 13 `laya*`, 9 `mu2*`, 7 `mount*`, 6 `mex*`, plus the
  `sft`, `kd`, `lora`, `gdn` and `yarn` families).

### 1.1 Where V2 is genuinely better than every other pipeline

| Capability | V2 location | Why it beats the others |
|---|---|---|
| Atomic, manifest-bound artifacts | `artifacts/atomic.py`, `artifacts/markers.py`, `orchestration/stages.py:198-234` | every stage marker is written from bytes re-read off disk; V1 writes `train_summary.json`, `eval_report.json` and `*.bin` in place |
| Exact-resume identity | `training/resume.py`, `identity/*` | a 16-key identity (command, cwd, env, device, code hash) refuses a changed resume; V1 auto-resumes on a directory scan (`scripts/train.py:25-33`) |
| Checkpoint selection from manifests | `training/selection.py:42-57` | only `step-NNNNNNNN` directories **with** a verifying manifest are candidates; V1 globs `checkpoint-*` |
| Cross-process GPU lease | `resources/lease.py` | filelock plus sidecar, fail-closed, never steals, never kills; V1 has no lock anywhere in the repo |
| External gates | `evaluation/external.py`, `evaluation/suites.py:106-110` | `artifact_integrity`, `checkpoint_reload`, `retention_policy` re-derive claims from published bytes; V1 has none |
| Raise-only gates | `evaluation/gates.py` | a gate returns `None` or raises, so ignoring the exception still fails |
| Deterministic dedupe-before-split | `data/dedupe.py:34-54` | V1 dedupes per candidate in legacy mode |
| Path containment and value-based redaction | `security/paths.py`, `security/secrets.py:28-46` | V1 redacts by key only, and nothing in the V1 training path redacts at all |
| Allocated VRAM ceiling before any tensor | `resources/device.py`, called at `orchestration/handlers.py:1287` | implements the `rules/gpu.md` hard-ceiling rule |
| OOM is `BLOCKED`, never a silent reduction | `orchestration/handlers.py:1447-1453` | V1 has no such policy |

---

## 2. Class A - REAL DEFECTS: config fields V2 declares and then ignores

These are the most serious findings, because they are not missing features. They
are fields a user can set in a YAML that `validate` returns `PASS` for, and which
change nothing about the run. A config that lies is worse than a config that
cannot express the thing.

### A1. `train.gradient_accumulation` is never applied - CRITICAL

- Declared: `configs/base/r1.yaml:54` (8), `nano_codesechnet.yaml:51` (8),
  `micro_full.yaml:60` (2). Schema: `config/models.py:91`.
- Actual: `orchestration/handlers.py:1358-1374` does one `backward()` and one
  `optimizer.step()` **per micro-batch**.
- Effect: the effective batch is `batch_size` (1), not `batch_size * accum` (8).
  The declared run is an 8x smaller run than the file says, with no warning.

### A2. `train.scheduler` is never read - CRITICAL

- Every committed config sets `cosine` (`base/r1.yaml:50`, `nano_codesechnet.yaml:45`,
  `micro_full.yaml:54`).
- There is **no** `torch.optim.lr_scheduler` anywhere in the package (zero grep
  hits). The learning rate is flat for the whole run.
- The checkpoint state pretends otherwise: the `scheduler` role is the literal
  `{"kind": "cosine-constant-lr"}` (`handlers.py:1554`).
- The unused HF path hardcodes a *different* scheduler: `lr_scheduler_type="linear"`
  (`training/hf_causal_lm.py:54`).

### A3. `train.optimizer` is never read - HIGH

- `config/models.py:86` declares `optimizer: str`; configs set `adamw`.
- `handlers.py:1277-1279` hardcodes `torch.optim.AdamW(..., weight_decay=0.0)`.
- There is **no `weight_decay` field** in `TrainingConfig`, so weight decay is
  permanently 0 and cannot be set from YAML at all.
- V1 selects `optim` per config, including `adamw_bnb_8bit` (bitsandbytes 8-bit
  Adam), and sets `adam_beta1=0.9`, `beta2=0.95`, `weight_decay`, `warmup_steps`.

### A4. `runtime.worker_count` is never read - MEDIUM

- `config/models.py:121`; configs set `worker_count: 0`.
- `stage_train` iterates an in-memory list of blocks directly
  (`handlers.py:1344-1351`). No `DataLoader` is built anywhere in the training path.

### A5. `train.evaluation_cadence` never applies independently - MEDIUM

- Declared separately from `checkpoint_every_steps` (`base/r1.yaml:55-56`), but the
  held-out loss is measured only inside the `final_step % checkpoint_every_steps`
  branch (`handlers.py:1412-1428`). The two cadences cannot differ in practice.

### A6. `data.filters` is recorded but never applied - HIGH

- `config/models.py:52`; all configs set `filters: []`.
- `handlers.py:647` writes it to the manifest as `filters={"declared":
  list(data.filters)}` - a record of an *intent*, stored beside a real filter
  value at `data/prepare.py:246-250`.
- The real filter is hardcoded: `"min_chars": 0` (`handlers.py:577`).
  `data/prepare.py:109-114` supports `min_chars`/`max_chars`, but no config value
  can reach them. V1 exposes `min_chars` per config and reports `filter_rejected`.

### A7. `data.normalization` is certified but never performed - HIGH (evidence integrity)

- Declared `unicode_nfkc` everywhere; recorded into `DataManifest.normalization`
  (`handlers.py:648`).
- The only normalization applied is `normalized_text()`, a whitespace-run collapse
  (`data/dedupe.py:11-15`, used at `data/prepare.py:108`). `prepare.py:250` and
  `:274` record the truth as `"ws-collapsed"`, which nobody reads, because the
  manifest that ships says `unicode_nfkc`.
- Net effect: **the published data manifest certifies a normalization that was
  never applied.** This is exactly the class of defect `rules/evidence.md` and the
  V2 marker design exist to prevent.

### A8. Only `data.sources[0]` is ever read - HIGH (evidence integrity)

- `config/models.py:47` declares a tuple of sources; `validate` accepts any number
  and requires each to pin a file set.
- `handlers.py:550` builds the reader from `data.sources[0]` and nothing else.
- `handlers.py:629-630` writes **all** declared `source_ids` into the
  `DataManifest`. A two-source config publishes a manifest claiming two pinned
  sources while only one was read.
- V1 (`scripts/prepare_data.py:14,238-239`) streams every candidate against **one
  shared SHA-1 set** (cross-source dedupe) and writes per-source stats plus a
  corpus-wide `dedupe_rate`.

### A9. `kind: huggingface` sources are unreachable - HIGH

- `config/models.py:33` allows `kind: huggingface`; a real `HFSourceReader` exists
  at `data/sources/huggingface.py` and is covered by `tests/network/test_hf_source.py`.
- `_source_reader` (`handlers.py:497-501`) imports `LocalSourceReader` and
  constructs it unconditionally. No stage references the HF reader.
- So a config declaring a Hugging Face source is resolved against a local file
  path. The approved R1 scope lists "local + declared Hugging Face data sources",
  so this is an R1 gap, not an R2 one.

### A10. `model.base_model` and the R2 surface - DELIBERATE

`config/models.py:9` and `features/registry.py:103-119` refuse these by name, and
`migrations/README.md` refuses to carry `pretrained_checkpoint`/`resume_from`
across. Correct as designed; see section 4.

### A11. `role: test` and `role: external-gate` sources are silently trained on - CRITICAL

- `config/models.py:11` declares `SourceRole = Literal["train", "validation",
  "test", "external-gate"]`, and `validate` accepts all four.
- `data/split.py:26-31` (`assign_split`) can only ever return `"train"` or
  `"validation"`. There is no third branch and no test-set carve-out.
- `handlers.py:632` records every source's declared role in
  `DataManifest.source_roles`, so the manifest faithfully says `external-gate`
  for a corpus that `assign_split` then placed in the training split.
- Effect: a corpus declared as held-out evaluation data is trained on, and the
  published data manifest attests that it was held out. `config/models.py:9`
  reserves a source role for external gates that the pipeline cannot honour.
- V1 has the same two-way split (train/val, positional cut) but does not *declare*
  a role it cannot honour, so it fails more honestly than V2 here.

## 3. Class B - REAL DEFECT: `migrate-v1` emits a config that always fails validation

Proved by running it, not by reading it. Command:
`& .\.venv\Scripts\python.exe pipeline_v2\run.py migrate-v1 --input configs\pilot.yaml`,
output written to `scratch/`, artifact deleted afterwards.

- The command reports `"status": "PASS"` and correctly proves the V1 input is
  byte-identical (`source_unchanged: true`).
- It migrated **2 of roughly 50 fields**: `name -> run.name`, `seed -> run.seed`.
- The rest were dumped as `not_migrated`, including shredded inline list content
  such as `- "def fibonacci(n)` and `- "class Stack`, which shows it parsed YAML
  sequence literals as if they were mapping keys.
- Cause 1: `_read_mapping` (`adapters/legacy_v1.py:59-74`) is a hand-rolled line
  splitter, **not a YAML parser**, although PyYAML is a hard dependency.
- Cause 2: `DIRECT_FIELDS` (`adapters/legacy_v1.py:47-54`) maps to V2 paths that
  **do not exist** in the V2 schema: `model.name`, `train.epochs`, `optim.lr`.
  `TrainingConfig` has no `epochs`, and there is no `optim` section at all.
- Cause 3: `render_v2_config` (`:77-95`) emits `provenance:`, a key absent from
  `ConfigDocument`, which is `extra="forbid"` (`config/models.py:14`), and never
  emits the required `schema_version: 2`.
- Verified consequence: `validate --config <migrated>` returns
  `{"status": "FAIL", "reason": "... No such file or directory: base\r1.yaml"}`.
  Even after fixing the `extends` path it would still fail on the missing
  `schema_version` and the forbidden `provenance` key.
- The suite never closes this: `tests/contract/test_v1_boundary.py` asserts the
  policy text and that the command is reachable, but never asserts that a migrated
  file validates.

## 4. Class C - DELIBERATE scope boundaries: not gaps

Source: `docs/superpowers/specs/2026-09-25-core-training-pipeline-v2-design.md`
sections 4.2 and 3.2, enforced machine-side at `features/registry.py:72`
(`_BY_NAME_R2 = {}`) and `config/models.py:9`.

| Capability | V1 status | V2 status | Verdict |
|---|---|---|---|
| SFT | `scripts/sft.py`, `scripts/sft_data.py`, 5 configs | refused by name | **R2 by design** |
| LoRA / PEFT | `src/model.py::maybe_wrap_peft`, `lora_example.yaml`, `h2*.yaml` | refused by name | **R2 by design** |
| Knowledge distillation | `scripts/kd.py`, 5 `kd_*.yaml`, teacher streaming | refused by name | **R2 by design** |
| GDN hybrid attention | `src/gdn.py` (495 lines), `GDNHybrid*` classes, `gdn_smoke.yaml` | refused by name | **R2 by design** |
| Mounting / gated attention | `scripts/mount.py`, 7 `mount_*.yaml`, A/B report | refused by name | **R2 by design** |
| bf16, FlashAttention | V1 is `fp16` only | refused | explicit hardware contract |
| Distributed workers, remote store, plugin runtime | - | refused | V3, explicitly reserved |
| Component-line migration | - | refused | spec 3.2, no component-line migration |
| WebUI / Flow / backup wiring | V1 has entry points | refused | spec L647 |
| Weight continuation from a checkpoint | V1 has it | refused | spec 4.1 L81, not R1 |
| Automatic checkpoint/log deletion | V1 `save_total_limit: 3` | refused | plan L31, needs a decision |
| Automatic V1 cutover or fallback | - | forbidden | spec L6 and section 17 |
| MoE / expert routing | **absent from V1 too** | absent | neither has it |
| DDP/FSDP, EMA, LR finder, label smoothing, focal loss, curriculum | absent from V1 | absent | spec 4.2 defers extra optimizers/losses/augmentation |

Correct framing for the five R2 capabilities: **a port backlog exists, a port plan
does not.** The spec calls them "these five *existing* advanced paths" and all
five are already implemented and measured on the V1 side. Porting them is a
release decision, not a bug fix, and each needs its own schema, identity rules,
tests, smoke run and acceptance gate.

## 5. Class D - Operational capabilities V1 has and V2 has no equivalent for

| Capability | V1 | V2 | Severity |
|---|---|---|---|
| **Cooperative stop** | `src/stop.py`: a `STOP` flag file honoured in `TrainerCallback.on_save`, so a stop always leaves a valid checkpoint | **none**. `run` has no cancel path and the loop runs to `max_steps` | **CRITICAL for operation.** A long T run cannot be halted without killing the process, which the repo forbids and which forfeits the last checkpoint. V1's own note: the UI never kills a live run |
| **Warmup** | `warmup_steps` per config (150 in `configs/next_pretrain.yaml`) | **none** | HIGH - full-rate LR from step 1 |
| **Best-model-at-end** | `load_best_model_at_end=True` with `metric_for_best_model=eval_loss` and `greater_is_better=False` | best-by-val-loss exists **only** inside `stage_train` (`_best_checkpoint`, `handlers.py:345`) and is unreachable from `training/selection.py::select_checkpoint`, which picks the *highest-numbered* verified checkpoint | HIGH - selected and best are different checkpoints |
| **`--dry-run`** | `scripts/run_custom.py --dry-run` prints the whole chain before running it | `plan` prints a static stage list, not the resolved commands | MEDIUM |
| **Multi-run status dashboard** | `scripts/status.py --runs --curve 5`: every phase, loss curve, tokens/sec, indicative MFU | `status --run <one path>` reads one published status; no curve, no cross-run comparison, no MFU | MEDIUM |
| **Hardware state report** | `scripts/hardware_state.py --all`: CPU, RAM, disk, network, GPU, VRAM, per-process | `doctor` is an env/import check; `preflight --gpu` reports VRAM only | MEDIUM |
| **Resource profiling over time** | `scripts/profile_resources.py --interval --minutes --summary` | a single `peak_reserved_bytes` and wall-clock at end of stage | LOW |
| **Generation sampling parameters** | `scripts/infer.py --temperature --top_k --top_p --sample` | greedy only; `evaluation/generation.py` is a 1-prompt did-it-produce-text smoke | MEDIUM - blocks any sampling-based evaluator |
| **Instruction-style eval** | `scripts/eval.py` instruction eval, `--label`, `--report-out` | generation and syntax smoke only | MEDIUM |
| **Context extrapolation probes** | `scripts/eval.py --ctx --rope-scaling --rope-factor --stream-window --sink-tokens --stream-positions`; `src/model.py::apply_rope_scaling`, `build_streaming_sink_mask`, `streaming_position_ids` | `ModelConfig` has no rope scaling, no window, no sink tokens; `LlamaConfig` is built with defaults | MEDIUM - also V1's only way to measure long-context claims |
| **YaRN context extension** | `rope_type: yarn` with `factor` and `original_max_position_embeddings` (`src/model.py:58-69`), `configs/yarn_4096.yaml` | no `rope_theta`, no `rope_scaling`, no `head_dim` in `ModelConfig` | MEDIUM |
| **Embedding transplant / widening** | `scripts/embed_transplant.py --dim --teacher --vocab-size` | none; `build_causal_lm` always builds a fresh model at `vocab_size` | LOW-MEDIUM |
| **Model soup / weight merging** | `scripts/soup_merge.py --config-from --mix --out` | none | LOW |
| **Checkpoint bundling / resume packs** | `scripts/bundle_checkpoint.py`, `scripts/bundle_resume_pack.py` (`BUNDLE_MANIFEST.json` with crc32) | run roots are repository-relative by design (`security/paths.py`, `resolve_output_dir`) | LOW - arguably a deliberate trade |
| **Hub publish** | `scripts/backup_to_hub.py --repo --path_in_repo --execute` (execute-gated) | `export` copies an allowlist locally; no Hub | LOW |
| **Teacher download / distillation cache** | `scripts/download_teacher.py`, `mex/scripts/distill_cache.py` | none | R2 |
| **Corpus build tooling** | `scripts/h2p2_build_corpus.py`, `h2_validate_corpus.py --allowed-oov`, `h2p2_validate_corpus.py` | `data/sources/*` read declared files; no corpus construction, no OOV gate, no license gate | MEDIUM |
| **TensorBoard / tfevents** | `TENSORBOARD_LOGGING_DIR` written by `train.py` | `hf_causal_lm.build_training_arguments` sets `report_to=[tensorboard]` but is **dead code**; `stage_train` uses a hand-rolled loop, so **no tfevents are ever written**. Curves exist only in `train/logs/metrics/*.jsonl` | MEDIUM |
| **Retrieval / agent-memory eval** | `scripts/agent_memory_eval.py` (recall@k, top-k) | none | LOW - different task |

### D1. Dead code that hides all of the above

`training/hf_causal_lm.py::HFCausalLMTrainer` implements only `prepare()`; `fit`,
`evaluate`, `checkpoint`, `resume` and `finalize` all raise a `PipelineError` saying
each is owned by a later orchestration task (lines 132-155). Its
`build_training_arguments()` is referenced only by `tests/unit/test_model_factory.py`.
So the module documented as the single place for `TrainingArguments` is never used
by a stage, and the real loop is the hand-rolled one in `orchestration/handlers.py`.
Every item in section 2 exists because the training loop was written outside the
declared training abstraction (`training/backend.py::TrainerBackend`), which nothing
in the package implements.

### D2. Further V1 capabilities found in the full source inventory

| Capability | V1 | V2 |
|---|---|---|
| **Executable code bench** | `scripts/mini_eval.py`: 16 tasks x 2 assert-tests, each written to a fresh `tempfile.TemporaryDirectory` and run via `subprocess.run` with a timeout, credited only on returncode 0 **and** a marker in stdout. Writes `pass_at_1` and per-task test credit | none. V2's `generation_smoke` is "did the decode produce text" (`evaluation/generation.py:10-20`) |
| **Achieved-capacity VRAM probe** | `scripts/vram_probe.py`: synthetic tokens, 2 warm-up steps excluded, `reset_peak_memory_stats`, optimizer-aware, `--lora` arm, and documented decision thresholds (peak <= 6.5 GB approve / 6.5-7.8 re-probe with 8-bit Adam / OOM stay at pilot) | `preflight --gpu` reports free VRAM; it never measures what actually fits at a given batch x sequence |
| **Regression gate against a previous eval** | `scripts/ctx_probe.py::delta_vs_history` compares each probe point to the checkpoint's stored `eval_report.json` | none; a run that gets worse scores the same as one that improves |
| **Per-point fault isolation and crash-resilient probe persistence** | `ctx_probe._flush()` writes both the per-point JSON and `summary.json` after every point, so a suspend mid-matrix loses nothing; OOM at one point is recorded and the matrix continues | none - V2 has no long-running multi-point probe |
| **Launch gate that inspects the data, not just the device** | `scripts/c12_preflight.py`: 7 checks including prompt-masking integrity (row 0 label is -100, last label is not), forgetting-guard shards present, SFT row counts matching `meta.json`, disk headroom, and a param target within +/-2% | `preflight` checks config, device and VRAM only |
| **On-policy judge pipeline** | `scripts/kt2_judge.py`: 4 stages (prompts, sample, score, select) with a deterministic prompt pool, append-only candidate sampling with torn-line-tolerant resume, token-budget-packed teacher scoring, a frozen rubric with a 5-step JSON parse ladder, an unparseable-rate gate, AST-gated selection, and a judge-vs-likelihood agreement metric | none |
| **Task-capability verification** | `scripts/eval.py` reports greedy and sampled AST parse pass rates; `sft_data.py` gates the corpus on `ast.parse` | V2's `syntax` suite measures the same thing (`evaluation/syntax.py`) - **V2 has this** |
| **Cross-tokenizer embedding alignment** | `scripts/embed_transplant.py`: raw-byte alignment, re-tokenise-and-average fallback, seeded Gaussian fallback, isometric QR lift, and a >= 60% exact-match gate (`sys.exit(2)` below it) | none |
| **HF dataset download / streaming mode** | `prepare_data.py`: `data_mode: stream|download`, `cache_dir`, `.env` merge, and a specific hint on `401|403|gated` | `data/sources/huggingface.py` exists but is unreachable (A9); no download/stream switch |
| **Supervision-mask correctness gate** | `c12_preflight.py` verifies prompt masking on the real dataset before a launch | none |

### D3. Where V1 is genuinely worse and must not be ported as-is

Recorded here so the comparison is honest in both directions:

- **No file leases or advisory locks anywhere.** V2's `resources/lease.py` is the
  only real mutual exclusion in the repository.
- **No atomic writes.** No `os.replace`, `tempfile` or `.tmp` in V1; 
  `train_summary.json`, `eval_report.json`, `custom_chain.json`, `mount_schedule.json`,
  `pairs.jsonl` and every `*.bin` shard are written in place, and
  `closeout_reflection.py` documents the partial-write risk in its own docstring.
- **Exactly one manifest** (`BUNDLE_MANIFEST.json`); no shard manifests, no
  checkpoint manifests, no per-run manifest, no per-shard checksum for token `.bin`.
- **No run identity.** No `run_id` field anywhere; provenance is three ad-hoc
  mechanisms.
- **No data-stage resume.** `tokenize_data.py` and `sft_data.py` overwrite wholesale
  on retry; auto-resume exists only in the four training scripts.
- **No network policy, no secrets handling in the data path**, exact-SHA1-only dedup
  (no near-dup), no per-document offsets, and a two-way train/val split with a
  positional cut rather than a salted split.
- **`rope_theta = 10000.0` is hardcoded in four places** with no config key, and
  token-id range validation on the uint32 cast is absent.

One V1 pattern is worth copying verbatim: the **ancestor-aware WDDM GPU guard** in
`run_custom.py`. It resolves the full process ancestor chain through a hand-rolled
`CreateToolhelp32Snapshot` walk and treats any liveness probe error as alive, which
is what makes "never co-run" hold on Windows. V2's lease is the better mechanism,
but this guard is the better *diagnostic* and is worth keeping as the message a
blocked operator sees.

## 6. Class E - Smaller verified defects

| Finding | Location | Note |
|---|---|---|
| Lease liveness ignores process start time | `resources/lease.py:44,58,198-199,240-243` | `recover_stale` checks only `process_is_alive(pid)`. A recycled PID reads as a live owner, so the lease stays `BLOCKED`. Fail-closed, so not a safety hole, but it is a recovery-availability gap |
| Heartbeat monitor is never called | `monitoring/heartbeat.py:7,16,49` | `HealthSnapshot` and `observe_liveness` are referenced by nothing outside their own module, although the design requires the monitor to check heartbeat |
| Secrets shorter than 6 characters are never redacted | `security/secrets.py:49,62` | `_MINIMUM_SECRET_LENGTH = 6`; a real short secret passes through |
| The declared run-tree layout differs from the spec | per-stage subtrees under `runs/v2/<ns>/<run>/<stage>/` | the spec declares a flat run tree; the deviation is recorded only in `.superpowers/sdd/pipeline-v2-progress.md`, which is outside the repository document map |

Checked and found **sound**, so not listed as gaps: every declared metric must have a
threshold and a direction (`config/models.py:208-211`), so the resources suite cannot
silently pass unbounded; external gates are populated
(`evaluation/suites.py:106-110`); the V1 boundary is machine-enforced and the frozen
20-file inventory is re-verified on every test run
(`tests/contract/test_v1_boundary.py:96-104`).

---

## 7. Class F - Documentation and routing drift (fix these, do not analyse them)

| Finding | Evidence |
|---|---|
| `pipeline_v2/README.md:62-64` still says `run` "is currently wired to refuse with BLOCKED" | `cli.py:273` implements `cmd_run`; `runs/v2/core/` holds 5 published runs with all 13 stage directories |
| `pipeline_v2/docs/acceptance.md:118-124` says the GPU sequence "has NOT been executed" | contradicted by the published `runs/v2/core/*` runs. Note a contract test currently pins that stale text, so the runbook and the test must move together |
| The repository router does not mention `pipeline_v2` | neither `AGENTS.md` (portfolio map) nor `rules/components/core.md` has a V2 entry, no V2 entry point, and no V2 smallest-validation command. The next agent routing by the map will not find V2 |
| V1 and V2 progress live outside the document map | no `pipeline_v2` task exists in `state/tasks.yaml`; the narrative record is `.superpowers/sdd/pipeline-v2-progress.md` |
| `configs/base/r1.yaml` is no longer a template | it has no `replace_me` placeholders, so `configs/experiments/{smoke,pilot,target}.yaml` are three-line extends that differ only in `run.name` and `run.stage`. The committed S/P/T templates no longer express three distinct experiments |

## 8. Class G - Component lines: what they have that V2 does not

Read from each line's scoped `AGENTS.md` plus its declared entry points.

| Line | Capability V2 lacks | V2 equivalent |
|---|---|---|
| `diacritizer/` | **Non-LM task head**: a char-level bidirectional encoder with a 15-class diacritic head, i.e. token classification, not next-token prediction | none. `config/models.py:72` fixes `architecture: llama` and `models/causal_lm.py` only builds a causal LM |
| `diacritizer/` | **Postprocessing / codec round-trip** (`src/` passthrough codec, `scripts/postproc.py`), with a rule that postprocessing edits must re-run the registered gate on the unchanged split | none; V2 goes tokenizer bytes straight to loss |
| `diacritizer/` | **Gold/anchor checkpoint registry**: the deployed anchor stays gold until a registered verdict changes it | none. The experiment registry records the same gap for the core line: there is no anchor evaluator |
| `laya/` | **Typed-decision scoring heads** and a **panel evaluator** family (`eval_phish_sym.py`, `eval_public_choice_sym.py`) | none; `evaluation/suites.py` has exactly four suites, all for one task |
| `laya/` | **Script-bound preflight**: `preflight_check.py --script <trainer>.py --config <cfg>.yaml`, required before ANY unattended long launch | V2's `preflight` binds a config and a device, never the trainer that will run |
| `laya/` | **Param-count as a pre-registration artifact** (`param_count.py`) | partly covered by the `model_validation` stage and `manifests/model.json` |
| `ui/` | **Structured output with a round-trip equality assertion** (`catalog.py`, `chain.py`, `validate.py`, `generator.py`) | none; V2 measures free text |
| `ui/` | **Corpus builder with a registered seed and per-archetype caps** (`build_corpus.py`, `build_space_bunny_50k.py`) | none; V2 reads declared files but never constructs a corpus |
| `mex/` | **Model surgery**: `net2net_widen.py`, `merge_soup.py`, adapter mounting | none |
| `mex/` | **Deterministic replay packs** that must never be silently recomposed | V2 is stronger here: generation identity makes a changed policy a new generation rather than an overwrite |
| `mex/`, `langid/` | **Multiple eval task families per checkpoint** (langid runs langid, topic, ner, emo, schemer, rank off one model) | one task, four suites |
| `langid/` | **CPU-only by design, with export steps** | V2's `export` is a local allowlist copy; no CPU/CUDA profile separation |
| all | **Loss-curve plotting** (`mex/scripts/plot_losses.py`, V1 `status.py --curve`) | none; V2 records JSONL segments and never plots |

### G1. The structural conclusion

The component lines are not smaller versions of V2. They train **different things**
(a 15-class tagger, a typed-decision scorer, a structured chain generator, a replay
expert, a language tagger) through **different evaluation contracts** (postprocessing
round-trips, safety panels, per-task families, anchor-gold verdicts). Porting them
into V2 is not a feature-list exercise: it means V2's hardcoded single shape -
`Literal["llama"]`, one causal-LM head, four suites - becomes a registry of task
types. The approved design defers exactly that with "No component-line migration"
(spec 3.2), which is the right call at this stage.

The parts worth taking from them now are narrow and cheap:

1. script-bound preflight (laya),
2. an anchor-evaluator concept (diacritizer, langid),
3. corpus builders with registered seeds (ui),
4. a postprocessing / decode stage, so an evaluation measures the published text
   rather than raw token ids.

## 9. Prioritized remediation plan

Ordered by blast radius, not by effort. P0 items are cases where pipeline_v2
currently **publishes a false claim** or **silently trains something other than
what the config says** - those are worse than a missing feature, because every
artifact the run publishes inherits the error.

### P0 - make the config true again (no new capability, only correctness)

| # | Item | Where | Why first |
|---|---|---|---|
| 1 | Apply `train.gradient_accumulation` in the loop | `orchestration/handlers.py:1358-1374` | every published run so far trained with an 8x smaller batch than its own config declared |
| 2 | Apply `train.scheduler` (cosine/linear) + add `warmup_steps` and `weight_decay` to `TrainingConfig` | `config/models.py:85-99`, `handlers.py:1277` | the LR schedule is currently flat and undeclared; every existing checkpoint's `scheduler` state says `cosine-constant-lr` |
| 3 | Honour `train.optimizer` | `handlers.py:1277-1279` | same class of defect |
| 4 | Perform, or refuse to certify, `data.normalization` | `data/prepare.py:108`, `handlers.py:648` | the data manifest currently certifies `unicode_nfkc` over data that was only whitespace-collapsed |
| 5 | Read every declared `data.sources` entry, or refuse more than one | `handlers.py:550,629-630` | the manifest currently claims pinned sources that were never opened |
| 6 | Wire `HFSourceReader` into `_source_reader` by `kind` | `handlers.py:497-501` | an R1 scope item that exists but is unreachable |
| 7 | Make `data.filters` reach `min_chars`/`max_chars`, or delete the field | `handlers.py:577,647` | a declared field that is recorded as intent and never applied |
| 8 | Add a cooperative stop: a `STOP` flag honoured at checkpoint boundaries, plus a `run --stop` | new; V1 reference `src/stop.py` | without it a long run can only be stopped by killing it, which forfeits the checkpoint and breaks the repo's no-kill rule |
| 9 | Fix `migrate-v1`: real YAML parsing, a field map that matches the V2 schema, emit `schema_version: 2`, no `provenance` key; add a test that a migrated file **validates** | `adapters/legacy_v1.py` | today it reports `PASS` and emits a file that cannot load |

Items 1-3 and 4-7 are each small and independently testable. Item 8 is the only
one that needs a new command. Item 9 is the only one with a user-visible
behaviour change (`migrate-v1` will start reporting `not_migrated` honestly rather
than claiming a translation it did not perform).

### P0b - two more correctness items from the full inventory

| # | Item | Where | Why |
|---|---|---|---|
| 10 | Honour `SourceRole` `test` and `external-gate`, or remove them from the enum | `config/models.py:11`, `data/split.py:26-31` | a declared held-out corpus is currently trained on and the manifest attests it was held out |
| 11 | Add an achieved-capacity probe to `preflight --gpu` (synthetic tokens, warm-up excluded, optimizer-aware) and a regression gate that compares a new eval report against the stored one | new; V1 references `scripts/vram_probe.py`, `scripts/ctx_probe.py::delta_vs_history` | V2 measures free VRAM, not what fits; and a run that regresses scores identically to one that improves |

### P1 - close declared-but-unfinished R1 items

| # | Item | Note |
|---|---|---|
| 10 | Reconnect `training/backend.py::TrainerBackend` and either implement `HFCausalLMTrainer.fit` or delete the module | removes the dead abstraction that caused 1-3, and makes a future LoRA/KD backend a real seam rather than a rewrite |
| 11 | Make `training/selection.py` select by the declared `selection_policy`, not by highest step number | today `select_checkpoint` and `_best_checkpoint` can disagree about which checkpoint the run stands for |
| 12 | Emit a heartbeat from long stages and read it in the monitor | `monitoring/heartbeat.py` is currently unreachable; the design requires it |
| 13 | Compare `process_start_time` in the lease | `resources/lease.py:240-243`; fail-closed today, but a recycled PID strands the lease |
| 14 | Restore a multi-run status view (curve, tokens/sec, cross-run comparison) | V1 `status.py` parity; this is what an operator actually reads |

### P2 - Release 2 decision (not a bug fix)

Porting SFT, LoRA, KD, GDN and mounting into V2 is a **release decision** that needs
its own approved spec, per spec 4.2. The V1 implementations and their measured
results are the input; the work is re-expressing them as registered V2 feature
modules with their own schemas, identity rules, tests, smoke runs and gates. This
is weeks of work, and doing it piecemeal inside a gap-fix change would put new
capabilities on top of a loop that is still wrong in items 1-3.

### P3 - documentation and routing

| # | Item |
|---|---|
| 15 | Update `pipeline_v2/README.md` and `pipeline_v2/docs/acceptance.md` to match the published runs; move the contract test that pins the stale text in the same change |
| 16 | Add a `pipeline_v2` entry to `AGENTS.md` (portfolio map) and to `rules/components/core.md`, with the smallest validation command |
| 17 | Register a `pipeline_v2` task in `state/tasks.yaml`; record the two adopted decisions in `state/decisions.yaml` |
| 18 | Regenerate `CURRENT_STATE.md` rather than hand-editing it |
| 19 | Restore distinct S/P/T templates, or document that S/P/T differ only in name and stage |

---

## 10. What this document does not claim

- No GPU work was run. Nothing here is a claim about trained-model quality.
- No file outside `pipeline_v2/docs/gap-analysis.md` was modified, and no V1
  artifact was touched. The one scratch artifact created for the `migrate-v1`
  proof was deleted.
- The R1/R2 boundary, the validation vocabulary and the evidence rules are taken
  from the approved design and `rules/`, not invented here.
- An earlier SDD ledger lists several of these as already closed (external gates
  empty, the resources suite unthresholded). Both were re-checked first-hand and
  are **sound** today: `EXTERNAL_GATE_NAMES` is populated
  (`evaluation/suites.py:106-110`) and every declared metric must carry a threshold
  (`config/models.py:208-211`). The ledger is stale, not the code.


---

---

---

---


---