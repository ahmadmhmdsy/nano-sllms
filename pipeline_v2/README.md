# Nano SLMs Pipeline V2

This directory contains the isolated V2 training package. It is a
self-contained modular monolith: it never imports V1, never discovers plugins at
runtime, and never reaches the network or a device without being told to.

## Dependency setup

For a newly approved dependency-setup window, run these commands from the
repository root. The lock command is the canonical resolution step, and the safe
sync installs the locked project plus its test extra into the existing repository
environment:

```powershell
uv lock --project pipeline_v2
uv sync --project pipeline_v2 --active --inexact --locked --extra test
```

Do not run a bare `uv sync`: a default exact sync may remove packages that are
not declared by `pipeline_v2/uv.lock` from the shared environment. Always retain
`--active`, `--inexact`, `--locked`, and `--extra test` in the install command.
Do not create a nested `pipeline_v2/.venv`; all V2 commands and pytest use the
repository-root `.venv`.

The current shared `.venv` is frozen. The one-time shared-venv drift was
explicitly authorized and accepted by the user, and the setup above records that
already-completed operation. It is not a current instruction to run lock or sync.
Do not perform further lock, sync, or environment changes until a new user
decision is recorded. The evidence for that decision is
`.superpowers/sdd/pipeline-v2-task-1-report.md`.

## CPU-safe command shell

Run the command shell from the repository root:

```powershell
& .\.venv\Scripts\python.exe pipeline_v2\run.py --help
```

Run the focused package-contract test through the repository venv:

```powershell
& .\.venv\Scripts\python.exe -m pytest pipeline_v2/tests/unit/test_package_contract.py -q
```

## The thirteen commands

| Command | What it does |
|---|---|
| `init` | write a complete validated template from `--template` (default `base/r1.yaml`), refusing to overwrite |
| `validate` | check schema, paths, dimensions, tokenizer, capabilities |
| `plan` | print the stage graph and file effects; writes nothing, touches no device; `--dry-run` also prints the resolved config |
| `doctor` | CPU-only environment and import checks |
| `hardware-state` | read-only CPU, RAM, and disk report; GPU is not probed (use `preflight --gpu`) |
| `preflight` | CPU checks, or a device report with `--gpu` |
| `run` | execute or exactly resume the stage graph; `--seed`, `--device`, `--precision` are the only accepted overrides |
| `status` | read published status and evidence only |
| `status-runs` | the latest published status per run under a namespace, read-only, with each run's loss-curve summary and tokens/sec where train published them |
| `explain` | resolved fields, overrides, capabilities, evaluators, outputs |
| `diff` | compare two resolved semantic configurations |
| `export` | copy exactly the allowlist it was given; there is no default |
| `migrate-v1` | translate a V1 config as data; never launches anything |

`preflight --gpu` is the only V2 command that probes a device. `run` executes
the complete 13-stage graph on the configured device, or exactly resumes a run
whose stages are already published; a stage that cannot run is reported
`BLOCKED` by the engine rather than guessed. A cooperative stop (`run --stop`)
places a STOP flag the train stage honours at its next optimizer-step
boundary, always preserving the checkpoint from that step.

## Where the pieces are

- `slm_train/contracts` — the typed, immutable, strict schemas everything binds to
- `slm_train/config` — loading, inheritance, canonicalization, safe overrides
- `slm_train/artifacts` — atomic writes, manifests, markers, log index, publication
- `slm_train/orchestration` — the stage graph, the engine, identity handoff, status
- `slm_train/training` — cursor and checkpoint state, exact resume, selection
- `slm_train/monitoring` — event bus, health classification, metrics
- `slm_train/evaluation` — declared suites, measurement-only evaluators, gates
- `slm_train/security` — path containment, model loading, redaction, export
- `slm_train/resources` — device, lease, preflight, doctor
- `slm_train/data` — source readers, preparation, tokenization, packing, validation, dedupe
- `slm_train/models` — model factory, causal LM, manifest, decode, yarn
- `slm_train/identity` — run ID, source, code, runtime, invocation fingerprints
- `slm_train/adapters` — legacy V1 migration, registry
- `slm_train/features` — capability registry

## Adapting it to a new experiment

YAML is the adaptation surface. Copy the closest of
`configs/experiments/{smoke,pilot,target}.yaml` and edit only the declared
fields; an unknown key is a validation error rather than a silent no-op, so a
mistake is visible instead of ignored. No Python edit is required for an
ordinary experiment.

## Acceptance

`docs/acceptance.md` is the Release 1 runbook: the GPU gate, the OOM policy,
the exact-command resume rule, the required read-back paths, and the acceptance
checklist. The CPU lanes pass. Several GPU criteria are settled by published
runs under `runs/v2/core/`, each cited in `docs/acceptance.md` by its exact run
path; the criteria no published run has executed yet stay labelled
`NEEDS_USER_DECISION`, not PASS.
