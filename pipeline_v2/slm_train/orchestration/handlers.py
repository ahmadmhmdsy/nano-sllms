"""R1 stage handlers, wired to the real building blocks.

Each handler is a pure function of its StageContext. A handler may only read
artifacts an EARLIER stage published, may only write outputs it declared, and
must write its own completion marker binding those outputs before returning
PASS. The engine checks the declarations independently; the handler is trusted
only because the engine does not trust it.

A stage with no handler here is a stage the engine will refuse to pass. That is
deliberate: an unimplemented stage must block the run, never be skipped.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml

from slm_train.orchestration.stages import StageContext, StageOutcome

__all__ = ["HANDLERS", "handler_for"]


def _fail(context: StageContext, reason: str) -> StageOutcome:
    return StageOutcome(label="FAIL", reason=reason, detail={"stage": context.spec.name})


def _blocked(context: StageContext, reason: str) -> StageOutcome:
    return StageOutcome(label="BLOCKED", reason=reason, detail={"stage": context.spec.name})


# ---------------------------------------------------------------- validate


def stage_validate(context: StageContext) -> StageOutcome:

    if context.already_complete():
        return StageOutcome(label="PASS", detail={"stage": context.spec.name, "reused": True})
    """Publish the resolved configuration every later stage reads.

    Two forms are written deliberately: YAML, because a human has to be able to
    read what the run was actually asked to do, and JSON, because the machine
    has to bind it by hash without a parser round trip. The config is the
    literal in-memory document the loader returned, not a re-read of the source
    file, so an override passed on the command line is visible here.
    """
    from slm_train.contracts.hash import canonical_json_bytes, canonical_sha256

    document = context.config
    if document is None:
        return _blocked(context, "the engine supplied no resolved configuration")

    as_json = document.model_dump(mode="json")
    payload = canonical_json_bytes(as_json)

    json_path = context.output_path("config/resolved.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(payload)

    yaml_path = context.output_path("config/resolved.yaml")
    yaml_path.write_text(
        yaml.safe_dump(as_json, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )

    try:
        context.write_stage_marker(
            intended={
                "config/resolved.json": payload,
                "config/resolved.yaml": yaml_path.read_bytes(),
            }
        )
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(
        label="PASS",
        detail={
            "config_hash": canonical_sha256(as_json),
            "run": document.run.name,
            "stage": document.run.stage,
            "device": document.runtime.device,
        },
    )


# --------------------------------------------------------------- bootstrap


def stage_bootstrap(context: StageContext) -> StageOutcome:

    if context.already_complete():
        return StageOutcome(label="PASS", detail={"stage": context.spec.name, "reused": True})
    """Compute what the run was ASKED to do, before any hardware is touched.

    The bootstrap identity deliberately excludes the attempt, so a retry of the
    same work is recognisable as the same work. The attempt id then distinguishes
    two launches of it. Both are recorded, and neither is a clock or a random
    value: bootstrap_attempt_id is pure in its four inputs.
    """
    from slm_train.config.canonical import semantic_hash
    from slm_train.orchestration.bootstrap import BootstrapIdentity, bootstrap_attempt_id

    resolved = context.read_published_json("validate", "config/resolved.json")
    if not resolved:
        return _blocked(context, "the validate stage published an empty configuration")

    document = context.config
    identity = BootstrapIdentity(
        logical_experiment=str(document.run.name) + "-" + str(document.run.stage),
        config_hash=semantic_hash(resolved),
        source_identity_hash=semantic_hash(
            {
                "sources": [
                    source.model_dump(mode="json") for source in document.data.sources
                ]
            }
        ),
        tokenizer_identity_hash=semantic_hash(
            {
                "source": document.tokenizer.source,
                "revision_or_files": (
                    list(document.tokenizer.revision_or_files)
                    if isinstance(document.tokenizer.revision_or_files, tuple)
                    else document.tokenizer.revision_or_files
                ),
            }
        ),
        launch_profile_hash=semantic_hash(
            {
                "launch_profile": document.runtime.launch_profile,
            }
        ),
        requested_runtime_hash=semantic_hash(
            {
                "device": document.runtime.device,
                "precision": document.runtime.precision,
            }
        ),
        code_dependency_hash=semantic_hash({"release": document.release}),
        command_hash=semantic_hash({"argv": list(sys_argv())}),
    )
    bootstrap_id = identity.digest()

    launch_nonce = semantic_hash(
        {"pid": os.getpid(), "started": round(float(psutil_create_time()), 3)}
    )[:32]
    attempt = bootstrap_attempt_id(
        bootstrap_id=bootstrap_id,
        launch_nonce=launch_nonce,
        pid=os.getpid(),
        process_start_time=float(psutil_create_time()),
    )

    record = {
        "schema_version": 1,
        "bootstrap_id": bootstrap_id,
        "attempt_id": attempt,
        "logical_experiment": identity.logical_experiment,
        "request": identity.as_dict(),
    }
    target = context.output_path("attempt/bootstrap.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        json.dumps(record, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )

    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(
        label="PASS", detail={"bootstrap_id": bootstrap_id, "attempt_id": attempt}
    )


# ---------------------------------------------------------- gpu_preflight


def stage_gpu_preflight(context: StageContext) -> StageOutcome:

    if context.already_complete():
        return StageOutcome(label="PASS", detail={"stage": context.spec.name, "reused": True})
    """Decide the device, and never touch one the configuration did not ask for.

    A CPU configuration takes the CPU preflight, which by construction never
    reaches a device. A GPU configuration would acquire the lease and probe.
    There is no path here that probes a GPU for a CPU run: an ordinary run must
    not silently claim hardware it was not given.
    """
    from slm_train.resources.preflight import run_cpu_preflight, run_gpu_preflight

    document = context.config
    device = str(document.runtime.device)
    if device != "cpu":
        # The GPU lane. This stage does not probe anything itself: it delegates
        # to resources.preflight, the only module allowed to call the probe, and
        # that module report is the stage evidence. A probe that fails is
        # BLOCKED. It is never a quiet CPU downgrade.
        report = run_gpu_preflight(Path(context.repo_root), device)
        _write_preflight(context, report)
        if report.get("decision") != "PASS":
            return _blocked(
                context,
                "the GPU preflight did not pass: " + str(report.get("reason", "")),
            )
        if report.get("gpu_touched") is not True:
            return _fail(
                context,
                "a GPU preflight reported that it did not touch a device",
            )
        if report.get("mode") != "gpu":
            return _fail(context, "a GPU preflight reported the wrong mode")
        try:
            context.write_stage_marker()
        except (OSError, ValueError) as error:
            return _fail(context, "could not bind the stage marker: " + str(error))
        return StageOutcome(
            label="PASS",
            detail={
                "mode": "gpu",
                "device": str(report.get("device", device)),
                "free_memory_bytes": int(
                    (report.get("estimate") or {}).get("free_memory_bytes", 0)
                ),
            },
        )

    report = run_cpu_preflight(Path(context.repo_root))
    _write_preflight(context, report)

    if report.get("status") != "PASS":
        return _blocked(
            context, "the CPU preflight did not pass: " + str(report.get("reason", ""))
        )
    if report.get("gpu_touched") is not False:
        return _fail(context, "a CPU preflight reported that it touched a device")

    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(label="PASS", detail={"mode": "cpu", "gpu_touched": False})


def sys_argv():
    import sys

    return list(sys.argv)


def psutil_create_time() -> float:
    """This process's start time, or the process start if psutil is unavailable."""
    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        return float(time.time())


# ------------------------------------------------------- resume and retention
def _step_dir_name(step: int) -> str:
    return "step-" + str(step).zfill(8)


def _published_steps(stage_root: Path) -> list[int]:
    """Every complete checkpoint this run published, oldest first."""
    from slm_train.training.checkpoint import is_complete

    root = stage_root / "checkpoints"
    if not root.is_dir():
        return []
    found: list[int] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("step-"):
            continue
        try:
            step = int(entry.name[5:])
        except ValueError:
            continue
        if is_complete(stage_root, step):
            found.append(step)
    return found


def _scores(stage_root: Path) -> dict[int, float]:
    """Held-out loss per published step, recorded so best is measurable."""
    from slm_train.artifacts.atomic import atomic_write_json

    path = stage_root / "checkpoints" / "scores.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, float] = {}
    for key, value in (raw.get("val_loss") or {}).items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _remember_score(
    stage_root: Path, step: int, val_loss: float, train_loss: float
) -> None:
    from slm_train.artifacts.atomic import atomic_write_json

    scores = _scores(stage_root)
    scores[step] = float(val_loss)
    trains = _train_losses(stage_root)
    trains[step] = float(train_loss)
    payload = {
        "val_loss": {str(k): v for k, v in sorted(scores.items())},
        "train_loss": {str(k): v for k, v in sorted(trains.items())},
    }
    target = stage_root / "checkpoints" / "scores.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, payload, replace_existing=True)


def _train_losses(stage_root: Path) -> dict[int, float]:
    """Training loss per published step, carried across a resume."""
    path = stage_root / "checkpoints" / "scores.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, float] = {}
    for key, value in (raw.get("train_loss") or {}).items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _best_checkpoint(stage_root: Path) -> str | None:
    """The published step with the lowest recorded held-out loss."""
    scores = _scores(stage_root)
    if not scores:
        return None
    best = min(scores, key=lambda step: scores[step])
    return _step_dir_name(best)


def _retained(stage_root: Path) -> list[str]:
    return [_step_dir_name(step) for step in _published_steps(stage_root)]


def _prune_checkpoints(
    stage_root: Path, keep_last: int, keep_best: int, protect: int
) -> list[str]:
    """Keep the best N and the last N checkpoints; delete the rest.

    Deleting a published checkpoint is the one destructive act this pipeline
    performs, so it is narrow and explicit: only directories this run itself
    published under its own checkpoints directory, never a best or a last that
    the policy asked to keep, and never the step just written.
    """
    import shutil

    steps = _published_steps(stage_root)
    if not steps:
        return []
    scores = _scores(stage_root)
    keep: set[int] = set(steps[-keep_last:]) if keep_last else set()
    if keep_best and scores:
        ranked = sorted((s for s in steps if s in scores), key=lambda s: scores[s])
        keep.update(ranked[:keep_best])
    keep.add(int(protect))
    keep.update(steps[-1:] if keep_last else [])
    for step in steps:
        if step in keep:
            continue
        shutil.rmtree(stage_root / "checkpoints" / _step_dir_name(step),
                      ignore_errors=True)
    return sorted(_step_dir_name(step) for step in _published_steps(stage_root))


def _release_device_cache() -> None:
    """Hand cached device blocks back before a second pass. Never raises."""
    try:
        from slm_train.resources.gpu_probe import release_cached_memory

        release_cached_memory()
    except Exception:
        return


def _labels_for_block(ids, block_mask):
    """Causal-LM labels for one block: -100 where the loss mask bit is 0.

    A document_masked packing block carries one loss bit per token in the
    packer's convention (1 = loss on, 0 = loss off); absent masks (concat
    packing) keep the historical labels-equal-ids behaviour.
    """
    import torch

    if block_mask is None:
        return ids
    loss_off = torch.as_tensor(bool_mask_off(block_mask), device=ids.device)
    masked = ids.masked_fill(loss_off, -100)
    if not bool((masked != -100).any()):
        # A fully masked block (a pure pad block) carries no loss by design:
        # cross-entropy over -100 is a NaN, not a measurement, so the block is
        # signaled as skipped rather than averaged in.
        return None
    return masked


def bool_mask_off(block_mask):
    import numpy as np
    return ~np.asarray(block_mask, dtype=bool)


def bool_mask(block_mask):
    import numpy as np
    return np.asarray(block_mask, dtype=bool)


def _held_out_loss(
    model,
    blocks,
    device: str,
    compute_dtype,
    autocast_device: str,
    autocast_enabled: bool,
    masks=None,
) -> float:
    """Held-out loss on the reserved blocks, with no grad and no training mode."""
    import torch

    if not blocks:
        return float("inf")
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        counted = 0
        with torch.no_grad():
            for index, block in enumerate(blocks):
                block_mask = masks[index] if isinstance(masks, list) else None
                ids = torch.tensor(
                    [list(map(int, block))], dtype=torch.long
                ).to(device)
                labels = _labels_for_block(ids, block_mask)
                if labels is None:
                    continue
                counted += 1
                with torch.autocast(
                    device_type=autocast_device,
                    dtype=compute_dtype,
                    enabled=autocast_enabled,
                    cache_enabled=False,
                ):
                    total += float(model(input_ids=ids, labels=labels).loss)
        return total / max(counted, 1)
    finally:
        model.train(was_training)


def _resume_from(
    stage_root: Path, identity: dict, model, optimizer, scheduler,
    autocast_device: str, compute_dtype, autocast_enabled: bool,
    scaler=None,
) -> tuple[int, float | None] | None:
    """Continue from the newest checkpoint THIS run published, or return None.

    A checkpoint is only accepted when its manifest carries this run id and
    this config hash. A foreign or mismatched checkpoint is never loaded: the
    run starts clean instead of silently training on someone else weights.
    Returns (step, train_loss) so a resume that needs no work still reports the
    step it resumed at and the loss it resumed with.
    """
    import torch

    from slm_train.training.checkpoint import is_complete

    steps = _published_steps(stage_root)
    if not steps:
        return None
    step = steps[-1]
    directory = stage_root / "checkpoints" / _step_dir_name(step)
    try:
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if str(manifest.get("run_id", "")) != str(identity["run_id"]):
        return None
    if str(manifest.get("config_hash", "")) != str(identity["config_hash"]):
        return None
    try:
        state = torch.load(
            directory / "state" / "model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        optim_state = torch.load(
            directory / "state" / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(optim_state)
        scheduler_state = torch.load(
            directory / "state" / "scheduler.pt",
            map_location="cpu",
            weights_only=False,
        )
        scheduler.load_state_dict(scheduler_state)
        if scaler is not None:
            scaler_state = torch.load(
                directory / "state" / "scaler.pt",
                map_location="cpu",
                weights_only=False,
            )
            scaler.load_state_dict(scaler_state)
    except (OSError, RuntimeError, ValueError, KeyError):
        return None
    resumed_step = int(manifest.get("global_step", step))
    losses = _train_losses(stage_root)
    return resumed_step, losses.get(resumed_step)


def _write_preflight(context: StageContext, report: dict) -> None:
    """Persist the preflight report as this stage evidence."""
    from slm_train.artifacts.atomic import atomic_write_json

    target = context.output_path("attempt/preflight.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report, replace_existing=True)


# ------------------------------------------------------------------ prepare


def _source_reader(context: StageContext, declaration):
    """Build the pinned reader for one declared source.

    The reader is constructed from the declared file set, not from a directory
    scan. A scan would silently pick up a file the operator never declared and
    then pin it, which is the opposite of pinned.
    """
    if declaration.kind == "huggingface":
        from slm_train.data.sources.huggingface import HFSourceReader

        return HFSourceReader.from_repository(
            repository=declaration.path_or_repo,
            revision=declaration.revision,
            files=declaration.file_set,
            role=declaration.role,
            declared_source_id=declaration.source_id,
            expected_row_budget=declaration.expected_rows,
            expected_token_budget=declaration.expected_tokens,
            license_note=declaration.license,
        )
    from slm_train.data.sources.local import LocalSourceReader

    root = Path(context.repo_root) / Path(declaration.path_or_repo).parent
    files = tuple(root / name for name in declaration.file_set)
    return LocalSourceReader.from_file_set(
        files,
        role=declaration.role,
        declared_source_id=declaration.source_id,
        expected_row_budget=declaration.expected_rows,
        expected_token_budget=declaration.expected_tokens,
        license_note=declaration.license,
    )


def _reopen_prepared(published: Path, reader, data):
    """Reopen an already-published generation, reading its facts back from disk.

    The counts are taken from the published manifest rather than recomputed, so
    a reused generation reports exactly what was published the first time.
    """
    from slm_train.data.prepare import PreparedGeneration

    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    raw = json.loads((published / "raw-manifest.json").read_text(encoding="utf-8"))
    return PreparedGeneration(
        semantic_hash=manifest["content_hash"],
        counts=manifest.get("document_counts", {}),
        published_path=published,
        generation_key=published.name,
        split_assignments=raw.get("split_assignments", {}),
        instance_label="PASS",
        provenance=raw.get("provenance", {}),
    )


def stage_prepare(context: StageContext) -> StageOutcome:

    if context.already_complete():
        return StageOutcome(label="PASS", detail={"stage": context.spec.name, "reused": True})
    """Prepare one data generation, deterministically, and bind its manifest."""
    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.artifacts.markers import write_completion_marker
    from slm_train.config.canonical import semantic_hash
    from slm_train.contracts.data import DataManifest
    from slm_train.contracts.hash import canonical_json_bytes, sha256_bytes
    from slm_train.data.prepare import PreparationError, prepare_generation

    document = context.config
    data = document.data
    if not data.sources:
        return _blocked(context, "no source is declared, so there is nothing to prepare")

    try:
        reader = _source_reader(context, data.sources[0])
    except (OSError, ValueError) as error:
        return _fail(context, "the declared source cannot be pinned: " + str(error))

    from slm_train.data.generation import generation_key_for

    # The generation id is a function of the recorded preparation semantics, so
    # a changed source or a changed policy yields a NEW generation rather than
    # an implicit refresh of the old one.
    semantic = semantic_hash(
        {
            "dataset_id": data.dataset_id,
            "salt": data.split_salt,
            "validation_fraction": data.validation_fraction,
            "dedupe": data.dedupe,
            "normalization": data.normalization,
            # The declared filters are part of the preparation semantics: a
            # changed filter policy must yield a NEW generation rather than
            # an implicitly refreshed one that was never filtered.
            "filters": list(data.filters),
            "source_root_hash": reader.manifest.root_hash,
        }
    )
    generation_id = generation_key_for(data.dataset_id, semantic)

    # Parse declared filters: a "min_chars:n" or "max_chars:n" string
    # in data.filters is a filter value, not a recorded intent. The declared
    # filter must reach the real filter, or the manifest certifies a filter
    # that was never applied.
    min_chars = 0
    max_chars = None
    for item in data.filters:
        name, _sep, raw = str(item).partition(":")
        name = name.strip()
        try:
            value = int(raw.strip())
        except ValueError:
            raise ValueError("declared filter value is not an integer: " + str(item))
        if name == "min_chars":
            if value < 0:
                raise ValueError("min_chars filter must be nonnegative")
            min_chars = value
        elif name == "max_chars":
            if value < 1:
                raise ValueError("max_chars filter must be positive")
            max_chars = value
        else:
            raise ValueError("unknown data filter: " + str(item))

    options = {
        "dataset_id": data.dataset_id,
        "generation_id": generation_id,
        "split_salt": data.split_salt,
        "validation_fraction": data.validation_fraction,
        "dedupe": data.dedupe != "none",
        # The declared normalization is the applied one: a reader-level default
        # would silently flatten the ui contract's task/chain newline.
        "normalization": data.normalization,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "regeneration_command": tuple(data.regeneration_command),
        "license_note": data.sources[0].license,
    }

    repo_root = Path(context.repo_root)
    published = repo_root / "data/v2/prepared" / data.dataset_id / generation_id
    try:
        if published.is_dir():
            # A published generation is never overwritten, and it must not be
            # silently trusted either. An exact re-run REUSES the generation
            # only after re-verifying its marker against the bytes on disk; a
            # generation that does not verify is a real failure, not something
            # to republish over.
            from slm_train.artifacts.markers import verify_completion_marker
            from slm_train.data.prepare import PAYLOAD_NAME, RAW_MANIFEST_NAME

            if verify_completion_marker(
                published, [PAYLOAD_NAME, RAW_MANIFEST_NAME, "manifest.json"]
            ) is not True:
                return _fail(
                    context,
                    "a generation is already published at " + generation_id
                    + " but its completion marker does not verify",
                )
            prepared = _reopen_prepared(published, reader, data)
        else:
            prepared = prepare_generation(reader, options, repo_root)
    except PreparationError as error:
        return StageOutcome(
            label=getattr(error, "label", "FAIL") or "FAIL",
            reason=str(error),
            detail={"stage": "prepare"},
        )
    except FileExistsError:
        return _fail(
            context,
            "a generation is already published at " + generation_id
            + " and could not be verified for reuse",
        )
    except (OSError, ValueError) as error:
        return _fail(context, "preparation failed: " + str(error))

    if prepared.label != "PASS":
        return StageOutcome(
            label=prepared.label,
            reason="preparation is " + str(prepared.label),
            detail={"stage": "prepare"},
        )

    manifest = DataManifest(
        schema_version=1,
        source_ids=tuple(source.source_id for source in data.sources),
        source_roles={source.source_id: source.role for source in data.sources},
        source_identity={
            "kind": reader.manifest.kind,
            "root_hash": reader.manifest.root_hash,
        },
        source_fingerprint=reader.manifest.root_hash,
        dataset_config={
            "dataset_id": data.dataset_id,
            "generation_key": prepared.generation_key,
            "dedupe": data.dedupe,
            "normalization": data.normalization,
        },
        split_policy={
            "salt": data.split_salt,
            "validation_fraction": data.validation_fraction,
            "policy": "deterministic-hash",
        },
        # The applied filter values are what the preparation lane
        # actually enforced (the parsed declared values above, passed to
        # prepare_generation); the declared strings stay recorded for
        # traceability. A manifest that recorded only the declared intent
        # would certify a filter that was never proven to be applied.
        filters={
            "min_chars": min_chars,
            "max_chars": max_chars,
            "declared": list(data.filters),
        },
        normalization=data.normalization,
        dedupe_mode=data.dedupe,
        document_counts=dict(prepared.counts),
        license_note=data.sources[0].license,
        regeneration_command=tuple(data.regeneration_command),
        content_hash=prepared.semantic_hash,
    )

    # The prepared generation's own marker is bound here as evidence, so this
    # stage does not merely assert that preparation happened: it binds the exact
    # marker the preparation lane wrote.
    source_marker = Path(prepared.published_path) / "COMPLETE.json"
    if not source_marker.is_file():
        return _fail(context, "the prepared generation published no completion marker")

    # The stage publishes a RECORD naming the published generation, not a copy
    # of its marker. A later stage must be able to find the generation by name;
    # scanning the published tree for whichever directory looks right is exactly
    # how a stage ends up reading another run's data.
    record = {
        "status": "PASS",
        "kind": "prepared-generation",
        "published_path": Path(prepared.published_path).relative_to(
            Path(context.repo_root)
        ).as_posix(),
        "generation_key": prepared.generation_key,
        "source_marker_sha256": sha256_bytes(source_marker.read_bytes()),
    }
    target = context.output_path("prepared_generation/COMPLETE.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, canonical_json_bytes(record))

    data_target = context.output_path("manifests/data.json")
    data_target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(data_target, canonical_json_bytes(manifest.model_dump(mode="json")))

    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(
        label="PASS",
        detail={
            "content_hash": manifest.content_hash,
            "generation_key": prepared.generation_key,
            "counts": dict(prepared.counts),
        },
    )


# ------------------------------------------------------------------ tokenize


def _pinned_tokenizer(context: StageContext, data_content_hash: str):
    """Pin the declared tokenizer to its exact local file set."""
    from slm_train.data.tokenizer import pin_local_tokenizer

    declared = context.config.tokenizer
    raw = declared.revision_or_files
    if isinstance(raw, tuple):
        paths = [Path(context.repo_root) / name for name in raw]
    else:
        paths = [Path(context.repo_root) / str(raw)]

    specials = declared.special_ids
    return pin_local_tokenizer(
        paths,
        vocabulary_size=_tokenizer_vocabulary(paths[0]),
        max_token_id=_tokenizer_vocabulary(paths[0]) - 1,
        package_version=_tokenizer_version(),
        bos=specials.get("bos"),
        eos=specials.get("eos"),
        pad=specials.get("pad"),
        special_tokens=["<pad>", "<bos>", "<eos>"],
        context_length=declared.context,
        offset_behavior=declared.offsets,
    )


def _tokenizer_vocabulary(path: Path) -> int:
    """The vocabulary size, read from the tokenizer file, never declared by hand."""
    document = json.loads(path.read_text(encoding="utf-8"))
    model = document.get("model") or {}
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        return len(vocab)
    raise ValueError("the tokenizer file declares no readable vocabulary: " + str(path))


def _tokenizer_version() -> str:
    try:
        import tokenizers

        return "tokenizers==" + str(tokenizers.__version__)
    except Exception:
        return "tokenizers==unknown"


def stage_tokenize(context: StageContext) -> StageOutcome:
    """Tokenize the prepared generation and pack it into verified shards."""
    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.contracts.hash import canonical_json_bytes, sha256_bytes
    from slm_train.contracts.tokenizer import TokenizerManifest
    from slm_train.data.packing import pack_tokens

    if context.already_complete():
        return StageOutcome(label="PASS", detail={"stage": "tokenize", "reused": True})

    data_manifest = context.read_published_json("prepare", "manifests/data.json")
    data_content_hash = data_manifest["content_hash"]

    try:
        pinned = _pinned_tokenizer(context, data_content_hash)
    except (OSError, ValueError) as error:
        return _fail(context, "the tokenizer cannot be pinned: " + str(error))

    documents = _read_prepared_documents(context, data_manifest)
    if not documents:
        return _fail(context, "the prepared generation contains no documents to tokenize")

    from tokenizers import Tokenizer

    encoder = Tokenizer.from_file(
        str(Path(context.repo_root) / str(context.config.tokenizer.revision_or_files))
    )

    sequence_length = context.config.packing.sequence_length
    retain_tail = context.config.packing.dropped_tail_policy == "retain"
    mode = context.config.packing.mode
    repo_root = Path(context.repo_root)

    if mode == "document_masked":
        # Prompt-masked packing: each document is tokenized on its own, and the
        # TASK prefix (everything through the first newline, exactly the
        # contract ui/scripts/prepare_u1.py wrote) gets loss-mask bit 0;
        # the chain spec body keeps loss. A token that straddles the boundary
        # keeps loss, so no chain token is ever silently masked.
        tokens = []
        docs = []
        for text in documents:
            text = str(text)
            encoding = encoder.encode(text)
            cut = text.find("\n")
            cut = (cut + 1) if cut != -1 else len(text)
            doc_mask = [1 if off[1] > cut else 0 for off in encoding.offsets]
            tokens.extend(int(v) for v in encoding.ids)
            docs.append({"tokens": list(encoding.ids), "mask": doc_mask})
        documents_payload = docs
    else:
        # Concat packing keeps the historical whole-stream encode.
        tokens = []
        documents_payload = None
        for text in documents:
            tokens.extend(encoder.encode(str(text)).ids)

    if not tokens:
        return _fail(context, "the prepared generation tokenized to zero tokens: FAIL")

    # The token generation is content-addressed, so an exactly repeated run
    # addresses the same directory. It must be REUSED after its marker is
    # re-verified, exactly as a published prepared generation is: never
    # overwritten, never silently trusted.
    from slm_train.contracts.hash import canonical_sha256
    from slm_train.data.packing import PackResult

    stage_hash_payload = {
        "mode": mode,
        "sequence_length": sequence_length,
        "retain_tail": bool(retain_tail),
        "tokens": [int(v) for v in tokens],
    }
    if mode == "document_masked":
        # mirror the packer's mask_schema number: retained-tail mask bits are
        # part of a published document_masked generation, so a pre-fix
        # generation (all-pad masks) can not be silently reused.
        stage_hash_payload["mask_schema"] = 2
    content_hash = sha256_bytes(canonical_sha256(stage_hash_payload).encode("utf-8"))
    existing = repo_root / "data/v2/tokens" / content_hash[:16]

    if existing.is_dir():
        from slm_train.artifacts.markers import verify_completion_marker
        from slm_train.data.shards import ShardRecord

        if verify_completion_marker(existing, ["manifest.json"]) is not True:
            return _fail(
                context,
                "a token generation is already published at "
                + existing.name
                + " but its completion marker does not verify",
            )
        published_manifest = json.loads(
            (existing / "manifest.json").read_text(encoding="utf-8")
        )
        result = PackResult(
            counts=published_manifest.get("counts", {}),
            shards=[
                ShardRecord(path=str(s["path"]), size_bytes=int(s["size_bytes"]), sha256=str(s["sha256"]))
                for s in published_manifest.get("shards", [])
            ],
            manifest=published_manifest,
            manifest_path=existing / "manifest.json",
            published_path=existing,
            tokenizer=pinned,
        )
    else:
        try:
            result = pack_tokens(
                tokens,
                sequence_length=sequence_length,
                mode=mode,
                root=repo_root,
                retain_tail=retain_tail,
                pad_token_id=context.config.tokenizer.special_ids.get("pad"),
                tokenizer=pinned,
                source_data_hash=data_content_hash,
                documents=documents_payload,
            )
        except FileExistsError:
            return _fail(
                context,
                "a token generation was published concurrently and could not be verified",
            )
        except (OSError, ValueError) as error:
            return _fail(context, "packing failed: " + str(error))

    manifest = pinned.manifest
    tokenizer_target = context.output_path("manifests/tokenizer.json")
    tokenizer_target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        tokenizer_target, canonical_json_bytes(manifest.model_dump(mode="json"))
    )

    # The published token generation's own marker is bound as evidence.
    source_marker = Path(result.published_path) / "COMPLETE.json"
    if not source_marker.is_file():
        return _fail(context, "the tokenized generation published no completion marker")
    record = {
        "status": "PASS",
        "kind": "tokenized-generation",
        "published_path": Path(result.published_path).relative_to(
            Path(context.repo_root)
        ).as_posix(),
        "source_marker_sha256": sha256_bytes(source_marker.read_bytes()),
    }
    target = context.output_path("tokenized_generation/COMPLETE.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, canonical_json_bytes(record))

    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(
        label="PASS",
        detail={
            "tokenizer_hash": manifest.content_hash,
            "blocks": int(getattr(result, "usable_blocks", 0) or 0),
            "documents": len(documents),
        },
    )


def _read_prepared_documents(context: StageContext, data_manifest: dict) -> list:
    """Read the prepared documents back from the published generation."""
    # The path comes from the prepare stage's own published record, never from
    # a directory scan: a scan could hand this stage a different run's corpus.
    record = context.read_published_json("prepare", "prepared_generation/COMPLETE.json")
    published = record.get("published_path")
    if not isinstance(published, str) or not published:
        return []
    payload = Path(context.repo_root) / published / "prepared.jsonl"
    if not payload.is_file():
        return []
    texts = []
    for line in payload.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            texts.append(record.get("text", ""))
    return texts


# ------------------------------------------------------ dataset_validation


def stage_dataset_validation(context: StageContext) -> StageOutcome:
    """Reopen the published token generation and validate it against the contract."""
    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.contracts.hash import canonical_json_bytes, canonical_sha256
    from slm_train.contracts.tokenized import TokenizedDataManifest
    from slm_train.data.validate import validate_packed_tokens

    if context.already_complete():
        return StageOutcome(
            label="PASS", detail={"stage": "dataset_validation", "reused": True}
        )

    tokenizer_manifest = context.read_published_json("tokenize", "manifests/tokenizer.json")
    data_manifest = context.read_published_json("prepare", "manifests/data.json")

    # The path comes from the tokenize stage's own published record. Scanning the
    # tokens tree for a directory that looks right would let this stage read
    # another run's shards, so there is no scan here at all.
    record = context.read_published_json("tokenize", "tokenized_generation/COMPLETE.json")
    published = record.get("published_path")
    if not isinstance(published, str) or not published:
        return _fail(context, "the tokenize stage recorded no published token path")
    # The validator reads the PUBLISHED RECORD, which carries the pinned
    # tokenizer identity alongside the shards. The bare contract
    # (tokenized-manifest.json) deliberately carries no tokenizer key, so
    # validating that file instead would be validating the wrong artifact.
    record_path = Path(context.repo_root) / published / "manifest.json"
    token_manifest_path = Path(context.repo_root) / published / "tokenized-manifest.json"
    if not record_path.is_file() or not token_manifest_path.is_file():
        return _fail(
            context, "the recorded token generation is not on disk: " + published
        )

    result = validate_packed_tokens(
        record_path,
        sequence_length=context.config.packing.sequence_length,
        expected_tokenizer_hash=canonical_sha256(tokenizer_manifest),
        strict=False,
    )
    if result.get("label") != "PASS":
        return _fail(
            context,
            "packed-token validation is "
            + str(result.get("label"))
            + ": "
            + str(result.get("reason", "")),
        )

    # The published record is richer than the contract, so it goes through the
    # packing module's own adapter rather than being validated directly, which
    # would reject the record's kind and per-shard mask fields as unknown keys.
    from slm_train.data.packing import tokenized_manifest_from_record

    manifest = tokenized_manifest_from_record(
        json.loads(token_manifest_path.read_text(encoding="utf-8"))
    )
    target = context.output_path("manifests/tokenized_data.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, canonical_json_bytes(manifest.model_dump(mode="json")))

    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(
        label="PASS",
        detail={"shards": len(manifest.shards), "usable_blocks": manifest.usable_blocks},
    )


# -------------------------------------------------------- model_validation


def stage_model_validation(context: StageContext) -> StageOutcome:
    """Build the declared model and record its real identity."""
    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.config.canonical import semantic_hash
    from slm_train.contracts.hash import canonical_json_bytes
    from slm_train.models.causal_lm import build_causal_lm
    from slm_train.models.manifest import build_model_manifest

    if context.already_complete():
        return StageOutcome(
            label="PASS", detail={"stage": "model_validation", "reused": True}
        )

    document = context.config
    tokenizer_manifest = context.read_published_json("tokenize", "manifests/tokenizer.json")

    try:
        model = build_causal_lm(
            model_config=document.model,
            capabilities=list(document.capabilities),
            data_context=int(context.config.packing.sequence_length),
            vocab_size=int(tokenizer_manifest["vocabulary_size"]),
            seed=int(document.run.seed),
        )
    except (OSError, ValueError) as error:
        return _fail(context, "the model cannot be built: " + str(error))

    manifest = build_model_manifest(
        model_config=document.model,
        model=model,
        attn_implementation=document.model.attention,
        source_hash=semantic_hash(
            {"architecture": document.model.architecture, "release": document.release}
        ),
    )
    # Phase 2 (operator-friction plan, item 4): the same disk gate the train
    # stage applies, so an undersized volume is refused before any evidence is
    # written rather than discovered after hours of training.
    from slm_train.resources.disk_gate import disk_gate_decision

    disk_decision = disk_gate_decision(
        context.stage_root(),
        checkpoint_bytes=int(manifest.capability_metadata["num_parameters"]) * 6,
        keep=int(context.config.train.keep_last_checkpoints)
        + int(context.config.train.keep_best_checkpoints),
    )
    if disk_decision["label"] == "BLOCKED":
        return _fail(context, disk_decision["reason"])

    target = context.output_path("manifests/model.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, canonical_json_bytes(manifest.model_dump(mode="json")))

    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(
        label="PASS",
        detail={"parameters": manifest.capability_metadata.get("num_parameters")},
    )


# --------------------------------------------------- evaluator_validation


def stage_evaluator_validation(context: StageContext) -> StageOutcome:
    """Bind the declared evaluators, their gates, and their thresholds."""
    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.contracts.evaluator import EvaluatorManifest
    from slm_train.contracts.hash import canonical_json_bytes
    from slm_train.evaluation.suites import suite_for

    if context.already_complete():
        return StageOutcome(
            label="PASS", detail={"stage": "evaluator_validation", "reused": True}
        )

    declared = context.config.evaluation
    for name in declared.suites:
        if suite_for(name) is None:
            return _blocked(
                context, "the suite " + str(name) + " is not declared in the R1 catalogue"
            )

    manifest = EvaluatorManifest(
        schema_version=1,
        evaluator_ids=tuple(declared.ids),
        evaluator_versions={
            identifier: version
            for identifier, version in zip(declared.ids, declared.versions)
        },
        suite_gates=tuple(declared.suites),
        metric_schemas={
            "metric": "finite-real; boolean and string values are not reportable"
        },
        cadence=str(declared.cadence_steps),
        thresholds=dict(declared.thresholds),
        threshold_directions=dict(declared.directions),
    )
    target = context.output_path("manifests/evaluator.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, canonical_json_bytes(manifest.model_dump(mode="json")))

    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(label="PASS", detail={"suites": list(declared.suites)})


# ------------------------------------------------ resolve_final_identity


def stage_resolve_final_identity(context: StageContext) -> StageOutcome:
    """Assemble the run identity from every input the run actually used.

    The fingerprint is computed from the manifests the earlier stages published,
    not from the configuration. That is the whole point: a run whose inputs were
    changed after the config was written gets a different identity, so it cannot
    be mistaken for a continuation of the earlier one.
    """
    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.config.canonical import semantic_hash
    from slm_train.contracts.hash import canonical_json_bytes, canonical_sha256
    from slm_train.contracts.run import RunIdentityManifest
    from slm_train.identity.run_id import InputFingerprint, final_run_id

    if context.already_complete():
        return StageOutcome(
            label="PASS", detail={"stage": "resolve_final_identity", "reused": True}
        )

    data_manifest = context.read_published_json("prepare", "manifests/data.json")
    tokenizer_manifest = context.read_published_json("tokenize", "manifests/tokenizer.json")
    tokenized_manifest = context.read_published_json(
        "dataset_validation", "manifests/tokenized_data.json"
    )
    model_manifest = context.read_published_json("model_validation", "manifests/model.json")
    evaluator_manifest = context.read_published_json(
        "evaluator_validation", "manifests/evaluator.json"
    )
    resolved = context.read_published_json("validate", "config/resolved.json")

    fingerprint = InputFingerprint(
        config_hash=canonical_sha256(resolved),
        data_hash=data_manifest["content_hash"],
        tokenizer_hash=tokenizer_manifest["content_hash"],
        tokenized_hash=canonical_sha256(tokenized_manifest),
        model_hash=canonical_sha256(model_manifest),
        evaluator_hash=canonical_sha256(evaluator_manifest),
        code_hash=semantic_hash({"release": context.config.release}),
        release="r1",
        launch_profile_hash=semantic_hash(
            {"launch_profile": context.config.runtime.launch_profile}
        ),
        runtime_hash=semantic_hash(
            {
                "device": context.config.runtime.device,
                "precision": context.config.runtime.precision,
            }
        ),
        invocation_hash=semantic_hash(
            {"config": str(context.config.run.name), "stage": str(context.config.run.stage)}
        ),
    )

    experiment = str(context.config.run.name) + "-" + str(context.config.run.stage)
    run_id = final_run_id(experiment, fingerprint)
    manifest = RunIdentityManifest(
        schema_version=1,
        run_id=run_id,
        experiment_id=experiment,
        config_path=str(context.config.run.name),
        config_hash=fingerprint.config_hash,
        data_hash=fingerprint.data_hash,
        tokenizer_hash=fingerprint.tokenizer_hash,
        tokenized_hash=fingerprint.tokenized_hash,
        model_hash=fingerprint.model_hash,
        evaluator_hash=fingerprint.evaluator_hash,
        code_hash=fingerprint.code_hash,
        device_identity={"kind": str(context.config.runtime.device), "index": "0"},
        working_directory=".",
        evidence_root=run_id,
    )
    target = context.output_path("manifests/run_identity.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, canonical_json_bytes(manifest.model_dump(mode="json")))

    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(label="PASS", detail={"run_id": run_id})


# -------------------------------------------------------------------- train


def _load_blocks(context: StageContext, tokenized: dict) -> list:
    """Read the packed token shards back from disk into training blocks."""
    import numpy as np

    from slm_train.data.packing import TOKENIZED_MANIFEST_NAME

    record = context.read_published_json("tokenize", "tokenized_generation/COMPLETE.json")
    published = record.get("published_path")
    if not isinstance(published, str) or not published:
        return []
    root = Path(context.repo_root) / published
    manifest = json.loads((root / TOKENIZED_MANIFEST_NAME).read_text(encoding="utf-8"))
    sequence_length = int(manifest["sequence_length"])
    blocks = []
    for shard in manifest.get("shards", []):
        raw = np.fromfile(root / str(shard["path"]), dtype="<u4")
        for start in range(0, len(raw) - sequence_length + 1, sequence_length):
            blocks.append(raw[start : start + sequence_length])
    del tokenized
    return blocks


def _load_block_masks(context: StageContext, tokenized: dict) -> "list | None":
    """Per-token loss masks packed in document_masked mode, else None.

    The mask shard is the packing stage's own published companion and is
    validated for length exactly like the token shards it accompanies; a
    mismatch is a labeled FAIL, never a silent full-ones upgrade.
    """
    import numpy as np

    from slm_train.data.packing import MASK_UINT8, TOKENIZED_MANIFEST_NAME

    record = context.read_published_json("tokenize", "tokenized_generation/COMPLETE.json")
    published = record.get("published_path")
    if not isinstance(published, str) or not published:
        return None
    root = Path(context.repo_root) / published
    manifest = json.loads((root / TOKENIZED_MANIFEST_NAME).read_text(encoding="utf-8"))
    sequence_length = int(manifest["sequence_length"])
    has_masks = any(shard.get("mask_path") for shard in manifest.get("shards", []))
    if not has_masks:
        return None
    masks = []
    for shard in manifest.get("shards", []):
        mask_path = shard.get("mask_path")
        if not mask_path:
            raise ValueError("a document_masked token generation published a shard without a mask: FAIL")
        raw = np.fromfile(root / str(mask_path), dtype=np.uint8)
        token_count = int(shard["size_bytes"]) // 4
        if len(raw) > token_count:
            raise ValueError("mask shard is longer than its token shard: FAIL")
        if len(raw) < token_count:
            # The packer stores the mask for the real tokens only; the pad
            # region carries no loss by definition, so its mask bit is 0.
            raw = np.concatenate(
                [raw, np.zeros(token_count - len(raw), dtype=np.uint8)]
            )
        for start in range(0, len(raw) - sequence_length + 1, sequence_length):
            masks.append(raw[start : start + sequence_length])
    if not masks:
        raise ValueError("a document_masked token generation published mask shards but no usable mask blocks: FAIL")
    return masks


def _build_teacher(teacher_checkpoint: str, device: str):
    """Teacher reconstruction convention (ui/E-008): the checkpoint argument
    may point to a published model directory containing model_config.json, or
    to a weight-average.pt path whose sibling run root holds
    publish_final/final/model/model_config.json. State loads fp32 on CPU."""
    import json
    from pathlib import Path
    import torch
    from slm_train.models.causal_lm import build_causal_lm

    ckpt = Path(teacher_checkpoint)
    if ckpt.is_dir() and (ckpt / "model_config.json").exists():
        cfg_path = ckpt / "model_config.json"
    else:
        found = None
        probe = ckpt.parent
        for _ in range(6):
            cand = probe / "publish_final" / "final" / "model" / "model_config.json"
            if cand.exists():
                found = cand
                break
            probe = probe.parent
        if found is None:
            raise FileNotFoundError(
                "teacher model_config.json not found for: " + teacher_checkpoint
            )
        cfg_path = found
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    from slm_train.config.models import ModelConfig
    model_config = ModelConfig(
        architecture="llama", hidden=int(cfg["hidden"]), intermediate=int(cfg["intermediate"]),
        layers=int(cfg["layers"]), heads=int(cfg["heads"]), kv_heads=int(cfg["kv_heads"]),
        context=int(cfg["context"]), tied_embeddings=bool(cfg.get("tied_embeddings", True)),
        attention=str(cfg.get("attention", "sdpa")),
        precision=str(cfg.get("precision", "fp16")),
        base_model=None,
    )
    state_path = ckpt if ckpt.is_file() else ckpt / "model.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    vocab_expanded = int(state["model.embed_tokens.weight"].shape[0])
    teacher = build_causal_lm(model_config, capabilities=("pretrain",), vocab_size=vocab_expanded)
    missing, _ = teacher.load_state_dict(state, strict=False)
    if missing:
        raise ValueError("teacher state missing keys: " + ", ".join(list(missing)[:6]))
    return teacher.to(device=device)


def _kd_teacher_kl(
    teacher_checkpoint: str,
    input_ids,
    student_logits,
    device: str,
    beta_kl: float,
    temperature: float,
    chunk: int = 128,
    cache: dict | None = None,
):
    """Chunked teacher KL (ui/E-008, memory-safe): the teacher forward and the
    KL reduction run over sequence slices of size CHUNK, so peak memory stays
    the student's own forward, not the student's forward PLUS two fp32
    logit tensors. alpha weight is NOT applied here: the caller adds only the
    KL term to the student loss (alpha_ce stays as the plain student CE).

    The cache is per-run: a local dict passed by the caller, not a module-level
    global. This prevents cross-run contamination and memory leaks.
    """
    import torch
    import torch.nn.functional as F

    if cache is None:
        cache = {}
    cached = cache.get(str(teacher_checkpoint))
    if cached is None:
        teacher = _build_teacher(teacher_checkpoint, device=device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        cache[str(teacher_checkpoint)] = teacher
        cached = teacher
    T = max(float(temperature), 1e-6)
    total_kl = student_logits.new_zeros(())
    B, L, V = student_logits.shape
    c = max(int(chunk), 16)
    with torch.no_grad():
        for start in range(0, L, c):
            end = min(L, start + c)
            ids_chunk = input_ids[:, start:end]
            t_logits = cached(input_ids=ids_chunk).logits
            t_logits = t_logits.float()
            s_log = F.log_softmax(student_logits[:, start:end].float() / T, dim=-1)
            target = F.softmax(t_logits / T, dim=-1)
            total_kl = total_kl + F.kl_div(
                s_log, target, reduction="sum"
            ) / float(B * L)
    return beta_kl * total_kl * (T * T)


def stage_train(context: StageContext) -> StageOutcome:
    """Train on the configured device, checkpointing through the real publication path.

    The loop is a real one: real batches, a real optimizer, a real backward
    pass. It is bounded by the data rather than by a config field, because an
    unbounded loop is not a test.
    """
    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.artifacts.atomic import atomic_write_bytes, atomic_write_json
    from slm_train.contracts.checkpoint import CheckpointManifest
    from slm_train.contracts.hash import canonical_json_bytes
    from slm_train.monitoring.bus import EventBus
    from slm_train.monitoring.health import classify_health
    from slm_train.training.checkpoint import is_complete, publish_checkpoint
    from slm_train.training.state import CheckpointState, TrainingCursor

    if context.already_complete():
        return StageOutcome(label="PASS", detail={"stage": "train", "reused": True})

    document = context.config
    # The stage trains on the device the configuration named. gpu_preflight has
    # already decided that device is usable and has bound the probe report as
    # evidence, so this stage never selects one for itself and never falls back.
    device = str(document.runtime.device)

    identity = context.read_published_json(
        "resolve_final_identity", "manifests/run_identity.json"
    )
    from slm_train.contracts.checkpoint import CheckpointManifest
    model_manifest = context.read_published_json("model_validation", "manifests/model.json")

    tokenizer_manifest = context.read_published_json("tokenize", "manifests/tokenizer.json")
    tokenized_manifest = context.read_published_json(
        "dataset_validation", "manifests/tokenized_data.json"
    )
    evaluator_manifest = context.read_published_json(
        "evaluator_validation", "manifests/evaluator.json"
    )

    # Phase 2 (operator-friction plan): invalidation made loud. If the config
    # context moved after tokenization, these stale blocks would still train
    # with the wrong warmup mask and pass quietly. Refuse instead.
    prepared_context = int(tokenized_manifest["sequence_length"])
    configured_context = int(document.packing.sequence_length)
    if prepared_context != configured_context:
        return _fail(
            context,
            "the prepared data was tokenized for context "
            + str(prepared_context) + " but the configuration now declares "
            + str(configured_context)
            + "; re-run the prepare and tokenize stages for the new context.",
        )

    train_blocks, _held_out, _train_masks, _held_masks = _split_blocks(
        context, tokenized_manifest, float(document.data.validation_fraction)
    )
    if not train_blocks:
        return _fail(context, "the published token shards yield no complete training block")

    import torch
    from slm_train.models.causal_lm import build_causal_lm

    torch.manual_seed(int(document.run.seed))
    model = build_causal_lm(
        model_config=document.model,
        capabilities=list(document.capabilities),
        data_context=int(document.packing.sequence_length),
        vocab_size=int(tokenizer_manifest["vocabulary_size"]),
        seed=int(document.run.seed),
    )

    # E-007 native warm start: copy masked transplant rows into the student
    # embeddings at build time. Tied embeddings => lm_head is the same tensor.
    _warm = str(getattr(document.train, "warm_start_embeddings", "") or "")
    _warm_info = None
    if _warm:
        from pathlib import Path as _Path
        import json as _json
        _wp = _Path(_warm)
        if not _wp.is_absolute():
            _wp = _Path(context.repo_root) / _wp
        _blob = torch.load(_wp, map_location="cpu", weights_only=True)
        _W = _blob["embeddings"].float()
        _mask = _blob["mask"].bool()
        _emb = model.get_input_embeddings().weight
        if _W.shape[1] != _emb.shape[1]:
            raise ValueError(
                "warm-start hidden mismatch: transplant "
                + str(_W.shape[1]) + " vs model " + str(_emb.shape[1])
            )
        _n_copy = int(_mask.sum())
        _rows = _W[_mask]
        _idx = _mask.nonzero(as_tuple=True)[0]
        _idx = _idx[_idx < _emb.shape[0]]
        _rows = _rows[: _idx.shape[0]]
        with torch.no_grad():
            _emb[_idx] = _rows.to(_emb.dtype)
        _warm_info = {
            "source": str(_wp),
            "rows_transplanted": int(_idx.shape[0]),
            "student_vocab": int(_emb.shape[0]),
            "student_hidden": int(_emb.shape[1]),
        }
    # The declared precision is the PUBLISHED precision. Compute on CPU keeps
    # fp32 master weights and an fp32 optimizer, because fp16 CPU AdamW goes
    # non-finite within two steps on this profile. Downcasting the master
    # weights here would be a silent numerical-policy change; the split is
    # recorded in the stage record instead.
    # Master weights and optimizer state stay fp32 on EVERY device. Pure fp16
    # AdamW produces a non-finite loss at step 2 on this profile; that was
    # measured on CPU first and then reproduced on the GPU (step 1 loss 3.3634,
    # non-finite at step 2, evidence in train/logs/monitor). Downcasting the
    # master weights would be a silent numerical-policy change.
    #
    # What the declared precision buys on a GPU is the forward pass: it runs
    # under autocast, so the matmuls are fp16 while gradients accumulate in
    # fp32. That is the standard mixed-precision recipe, and the declared
    # precision still governs the PUBLISHED artifact. Both dtypes are recorded
    # in the stage record.
    master_dtype = torch.float32
    compute_dtype = master_dtype if device == "cpu" else torch.float16
    autocast_enabled = device != "cpu"
    autocast_device = torch.device(device).type
    # GradScaler for fp16 training: prevents gradient underflow in the backward pass.
    # The scaler is only used on GPU; CPU training uses fp32 and does not need it.
    # Use torch.amp.GradScaler to avoid a direct device reference in this module.
    scaler = torch.amp.GradScaler("cuda") if autocast_enabled else None
    # The device was named by the configuration and already probed by
    # gpu_preflight. This stage places the model there and never selects one
    # for itself, so a run cannot quietly train somewhere the operator did not
    # ask for.
    model = model.to(master_dtype).to(device)
    model.train()
    # The learning rate, the step budget, and the checkpoint retention policy
    # are config fields. They used to be named constants buried in this file,
    # which meant changing them required a code edit: exactly what a
    # config-driven pipeline must not make the operator do.
    learning_rate = float(document.train.learning_rate)
    max_steps = int(document.train.max_steps)
    from slm_train.training.schedule import build_optimizer, build_scheduler

    optimizer = build_optimizer(document.train, model.parameters(), learning_rate)
    scheduler = build_scheduler(document.train, optimizer, max_steps)

    # E-005 native monitor (reusable, config-driven): multi-metric probes at
    # a cadence aligned to the checkpoint boundary but respecting the
    # declared 1/N evaluation fraction, and (optionally) early stop driven
    # by the VALIDATION loss only - the optional probe TASKS file is a
    # reserved, never-trained file measured READ-ONLY.
    from slm_train.training.monitor import TrainMonitor, TrainMonitorConfig

    monitor_cfg = TrainMonitorConfig(document.train)
    monitor_cadence = max(
        1, min(max_steps, max(1, int(-(-max_steps * monitor_cfg.eval_fraction // 1))))
    )
    monitor_cadence = max(monitor_cadence, int(document.train.checkpoint_every_steps))
    _probe_task_rows = []
    if monitor_cfg.enabled and monitor_cfg.probe_tasks_file:
        from slm_train.evaluation.probes import load_probe_tasks

        _probe_path = Path(monitor_cfg.probe_tasks_file)
        if not _probe_path.is_absolute():
            _probe_path = Path(context.repo_root) / monitor_cfg.probe_tasks_file
        _probe_task_rows = load_probe_tasks(str(_probe_path), monitor_cfg.probe_tasks or 24, seed=int(document.run.seed))
    _loss_window: list = []

    def _monitor_task_probe():
        if not _probe_task_rows:
            return {}
        import torch
        from tokenizers import Tokenizer

        from slm_train.evaluation.probes import probe_result, pass_k_result

        encoder = Tokenizer.from_file(
            str(Path(context.repo_root) / str(document.tokenizer.revision_or_files))
        )
        pad_id = int(document.tokenizer.special_ids["pad"])
        pass_k = int(getattr(monitor_cfg, "probe_pass_k", 1) or 1)

        def gen_chunks(texts, do_sample, n):
            # Batched generation natively (monitor option, shared w/ E-005..E-012):
            # chunk tasks by padded length, one generate per chunk; the prompt
            # convention (task + newline) is unchanged. Returns per-task lists
            # of n decoded chains (n=1 -> single greedy/sampled chain).
            out = []
            for i0 in range(0, len(texts), 8):
                chunk = texts[i0 : i0 + 8]
                encs = [list(map(int, encoder.encode(tx).ids)) for tx in chunk]
                L = max(len(e) for e in encs)
                input_ids = torch.tensor(
                    [[0] * (L - len(e)) + e for e in encs], dtype=torch.long
                )
                attn = torch.tensor(
                    [[0] * (L - len(e)) + [1] * len(e) for e in encs], dtype=torch.long
                )
                kw = dict(
                    max_new_tokens=int(monitor_cfg.probe_max_new_tokens),
                    do_sample=do_sample,
                    num_beams=1,
                    pad_token_id=pad_id,
                )
                if do_sample:
                    kw["temperature"] = float(monitor_cfg.probe_temperature)
                    kw["top_p"] = float(monitor_cfg.probe_top_p)
                if n > 1:
                    kw["num_return_sequences"] = n
                generated = model.generate(
                    input_ids=input_ids.to(device),
                    attention_mask=attn.to(device),
                    **kw,
                )
                for j in range(len(chunk)):
                    n_in = L
                    if n > 1:
                        out.append([
                            encoder.decode(
                                [int(t) for t in generated[j * n + r, n_in:]],
                                skip_special_tokens=True,
                            )
                            for r in range(n)
                        ])
                    else:
                        out.append([
                            encoder.decode(
                                [int(t) for t in generated[j, n_in:]],
                                skip_special_tokens=True,
                            )
                        ])
            return out

        tasks = [str(row.get("task", "")) + "\n" for row in _probe_task_rows]
        greedy_chains = gen_chunks(tasks, False, 1)
        for j, row in enumerate(_probe_task_rows):
            row["chain"] = greedy_chains[j][0]
        result = probe_result(_probe_task_rows)
        if pass_k > 1:
            sampled = gen_chunks(tasks, True, pass_k)
            result.update(pass_k_result(_probe_task_rows, sampled))
        return result

    _monitor = TrainMonitor(
        monitor_cfg, max_steps,
        context.output_path("logs/monitor_report.jsonl"),
        context.output_path("logs/monitor_summary.json"),
        task_probe=_monitor_task_probe if monitor_cfg.enabled else None,
    )
    _monitor.cadence = monitor_cadence

    # A hard allocator ceiling, applied before any tensor is made. rules/gpu.md
    # treats a spill into shared GPU memory as an OOM-equivalent event and
    # forbids trusting driver fallback behaviour, so the cap is set explicitly
    # by the one module allowed to touch a device.
    ceiling = float(document.runtime.vram_fraction)
    try:
        from slm_train.resources.device import enforce_allocator_ceiling

        ceiling = enforce_allocator_ceiling(device, ceiling)
    except Exception as error:
        return _fail(
            context, "the allocator ceiling could not be applied: " + str(error)
        )

    stage_root = context.stage_root()
    stage_root.mkdir(parents=True, exist_ok=True)
    bus = EventBus(stage_root, str(identity["run_id"]))
    bus.emit("run_started", {"stage": "train", "device": device})
    # TensorBoard writer: logs loss, LR, and gradient norm at each step so an
    # operator can visualize training curves in TensorBoard.
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_log_dir = stage_root / "logs" / "tensorboard"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_log_dir))
    except ImportError:
        pass  # tensorboard is optional; the run works without it
    # Phase 2 (operator-friction plan): refuse to train before any compute is
    # spent when the checkpoint evidence cannot physically fit on the disk.
    from slm_train.resources.disk_gate import disk_gate_decision

    disk_decision = disk_gate_decision(
        stage_root,
        checkpoint_bytes=int(model_manifest["capability_metadata"]["num_parameters"]) * 6,
        keep=int(document.train.keep_last_checkpoints)
        + int(document.train.keep_best_checkpoints),
    )
    if disk_decision["label"] == "BLOCKED":
        return _fail(context, disk_decision["reason"])

    # Throughput and peak memory are measured over the window that actually
    # trained, not estimated. The resources suite refuses to report a value it
    # was not given, so a run that skipped this measurement fails its own gate
    # instead of publishing a fabricated zero.
    from slm_train.resources.gpu_probe import peak_reserved_bytes, reset_peak_memory

    train_started = time.monotonic()
    reset_peak_memory(device)

    batch_size = int(document.train.batch_size)
    accum = int(document.train.gradient_accumulation)
    evaluation_cadence = int(document.train.evaluation_cadence)
    sequence_length = int(document.packing.sequence_length)
    kd_state = {"teacher": None}
    kd_teacher_cache: dict = {}
    if bool(document.train.kd.enabled):
        kd_invalid = []
        if not document.train.kd.teacher_checkpoint:
            kd_invalid.append("kd.teacher_checkpoint missing")
        elif not Path(document.train.kd.teacher_checkpoint).exists():
            kd_invalid.append("kd.teacher_checkpoint does not exist: " + str(document.train.kd.teacher_checkpoint))
        elif float(document.train.kd.alpha_ce) + float(document.train.kd.beta_kl) == 0.0:
            kd_invalid.append("kd alpha+beta sum to zero")
        if kd_invalid:
            raise ValueError("kd config invalid: " + "; ".join(kd_invalid))
        kd_state["teacher"] = str(document.train.kd.teacher_checkpoint)
        bus.emit("health_warning", {"reason": "kd teacher loaded: " + kd_state["teacher"]})

    final_step = 0
    last_loss = None
    out_of_memory = False
    step = 0
    last_epoch = 0
    stop = False

    # Auto-resume. If this run already published a checkpoint and the stage is
    # not complete, that checkpoint is the run own history, so training
    # continues from it instead of silently starting over. It only ever loads a
    # checkpoint this run wrote, matched on the run id, and the operator can
    # turn it off with train.resume.
    resumed_from = None
    resume_reason = "auto-resume is off (train.resume=false)"
    if bool(document.train.resume):
        carried = _resume_from(
            stage_root, identity, model, optimizer, scheduler, autocast_device,
            compute_dtype, autocast_enabled, scaler,
        )
        if carried is not None:
            resumed_from, carried_loss = carried
            step = int(resumed_from)
            resume_reason = ""
            # A resume that needs no further work still has a final step and a
            # last loss: both come from the checkpoint, not from a step this
            # invocation happened to run.
            final_step = step
            last_loss = carried_loss
            steps_per_epoch = max(1, len(train_blocks) // max(1, batch_size))
            last_epoch = step // steps_per_epoch
        if resumed_from is None:
            resume_reason = "no checkpoint of this run qualified for resume; starting fresh"

    # Phase 2 (operator-friction plan, item 5): the resume decision is an
    # operator-visible artifact, so a BAD run is never mistaken for a fresh
    # attempt. Written before training starts, whatever the decision was.
    resume_summary = {
        "resumed_from_step": resumed_from,
        "source_checkpoint": None if resumed_from is None
        else "step-%08d" % int(resumed_from),
        "identity_keys_compared": ["run_id", "config_hash", "tokenizer_hash",
                                   "model_hash", "code_hash"],
        "retrain_reason": resume_reason,
    }
    atomic_write_json(
        context.output_path("logs/resume_summary.json"), resume_summary,
        replace_existing=True,
    )

    # The budget is a step count, not an epoch count: a run over a large corpus
    # must be boundable without knowing how many epochs that corpus implies.
    # With gradient accumulation, max_steps is the number of optimizer steps,
    # and each optimizer step covers accum micro-batches.
    epoch = 0
    measured_tokens = 0
    # The optimizer-step counter is part of the resumed state: a checkpoint at
    # step N already consumed N optimizer steps of the budget, so a resumed run
    # must continue the budget from N instead of re-spending it from zero.
    optimizer_step = 0 if resumed_from is None else int(resumed_from)
    micro_batch = 0

    # Resource profiling: log CPU, RAM, GPU, VRAM every N seconds during training.
    # This allows an operator to diagnose memory leaks or performance degradation.
    import threading

    resource_timeseries_path = stage_root / "logs" / "resource_timeseries.jsonl"
    resource_timeseries_path.parent.mkdir(parents=True, exist_ok=True)
    resource_stop_event = threading.Event()

    def _resource_probe_worker():
        while not resource_stop_event.is_set():
            try:
                record = {
                    "timestamp": time.time(),
                    "step": step,
                    "optimizer_step": optimizer_step,
                }
                # CPU and RAM
                try:
                    import psutil
                    record["cpu_percent"] = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory()
                    record["ram_used_bytes"] = int(mem.used)
                    record["ram_total_bytes"] = int(mem.total)
                except ImportError:
                    pass
                # GPU and VRAM
                if autocast_enabled:
                    try:
                        from slm_train.resources.gpu_probe import memory_report
                        vram = memory_report(device)
                        record["vram_allocated_bytes"] = vram["allocated_bytes"]
                        record["vram_reserved_bytes"] = vram["reserved_bytes"]
                    except Exception:
                        pass
                with open(resource_timeseries_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass
            resource_stop_event.wait(10)  # Log every 10 seconds

    resource_thread = threading.Thread(target=_resource_probe_worker, daemon=True)
    resource_thread.start()

    val_loss = None  # Initialize so monitor can reference it before first eval
    while optimizer_step < max_steps and not stop:
        epoch += 1
        generator = torch.Generator().manual_seed(int(document.run.seed) + epoch)
        permutation = torch.randperm(len(train_blocks), generator=generator).tolist()
        for start in range(0, len(permutation) - batch_size + 1, batch_size):
            if optimizer_step >= max_steps:
                break
            # Cooperative stop: check for the STOP flag file at each step boundary.
            # The flag is written by run --stop and honoured here, so a long run
            # can be halted without killing the process. The checkpoint at the
            # current step is preserved.
            if (stage_root / "STOP").is_file():
                stop = True
                break
            batch = [train_blocks[i] for i in permutation[start : start + batch_size]]
            step += 1
            micro_batch += 1
            ids = torch.tensor(
                [list(map(int, row)) for row in batch], dtype=torch.long
            ).to(device)
            batch_labels = ids
            if isinstance(_train_masks, list):
                block_masks = torch.tensor(
                    [list(map(int, _train_masks[i])) for i in permutation[start : start + batch_size]],
                    dtype=torch.long,
                ).to(device)
                batch_labels = ids.masked_fill(
                    block_masks == 0, torch.tensor(-100, dtype=ids.dtype)
                )
            if isinstance(batch_labels, torch.Tensor) and not bool((batch_labels != -100).any()):
                # A batch of pure pad blocks carries no loss by design:
                # optimizing it would push a NaN through the optimizer.
                step -= 1
                continue
            measured_tokens += int(ids.numel())
            try:
                with torch.autocast(
                    device_type=autocast_device,
                    dtype=compute_dtype,
                    enabled=autocast_enabled,
                ):
                    output = model(input_ids=ids, labels=batch_labels)
                    loss = output.loss
                    if kd_state.get("teacher") is not None and optimizer_step <= int(
                        max(1, max_steps * float(getattr(document.train.kd, "active_fraction", 1.0)))
                    ):
                        # Native KD (ui/E-008): teacher-KL is ADDED to the same
                        # backward; the student's primary forward is reused
                        # (its logits were already paid for by the CE path),
                        # and the teacher forward + KL are CHUNKED over the
                        # sequence so no second full-precision logit tensor
                        # ever exists at once.
                        loss = loss + _kd_teacher_kl(
                            str(kd_state["teacher"]), input_ids=ids,
                            student_logits=output.logits, device=device,
                            beta_kl=float(document.train.kd.beta_kl),
                            temperature=float(document.train.kd.temperature),
                            chunk=int(getattr(document.train.kd, "kl_chunk", 128)),
                            cache=kd_teacher_cache,
                        )
                if scaler is not None:
                    scaler.scale(loss / accum).backward()
                else:
                    (loss / accum).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(document.train.max_grad_norm),
                    foreach=False,
                )
                if micro_batch % accum == 0 or step >= max_steps * accum:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer_step += 1
                    micro_batch = 0
                    # Gradients are dead weight after the step. Freeing them here,
                    # rather than at the top of the next iteration, keeps the peak
                    # from being the training pass PLUS the checkpoint evaluation.
                    optimizer.zero_grad(set_to_none=True)
                if step % 32 == 0:
                    _release_device_cache()
            except (RuntimeError, MemoryError) as failure:
                # The Task 0 policy: an own-job OOM is BLOCKED, the last valid
                # checkpoint is preserved, and nothing is silently reduced.
                #
                # The message is classified by the monitoring module rather than
                # by matching an exception type here: only the GPU preflight is
                # allowed to name a device API, and this stage must not probe
                # or reference one. A non-memory failure is not swallowed: it
                # stops the run and is reported with its own text.
                from slm_train.resources.gpu_probe import attempt_accounting

                bus.emit(
                    "oom_accounting",
                    {"step": step, "site": "micro", **attempt_accounting(device, 1)},
                )
                bus.flush()
                decision = classify_health(message=str(failure))
                if not decision.oom:
                    bus.emit(
                        "health_failure",
                        {"step": step, "reason": str(failure)[:200]},
                    )
                    bus.flush()
                    return _fail(
                        context,
                        "training failed at step " + str(step) + ": " + str(failure),
                    )
                out_of_memory = True
                bus.emit("health_failure", {"step": step, "reason": decision.reason})
                stop = True
                break
            if not torch.isfinite(loss):
                # A non-finite loss stops the run at once. It is never skipped,
                # never clamped, and never allowed to continue into the next
                # step: the weights are already damaged at that point.
                bus.emit("health_failure", {"step": step, "reason": "non-finite loss"})
                bus.flush()
                return _fail(
                    context, "training produced a non-finite loss at step " + str(step)
                )
            last_loss = float(loss.detach())
            _loss_window.append(last_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            bus.emit("step_completed", {
                "step": step,
                "loss": last_loss,
                "lr": current_lr,
                "max_steps": max_steps * accum,
                "progress": step / (max_steps * accum),
            })
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", last_loss, step)
                tb_writer.add_scalar("train/lr", current_lr, step)
            from slm_train.resources.gpu_probe import attempt_accounting

            bus.emit("vram_probe", {"step": step, **attempt_accounting(device, 1)})
            bus.flush()
            final_step = optimizer_step
            last_epoch = epoch
            if final_step % int(document.train.checkpoint_every_steps) == 0:
                _publish_step(
                    stage_root, identity, model_manifest, tokenizer_manifest,
                    evaluator_manifest, model, optimizer, scheduler, final_step,
                    last_loss, batch_size, scaler,
                )
                # A checkpoint is only "best" against a measured number, so the
                # held-out loss is measured here, on the same blocks the
                # evaluation stage will use and on the same device.
                # The evaluation runs on the same blocks the evaluation stage
                # will use, on the same device, with the training graph and the
                # cached device blocks released first.
                _release_device_cache()
                val_loss = _held_out_loss(
                    model, _held_out, device, compute_dtype, autocast_device,
                    autocast_enabled,
                    masks=_held_masks,
                )
                _remember_score(stage_root, final_step, val_loss, last_loss)
                _prune_checkpoints(
                    stage_root,
                    keep_last=int(document.train.keep_last_checkpoints),
                    keep_best=int(document.train.keep_best_checkpoints),
                    protect=final_step,
                )
                bus.emit(
                    "checkpoint_completed",
                    {
                        "checkpoint": final_step,
                        "val_loss": val_loss,
                        "best": _best_checkpoint(stage_root),
                    },
                )
                if str(document.train.checkpoint_averaging) == "progressive":
                    _progressive_average(
                        model, stage_root,
                        int(document.train.checkpoint_averaging_n), device,
                        final_step,
                    )
            # Independent evaluation cadence: measure held-out loss at the declared
            # cadence, not just at checkpoint boundaries. This allows evaluation to
            # happen more frequently than checkpointing.
            if final_step % evaluation_cadence == 0:
                _release_device_cache()
                val_loss = _held_out_loss(
                    model, _held_out, device, compute_dtype, autocast_device,
                    autocast_enabled,
                    masks=_held_masks,
                )
                _remember_score(stage_root, final_step, val_loss, last_loss)
                bus.emit(
                    "evaluation_completed",
                    {"step": final_step, "kind": "cadence", "val_loss": val_loss},
                )
                bus.flush()
        # Monitor: probes fire at their OWN cadence (monitor_cadence), not the
        # evaluation cadence. The val_loss is the most recently measured one
        # (checkpoint or evaluation block), so it may lag by up to
        # evaluation_cadence steps - acceptable for trend monitoring.
        if _monitor.cfg.enabled and val_loss is not None and final_step % _monitor.cadence == 0:
            record = _monitor.record(final_step, val_loss, _loss_window)
            bus.emit("evaluation_completed", {
                "step": final_step, "kind": "monitor_probe",
                "val_loss": val_loss,
                "probe": {k: v for k, v in (record or {}).items() if k.startswith("probe_")},
            })
            bus.flush()
            if not _monitor.keep_training(record or {}):
                bus.emit("health_warning", {
                    "step": final_step,
                    "reason": "early stop: " + str(_monitor.stop_reason),
                })
                bus.flush()
                stop = True
        if stop:
            break
    bus.flush()

    # Stop the resource profiling thread
    resource_stop_event.set()
    resource_thread.join(timeout=5)

    # Monitor artifacts, always: the summary exists even for a run that
    # declared no monitor, so a reader never has to guess which regime ran.
    monitor_summary = _monitor.summarize()
    # The report is a declared stage output, so it must exist even when the
    # declared monitor never probed: an empty report is the honest record of
    # a run that measured nothing.
    if not _monitor.report_path.is_file():
        _monitor.report_path.write_text("", encoding="utf-8")
    bus.emit("evaluation_completed", {"step": int(final_step), "kind": "monitor_summary", **monitor_summary})
    bus.flush()

    if out_of_memory:
        return _blocked(
            context,
            "the run ran out of memory at step " + str(step)
            + "; the last valid checkpoint is preserved and batch, context, "
            "optimizer, precision, and command are unchanged",
        )
    if final_step == 0:
        return _fail(context, "no training step completed")

    # A checkpoint is never republished. When the cadence already published the
    # final step, that checkpoint IS the final one and is reused as it stands.
    if not is_complete(stage_root, final_step):
        state = _final_state(final_step, last_loss, batch_size, scheduler)
        payloads = _real_payloads(model, optimizer, scheduler, final_step)
        manifest = CheckpointManifest(
            schema_version=1,
            run_id=str(identity["run_id"]),
            config_hash=str(identity["config_hash"]),
            tokenizer_hash=str(identity["tokenizer_hash"]),
            model_hash=str(identity["model_hash"]),
            code_hash=str(identity["code_hash"]),
            global_step=final_step,
            epoch=float(last_epoch),
            files=(),
            complete=True,
        )
        publish_checkpoint(stage_root, final_step, state, payloads, manifest)
    # R2 adoption: a declared averaging run publishes the weight average of
    # the last N complete checkpoints as a verified artifact. Failure to
    # produce it is a labeled FAIL, never a silent skip: a run that declares
    # averaging and does not publish its average is not evidence.
    weight_average = None
    if str(document.train.checkpoint_averaging) != "off":
        from slm_train.training.averaging import publish_weight_average

        def _load_one(train_root_arg, step):
            import torch as _torch

            path = (Path(train_root_arg) / "checkpoints"
                    / ("step-" + str(step).zfill(8)) / "state" / "model.pt")
            return _torch.load(path, map_location="cpu", weights_only=True)

        weight_average = publish_weight_average(
            stage_root, identity, int(document.train.checkpoint_averaging_n),
            lambda pin, step: _load_one(pin, step),
        )
    bus.emit("stage_completed", {"stage": "train", "steps": final_step})
    bus.flush()

    elapsed_seconds = max(time.monotonic() - train_started, 1e-6)
    target = context.output_path("checkpoints/COMPLETE.json")
    atomic_write_bytes(
        target,
        canonical_json_bytes(
            {
                "status": "PASS",
                "run_id": str(identity["run_id"]),
                "final_step": final_step,
                "last_loss": last_loss,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
                "last_epoch": last_epoch,
                "resumed_from_step": resumed_from,
                "vram_fraction": ceiling,
                "best_checkpoint": _best_checkpoint(stage_root),
                "best_val_loss": _best_score(stage_root),
                "retained_checkpoints": _retained(stage_root),
                "train_blocks": len(train_blocks),
                "held_out_blocks": len(_held_out),
                "declared_precision": str(document.runtime.precision),
                "compute_dtype": str(compute_dtype),
                "train_tokens": int(measured_tokens),
                "train_seconds": float(elapsed_seconds),
                "peak_vram_bytes": int(peak_reserved_bytes(device)),
                "checkpoint": "step-" + str(final_step).zfill(8),
                "weight_average": weight_average,
            }
        ),
    )
    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    _detail = {"steps": final_step, "final_loss": last_loss}
    if _warm_info:
        _detail["warm_start"] = _warm_info
    return StageOutcome(label="PASS", detail=_detail)




def _progressive_average(model, stage_root, n: int, device, applied_at_step: int) -> None:
    """Average the last n complete checkpoint weights into the live model.

    R2 progressive mode (user-directed, research/ui_u2_full_train_plan.md):
    continues training FROM the running average of the most recent survival.
    load_state_dict copies values into the LIVE parameter tensors, so the
    optimizer moments stay aligned; a fresh re-init would be a different,
    unregistered policy. The applied step is recorded for evidence.
    """
    import torch

    from slm_train.training.averaging import average_states, average_steps

    steps = average_steps(stage_root, max(int(n), 2))
    if len(steps) < 2:
        return
    states = []
    for step in steps:
        path = (Path(stage_root) / "checkpoints"
                / ("step-" + str(step).zfill(8)) / "state" / "model.pt")
        if not path.is_file():
            return
        states.append(torch.load(path, map_location="cpu", weights_only=True))
    averaged = average_states(states)
    import io, json as _json

    model.load_state_dict(averaged)
    record = {
        "applied_at_step": int(applied_at_step),
        "averaged_steps": steps,
        "n": len(steps),
    }
    log_path = Path(stage_root) / "progressive_averaging.jsonl"
    log_path.open("a", encoding="utf-8").write(_json.dumps(record) + "\n")


def _best_score(stage_root) -> float | None:
    """The recorded held-out loss of the best checkpoint, or None if there is none.

    _scores returns a FLAT {step: loss} mapping with integer keys and takes the
    STAGE root, not the checkpoints directory. Reading it as a nested
    {"val_loss": {...}} document made every best score silently None.
    """
    scores = _scores(stage_root)
    if not scores:
        return None
    best = min(scores, key=lambda step: scores[step])
    return float(scores[best])


def _step_of(name: str) -> int:
    digits = "".join(character for character in str(name) if character.isdigit())
    return int(digits) if digits else 0


def _real_payloads(model, optimizer, scheduler, step: int, scaler=None) -> dict:
    """Serialize the real tensors a checkpoint must carry.

    Placeholder bytes would make a checkpoint that verifies while carrying
    nothing, and any later stage that trusted it would silently evaluate the
    initial weights instead of the trained ones. Every role gets real content.
    """
    import io

    import torch

    def dump(value) -> bytes:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()

    return {
        "model": dump({k: v.detach().cpu() for k, v in model.state_dict().items()}),
        "optimizer": dump(optimizer.state_dict()),
        "scheduler": dump(scheduler.state_dict()),
        "scaler": dump(scaler.state_dict() if scaler is not None else {"kind": "none", "reason": "fp32 master weights on CPU"}),
        "global_rng": dump(torch.get_rng_state()),
        "sampler": dump({"order_seed": int(step), "shard": 0}),
        "dataloader_generator": dump({"seed": int(step)}),
        # worker_rng is a TUPLE role: the checkpoint expects one payload per
        # worker, addressed worker_rng.0, worker_rng.1, ... not a single blob.
        "worker_rng.0": dump(torch.get_rng_state()),
        "early_stop": dump({"stopped": False, "reason": ""}),
        "evaluation": dump({"pending": True}),
        "lifecycle": dump({"stage": "train", "step": int(step)}),
    }


def _final_state(step: int, loss, batch_size: int = 1, scheduler=None):
    from slm_train.training.state import CheckpointState, TrainingCursor

    return CheckpointState(
        cursor=TrainingCursor(step * batch_size, 0, 0, 1),
        model="model.pt",
        optimizer="optimizer.pt",
        scheduler="scheduler.pt",
        scaler="scaler.pt",
        global_rng="global_rng.pt",
        sampler="sampler.pt",
        dataloader_generator="dataloader_generator.pt",
        worker_rng=("worker_rng_0.pt",),
        early_stop="early_stop.json",
        evaluation="evaluation.json",
        lifecycle="lifecycle.json",
    )


def _publish_step(
    root, identity, model_manifest, tokenizer_manifest, evaluator_manifest,
    model, optimizer, scheduler, step: int, loss, batch_size: int, scaler=None,
) -> None:
    """Publish one intermediate checkpoint through the real create-if-absent path."""
    from slm_train.contracts.checkpoint import CheckpointManifest
    from slm_train.training.checkpoint import is_complete, publish_checkpoint

    # A complete checkpoint is immutable: an already-published step is the
    # run own history and is republished as it stands, never rewritten. This
    # is what lets an auto-resumed run replay the cadence without a conflict.
    if is_complete(root, step):
        return

    state = _final_state(step, loss, batch_size, scheduler)
    payloads = _real_payloads(model, optimizer, scheduler, step, scaler)
    manifest = CheckpointManifest(
        schema_version=1,
        run_id=str(identity["run_id"]),
        config_hash=str(identity["config_hash"]),
        tokenizer_hash=str(identity["tokenizer_hash"]),
        model_hash=str(identity["model_hash"]),
        code_hash=str(identity["code_hash"]),
        global_step=step,
        epoch=1.0,
        files=(),
        complete=True,
    )
    publish_checkpoint(root, step, state, payloads, manifest)


# ----------------------------------------------------- internal_evaluation


def stage_internal_evaluation(context: StageContext) -> StageOutcome:
    """Score the final checkpoint and gate the declared internal suites."""
    import math

    import torch

    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.contracts.hash import canonical_json_bytes, canonical_sha256
    from slm_train.evaluation.generation import generation_details, generation_metrics
    from slm_train.evaluation.internal import validation_metrics
    from slm_train.evaluation.syntax import python_parser, syntax_metrics
    from slm_train.evaluation.reports import evaluate, write_report
    from slm_train.evaluation.suites import suite_for

    if context.already_complete():
        return StageOutcome(
            label="PASS", detail={"stage": "internal_evaluation", "reused": True}
        )

    document = context.config
    identity = context.read_published_json(
        "resolve_final_identity", "manifests/run_identity.json"
    )
    evaluator_manifest = context.read_published_json(
        "evaluator_validation", "manifests/evaluator.json"
    )
    tokenizer_manifest = context.read_published_json("tokenize", "manifests/tokenizer.json")
    train_record = context.read_published_json("train", "checkpoints/COMPLETE.json")
    tokenized_manifest = context.read_published_json(
        "dataset_validation", "manifests/tokenized_data.json"
    )

    if train_record.get("last_loss") is None:
        return _fail(context, "the train stage recorded no loss to evaluate")

    # A real forward pass over the held-out blocks, not a restatement of the
    # training loss. The validation metric must not be the training metric.
    from slm_train.models.causal_lm import build_causal_lm

    _unused_train, held_out, _held_maskother, held_masks = _split_blocks(
        context, tokenized_manifest, float(document.data.validation_fraction)
    )
    if not held_out:
        return _fail(context, "the run produced no held-out block to evaluate")

    torch.manual_seed(int(document.run.seed))
    model = build_causal_lm(
        model_config=document.model,
        capabilities=list(document.capabilities),
        data_context=int(document.packing.sequence_length),
        vocab_size=int(tokenizer_manifest["vocabulary_size"]),
        seed=int(document.run.seed),
    )
    # The trained weights MUST be loaded. Rebuilding the model here would score
    # the freshly initialised weights and report the same number as an untrained
    # run, which is a real measurement dressed as evidence.
    loaded = _load_trained_weights(
        context, model, str(train_record["checkpoint"]),
        weight_average=(train_record.get("weight_average")
                        if str(context.config.train.checkpoint_averaging) != "off"
                        else None)
    )
    if loaded is None:
        return _fail(
            context,
            "the selected checkpoint carries no loadable model state: "
            + str(train_record["checkpoint"]),
        )
    # Evaluation runs on the same device the run trained on, so the recorded
    # number measures the published weights on the declared device rather than a
    # CPU copy of them. The dtype split matches the train stage.
    device = str(document.runtime.device)
    eval_dtype = torch.float32 if device == "cpu" else torch.float16
    model = model.to(torch.float32).to(device)
    model.eval()
    losses = []
    with torch.no_grad():
        for block_index, block in enumerate(held_out):
            block_mask = held_masks[block_index] if isinstance(held_masks, list) else None
            ids = torch.tensor(
                [list(map(int, block))], dtype=torch.long
            ).to(device)
            block_labels = _labels_for_block(ids, block_mask)
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=eval_dtype,
                enabled=device != "cpu",
            ):
                losses.append(float(model(input_ids=ids, labels=block_labels).loss))
    mean_loss = sum(losses) / len(losses)
    if not math.isfinite(mean_loss):
        return _fail(context, "the held-out loss is not finite")

    # Every declared suite is measured, and only the declared ones. The previous
    # shape hardcoded validation_metrics() here, so declaring generation, syntax
    # or resources produced a guaranteed "missing required metric" FAIL: three
    # suites were advertised by the catalogue and unreachable by any run.
    declared_suites = set(document.evaluation.suites)
    values = {}
    if "validation" in declared_suites:
        values.update(validation_metrics(mean_loss, len(losses)))

    samples = ()
    needs_text = bool(declared_suites & {"generation", "syntax"})
    if needs_text:
        decoded = _decode_for_evaluation(context, document, model, device, eval_dtype)
        if isinstance(decoded, str):
            return _fail(context, decoded)
        samples = decoded

    if "generation" in declared_suites:
        try:
            values.update(generation_metrics(samples))
        except ValueError as error:
            return _fail(context, "generation smoke could not be scored: " + str(error))

    syntax_failures = ()
    if "syntax" in declared_suites:
        try:
            scored = syntax_metrics(samples, python_parser())
        except ValueError as error:
            return _fail(context, "syntax could not be scored: " + str(error))
        # "failures" is a diagnostic, not a metric: leaving it in the value map
        # would hand the gate a metric the syntax suite never declared.
        syntax_failures = tuple(scored.get("failures", ()))
        values["syntax_error_rate"] = scored["syntax_error_rate"]

    if "resources" in declared_suites:
        measured = _resource_metrics(train_record)
        if isinstance(measured, str):
            return _fail(context, measured)
        values.update(measured)

    if not values:
        return _fail(context, "the run declared no evaluable suite")
    expected_hash = canonical_sha256(evaluator_manifest)
    stage_root = context.stage_root()
    reports = []
    for name in document.evaluation.suites:
        suite = suite_for(name)
        # Each suite is handed EXACTLY the metrics it declares. Passing the
        # union of every suite's metrics is what made the strict completeness
        # check report "unexpected metrics" and fail a run that had measured
        # everything correctly.
        scoped = {
            metric.name: values[metric.name]
            for metric in suite.metrics
            if metric.name in values
        }
        missing = [metric.name for metric in suite.metrics if metric.name not in scoped]
        if missing:
            return _fail(
                context,
                "the " + str(name) + " suite declares metrics this run did not measure: "
                + str(missing),
            )
        report = evaluate(
            suite,
            dict(identity),
            scoped,
            expected_evaluator_hash=expected_hash,
            actual_evaluator_hash=expected_hash,
            # The decoded text is recorded verbatim, not summarised: a
            # generation gate worth trusting has to show what it generated.
            details=(
                generation_details(samples) if samples else {"held_out_blocks": len(losses)}
            ),
            # The declared thresholds are what the operator configured; the
            # catalogue only supplies the metric names and their directions.
            thresholds=dict(document.evaluation.thresholds),
        )
        reports.append((name, write_report(stage_root, name, report)))
        if report.status != "PASS":
            return _fail(
                context,
                "internal suite " + str(name) + " is " + str(report.status)
                + ": " + str(report.details.get("reason", "")),
            )

    target = context.output_path("evaluations/COMPLETE.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        target,
        canonical_json_bytes(
            {
                "schema_version": 1,
                "run_id": str(identity["run_id"]),
                "metrics": values,
                "suites": [name for name, _ in reports],
                "reports": [path for _, path in reports],
            }
        ),
    )
    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(
        label="PASS", detail={"metrics": values, "suites": len(reports)}
    )


def _load_trained_weights(context: StageContext, model, checkpoint_name: str,
                          weight_average=None):
    """Load the published checkpoint state into the model, or return None.

    When the run declared checkpoint averaging and the train stage published a
    weight average, the AVERAGED weights are the trained weights (R2 adoption):
    fall back to the named checkpoint only when no average is available.
    """
    import torch

    from slm_train.training.checkpoint import checkpoint_dir
    if weight_average:
        avg_path = (Path(context.run_root) / "train"
                    / str(weight_average["path"]))
        if avg_path.is_file():
            state = torch.load(avg_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.weight_average_used = True  # type: ignore[attr-defined]
            return model

    # The checkpoint lives under the TRAIN stage root, not under this stage.
    # Addressing the wrong root looks for a file that was never written and
    # would report a real run as unevaluable.
    train_root = Path(context.run_root) / "train"
    path = checkpoint_dir(train_root, int(checkpoint_name.split("-")[-1]))
    state_path = path / "state" / "model.pt"
    if not state_path.is_file():
        return None
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except (OSError, ValueError, RuntimeError):
        return None
    try:
        model.load_state_dict(state)
    except (RuntimeError, KeyError):
        return None
    return model


def _split_blocks(context: StageContext, tokenized: dict, fraction: float) -> tuple:
    """Partition the packed blocks into training and held-out sets.

    The split is a tail reservation of the declared validation_fraction, applied
    to the blocks in publication order so it is deterministic. The held-out set
    is EXCLUDED from training, which is the point: an evaluation block that the
    optimizer has already seen is not evidence of anything.

    Both stages call this one function. A stage that re-derived the split could
    quietly evaluate on blocks the trainer also trained on.
    """
    blocks = _load_blocks(context, tokenized)
    if not blocks:
        return [], [], [], []
    masks = _load_block_masks(context, tokenized)
    if isinstance(masks, list) and len(masks) != len(blocks):
        raise ValueError("mask block count disagrees with token block count: FAIL")
    cut = len(blocks) - max(1, int(len(blocks) * fraction))
    if masks is None:
        return blocks[:cut], blocks[cut:], None, None
    return blocks[:cut], blocks[cut:], masks[:cut], masks[cut:]


# ------------------------------------------------------------ external_gates


def stage_external_gates(context: StageContext) -> StageOutcome:
    """Bind the external gate reports, and never pass an absent one."""
    from slm_train.artifacts.atomic import atomic_write_bytes
    from slm_train.contracts.hash import canonical_json_bytes

    if context.already_complete():
        return StageOutcome(
            label="PASS", detail={"stage": "external_gates", "reused": True}
        )

    from slm_train.evaluation.external import run_external_gates

    # These gates re-derive the run's claims from the published bytes. The
    # previous implementation copied the INTERNAL suite reports and relabelled
    # them "external", so the stage could not fail: it graded the run using the
    # run's own report of itself.
    results = run_external_gates(context, context.config)
    bound = {result.name: result.as_dict() for result in results}

    if not results:
        return _blocked(
            context, "no external gate is registered, so nothing was verified"
        )

    failing = sorted(r.name for r in results if r.status == "FAIL")
    blocked = sorted(r.name for r in results if r.status == "BLOCKED")
    overall = "FAIL" if failing else ("BLOCKED" if blocked else "PASS")

    target = context.output_path("reports/COMPLETE.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        target,
        canonical_json_bytes(
            {"schema_version": 1, "status": overall, "gates": bound}
        ),
    )
    if overall != "PASS":
        return (
            _fail(context, "an external gate failed: " + str(failing))
            if failing
            else _blocked(context, "an external gate could not run: " + str(blocked))
        )
    try:
        context.write_stage_marker()
    except (OSError, ValueError) as error:
        return _fail(context, "could not bind the stage marker: " + str(error))
    return StageOutcome(label="PASS", detail={"gates": list(bound)})


# ------------------------------------------------------------- publish_final


def _decode_for_evaluation(context: StageContext, document, model, device, eval_dtype):
    """Decode from the trained weights, or return why it could not.

    The samples are recorded verbatim in the evaluation report, so a reader can
    see what the model actually produced instead of trusting a 1.0.
    """
    import torch
    from tokenizers import Tokenizer

    from slm_train.evaluation.generation import smoke_prompt
    from slm_train.models.decode import DecodeError, decode_samples

    # The SAME pinned local tokenizer file the tokenize stage used. Decoding
    # through anything else would measure the run against a vocabulary it was
    # never trained on.
    try:
        encoder = Tokenizer.from_file(
            str(Path(context.repo_root) / str(document.tokenizer.revision_or_files))
        )
    except Exception as error:
        return "the declared tokenizer could not be reopened: " + str(error)

    prompt = smoke_prompt()
    try:
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=eval_dtype,
            enabled=device != "cpu",
        ):
            return decode_samples(
                model, encoder, prompt,
                sample_count=int(document.evaluation.generation_samples),
                max_new_tokens=int(document.evaluation.generation_max_new_tokens),
                device=device,
                seed=int(document.run.seed),
                pad_id=int(document.tokenizer.special_ids["pad"]),
                temperature=float(document.evaluation.generation_temperature),
                top_k=int(document.evaluation.generation_top_k),
                top_p=float(document.evaluation.generation_top_p),
            )
    except DecodeError as error:
        return "decoding from the published checkpoint failed: " + str(error)


def _resource_metrics(train_record):
    """Throughput and peak memory, or why the run did not measure them."""
    from slm_train.evaluation.resources import resource_metrics

    try:
        return resource_metrics(
            tokens=train_record.get("train_tokens"),
            seconds=train_record.get("train_seconds"),
            peak_vram_bytes=train_record.get("peak_vram_bytes"),
        )
    except ValueError as error:
        return "the resources suite needs measurements the run did not take: " + str(error)


def stage_publish_final(context: StageContext) -> StageOutcome:
    """Publish the run, read it back, and bind the run marker last."""
    from slm_train.artifacts.publication import publish_final, write_run_complete
    from slm_train.contracts.hash import canonical_json_bytes, sha256_bytes, sha256_file

    if context.already_complete():
        return StageOutcome(
            label="PASS", detail={"stage": "publish_final", "reused": True}
        )

    identity = context.read_published_json(
        "resolve_final_identity", "manifests/run_identity.json"
    )
    train_record = context.read_published_json("train", "checkpoints/COMPLETE.json")
    tokenizer_manifest = context.read_published_json("tokenize", "manifests/tokenizer.json")
    publish_root = context.stage_root()

    model_payloads = {
        "model_config.json": canonical_json_bytes(context.config.model.model_dump(mode="json")),
        "run_identity.json": canonical_json_bytes(identity),
    }
    tokenizer_payloads = {
        "tokenizer_manifest.json": canonical_json_bytes(tokenizer_manifest),
    }
    # The identity hashes are recorded in the run identity manifest itself, not
    # smuggled into the publication payload map as if they were file paths.
    # publish_final certifies the staged files it actually wrote.
    required_hashes = {}

    try:
        publish_final(
            publish_root,
            run_id=str(identity["run_id"]),
            selected_checkpoint=str(train_record["checkpoint"]),
            model_payloads=model_payloads,
            tokenizer_payloads=tokenizer_payloads,
            required_hashes=required_hashes,
        )
    except (OSError, ValueError) as error:
        return _fail(context, "final publication failed: " + str(error))

    # COMPLETE.json is bound last and only through write_run_complete, so no
    # path exists that marks a run complete without the read-back above.
    # COMPLETE.json binds only what the final publication manifest certifies.
    # The manifest itself is not in that certified set, so binding it here would
    # be refused: the marker may not claim an artifact publication never recorded.
    payloads = {}
    # The marker re-reads every bound path from disk and compares bytes, so
    # the map carries the published bytes, not a digest of them.
    payloads.update(
        {
            "final/model/" + name: (publish_root / "final" / "model" / name).read_bytes()
            for name in model_payloads
        }
    )
    payloads.update(
        {
            "final/tokenizer/" + name: (
                publish_root / "final" / "tokenizer" / name
            ).read_bytes()
            for name in tokenizer_payloads
        }
    )
    try:
        write_run_complete(publish_root, payloads)
    except (OSError, ValueError) as error:
        return _fail(context, "the run marker could not be bound: " + str(error))

    # No separate stage marker is written here: write_run_complete already
    # wrote COMPLETE.json for this stage, and the graph declares that file as
    # this stage's marker. Writing a second one would overwrite the run marker
    # with a weaker one that no longer certifies the publication.
    return StageOutcome(
        label="PASS", detail={"run_id": str(identity["run_id"]), "files": len(payloads)}
    )


def _guarded(handler):
    """Report an unexpected filesystem failure as a FAIL, not as a crash.

    A handler that raises leaves the engine recording a bare exception string.
    That string is an implementation detail, while the label and reason are what
    an operator actually reads. Converting the failure here keeps every stage
    speaking the same validation vocabulary, and keeps the exception TYPE in the
    reason because the path is legitimate evidence, not a secret.
    """

    def wrapper(context: StageContext) -> StageOutcome:
        try:
            return handler(context)
        except (OSError, MemoryError) as error:
            return _fail(
                context,
                type(error).__name__
                + " while the stage was writing or reading its evidence: "
                + str(error),
            )

    wrapper.__name__ = getattr(handler, "__name__", "handler")
    wrapper.__doc__ = handler.__doc__
    return wrapper


HANDLERS = {
    "validate": stage_validate,
    "bootstrap": stage_bootstrap,
    "gpu_preflight": stage_gpu_preflight,
    "prepare": stage_prepare,
    "tokenize": stage_tokenize,
    "dataset_validation": stage_dataset_validation,
    "model_validation": stage_model_validation,
    "evaluator_validation": stage_evaluator_validation,
    "resolve_final_identity": stage_resolve_final_identity,
    "train": stage_train,
    "internal_evaluation": stage_internal_evaluation,
    "external_gates": stage_external_gates,
    "publish_final": stage_publish_final,
}

# Every handler is guarded, so a filesystem failure is reported as a FAIL in
# the shared vocabulary instead of escaping as a bare exception string.
HANDLERS = {name: _guarded(handler) for name, handler in HANDLERS.items()}


def handler_for(name: str):
    return HANDLERS.get(name)
