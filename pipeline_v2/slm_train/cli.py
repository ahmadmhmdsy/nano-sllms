from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

"""The eleven commands, and the one rule that shapes all of them.

No command has a side effect its name does not promise. plan writes nothing.
doctor touches no device. preflight probes a device only when --gpu is given,
and it is the only V2 path that does. status reads published evidence and never
recomputes it. export copies exactly the allowlist it was handed. migrate-v1
reads a V1 file as data and never launches anything.

Every command prints its outcome in the shared validation vocabulary, so a
caller never has to guess whether a quiet exit was a pass.
"""

__all__ = ["COMMANDS", "build_parser", "main"]

COMMANDS = (
    "init", "validate", "plan", "doctor", "hardware-state", "preflight", "run",
    "status", "status-runs", "explain", "diff", "export", "migrate-v1",
)

#: The only three overrides `run` will accept. Anything else is a config change
#: and must be made in the file, where it leaves a reviewable diff.
SAFE_OVERRIDES = ("seed", "device", "precision")

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_INVALID = 3


def _emit(document, stream=None) -> None:
    print(json.dumps(document, indent=2, sort_keys=True, default=str),
          file=stream or sys.stdout)


def _fail(label: str, reason: str, stream=None) -> int:
    _emit({"status": label, "reason": reason}, stream)
    return EXIT_REFUSED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline_v2")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, *arguments):
        node = sub.add_parser(name)
        for flag, kwargs in arguments:
            node.add_argument(flag, **kwargs)
        return node

    add("init",
        ("--template", {"default": "base/r1.yaml"}),
        ("--name", {"default": None}),
        ("--out", {"required": True}))
    add("validate", ("--config", {"required": True}))
    plan = add("plan", ("--config", {"required": True}))
    plan.add_argument("--dry-run", action="store_true")
    add("doctor")
    add("hardware-state")
    preflight = add("preflight", ("--config", {"default": None}), ("--gpu", {"action": "store_true"}))
    preflight.add_argument("--device", default="cuda:0")
    run = add("run",
        ("--config", {"required": True}),
        ("--seed", {"default": None}),
        ("--device", {"default": None}),
        ("--precision", {"default": None}))
    # Cooperative stop: instead of launching, place the STOP flag the train
    # loop honours at its next optimizer-step boundary. No process is killed;
    # the run ends with a preserved checkpoint.
    run.add_argument("--stop", action="store_true")
    add("status", ("--run", {"required": False}))
    add("status-runs", ("--namespace", {"default": None}))
    add("explain", ("--config", {"required": True}))
    add("diff", ("--left", {"required": True}), ("--right", {"required": True}))
    export = add("export",
                 ("--run", {"required": True}),
                 ("--out", {"required": True}),
                 ("--allowlist", {"nargs": "+", "default": None}))
    add("migrate-v1", ("--input", {"required": True}), ("--output", {"required": True}))
    return parser


# --- command handlers -----------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load(args):
    from slm_train.config.loader import load_config

    return load_config(Path(args.config), root=_repo_root())


def cmd_validate(args) -> int:
    try:
        config = _load(args)
    except Exception as error:
        return _fail("FAIL", str(error))
    from slm_train.features.registry import validate_capabilities

    try:
        validate_capabilities(config.release, list(config.capabilities))
    except Exception as error:
        return _fail("FAIL", str(error))
    _emit({"status": "PASS", "command": "validate",
           "run": config.run.name, "stage": config.run.stage,
           "capabilities": list(config.capabilities)})
    return EXIT_OK


def cmd_plan(args) -> int:
    try:
        config = _load(args)
    except Exception as error:
        return _fail("FAIL", str(error))
    from slm_train.orchestration.graph import StageGraph

    graph = StageGraph.r1()
    stages = [
        {
            "name": spec.name,
            "requires": list(spec.requires),
            "required_inputs": list(spec.required_inputs),
            "outputs": list(spec.outputs),
            "completion_marker": spec.completion_marker,
        }
        for spec in graph.specs()
    ]
    result = {"status": "PASS", "command": "plan", "stages": stages,
              "writes": False, "acquires_device": False}
    if getattr(args, "dry_run", False):
        result["resolved_config"] = {
            "run": {
                "name": config.run.name,
                "namespace": config.run.namespace,
                "stage": config.run.stage,
                "seed": config.run.seed,
            },
            "data": {
                "dataset_id": config.data.dataset_id,
                "sources": [
                    {
                        "source_id": s.source_id,
                        "kind": s.kind,
                        "path_or_repo": s.path_or_repo,
                        "role": s.role,
                    }
                    for s in config.data.sources
                ],
                "normalization": config.data.normalization,
                "filters": list(config.data.filters),
            },
            "model": {
                "architecture": config.model.architecture,
                "hidden": config.model.hidden,
                "layers": config.model.layers,
                "context": config.model.context,
                "precision": config.model.precision,
            },
            "train": {
                "learning_rate": config.train.learning_rate,
                "max_steps": config.train.max_steps,
                "batch_size": config.train.batch_size,
                "gradient_accumulation": config.train.gradient_accumulation,
                "optimizer": config.train.optimizer,
                "scheduler": config.train.scheduler,
                "warmup_steps": config.train.warmup_steps,
                "checkpoint_every_steps": config.train.checkpoint_every_steps,
                "checkpoint_policy": config.train.checkpoint_policy,
            },
            "runtime": {
                "device": config.runtime.device,
                "precision": config.runtime.precision,
            },
        }
        result["run_command"] = f"run --config {args.config}"
    _emit(result)
    return EXIT_OK


def cmd_doctor(args) -> int:
    from slm_train.resources.doctor import run_doctor

    result = run_doctor(_repo_root())
    _emit(result)
    return EXIT_OK if result.get("status") == "PASS" else EXIT_REFUSED


def cmd_hardware_state(args) -> int:
    """Report CPU, RAM, disk, GPU, and VRAM information. Read-only."""
    import os
    import shutil
    import platform

    root = _repo_root()
    report = {
        "status": "PASS",
        "command": "hardware-state",
        "cpu": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cores": os.cpu_count(),
        },
        "memory": {},
        "disk": {},
        "gpu": {},
    }

    # RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        report["memory"] = {
            "total_bytes": int(mem.total),
            "available_bytes": int(mem.available),
            "used_bytes": int(mem.used),
            "percent": float(mem.percent),
        }
    except ImportError:
        report["memory"] = {"detail": "psutil not available"}

    # Disk info
    try:
        usage = shutil.disk_usage(str(root))
        report["disk"] = {
            "total_bytes": int(usage.total),
            "free_bytes": int(usage.free),
            "used_bytes": int(usage.used),
        }
    except OSError as error:
        report["disk"] = {"detail": str(error)}

    # GPU info is intentionally NOT probed here: only preflight.py may
    # call probe_gpu, and the doctor command is a diagnostic, not a preflight.
    # A GPU probe from doctor would violate the release boundary.
    report["gpu"] = {"detail": "use 'preflight' command to probe GPU"}

    _emit(report)
    return EXIT_OK


def cmd_preflight(args) -> int:
    from slm_train.resources.preflight import run_cpu_preflight, run_gpu_preflight, write_preflight

    if not args.gpu:
        report = run_cpu_preflight(_repo_root())
        _emit(report)
        return EXIT_OK if report.get("status") == "PASS" else EXIT_REFUSED
    report = run_gpu_preflight(_repo_root(), args.device)
    write_preflight(_repo_root(), report)
    # The GPU report keys its verdict "decision" and the CPU report keys it
    # "status". A caller gates on the EXIT CODE, so the two shapes must be
    # reconciled here: otherwise a device that passed its probe exits REFUSED
    # and the launch gate cannot tell a healthy GPU from a blocked one.
    report["status"] = report.get("decision", "BLOCKED")
    _emit(report)
    return EXIT_OK if report.get("status") == "PASS" else EXIT_REFUSED


def cmd_explain(args) -> int:
    try:
        config = _load(args)
    except Exception as error:
        return _fail("FAIL", str(error))
    from slm_train.config.canonical import semantic_hash

    _emit({
        "status": "PASS",
        "command": "explain",
        "run": config.run.model_dump(mode="json"),
        "model": config.model.model_dump(mode="json"),
        "runtime": config.runtime.model_dump(mode="json"),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "features": config.features.model_dump(mode="json"),
        "capabilities": list(config.capabilities),
        "outputs": ["checkpoints/", "logs/", "evaluations/", "final/", "manifests/"],
        "semantic_hash": semantic_hash(config.model_dump(mode="json")),
        "resume_eligible": True,
        "resolved_overrides": {},
    })
    return EXIT_OK


def cmd_diff(args) -> int:
    from slm_train.config.loader import load_config
    from slm_train.config.canonical import semantic_hash

    try:
        left = load_config(Path(args.left), root=_repo_root())
        right = load_config(Path(args.right), root=_repo_root())
    except Exception as error:
        return _fail("FAIL", str(error))
    left_document = left.model_dump(mode="json")
    right_document = right.model_dump(mode="json")
    changed = sorted(
        key for key in set(left_document) | set(right_document)
        if left_document.get(key) != right_document.get(key)
    )
    identical = semantic_hash(left_document) == semantic_hash(right_document)
    _emit({
        "status": "PASS",
        "command": "diff",
        "identical": identical,
        "changed_sections": changed,
        "left_semantic_hash": semantic_hash(left_document),
        "right_semantic_hash": semantic_hash(right_document),
    })
    return EXIT_OK


def cmd_init(args) -> int:
    """Write a complete template, refusing to overwrite anything."""
    source = _repo_root() / "pipeline_v2" / "configs" / str(args.template)
    if not source.is_file():
        return _fail("FAIL", "no such template: " + str(source))
    destination = Path(args.out)
    if destination.exists():
        return _fail("FAIL", "refusing to overwrite: " + str(destination))
    text = source.read_text(encoding="utf-8")
    if args.name:
        text = text.replace("name: replace_me", "name: " + str(args.name))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    _emit({"status": "PASS", "command": "init", "out": str(destination),
           "template": str(args.template)})
    return EXIT_OK


def cmd_status(args) -> int:
    """Published status only. A run with no published status is NOT a pass."""
    from slm_train.orchestration.status import status_path

    published = status_path(Path(args.run))
    if not published.is_file():
        return _fail("BLOCKED", "no published status: " + str(published))
    try:
        document = json.loads(published.read_text(encoding="utf-8"))
    except ValueError as error:
        return _fail("BLOCKED", "published status is unreadable: " + str(error))
    _emit(document)
    label = document.get("status")
    return EXIT_OK if label == "PASS" else EXIT_REFUSED




def _summarise_run(run_root):
    """One run's published facts, read-only, or the honest unknown."""
    import json as _json
    from slm_train.orchestration.status import status_path

    published = status_path(Path(run_root))
    if not published.is_file():
        return {"run": Path(run_root).name, "status": "NEEDS_USER_DECISION",
                "reason": "no published status artifact"}
    try:
        document = _json.loads(published.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"run": Path(run_root).name, "status": "FAIL",
                "reason": "the status artifact is unreadable"}
    summary = {
        "run": Path(run_root).name,
        "status": document.get("status"),
        "failed_stage": document.get("failed_stage"),
        "completed_stages": document.get("completed_stages", []),
        "reason": document.get("reason", ""),
    }
    complete = Path(run_root) / "train" / "checkpoints" / "COMPLETE.json"
    if complete.is_file():
        try:
            record = _json.loads(complete.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        curve = Path(run_root) / "train" / "checkpoints" / "scores.json"
        scores = {}
        if curve.is_file():
            try:
                raw = _json.loads(curve.read_text(encoding="utf-8"))
                scores = {int(k): float(v) for k, v in (raw.get("val_loss") or {}).items()
                          if _is_number(v)}
            except (OSError, ValueError):
                scores = {}
        summary["loss_curve_points"] = sorted(scores)
        summary["best_val_loss"] = min(scores.values()) if scores else None
        summary["final_val_loss"] = record.get("last_loss")
        tokens = record.get("train_tokens")
        seconds = record.get("train_seconds")
        summary["tokens_per_second"] = (float(tokens) / float(seconds)
                                        if isinstance(tokens, (int, float))
                                        and isinstance(seconds, (int, float))
                                        and seconds > 0 else None)
    return summary


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def cmd_status_runs(args) -> int:
    """The latest published status per run under a namespace. Read-only."""
    root = _repo_root()
    base = root / "runs" / "v2"
    target = base if not getattr(args, "namespace", None) else base / str(args.namespace)
    if not target.is_dir():
        return _fail("BLOCKED", "no run namespace directory: " + str(target))
    runs = []
    for directory in sorted(target.iterdir()):
        if directory.is_dir():
            runs.append(_summarise_run(directory))
    _emit({"status": "PASS", "command": "status", "namespace": target.name,
           "runs": runs})


def _recorded_status(path) -> dict:
    """Read the status the engine wrote. Never infer the label from an exception.

    If the status file is missing or unreadable the run is FAIL, not a guess:
    an unreadable status means nobody can tell what happened, and a run whose
    outcome cannot be read is not a pass.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        status = document.get("status")
        if status in {"PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION"}:
            return {
                "status": status,
                "failed_stage": document.get("failed_stage"),
                "completed_stages": document.get("completed_stages", []),
                "reason": document.get("reason", ""),
            }
    except (OSError, ValueError):
        pass
    return {
        "status": "FAIL",
        "failed_stage": None,
        "completed_stages": [],
        "reason": "the run status artifact is missing or unreadable",
    }


def cmd_run(args) -> int:
    """Execute or exactly resume. Only three overrides are accepted."""
    from slm_train.config.overrides import apply_safe_overrides

    if getattr(args, "stop", False):
        # Cooperative stop: create the flag, never signal a process. The train
        # stage honours it at its next optimizer-step boundary; the checkpoint
        # from that step is preserved, and the run resumes from it exactly.
        try:
            config = _load(args)
        except Exception as error:
            return _fail("FAIL", str(error))
        root = _repo_root()
        # The train stage honours the flag at stage_root / "STOP" (handlers.py),
        # where stage_root = run_root / "train". Write it there, not at run level.
        flag = (root / "runs" / "v2" / str(config.run.namespace)
                / str(config.run.name) / "train" / "STOP")
        flag.parent.mkdir(parents=True, exist_ok=True)
        try:
            flag.write_text("cooperative stop\n", encoding="utf-8")
        except OSError as error:
            return _fail("FAIL", "could not write the stop flag: " + str(error))
        _emit({"status": "PASS", "command": "run-stop",
               "flag": flag.relative_to(root).as_posix(),
               "note": "the train stage will stop cleanly at its next step boundary"})
        return EXIT_OK

    try:
        config = _load(args)
    except Exception as error:
        return _fail("FAIL", str(error))
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.device is not None:
        overrides["device"] = args.device
    if args.precision is not None:
        overrides["precision"] = args.precision
    unknown = sorted(set(overrides) - set(SAFE_OVERRIDES))
    if unknown:
        return _fail("FAIL", "unsupported override: " + ", ".join(unknown))
    try:
        config = apply_safe_overrides(config, overrides)
    except Exception as error:
        return _fail("FAIL", str(error))
    from slm_train.orchestration.engine import Engine
    from slm_train.orchestration.graph import StageGraph
    from slm_train.orchestration.handlers import HANDLERS
    from slm_train.orchestration.status import status_path

    root = _repo_root()
    run_root = root / "runs" / "v2" / str(config.run.namespace) / str(config.run.name)
    engine = Engine(
        graph=StageGraph.r1(),
        handlers=HANDLERS,
        run_root=run_root,
        config=config,
        repo_root=root,
    )
    try:
        result = engine.run()
    except Exception as error:
        # The engine is the single authority on the label. It has already
        # written the terminal status; re-deciding it here from the exception
        # type is exactly the bug this branch exists to prevent, so the recorded
        # status is read back instead of being guessed.
        recorded = _recorded_status(status_path(run_root))
        _emit({
            "status": recorded["status"],
            "command": "run",
            "config": str(args.config),
            "overrides": overrides,
            "run_root": run_root.relative_to(root).as_posix(),
            "failed_stage": recorded["failed_stage"],
            "completed_stages": recorded["completed_stages"],
            "reason": recorded["reason"] or str(error),
            "status_artifact": status_path(run_root).relative_to(root).as_posix(),
        })
        return EXIT_REFUSED
    _emit({
        "status": result.status,
        "command": "run",
        "config": str(args.config),
        "overrides": overrides,
        "run_root": run_root.relative_to(root).as_posix(),
        "completed_stages": list(result.completed_stages),
        "status_artifact": status_path(run_root).relative_to(root).as_posix(),
    })
    return EXIT_OK


def cmd_export(args) -> int:
    """Copy exactly the allowlist. There is no default and no glob."""
    from slm_train.security.export import ExportError, ExportRequest, export_allowlist

    if not args.allowlist:
        return _fail("FAIL", "export needs an explicit --allowlist; there is no default")
    try:
        copied = export_allowlist(
            ExportRequest(Path(args.run), Path(args.out), args.allowlist)
        )
    except ExportError as error:
        return _fail("FAIL", str(error))
    _emit({"status": "PASS", "command": "export", "copied": copied})
    return EXIT_OK


def cmd_migrate_v1(args) -> int:
    """Translate a V1 config. Never launches training, never touches V1."""
    from slm_train.adapters.legacy_v1 import MigrationError, migrate_v1_config

    try:
        result = migrate_v1_config(args.input, args.output)
    except MigrationError as error:
        return _fail("FAIL", str(error))
    _emit({
        "status": "PASS",
        "command": "migrate-v1",
        "output": str(result.target),
        "source_sha256_before": result.source_sha256_before,
        "source_sha256_after": result.source_sha256_after,
        "source_unchanged": result.source_sha256_before == result.source_sha256_after,
        "migrated_fields": list(result.migrated_fields),
        "not_migrated": list(result.notes),
        "note": "migration is translation, not adoption; it will not launch training",
    })
    return EXIT_OK


HANDLERS = {
    "init": cmd_init,
    "validate": cmd_validate,
    "plan": cmd_plan,
    "doctor": cmd_doctor,
    "hardware-state": cmd_hardware_state,
    "preflight": cmd_preflight,
    "run": cmd_run,
    "status": cmd_status,
    "status-runs": cmd_status_runs,
    "explain": cmd_explain,
    "diff": cmd_diff,
    "export": cmd_export,
    "migrate-v1": cmd_migrate_v1,
}


def main(argv: list = None) -> int:
    args = build_parser().parse_args(argv)
    handler = HANDLERS.get(args.command)
    if handler is None:
        _emit({"status": "FAIL", "reason": "no such command: " + str(args.command)})
        return EXIT_INVALID
    return handler(args)
