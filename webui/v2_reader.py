r"""Read-only introspection over pipeline_v2 run artifacts (V2 mode).

The same contract as webui/artifacts.py holds for V1: this module opens small
JSON files and JSONL log segments, and it never loads weights, never touches a
device, never allocates VRAM and never writes. It is therefore safe to poll
beside a live V2 run, which is what the Monitor tab does every 10 seconds.

Two rules make this module deliberately not a re-implementation of the V1
readers, and both come from defects the V1 to V2 work already paid for:

1. The label is the engine's, not this module's. status.json is written by the
   orchestration engine and is displayed verbatim. Nothing here infers a PASS
   from the presence of artifacts, and an unreadable or absent status is
   reported as FAIL for the same reason pipeline_v2/run.py status refuses: a
   run whose outcome cannot be read is not a pass.
2. A published marker is not the same claim as its target. COMPLETE.json is
   read as a marker plus the hashes it certifies, and this module never
   restates a metric the marker does not carry.

The read path is the published one. pipeline_v2/run.py status reads exactly
<run_root>/status.json, so this module reads the same file rather than
inferring status from the stage tree.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "runs" / "v2"

# The five validation labels, anchored in docs/agent/SCHEMA.md. Anything the
# engine writes outside this set is surfaced as-is and never coerced.
LABELS = ("PASS", "FAIL", "SKIPPED", "BLOCKED", "NEEDS_USER_DECISION")

STATUS_FILE = "status.json"
# Stage order is the R1 graph order, used only to sort the stage table so it
# reads in execution order. It is never used to decide what ran.
R1_STAGES = (
    "validate", "bootstrap", "gpu_preflight", "prepare", "tokenize",
    "dataset_validation", "model_validation", "evaluator_validation",
    "resolve_final_identity", "train", "internal_evaluation",
    "external_gates", "publish_final",
)

MAX_CURVE_POINTS = 400

# markdown inline-code tick, kept as a constant so the rendered paths read as
# code without the module source itself being littered with quote characters.
TICK = chr(96)
DASH = "\u2014"


def code(value) -> str:
    """Render a value as markdown inline code."""
    return TICK + str(value) + TICK


def load_json(path: Path):
    """Read a small JSON file; None when missing/corrupt (never raises)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------- discovery

def v2_runs() -> list[tuple[str, str]]:
    """(namespace, run_id) for every V2 run on disk, sorted.

    _attempts is V2's internal attempt space (V2Layout.attempts) and is not a
    run, so it is skipped rather than rendered as an empty run.
    """
    if not V2_ROOT.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for namespace in sorted(p for p in V2_ROOT.iterdir() if p.is_dir()):
        if namespace.name.startswith("_"):
            continue
        for run_dir in sorted(p for p in namespace.iterdir() if p.is_dir()):
            if not run_dir.name.startswith("_"):
                out.append((namespace.name, run_dir.name))
    return out


def run_choices() -> list[str]:
    """Display labels for the run dropdowns: namespace/run_id."""
    return [f"{ns}/{rid}" for ns, rid in v2_runs()]


def run_root(label: str) -> Path | None:
    """Resolve a namespace/run_id label to its run root, or None.

    The label arrives from a user-writable dropdown, so it is split once and
    re-checked against the enumerated pairs rather than joined onto a path.
    That keeps a crafted label from walking out of runs/v2.
    """
    if not label or "/" not in label:
        return None
    namespace, _, run_id = label.partition("/")
    for known_ns, known_rid in v2_runs():
        if known_ns == namespace and known_rid == run_id:
            return V2_ROOT / namespace / run_id
    return None


def footprint(path: Path) -> int:
    """Total bytes under a directory; 0 when it is not a directory."""
    p = Path(path)
    if not p.is_dir():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# -------------------------------------------------------------------- status

def run_label(run_root_path: Path) -> dict:
    """The engine-published status, displayed verbatim.

    A missing or unreadable status.json is FAIL with the reason spelled out.
    That is deliberately not "unknown" and not "pass": the engine is the only
    authority on the label, and if its label cannot be read then nobody can
    tell what happened.
    """
    published = load_json(Path(run_root_path) / STATUS_FILE)
    if not isinstance(published, dict):
        return {
            "status": "FAIL",
            "failed_stage": None,
            "completed_stages": [],
            "reason": "no readable " + STATUS_FILE,
            "readable": False,
        }
    status = published.get("status")
    return {
        "status": status if status in LABELS else str(status),
        "failed_stage": published.get("failed_stage"),
        "completed_stages": list(published.get("completed_stages") or []),
        "reason": str(published.get("reason") or ""),
        "readable": True,
    }


def stage_table(run_root_path: Path) -> list[list[str]]:
    """One row per stage: stage, state, marker path.

    A stage is completed when the engine listed it in completed_stages;
    otherwise it is reported as the engine left it. A stage directory with no
    COMPLETE.json is reported as such rather than assumed complete, because
    the marker is the proof and its absence is a real state.
    """
    root = Path(run_root_path)
    label = run_label(root)
    done = set(label["completed_stages"])
    names = [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    ordered = [s for s in R1_STAGES if s in names]
    ordered += sorted(n for n in names if n not in R1_STAGES)

    rows: list[list[str]] = []
    for stage in ordered:
        marker = root / stage / "COMPLETE.json"
        record = load_json(marker) if marker.is_file() else None
        if stage in done:
            state = "completed"
        elif record is not None:
            state = "marker written, not listed by the engine"
        else:
            state = "no marker"
        if isinstance(record, dict) and record.get("status") in LABELS:
            state += " (marker: " + str(record["status"]) + ")"
        rel = marker.relative_to(ROOT).as_posix() if marker.is_file() else "-"
        rows.append([stage, state, rel])
    return rows


# --------------------------------------------------------------------- logs

def _tail_lines(path: Path, n: int) -> list[str]:
    """The last n complete lines of a text file.

    Metrics segments are hundreds of KB and the Monitor tab re-reads them every
    10 seconds, so the head of the file is never parsed. A first partial line
    after the seek is dropped rather than parsed as a record.
    """
    if n <= 0:
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            window = 65536
            while True:
                start = max(0, size - window)
                handle.seek(start)
                chunk = handle.read()
                if start <= 0 or chunk.count(b"\n") > n:
                    break
                window *= 4
    except OSError:
        return []
    if not chunk:
        return []
    lines = chunk.decode("utf-8", "replace").splitlines()
    if start > 0:
        # the first line of the window is a fragment of a longer record
        lines = lines[1:]
    return lines[-n:]


def _records(root: Path, kind: str, limit: int) -> list[dict]:
    """Records from logs/<kind>/segment-*.jsonl, oldest first, bounded.

    Segments are walked newest-first so a bounded read still yields the most
    recent records. A file that is not a numbered segment is skipped: this is
    a read-only view and a stray file must not raise out of a poller.
    """
    kind_dir = Path(root) / "logs" / kind
    if not kind_dir.is_dir() or limit <= 0:
        return []
    out: list[dict] = []
    for segment in sorted(kind_dir.glob("segment-*.jsonl"), reverse=True):
        if len(out) >= limit:
            break
        batch: list[dict] = []
        for line in _tail_lines(segment, limit - len(out)):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                batch.append(record)
        # _tail_lines already returns file order, so batch is oldest-first and
        # needs no reversal. Segments are walked newest-first, so prepending
        # keeps the whole stream in ascending record order.
        out = batch + out
    return out[-limit:]


def loss_points(run_root_path: Path,
                limit: int = MAX_CURVE_POINTS) -> list[tuple[int, float]]:
    """(step, loss) from the train stage metrics stream, oldest first.

    Read from the published metrics segments rather than from the training
    state, because the segments are what the run actually wrote while it was
    live; scores.json only covers retained checkpoints and is read separately.
    """
    points: list[tuple[int, float]] = []
    for record in _records(Path(run_root_path) / "train", "metrics", limit * 2):
        if record.get("event") != "step_completed":
            continue
        payload = record.get("payload") or {}
        step, loss = payload.get("step"), payload.get("loss")
        if isinstance(step, int) and isinstance(loss, (int, float)):
            points.append((step, float(loss)))
    if len(points) > limit:
        stride = len(points) / float(limit)
        points = [points[int(i * stride)] for i in range(limit)]
    return points


def monitor_events(run_root_path: Path, limit: int = 200) -> list[dict]:
    """The train stage monitor stream, oldest first."""
    return _records(Path(run_root_path) / "train", "monitor", limit)


def last_monitor_line(run_root_path: Path) -> str:
    """The newest monitor event as one readable line, or empty when none."""
    events = monitor_events(run_root_path, limit=1)
    if not events:
        return ""
    record = events[-1]
    payload = record.get("payload") or {}
    stage = payload.get("stage", "?")
    kind = str(record.get("event", "?")).replace("_", " ")
    details = ", ".join(f"{k}={v}" for k, v in payload.items() if k != "stage")
    return f"{stage}: {kind}" + (f" ({details})" if details else "")


# --------------------------------------------------------------- checkpoints

def checkpoint_record(run_root_path: Path) -> dict | None:
    """The train stage selection record, or None when train never finished."""
    return load_json(
        Path(run_root_path) / "train" / "checkpoints" / "COMPLETE.json")


def scores(run_root_path: Path) -> dict:
    """train_loss/val_loss per retained checkpoint step."""
    data = load_json(
        Path(run_root_path) / "train" / "checkpoints" / "scores.json")
    return data if isinstance(data, dict) else {}


def checkpoint_manifest(run_root_path: Path, step: str) -> dict | None:
    """The manifest of one retained checkpoint, read-only."""
    if not step or "/" in step or "\\" in step or step in (".", ".."):
        return None
    return load_json(
        Path(run_root_path) / "train" / "checkpoints" / step / "manifest.json")


def retained_checkpoints(run_root_path: Path) -> list[str]:
    record = checkpoint_record(run_root_path) or {}
    retained = record.get("retained_checkpoints")
    if isinstance(retained, list):
        return [str(s) for s in retained]
    return []


# ------------------------------------------------------------- eval + gates

def eval_reports(run_root_path: Path) -> dict[str, dict]:
    """evaluator name -> the report the internal evaluation stage published."""
    base = Path(run_root_path) / "internal_evaluation" / "evaluations"
    out: dict[str, dict] = {}
    if not base.is_dir():
        return out
    for report in sorted(base.glob("*/report.json")):
        data = load_json(report)
        if isinstance(data, dict):
            out[report.parent.name] = data
    return out


def external_gates(run_root_path: Path) -> dict | None:
    return load_json(
        Path(run_root_path) / "external_gates" / "reports" / "COMPLETE.json")


def metric_rows(run_root_path: Path) -> list[list[str]]:
    """[evaluator, metric, value, gate, status] for every published metric."""
    rows: list[list[str]] = []
    for evaluator, report in sorted(eval_reports(run_root_path).items()):
        values = report.get("metric_values") or {}
        gate = report.get("gate_threshold")
        for metric, value in sorted(values.items()):
            rows.append([
                evaluator,
                metric,
                f"{value:.6g}" if isinstance(value, (int, float)) else str(value),
                "-" if gate is None else str(gate),
                str(report.get("status", "-")),
            ])
    return rows


# ------------------------------------------------------------------ identity

def resolved_config(run_root_path: Path) -> dict | None:
    """The config the engine resolved and recorded, or None.

    This is the authoritative version of the run configuration. It is
    preferred over any repo config file because it is what the run actually
    bound to, including inheritance and safe overrides.
    """
    return load_json(Path(run_root_path) / "validate" / "config" / "resolved.json")


def run_identity(run_root_path: Path) -> dict | None:
    return load_json(
        Path(run_root_path) / "resolve_final_identity"
        / "manifests" / "run_identity.json")


def preflight(run_root_path: Path) -> dict | None:
    """The device report the gpu_preflight stage recorded, or None."""
    return load_json(
        Path(run_root_path) / "gpu_preflight" / "attempt" / "preflight.json")


def model_identity(run_root_path: Path) -> dict | None:
    """The declared model identity, or None.

    This is the declared architecture, not a tensor inventory: the V2
    publish_final stage certifies model_config.json and run_identity.json and
    does not write weight files, so there is no safetensors header to read.
    Callers must say so rather than implying a weight-level view.
    """
    return load_json(
        Path(run_root_path) / "model_validation" / "manifests" / "model.json")


def published_model_config(run_root_path: Path) -> dict | None:
    return load_json(
        Path(run_root_path) / "publish_final" / "final" / "model"
        / "model_config.json")


def final_publication(run_root_path: Path) -> dict | None:
    return load_json(
        Path(run_root_path) / "publish_final" / "manifests"
        / "final_publication.json")


def servable_weights(run_root_path: Path) -> Path | None:
    """A servable HF checkpoint file for this run, or None.

    V2 keeps training state as state/model.pt inside a hash-chained checkpoint
    and publish_final certifies metadata only. A .pt training state is not a
    servable checkpoint, so it is never returned here: a caller that gets None
    must report that chat is unavailable rather than loading a training state
    and calling it a model.
    """
    root = Path(run_root_path)
    if not root.is_dir():
        return None
    for directory in (root / "publish_final" / "final" / "model",
                      root / "final" / "model"):
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.iterdir()):
            if candidate.is_file() and candidate.suffix in (".safetensors", ".bin"):
                return candidate
    return None


# ---------------------------------------------------------------- rendering

def _fmt(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def summary_row(label: str) -> list:
    """One Dashboard row: run, status, stages, checkpoint, best val, size."""
    root = run_root(label)
    if root is None:
        return [label, "-", "-", "-", "-", "-"]
    status = run_label(root)
    record = checkpoint_record(root) or {}
    val = scores(root).get("val_loss") or {}
    best_val = min(val.values()) if val else None
    return [
        label,
        status["status"],
        f"{len(status['completed_stages'])}/{len(R1_STAGES)}",
        str(record.get("checkpoint", "-")),
        _fmt(best_val) if best_val is not None else "-",
        f"{footprint(root) / 2**30:.1f} GiB",
    ]


def detail_markdown(label: str) -> str:
    """The per-run detail body, with every claim tied to a relative path."""
    root = run_root(label)
    if root is None:
        return "### " + label + "\n\n*Unknown or unreadable V2 run.*"

    status = run_label(root)
    rel = root.relative_to(ROOT).as_posix()
    lines = ["### " + label, "",
             "**" + status["status"] + "** " + DASH + " "
             + code(rel + "/" + STATUS_FILE), ""]

    if status["reason"]:
        lines.append("reason: " + status["reason"])
    if status["failed_stage"]:
        lines.append("failed stage: " + code(status["failed_stage"]))
    lines.append("stages completed: "
                 + str(len(status["completed_stages"])) + "/"
                 + str(len(R1_STAGES)))
    lines.append("")

    identity = run_identity(root)
    if identity:
        for key in ("run_id", "namespace", "config_hash", "code_hash", "seed"):
            if identity.get(key) is not None:
                lines.append("- " + key + ": " + code(_fmt(identity[key])))
        lines.append("")

    pre = preflight(root)
    if pre:
        lines.append("**preflight: " + str(pre.get("decision", "?"))
                     + "** " + DASH + " " + str(pre.get("reason", "")))
        observation = pre.get("observation") or {}
        if observation.get("name"):
            free = observation.get("free_memory_bytes")
            total = observation.get("total_memory_bytes")
            gib = ""
            if isinstance(free, int) and isinstance(total, int):
                gib = (", " + f"{free / 2**30:.1f}" + " GiB free of "
                       + f"{total / 2**30:.1f}")
            lines.append("- device: " + code(observation.get("name")) + " "
                         + code(observation.get("device", "")) + gib)
        lines.append("")

    rows = metric_rows(root)
    if rows:
        lines.append("| evaluator | metric | value | gate | status |")
        lines.append("|---|---|---|---|---|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    record = checkpoint_record(root)
    if record:
        lines.append(
            "checkpoint " + code(record.get("checkpoint"))
            + " (best " + code(record.get("best_checkpoint"))
            + ", step " + str(record.get("final_step")) + "/"
            + str(record.get("max_steps"))
            + ", last_loss " + _fmt(record.get("last_loss"))
            + ", " + str(record.get("declared_precision")) + ")")
        retained = retained_checkpoints(root)
        if retained:
            lines.append("retained: " + ", ".join(code(s) for s in retained))
        lines.append("")

    model = model_identity(root)
    if model:
        # num_parameters, precision and source_hash live inside
        # capability_metadata, not at the top level of the manifest.
        meta = model.get("capability_metadata") or {}
        lines.append(
            "model: " + str(meta.get("layers")) + "L h"
            + str(meta.get("hidden")) + " heads "
            + str(meta.get("heads")) + "/" + str(meta.get("kv_heads"))
            + " ffn " + str(meta.get("intermediate")) + " ctx "
            + str(meta.get("context")) + " vocab "
            + str(meta.get("vocab_size")) + " ("
            + f"{meta.get('num_parameters', 0):,}" + " params, "
            + str(meta.get("precision")) + ")")
        lines.append("")

    lines.append("| stage | state | marker |")
    lines.append("|---|---|---|")
    for row in stage_table(root):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def chat_blocked_markdown(label: str) -> str:
    """Why the Chat tab cannot serve a V2 run, with the evidence paths."""
    root = run_root(label)
    if root is None:
        return "**BLOCKED** " + DASH + " unknown V2 run " + label
    if servable_weights(root) is not None:
        return ("A servable checkpoint exists for " + label
                + "; the V1 chat loader is not wired to it yet.")
    rel = root.relative_to(ROOT).as_posix()
    return (
        "**BLOCKED** " + DASH + " " + label + " publishes no servable weights."
        "\n\nThe V2 publish_final stage certifies metadata only:\n\n"
        + "- " + code(rel + "/publish_final/final/model/model_config.json")
        + "\n"
        + "- " + code(rel + "/publish_final/final/model/run_identity.json")
        + "\n"
        + "- " + code(rel + "/publish_final/manifests/final_publication.json")
        + "\n\nTrained weights exist as "
        + code(rel + "/train/checkpoints/step-*/state/model.pt")
        + ", which is a hash-chained training state, not a servable HF "
        + "checkpoint. The Chat tab serves V1 checkpoints only; serving a V2 "
        + "run needs a pipeline decision to export a servable checkpoint, "
        + "not a UI change.")
