"""Deterministic renderer for the repository's current-state snapshot.

Reads the canonical registries (state/tasks.yaml, state/decisions.yaml,
research/experiments.yaml, state/environment.json, state/jobs.json) and
renders CURRENT_STATE.md plus state/current.json byte-for-byte
deterministically from those inputs:

- no wall-clock time, hostname, or process inspection goes into the
  output -- every fact comes from a registry file or Git itself;
- missing or malformed registries render as UNKNOWN instead of raising,
  so a fresh clone produces a stable snapshot too;
- --check never writes; --write updates only CURRENT_STATE.md and
  state/current.json.

The renderer must never inspect or terminate processes: the active job
is read from state/jobs.json only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - reported by main when needed
    yaml = None


UNKNOWN = "UNKNOWN"
MODES = ("write", "check")

# (registry label, relative path, format). Labels double as section names
# and keys in state/current.json.
REGISTRY_FILES: tuple[tuple[str, str, str], ...] = (
    ("tasks", "state/tasks.yaml", "yaml"),
    ("experiments", "research/experiments.yaml", "yaml"),
    ("decisions", "state/decisions.yaml", "yaml"),
    ("environment", "state/environment.json", "json"),
    ("jobs", "state/jobs.json", "json"),
)

OUTPUT_MD = "CURRENT_STATE.md"
OUTPUT_JSON = "state/current.json"
GENERATED_OUTPUTS = frozenset((OUTPUT_MD, OUTPUT_JSON))


@dataclass(frozen=True)
class RenderResult:
    """Outcome of one renderer run: the markdown text and pass/fail."""

    text: str
    ok: bool


def _load_yaml(path: Path) -> Any:
    if yaml is None:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def load_registries(root: Path) -> dict:
    """Load tasks, experiments, decisions, environment, and jobs.

    Missing or malformed registries load as None so the renderer can
    emit a deterministic UNKNOWN snapshot instead of crashing.
    """

    root = Path(root)
    registries: dict[str, Any] = {}
    for label, rel, kind in REGISTRY_FILES:
        path = root / rel
        if kind == "yaml":
            registries[label] = _load_yaml(path)
        else:
            registries[label] = _load_json(path)
    return registries


def _git(args: list[str], root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return completed.stdout.strip()


def collect_git_metadata(root: Path) -> dict:
    """Read branch, commit, and dirty-state counts without changing Git."""

    root = Path(root)
    if not (root / ".git").exists():
        return {"branch": UNKNOWN, "commit": UNKNOWN, "dirty_count": UNKNOWN}
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    commit = _git(["rev-parse", "HEAD"], root)
    porcelain = _git(
        ["status", "--porcelain=v1", "--untracked-files=all"], root
    )
    if porcelain == UNKNOWN:
        dirty_count: Any = UNKNOWN
    else:
        # The renderer writes these two files itself. Excluding them keeps
        # write -> check byte-stable while the snapshot is not committed yet.
        dirty_count = 0
        for line in porcelain.splitlines():
            if not line.strip():
                continue
            record = line[3:] if len(line) >= 3 else line
            path_tokens = {
                token.strip().strip('"').replace("\\", "/")
                for token in record.split(" -> ")
            }
            if path_tokens.isdisjoint(GENERATED_OUTPUTS):
                dirty_count += 1
    return {"branch": branch, "commit": commit, "dirty_count": dirty_count}


def _entries(value: Any, key: str) -> list:
    """Normalise a registry value into a list of record mappings."""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        inner = value.get(key, [])
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []


def _fmt(value: Any) -> str:
    """Format one value for markdown; None and unknowns become UNKNOWN."""

    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return ",".join(_fmt(item) for item in value) if value else "-"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _jsonable(value: Any) -> Any:
    """Recursively convert parsed YAML values to JSON-safe structures."""

    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _active_job(jobs_value: Any) -> str:
    """Resolve the active job from state/jobs.json only (no process peeking).

    More than one record claiming status 'active' is a conflict and
    renders as UNKNOWN -- ownership cannot be verified from the registry.
    A missing registry is also UNKNOWN (unverified).
    """

    jobs = _entries(jobs_value, "jobs")
    active = [
        str(record.get("id", UNKNOWN))
        for record in jobs
        if record.get("status") == "active"
    ]
    if len(active) > 1:
        return (
            f"{UNKNOWN} (conflicting active job state: "
            f"{len(active)} jobs claim active)"
        )
    if len(active) == 1:
        return active[0]
    return UNKNOWN


def _record_sort_key(record: dict) -> tuple[int, str]:
    identifier = record.get("id")
    if isinstance(identifier, str):
        return (0, identifier)
    return (1, "")


def _record_lines(entries: list) -> list[str]:
    lines = []
    for record in sorted(entries, key=_record_sort_key):
        fields = " | ".join(f"{key}={_fmt(record[key])}" for key in sorted(record))
        lines.append(f"- {fields}")
    return lines


def _render_markdown(git_meta: dict, active_job: str, registries: dict) -> str:
    lines = [
        "# CURRENT_STATE",
        "",
        "<!-- generated by scripts/render_current_state.py "
        "-- do not edit by hand -->",
        "",
        "## Git",
        "",
        f"- Branch: {git_meta['branch']}",
        f"- Commit: {git_meta['commit']}",
        f"- Dirty files: {git_meta['dirty_count']}",
        "",
        "## Active job",
        "",
        f"- {active_job}",
        "",
    ]
    for label, rel, _kind in REGISTRY_FILES:
        lines.append(f"## {label.capitalize()}")
        lines.append("")
        value = registries[label]
        if value is None:
            lines.append(f"- {UNKNOWN} ({rel} missing)")
        else:
            entries = _entries(value, label)
            if entries:
                lines.extend(_record_lines(entries))
            else:
                lines.append("- (none recorded)")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_json(git_meta: dict, active_job: str, registries: dict) -> str:
    # The machine-readable field stays the bare UNKNOWN sentinel; the
    # human-readable conflict detail lives in the markdown only.
    payload_active_job = UNKNOWN if active_job.startswith(UNKNOWN) else active_job
    payload = {
        "schema_version": 1,
        "git": _jsonable(git_meta),
        "active_job": payload_active_job,
        "registries": {
            label: _jsonable(registries[label])
            for label, _rel, _kind in REGISTRY_FILES
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def render_state(fixture_root: Path, mode: str = "write") -> RenderResult:
    """Render deterministic CURRENT_STATE.md and state/current.json.

    mode 'write' updates only those two paths; mode 'check' writes
    nothing and reports ok=True only when both files already match a
    fresh deterministic render.
    """

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got: {mode!r}")
    fixture_root = Path(fixture_root)
    registries = load_registries(fixture_root)
    git_meta = collect_git_metadata(fixture_root)
    active_job = _active_job(registries["jobs"])
    md_text = _render_markdown(git_meta, active_job, registries)
    json_text = _render_json(git_meta, active_job, registries)

    md_path = fixture_root / OUTPUT_MD
    json_path = fixture_root / OUTPUT_JSON

    if mode == "check":
        ok = (
            md_path.is_file()
            and md_path.read_bytes() == md_text.encode("utf-8")
            and json_path.is_file()
            and json_path.read_bytes() == json_text.encode("utf-8")
        )
        return RenderResult(text=md_text, ok=ok)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8", newline="\n")
    json_path.write_text(json_text, encoding="utf-8", newline="\n")
    return RenderResult(text=md_text, ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render CURRENT_STATE.md and state/current.json from the "
            "canonical registries. --check verifies without writing; "
            "--write updates only those two files."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write",
        dest="mode",
        action="store_const",
        const="write",
        help="regenerate CURRENT_STATE.md and state/current.json",
    )
    group.add_argument(
        "--check",
        dest="mode",
        action="store_const",
        const="check",
        help="verify the snapshot matches a fresh render (never writes)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root (default: the repo containing this script)",
    )
    args = parser.parse_args(argv)
    mode = args.mode or "check"  # safe default: never write unless asked
    root = args.root or Path(__file__).resolve().parents[1]

    result = render_state(fixture_root=root, mode=mode)
    if result.ok:
        print(f"render: PASS ({mode} mode, root={root})")
        return 0
    print(f"render: FAIL ({mode} mode, root={root}): snapshot is missing or stale")
    return 1


if __name__ == "__main__":
    sys.exit(main())
