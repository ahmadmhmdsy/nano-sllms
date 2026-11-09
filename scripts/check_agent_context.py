"""Read-only validation helpers for the repository's generated context files.

The linter intentionally has no process, GPU, or network side effects. Its
public helpers accept text or parsed records so tests can exercise the policy
without touching live state; lint_tree is the public read-only entry point
for a complete context tree, with optional policy and evidence-root coverage.
All filesystem inspection is lexical and no-follow: symlinks, reparse points,
hard-linked regular files, special files, and indeterminate metadata are
rejected before a file is opened.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

try:
    import yaml
except ImportError:  # pragma: no cover - reported by main when needed
    yaml = None


_WIN = os.name == "nt"
_YAML_MISSING = object()
_MAX_CONTEXT_FILE_BYTES = 1 << 20
_MAX_DIAGNOSTIC_CHARS = 240
_REDACTED_DIAGNOSTIC = "<redacted-diagnostic>"
_FORMAT_DIAGNOSTIC = re.compile(
    r"[\u00ad\u0600\u061c\u180e\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufff9\ufeff\U000E0000-\U000E007F]"
)

TASK_STATUSES = frozenset({"pending", "in_progress", "blocked", "done"})
EXPERIMENT_STATUSES = frozenset({"planned", "running", "closed", "blocked"})
EXPERIMENT_VERDICTS = frozenset(
    {"accepted", "rejected", "partial", "instrument_unknown", "none"}
)

FindingKey = tuple[str, int, str, str | None]

_CONTEXT_EXCEPTION_KINDS = frozenset(
    {"malformed_table_row", "broken_markdown_link"}
)
_MALFORMED_TABLE_ISSUE = re.compile(
    r"^malformed table row: (?P<path>[^:]+):(?P<line>[0-9]+)$"
)
_BROKEN_LINK_ISSUE = re.compile(
    r"^broken markdown link: (?P<path>[^:]+):(?P<line>[0-9]+): (?P<target>.+)$"
)

_NAMESPACED_ID = re.compile(r"^[^/\s]+/[^/\s]+(?:/[^/\s]+)*$")
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
_CONTROL_DIAGNOSTIC = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SECRET_DIAGNOSTIC_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
    ),
    re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(
        r"(?<![A-Za-z0-9_])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_])"
    ),
    re.compile(r"(?i:\bbearer\s*[:=]?\s*)[A-Za-z0-9._~+/-]{20,}=*"),
    re.compile(
        r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9]+[_-])*"
        r"(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|"
        r"client[_-]?secret|private[_-]?key|authorization)"
        r"(?![A-Za-z0-9_])[ \t]*[=:][ \t]*\S",
        re.IGNORECASE,
    ),
)
_BARE_PYTHON = re.compile(r"(?<![\w.\\/-])python(?:3(?:\.\d+)?|\.exe)?(?:\s|$)", re.I)
_BARE_PIP = re.compile(r"(?<![\w.\\/-])pip(?:3|\.exe)?(?:\s|$)", re.I)
_LIVE_CLAIM = (
    re.compile(r"\bPID\s*#?\s*\d+\b", re.I),
    re.compile(
        r"\b(?:job|process)\s+(?:id\s*)?[:#]?\s*[A-Za-z0-9_.-]+\s+"
        r"(?:is\s+)?(?:live|running|active)\b",
        re.I,
    ),
    re.compile(r"\blive\s+as\s+of\b", re.I),
    re.compile(r"\bcurrently\s+(?:running|active)\s+(?:job|process)\b", re.I),
    re.compile(r"\bcurrent\s+(?:machine|host)\s+(?:is|has)\b", re.I),
)
_ARCHIVED_AGENTS = re.compile(
    r"(?:\b(?:archive|snapshot|historical|backup)[^\n]*AGENTS\.md\b|"
    r"\barchived\s+AGENTS\.md\b)",
    re.I,
)

# These directories are deliberately not consumed by the context linter.  The
# preflight still inspects the directory entry itself, so a link named like an
# ignored directory is rejected, but it does not walk large generated stores.
_IGNORED_TREE_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        ".skillopt-sleep",
        "__pycache__",
        "data",
        "scratch",
        "checkpoint_backup",
        "runs",
        "outputs",
        "artifacts",
        "logs",
        "models",
        "raw",
        "wandb",
    }
)


class UnsafeContextError(ValueError):
    """Raised internally when a context path cannot be safely inspected."""


def _absolute_lexical(path: Any) -> Path:
    """Return an absolute spelling without resolving filesystem links."""
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError):
        raise UnsafeContextError(
            "unsafe context path: absolute path could not be normalized: "
            + _REDACTED_DIAGNOSTIC
        ) from None


def _display_path(path: Path, root: Path) -> str:
    """Return a bounded, redacted path spelling for diagnostics."""
    try:
        text = path.relative_to(root).as_posix()
    except ValueError:
        text = path.as_posix()
    return _safe_diagnostic(text)


def _diagnostic_text(value: str) -> str:
    """Return the bounded, redacted diagnostic form of untrusted link text."""
    return _safe_diagnostic(value)


def _safe_diagnostic(value: Any, *, limit: int = _MAX_DIAGNOSTIC_CHARS) -> str:
    """Return a bounded label without caller-controlled paths or secrets.

    Ordinary relative paths keep their useful spelling; secret-shaped,
    control/format-bearing, or overlong values become a fixed placeholder.
    """
    try:
        text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    except (TypeError, ValueError):
        return _REDACTED_DIAGNOSTIC
    if (
        _CONTROL_DIAGNOSTIC.search(text)
        or _FORMAT_DIAGNOSTIC.search(text)
        or any(pattern.search(text) for pattern in _SECRET_DIAGNOSTIC_PATTERNS)
    ):
        return _REDACTED_DIAGNOSTIC
    if len(text) > limit:
        return _REDACTED_DIAGNOSTIC
    return text


class _SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser whose diagnostics never echo raw argument values."""

    def _print_message(self, message: str, file: Any = None) -> None:
        stream = sys.stderr if file is None else file
        for line in str(message).splitlines():
            _safe_print(_safe_diagnostic(line), file=stream)

    def error(self, message: str) -> None:
        self._print_message(f"error: {_safe_diagnostic(message)}")
        raise SystemExit(2)


def _unsafe_path_issue(label: Any, reason: Any) -> str:
    """Return one controlled "unsafe" issue for a path or metadata failure."""
    return f"unsafe {_safe_diagnostic(label)}: {_safe_diagnostic(reason)}"


def _stream_safe_text(value: str, stream: Any) -> str:
    """Return text encodable by a CLI stream, escaping unsupported code points."""
    encoding = getattr(stream, "encoding", None)
    if isinstance(encoding, str):
        try:
            value.encode(encoding)
        except (LookupError, UnicodeError):
            return value.encode("ascii", "backslashreplace").decode("ascii")
    return value


def _safe_print(value: Any, *, file: Any = None) -> None:
    """Print bounded text without allowing stream encoding errors to escape."""
    stream = sys.stdout if file is None else file
    text = str(value)
    try:
        print(_stream_safe_text(text, stream), file=stream)
    except UnicodeEncodeError:
        # Some test doubles and redirected streams enforce encoding in
        # write rather than exposing an encoding attribute.
        try:
            print(text.encode("ascii", "backslashreplace").decode("ascii"), file=stream)
        except UnicodeEncodeError:
            pass


def _safe_evidence_diagnostic(value: str) -> str:
    """Return a bounded, non-secret representation of an evidence value."""
    if any(pattern.search(value) for pattern in _SECRET_DIAGNOSTIC_PATTERNS):
        return "<redacted-secret-evidence>"
    if _CONTROL_DIAGNOSTIC.search(value) or _FORMAT_DIAGNOSTIC.search(value):
        return "<redacted-control-evidence>"
    if len(value) > 256:
        return value[:240] + "<truncated-evidence>"
    return value


def _is_absolute_reference(value: str) -> bool:
    # Check the spelling before constructing a Path: on POSIX a Windows drive
    # path is otherwise treated as an ordinary relative filename.
    return (
        value.startswith(("/", "\\\\"))
        or bool(re.match(r"^[A-Za-z]:", value))
        or bool(Path(value).is_absolute())
    )


def _windows_file_attributes(path: Path) -> tuple[int, int]:
    """Return Windows file attributes and the API error, if any."""
    import ctypes  # noqa: PLC0415 - Windows-only, local import keeps POSIX clean

    kernel32 = ctypes.windll.kernel32
    get_attributes = kernel32.GetFileAttributesW
    get_attributes.argtypes = [ctypes.c_wchar_p]
    get_attributes.restype = ctypes.c_uint32
    attributes = int(get_attributes(str(path)))
    error = (
        int(kernel32.GetLastError())
        if attributes == 0xFFFFFFFF
        else 0
    )
    return attributes, error


def _lstat_component(path: Path) -> os.stat_result | None:
    """Inspect one component without following it."""
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except NotADirectoryError as exc:
        raise UnsafeContextError(
            "unsafe context path: a component is not a directory: "
            + _REDACTED_DIAGNOSTIC
        ) from None
    except ValueError as exc:
        raise UnsafeContextError(
            "unsafe context path: invalid component spelling: "
            + _REDACTED_DIAGNOSTIC
        ) from None
    except OSError as exc:
        raise UnsafeContextError(
            "unsafe context path: metadata could not be inspected: "
            + _REDACTED_DIAGNOSTIC
        ) from None


def _is_reparse_component(path: Path, metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    if not _WIN:
        return False
    try:
        attributes, error = _windows_file_attributes(path)
    except (AttributeError, OSError, ValueError):
        raise UnsafeContextError(
            "unsafe context path: reparse metadata could not be inspected: "
            + _REDACTED_DIAGNOSTIC
        ) from None
    if attributes == 0xFFFFFFFF:
        # The path is an internal absolute spelling; only the fixed reason and
        # numeric API error are safe to report.
        raise UnsafeContextError(
            "unsafe context path: reparse metadata could not be inspected: "
            + _REDACTED_DIAGNOSTIC
            + f" (Windows error: {int(error)})"
        )
    return bool(attributes & 0x400)


def _inspect_components(
    path: Path, *, allow_missing: bool
) -> tuple[str, os.stat_result | None, Path | None]:
    """Inspect an absolute path and every existing component without following."""
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[len(current.parts) :]
    for index, part in enumerate(parts):
        current = current / part
        metadata = _lstat_component(current)
        if metadata is None:
            if allow_missing:
                return "missing", None, current
            raise UnsafeContextError(
                f"missing context path: {_safe_diagnostic(current)}"
            )
        if _is_reparse_component(current, metadata):
            return "unsafe", metadata, current
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise UnsafeContextError(
                "unsafe context path: a component is not a directory: "
                f"{_safe_diagnostic(current)}"
            )
    return "exists", metadata, None


def _classify_existing(
    path: Path, *, allow_missing: bool = True
) -> tuple[str, os.stat_result | None]:
    state, metadata, _ = _inspect_components(path, allow_missing=allow_missing)
    if state == "missing":
        return "missing", None
    if state == "unsafe" or metadata is None:
        raise UnsafeContextError(f"unsafe context path: {_safe_diagnostic(path)}")
    if not stat.S_ISREG(metadata.st_mode) and not stat.S_ISDIR(metadata.st_mode):
        raise UnsafeContextError(
            f"unsafe context path: special file: {_safe_diagnostic(path)}"
        )
    if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
        raise UnsafeContextError(
            f"unsafe context path: hard-linked regular file: {_safe_diagnostic(path)}"
        )
    return "exists", metadata


def _preflight_tree(root: Path, label: str) -> None:
    """Reject unsafe entries before any context file is read."""
    absolute = _absolute_lexical(root)
    try:
        state, metadata, _ = _inspect_components(absolute, allow_missing=False)
        if state != "exists" or metadata is None or not stat.S_ISDIR(metadata.st_mode):
            raise UnsafeContextError(
                f"unsafe {_safe_diagnostic(label)} root: not a directory: "
                f"{_safe_diagnostic(absolute)}"
            )
    except UnsafeContextError:
        raise
    except OSError:
        # An unexpected filesystem failure is reported without its path or
        # exception text, which is untrusted diagnostic content.
        raise UnsafeContextError(
            "unsafe context path: metadata could not be inspected: "
            + _REDACTED_DIAGNOSTIC
        ) from None

    ignored = {name.casefold() for name in _IGNORED_TREE_DIRS}
    pending = [absolute]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            # The directory is an absolute internal path; only the caller label and
            # a fixed placeholder are safe to report.
            raise UnsafeContextError(
                f"unsafe {_safe_diagnostic(label)} directory: metadata could not be "
                f"inspected: {_REDACTED_DIAGNOSTIC}"
            ) from None
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = _lstat_component(path)
                if metadata is None:
                    raise UnsafeContextError(
                        f"unsafe {_safe_diagnostic(label)} path disappeared during "
                        f"preflight: {_safe_diagnostic(path)}"
                    )
                if _is_reparse_component(path, metadata):
                    raise UnsafeContextError(
                        f"unsafe {_safe_diagnostic(label)} path is a symlink or reparse "
                        f"point: {_safe_diagnostic(path)}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    if entry.name.casefold() not in ignored:
                        pending.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    if metadata.st_nlink > 1:
                        raise UnsafeContextError(
                            f"unsafe {_safe_diagnostic(label)} path is a hard-linked "
                            f"regular file: {_safe_diagnostic(path)}"
                        )
                else:
                    raise UnsafeContextError(
                        f"unsafe {_safe_diagnostic(label)} path is a special file: "
                        f"{_safe_diagnostic(path)}"
                    )
            except UnsafeContextError:
                raise
            except OSError:
                # The entry path is derived from an absolute internal tree.
                raise UnsafeContextError(
                    f"unsafe {_safe_diagnostic(label)} path metadata could not be "
                    f"inspected: {_REDACTED_DIAGNOSTIC}"
                ) from None


def _safe_existing_file(
    path: Path, label: str, issues: list[str], *, missing_ok: bool = True
) -> bool:
    """Return whether a regular file is safe to read, recording controlled issues."""
    try:
        state, metadata = _classify_existing(path, allow_missing=missing_ok)
    except UnsafeContextError as exc:
        issues.append(_unsafe_path_issue(label, exc))
        return False
    if state == "missing":
        return False
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        issues.append(_unsafe_path_issue(f"{_safe_diagnostic(label)}: not a regular file", path))
        return False
    return True


def _read_bounded_text(path: Path, label: str, issues: list[str]) -> str | None:
    """Read at most one bounded chunk from a context text file."""
    safe_label = _safe_diagnostic(label)
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_CONTEXT_FILE_BYTES + 1)
    except OSError:
        issues.append(f"cannot read {safe_label}")
        return None
    if len(data) > _MAX_CONTEXT_FILE_BYTES:
        issues.append(f"{safe_label} exceeds the size limit")
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        issues.append(f"cannot decode {safe_label}")
        return None


def _read_regular_text(path: Path, label: str, issues: list[str]) -> str | None:
    if not _safe_existing_file(path, label, issues, missing_ok=True):
        return None
    return _read_bounded_text(path, label, issues)


def _malformed_table_rows(text: str, source: Any, issues: list[str]) -> None:
    source_label = _safe_diagnostic(source)
    in_fence = False
    fence_marker = chr(96) * 3
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith(fence_marker):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.strip()
        if stripped.startswith("|") and "||" in stripped:
            issues.append(f"malformed table row: {source_label}:{line_number}")


def _validate_evidence(
    record: dict[str, Any], label: str, index: int, root: Path, issues: list[str]
) -> None:
    evidence = record.get("evidence", [])
    if evidence is None:
        return
    if not isinstance(evidence, list):
        issues.append(f"{label}[{index}]: evidence must be a list")
        return
    for evidence_index, value in enumerate(evidence):
        item_label = f"{label}[{index}]: evidence[{evidence_index}]"
        if not isinstance(value, str) or not value:
            issues.append(f"{item_label}: invalid evidence path")
            continue
        parts = value.split("/")
        invalid = (
            value != value.strip()
            or "\\" in value
            or "\x00" in value
            or value.startswith("/")
            or value.startswith("//")
            or any(not part or part in (".", "..") for part in parts)
            or any(":" in part for part in parts)
            or bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value))
            or bool(PurePosixPath(value).is_absolute())
            or (_WIN and any(part.endswith((".", " ")) for part in parts))
        )
        if invalid:
            issues.append(f"{item_label}: invalid local repo-relative evidence path")
            continue
        artifact = root.joinpath(*parts)
        try:
            state, _ = _classify_existing(artifact, allow_missing=True)
        except UnsafeContextError:
            issues.append(f"{item_label}: unsafe evidence path")
            continue
        if state == "missing":
            issues.append(
                f"{item_label}: missing evidence path: "
                f"{_safe_evidence_diagnostic(value)}"
            )


def check_records(records: dict, root: Path) -> list[str]:
    """Return stable, human-readable record and table validation issues."""
    issues: list[str] = []
    if not isinstance(records, dict):
        return ["records must be a mapping"]

    try:
        root = _absolute_lexical(root)
    except UnsafeContextError as exc:
        return [_unsafe_path_issue("evidence root", exc)]
    try:
        _classify_existing(root, allow_missing=False)
    except UnsafeContextError as exc:
        issues.append(_unsafe_path_issue("evidence root", exc))

    checks = (
        ("tasks", TASK_STATUSES, None),
        ("experiments", EXPERIMENT_STATUSES, EXPERIMENT_VERDICTS),
    )
    for label, statuses, verdicts in checks:
        entries = records.get(label, [])
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            issues.append(f"{label} must be a list")
            continue
        seen: set[str] = set()
        for index, record in enumerate(entries):
            if not isinstance(record, dict):
                issues.append(f"{label}[{index}]: record must be a mapping")
                continue
            identifier = record.get("id")
            if not isinstance(identifier, str) or not _NAMESPACED_ID.fullmatch(identifier):
                issues.append(f"{label}[{index}]: id must be a non-empty namespaced id")
                id_label = _safe_diagnostic(identifier)
            else:
                id_label = _safe_diagnostic(identifier)
                if identifier in seen:
                    kind = "task" if label == "tasks" else "experiment"
                    issues.append(f"duplicate {kind} id: {id_label}")
                seen.add(identifier)

            status = record.get("status")
            if status is not None and (
                not isinstance(status, str) or status not in statuses
            ):
                issues.append(f"invalid {label[:-1]} status: {id_label}")
            if verdicts is not None:
                verdict = record.get("verdict")
                if verdict is not None and (
                    not isinstance(verdict, str) or verdict not in verdicts
                ):
                    issues.append(f"invalid {label[:-1]} verdict: {id_label}")
            _validate_evidence(record, label, index, root, issues)

    tables = records.get("tables", {})
    if tables is None:
        tables = {}
    if not isinstance(tables, dict):
        issues.append("tables must be a mapping")
    else:
        for source, text in tables.items():
            if not isinstance(text, str):
                issues.append(f"table source must be text: {_safe_diagnostic(source)}")
                continue
            _malformed_table_rows(text, _safe_diagnostic(source), issues)
    return issues


def _iter_safe_markdown_files(root: Path, issues: list[str]) -> list[Path]:
    """Yield regular Markdown files below an already-preflighted rules tree."""
    result: list[Path] = []
    pending = [_absolute_lexical(root)]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError:
            issues.append(
                f"cannot enumerate policy directory: {_safe_diagnostic(directory)}"
            )
            continue
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = _lstat_component(path)
                if metadata is None or _is_reparse_component(path, metadata):
                    issues.append(_unsafe_path_issue("policy path", path))
                    continue
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode) and path.suffix.casefold() == ".md":
                    if metadata.st_nlink > 1:
                        issues.append(
                            _unsafe_path_issue("policy path: hard-linked regular file", path)
                        )
                    else:
                        result.append(path)
                elif not stat.S_ISREG(metadata.st_mode) and not stat.S_ISDIR(metadata.st_mode):
                    issues.append(_unsafe_path_issue("policy path: special file", path))
            except UnsafeContextError as exc:
                issues.append(_unsafe_path_issue("policy path", exc))
    return sorted(result, key=lambda item: item.as_posix())


def check_markdown_links(path: Path, root: Path) -> list[str]:
    """Check local Markdown links without following filesystem links."""
    try:
        root = _absolute_lexical(root)
        path = _absolute_lexical(path)
    except UnsafeContextError as exc:
        return [_unsafe_path_issue("markdown path", exc)]
    try:
        root_state, root_metadata = _classify_existing(root, allow_missing=False)
    except UnsafeContextError as exc:
        return [_unsafe_path_issue("markdown root", exc)]
    if root_state != "exists" or root_metadata is None or not stat.S_ISDIR(root_metadata.st_mode):
        return [_unsafe_path_issue("markdown root: not a directory", root)]

    source_issues: list[str] = []
    if not _safe_existing_file(path, "markdown file", source_issues, missing_ok=True):
        return source_issues or [f"missing markdown file: {_display_path(path, root)}"]
    text = _read_bounded_text(path, "markdown file", source_issues)
    if text is None:
        return source_issues

    def link_issue(kind: str, line_number: int, target: str) -> str:
        return (
            f"{kind}: {_display_path(path, root)}:{line_number}: "
            f"{_diagnostic_text(target)}"
        )

    issues: list[str] = []
    root_name = os.path.normcase(os.path.abspath(str(root)))
    for match in _MARKDOWN_LINK.finditer(text):
        raw_target = match.group(1).strip()
        line_number = text.count("\n", 0, match.start()) + 1
        if raw_target.startswith("<"):
            end = raw_target.find(">")
            target = raw_target[1:end] if end > 0 else raw_target[1:]
        else:
            target = raw_target.split(None, 1)[0]
        target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if not target:
            continue
        if re.match(r"^(?:https?|mailto):", target, re.I):
            continue
        if (
            "\\" in target
            or ":" in target
            or target.startswith("/")
            or bool(re.match(r"^[A-Za-z]:", target))
        ):
            issues.append(link_issue("unsafe markdown link", line_number, target))
            continue
        if _WIN and any(
            part not in (".", "..") and part.endswith((".", " "))
            for part in target.split("/")
        ):
            issues.append(link_issue("unsafe markdown link", line_number, target))
            continue
        try:
            target_absolute = _absolute_lexical(Path(os.path.join(str(path.parent), target)))
        except (OSError, TypeError, ValueError):
            issues.append(link_issue("unsafe markdown link", line_number, target))
            continue
        try:
            inside = os.path.commonpath(
                [root_name, os.path.normcase(os.path.abspath(str(target_absolute)))]
            ) == root_name
        except ValueError:
            inside = False
        if not inside:
            issues.append(link_issue("markdown link escapes root", line_number, target))
            continue
        try:
            state, _ = _classify_existing(target_absolute, allow_missing=True)
        except UnsafeContextError:
            issues.append(link_issue("unsafe markdown link", line_number, target))
            continue
        if state == "missing":
            issues.append(link_issue("broken markdown link", line_number, target))
    return issues


def check_static_policy(files: dict[str, str], root: Path) -> list[str]:
    """Reject live claims, bare package commands, and archived instructions."""
    issues: list[str] = []
    if not isinstance(files, dict):
        return ["policy files must be a mapping"]
    for source, text in sorted(files.items(), key=lambda item: _safe_diagnostic(item[0])):
        source_label = _safe_diagnostic(source)
        if not isinstance(text, str):
            issues.append(f"policy source must be text: {source_label}")
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in _LIVE_CLAIM):
                issues.append(f"live process claim: {source_label}:{line_number}")
            if _BARE_PYTHON.search(line):
                issues.append(f"bare Python command: {source_label}:{line_number}")
            if _BARE_PIP.search(line):
                issues.append(f"bare pip command: {source_label}:{line_number}")
            if _ARCHIVED_AGENTS.search(line):
                issues.append(f"archived AGENTS.md reference: {source_label}:{line_number}")
    return issues


def _load_yaml(path: Path, label: str, issues: list[str]) -> Any:
    try:
        state, metadata = _classify_existing(path, allow_missing=True)
    except UnsafeContextError as exc:
        issues.append(_unsafe_path_issue(label, exc))
        return None
    if state == "missing":
        return _YAML_MISSING
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        issues.append(_unsafe_path_issue(f"{_safe_diagnostic(label)}: not a regular file", path))
        return None
    if yaml is None:
        issues.append(f"cannot parse {label}: PyYAML is unavailable")
        return None
    text = _read_bounded_text(path, label, issues)
    if text is None:
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        # Do not echo parser snippets: YAML errors can include untrusted text.
        issues.append(f"cannot parse {label}")
    except RecursionError:
        # Deeply nested input can exhaust the parser stack; report a bounded
        # diagnostic without echoing any of the input.
        issues.append(f"cannot parse {label}: nesting exceeds recursion limit")
    return None


def _current_records(root: Path) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    records: dict[str, Any] = {
        "tasks": [],
        "experiments": [],
        "tables": {},
    }
    task_data = _load_yaml(root / "state" / "tasks.yaml", "state/tasks.yaml", issues)
    experiment_data = _load_yaml(
        root / "research" / "experiments.yaml", "research/experiments.yaml", issues
    )
    if isinstance(task_data, dict):
        records["tasks"] = task_data.get("tasks", [])
    elif task_data is not None:
        issues.append("state/tasks.yaml must contain a mapping")
    if isinstance(experiment_data, dict):
        records["experiments"] = experiment_data.get("experiments", [])
    elif experiment_data is _YAML_MISSING and isinstance(task_data, dict):
        # Some small context fixtures keep both record collections in the
        # task registry; use that explicit fallback only when the dedicated
        # experiment file is absent, without masking a present malformed file.
        records["experiments"] = task_data.get("experiments", [])
    elif experiment_data is not None:
        issues.append("research/experiments.yaml must contain a mapping")

    for relative in ("TASKS.md", "research/EXPERIMENTS.md", "README.md"):
        path = root / relative
        text = _read_regular_text(path, relative, issues)
        if text is not None:
            records["tables"][relative] = text
    return records, issues


def _context_exception_key(issue: str) -> FindingKey | None:
    """Return the exact (path, line, kind, target) key for a known finding."""
    table = _MALFORMED_TABLE_ISSUE.fullmatch(issue)
    if table is not None:
        return (
            table.group("path"),
            int(table.group("line")),
            "malformed_table_row",
            None,
        )
    link = _BROKEN_LINK_ISSUE.fullmatch(issue)
    if link is not None:
        return (
            link.group("path"),
            int(link.group("line")),
            "broken_markdown_link",
            link.group("target"),
        )
    return None


def _registry_relative_path(value: Any) -> bool:
    """Return True for a safe, repository-relative POSIX path spelling."""
    if not isinstance(value, str) or not value or len(value) > 240:
        return False
    if _is_absolute_reference(value) or "\\" in value or ":" in value:
        return False
    if value.strip() != value:
        return False
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    if _WIN and any(part.endswith((".", " ")) for part in parts):
        return False
    return _safe_diagnostic(value) == value


def _registry_path_issue(value: Any, label: str, index: int) -> str:
    """Return an invalid-path or unsafe-path diagnostic for a registry value."""
    if not isinstance(value, str) or not value:
        return f"context exception[{index}]: invalid {label}"
    parts = value.split("/")
    if (
        _is_absolute_reference(value)
        or "\\" in value
        or ":" in value
        or value.strip() != value
        or any(part in {"", ".", ".."} for part in parts)
        or (_WIN and any(part.endswith((".", " ")) for part in parts))
        or _safe_diagnostic(value) != value
    ):
        return f"context exception[{index}]: unsafe {label}"
    return f"context exception[{index}]: invalid {label}"


def _registry_text(
    value: Any, label: str, issues: list[str], index: int
) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 240
        or _safe_diagnostic(value) != value
    ):
        issues.append(f"context exception[{index}]: invalid {label}")
        return False
    return True


def load_context_exceptions(
    root: Path,
) -> tuple[dict[FindingKey, dict[str, Any]], list[str]]:
    """Load and fully validate the exact historical exception registry.

    Any malformed, missing, unsafe, or unreadable registry returns no
    exceptions at all so a bad registry can never accept a finding.
    """
    try:
        root = _absolute_lexical(root)
    except UnsafeContextError:
        return {}, ["context exception registry root is unsafe"]
    issues: list[str] = []
    data = _load_yaml(
        root / "state" / "context_exceptions.yaml",
        "state/context_exceptions.yaml",
        issues,
    )
    if data is _YAML_MISSING:
        return {}, [
            "missing required context exception registry: "
            "state/context_exceptions.yaml"
        ]
    if issues:
        return {}, issues
    if (
        not isinstance(data, dict)
        or isinstance(data.get("schema_version"), bool)
        or data.get("schema_version") != 1
    ):
        return {}, ["context exception registry must contain schema_version: 1"]
    if set(data) - {"schema_version", "exceptions"}:
        return {}, ["context exception registry has an unknown top-level field"]
    records = data.get("exceptions")
    if not isinstance(records, list):
        return {}, ["context exception registry exceptions must be a list"]

    parsed: dict[FindingKey, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    allowed_fields = {
        "id",
        "status",
        "path",
        "line",
        "kind",
        "target",
        "reason",
        "accepted_at",
        "decision_source",
        "evidence",
    }
    for index, record in enumerate(records):
        local: list[str] = []
        if not isinstance(record, dict):
            issues.append(f"context exception[{index}]: record must be a mapping")
            continue
        if set(record) - allowed_fields:
            local.append(f"context exception[{index}]: unknown field")
        identifier = record.get("id")
        if (
            not isinstance(identifier, str)
            or _NAMESPACED_ID.fullmatch(identifier) is None
        ):
            local.append(f"context exception[{index}]: invalid id")
        elif identifier in seen_ids:
            local.append(f"context exception[{index}]: duplicate id")
        else:
            seen_ids.add(identifier)
        if record.get("status") != "accepted_historical":
            local.append(f"context exception[{index}]: invalid status")
        finding_path = record.get("path")
        if not _registry_relative_path(finding_path):
            local.append(_registry_path_issue(finding_path, "path", index))
        line = record.get("line")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            local.append(f"context exception[{index}]: invalid line")
        kind = record.get("kind")
        if kind not in _CONTEXT_EXCEPTION_KINDS:
            local.append(f"context exception[{index}]: invalid kind")
        target = record.get("target")
        if kind == "broken_markdown_link":
            if not _registry_relative_path(target):
                local.append(_registry_path_issue(target, "target", index))
        elif "target" in record:
            local.append(
                f"context exception[{index}]: target is not allowed for "
                "malformed_table_row"
            )
        _registry_text(record.get("reason"), "reason", local, index)
        accepted_at = record.get("accepted_at")
        if not isinstance(accepted_at, str):
            local.append(f"context exception[{index}]: invalid accepted_at")
        else:
            try:
                if date.fromisoformat(accepted_at).isoformat() != accepted_at:
                    raise ValueError
            except ValueError:
                local.append(f"context exception[{index}]: invalid accepted_at")
        if record.get("decision_source") != "direct-user-instruction":
            local.append(f"context exception[{index}]: invalid decision_source")
        evidence = record.get("evidence")
        if not isinstance(evidence, list):
            local.append(f"context exception[{index}]: evidence must be a list")
        else:
            _validate_evidence(record, "context exception", index, root, local)
        if local:
            issues.extend(local)
            continue
        key: FindingKey = (
            finding_path,
            line,
            kind,
            target if kind == "broken_markdown_link" else None,
        )
        if key in parsed:
            issues.append(f"context exception[{index}]: duplicate finding key")
            continue
        parsed[key] = record
    if issues:
        return {}, issues
    return parsed, []


def apply_context_exceptions(
    issues: list[str],
    exceptions: dict[FindingKey, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Split issues into unaccepted issues and exact accepted historical ones."""
    remaining: list[str] = []
    accepted: list[str] = []
    for issue in issues:
        key = _context_exception_key(issue)
        if key is None or key not in exceptions:
            remaining.append(issue)
            continue
        path, line, kind, target = key
        suffix = f": {target}" if target is not None else ""
        accepted.append(
            f"ACCEPTED HISTORICAL: {path}:{line} ({kind}{suffix})"
        )
    return remaining, accepted


def lint_tree(
    root: Path,
    *,
    include_policy: bool = True,
    evidence_root: Path | None = None,
    accepted_exceptions: list[str] | None = None,
) -> list[str]:
    """Read-only lint of a repository/context tree; return all issues.

    The complete relevant input tree is preflighted before record or policy
    reads. Large ignored stores are not traversed, but links in files consumed
    by the linter are always rejected. Policy mode is the default: AGENTS.md,
    CLAUDE.md, SOUL.md, and rules/*.md are required and their content is
    checked for live PID/job claims, bare python/pip commands, and archived
    AGENTS.md references; pass include_policy=False for record-only use.
    """
    try:
        return _lint_tree(
            root,
            include_policy=include_policy,
            evidence_root=evidence_root,
            accepted_exceptions=accepted_exceptions,
        )
    except Exception:  # noqa: BLE001 - no raw diagnostic may escape the API
        return ["context validation failed unexpectedly: " + _REDACTED_DIAGNOSTIC]


def _lint_tree(
    root: Path,
    *,
    include_policy: bool,
    evidence_root: Path | None,
    accepted_exceptions: list[str] | None = None,
) -> list[str]:
    try:
        root = _absolute_lexical(root)
        evidence = (
            _absolute_lexical(evidence_root) if evidence_root is not None else root
        )
    except UnsafeContextError as exc:
        return [_unsafe_path_issue("context root", exc)]
    issues: list[str] = []
    checked: set[Path] = set()
    for label, path in (("context", root), ("evidence", evidence)):
        if path in checked:
            continue
        checked.add(path)
        try:
            _preflight_tree(path, label)
        except UnsafeContextError as exc:
            issues.append(_safe_diagnostic(exc))
    if issues:
        return issues

    records, pre_issues = _current_records(root)
    issues.extend(pre_issues)
    issues.extend(check_records(records, evidence))

    for relative in ("README.md", "TASKS.md", "research/EXPERIMENTS.md"):
        path = root / relative
        text_issues: list[str] = []
        if _safe_existing_file(path, relative, text_issues, missing_ok=True):
            issues.extend(check_markdown_links(path, root))
        else:
            issues.extend(text_issues)

    if include_policy:
        policy_files: dict[str, str] = {}
        for relative in ("AGENTS.md", "CLAUDE.md", "SOUL.md"):
            path = root / relative
            text = _read_regular_text(path, relative, issues)
            if text is None and not any(
                issue.startswith(f"unsafe {relative}:") for issue in issues
            ):
                issues.append(f"missing required policy file: {relative}")
            elif text is not None:
                policy_files[relative] = text
        rules_dir = root / "rules"
        try:
            rules_state, rules_metadata = _classify_existing(
                rules_dir, allow_missing=True
            )
        except UnsafeContextError as exc:
            issues.append(_unsafe_path_issue("policy path", exc))
        else:
            if rules_state == "exists" and rules_metadata is not None and stat.S_ISDIR(
                rules_metadata.st_mode
            ):
                for path in _iter_safe_markdown_files(rules_dir, issues):
                    relative = _safe_diagnostic(path.relative_to(root).as_posix())
                    text = _read_regular_text(path, relative, issues)
                    if text is not None:
                        policy_files[relative] = text
        if not any(
            name.startswith("rules/") and name.endswith(".md") for name in policy_files
        ):
            issues.append("missing required policy file: rules/*.md")
        issues.extend(check_static_policy(policy_files, root))
        exceptions, exception_issues = load_context_exceptions(root)
        issues.extend(exception_issues)
        if not exception_issues:
            issues, accepted = apply_context_exceptions(issues, exceptions)
            if accepted_exceptions is not None:
                accepted_exceptions.extend(accepted)
    return issues


def main(argv: list[str] | None = None) -> int:
    """Print all discovered issues and return zero only for a valid context."""
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--max-issues",
        type=int,
        default=200,
        help="maximum number of issues to print (default: 200)",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help=(
            "record-only mode: skip the required policy files "
            "(AGENTS.md, CLAUDE.md, SOUL.md, rules/*.md) and their checks"
        ),
    )
    args = parser.parse_args(argv)
    if args.max_issues < 1:
        _safe_print("ISSUE: --max-issues must be positive")
        return 2
    # Do not resolve the user-supplied root: resolution would follow a link
    # before lint_tree gets a chance to reject it.
    try:
        root = _absolute_lexical(args.root.expanduser())
    except UnsafeContextError as exc:
        _safe_print(f"ISSUE: {_safe_diagnostic(exc)}")
        return 1
    try:
        accepted: list[str] = []
        issues = lint_tree(
            root,
            include_policy=not args.no_policy,
            accepted_exceptions=accepted,
        )
    except Exception:  # noqa: BLE001 - no raw diagnostic may escape the CLI
        _safe_print("ISSUE: context validation failed unexpectedly: " + _REDACTED_DIAGNOSTIC)
        return 1

    # Accepted records are reported before ordinary issues so the exact
    # historical exceptions stay visible; the exit status depends only on
    # unaccepted issues.
    combined = [("accepted", entry) for entry in accepted]
    combined.extend(("issue", issue) for issue in issues)
    for kind, entry in combined[: args.max_issues]:
        _safe_print(entry if kind == "accepted" else f"ISSUE: {entry}")
    if len(combined) > args.max_issues:
        _safe_print(
            f"ISSUE: additional issues omitted: {len(combined) - args.max_issues}"
        )
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
