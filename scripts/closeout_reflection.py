"""Deterministic, offline task-closeout reflections.

The default CLI mode is a dry-run: it validates a JSON input and prints a
reflection draft without writing any file. Repository output requires an
explicit --write plus --output (never overwriting an existing file without
--replace), and lesson staging requires an explicit --stage-lesson plus
--write.

This tool deliberately cannot call a model, launch jobs, inspect secrets, or
edit policy or persona files (SOUL.md, AGENTS.md, CLAUDE.md, rules/). The CLI
refuses an --output path that names SOUL.md, AGENTS.md, CLAUDE.md, or anything
under rules/. It also refuses alternate-stream and multi-link paths, validates
existing repository-relative evidence, and rejects an --output/--state-path
pair that resolves to one canonical destination before writing either file.
Metadata and reparse inspection fails closed; invalid UTF-8 and excessive state
nesting are preflighted before destinations are opened. Filesystem errors
during the sequential write phase are reported: either write can be partial,
and a later failure can leave earlier output or state because this path-based
implementation is not transactional. The remaining validation/use gap is a
documented path-based TOCTOU limitation. Repository-contained lesson staging is
restricted to the canonical state/lessons_pending.json path; an explicitly
external lesson-state path is allowed only for isolated validation and is never
treated as live repository state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

MAX_STRING_CHARS = 4000
MAX_LIST_ITEMS = 100
MAX_INPUT_BYTES = 1 << 20

POLICY_FILENAMES = frozenset(("soul.md", "agents.md", "claude.md"))
_WIN = sys.platform == "win32"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3

REQUIRED_STRING_FIELDS = (
    "requested_outcome",
    "actual_outcome",
    "reusable_lesson",
    "rule_candidate",
    "confidence_limitations",
)

OPTIONAL_STRING_FIELDS = (
    "scope",
    "source_reflection",
)

REQUIRED_LIST_FIELDS = (
    "changed_files",
    "validation_performed",
    "failures_near_misses",
    "root_causes",
    "fixes",
    "evidence",
    "open_questions",
)

_MANDATORY_LIST_FIELDS = frozenset(("changed_files", "evidence"))

SECRET_PATTERNS = (
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws temporary access key id", re.compile(r"\bASIA[0-9A-Z]{16}\b")),
    ("openai token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("gitlab token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{35,}\b")),
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{32,}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("private key block", re.compile("-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("github fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    (
        "authorization bearer token",
        re.compile(
            r"\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{12,}",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit bearer token",
        re.compile(
            r"\bbearer(?:\s+|[:=]\s*)[A-Za-z0-9._~+/=-]{12,}",
            re.IGNORECASE,
        ),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"),
    ),
    (
        "credential assignment",
        re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|"
            r"client[_-]?secret|private[_-]?key|aws[_-]?secret[_-]?access[_-]?key|"
            r"database[_-]?password|db[_-]?password|jwt)"
            r"(?![A-Za-z0-9])[\"']?[ \t]*[=:][ \t]*[\"']?\S",
            re.IGNORECASE,
        ),
    ),
)

_CONTROL_CHARACTERS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
_DIAGNOSTIC_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_FORMAT_CHARACTERS = re.compile(
    r"[\u00ad\u0600\u061c\u180e\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufff9\ufeff\U000e0000-\U000e007f]"
)
# Exception text is not a trusted diagnostic source.  Absolute, UNC, URI,
# and drive-qualified path fragments are treated as untrusted even when they
# contain no control characters or obvious token prefix.  Requested values
# are still rendered separately through _safe_diagnostic(), which preserves
# bounded ordinary paths when they are known inputs rather than exception
# text.
_UNSAFE_EXCEPTION_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\|/(?=[^ \t\r\n]))"
)
MAX_DIAGNOSTIC_CHARS = 240

DEFAULT_SCOPE = "core"
DEFAULT_STATE_PATH = Path("state") / "lessons_pending.json"
REPOSITORY_ROOT = Path(os.path.abspath(__file__)).parent.parent
_REPOSITORY_OUTPUT_PREFIX = ("docs", "reflections")
_REPOSITORY_STATE_PATH = ("state", "lessons_pending.json")
_EVIDENCE_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

EMPTY_LIST_NOTE = "(none recorded)"

SCHEMA_VERSION = 1
_LESSON_STATUSES = frozenset(
    ("observed", "candidate", "validated", "promoted", "rejected")
)
_REQUIRED_LESSON_FIELDS = (
    "id",
    "scope",
    "status",
    "lesson",
    "source_reflection",
    "evidence",
    "promoted_to",
)


class ValidationError(ValueError):
    """Raised when the reflection payload is missing or malformed."""


class SecretDetectedError(ValueError):
    """Raised when a payload value looks like a likely secret."""


class ProtectedPathError(ValueError):
    """Raised when an output path targets a protected policy/persona file."""


def _ensure_utf8_text(text: str, label: str) -> None:
    """Reject text that cannot be encoded before any destination is opened."""
    try:
        text.encode("utf-8")
    except UnicodeError as exc:
        raise ValidationError(
            f"{_safe_diagnostic(label)} contains text that cannot be encoded as UTF-8"
        ) from None


def _ensure_safe_text(text: str, label: str) -> None:
    """Reject control and bidi/format characters in persisted text."""
    _ensure_utf8_text(text, label)
    match = _CONTROL_CHARACTERS.search(text) or _FORMAT_CHARACTERS.search(text)
    if match is not None:
        codepoint = ord(match.group())
        raise ValidationError(
            f"{_safe_diagnostic(label)} contains unsupported control character U+{codepoint:04X}"
        )


def _reject_json_constant(value: str):
    """Reject non-standard JSON numeric constants before they become state."""
    raise ValueError(
        f"non-finite JSON constant is not allowed: {_safe_diagnostic(value)}"
    )


def _parse_finite_float(value: str):
    """Parse JSON floats while rejecting overflow to infinity."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not allowed")
    return parsed


def _loads_strict_json(raw: str, label: str):
    """Load JSON without accepting non-standard or overflowing numbers."""
    try:
        return json.loads(
            raw,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except RecursionError as exc:
        raise ValidationError(
            f"{_safe_diagnostic(label)} nesting is too deep to parse safely"
        ) from None
    except ValueError as exc:
        message = _safe_exception_detail(exc)
        raise ValidationError(
            f"{_safe_diagnostic(label)} is not valid JSON: {message}"
        ) from None


def _reject_nonfinite(value, label: str) -> None:
    """Reject non-finite floats anywhere in an already parsed JSON value."""
    pending = [(value, label)]
    while pending:
        current, current_label = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            raise ValidationError(
                f"{_safe_diagnostic(current_label)} contains a non-finite number"
            )
        if isinstance(current, dict):
            pending.extend((item, f"{current_label}.{key}") for key, item in current.items())
        elif isinstance(current, list):
            pending.extend(
                (item, f"{current_label}[{index}]")
                for index, item in enumerate(current)
            )


def _serialize_state(
    state: dict,
    state_path: Path,
    *,
    indent: int | None = None,
) -> str:
    """Serialize state with controlled depth, type, and UTF-8 errors."""
    options = {"ensure_ascii": False, "allow_nan": False}
    if indent is not None:
        options["indent"] = indent
    try:
        serialized = json.dumps(state, **options)
    except RecursionError as exc:
        raise ValidationError(
            f"state file nesting depth exceeds the safe serialization limit: {_safe_diagnostic(state_path)}"
        ) from None
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"state file cannot be serialized as JSON: {_safe_diagnostic(state_path)}"
        ) from None
    _ensure_safe_text(serialized, f"state file {_safe_diagnostic(state_path)}")
    return serialized


def _ensure_state_serializable(state: dict, state_path: Path) -> None:
    """Reject parsed state that cannot be written as UTF-8 JSON."""
    _serialize_state(state, state_path)


def _path_exists(path: Path) -> bool:
    """Classify existence without treating metadata errors as safe absence.

    A missing path may be a new output; an indeterminate stat result is a
    controlled protection error.
    """
    try:
        os.stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except ValueError as exc:
        raise ValidationError(
            f"invalid destination path (embedded NUL or invalid value): {_safe_diagnostic(path)}"
        ) from None
    except OSError as exc:
        raise ProtectedPathError(
            "refusing to proceed because path existence could not be safely "
            f"inspected: {_safe_diagnostic(path)} ({_safe_exception_detail(exc)})"
        ) from None
    return True


def _validate_string(payload, field, *, allow_empty=False):
    value = payload.get(field)
    if value is None:
        raise ValidationError(f"missing required field: {_safe_diagnostic(field)}")
    if not isinstance(value, str):
        raise ValidationError(f"field must be a string: {_safe_diagnostic(field)}")
    if not value.strip():
        if allow_empty:
            return ""
        raise ValidationError(f"field must not be empty: {_safe_diagnostic(field)}")
    if len(value) > MAX_STRING_CHARS:
        raise ValidationError(
            f"field is longer than {MAX_STRING_CHARS} characters: {_safe_diagnostic(field)}"
        )
    return value


def _validate_list(payload, field, *, allow_empty=False):
    value = payload.get(field)
    if value is None:
        raise ValidationError(f"missing required field: {_safe_diagnostic(field)}")
    if not isinstance(value, list):
        raise ValidationError(f"field must be a list of strings: {_safe_diagnostic(field)}")
    if not allow_empty and not value:
        raise ValidationError(f"field must contain at least one item: {_safe_diagnostic(field)}")
    if len(value) > MAX_LIST_ITEMS:
        raise ValidationError(
            f"field has more than {MAX_LIST_ITEMS} items: {_safe_diagnostic(field)}"
        )
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValidationError(
                f"field must contain non-empty strings only: {_safe_diagnostic(field)}"
            )
    return list(value)


def _scan_for_secret(field, text):
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            # The matched text itself is never included in the error message.
            raise SecretDetectedError(
                f"likely secret detected ({name}) in field: {_safe_diagnostic(field)}"
            )


def _safe_diagnostic(value, *, limit: int = MAX_DIAGNOSTIC_CHARS) -> str:
    """Return a bounded diagnostic label without caller-controlled secrets."""
    try:
        text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
        text.encode("utf-8")
    except Exception:
        return "<redacted diagnostic>"
    if (
        _DIAGNOSTIC_CONTROL_CHARACTERS.search(text)
        or _FORMAT_CHARACTERS.search(text)
        or any(pattern.search(text) for _, pattern in SECRET_PATTERNS)
    ):
        return "<redacted diagnostic>"
    try:
        limit = int(limit)
    except (TypeError, ValueError, OverflowError):
        limit = MAX_DIAGNOSTIC_CHARS
    if limit <= 0:
        return ""
    if len(text) > limit:
        marker = "…"
        # A one-character budget cannot show both content and the marker;
        # preserve the first safe character instead of replacing it entirely.
        if limit == 1:
            return text[:1]
        if limit < len(marker):
            return marker[:limit]
        keep_tail = min(48, limit // 4)
        keep_head = max(0, limit - keep_tail - len(marker))
        tail = text[-keep_tail:] if keep_tail else ""
        return f"{text[:keep_head]}{marker}{tail}"
    return text


def _safe_exception_detail(exc: BaseException) -> str:
    """Keep useful non-filesystem errors while bounding and redacting them."""
    if isinstance(exc, OSError):
        return "filesystem operation failed"
    try:
        detail = str(exc)
        detail.encode("utf-8")
    except Exception:
        return "filesystem operation failed"
    if (
        _DIAGNOSTIC_CONTROL_CHARACTERS.search(detail)
        or _FORMAT_CHARACTERS.search(detail)
        or any(pattern.search(detail) for _, pattern in SECRET_PATTERNS)
        or len(detail) > MAX_DIAGNOSTIC_CHARS
    ):
        return "filesystem operation failed"
    return detail


def _safe_cli_exception_detail(
    exc: BaseException, *, known_paths: tuple[object, ...] = ()
) -> str:
    """Return a CLI-safe detail while retaining known path context.

    Internal path errors are allowed to name a requested input/output/state
    path, but exception text supplied by a lower-level or patched component is
    not trusted.  If an absolute/UNC/URI path appears and does not match one
    of the paths supplied by this CLI invocation, fail closed with a fixed
    filesystem diagnostic.  Known paths are replaced with a bounded marker.
    """
    detail = _safe_exception_detail(exc)
    if not _UNSAFE_EXCEPTION_PATH.search(detail):
        return detail
    known = []
    for value in known_paths:
        try:
            text = os.fspath(value)
        except Exception:
            continue
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8")
            except UnicodeError:
                continue
        if not isinstance(text, str) or not text:
            continue
        start = 0
        while True:
            index = detail.find(text, start)
            if index < 0:
                break
            end = index + len(text)
            if end == len(detail) or detail[end] in " \t\r\n,;)]}":
                known.append(text)
                break
            start = end
    if not known:
        return "filesystem operation failed"
    for text in sorted(set(known), key=len, reverse=True):
        detail = detail.replace(text, "<redacted diagnostic>")
    if _UNSAFE_EXCEPTION_PATH.search(detail):
        return "filesystem operation failed"
    if len(detail) > MAX_DIAGNOSTIC_CHARS:
        detail = _safe_diagnostic(detail, limit=MAX_DIAGNOSTIC_CHARS)
    return detail


def _safe_api_exception_detail(
    exc: BaseException, *, requested_path: object
) -> str:
    """Keep a safe API diagnostic and separately label the requested path."""
    raw_detail = _safe_exception_detail(exc)
    detail = _safe_cli_exception_detail(exc, known_paths=(requested_path,))
    if detail != "filesystem operation failed" and not _UNSAFE_EXCEPTION_PATH.search(
        raw_detail
    ):
        return detail

    suffix = f" ({detail})"
    budget = MAX_DIAGNOSTIC_CHARS - len(suffix)
    if budget <= 0:
        return "filesystem operation failed"
    path_label = _safe_diagnostic(requested_path, limit=budget)
    if not path_label or path_label in detail:
        return detail
    return f"{path_label}{suffix}"


def _write_cli_text(message: str, *, stream=None, end: str = "\n") -> None:
    """Write CLI text without allowing the output encoding to leak a traceback."""
    if stream is None:
        stream = sys.stdout
    try:
        encoding = getattr(stream, "encoding", None)
        if encoding:
            message.encode(encoding)
        stream.write(message + end)
    except (LookupError, UnicodeError):
        encoding = getattr(stream, "encoding", None) or "ascii"
        try:
            fallback = message.encode(encoding, errors="backslashreplace").decode(encoding)
            stream.write(fallback + end)
        except (LookupError, UnicodeError):
            stream.write("<redacted diagnostic>" + end)


def _validate_persisted_strings(value, label: str) -> None:
    """Fail closed on unsafe text anywhere in a persisted JSON value tree."""
    pending = [(value, label)]
    while pending:
        current, current_label = pending.pop()
        if isinstance(current, str):
            _ensure_safe_text(current, current_label)
            _scan_for_secret(current_label, current)
        elif isinstance(current, dict):
            for key, item in current.items():
                if isinstance(key, str):
                    _ensure_safe_text(key, f"{current_label} key")
                    _scan_for_secret(f"{current_label} key", key)
                pending.append((item, current_label))
        elif isinstance(current, list):
            pending.extend((item, current_label) for item in current)


def _lstat_no_follow(path: Path, label: str, *, allow_missing: bool = False):
    """Return lstat metadata, or a controlled validation error.

    Persisted evidence may refer to an artifact that has since disappeared. In
    that compatibility mode a missing final/ancestor is represented by None so
    callers can stop walking that branch; every other metadata failure remains
    fail-closed.
    """
    try:
        return os.lstat(path)
    except FileNotFoundError as exc:
        if allow_missing:
            return None
        raise ValidationError(f"{_safe_diagnostic(label)} does not exist: {_safe_diagnostic(path)}") from None
    except NotADirectoryError as exc:
        raise ValidationError(f"{_safe_diagnostic(label)} does not exist: {_safe_diagnostic(path)}") from None
    except ValueError as exc:
        raise ValidationError(f"{_safe_diagnostic(label)} has an invalid path: {_safe_diagnostic(path)}") from None
    except OSError as exc:
        raise ValidationError(
            f"{_safe_diagnostic(label)} could not be inspected without following links: {_safe_diagnostic(path)}"
        ) from None


def _inspect_no_follow(
    path: Path,
    label: str,
    *,
    allow_directory: bool,
    directory_only: bool = False,
    allow_missing: bool = False,
) -> bool:
    """Reject link/reparse, hard-linked, and special evidence entries.

    Return False only when compatibility mode explicitly permits a missing
    path; all existing entries are fully inspected without following links.
    """
    metadata = _lstat_no_follow(path, label, allow_missing=allow_missing)
    if metadata is None:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise ValidationError(f"{_safe_diagnostic(label)} is a symlink or reparse point: {_safe_diagnostic(path)}")
    if _WIN:
        try:
            attributes, error = _windows_file_attributes(path)
        except (AttributeError, OSError, ValueError) as exc:
            raise ValidationError(
                f"{_safe_diagnostic(label)} reparse metadata is indeterminate: {_safe_diagnostic(path)}"
            ) from None
        if attributes == _INVALID_FILE_ATTRIBUTES:
            raise ValidationError(
                f"{_safe_diagnostic(label)} reparse metadata is indeterminate: {_safe_diagnostic(path)} "
                f"(Windows error: {_safe_diagnostic(error)})"
            )
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValidationError(f"{_safe_diagnostic(label)} is a symlink or reparse point: {_safe_diagnostic(path)}")
    if stat.S_ISREG(metadata.st_mode):
        if directory_only:
            raise ValidationError(f"{_safe_diagnostic(label)} is not a directory: {_safe_diagnostic(path)}")
        if metadata.st_nlink > 1:
            raise ValidationError(f"{_safe_diagnostic(label)} is a hard-linked regular file: {_safe_diagnostic(path)}")
        return True
    if stat.S_ISDIR(metadata.st_mode):
        if directory_only or allow_directory:
            return True
        raise ValidationError(f"{_safe_diagnostic(label)} is not a regular evidence artifact: {_safe_diagnostic(path)}")
    raise ValidationError(f"{_safe_diagnostic(label)} is a special file: {_safe_diagnostic(path)}")


def _lexical_components(path: Path):
    """Yield absolute path components without resolving links."""
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
        current = Path(absolute.anchor)
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(
            "path could not be inspected safely: "
            f"{_safe_diagnostic(path)} (filesystem operation failed)"
        ) from None
    for part in absolute.parts[len(current.parts) :]:
        current = current / part
        yield current


def _inspect_no_follow_path(
    path: Path,
    label: str,
    *,
    require_directory: bool = False,
    allow_missing: bool = False,
) -> bool:
    """Inspect every lexical ancestor and the final no-follow path entry."""
    components = list(_lexical_components(path))
    if not components:
        raise ValidationError(f"{_safe_diagnostic(label)} has no path components")
    for index, component in enumerate(components):
        is_final = index == len(components) - 1
        if not _inspect_no_follow(
            component,
            f"{_safe_diagnostic(label)} component {index}",
            allow_directory=not require_directory,
            directory_only=require_directory or not is_final,
            allow_missing=allow_missing,
        ):
            return False
    return True


def _read_input_text(path: Path) -> str:
    """Read a regular, no-follow input within the bounded payload limit."""
    try:
        absolute_path = Path(os.path.abspath(path))
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(
            "input file path could not be inspected safely: "
            f"{_safe_diagnostic(path)} (filesystem operation failed)"
        ) from None
    stream_component = _alternate_stream_component(absolute_path)
    if stream_component is not None:
        raise ValidationError(
            "input file must not use a Windows alternate data stream: "
            f"{_safe_diagnostic(path)}"
        )
    _inspect_no_follow_path(path, "input file")
    try:
        metadata = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValidationError(f"input file does not exist: {_safe_diagnostic(path)}") from None
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"input file metadata could not be inspected safely: {_safe_diagnostic(path)}"
        ) from None
    if metadata.st_size > MAX_INPUT_BYTES:
        raise ValidationError(
            f"input file exceeds the {MAX_INPUT_BYTES}-byte limit: {_safe_diagnostic(path)}"
        )
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot read input file safely: {_safe_diagnostic(path)}") from None
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError(
            f"input file exceeds the {MAX_INPUT_BYTES}-byte limit: {_safe_diagnostic(path)}"
        )
    return raw.decode("utf-8")


def _validate_evidence_paths(evidence, *, allow_missing: bool = False):
    """Require exact repo-relative, no-follow evidence artifact paths.

    Incoming evidence must exist. Persisted legacy evidence may be missing, but
    is still checked lexically and any path that does exist is checked without
    following links.
    """
    try:
        root = Path(os.path.abspath(os.fspath(REPOSITORY_ROOT)))
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(
            "evidence root could not be inspected safely "
            f"({_safe_exception_detail(exc)})"
        ) from None
    _inspect_no_follow_path(root, "evidence root", require_directory=True)

    for index, value in enumerate(evidence):
        label = f"evidence[{index}]"
        parts = value.split("/")
        invalid_syntax = (
            value != value.strip()
            or not value
            or "\\" in value
            or "\x00" in value
            or value.startswith("/")
            or any(":" in part for part in parts)
            or _WINDOWS_DRIVE_RE.match(value) is not None
            or _EVIDENCE_URI_RE.match(value) is not None
            or any(part in ("", ".", "..") for part in parts)
            or (_WIN and any(part.endswith((".", " ")) for part in parts))
        )
        if invalid_syntax or PurePosixPath(value).is_absolute():
            raise ValidationError(
                f"{_safe_diagnostic(label)} must be an exact repo-relative forward-slash artifact path"
            )
        _inspect_no_follow_path(
            root.joinpath(*parts),
            label,
            allow_missing=allow_missing,
        )


def validate_reflection(payload):
    """Return normalized required fields, rejecting missing or malformed input."""
    if not isinstance(payload, dict):
        raise ValidationError("reflection payload must be a JSON object")
    fields = {}
    for field in REQUIRED_STRING_FIELDS:
        value = _validate_string(payload, field)
        _ensure_safe_text(value, f"field {_safe_diagnostic(field)}")
        _scan_for_secret(field, value)
        fields[field] = value
    for field in OPTIONAL_STRING_FIELDS:
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValidationError(f"field must be a string: {_safe_diagnostic(field)}")
        if not value.strip():
            raise ValidationError(f"field must not be empty: {_safe_diagnostic(field)}")
        if len(value) > MAX_STRING_CHARS:
            raise ValidationError(
                f"field is longer than {MAX_STRING_CHARS} characters: {_safe_diagnostic(field)}"
            )
        _ensure_safe_text(value, f"field {_safe_diagnostic(field)}")
        _scan_for_secret(field, value)
        fields[field] = value
    for field in REQUIRED_LIST_FIELDS:
        allow_empty = field not in _MANDATORY_LIST_FIELDS
        value = _validate_list(payload, field, allow_empty=allow_empty)
        for index, item in enumerate(value):
            _ensure_safe_text(item, f"{_safe_diagnostic(field)}[{index}]")
            _scan_for_secret(field, item)
        fields[field] = value
    try:
        _validate_evidence_paths(fields["evidence"])
    except (ProtectedPathError, ValidationError, SecretDetectedError) as exc:
        # Boundary errors may have originated in filesystem helpers; preserve
        # their type while never forwarding unsanitized exception text.
        raise type(exc)(_safe_exception_detail(exc)) from None
    except OSError:
        raise ValidationError(
            "evidence paths could not be inspected safely "
            "(filesystem operation failed)"
        ) from None
    except (TypeError, ValueError):
        raise ValidationError("evidence paths could not be inspected safely") from None
    return fields


def _slug(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or DEFAULT_SCOPE


def lesson_candidate_id(fields, scope, source_reflection):
    """Return a stable ID for the persisted lesson-candidate identity.

    The identity includes every field stored with a new candidate, including
    the proposed rule and its evidence. Reflection narrative fields are not
    persisted as candidate identity and do not create a second candidate when
    they change.
    """
    identity = {
        "scope": scope,
        "source_reflection": source_reflection,
        "reusable_lesson": fields["reusable_lesson"],
        "rule_candidate": fields["rule_candidate"],
        "evidence": fields["evidence"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    return f"lesson-{_slug(scope)}-{digest}"


def _bullet_list(items):
    if not items:
        return f"- {EMPTY_LIST_NOTE}"
    return "\n".join(f"- {item}" for item in items)


def build_reflection(payload: dict) -> str:
    """Render a validated reflection from a JSON payload."""
    fields = validate_reflection(payload)
    scope = fields.get("scope") or DEFAULT_SCOPE
    source = fields.get("source_reflection") or "(not yet filed)"
    lines = [
        "# Task closeout reflection",
        "",
        f"Requested outcome: {fields['requested_outcome']}",
        "",
        f"Actual outcome: {fields['actual_outcome']}",
        "",
        "## Changed files",
        "",
        _bullet_list(fields["changed_files"]),
        "",
        "## Validation performed",
        "",
        _bullet_list(fields["validation_performed"]),
        "",
        "## Failures and near-misses",
        "",
        _bullet_list(fields["failures_near_misses"]),
        "",
        "## Root causes",
        "",
        _bullet_list(fields["root_causes"]),
        "",
        "## Fixes",
        "",
        _bullet_list(fields["fixes"]),
        "",
        "## Evidence",
        "",
        _bullet_list(fields["evidence"]),
        "",
        f"Reusable lesson: {fields['reusable_lesson']}",
        "",
        f"Rule candidate: {fields['rule_candidate']}",
        "",
        "## Open questions",
        "",
        _bullet_list(fields["open_questions"]),
        "",
        f"Confidence and limitations: {fields['confidence_limitations']}",
        "",
        f"Scope: {scope}",
        "",
        f"Source reflection: {source}",
        "",
        "## Lesson candidate",
        "",
        lesson_candidate_id(fields, scope, source),
        "",
    ]
    draft = "\n".join(lines)
    _ensure_safe_text(draft, "reflection draft")
    return draft


def is_protected_path(path: Path) -> bool:
    """Return True for protected names and conservative Windows rule aliases."""
    def normalized(component: str) -> str:
        if _WIN:
            # Win32 normalizes trailing dots and spaces in path components.
            return component.rstrip(" .").casefold()
        return component.lower()

    if normalized(path.name) in POLICY_FILENAMES:
        return True
    return "rules" in {normalized(part) for part in path.parts}


def _alternate_stream_component(path: Path) -> str | None:
    """Return a Windows alternate-stream component, if present."""
    if not _WIN:
        return None
    for part in path.parts[1:]:
        if ":" in part:
            return part
    return None


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
        if attributes == _INVALID_FILE_ATTRIBUTES
        else 0
    )
    return attributes, error


def _symlink_component(path: Path) -> bool | None:
    """Return whether a component is a symlink, or None when it is absent.

    Unlike os.path.islink, explicit lstat keeps metadata failures visible so
    an indeterminate inspection cannot be treated as a safe path.
    """
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except NotADirectoryError as exc:
        raise ProtectedPathError(
            "refusing to write because a path component is not a directory: "
            f"{_safe_diagnostic(path)}"
        ) from None
    except OSError as exc:
        raise ProtectedPathError(
            "refusing to write because a path component could not be "
            f"inspected for symlink protection: {_safe_diagnostic(path)} ({_safe_exception_detail(exc)})"
        ) from None
    return stat.S_ISLNK(metadata.st_mode)


def _reparse_component(path: Path) -> Path | None:
    """Return the first symlink, junction, or other reparse component.

    Windows junctions are not classic symlinks, so the reparse-point file
    attribute is checked explicitly. The path is expected to be absolute.
    Symlink metadata is inspected with fail-closed lstat semantics on every
    platform; only a genuinely absent component is accepted as a new output.
    """
    if not _WIN:
        current = Path(path.anchor)
        for part in path.parts[len(current.parts) :]:
            current = current / part
            link_status = _symlink_component(current)
            if link_status is None:
                # Once a component is genuinely absent, every later component
                # is absent as well; do not inspect past that point.
                return None
            if link_status:
                return current
        return None

    current = Path(path.anchor)
    for part in path.parts[len(current.parts) :]:
        current = current / part
        if _symlink_component(current):
            return current
        attributes, error = _windows_file_attributes(current)
        if attributes == _INVALID_FILE_ATTRIBUTES:
            if error in (_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND):
                # A genuinely absent component is allowed for a new output;
                # every later component is necessarily absent as well.
                return None
            raise ProtectedPathError(
                "refusing to write because a path component could not be "
                f"inspected for reparse protection: {_safe_diagnostic(current)} "
                f"(Windows error: {_safe_diagnostic(error)})"
            )
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            return current
    return None


def _multi_link_path(path: Path) -> Path | None:
    """Return a path whose existing identity is unsafe or a multi-link file."""
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        # A genuinely absent output (or absent parent) is allowed.
        return None
    except OSError:
        # Do not treat an indeterminate metadata query as a new file.
        return path
    except ValueError as exc:
        raise ValidationError(
            f"invalid destination path (embedded NUL or invalid value): {_safe_diagnostic(path)}"
        ) from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1:
        return path
    return None


def _repository_relative_parts(path: Path) -> tuple[str, ...] | None:
    """Return canonical relative parts when path is inside the repository root.

    Both sides are resolved before comparison so a short-name/case alias or a
    symlinked root cannot make a repository path appear to be outside its
    approved boundary.
    """
    try:
        canonical_path = os.path.normcase(
            os.path.realpath(os.path.abspath(str(path)))
        )
        canonical_root = os.path.normcase(
            os.path.realpath(os.path.abspath(str(REPOSITORY_ROOT)))
        )
    except (OSError, ValueError) as exc:
        raise ProtectedPathError(
            "refusing to write because the repository destination could not be "
            f"canonically inspected: {_safe_diagnostic(path)}"
        ) from None
    try:
        relative = os.path.relpath(canonical_path, canonical_root)
    except ValueError:
        # A different drive/UNC namespace cannot be inside this repository root.
        return None
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return None
    if relative == os.curdir:
        return ()
    return tuple(Path(relative).parts)


def _validate_repository_destination(
    requested: Path, resolved: Path, purpose: str
) -> None:
    """Restrict repository-contained destinations to their approved trees."""
    if purpose not in ("output", "state"):
        raise ValidationError("destination purpose must be 'output' or 'state'")
    relative_parts = _repository_relative_parts(resolved)
    if relative_parts is None:
        return
    if purpose == "output":
        allowed = (
            len(relative_parts) > len(_REPOSITORY_OUTPUT_PREFIX)
            and relative_parts[: len(_REPOSITORY_OUTPUT_PREFIX)]
            == _REPOSITORY_OUTPUT_PREFIX
        )
        approved = "docs/reflections/"
    else:
        allowed = relative_parts == _REPOSITORY_STATE_PATH
        approved = "state/lessons_pending.json"
    if not allowed:
        raise ProtectedPathError(
            f"refusing to write a repository {purpose} outside {approved}: {_safe_diagnostic(requested)}"
        )


def _validate_output_path(path: Path, *, purpose: str = "output") -> Path:
    """Refuse protected, stream, reparse, unsafe, or misplaced destinations.

    Repository-contained reflection output is limited to docs/reflections/;
    lesson staging is limited to the canonical state/lessons_pending.json file.
    Paths outside the repository remain available for isolated test and scratch
    outputs. Checks are conservative and path-based: reparse and metadata
    failures are closed, and existing
    non-regular destinations are refused along with hard-linked regular files.
    A concurrent replacement after validation is not prevented.
    """
    try:
        requested = Path(os.path.abspath(path))
        stream_component = _alternate_stream_component(requested)
        if stream_component is not None:
            raise ProtectedPathError(
                "refusing to write through a Windows alternate data stream: "
                f"{_safe_diagnostic(requested)} (stream component: {_safe_diagnostic(stream_component)})"
            )
        escape = _reparse_component(requested)
        if escape is not None:
            raise ProtectedPathError(
                "refusing to write through a symlink or junction path: "
                f"{_safe_diagnostic(requested)} (reparse component: {_safe_diagnostic(escape)})"
            )
        try:
            resolved = Path(os.path.realpath(requested))
        except OSError as exc:
            raise ProtectedPathError(
                "refusing to write because destination resolution could not be "
                f"safely inspected: {_safe_diagnostic(requested)} ({_safe_exception_detail(exc)})"
            ) from None
        if is_protected_path(resolved):
            raise ProtectedPathError(
                "refusing to write a policy or persona file: SOUL.md, AGENTS.md, "
                f"CLAUDE.md, and rules/** are protected (requested: {_safe_diagnostic(path)})"
            )
        _validate_repository_destination(requested, resolved, purpose)
        linked = _multi_link_path(requested)
        if linked is not None:
            try:
                link_count = os.stat(linked, follow_symlinks=False).st_nlink
            except OSError:
                link_count = "unknown"
            raise ProtectedPathError(
                "refusing to write because output file identity cannot be safely "
                f"classified (non-regular or possible hard link): {_safe_diagnostic(linked)} "
                f"(link count: {link_count})"
            )
        return requested
    except (ProtectedPathError, ValidationError):
        raise
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"invalid destination path (embedded NUL or invalid value): {_safe_diagnostic(path)}"
        ) from None
    except OSError as exc:
        raise ProtectedPathError(
            "destination path could not be inspected safely: "
            f"{_safe_diagnostic(path)} (filesystem operation failed)"
        ) from None


def _canonical_destination(path: Path) -> str:
    """Return a normalized spelling for platform-aware destination comparison.

    Windows spellings are case-folded; POSIX spellings retain case unless the
    platform's normcase implementation says otherwise.
    """
    try:
        canonical = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(
            "destination identity could not be inspected safely: "
            f"{_safe_diagnostic(path)} (filesystem operation failed)"
        ) from None
    return canonical.casefold() if _WIN else canonical


def _same_destination(left: Path, right: Path) -> bool:
    """Compare normalized paths and, when possible, existing file identities.

    Case folding is applied on Windows only; unavailable samefile metadata is
    not treated as proof that two destinations are identical.
    """
    if _canonical_destination(left) == _canonical_destination(right):
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _reject_evidence_destination(destination: Path, evidence: list[str]) -> None:
    """Never use a cited source artifact as a reflection/state destination."""
    try:
        root = Path(os.path.abspath(os.fspath(REPOSITORY_ROOT)))
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(
            "evidence destination could not be inspected safely "
            f"({_safe_exception_detail(exc)})"
        ) from None
    for index, relative in enumerate(evidence):
        source = root.joinpath(*relative.split("/"))
        try:
            same = _same_destination(destination, source)
        except (ProtectedPathError, ValidationError):
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise ValidationError(
                "evidence destination identity could not be inspected safely: "
                f"{_safe_diagnostic(destination)} (filesystem operation failed)"
            ) from None
        if same:
            raise ValidationError(
                "destination must not overwrite a payload evidence artifact "
                f"(evidence[{index}])"
            )


def write_reflection(path: Path, content: str, replace: bool = False) -> None:
    """Write one new reflection without silently overwriting another.

    The write is a single path-based operation and may be partial if the
    filesystem fails after bytes have been emitted.
    """
    _ensure_safe_text(content, "reflection content")
    _scan_for_secret("reflection content", content)
    try:
        path = _validate_output_path(path)
        exists = _path_exists(path)
    except (ProtectedPathError, ValidationError, SecretDetectedError) as exc:
        raise type(exc)(_safe_api_exception_detail(exc, requested_path=path)) from None
    except OSError:
        raise ProtectedPathError(
            "reflection destination could not be inspected safely: "
            f"{_safe_diagnostic(path)} (filesystem operation failed)"
        ) from None
    except (TypeError, ValueError):
        raise ValidationError(
            f"invalid reflection destination path: {_safe_diagnostic(path)}"
        ) from None
    except Exception:
        raise ProtectedPathError(
            "reflection destination could not be processed safely: "
            f"{_safe_diagnostic(path)} (filesystem operation failed)"
        ) from None
    if exists and not replace:
        raise ValidationError(
            f"refusing to overwrite existing file without --replace: {_safe_diagnostic(path)}"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except UnicodeError as exc:
        raise ValidationError(
            "reflection output could not be encoded as UTF-8"
        ) from None
    except (OSError, ValueError):
        raise ProtectedPathError(
            "reflection output could not be written safely: "
            f"{_safe_diagnostic(path)} (filesystem operation failed)"
        ) from None
    except Exception:
        raise ProtectedPathError(
            "reflection output could not be processed safely: "
            f"{_safe_diagnostic(path)} (filesystem operation failed)"
        ) from None


def _validate_state_lesson(record, index: int) -> None:
    """Validate one persisted lesson, including safe legacy evidence paths."""
    if not isinstance(record, dict):
        raise ValidationError(f"state lesson {index} must be a JSON object")
    for field in _REQUIRED_LESSON_FIELDS:
        if field not in record:
            raise ValidationError(f"state lesson {index} is missing field: {_safe_diagnostic(field)}")
    for field in ("id", "scope", "status", "lesson", "source_reflection"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(
                f"state lesson {index} field must be a non-empty string: {_safe_diagnostic(field)}"
            )
        _ensure_safe_text(value, f"state lesson {index} field: {_safe_diagnostic(field)}")
        _scan_for_secret(f"state lesson {index} field: {_safe_diagnostic(field)}", value)
    if record["status"] not in _LESSON_STATUSES:
        raise ValidationError(f"state lesson {index} has an invalid status")
    for field in ("evidence", "promoted_to"):
        value = record[field]
        if not isinstance(value, list):
            raise ValidationError(f"state lesson {index} field must be a list: {_safe_diagnostic(field)}")
        for item_index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValidationError(
                    f"state lesson {index} field must contain non-empty strings: {_safe_diagnostic(field)}"
                )
            label = f"state lesson {index} field: {_safe_diagnostic(field)}[{item_index}]"
            _ensure_safe_text(item, label)
            _scan_for_secret(label, item)
    _validate_evidence_paths(record["evidence"], allow_missing=True)
    if "rule_candidate" in record:
        value = record["rule_candidate"]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(
                f"state lesson {index} rule_candidate must be a non-empty string"
            )
        _ensure_safe_text(value, f"state lesson {index} rule_candidate")
        _scan_for_secret(f"state lesson {index} rule_candidate", value)

    known_fields = frozenset(_REQUIRED_LESSON_FIELDS) | {"rule_candidate"}
    for field, value in record.items():
        if field in known_fields:
            continue
        _ensure_safe_text(field, f"state lesson {index} unknown field name")
        _scan_for_secret(f"state lesson {index} unknown field name", field)
        _validate_persisted_strings(value, f"state lesson {index} unknown field")


def _validate_state_schema(state, state_path: Path) -> dict:
    """Validate the persisted state shape and duplicate identities."""
    if not isinstance(state, dict):
        raise ValidationError(f"state file must be a JSON object: {_safe_diagnostic(state_path)}")
    if type(state.get("schema_version")) is not int or state.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            f"state file schema_version must be the integer {SCHEMA_VERSION}: {_safe_diagnostic(state_path)}"
        )
    lessons = state.get("lessons")
    if not isinstance(lessons, list):
        raise ValidationError(f"state file 'lessons' must be a list: {_safe_diagnostic(state_path)}")
    for field, value in state.items():
        if field in ("schema_version", "lessons"):
            continue
        _ensure_safe_text(field, "state unknown field name")
        _scan_for_secret("state unknown field name", field)
        _validate_persisted_strings(value, "state unknown field")
    seen_ids = set()
    # The historical three-field identity has no rule_candidate field. Keep a
    # per-base record of explicit rules plus an optional unknown-rule ID so a
    # legacy record and a full record cannot coexist under different IDs when
    # their other persisted identity facts match.
    seen_identity_bases = {}
    for index, record in enumerate(lessons):
        _validate_state_lesson(record, index)
        record_id = record["id"]
        if record_id in seen_ids:
            prefix = "state contains duplicate lesson id: "
            raise ValidationError(
                prefix
                + _safe_diagnostic(
                    record_id,
                    limit=MAX_DIAGNOSTIC_CHARS - len(prefix),
                )
            )
        seen_ids.add(record_id)
        identity_base = (
            record["scope"],
            record["source_reflection"],
            record["lesson"],
            tuple(record["evidence"]),
        )
        rule_present = "rule_candidate" in record
        rule_value = record.get("rule_candidate")
        prior = seen_identity_bases.get(identity_base)
        if prior is None:
            prior = {"unknown_id": None, "rules": {}}
            seen_identity_bases[identity_base] = prior
        prior_id = None
        if not rule_present:
            if prior["unknown_id"] is not None or prior["rules"]:
                prior_id = prior["unknown_id"] or next(iter(prior["rules"].values()))
            else:
                prior["unknown_id"] = record_id
        elif prior["unknown_id"] is not None or rule_value in prior["rules"]:
            prior_id = prior["unknown_id"] or prior["rules"][rule_value]
        else:
            prior["rules"][rule_value] = record_id
        if prior_id is not None:
            prefix = "state contains duplicate lesson identity under different ids: "
            available = MAX_DIAGNOSTIC_CHARS - len(prefix) - len(", ")
            prior_limit = available // 2
            record_limit = available - prior_limit
            raise ValidationError(
                prefix
                + _safe_diagnostic(prior_id, limit=prior_limit)
                + ", "
                + _safe_diagnostic(record_id, limit=record_limit)
            )
    _reject_nonfinite(state, "state")
    _ensure_state_serializable(state, state_path)
    return state


def _read_state(state_path: Path):
    if not _path_exists(state_path):
        return {"schema_version": SCHEMA_VERSION, "lessons": []}
    try:
        raw = state_path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ValidationError(
            f"state file is not valid UTF-8: {_safe_diagnostic(state_path)}"
        ) from None
    except (OSError, ValueError) as exc:
        raise ValidationError(
            "state file could not be read safely: "
            f"{_safe_diagnostic(state_path)} ({_safe_exception_detail(exc)})"
        ) from None
    state = _loads_strict_json(raw, f"state file {_safe_diagnostic(state_path)}")
    try:
        return _validate_state_schema(state, state_path)
    except RecursionError as exc:
        raise ValidationError(
            f"state file nesting depth exceeds the safe limit: {_safe_diagnostic(state_path)}"
        ) from None


def _legacy_lesson_id(scope, source_reflection, reusable_lesson):
    """Return the historical three-field lesson ID when compatibility applies."""
    digest = hashlib.sha256(
        json.dumps([scope, source_reflection, reusable_lesson], sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()[:8]
    return f"lesson-{_slug(scope)}-{digest}"


def _lesson_facts_match(
    record: dict,
    *,
    scope: str,
    source: str,
    lesson: str,
    rule_candidate: str,
    evidence: list,
    allow_missing_rule: bool,
) -> bool:
    """Compare persisted identity facts without treating the ID as proof."""
    if (
        record.get("scope") != scope
        or record.get("source_reflection") != source
        or record.get("lesson") != lesson
        or record.get("evidence") != evidence
    ):
        return False
    if "rule_candidate" not in record:
        return allow_missing_rule
    return record["rule_candidate"] == rule_candidate


def _same_lesson_identity(
    left: dict,
    right: dict,
) -> bool:
    """Return whether two persisted records represent the same identity.

    A missing optional rule is the historical three-field wildcard. This
    keeps a legacy record from being silently duplicated by a later full
    candidate, while still allowing two explicit rule candidates to coexist.
    """
    if (
        left.get("scope") != right.get("scope")
        or left.get("source_reflection") != right.get("source_reflection")
        or left.get("lesson") != right.get("lesson")
        or left.get("evidence") != right.get("evidence")
    ):
        return False
    left_rule_present = "rule_candidate" in left
    right_rule_present = "rule_candidate" in right
    return (
        not left_rule_present
        or not right_rule_present
        or left["rule_candidate"] == right["rule_candidate"]
    )


def _prepare_lesson_candidate(
    payload: dict,
    state_path: Path,
) -> tuple[Path, dict, str | None]:
    """Validate state and fully preflight an identity-aware update."""
    state_path = _validate_output_path(state_path, purpose="state")
    fields = validate_reflection(payload)
    _reject_evidence_destination(state_path, fields["evidence"])
    state = _read_state(state_path)
    lessons = list(state.get("lessons", []))
    scope = fields.get("scope") or DEFAULT_SCOPE
    source = fields.get("source_reflection") or "(not yet filed)"
    lesson = fields["reusable_lesson"]
    rule_candidate = fields["rule_candidate"]
    evidence = list(fields["evidence"])
    candidate_id = lesson_candidate_id(fields, scope, source)
    legacy_id = _legacy_lesson_id(scope, source, lesson)

    for existing in lessons:
        existing_id = existing.get("id")
        if existing_id == candidate_id:
            if not _lesson_facts_match(
                existing,
                scope=scope,
                source=source,
                lesson=lesson,
                rule_candidate=rule_candidate,
                evidence=evidence,
                allow_missing_rule=False,
            ):
                raise ValidationError(
                    f"lesson candidate id collision: {_safe_diagnostic(candidate_id)}"
                )
            return state_path, existing, None
        if existing_id == legacy_id:
            if (
                existing.get("scope") != scope
                or existing.get("source_reflection") != source
                or existing.get("lesson") != lesson
            ):
                raise ValidationError(
                    f"legacy lesson id collision: {_safe_diagnostic(legacy_id)}"
                )
            if _lesson_facts_match(
                existing,
                scope=scope,
                source=source,
                lesson=lesson,
                rule_candidate=rule_candidate,
                evidence=evidence,
                allow_missing_rule=True,
            ):
                return state_path, existing, None
            # The legacy ID encodes only the three legacy facts. An explicit
            # rule/evidence change therefore creates a new current candidate.
            continue

    incoming = {
        "scope": scope,
        "source_reflection": source,
        "lesson": lesson,
        "rule_candidate": rule_candidate,
        "evidence": evidence,
        "promoted_to": [],
    }
    for existing in lessons:
        if _same_lesson_identity(existing, incoming):
            raise ValidationError(
                "state contains a lesson identity already stored under a different id: "
                f"{_safe_diagnostic(existing.get('id'))}"
            )

    candidate = {
        "id": candidate_id,
        "scope": scope,
        "status": "candidate",
        "lesson": lesson,
        "rule_candidate": rule_candidate,
        "source_reflection": source,
        "evidence": evidence,
        "promoted_to": [],
    }
    lessons.append(candidate)
    state["lessons"] = lessons
    serialized = _serialize_state(state, state_path, indent=2) + "\n"
    return state_path, candidate, serialized


def stage_lesson_candidate(payload: dict, state_path: Path) -> dict:
    """Append a lesson candidate without activating a permanent rule.

    State is preflighted for UTF-8 and depth, but the subsequent in-place
    rewrite is sequential and non-transactional and may be partial.
    """
    try:
        state_path, candidate, serialized = _prepare_lesson_candidate(payload, state_path)
    except (ProtectedPathError, ValidationError, SecretDetectedError) as exc:
        raise type(exc)(_safe_api_exception_detail(exc, requested_path=state_path)) from None
    except OSError:
        raise ProtectedPathError(
            "lesson state destination could not be inspected safely: "
            f"{_safe_diagnostic(state_path)} (filesystem operation failed)"
        ) from None
    except (TypeError, ValueError):
        raise ValidationError(
            f"invalid lesson state destination path: {_safe_diagnostic(state_path)}"
        ) from None
    except Exception:
        raise ProtectedPathError(
            "lesson state destination could not be processed safely: "
            f"{_safe_diagnostic(state_path)} (filesystem operation failed)"
        ) from None
    if serialized is None:
        return candidate
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(serialized, encoding="utf-8")
    except UnicodeError as exc:
        raise ValidationError(
            f"state file could not be encoded as UTF-8: {_safe_diagnostic(state_path)}"
        ) from None
    except (OSError, ValueError):
        raise ProtectedPathError(
            "state file could not be written safely: "
            f"{_safe_diagnostic(state_path)} (filesystem operation failed)"
        ) from None
    except Exception:
        raise ProtectedPathError(
            "state file could not be processed safely: "
            f"{_safe_diagnostic(state_path)} (filesystem operation failed)"
        ) from None
    return candidate


class _SafeArgumentParser(argparse.ArgumentParser):
    """Keep argparse diagnostics bounded before they reach the terminal."""

    def error(self, message):
        usage = self.format_usage().rstrip("\n")
        if usage:
            _write_cli_text(usage, stream=sys.stderr)
        _write_cli_text(
            f"{_safe_diagnostic(self.prog)}: error: {_safe_diagnostic(message)}",
            stream=sys.stderr,
        )
        raise SystemExit(2)


def _parse_args(argv):
    parser = _SafeArgumentParser(
        prog="closeout_reflection",
        description="Deterministic, offline task-closeout reflection tool.",
    )
    parser.add_argument(
        "--input", required=True, help="Path to the reflection payload JSON file."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the reflection draft to --output (off by default).",
    )
    parser.add_argument(
        "--output",
        help="Output path for the reflection markdown (required with --write).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Allow overwriting an existing --output file.",
    )
    parser.add_argument(
        "--stage-lesson",
        action="store_true",
        help="Also stage a lesson candidate (requires --write).",
    )
    parser.add_argument(
        "--state-path",
        default=str(Path(REPOSITORY_ROOT) / DEFAULT_STATE_PATH),
        help="Path to lessons_pending.json for --stage-lesson (default: repository state).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Default to dry-run; require --write for repository output.

    With lesson staging, all destinations and the prospective state are checked
    before the reflection write; filesystem errors are reported without a
    traceback, and partial sequential writes are not rolled back.
    """
    args = _parse_args(argv)
    input_path = Path(args.input)
    try:
        raw = _read_input_text(input_path)
        payload = _loads_strict_json(raw, f"input file {_safe_diagnostic(args.input)}")
        draft = build_reflection(payload)
    except OSError as exc:
        _write_cli_text(
            f"error: cannot read input file: {_safe_diagnostic(args.input)}: "
            f"{_safe_cli_exception_detail(exc, known_paths=(args.input, input_path))}",
            stream=sys.stderr,
        )
        return 2
    except UnicodeError:
        _write_cli_text(
            f"error: input file is not valid UTF-8: {_safe_diagnostic(args.input)}",
            stream=sys.stderr,
        )
        return 2
    except json.JSONDecodeError as exc:
        _write_cli_text(
            f"error: invalid JSON in input: {_safe_diagnostic(args.input)}: "
            f"{_safe_cli_exception_detail(exc, known_paths=(args.input, input_path))}",
            stream=sys.stderr,
        )
        return 2
    except RecursionError:
        _write_cli_text(
            f"error: input nesting is too deep to parse safely: {_safe_diagnostic(args.input)}",
            stream=sys.stderr,
        )
        return 2
    except (ValidationError, SecretDetectedError) as exc:
        _write_cli_text(
            f"error: {_safe_cli_exception_detail(exc, known_paths=(args.input, input_path))}",
            stream=sys.stderr,
        )
        return 2
    except ValueError as exc:
        _write_cli_text(
            f"error: cannot read input file: {_safe_diagnostic(args.input)}: "
            f"{_safe_cli_exception_detail(exc, known_paths=(args.input, input_path))}",
            stream=sys.stderr,
        )
        return 2
    except Exception:
        _write_cli_text(
            "error: reflection input could not be processed safely: "
            "filesystem operation failed",
            stream=sys.stderr,
        )
        return 2

    if not args.write:
        _write_cli_text(draft, end="")
        _write_cli_text(
            "dry-run: no file written; pass --write with an explicit --output "
            "to create one"
        )
        return 0

    if not args.output:
        _write_cli_text(
            "error: --write requires an explicit --output path",
            stream=sys.stderr,
        )
        return 2
    output_path = Path(args.output)
    state_path = Path(args.state_path) if args.stage_lesson else None
    try:
        # Validate every requested write destination before creating either file.
        output_path = _validate_output_path(output_path)
        if _same_destination(input_path, output_path):
            raise ValidationError(
                "--input and --output must identify distinct destinations"
            )
        _reject_evidence_destination(output_path, payload["evidence"])
        if state_path is not None:
            state_path = _validate_output_path(state_path, purpose="state")
            if _same_destination(input_path, state_path):
                raise ValidationError(
                    "--input and --state-path must identify distinct destinations"
                )
            if _same_destination(output_path, state_path):
                raise ValidationError(
                    "--output and --state-path must identify distinct destinations"
                )
            _reject_evidence_destination(state_path, payload["evidence"])
            # Precompute the exact candidate-augmented state so deterministic
            # encoding/depth failures happen before the reflection is written.
            _prepare_lesson_candidate(payload, state_path)
    except (ProtectedPathError, ValidationError, OSError) as exc:
        known_paths = (args.output, output_path, args.state_path, state_path)
        _write_cli_text(
            f"error: {_safe_cli_exception_detail(exc, known_paths=known_paths)}",
            stream=sys.stderr,
        )
        return 2
    except (UnicodeError, ValueError) as exc:
        known_paths = (args.output, output_path, args.state_path, state_path)
        _write_cli_text(
            f"error: {_safe_cli_exception_detail(exc, known_paths=known_paths)}",
            stream=sys.stderr,
        )
        return 2
    except Exception:
        _write_cli_text(
            "error: output could not be processed safely: "
            "filesystem operation failed",
            stream=sys.stderr,
        )
        return 2

    write_phase = "reflection"
    try:
        write_reflection(output_path, draft, replace=args.replace)
        staged = None
        if state_path is not None:
            write_phase = "state"
            staged = stage_lesson_candidate(payload, state_path)
    except OSError:
        _write_cli_text(
            f"error: {write_phase} write failed: filesystem operation failed",
            stream=sys.stderr,
        )
        return 2
    except (ProtectedPathError, ValidationError, SecretDetectedError, UnicodeError, ValueError) as exc:
        known_paths = (args.output, output_path, args.state_path, state_path)
        _write_cli_text(
            f"error: {write_phase} write failed: "
            f"{_safe_cli_exception_detail(exc, known_paths=known_paths)}",
            stream=sys.stderr,
        )
        return 2
    except Exception:
        _write_cli_text(
            f"error: {write_phase} write failed: filesystem operation failed",
            stream=sys.stderr,
        )
        return 2

    _write_cli_text(f"wrote: {_safe_diagnostic(output_path)}")
    if staged is not None:
        try:
            staged_id = staged.get("id")
        except Exception:
            staged_id = "<redacted diagnostic>"
        _write_cli_text(
            f"staged lesson candidate: {_safe_diagnostic(staged_id)}",
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
