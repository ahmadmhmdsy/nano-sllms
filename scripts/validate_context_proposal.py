"""Gated, non-adopting validation for proposed agent-context changes.
This tool compares a baseline context tree with a candidate context tree,
runs the repository's public read-only lint_tree helper
(scripts/check_agent_context.py) against both, classifies lint regressions
versus improvements, and renders a unified diff for user review. Diff keys are
intentionally relative to each context-tree root so a rename is an honest,
applyable delete/add pair; repository metadata and virtual-environment
contents are excluded.
``root`` is the evidence-resolution root: evidence paths inside the candidate
records must exist there, so a proposal may only cite evidence that already
exists in the real repository.

It can only *stage* outputs - an explicit report path and an optional diff
path. It cannot adopt live policy and has no flag, hook, or subprocess that
calls skillopt-sleep adoption. Output paths are resolved (junctions and
symlinks included) and refused if they land on SOUL.md, AGENTS.md,
CLAUDE.md, anything under rules/, or the repository's live policy roots.
Adoption is always an explicit human decision.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import re
import stat
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

PROTECTED_FILENAMES = frozenset(("soul.md", "agents.md", "claude.md"))
_IGNORED_DIRS = frozenset(
    (
        ".git",
        ".venv",
        ".skillopt-sleep",
        ".pytest_cache",
        ".superpowers",
        "__pycache__",
        "artifacts",
        "checkpoint_backup",
        "data",
        "logs",
        "models",
        "node_modules",
        "outputs",
        "raw",
        "resources",
        "runs",
        "scratch",
        "wandb",
    )
)

_MAX_DIFF_FILE_BYTES = 1 << 20
_SECRET_PATTERNS = (
    ("AWS access key ID", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "private key block",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
            r"(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|$)",
            re.DOTALL,
        ),
    ),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}\b")),
    (
        "OpenAI-style token",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ),
    ("GitLab token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    (
        "Bearer token",
        re.compile(
            r"\b(?P<bearer_label>Bearer)(?:\s+|\s*[:=]\s*|"
            r"\x1f+\s*)"
            r"(?P<bearer_value>(?:[A-Za-z0-9._~+/=-]|\x1f)+)",
            re.IGNORECASE,
        ),
    ),
    (
        "JSON Web Token",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]*\."
            r"(?:[A-Za-z0-9_-]*={0,2}\.){1,3}"
            r"[A-Za-z0-9_-]*={0,2}(?![A-Za-z0-9_-])"
        ),
    ),
    (
        "credential assignment",
        re.compile(
            r"(?<![A-Za-z0-9])(?P<assignment_quote>[\"']?)"
            r"(?P<assignment_key>_*(?:[A-Za-z0-9]+[_-])*"
            r"(?:password|passwd|secret|credentials|passphrase|token|"
            r"(?:refresh|access|auth|session|oauth|client)[_-]?token|"
            r"api[_-]?(?:key|token)|bearer[_-]?token|"
            r"secret[_-]?key|access[_-]?key|"
            r"aws[_-]?secret[_-]?access[_-]?key|client[_-]?secret|"
            r"private[_-]?key|authorization)"
            r"(?:[_-][A-Za-z0-9]+)*)"
            r"(?P<assignment_quote_end>[\"']?)(?![A-Za-z0-9_])"
            r"(?P<assignment_separator>\s*[=:]\s*)"
            r"(?P<assignment_value>"
            r"\"(?:\\.|[^\"\\\r\n])*\""
            r"|'(?:\\.|[^'\\\r\n])*'"
            r"|\x60[^\x60\r\n]*\x60"
            r"|\([^)\r\n]*\)"
            r"|\[[^\]\r\n]*\]"
            r"|\{[^{}\r\n]*\}"
            r"|(?:\.{3,}|…+|\*+|_+)(?=[\x60\'\".,;:)\]}]|\s|$)"
            r"|\S[^\r\n]*"
            r")?",
            re.IGNORECASE,
        ),
    ),
)
_CONTROL_CHARACTERS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
_PATH_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_FORMAT_CATEGORIES = frozenset(("Cf", "Zl", "Zp"))
_MAX_DIAGNOSTIC_CHARS = 512
_DIAGNOSTIC_TRUNCATION = "...<truncated>"

_WIN = sys.platform == "win32"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LIVE_STAGING_ROOT = _REPOSITORY_ROOT / ".skillopt-sleep" / "staging"
_LINTER_PATH = Path(__file__).resolve().parent / "check_agent_context.py"


def _is_format_character(character: str) -> bool:
    """Return whether a character is a format/control diagnostic hazard."""
    # U+E0000 is unassigned in some Unicode tables, while the remainder of
    # the tag range is Cf. Keep the whole range explicit so it cannot hide a
    # token on Python runtimes whose Unicode database predates the tag.
    return (
        unicodedata.category(character) in _FORMAT_CATEGORIES
        or "\U000e0000" <= character <= "\U000e007f"
    )


def _contains_format_character(value: str) -> bool:
    """Return whether text contains a Unicode format or line-separator char."""
    return any(_is_format_character(character) for character in value)


def _strip_format_characters(value: str) -> str:
    """Remove Unicode format controls and line separators from diagnostics."""
    return "".join(
        character for character in value if not _is_format_character(character)
    )


def _mark_diagnostic_controls(value: str) -> str:
    """Preserve control boundaries while scanning a Bearer label/value."""
    return "".join(
        "\x1f"
        if _PATH_CONTROL_CHARACTERS.match(character)
        or _is_format_character(character)
        else character
        for character in value
    )


_CREDENTIAL_PLACEHOLDER_VALUES = frozenset(
    (
        "change-me",
        "changeme",
        "example",
        "fake",
        "insert-value",
        "not-a-secret",
        "not_a_secret",
        "placeholder",
        "redacted",
        "replace-me",
        "sample",
        "your-api-key",
        "your-api-token",
        "your-password",
        "your-token",
        "your_api_key",
        "your_api_token",
        "your_password",
        "your_token",
    )
)


def _is_credential_placeholder(value: str | None) -> bool:
    """Return whether a credential assignment is an explicit documentation stub."""
    if value is None:
        return True
    stripped = value.strip()
    if re.match(
        r"^[\x60'\"](?:\.{2,}|…+|\*+|_+)[\x60'\"](?=$|[\s)\]},.;:])",
        stripped,
    ):
        return True
    unwrapped = stripped
    if (
        len(unwrapped) >= 2
        and unwrapped[0] in "'\x60\""
        and unwrapped[-1] == unwrapped[0]
    ):
        unwrapped = unwrapped[1:-1]
    if re.fullmatch(r"\x24\{[^{}\r\n]+\}", unwrapped):
        return True
    if re.fullmatch(r"\{\{[^{}\r\n]+\}\}", unwrapped):
        return True

    normalized = stripped.strip(chr(96) + "'\".,;:()[]{}<>")
    if not normalized:
        return True
    if normalized.casefold() in _CREDENTIAL_PLACEHOLDER_VALUES:
        return True
    if re.fullmatch(r"(?:\.{2,}|…+|\*+|_+)", normalized):
        return True
    return False


def _pattern_has_live_secret(
    category: str, pattern: re.Pattern[str], text: str
) -> bool:
    """Return whether a pattern identifies a non-placeholder secret."""
    if category == "Bearer token":
        return pattern.search(_mark_diagnostic_controls(text)) is not None
    if category != "credential assignment":
        return pattern.search(text) is not None
    return any(
        not _is_credential_placeholder(match.group("assignment_value"))
        for match in pattern.finditer(text)
    )


def _safe_diagnostic(value: object) -> str:
    """Return bounded UTF-8-safe text with secret-like values redacted."""
    try:
        text = str(value).encode("utf-8", errors="ignore").decode("utf-8")
    except Exception:
        return "<diagnostic unavailable>"

    # Preserve control boundaries for Bearer matching first.  Then remove
    # controls before the remaining patterns, so provider tokens split by
    # allowed LF/CR/tab content cannot evade detection.
    marked_text = _mark_diagnostic_controls(text)
    for category, pattern in _SECRET_PATTERNS:
        if category == "Bearer token":
            marked_text = pattern.sub(
                lambda match: (
                    f"{match.group('bearer_label')} "
                    "<redacted:Bearer token>"
                ),
                marked_text,
            )
            break
    text = _PATH_CONTROL_CHARACTERS.sub("", marked_text)
    text = _strip_format_characters(text)
    for category, pattern in _SECRET_PATTERNS:
        if category == "Bearer token":
            continue
        if category == "credential assignment":
            def redact_assignment(match: re.Match[str]) -> str:
                if _is_credential_placeholder(
                    match.group("assignment_value")
                ):
                    return match.group(0)
                return (
                    f"{match.group('assignment_key')}"
                    f"{match.group('assignment_separator')}"
                    "<redacted:credential assignment>"
                )

            text = pattern.sub(redact_assignment, text)
        else:
            text = pattern.sub(f"<redacted:{category}>", text)
    if len(text) > _MAX_DIAGNOSTIC_CHARS:
        text = (
            text[:_MAX_DIAGNOSTIC_CHARS - len(_DIAGNOSTIC_TRUNCATION)]
            + _DIAGNOSTIC_TRUNCATION
        )
    return text



_PATH_DIAGNOSTIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\|//|/)"
    r"[^\s<>\"'|,;]+"
)
_PATH_QUOTED_DIAGNOSTIC_PATTERN = re.compile(
    r"(?P<quote>[\"'])(?P<path>(?:[A-Za-z]:[\\/]|\\\\|//|/)[^\"'\r\n]*)(?P=quote)"
)


def _path_diagnostic_text(path: object) -> str | None:
    try:
        text = str(path)
    except Exception:
        return None
    return text if text else None


def _safe_exception_diagnostic(
    value: object, known_paths: tuple[object, ...] = ()
) -> str:
    """Sanitize exception text while retaining explicitly requested paths."""
    try:
        text = str(value)
    except Exception:
        return "<diagnostic unavailable>"

    protected: dict[str, str] = {}
    for index, path in enumerate(known_paths):
        known = _path_diagnostic_text(path)
        if known is None or known not in text:
            continue
        marker = f"__dsh_known_path_{index}__"
        protected[marker] = known
        text = text.replace(known, marker)

    def replace_path(match: re.Match[str]) -> str:
        candidate = match.group(0)
        core = candidate.rstrip(".,;:!?)]}")
        if not core:
            return candidate
        return "<path>" + candidate[len(core):]

    def replace_quoted_path(match: re.Match[str]) -> str:
        return f"{match.group('quote')}<path>{match.group('quote')}"

    text = _PATH_QUOTED_DIAGNOSTIC_PATTERN.sub(replace_quoted_path, text)
    text = _PATH_DIAGNOSTIC_PATTERN.sub(replace_path, text)
    for marker, known in protected.items():
        text = text.replace(marker, known)
    return _safe_diagnostic(text)


def _safe_cli_line(value: object) -> str:
    """Return a diagnostic safe for a newline-terminated CLI line."""
    return _safe_diagnostic(value)[: _MAX_DIAGNOSTIC_CHARS - 1]


def _write_cli_text(
    message: str,
    *,
    stream: object = None,
    end: str = "\n",
    json_text: bool = False,
) -> None:
    """Write CLI text without allowing an output encoding error to escape."""
    if stream is None:
        stream = sys.stdout
    try:
        encoding = getattr(stream, "encoding", None)
        if encoding:
            message.encode(encoding)
        stream.write(message + end)
        return
    except (LookupError, UnicodeError):
        if json_text:
            try:
                message = json.dumps(
                    json.loads(message), ensure_ascii=True, indent=2
                )
            except (TypeError, ValueError, UnicodeError):
                pass
        encoding = getattr(stream, "encoding", None) or "ascii"
        try:
            fallback = message.encode(
                encoding, errors="backslashreplace"
            ).decode(encoding)
            stream.write(fallback + end)
        except (LookupError, UnicodeError):
            try:
                stream.write("<diagnostic unavailable>" + end)
            except (LookupError, UnicodeError):
                pass


def _safe_value_error(message: object) -> ValueError:
    """Return a ValueError whose complete diagnostic is bounded and safe."""
    return ValueError(_safe_diagnostic(message))


class StagingError(ValueError):
    """Raised when staging would escape, hit policy, or overwrite silently."""

    def __init__(self, message: object = "", *args: object) -> None:
        safe_args = tuple(_safe_exception_diagnostic(arg) for arg in args)
        super().__init__(_safe_diagnostic(message), *safe_args)

    def __str__(self) -> str:
        return _safe_diagnostic(super().__str__())


def is_protected_output(path: Path) -> bool:
    """Return True for protected names and conservative Windows aliases."""
    path = Path(path)

    def normalized(component: str) -> str:
        if _WIN:
            # Win32 normalizes trailing dots and spaces in path components.
            return component.rstrip(" .").casefold()
        return component.lower()

    if normalized(path.name) in PROTECTED_FILENAMES:
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

    Explicit lstat keeps metadata failures visible so an indeterminate
    inspection cannot be treated as a safe path.
    """
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except (ValueError, NotADirectoryError) as exc:
        raise StagingError(
            "refusing to stage because a path component is invalid: "
            f"{_safe_diagnostic(path)} ({_safe_exception_diagnostic(exc)})"
        ) from None
    except OSError as exc:
        raise StagingError(
            "refusing to stage because a path component could not be "
            "inspected for symlink protection: "
            f"{_safe_diagnostic(path)} ({_safe_exception_diagnostic(exc)})"
        ) from None
    return stat.S_ISLNK(metadata.st_mode)


def _reparse_component(path: Path) -> Path | None:
    """Return the first symlink, junction, or other reparse component.

    Only a genuinely absent component is accepted as a new output. Every
    other metadata error fails closed. The path is expected to be absolute.
    """
    current = Path(path.anchor)
    parts = path.parts[len(current.parts):]
    for part in parts:
        current = current / part
        symlink_state = _symlink_component(current)
        if symlink_state:
            return current
        if symlink_state is None:
            # Once a component is genuinely absent, all later components are
            # absent too; do not continue into a path that cannot exist.
            return None
        if not _WIN:
            continue
        try:
            attributes, error = _windows_file_attributes(current)
        except Exception as exc:
            raise StagingError(
                "refusing to stage because a path component could not be "
                "inspected for reparse protection: "
                f"{_safe_diagnostic(current)} ({_safe_exception_diagnostic(exc)})"
            ) from None
        if attributes == _INVALID_FILE_ATTRIBUTES:
            if error in (_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND):
                return None
            raise StagingError(
                "refusing to stage because a path component could not be "
                "inspected for reparse protection: "
                f"{_safe_diagnostic(current)} "
                f"(Windows error: {_safe_diagnostic(error)})"
            )
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            return current
    return None


def _destination_metadata(path: Path) -> tuple[bool, os.stat_result | None]:
    """Classify an existing destination without fail-open existence checks."""
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        return False, None
    except ValueError as exc:
        raise StagingError(
            "invalid staging path (embedded NUL or invalid value): "
            f"{_safe_diagnostic(path)}"
        ) from None
    except OSError as exc:
        raise StagingError(
            "refusing to stage because destination metadata could not be "
            "safely inspected: "
            f"{_safe_diagnostic(path)} ({_safe_exception_diagnostic(exc)})"
        ) from None
    return True, metadata


def _check_staging(path: Path, replace: bool) -> Path:
    """Validate and return the absolute lexical destination to write.

    The check is conservative and path-based. It rejects protected names and
    aliases, Windows alternate streams, symlinks/junctions/reparse points,
    indeterminate metadata, hard-linked regular files, and all other existing
    non-regular destinations. A concurrent replacement after this check is
    not prevented. Callers must use the returned normalized path rather than
    the caller's original spelling, which may contain dot segments.
    """
    try:
        raw_path = str(path)
        if "\x00" in raw_path:
            raise StagingError(
                "invalid staging path (embedded NUL): "
                f"{_safe_diagnostic(path)}"
            )
        requested = Path(os.path.abspath(path))
        stream_component = _alternate_stream_component(requested)
        if stream_component is not None:
            raise StagingError(
                "refusing to stage through a Windows alternate data stream: "
                f"{_safe_diagnostic(requested)} "
                f"(stream component: {_safe_diagnostic(stream_component)})"
            )
        escape = _reparse_component(requested)
        if escape is not None:
            raise StagingError(
                "refusing to stage through a symlink or junction path: "
                f"{_safe_diagnostic(requested)} "
                f"(reparse component: {_safe_diagnostic(escape)})"
            )
        try:
            resolved = Path(os.path.realpath(requested))
        except (OSError, ValueError) as exc:
            raise StagingError(
                "refusing to stage because destination resolution could not "
                "be safely inspected: "
                f"{_safe_diagnostic(requested)} ({_safe_exception_diagnostic(exc)})"
            ) from None
        if is_protected_output(requested) or is_protected_output(resolved):
            raise StagingError(
                "refusing to stage a protected path "
                "(SOUL.md, AGENTS.md, CLAUDE.md, rules/**): "
                f"{_safe_diagnostic(path)}"
            )
        if _is_within_or_different_drive(
            resolved, _REPOSITORY_ROOT
        ) and not _is_within_or_different_drive(resolved, _LIVE_STAGING_ROOT):
            raise StagingError(
                "refusing to write inside the live repository; use "
                ".skillopt-sleep/staging for staged outputs: "
                f"{_safe_diagnostic(path)}"
            )
        exists, metadata = _destination_metadata(requested)
        if exists and metadata is not None:
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1:
                raise StagingError(
                    "refusing to stage through an unsafe existing destination "
                    f"(non-regular or hard-linked): {_safe_diagnostic(requested)}"
                )
            if not replace:
                raise StagingError(
                    "refusing to overwrite existing file without --replace: "
                    f"{_safe_diagnostic(path)}"
                )
    except StagingError:
        raise
    except OSError as exc:
        raise StagingError(
            "refusing to stage because destination path could not be "
            f"resolved: {_safe_diagnostic(path)} ({_safe_exception_diagnostic(exc)})"
        ) from None
    except (TypeError, ValueError) as exc:
        raise StagingError(
            "invalid staging path (embedded NUL or invalid value): "
            f"{_safe_diagnostic(path)} "
            f"({_safe_exception_diagnostic(exc, (path,))})"
        ) from None
    except Exception as exc:
        raise StagingError(
            "invalid staging path (embedded NUL or invalid value): "
            f"{_safe_diagnostic(path)} "
            f"({_safe_exception_diagnostic(exc, (path,))})"
        ) from None
    return requested


def _canonical_staging_path(path: Path) -> str:
    """Return a platform-aware canonical spelling for collision checks."""
    try:
        absolute = Path(os.path.abspath(path))
        resolved = Path(os.path.realpath(absolute))
    except Exception as exc:
        raise StagingError(
            "refusing to stage because destination could not be safely "
            "resolved: "
            f"{_safe_diagnostic(path)} "
            f"({_safe_exception_diagnostic(exc, (path,))})"
        ) from None
    return os.path.normcase(str(resolved))


def _is_within(path: Path, root: Path) -> bool:
    """Compare canonical absolute paths while callers retain no-follow checks."""
    try:
        path_name = os.path.normcase(
            os.path.realpath(os.path.abspath(os.fspath(path)))
        )
        root_name = os.path.normcase(
            os.path.realpath(os.path.abspath(os.fspath(root)))
        )
        return os.path.commonpath([path_name, root_name]) == root_name
    except Exception as exc:
        raise StagingError(
            "refusing to compare staging path safely: "
            f"{_safe_diagnostic(path)} "
            f"({_safe_exception_diagnostic(exc, (path, root))})"
        ) from None


def _is_within_or_different_drive(path: Path, root: Path) -> bool:
    """Return false for a different-drive path without weakening _is_within."""
    try:
        path_name = os.path.normcase(
            os.path.realpath(os.path.abspath(os.fspath(path)))
        )
        root_name = os.path.normcase(
            os.path.realpath(os.path.abspath(os.fspath(root)))
        )
    except Exception as exc:
        raise StagingError(
            "refusing to compare staging path safely: "
            f"{_safe_diagnostic(path)} "
            f"({_safe_exception_diagnostic(exc, (path, root))})"
        ) from None
    if os.path.splitdrive(path_name)[0].casefold() != os.path.splitdrive(
        root_name
    )[0].casefold():
        return False
    return _is_within(path, root)


def _path_component_has_secret(component: str) -> bool:
    """Return whether a path component matches a likely-secret pattern."""
    return any(
        _pattern_has_live_secret(category, pattern, component)
        for category, pattern in _SECRET_PATTERNS
    )


def _absolute_input_path(path: Path, label: str) -> Path:
    """Make an input path absolute without exposing load diagnostics."""
    # Keep the controlled role label short enough that the useful exception
    # reason survives the final composed diagnostic bound.
    label = _safe_diagnostic(label)[:128]
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except Exception as exc:
        raise _safe_value_error(
            f"cannot load {label} path: {_safe_exception_diagnostic(exc)}"
        ) from None


def _preflight_tree(root: Path, label: str) -> None:
    """Fail closed on unsafe tree entries before any linter or diff read.

    Diagnostics intentionally identify only the static tree role.  Absolute
    roots and untrusted path values are not safe diagnostic material because
    caller-controlled names can contain secrets or control characters.
    """
    label = _safe_diagnostic(label)
    absolute = _absolute_input_path(root, label)
    current = Path(absolute.anchor)
    parts = absolute.parts[len(current.parts):]
    for index, part in enumerate(parts):
        if _path_component_has_secret(part):
            raise _safe_value_error(f"unsafe {label} path may contain a secret")
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise _safe_value_error(f"missing {label} root") from None
        except (NotADirectoryError, ValueError) as exc:
            raise _safe_value_error(f"unsafe {label} root component") from None
        except OSError as exc:
            raise _safe_value_error(f"unsafe {label} root metadata") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise _safe_value_error(
                f"unsafe {label} path is a symlink or reparse point"
            )
        if _WIN:
            try:
                attributes, error = _windows_file_attributes(current)
            except (AttributeError, OSError, ValueError) as exc:
                raise _safe_value_error(
                    f"unsafe {label} reparse metadata"
                ) from None
            if attributes == _INVALID_FILE_ATTRIBUTES:
                raise _safe_value_error(
                    f"unsafe {label} reparse metadata "
                    f"(Windows error: {error})"
                )
            if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise _safe_value_error(
                    f"unsafe {label} path is a symlink or reparse point"
                )
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise _safe_value_error(
                f"unsafe {label} path component is not a directory"
            )
        if index == len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise _safe_value_error(f"unsafe {label} root is not a directory")

    ignored = {directory.casefold() for directory in _IGNORED_DIRS}
    pending = [absolute]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise _safe_value_error(
                f"unsafe {label} directory metadata"
            ) from None
        for entry in entries:
            path = Path(entry.path)
            if _path_component_has_secret(entry.name):
                raise _safe_value_error(f"unsafe {label} path may contain a secret")
            try:
                metadata = os.lstat(path)
            except FileNotFoundError as exc:
                raise _safe_value_error(
                    f"unsafe {label} path disappeared during preflight"
                ) from None
            except (NotADirectoryError, ValueError) as exc:
                raise _safe_value_error(
                    f"unsafe {label} path metadata"
                ) from None
            except OSError as exc:
                raise _safe_value_error(
                    f"unsafe {label} path metadata"
                ) from None
            if stat.S_ISLNK(metadata.st_mode):
                raise _safe_value_error(
                    f"unsafe {label} path is a symlink or reparse point"
                )
            if _WIN:
                try:
                    attributes, error = _windows_file_attributes(path)
                except (AttributeError, OSError, ValueError) as exc:
                    raise _safe_value_error(
                        f"unsafe {label} reparse metadata"
                    ) from None
                if attributes == _INVALID_FILE_ATTRIBUTES:
                    raise _safe_value_error(
                        f"unsafe {label} reparse metadata "
                        f"(Windows error: {error})"
                    )
                if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                    raise _safe_value_error(
                        f"unsafe {label} path is a symlink or reparse point"
                    )
            if stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink > 1:
                    raise _safe_value_error(
                        f"unsafe {label} path is a hard-linked regular file"
                    )
            elif stat.S_ISDIR(metadata.st_mode):
                if entry.name.casefold() not in ignored:
                    pending.append(path)
            else:
                raise _safe_value_error(
                    f"unsafe {label} path is a special file"
                )


def _preflight_staging(
    paths: list[Path],
    replace: bool,
    protected_roots: tuple[Path, ...] = (),
) -> None:
    """Validate outputs and their canonical boundary before input traversal."""
    canonical: list[str] = []
    for path in paths:
        validated = _check_staging(path, replace)
        for protected_root in protected_roots:
            if _is_within_or_different_drive(validated, protected_root):
                # The designated live staging area is the sole exception when a
                # protected root (commonly the repository/evidence root)
                # encloses it.  Input roots elsewhere remain fail-closed.
                if (
                    _is_within_or_different_drive(
                        validated, _LIVE_STAGING_ROOT
                    )
                    and _is_within_or_different_drive(
                        _LIVE_STAGING_ROOT, protected_root
                    )
                ):
                    continue
                raise StagingError(
                    "refusing to stage an output under an input or evidence root: "
                    f"{_safe_diagnostic(path)}"
                )
        canonical.append(_canonical_staging_path(validated))
    if len(set(canonical)) != len(canonical):
        raise StagingError(
            "refusing to stage report and diff at the same destination"
        )


def _tree_display_path(path: Path, root: Path) -> str:
    """Return a relative, log-safe path label for tree diagnostics."""
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return "<outside-tree>"
    if (
        _PATH_CONTROL_CHARACTERS.search(relative)
        or _contains_format_character(relative)
    ):
        return "<unsafe-path>"
    return _safe_diagnostic(relative)


def _read_tree(root: Path) -> dict[str, str]:
    """Collect relative-path -> text after a no-follow tree preflight."""
    absolute = _absolute_input_path(root, "diff tree")
    _preflight_tree(absolute, "diff tree")
    files: dict[str, str] = {}
    ignored = {directory.casefold() for directory in _IGNORED_DIRS}
    pending = [absolute]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            display_directory = _tree_display_path(directory, absolute)
            raise _safe_value_error(
                "unsafe diff tree directory metadata: "
                f"{display_directory}"
            ) from None
        for entry in entries:
            path = Path(entry.path)
            display_path = _tree_display_path(path, absolute)
            if (
                _PATH_CONTROL_CHARACTERS.search(entry.name)
                or _contains_format_character(entry.name)
            ):
                raise _safe_value_error(
                    "unsafe diff tree path contains control characters: "
                    "<unsafe-path>"
                )
            try:
                metadata = os.lstat(path)
            except FileNotFoundError as exc:
                raise _safe_value_error(
                    "unsafe diff tree path disappeared during traversal: "
                    f"{display_path}"
                ) from None
            except OSError as exc:
                raise _safe_value_error(
                    f"unsafe diff tree path metadata: {display_path}"
                ) from None
            if stat.S_ISLNK(metadata.st_mode):
                raise _safe_value_error(
                    "unsafe diff tree path is a symlink or reparse point: "
                    f"{display_path}"
                )
            if _WIN:
                try:
                    attributes, error = _windows_file_attributes(path)
                except (AttributeError, OSError, ValueError) as exc:
                    raise _safe_value_error(
                        f"unsafe diff tree reparse metadata: {display_path}"
                    ) from None
                if attributes == _INVALID_FILE_ATTRIBUTES:
                    raise _safe_value_error(
                        f"unsafe diff tree reparse metadata: {display_path} "
                        f"(Windows error: {error})"
                    )
                if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                    raise _safe_value_error(
                        "unsafe diff tree path is a symlink or reparse point: "
                        f"{display_path}"
                    )
            if stat.S_ISDIR(metadata.st_mode):
                if entry.name.casefold() not in ignored:
                    pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink > 1:
                    raise _safe_value_error(
                        "unsafe diff tree path is a hard-linked regular file: "
                        f"{display_path}"
                    )
                rel = path.relative_to(absolute).as_posix()
                try:
                    with path.open("rb") as handle:
                        data = handle.read(_MAX_DIFF_FILE_BYTES + 1)
                except OSError as exc:
                    raise _safe_value_error(
                        "unsafe diff tree file could not be read: "
                        f"{display_path}"
                    ) from None
                if len(data) > _MAX_DIFF_FILE_BYTES:
                    raise _safe_value_error(
                        "unsafe diff tree file exceeds the size limit: "
                        f"{display_path}"
                    )
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise _safe_value_error(
                        "unsafe diff tree file is not valid UTF-8: "
                        f"{display_path}"
                    ) from None
                if (
                    _CONTROL_CHARACTERS.search(text)
                    or _contains_format_character(text)
                ):
                    raise _safe_value_error(
                        "unsafe diff tree file contains control characters: "
                        f"{display_path}"
                    )
                # Scan a control-stripped copy as well as the content. LF, CR,
                # and tab are valid file text controls, but removing them before
                # matching prevents a wrapped provider token from bypassing the
                # secret detector and then appearing in the rendered diff.
                secret_scan_text = _PATH_CONTROL_CHARACTERS.sub("", text)
                marked_scan_text = _mark_diagnostic_controls(text)
                for category, pattern in _SECRET_PATTERNS:
                    scan_text = (
                        marked_scan_text
                        if category == "Bearer token"
                        else secret_scan_text
                    )
                    if _pattern_has_live_secret(category, pattern, scan_text):
                        raise _safe_value_error(
                            "unsafe diff tree file may contain a secret "
                            f"({category}): {display_path}"
                        )
                files[rel] = text
            else:
                raise _safe_value_error(
                    f"unsafe diff tree path is a special file: {display_path}"
                )
    return files


def _load_linter():
    """Load the existing read-only linter module by file path."""
    spec = importlib.util.spec_from_file_location(
        "_agent_context_linter", _LINTER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load linter module: {_safe_diagnostic(_LINTER_PATH)}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lint_issues(root: Path, evidence_root: Path) -> list[str]:
    """Run the repository linter against the given tree (policy included)."""
    try:
        linter = _load_linter()
        return linter.lint_tree(
            root, include_policy=True, evidence_root=evidence_root
        )
    except Exception as exc:
        raise _safe_value_error(
            f"context static check failed: {_safe_exception_diagnostic(exc)}"
        ) from None


@dataclass(frozen=True)
class ProposalReport:
    baseline_ok: bool
    candidate_ok: bool
    regressions: tuple[str, ...]
    improvements: tuple[str, ...]
    diff: str


def _validate_proposal_impl(baseline: Path, candidate: Path, root: Path) -> ProposalReport:
    """Run static/context checks and return a non-adopting report.

    The baseline tree is linted as-is; the candidate is linted against policy
    coverage with ``root`` resolving evidence paths, so proposed evidence must
    already exist in the real repository.
    """
    baseline = _absolute_input_path(baseline, "baseline")
    candidate = _absolute_input_path(candidate, "candidate")
    root = _absolute_input_path(root, "evidence root")
    # Preflight every untrusted input before the public linter or diff reader
    # gets a chance to open anything. These checks are lexical/no-follow.
    _preflight_tree(baseline, "baseline")
    _preflight_tree(candidate, "candidate")
    _preflight_tree(root, "evidence root")

    baseline_issues = sorted(_lint_issues(baseline, baseline))
    candidate_issues = sorted(_lint_issues(candidate, root))
    baseline_set = set(baseline_issues)
    candidate_set = set(candidate_issues)
    raw_regressions = candidate_set - baseline_set
    raw_improvements = baseline_set - candidate_set
    regressions = tuple(sorted(_safe_diagnostic(issue) for issue in raw_regressions))
    improvements = tuple(
        sorted(_safe_diagnostic(issue) for issue in raw_improvements)
    )
    return ProposalReport(
        baseline_ok=not baseline_issues,
        candidate_ok=(
            not candidate_issues
            or (not raw_regressions and bool(raw_improvements))
        ),
        regressions=regressions,
        improvements=improvements,
        diff=render_diff(baseline, candidate),
    )


def validate_proposal(baseline: Path, candidate: Path, root: Path) -> ProposalReport:
    """Validate a proposal without exposing unexpected public-API errors."""
    try:
        return _validate_proposal_impl(baseline, candidate, root)
    except Exception as exc:
        raise _safe_value_error(
            f"proposal validation failed: {_safe_exception_diagnostic(exc)}"
        ) from None


def _render_diff_impl(baseline: Path, candidate: Path) -> str:
    """Return a unified diff for user review."""
    def public_path(value: object, role: str) -> Path:
        try:
            return Path(value)
        except TypeError:
            raise _safe_value_error(
                f"cannot load {role} path: invalid path value"
            ) from None
        except Exception as exc:
            raise _safe_value_error(
                f"cannot load {role} path: {_safe_exception_diagnostic(exc)}"
            ) from None

    baseline_files = _read_tree(public_path(baseline, "baseline"))
    candidate_files = _read_tree(public_path(candidate, "candidate"))
    lines: list[str] = []
    for rel in sorted(set(baseline_files) | set(candidate_files)):
        before = baseline_files.get(rel)
        after = candidate_files.get(rel)
        if before == after:
            continue
        before_lines = (before if before is not None else "").splitlines(keepends=True)
        after_lines = (after if after is not None else "").splitlines(keepends=True)
        lines.extend(difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        ))
    return "".join(lines)


def render_diff(baseline: Path, candidate: Path) -> str:
    """Render a diff without exposing unexpected public-API errors."""
    try:
        return _render_diff_impl(baseline, candidate)
    except Exception as exc:
        raise _safe_value_error(
            f"diff rendering failed: {_safe_exception_diagnostic(exc)}"
        ) from None


class _SafeArgumentParser(argparse.ArgumentParser):
    """Keep argparse errors bounded and free of caller-controlled values."""

    def _print_message(self, message: str, file: object = None) -> None:
        stream = sys.stderr if file is None else file
        _write_cli_text(str(message), stream=stream, end="")

    def error(self, message: str) -> None:
        diagnostic = _safe_diagnostic(f"{self.prog}: error: {message}")
        # Reserve one character for the terminating newline so the complete
        # emitted diagnostic remains within the shared 512-character bound.
        diagnostic = diagnostic[: _MAX_DIAGNOSTIC_CHARS - 1]
        self.exit(2, f"{diagnostic}\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = _SafeArgumentParser(
        prog="validate_context_proposal",
        description=(
            "Validate a proposed context change: lint baseline and candidate "
            "(policy files included), classify regressions vs improvements, "
            "and stage a report/diff whose evidence paths must already exist "
            "under the repository root given by --root. This tool cannot "
            "adopt live policy and never calls skillopt-sleep adoption; "
            "SOUL.md, AGENTS.md, CLAUDE.md, and rules/** can only be changed "
            "by an explicit human decision."
        ),
    )
    parser.add_argument(
        "--baseline", required=True, help="baseline context tree directory"
    )
    parser.add_argument(
        "--candidate", required=True, help="candidate context tree directory"
    )
    parser.add_argument(
        "--root", default=".",
        help="repository root that evidence paths in the candidate must resolve against",
    )
    parser.add_argument(
        "--output-report",
        help="explicit staged report path (off by default; refuses existing files without --replace)",
    )
    parser.add_argument(
        "--output-diff",
        help="explicit staged unified diff path (independent of --output-report; same replace rules)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="allow overwriting an existing staged report/diff file",
    )
    return parser.parse_args(argv)


_PROTECTED_MESSAGE = (
    "error: refusing to write a protected policy or persona file after path "
    "resolution (SOUL.md, AGENTS.md, CLAUDE.md, rules/**)"
)


def _validate_report_payload(payload: dict) -> None:
    """Reject malformed or live-secret report data before emission."""
    if not isinstance(payload, dict):
        raise StagingError("proposal report payload is not an object")
    for field in ("baseline_ok", "candidate_ok"):
        if not isinstance(payload.get(field), bool):
            raise StagingError(f"proposal report field {field} is not boolean")
    for field in ("regressions", "improvements"):
        values = payload.get(field)
        if not isinstance(values, (tuple, list)):
            raise StagingError(f"proposal report field {field} is not a sequence")
        for value in values:
            if not isinstance(value, str):
                raise StagingError(f"proposal report field {field} contains non-text data")
            if _contains_live_secret(value):
                raise StagingError("proposal report contains a live secret")
    diff = payload.get("diff")
    if not isinstance(diff, str):
        raise StagingError("proposal report field diff is not text")
    if _contains_live_secret(diff):
        raise StagingError("proposal report contains a live secret")


def _contains_live_secret(text: str) -> bool:
    """Return whether report text contains a non-placeholder secret."""
    return any(
        _pattern_has_live_secret(category, pattern, text)
        for category, pattern in _SECRET_PATTERNS
    )


def _serialize_report(payload: dict, *, trailing_newline: bool) -> str:
    """Serialize a report and reject text that cannot be UTF-8 encoded."""
    try:
        _validate_report_payload(payload)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if trailing_newline:
            text += "\n"
        text.encode("utf-8")
    except StagingError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise StagingError(
            "proposal report contains text that cannot be encoded as UTF-8: "
            f"{_safe_exception_diagnostic(exc)}"
        ) from None
    except Exception as exc:
        raise StagingError(
            "proposal report serialization failed: "
            f"{_safe_exception_diagnostic(exc)}"
        ) from None
    return text


def _encode_staged_content(content: str) -> bytes:
    """Encode staged text before any destination can be opened."""
    try:
        return content.encode("utf-8")
    except UnicodeError as exc:
        raise StagingError(
            "staged output contains text that cannot be encoded as UTF-8"
        ) from None


def _stage(path: Path, content: str, replace: bool) -> None:
    try:
        checked_path = _check_staging(Path(path), replace)
        encoded = _encode_staged_content(content)
        checked_path.parent.mkdir(parents=True, exist_ok=True)
        checked_path.write_bytes(encoded)
    except StagingError:
        raise
    except UnicodeError:
        raise StagingError(
            "staged output could not be encoded as UTF-8"
        ) from None
    except Exception as exc:
        raise StagingError(
            "staged output could not be written safely: "
            f"{_safe_diagnostic(path)} "
            f"({_safe_exception_diagnostic(exc, (path,))})"
        ) from None


def main(argv: list[str] | None = None) -> int:
    """Default to a dry-run; write only explicitly requested staged outputs."""
    args = _parse_args(argv)
    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    evidence_root = Path(args.root)
    requested_outputs = [
        Path(value)
        for value in (args.output_report, args.output_diff)
        if value
    ]
    try:
        # Validate requested output destinations before any untrusted input
        # traversal or linter/diff read.
        _preflight_staging(
            requested_outputs,
            args.replace,
            protected_roots=(baseline, candidate, evidence_root),
        )
        report = validate_proposal(baseline, candidate, evidence_root)
    except Exception as exc:
        known_paths = (baseline, candidate, evidence_root)
        _write_cli_text(
            _safe_cli_line(
                f"error: {_safe_exception_diagnostic(exc, known_paths)}"
            ),
            stream=sys.stderr,
        )
        return 2

    try:
        payload = asdict(report)
        report_text = _serialize_report(
            payload, trailing_newline=bool(args.output_report)
        )
        if args.output_report:
            _stage(
                Path(args.output_report),
                report_text,
                args.replace,
            )
            _write_cli_text(
                _safe_cli_line(
                    f"staged report (non-adopting): {args.output_report}"
                )
            )
        if args.output_diff:
            _stage(Path(args.output_diff), report.diff, args.replace)
            _write_cli_text(
                _safe_cli_line(
                    f"staged diff (non-adopting): {args.output_diff}"
                )
            )
    except Exception as exc:
        known_paths = (
            baseline,
            candidate,
            evidence_root,
            *requested_outputs,
        )
        _write_cli_text(
            _safe_cli_line(
                f"error: {_safe_exception_diagnostic(exc, known_paths)}"
            ),
            stream=sys.stderr,
        )
        return 2
    if not args.output_report and not args.output_diff:
        _write_cli_text(report_text, end="", json_text=True)
        _write_cli_text(
            _safe_cli_line(
                "staged: no file written (dry-run); pass --output-report/--output-diff "
                "to stage into an approved proposal directory only"
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
