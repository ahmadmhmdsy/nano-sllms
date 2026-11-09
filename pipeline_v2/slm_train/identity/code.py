"""Deterministic V2 code identity over the source set and uv.lock (fail closed)."""
import pathlib
from pathlib import Path

import slm_train

from slm_train.contracts.hash import sha256_bytes, sha256_file
from slm_train.identity.source import SourceIdentity


_CODE_SKIPS = {"__pycache__", ".pytest_cache", ".venv", "node_modules"}


def code_identity(repository_root: Path) -> SourceIdentity:
    """Hash the V2 source/schema/contract tree plus uv.lock.

    Every required input fails closed: a missing uv.lock is never silently
    omitted, because omitting it silently changes the code identity.
    """
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {repository_root}")
    package_root = root / "pipeline_v2" / "slm_train"
    if not package_root.is_dir():
        installed = pathlib.Path(slm_train.__file__).resolve().parent
        if not installed.is_relative_to(root):
            raise ValueError(
                f"V2 source tree (pipeline_v2/slm_train) not found under: {root}")
        package_root = installed
    files: "list[Path]" = []
    for path in sorted(package_root.rglob("*")):
        if any(part in _CODE_SKIPS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    file_digests: "dict[str, str]" = {}
    for path in files:
        file_digests[path.relative_to(root).as_posix()] = sha256_file(path)
    lock = root / "uv.lock"
    if not lock.is_file():
        raise FileNotFoundError(f"uv.lock is required for code identity and is missing: {lock}")
    file_digests["uv.lock"] = sha256_file(lock)
    ordered = sorted(file_digests.items())
    source_digest = sha256_bytes(
        "".join(f"{p}\x00{h}\n" for p, h in ordered).encode("utf-8"))
    return SourceIdentity(schema_version=1, kind="code", root_hash=source_digest, file_digests=file_digests)
