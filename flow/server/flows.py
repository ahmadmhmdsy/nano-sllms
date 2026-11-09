"""flows/ directory store for saved .flow.json documents (flow/ MVP Task 4).

flow/flows/ holds <slug>.flow.json files and is GIT-TRACKED (user decision
Q2, plan "File structure (locked)"). This module is the only writer/reader
of that directory; the repo's flow/flows/ is used when flows_dir is omitted
(tests pass a tempfile directory and never touch the repo one).

API contract (fixed by Task 4):

- save(name, g, flows_dir=None) -> pathlib.Path
    Gathers errors from BOTH validators first —
    flow.server.graph_schema.validate(g) for the document shape and
    flow.server.nodes.validate_props(kind, props) per node (Task-3
    reviewer-noted wiring) — and, if any, raises ValueError with the
    reasons newline-joined (NOT a partial write: nothing is touched on
    disk until validation passes). The name must be a slug: lowercase
    [a-z0-9-]{1,64}; any name containing a slash or ".." is rejected (no
    path traversal; backslashes too, enforced in code). On success writes flows/<slug>.flow.json and fsyncs
    the file before returning OK (data-preservation rule) — the written
    Path is returned.

- load(name, flows_dir=None) -> dict
    Returns the parsed JSON document, or raises FileNotFoundError with a
    clean message naming the flow (no traversal names get here: they are
    rejected with ValueError first).

- list_flows(flows_dir=None) -> list[str]
    Sorted slugs; [] when the directory does not exist yet.
"""

import json
import os
import re
import tempfile
from pathlib import Path

from flow.server import graph_schema, nodes

# flow/server/flows.py -> flow/flows/ (git-tracked user decision Q2).
FLOWS_DIR = Path(__file__).resolve().parent.parent / "flows"

_SUFFIX = ".flow.json"
_SLUG_RE = re.compile(r"^[a-z0-9-]{1,64}$")


def _validate_slug(name) -> str:
    """Return the validated slug, or raise ValueError with the reason."""
    if not isinstance(name, str):
        raise ValueError("flow name: expected a string, got %r" % (name,))
    # Belt-and-braces traversal checks; the slug regex below already
    # forbids every character these contain, but state the rule explicitly
    # so the error names the traversal when that is what the caller sent.
    if "/" in name or chr(92) in name or ".." in name:
        raise ValueError(
            "flow name: %r contains a path separator or '..' — "
            "path traversal is not allowed" % (name,)
        )
    if not _SLUG_RE.match(name):
        raise ValueError(
            "flow name: %r is not a valid slug — lowercase [a-z0-9-]{1,64} "
            "required" % (name,)
        )
    return name


def collect_validation_errors(g) -> "list[str]":
    """Run BOTH validators (document schema + per-node registry props)."""
    errors = list(graph_schema.validate(g))
    if isinstance(g, dict):
        graph = g.get("graph")
        if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
            for node in graph["nodes"]:
                if not isinstance(node, dict):
                    continue  # graph_schema already reported the shape error
                kind = node.get("kind")
                nid = node.get("id", "?")
                for reason in nodes.validate_props(kind, node.get("props")):
                    errors.append("graph.nodes[%s]: %s" % (nid, reason))
    return errors


def _dir(flows_dir) -> Path:
    return Path(flows_dir) if flows_dir is not None else FLOWS_DIR


def atomic_write(path, text):
    """Atomically (re)write `path` with `text` (UTF-8, LF newlines).

    The single shared writer for the directory stores (flows.save and
    the app.py model-graph endpoint): payload first goes to a temp file
    IN the target dir (name derived from the target name minus its
    dotted suffixes, e.g. demo.flow.json -> "demo-<rand>.tmp"), is
    fsynced durable there, then os.replace() swaps it into place —
    os.replace is atomic on Windows and POSIX, and on any crash the
    previous (or no) file remains exactly as it was. On failure the
    temp file is unlinked best-effort (no .tmp residue) and the
    original error propagates. Parent directories are created when
    missing. Returns the written Path.
    """
    path = Path(path)
    target_dir = path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    name_base = path.name
    for suffix in reversed(path.suffixes):
        if name_base.endswith(suffix):
            name_base = name_base[: -len(suffix)]
    fd, tmp_path = tempfile.mkstemp(prefix=name_base + "-", suffix=".tmp",
                                    dir=str(target_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())  # data-preservation rule: durable before OK
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass  # temp cleanup is best-effort; the original error propagates
        raise
    return path


def save(name, g, flows_dir=None):
    """Validate then write flows/<slug>.flow.json; fsync; return the Path.

    Raises ValueError (newline-joined reason list from BOTH validators, or
    a slug error) without writing anything on violation.
    """
    slug = _validate_slug(name)
    errors = collect_validation_errors(g)
    if errors:
        raise ValueError("\n".join(errors))

    target = _dir(flows_dir)
    path = target / (slug + _SUFFIX)
    payload = json.dumps(g, indent=2, ensure_ascii=False) + "\n"
    return atomic_write(path, payload)


def load(name, flows_dir=None):
    """Return the parsed .flow.json document dict for slug `name`.

    Raises ValueError on a traversal/non-slug name and a clean
    FileNotFoundError (message names the flow) when it does not exist.
    """
    slug = _validate_slug(name)
    path = _dir(flows_dir) / (slug + _SUFFIX)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("flow %r not found (looked for %s)" % (name, path))


# Backward-compat alias (the public name is collect_validation_errors).
_collect_validation_errors = collect_validation_errors


def delete_flow(name, flows_dir=None) -> Path:
    """Remove flows/<slug>.flow.json; FileNotFoundError when absent.

    The name is slug-validated with the same contract as save/load (no
    path traversal is possible), and a clean FileNotFoundError naming the
    flow is raised when it does not exist. Returns the deleted Path.
    """
    slug = _validate_slug(name)
    path = _dir(flows_dir) / (slug + _SUFFIX)
    try:
        path.unlink()
    except FileNotFoundError:
        raise FileNotFoundError("flow %r not found (looked for %s)" % (name, path))
    return path


def list_flows(flows_dir=None):
    """Sorted slugs of the saved flows; [] when the directory is absent."""
    target = _dir(flows_dir)
    if not target.is_dir():
        return []
    return sorted(
        p.name[: -len(_SUFFIX)]
        for p in target.iterdir()
        if p.is_file() and p.name.endswith(_SUFFIX)
    )
