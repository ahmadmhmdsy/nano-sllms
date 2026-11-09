# Reflection policy — task closeouts

Status: stable policy. Owner: core (this file). Storage: `docs/reflections/`.

## Why reflections exist

A closeout reflection is a short, deterministic record of what a task
requested, what actually happened, and what a later agent should reuse or
avoid. It exists so the next session inherits evidence instead of memory.

## Hybrid persistence policy

The policy is deliberately hybrid to keep repository noise low:

1. **Every non-trivial task gets a short closeout reflection.** A few
   sentences per section is enough; the generator prints a draft without
   writing anything by default.
2. **Durable reflection files are created in `docs/reflections/` only when
   the closeout includes any of:**
   - a failure or a near-miss worth recording,
   - a reusable lesson (and, only when explicitly requested with
     `--stage-lesson`, a lesson candidate in `state/lessons_pending.json`),
   - an evidence correction (a previously claimed result that turned out
     wrong or unverifiable),
   - a process change other agents must follow.
3. **Routine successful tasks do not create a repository file.** The
   dry-run draft may be discarded; only the task registry updates.

## Rules

- The generator is offline and deterministic: no model calls, no jobs, no
  network, no secret inspection.
- Validation runs before any write; likely-secret patterns are rejected and
  error messages never echo the matched text.
- The CLI preflights the completed draft, the reflection destination, and any
  lesson-state destination before opening a write. The caller-selected input
  payload is validated after it is read; it is not a repository-tree walk.
  Output and state destinations must be distinct, including canonical path
  collisions. When staging is requested, the existing state and the exact
  candidate-augmented state are parsed, serialized, and checked for UTF-8
  serializability, finite values, safe text, strict evidence syntax, and
  no-follow existing evidence targets before either file is opened. Historical
  evidence may be absent, but unsafe aliases, controls, secrets, and metadata
  failures are rejected without rewriting state. Invalid UTF-8, embedded
  non-whitespace control characters, excessive JSON nesting, embedded-NUL
  paths, unavailable metadata, and controlled filesystem errors are reported
  without a traceback. Repository-contained output is restricted to
  `docs/reflections/`; lesson staging is restricted to the canonical
  `state/lessons_pending.json`, so stable policy, generated state, and
  unrelated registries cannot be overwritten by this CLI. An external
  `--state-path` is an isolated-validation fixture only: it is never a live
  lesson registry or an alternative adoption target. Temporary outputs outside
  the repository remain available for isolated validation.
- Evidence pointers in a payload are exact forward-slash repository-root-relative
  paths. Real regular files and directories under the repository root are valid
  artifacts; lexical empty/dot/dot-dot components, backslashes or mixed
  separators, URI, drive, UNC, or alternate-stream forms, surrounding
  whitespace, Windows trailing-dot/space aliases, symlink/junction/reparse
  components, existing multi-link files, and metadata failures are rejected before
  evidence is accepted. Evidence targets are not followed.
- Output is opt-in: `--write` requires an explicit `--output` path, and an
  existing reflection output is never overwritten without `--replace`.
- Protected paths are refused under every code path: `SOUL.md`, `AGENTS.md`,
  `CLAUDE.md`, and anything under `rules/**` are never accepted as
  reflection or lesson-state destinations. The preflight also rejects every
  symlink, junction, or other reparse component; Windows alternate-stream
  paths; and existing multi-link files. Reparse and file-metadata inspection
  fails closed; only genuinely missing path components are treated as new
  outputs. On Windows, protected-name checks conservatively strip trailing
  dots and spaces, so aliases such as `rules..`, `rules `, and `SOUL.md.`
  are refused before directory creation. This is a conservative path-based
  preflight, not a handle-based no-follow or atomic-write guarantee: a
  concurrent replacement after validation remains outside the guarantee.
- On Windows, reparse components below the path anchor are refused even when
  the target looks safe, including junctions and other reparse placeholders;
  use a workspace path without reparse components for writes.
- Lesson staging is opt-in: `--stage-lesson` requires `--write`, touches only
  the canonical `state/lessons_pending.json` path, rewrites that state file in
  place, is idempotent for the same persisted candidate identity, and never
  promotes a candidate; promotion happens only through the gated rule path in
  [rules/README.md](../../rules/README.md).
- The output and lesson-state writes are sequential and non-transactional.
  Either write can be partial, and a later failure can leave earlier output or
  state; the error is reported, but no rollback is attempted.

## Usage

```powershell
# Dry run (default): prints the draft, writes nothing
& .\.venv\Scripts\python.exe scripts/closeout_reflection.py --input <payload.json>

# Durable write (opt-in)
& .\.venv\Scripts\python.exe scripts/closeout_reflection.py --input <payload.json> --write --output docs/reflections/<date>-<slug>.md

# Also stage a lesson candidate (requires --write)
& .\.venv\Scripts\python.exe scripts/closeout_reflection.py --input <payload.json> --write --output docs/reflections/<date>-<slug>.md --stage-lesson
```

Payload fields, size limits, and the lesson-candidate schema are documented
in [docs/reflections/README.md](../reflections/README.md).
