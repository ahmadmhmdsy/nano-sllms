---
id: policy/evidence
status: active
scope: Every claim of fact that lands in a report, record, ledger, reflection, or reply.
applies_when: The task writes or updates any record, report, ledger row, or evidence pointer.
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-10-06"
supersedes: []
evidence:
  - docs/agent/SCHEMA.md
  - docs/superpowers/specs/2026-09-24-agent-context-reliability-design.md
  - state/tasks.yaml
---

# Evidence policy

## Exact relative paths
Evidence pointers are exact repo-root-relative paths with forward slashes,
chosen so the linter can verify existence. Never store a bare line number as
the only pointer, and never format evidence as an absolute or
machine-specific path.

## Source artifact authority
Where a rendered summary and the underlying source artifact disagree, the
source artifact wins. Before quoting a measurement, open the artifact itself
(metrics JSON, gate-bench output, TensorBoard events, checkpoint metadata)
and parse it directly.

## No remembered numbers
Never quote a metric from memory. If the artifact cannot be opened this
session, the value's status is UNKNOWN, not "approximately what was seen
earlier". Cached numbers outside their source window are re-read or
re-registered before they are cited again.

## Stable IDs
Record IDs (tasks, experiments, decisions, lessons, rules) are namespaced and
never change after creation. Consolidations and renames use supersedes and
aliases, not a renamed record. Reusing a stable ID for a different fact is a
defect.

## Definition of done
A task is `done` only when all of these hold:

- The requested behavior is implemented and matches the nearest scoped
  conventions.
- Inputs are validated and expected errors are handled explicitly.
- Existing functionality is preserved (no unrelated changes ride along).
- The relevant validation command from the component's scoped `AGENTS.md`
  was run, and its result was read back from the artifact, not from
  console output.
- Every claim carries an exact relative evidence path; unavailable checks
  are reported `SKIPPED` with a reason.
- Documentation touched by the change is updated.
- No secrets were introduced; the final diff was reviewed.
- Known limitations are reported alongside the label, not hidden.

If any condition fails, the task stays `in_progress` or `blocked` — never
claim `done` on partial validation. `done` plus a material limitation is a
defect; report the limitation and keep work open instead.

## No machine-specific encoding
Evidence paths never encode machine-local absolute locations, credentials, or
live process identifiers. If a path embeds those, treat the path itself as
suspect and re-derive it from the workspace root.

## Metric direction comes from the defining code
Every metric carries a direction, and the direction is a property of the
code that computes it, not of prose about it. Before quoting or comparing a
metric, read the function or registry that defines it and record whether
higher or lower is better. A direction inherited from a summary, a resolved
config, or a previous session's wording is UNKNOWN until re-derived. An
inverted direction silently converts every later comparison into its
opposite, so it is a defect in the record, not a nuance.

## A recorded verdict with no artifact is a defect
A verdict row (PASS/FAIL/accepted/refuted) whose supporting artifact cannot
be produced is itself a defect in the record, independent of whether the
original conclusion was right. It may be wrong in either direction: a
recorded FAIL may have scored a slice or a misread number (a false FAIL),
and a recorded PASS may have no measurement behind it at all (a false PASS).
Correct it in place and label it, rather than leaving an artifactless claim
standing.

## Scope the conclusion to the evidence
State the exact part of a claim the artifact covers, and no more. "The arm
was under-trained" and "budget was the whole story" are different claims;
an artifact that supports the first does not support the second. When a
later run refutes part of an earlier conclusion, amend the record in place
so both the claim and its correction stay visible.
