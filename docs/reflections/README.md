# Reflections archive

Closeout reflections from individual tasks live in this directory, one file
per reflection, named `YYYY-MM-DD-<slug>.md`. Start from
[_template.md](_template.md); the deterministic generator renders the same
shape from a JSON payload.

## What belongs here

Per [docs/agent/REFLECTION.md](../agent/REFLECTION.md), a durable reflection
file is created only when the closeout records a failure or near-miss, a
reusable lesson, an evidence correction, or a process change. Routine
successful tasks do not add files here.

## Payload contract (for `scripts/closeout_reflection.py`)

The input is a JSON object with these fields:

- Required strings: `requested_outcome`, `actual_outcome`,
  `reusable_lesson`, `rule_candidate`, `confidence_limitations`.
- Optional strings: `scope` (defaults to `core`), `source_reflection`.
- Required string lists: `changed_files` and `evidence` must each contain at
  least one item; `validation_performed`, `failures_near_misses`,
  `root_causes`, `fixes`, `open_questions` may be empty lists.
- Limits: strings at most 4000 characters, lists at most 100 items; list
  items must be non-empty strings.
- Secret safety: values matching likely-secret shapes (AWS key ids, private
  key blocks, GitHub-style tokens, credential assignments such as
  `password=...`) are rejected. Error messages name the pattern and field
  but never echo the matched text. Non-whitespace control characters are also
  rejected before a draft or lesson candidate can be written.

## Lesson candidates

`--stage-lesson` is opt-in (it requires `--write`) and appends one entry to
`state/lessons_pending.json` keyed by a deterministic id
(`lesson-<scope>-<hash8>`). The id hashes the persisted candidate identity:
scope, source reflection, reusable lesson, rule candidate, and evidence. The
same identity therefore stages idempotently and never duplicates; changing a
lesson's proposed rule or evidence produces a distinct candidate rather than
silently returning an earlier one. Existing legacy candidate IDs computed
from the pre-extension identity are recognized and reused when the persisted
three-field identity (scope, source reflection, and reusable lesson) plus
evidence match. A legacy record with no optional `rule_candidate` is treated as
compatible with an incoming rule because the historical state cannot
distinguish whether that rule changed; this compatibility choice does not claim
to detect a changed rule that was never persisted. An explicitly persisted rule
mismatch or an evidence mismatch receives a new identity.
Reflection narrative fields that are not stored in the candidate do not affect
the id. Candidates carry
`status: "candidate"` and an empty `promoted_to` list; nothing here promotes a
lesson — adoption follows the gated path in
[rules/README.md](../../rules/README.md).

## Index

The table of contents for this archive is [INDEX.md](INDEX.md). Add one line
per durable reflection file.
