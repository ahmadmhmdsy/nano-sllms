
# Agent context schema — record fields and vocabularies

Canonical schemas for the agent context files. This document is the
authoritative reference for record fields and status vocabularies. The
tooling that validates them lives in `scripts/check_agent_context.py`
and the tests under `tests/agent_context/`.

## Generic conventions

- Record fields are keyed by stable namespaced IDs (`scope/name`). IDs
  never change after creation; history moves into `supersedes` and
  `aliases` lists, not a renamed record.
- Timestamps are ISO dates (`YYYY-MM-DD`). Where a live fact (machine,
  process, ownership, free VRAM, disk) cannot be verified this session, the
  recorded value is deliberately UNKNOWN, never an inherited old value.
- Evidence pointers are exact repo-root-relative paths (forward slashes),
  and the linter verifies they exist.

## Task records

Required record fields: id (namespaced stable ID), scope, title, status,
owner (UNKNOWN where unverified), created_at, updated_at, last_verified_at
(timestamp or UNKNOWN), next_action (one concrete action or none),
evidence (list of exact relative artifact paths), supersedes (prior IDs
replaced by this record), aliases (old IDs retained for migration).

## Experiment records

Required record fields: id (namespaced stable ID, for example
`diacritizer/E-062`), scope, status, question (the pre-registered
question), verdict, owner, dates (created/started/closed), evidence
(exact artifact paths), report (narrative report path), supersedes.

## Decision records

Required record fields: id, scope, status, decision, rationale, date,
evidence, supersedes.

## Lesson candidates

Required record fields: id, scope, status, lesson, source_reflection,
evidence, promoted_to. A lesson is not an active rule merely because it was
written down; promotion into `rules/` follows the review-gated policy in
`rules/context.md`.

## Context exception registry

'state/context_exceptions.yaml' (schema_version 1) is the single place where a
historical context finding may be accepted. It is an exact-match registry, not
a suppression list: a record is matched only when the linter produces the same
path, line, kind, and link target. A changed, shifted, new, or unlisted finding
still fails, and matching is never by substring.

Required record fields:

- id — namespaced stable ID (`context-exception/name`), unique in the file.
- status — exactly `accepted_historical`.
- path — repo-relative source path of the finding.
- line — 1-based line number of the finding.
- kind — one of the allowed kinds below.
- target — repo-relative link target; required for `broken_markdown_link` and
  forbidden for `malformed_table_row`.
- reason — short, non-secret explanation of why the finding is accepted.
- accepted_at — ISO date (`YYYY-MM-DD`) of the decision.
- decision_source — exactly `direct-user-instruction`.
- evidence — list of exact repo-relative artifact paths that must exist.

Allowed kinds:

- `malformed_table_row` — a legacy table row in a historical ledger.
- `broken_markdown_link` — a historical link whose target was never created.

Validation fails closed. A missing file, an unreadable or unsafe path, a
malformed YAML document, an unknown field, a wrong status, a bad date, a
duplicate id, or a duplicate finding key all make the registry accept
nothing and report a controlled issue instead. Accepted findings are printed
with the `ACCEPTED HISTORICAL:` prefix and do not affect the exit status; only
unaccepted findings do. Record-only mode (`--no-policy`) never applies the
registry, so the raw historical findings stay visible there.

## Status vocabularies

- Task status: pending / in_progress / blocked / done (the linter's
  canonical registry checks accept these four; the design vocabulary
  additionally records needs_user_decision for user-gated items).
- Experiment status: planned / running / closed / blocked.
- Experiment verdict: accepted / rejected / partial / instrument_unknown /
  none.
- Rule status: draft / active / superseded / retired.
- Decision status: proposed / active / superseded / revoked.
- Lesson status: observed / candidate / validated / promoted / rejected.
- Validation labels for reports and replies: PASS / FAIL / SKIPPED /
  BLOCKED / NEEDS_USER_DECISION.

## Rule file front matter

Every policy rule under `rules/` carries YAML front matter (an opening
`---` fence, the mapping, and a closing `---`) with exactly these
fields:

- id — namespaced stable ID (`policy/name` or `components/name`)
- status — one of the rule-status vocabulary values above
- scope — what the rule governs
- applies_when — the task class(es) that load the rule
- owner — manual-policy owner; rule changes are user-adoption-gated
- last_reviewed — the date the content was last reviewed
- supersedes — rule IDs this rule replaces (may be an empty list)
- evidence — a non-empty list of exact relative artifact paths the rule's
  content is grounded in

The index `rules/index.yaml` (schema_version 1) lists every rule file
exactly once, relative to `rules/`, and is linted to point only at
existing files.
