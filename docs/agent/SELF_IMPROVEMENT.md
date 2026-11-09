# Self-improvement proposals (SkillOpt-Sleep boundary)

This document is the staged, non-adopting boundary for any self-improvement
loop over the agent's own context files. It defines what an automated run
may produce and where the hard stop is.

## What an automated cycle may do

1. Harvest only task reflections and approved session evidence (no secrets,
   no raw user data).
2. Mine recurring failure and success patterns deterministically.
3. Replay a bounded set of tasks against a snapshot of the current context
   ("baseline") and against a proposed change ("candidate").
4. Run the static/context linter
   ([scripts/check_agent_context.py](../../scripts/check_agent_context.py))
   against both trees via
   [scripts/validate_context_proposal.py](../../scripts/validate_context_proposal.py).
5. Classify lint results as regressions versus improvements.
6. Stage up to the explicitly requested artifacts: a proposal report and/or a
   unified diff.

## What an automated cycle may never do

- edit SOUL.md, AGENTS.md, CLAUDE.md, or anything under rules/;
- call skillopt-sleep adoption or any optimizer adoption path;
- change validation/GPU policy or a user-approved decision;
- use an optimizer as an authority over policy;
- launch training, evaluation, inference, probes, downloads, or servers.

Adoption is always an explicit human decision that follows the gated
promotion path in [rules/README.md](../../rules/README.md): repeated
evidence, bounded diff, validation, explicit user adoption, post-adoption
verification, and a backup taken before any adoption.

## First-phase run profile

- Backend: deterministic mock only; no real model budget is spent.
- Edit budget: a bounded maximum number of changed hunks per proposal.
- Scores: baseline and candidate lint outcomes (issues found / gate pass)
  are reported side by side so the human sees both numbers.
- Held-out validation: a small set of scenarios not used for mining
  verifies the candidate before staging.
- Staged output location: an explicit, approved proposal directory. The
  non-adopting proposal validator writes nothing without explicit output
  paths. Before either report or diff is opened, it preflights every requested
  destination and rejects protected names and lexical aliases, Windows
  alternate streams, symlink/junction/reparse components, fail-closed metadata
  or resolution errors, hard-linked and other non-regular existing files, and
  canonical report/diff collisions. Containment comparisons canonicalize both
  paths, so Windows short-name aliases cannot bypass an input/evidence-root ban.
  It refuses silent overwrites (use --replace only for an explicitly approved
  existing staged file).
- Baseline and candidate context trees are untrusted inputs. Before linting or
  diffing, a no-follow walk rejects symlink, junction, reparse, hard-linked
  regular-file, special-file, and indeterminate-metadata entries; generated and
  sensitive stores are excluded from descendant traversal while their entry
  points are still inspected. Diff collection is bounded and fails closed on
  unreadable/non-UTF-8 files, likely secret or control content, and oversized
  artifacts, so matched content is never copied into a report or diff or echoed
  in an error. Evidence pointers are exact forward-slash repository-root-relative
  paths, while real in-root directories remain valid artifacts; traversal,
  alias, and link components are rejected. Requested outputs are preflighted
  before input reads: proposal outputs in the live repository are restricted to
  the designated `.skillopt-sleep/staging/` tree, and reflection outputs are
  restricted to `docs/reflections/`; lesson staging may update only the
  canonical `state/lessons_pending.json`. An external `--state-path` is only
  an isolated-validation fixture, not a live lesson registry or adoption target;
  it must not be copied into canonical state. An explicitly approved staging
  directory remains available.
- **Residual limitations:** both validators use path-based checks rather than
  handle-based no-follow or atomic-write guarantees. A concurrent replacement
  after validation remains outside the guarantee. The proposal validator
  writes report and diff sequentially after the shared preflight; a filesystem
  failure during the second write can still leave the first staged artifact.
  Portability of case-insensitive non-Windows filesystems is also not claimed.
  See [REFLECTION.md](REFLECTION.md) for the closeout preflight details.
- Backup before adoption: any adopted diff first has a recorded backup
  snapshot of the touched files.
- SOUL.md evolution is disabled in the first phase; no automatic persona
  revision is staged to adopt automatically.

## Report schema

The proposal report is JSON with the frozen fields: baseline_ok,
candidate_ok, regressions, improvements, and diff (see
scripts/validate_context_proposal.py for exact types).
