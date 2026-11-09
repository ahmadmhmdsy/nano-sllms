---
id: policy/experiments
status: active
scope: "Every experiment: registration, run control, metric extraction, verdict writing."
applies_when: The task plans, launches, evaluates, or closes an experiment on any component line.
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-10-06"
supersedes: []
evidence:
  - research/EXPERIMENTS.md
  - AGENTS.md
  - state/tasks.yaml
  - docs/agent/SCHEMA.md
---

# Experiments policy

## Never believe info alone; always test the hypothesis
Analysis, code reading, prior experiments, hearsay, intuition, and even
strong literature claims are only hypothesis generators. The default at
this task scale (single consumer GPU, small corpora, cheap smokes) is to
run the smallest real test on the actual data/model/combination and let
the measured artifact decide. Every architectural claim, dataset claim,
or "this usually works" suggestion must either cite a measured artifact
from THIS repo or be explicitly marked untested. Reading a config and
asserting a weakness is a hypothesis, not a verdict; the weakness
becomes knowledge only after a paired arm (with/without) on real data.

## Pre-register before launching
A run is armed only after a pre-registered record exists: the question, the
arms, the per-arm gates, the metric source, the GPU-window estimate, and the
evidence paths the run will produce. A run launched before its registration
exists is not "an experiment that happened anyway"; it is a BLOCKED run that
must be re-run inside its registered window.

## Anchor evaluator
Report results against the designated anchor — the run the line's existing
verdicts were measured against (for the D line, the deployed gold
checkpoint). The anchor changes only through the experiment lifecycle,
never implicitly or per-session.

## Source-parse metrics
Metrics come from parsing source artifacts (registered gate-bench outputs,
metrics JSON, checkpoint metadata, TensorBoard events). Never quote a
console-only number that has no artifact behind it, and never re-derive a
metric after the fact when the artifact records it.

## Hard-fail on overlap
If a provisional arm's input slices overlap data or items the anchor already
consumed, that is a **hard failure** of the registration (verdict BLOCKED),
not a footnote in the report. Overlap destroys the registration's clean
read; treat it as a launch error.

## Smoke must prove updates
A smoke claim is valid only when the smoke run demonstrably advanced the
update path — loss moved, a checkpoint or shard was written, metrics landed.
An exit code alone is not a smoke verdict. Every smoke conclusion cites the
artifact that advanced.

## Honest partial verdicts
When some rungs pass and others fail, the verdict is partial with each
sub-verdict listed; never round up to accepted. Unverifiable or
instrument-broken measurements are instrument_unknown, not guesses.

## Config declares intent; the log proves execution
A config states what was requested; only the run's own log shows what
executed. Before reasoning from any arm, confirm its mechanism actually
fired — warm-start tensor load counts (nonzero, not silently zero),
tokenizer phase, context length, resume step, checkpoint path. A run whose
declared mechanism did not fire is a control, not the arm it claims to be,
and every comparison built on it inherits the error. Make a silent no-op
fatal rather than a warning.

## Selection is not measurement
Choosing the best of several probes is a selection, not a measurement of
the arm. Report the matched-step number beside the selected best, and where
the claim matters, test the selection on a set that took no part in it. A
best-of-N over probes that oscillate inside a noise band will always
produce a favourable number; that number describes the selection procedure,
not the model.
