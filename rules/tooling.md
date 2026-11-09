---
id: policy/tooling
status: active
scope: Any agent executing repository work through a turn-based tool harness.
applies_when: Every session that performs discovery, edits, builds, or starts background work.
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-10-06"
supersedes: []
evidence:
  - AGENTS.md
  - rules/gpu.md
  - docs/agent/OPERATIONS.md
---

# Tool-use economy

A turn (one assistant generation) is the expensive unit; its cost grows with
accumulated context. Tool calls within a turn are nearly free by comparison.
Therefore: **minimize turns, maximize useful calls per turn.**

## Batching

1. Any turn holding exactly one tool call must either batch more calls into
   it or state in one line why the call must be serial (a dependency or a
   verification gate).
2. Front-load discovery: the first action turn carries every independent
   read, grep, glob, and status check that is foreseeable. Do not trickle
   files one per turn.
3. Read once, act many: [discover + read everything] -> [apply all
   edits/writes/commands] -> [verify]. Three batched turns beat ten mixed
   ones.
4. Independent calls go in one message; dependent work chains inside a
   single tool program where possible (one script performing A->B->C
   instead of three tool turns).
5. When a file tail/anchor is needed, read it in one generous offset window
   inside the discovery batch. If an edit anchor is rejected, re-read and
   retry in the same next program, not as a standalone probe turn.
6. Slice reads are fine; "slicing" never means reading twice to compensate
   for a narrow first slice.

## Output hygiene

7. Read only what is needed (offset/limit, targeted grep over full-file
   dumps); oversized results persist in context for the whole session.
8. Never echo large tool output back in prose; summarize conclusions,
   errors, and deltas only.
9. Do not re-read a file already read this session unless it changed since.
10. Do not re-derive facts already persisted in state files (ENVIRONMENT
    facts, MEMORY.md lessons); read them, do not rediscover them.

## Background and long-running work

11. Launch anything expected to take minutes (builds, training, data prep)
    as a background job immediately; take the job id and move on.
12. Never poll in a loop or sleep-wait. Collect a background job's output
    only on completion notification, or when the next action is genuinely
    blocked (then one wait call, not a spin loop).
13. While a background job runs: work on independent tasks or end the turn
    and wait; a silent wait is cheaper than idle polls.
14. Poll rarely, with a long interval, only when correctness requires
    monitoring intermediate progress.

## Delegation

15. Any self-contained subtree that would take several turns goes to a
    subagent in the background; the parent pays one result turn and the
    child's internal turns never grow the parent's context.
16. Multiple independent subtrees are spawned in one message, staggered.

## Anti-patterns (each wastes a turn)

- One read per turn for files known to be needed.
- Speculative "let me check..." probes with no concrete information need.
- Echoing tool results into narration before the next call.
- Manually polling a background job when completion notifications exist.
- Stacked calls whose results do not influence each other.
- Re-grepping something already found this session.
- Corrective re-reads (anchor misses, too-narrow slices, recomputing an
  index a generously-sized earlier read already covered).

Budget guide: a normal task targets ~5 turns (discover -> act -> verify,
plus one fix turn); planning-heavy or multi-phase tasks are exempt. Every
session ends with a one-line self-audit: was anything independent
serialized?

## Harness preconditions and text-channel hazards
Two mechanical rules, each learned by losing a turn:

- An edit requires a prior read of that exact file in the current session.
  Batch the reads for every file about to be edited before the edits, and
  re-read after any context compaction, which drops that state.
- Text passed through a program to a shell or a template literal is
  escaped, not literal. Never place raw backticks, dollar-brace forms, or
  unquoted nested quotes inside such a string; a value intended for the
  shell must be built so the outer language cannot consume it first.
