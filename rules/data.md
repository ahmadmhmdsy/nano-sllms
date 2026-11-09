---
id: policy/data
status: active
scope: Every dataset, tokenizer, checkpoint, log, and derived-artifact operation in this repository.
applies_when: The task creates, regenerates, re-chunks, re-encodes, cleans, or deletes any data or artifact.
owner: repository-owner (manual policy; changes require explicit user adoption)
last_reviewed: "2026-09-24"
supersedes: []
evidence:
  - AGENTS.md
  - docs/agent/OPERATIONS.md
  - state/environment.json
---

# Data policy

## Preserve source data first
User-supplied source data is immutable. Never delete it, overwrite it in
place, or "clean it up" as a side effect of an unrelated job. If space is
needed, report first; deletion happens only along an explicitly approved
boundary.

## Bounded chunking
When chunking a corpus, derive the chunk size from the corpus's own metadata
and state it in the record. Never chunk with an unbounded default or with a
count derived from memory. Chunk boundaries are recorded where the data
consumer can read them.

## Regeneration must be reproducible
Every derived artifact carries the exact venv-only command (and config) that
regenerated it. If the command is unknown, the artifact's lineage is
UNKNOWN — it is never guessed. Regeneration rebuilds into a fresh path or
via atomic replacement, never silently in place.

## Atomic and mmap-safe writes
Packed-shard and index writes go through a temp file plus an atomic rename
onto the final name. Never write in place over a shard a running job
currently has mmapped; if replacement is unavoidable, do it only when the
reader is verified idle. Streaming appenders record their own offsets, and a
restarted writer re-checks the written tail before appending.

## Never delete user data
Never delete, truncate, or overwrite user-owned data, checkpoints, or eval
artifacts without an explicit, reversible, consent-backed step. Report
before deleting anything, even when it looks stale.
