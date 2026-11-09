# Self-Reflection: Failure Analysis & Process Lessons - 2026-09-22

Written by the agent that lived the 2026-09-21/22 span (E-60a -> E-60b -> v3q data loss -> rebuild -> B1 divergence).
Purpose: name what *I* got wrong, not just what the plan got wrong, so the failure modes die here. Feeds the E-60 close and the v2 composite proposal (research/2026-09-22-mex-v2-composite-proposal.md).

## A. Failures I caused or co-caused in this session

### A1. One-metric blindness in my OWN verification harness (worst of the session)
My first three rebuild-proof evals produced 0.0997 and 0.0534 - numbers that contradicted two independent recorded anchors (A3 0.6496, B1 0.0889). The correct process gave the answer: probe the discriminating frame in order (attempt-2 checkpoint on rebuilt val -> 0.5781 vs recorded 0.5771 -> val EXONERATED; the instrument was on trial, not the data).
- Bug 1 - shifted-index protocol: predicted marks at position t against labels at t+1. Runs clean, prints clean, magnitude plausible - nothing flags it except a recorded anchor.
- Bug 2 - string-replace instrument surgery: a path replace missed (spacing/context differed), so an A3-trunk eval silently loaded B1's regressed head. I read that number as truth TWICE.
- Lesson L1 (instrument invariant): every ad-hoc eval script must assert stationarity against ONE pre-recorded anchor number before any product judgment. Otherwise the verdict is INSTRUMENT-UNKNOWN, not PASS/FAIL. The probe that found the truth (8 uniform row-chunks 0.650-0.652) is the pattern: evaluate every eval script ON a known anchor.
- Lesson L2 (self-edit invariance): I edited my own probe scripts by blind string-replace across files whose resulting state I did not re-read. Rule: patch -> READ BACK AND DIFF the semantic regions -> THEN launch a multi-minute job. No exceptions.

### A2. Harness arg-shape thrash
Whole turns crashed on a missing required field for pwsh, repeatedly. Lesson: after ONE binding-shape failure in a session, memo-ize the exact successful arg shape and reuse it verbatim; do not re-derive field names from memory.

### A3. Background-job read discipline
Wasted turns wait-polling jobs instead of using the notification stream; once lost the log stream entirely (empty read while running). Lessons: (a) take the job id, continue independent work until notified; (b) if the stream is EMPTY while status is running, assume spill-to-file and check filesystem evidence (checkpoints, log files), never poll in a loop.

### A4. Shared-memory spill: complicit twice
The user caught B1-prime spilling into WDDM shared GPU memory. I had NEVER been per-process monitoring: B1 ran its full 3000 steps silently spilling ~4 GiB into shared memory and no recorded number annotated residency. nvidia-smi per-process memory is [N/A] on this WDDM box; the only reliable split is the .NET GPU Process Memory PerformanceCounters with 2x NextValue samples (first sample is 0).
- Lesson L3 (residency invariant): every train/eval launch states its allocator budget at startup ([e61b] vram cap line); the watchdog scratch/gpu_shared_watch.py runs once per new shape; R7 probe now = shape + residency + throughput.
- Lesson L3b (do not chase the cap): when a capped job OOMs, resize the JOB (micro-batch x accumulation), never the cap. 256x2 == 512x1 EXACTLY for fixed-length rows (equal per-row CE weights); chasing the cap upward hit CUDA-driver OOM twice and wasted three launches.

### A5. Data-loss recovery posture
What went RIGHT: Option-A rebuild from raw reproduced recorded stats EXACTLY (12,702 quarantined / 7,581 deduped / per-source counts byte-consistent), and the rebuild-proof certified eval comparability for BOTH ladder checkpoints before any new launch.
What went wrong upstream, never to recur:
- Lesson L4 (regenerable is NOT backed up): the pack was 'regenerable in principle' but the regen chain had never been exercised end-to-end, so the OOM in the unbounded read_jsonl stayed invisible for weeks until it killed a subagent's box. Every durable artifact records (a) its regen command AND (b) its last successful end-to-end regen timestamp.
- Lesson L5 (RAM-bounded data code): any script reading user corpora loads in CHUNKS with index+length arrays and memmap writes (scratch/rebuild_pack_chunked.py pattern is now house style). Never load a whole corpus as Python objects.
- Lesson L5b (Windows mmap rename lock): mmap-opened files cannot be os.replace/unlinked while views exist - always design the final write step as copyfile-after-close (the patched e22_pack_qcri.py).

### A6. Mechanism-level failures of the MODEL work (recorded honestly, not relitigated)
- Frozen-growth (B1): zero-insert + frozen base = noise a frozen shell cannot compensate; 3000 steps regressed the warm head from 0.65 to 0.09. Structural failure #1.
- Co-train growth (B1-prime): diverged after warmup on unchanged LR 2e-4 (loss 3.03 -> 10.1; step-1000 val 0.0003 with 14/15 classes dead). Structural failure #2. R5 enforced: the recipe is RETIRED, no third variant of the same recipe. Depth-growth noise is not absorbable at this scale/LR/step budget. Verdict final unless user re-gates with a different factor structure.
- Lesson L6 (divergence is a runtime gate): a loss monotonically rising after warmup should trigger the next val eval by 500 and a kill decision in-session - not 2.5 more hours of wall time logging an already-proven collapse.

## B. What I got RIGHT (keep these behaviors)
1. Anchor-first verification: two independent recorded anchors caught the instrument bugs instead of mislabeling data. One known-good number reproduced = trusted frame; one failed = frame on trial.
2. Disk-not-models salvage: the git-history incident lost only regenerable npy because the raws were never deleted and weights were LFS-restored. Keep raws untouched, keep final weights tracked, keep the regen chain exercised.
3. Pre-registration honesty blocked a premature launch (no B1-prime until eval comparability certified).
4. Kill own-jobs cleanly, by id, with reasons logged; never collateral.

## C. Standing process upgrades adopted (immediate)

| New process | Why | Where |
|---|---|---|
| Anchored-eval: every gate script asserts vs one recorded anchor number within tolerance before judging | kills instrument false-verdicts (A1) | all future eval/probe scripts |
| Patch-then-read-back: blind string-replace forbidden in eval code | A1 bug 2 twice | all script surgery |
| Residency watchdog per new shape: scrape .NET counters dedicated-vs-shared for the actual pid, log scratch/vram_track.csv | VRAM-only user rule | R7 probe now = shape + residency + throughput |
| Micro-batch x accum is gradient-equivalent ONLY for fixed-length rows; re-prove before trusting again for variable-length data | 3 wasted launches | MEMORY.md standing note |
| Divergence watch: loss rising monotonically after warmup -> immediate val eval -> kill at criterion, no full-collapse logging | L6 | ops note + train_e61b follow-up |
| Job-id ledger per session: scrubatch/jobs_date.md; FS evidence over poll loops | A3 | session hygiene |

## D. Do NOT re-derive (already encoded, still binding)
CLAUDE/AGENTS rules; plan M1-M7 mistakes ledger (+ M8 = the VRAM-residency rule from A4); head-through eval protocol; same-index mark alignment; 15-class head; per-class readouts; full metric profile standing policy (E-55c).

## E. Honest 48-hour summary
Model progress: none (Rung A honest FAIL; Rung B structurally retired). Frame progress: data frame now rebuildable and verified; instrument frame anchored and multiplexed; memory frame monitored and policy-locked. Those three frames are what the v2 composite proposal builds on.