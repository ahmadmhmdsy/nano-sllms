# <E-NN> - <short title> (TASKS row N)

Copy this file to research/<yyyy-mm-dd>_<name>.md when starting a new
experiment report. Fill every section; FAIL/SKIPPED verdicts get reasons
(CLAUDE s10). Close-out checklist when the experiment settles:
(1) this report finished, (2) row appended to research/EXPERIMENTS.md,
(3) research/WHAT_WORKS.md levers updated if the verdict changes
what-future-training-should-do, (4) new gotchas -> MEMORY.md lessons,
(5) TASKS row status. Weights: local per AGENTS s5; commit only the
small summary artifacts.

## Question
One sentence: what does this experiment decide?

## Hypothesis
Falsifiable prediction + the gate numbers that would falsify it.

## Rig
- Data (source, tokenization, blocks train/val, ctx)
- Model(s) + params (student/teacher; frozen? dtype?)
- Arms (names, one-line description each, what is held IDENTICAL)
- Budget (steps, batch/accum, lr schedule, optimizer, seed, GPU, wall-clock target)

## Results table
Wall-clock is the headline column. Numbers from train_summary.json /
trainer_state.json curves / eval_report.json - cite the file for every table,
and NEVER silently merge different eval sources (see MEMORY 47).

| arm | params | wall-clock | in-run eval (cite source) | independent eval (cite source) | ppl | verdict |
|---|---|---|---|---|---|---|

## Gates and verdicts
Each pre-registered gate: ACTUAL vs REQUIRED -> PASS / FAIL with reason.
Divergence/stop rules as executed (did the cooperative stop fire?).

## Levers to carry
Concrete, transferable knobs proven here, with numbers + conditions:
what a future training run at this or larger scale SHOULD adopt.
Empty is a valid answer, not a shame.

## Mistakes and gotchas
What went wrong, root cause, the fix, and the MEMORY lesson it became
(link lesson numbers). Include anything a fresh agent would mis-spend
an hour rediscovering.

## Scale-transfer caveats
Honesty bounds: at which scales is this claim licensed? What must be
re-measured before extrapolating?
