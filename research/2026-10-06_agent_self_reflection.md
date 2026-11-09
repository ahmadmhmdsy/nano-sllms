# Self-reflection — 2026-10-06, after the verdict audit and E-070…E-072

Written because the audit found that a wrong verdict costs more than a failed run:
it propagates. Most of what follows is therefore about my own error patterns, not
about the models.

## What I got wrong

1. **I trusted a config's claim over the run's log.** E-065 F2's config said it
   warm-started; it actually trained from random init, and I built a comparison on
   top of that for a while. The check was always available — the load counts are in
   the log (`tensors_in / missing / unexpected`). *Fix:* read the load counts
   before reasoning from any warm-started arm; the D-12 guard now makes a zero-tensor
   load fatal, which is the same lesson turned into a mechanism.
2. **I inherited an inverted metric direction without re-deriving it.** A resolved
   config had the registered direction flipped; anything derived from it is
   sign-flipped. *Fix:* read the authority that defines the metric, not prose about
   it. Now recorded at the top of the audit ledger.
3. **I accepted several PASSes that had no artifact behind them** (E-38a, E-45,
   E-54c/d, E-63c, E-63s S3-best). *Fix:* a verdict without an exact artifact path is
   not a verdict. `rules/evidence.md` already said this; I was not applying it.
4. **I read a best-of-run minimum as if it were a measurement.** E-070's headline came
   from one probe of twelve on a curve oscillating in a ~5.7 % band. I flagged the
   caveat, but flagging is not testing. E-072 was the test it should have had from the
   start: a held-out set that took no part in the selection. *Fix:* every selection
   claim gets a matched-step number AND, where possible, an independent-set check.
5. **I nearly "fixed" a frozen historical file.** I was about to tidy the duplicate
   experiment ids in `research/EXPERIMENTS.md` — a file whose own banner says the
   duplicates are deliberate. Reading the banner first would have saved the detour.
   *Fix:* check a file's self-declared authority before editing it.
6. **I once killed a CPU job on a contention theory that then failed to replicate.**
   The rate barely moved (0.184 → 0.177 s/step), so the theory was refuted by the very
   measurement I took to justify it. *Fix:* measure the contention before acting on it.
7. **I overstated a conclusion and had to walk it back.** After E-069 I recorded that
   C4's problem was budget. E-070 vindicated the *under-trained* half and refuted the
   *whole story* half: past step 15000 the extra 45 k steps bought nothing. The
   correction is in the registry rather than quietly dropped. *Fix:* state which part
   of a claim the evidence actually covers.
8. **Tool-use errors cost me three round-trips this session** — a backtick inside a TS
   template literal, a dollar-brace interpolation that JS ate before pwsh saw it, and
   three edits to files I had not read this session. None were conceptual; all were
   avoidable. *Fix:* batch reads before edits; never put shell/JS metacharacters raw
   inside a template literal.

## What I got right, and should keep doing

- **Pre-registering the rule before running.** E-071's WIN/PARTIAL/FALSIFIED
  thresholds, its non-degeneracy test, and its ctx-256 scope limit were all fixed
  before the run. That is why "WIN" is a result rather than a narrative.
- **Reproducing the baseline inside the same harness.** E-071's single pass had to
  re-derive the recorded 0.374792 (it did, |diff| 0.00008) before the vote number was
  admissible. This is cheap and it is the difference between a gain and a harness
  difference.
- **Recording the counter-finding.** E-072 corroborated the selection on the
  registered metric — and I recorded that the ordering *inverts* on
  `der_canonical` and macro_f1 rather than flattening it into "corroborated".
- **Correcting myself in the record.** The E-069 → E-070 self-correction is written
  into the registry, not hidden.

## Standing personal rules from this

1. Read the log, not the config.
2. Read the authority that defines the metric, not prose about it.
3. No verdict without an artifact path.
4. A selection is not a measurement — test it off the selection set.
5. Check a file's authority banner before editing it.
6. Measure a contention before acting on it.
7. Say which part of a claim the evidence covers.
8. Batch reads before edits; never raw metacharacters in a template literal.
