# Per-part Python playground: decomposed sub-skill ladders

> RESTORED 2026-10-14 by the Lead. The original text was destroyed by an accidental full-file write.
> Sections 1-5 below are reconstructed faithfully from glm53_flash/playground.py (the source of truth for the
> ladders, tiers and API) and from the build report; sections 6-7 are the original survivors, preserved verbatim.

## 1. Measured motivation

After 6x more reference-SFT exposure the aspect-pool battery went 26/48 -> 32/48, which fixed exactly the three
aspects whose reference body is a SINGLE EXPRESSION (first_or_empty = "text[:1]", contains_zero = "0 in x",
is_palindrome = "x == x[::-1]"). Four aspects stayed at 0/4 at every exposure level:

    grade          multi-branch if/elif
    count_even     comprehension / generator
    count_positive comprehension / generator
    lookup         conditional expression + subscript / dict.get

Each needs a multi-statement or comprehension form. A corrected harvest run on 2026-10-14 (loops AND attributes
enabled, 640 sandbox requests, 0 timeouts) still produced 0/32 for all four, so they are outside the base model's
reach under sampling: pass@k = 0, which means RFT/RL cannot fix them and a reference-style corpus is the only
supported route. This module supplies that corpus, decomposed.

## 2. Files and API

    glm-5.3-flash-from-scratch/glm53_flash/playground.py
    glm-5.3-flash-from-scratch/tests/test_playground.py        (11 tests, PASS)

    PLAYGROUND: dict[str, list[SubSkill]]     construct -> ordered ladder, simplest first, canonical body last
    SubSkill(construct, rung, name, arguments, descriptions, body, generate, tier)
        .key                stable identifier
        .capabilities()     the Capabilities for this rung's tier
    playground_pool(tier="core"|"loops"|"attributes"|"all") -> list[SubSkill]
    make_playground_task(sub_skill, *, split, index, cases=3) -> CodingTask
        (same shape as make_aspect_task: prompt / reference_completion / cases / reference_source)
    select_playground_tasks(...)
    PLAYGROUND_SEEDS = {dev 4507, final 5623, rl 6733, confirm 7841, pretrain 8951}
        - disjoint from ASPECT_SEEDS and from the legacy frozen seeds.

Every expected output is computed by EXECUTING the rung's reference body at generation time (_run_body), never
hardcoded; a test independently re-executes each body and compares it to the stored cases.

## 3. Ladders

27 rungs total. Tiers: core 20, loops 6, attributes 1. Every ladder's LAST rung is byte-identical to the
corresponding canonical body in aspects.py (asserted by test).

### grade (7 rungs, all core)
1 single `if`  2 `if/else`  3 `if/elif`  4 `if/elif/else`  5 chained comparison  6 `and`/`or`
7 the full grade construct

### count_even (7 rungs; rungs 1-3 are loops tier, 4-7 core)
1 explicit loop + counter  2 loop with `if`  3 `+=` accumulator  4 list comprehension
5 `sum(generator)`  6 `sum(generator with if)`  7 the full count_even

### count_positive (7 rungs; rungs 1-3 loops tier, 4-7 core)
same ladder shape as count_even with a different predicate

### lookup (6 rungs; rung 3 is the attributes tier, the rest core)
1 plain subscript  2 membership test  3 `.get(k, default)`  4 ternary
5 conditional expression with subscript  6 the full lookup

## 4. How this corpus would be packed by reference_sft_bytes.py

A parallel `playground(index)` helper calls `make_playground_task` over `playground_pool(tier)` on the held-out
`pretrain` split, yielding the same (prompt, reference_completion) pairs the existing FAMILIES/ASPECTS sources
yield. The packer needs no shape change; the builder itself was deliberately NOT edited during design.

MEASURED 2026-10-14 (Phase 1.1, PASS). scripts/reference_sft_bytes.py now has playground(index) behind
--include-playground (default OFF) and --playground-tier. The flag-off path is byte-identical to the E-122 corpus
(sha256 train 572EED7F...E41840, val 9BDF6424...E756B9, meta.json identical), and the script REFUSES (exit 1) to
write the playground corpus into data/reference/bytes. The new corpus is data/reference/bytes_playground:
train.bin 6,812,468 bytes = 3,406,234 uint16 positions (18,800 examples, of which 10,800 = 27 sub-skills x 400),
val.bin 340,872 bytes (940 examples, 540 playground). Leakage control: 0 of 10,800 dev-split prompts appear in the
new train.bin, while 27/27 pretrain index-0 prompts do. Note this corpus is 3.13x the E-122 corpus (1,086,727
positions), so 600 steps is ~6 epochs here rather than ~18 - a fairer schedule, but not a 1-epoch match.

## 5. Falsifiable predictions

P1 the four constructs reach >= 1/4 on the aspect pool after a reference pass over these rungs
P2 the already-solved constructs do not regress (legacy battery stays 32/32; aspect pool >= 32/48)
P3 the explicit-loop rungs are what unlock the counters: a terminal-rung-only corpus underperforms the full ladder
P4 per-rung scores rise before construct scores do (the ladder is learnable bottom-up)
P5 after the pass, at least one of the four constructs shows pass@k > 0, which is the RL precondition
P6 no rung is learnable only because of leakage: dev prompts and prompt+body pairs stay absent from the corpus

MEASURED BASELINE 2026-10-14 (Phase 0.2, scratch/bench_playground.py, dev, tier all, per-rung 4, 160 bytes,
108 tasks, deterministic - two runs byte-identical after masking seconds):

  arm A (e121ref, 100 steps)  35/108   grade 0/28  count_even 11/28  count_positive 14/28  lookup 10/24
  E-122 best (600 steps)      31/108   grade 0/28  count_even 14/28  count_positive  3/28  lookup 14/24
  48-byte control (E-122)      4/108   <- the old budget erased 27 of 31 real passes

Findings: (a) the 160-byte budget is mandatory - 21 of 27 bodies exceed 48 bytes; (b) grade is 0/28 at EVERY rung
for BOTH arms, failing with "expected an indented block", so its blocker is the indentation syntax, not branch
complexity and not exposure; (c) more exposure is not uniformly better - E-122 trades count_positive (14 -> 3)
for count_even and lookup; (d) the terminal rungs (= canonical aspect bodies already in the corpus) pass 4/4 while
intermediate rungs mostly fail, so the model reproduces seen forms rather than generalising across rung forms -
which is precisely what an all-rungs corpus targets.

<!-- appended -->
## 6. What this design does not claim

- It is not measured: no rung has been shown to unlock anything.
- Case counts are three per task instance, matching the aspect pool; that is enough to
  score correctness, not to estimate generalisation.
- count_positive's terminal rung is a length-of-comprehension, so its ladder revisits the
  comprehension form at the end. That is a measured constraint of the canonical body in
  aspects.py, not a design preference.
- No GPU work is implied here; the corpus builder and these tests are CPU only.

## 7. Implementation order (approved 2026-10-14) - merged from "what we can do now" and the literature

The two lists are one roadmap. The reordering rationale is that a measurement must exist before a change can be
credited, and a reward change is only meaningful once the base can sometimes solve the target.

**Phase 0 - make the frontier measurable (blocks everything).**
  0.1 Playground battery: score every RUNG separately (generalise scratch/bench_aspect_pool.py to the playground
      pool). Graders hardened per the OpenAI RFT checklist: several cases per rung, edge cases included,
      execute-don't-match, never reference-string equality.
  INSTRUMENT NOTE (measured 2026-10-14): the aspect-pool battery's 48-byte completion budget TRUNCATES 21 of the
  27 rung bodies (max 133 bytes: grade/full 133, grade/and_or 124, count_even/loop_if 114). The rung battery must
  therefore run at --budget-bytes 160; the 48-byte reading is kept only as a truncation control. Rung-level and
  aspect-level numbers are then different instruments and must not be compared across batteries.
  0.2 Baseline it: arm A (26/48), E-122 best (32/48) and the 27 rungs individually -> the rung-level baseline
      every later change is measured against. Falsifiable: the four constructs stay 0/4 and the loop-tier rungs
      are 0/4 too (if a loop-tier rung already passes, the construct is reachable and the story changes).

**Phase 1 - teach the constructs (the only supported fix for pass@k = 0).**
  1.1 Wire playground(index) into scripts/reference_sft_bytes.py as a parallel source on the held-out pretrain split.
  1.2 Train a reference pass over the 27 rungs from E-122 best, using E-122's proven recipe (lr 1e-4, warmup 10,
      cosine, 600 steps, --checkpoint-fraction 0.05, keep best + last 2).
  1.3 Measure on the playground battery (per rung), the aspect pool (48) and the legacy battery (32) as a
      regression guard; test design-doc predictions P1-P6.

**Phase 2 - make the reward teach, not just score (from the literature).**
  2.1 Stepwise / partial credit (ExecVerify): reward a correct loop skeleton with a wrong predicate, so all-zero
      groups stop being dead.
  2.2 Execution-feedback grounding (RLEF, CODERL+): feed the interpreter's actual error back into the loop,
      instead of only applying the -1 penalty.
  2.3 Keep hardening the grader (OpenAI RFT guide) as partial credit is added - partial credit is exactly where
      reward hacking starts.

**Phase 3 - re-gate and run RL only where licensed.**
  3.1 Re-run the harvest with --allow-loops --allow-attributes and read keep rate PER RUNG; RFT/RL is licensed
      only where pass@k > 0 (RAFT / Reinforce-Rej: discard prompts with entirely incorrect responses).
  3.2 Run RFT then RLOO on the licensed rungs.

**Phase 4 - policy tuning (only after the constructs are reachable).**
  4.1 Shortest / fastest path: brevity weight + minimal-step preference.
  4.2 Repetition-penalty decoding.

**Phase 5 - scale the curriculum.**
  5.1 Extend the playground to the other Python parts (LeGo-Code modular curriculum; HS-STAR hierarchical
      sampling).

Standing rules that are not tasks: pass@k = 0 -> reference SFT, never RL, and discard all-zero groups; a perfect
score needs a leakage control; every cross-arm claim needs the same-evaluator control.
