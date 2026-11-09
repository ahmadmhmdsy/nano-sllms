# GLM-5.3-Flash RL playground v2 - design and suggestions (path 2)

Status: design, agreed scope for stage 1. Written 2026-10-10 while the E-117 byte arm
trained (single GPU: this work is CPU-only by construction).
Scope of record: TASKS.md row 163. Experiment ID for the first GPU stage: **E-118**.

## 1. The request

"Improve the RL section/part so we can cover many aspects of python and we can allow
model to train on it and it can learn by try; create an RL playground for each python
part so the model can learn step by step and later we combine them to make the model
fluent in python."

## 2. What the RL line is today (measured, not assumed)

- Reward is executable unit tests: glm53_flash/evaluator.py AST-validates the candidate,
  execs it in-process against the task's cases, and scores pass_fraction.
- train_rl.py is RLOO over a frozen split: --families selects a subset, schedule()
  shuffles the selected tasks and cycles epochs, and a deterministic schedule.json is
  checked before any resume. So **mixed-family replay already exists** - passing
  --families a,b,c interleaves them; no separate "combine" phase is needed.
- 8 families exist: increment, double, square, absolute, nonnegative, even, reverse,
  list_sum. Each is a one-line body with 3 fixed cases.

## 3. The four real ceilings (each is a reason the current loop cannot teach python)

C1. **Fixed cases per family.** Family.cases is a module constant, identical for every
    task instance and for every split. make_task() randomises only the function name and
    the description. The model can therefore satisfy the reward by memorising 3 answers,
    and the "confirm" split tests nothing but a renamed function. There is no input
    generalisation signal at all.

C2. **The AST allowlist is the curriculum ceiling.** validate_source denies For, While,
    Try, Raise, With, Import, ClassDef, Lambda, Global, Nonlocal, Delete, every
    ast.Attribute, dunder names, recursion, and any call outside
    {sum, len, abs, min, max, bool, int, str}. So: no loops, no .split()/.join()/
    .append()/.items(), no dict methods, no exceptions, no imports, no classes.
    Comprehensions and subscripting *are* allowed, which is why the current 8 families
    fit - but most of "python" is unexpressible.

C3. **RLOO needs within-group reward spread.** leave_one_out() returns zero advantage
    when all group members score the same (all-zero or all-one groups), and the run logs
    reward_spread and skips the update when spread is ~0. A family the model already
    solves 4/4 gives no gradient, so "more aspects" must keep a difficulty spread, not
    just more tasks.

C4. **No per-family instrument.** E-116's headline 3-family gain (15 -> 17/24) came with
    a post-hoc all-8 flat result (30/64 -> 30/64, p=1.0): a multi-family improvement can
    be pure redistribution between families. Without a per-family matrix at every
    checkpoint, "combine them" is unfalsifiable.

## 4. Suggestions, in the order they pay off

### S1. Generate cases per task instance (fixes C1) - highest value per line of code
Give each family a case generator seeded from (split, family, index). Every task instance
then gets its own inputs, the rl and confirm splits get *different* inputs for the same
description, and the reward measures generalisation. Determinism is preserved (the seed
is a pure function of the task identity), so schedule.json resume semantics are unchanged.
Cost: ~1 line per family plus a generator.

### S2. Add an aspect ladder instead of more one-liners (fixes C2)
Two capability tiers, deny-by-default in both:

- **Tier core** (works with today's in-process evaluator, no new sandbox): conditionals,
  strings, list aggregation, dict access - all expressible with if/else, subscripting,
  comprehensions and the existing safe calls. Proposed new families: sign, clamp,
  grade (conditionals); is_palindrome, repeat_text, first_or_empty (strings);
  count_even, count_positive, contains_zero (lists); lookup, key_count, has_key (dicts).
- **Tier loops+attrs** (needs S3): explicit for/while, attribute access, and an extended
  safe-call set (range, sorted, list, dict, set, tuple, enumerate, zip, reversed, any,
  all) - i.e. sum_positive, word_count, dict_values, sorted_desc.

Tier core is stage 1 because it needs no sandbox and cannot hang: the only iteration is
over comprehension inputs, and range is not callable. Tier loops+attrs is stage 2.

### S3. Evaluator v2: capability tiers + a subprocess sandbox (fixes C2's hard half)
- Capabilities as data (allow_loops, allow_attributes, extra safe calls), threaded through
  validate_source/evaluate_source with **defaults identical to today**, so every existing
  caller, test and resumed run behaves exactly as before.
- A persistent child-process runner (one spawn, JSON-lines over stdin/stdout) that executes
  candidates outside the trainer, with a wall-clock timeout and kill-and-respawn on
  overrun. Measured here (scratch/probe_subprocess.py): piped round-trip works, spawn cost
  0.081-0.111 s (mean 0.0935 s), subprocess.TimeoutExpired fires as required, and both
  clean exit codes (3) and hard os._exit(9) crashes are visible to the parent.
- Why a sandbox at all when loops are denied today: denying loops is *how* the in-process
  evaluator stays hang-proof. Allowing them without a timeout would let one bad completion
  hang the trainer forever. The sandbox is what converts "we cannot express python" into
  "we can express python safely".

### S4. Per-family forgetting matrix (fixes C4) - the instrument for "combine"
One script that loads a checkpoint once and reports pass@k per family on a chosen split,
for every checkpoint of a run, producing a family x checkpoint matrix. Run it after any
multi-family arm. A multi-family gain counts only if the diagonal does not pay for it.

### S5. Difficulty-aware scheduling (fixes C3)
Log reward_spread per family per checkpoint (the row already carries family and rewards).
When a family's spread collapses to 0, its gradient contribution is gone; the fix is to
keep sampling it (replay, so it is not forgotten) while adding harder instances from the
same family - which S1 makes possible, since generated cases can be difficulty-graded.

### S6. Init from real code, not the synthetic corpus
E-113 showed RL headroom is bounded by what the base model already represents. The RL line
today starts from a model pretrained only on the 8-family synthetic text. E-117 produces a
model pretrained on real Python; A/B the RL init (E-118 vs E-118-from-E-117) rather than
assuming.

## 5. Stage plan and gates

**Stage 1 (this work, CPU): implement S1 + tier-core aspects + evaluator v2 mechanism +
S4's matrix script + tests.** No GPU. Gates before stage 1 counts as done:
- G1: every existing test still passes (backward compatibility of the evaluator API).
- G2: generated cases are per-instance and rl/confirm disjoint (asserted in a test).
- G3: capability gating denies For/Attribute by default and permits them under the tier
  (asserted in a test), and the sandbox enforces its timeout on a hanging candidate.
- G4: reward parity - the in-process and sandbox paths return identical evaluations for a
  corpus of candidate sources.

**Stage 2 (GPU, after E-117): E-118 RL run.** Pre-registered when its launch approaches;
gates must include a per-family matrix (S4), not just an aggregate. Single GPU: it cannot
overlap E-117 or the B-BPE arm.

## 6. Open decisions for the user

- Aspect count: stage 1 adds 4 aspects (12 families total in the core pool). Add the
  loops+attrs tier in stage 2, or go straight for it?
- Sandbox default: on for all RL runs (slower, safer, richer) or only for tier loops+attrs?
- Should the RL split be regenerated (new generated cases) for E-118, and the old
  confirm split kept frozen as the historical reference?
## Implementation status - stage 1 delivered 2026-10-10

| Piece | File | State |
|---|---|---|
| 12 aspect families, per-task generated cases, tiered pool API | glm53_flash/aspects.py | done |
| Evaluator v2: Capabilities tiers, default = the exact v1 policy | glm53_flash/evaluator.py | done |
| Persistent child-process runner, JSON-lines, timeout + respawn | glm53_flash/sandbox.py | done |
| Trainer wiring: --pool/--tier/--sandbox/--allow-loops/--allow-attributes/--extra-safe-calls | scripts/train_rl.py | done |
| Per-family forgetting matrix | scripts/eval_forgetting_matrix.py | done |
| Tests (references pass, cases vary, tier gating, sandbox parity, timeout recovery) | tests/test_aspects_evaluator.py | 9 passed |

Evidence: 15 passed (new tests + test_lab.py); CPU smoke with --pool aspects --tier core
--sandbox on completed 2 groups in 6.8 s, rewards [-0.1, -0.1] (random-init model,
unparseable completions -> invalid penalty), checkpoints written, start event carries
pool/tier/sandbox/capabilities.

Not yet done (stage 2): real-code RL init from the path-1 pretrained glm, difficulty-aware
scheduling, and the A-ASPECTS/A-LEGACY/A-TIER comparison itself (pre-registered as E-118).

## 7. Approved additions - S7 to S10 (user decisions 2026-10-10 / 2026-10-11)

These were approved after the stage-1 write-up above, and they change the ORDER the ladder runs in. The
governing measurement is MEMORY 2026-10-11: pass@k = 0 is a hard wall - neither RL nor RFT can move an
aspect the base never solves (donor 0/8 aspects viable; arm A 7/8 after 3 epochs of SFT on reference
solutions). The ladder is therefore install -> amplify -> sharpen, not RL-first.

### S7. RFT/STaR between SFT and RL (approved: "3. The piece I'd add: RFT/STaR before RL")

scripts/train_rft.py. Each round samples every task with generate_group(sample=True), keeps ONLY the
completions whose evaluate_source(...).passed is true, and takes a masked completion-only supervised
step on those. RL then starts from the RFT policy instead of the SFT one.

Why it belongs before RL: RL multiplies the probability of an already-reachable success, while RFT turns
a rare success into a common one - so RFT both raises the starting point and widens the set of aspects
for which RL has any gradient at all. The reading is the per-family keep-rate curve, not the loss.

### S8. Shortest/fastest, gated on correctness (approved: "confirm the model is good enough then go shortest/fastest")

scripts/train_rl.py --brevity-weight (default 0.0). Applied ONLY to a completion that already passed
every test case: +w at a single byte, 0 at the reference body length, -w at twice it. Because it is
gated on a full pass it cannot buy correctness - it only orders correct answers against each other.
Default 0.0 keeps the existing reward bit-for-bit, so every earlier E-116/E-120/E-121 reading stays
comparable. Verified: at-ref 0.0, 1/6-ref 0.0833, 2x-ref -0.1, 3x-ref -0.1 (clamped), weight 0 -> 0.0.

Negative result that closed the other shortest-path lever (E-120): no-repeat n-gram-3 decoding HURTS
(rl-0096 9/32 -> 4/32) because Python's 4-space indent is itself a repeated n-gram. Byte-level
repetition control is closed off; a reward term is the remaining route.

### S9. Per-aspect temperature (measured, not assumed)

Temperature is a PER-ASPECT hyperparameter, not a global constant. At T=0.35 the E-120 pre-flight gives
list_sum 1.00, double 1.00, increment 1.00, square 0.50, reverse 0.50, absolute/nonnegative/even 0.00;
RFT yields at the same temperature are list_sum 1.000, double 0.625, increment 0.375, square 0.125,
reverse 0.125. A single --temperature for the whole pool spends budget on aspects that are already
saturated and starves the ones that need diversity.

### S10. LoRA / freeze + add layers (approved: "2. Item 6")

The user's constraint: "we do not re train in ladder but we can use lora or freeze weight and add some
layer or with another new to train model". The existing pattern to copy is
mex/train_mu2_lora.py --config configs/dia2f_replay.yaml. --train-scope last-block / last-block-head
already supplies the freeze half (2,140,232 trainable of 25.7M, measured on the RFT smoke); the LoRA
half and a per-aspect adapter-merge stage are what is left. This is what makes "combine them to make the
model fluent in python" concrete: one adapter per aspect, then a merge/router.

## 8. The ladder as it now runs

  1. reference SFT     install the idioms      (arm A: 0/8 -> 7/8 viable, 0/32 -> 9/32 greedy)
  2. RFT/STaR          amplify - sample, keep the passing, SFT on those (scripts/train_rft.py)
  3. RL (RLOO)         sharpen, per-aspect temperature, dense reward
  4. brevity           shortest/fastest, only once correctness is confirmed (--brevity-weight)

Route by the pass@k pre-flight, never by intuition (scratch/passk_preflight.py):
  pass@k > 0  -> RL viable now; RFT viable (amplify)
  pass@k = 0  -> route to reference SFT; RL/RFT burn budget for exactly zero gain

## 9. Status 2026-10-12

| Piece | File | State |
|---|---|---|
| RFT/STaR trainer | scripts/train_rft.py | built, CPU-verified (keep_rate 1.0, 4 supervised steps, auto-resume from round 2, retention + prune correct) |
| Shortest-path reward term | scripts/train_rl.py --brevity-weight | built, unit-verified, default off (backward compatible) |
| "pretrain" seed for the legacy pool | glm53_flash/tasks.py | done - select_tasks had silently restricted RFT to the aspects pool |
| RFT -> RL ladder on arm A | - | BLOCKED on arm B (single GPU, ~8.5 h at step 837/5000) |
| LoRA / per-aspect adapters + merge | - | not started (S10) |
| Keep-rate pre-flight on the 12 core aspects | research/raw/e121b_rft_harvest_preflight.json | done - keep_rate 0.4453 (171/384); 7/12 aspects solvable, 5/12 yield zero pairs and are skipped by RFT |


