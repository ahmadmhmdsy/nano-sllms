# SA-B: narrow the Python job to in-distribution task families (proposal)

Status: ADOPTED as option (a) by the user (2026-10-27, HANDOFF 2026-10-27h). PLAN §11 carries the scope, and the scoped gate is research/raw/sa_b_scoped_gate.json (99 gate tasks once the single-function filter is applied, not 103). The training-data view (step 2 below) is built in data/sa_b_scoped by scratch/sa_b_train_view.py (HANDOFF 2026-10-27i). No retrain has run on it.

## Question

The SA-E pass@10 run showed that `runs/sa4_scale075` solves only a small share of the benchmark tasks. Before we invest in SA-7 runtime tricks or more RL, we want to know two things. Which task families does the model already solve? And would narrowing the job to those families give us a target the model can reach?

## Method

There was no new generation. The script `scratch/sa_b_families.py` reused the existing k=10 samples:

- **Clean gate:** 370 tasks.
- **HumanEval:** 164 tasks.
- **MBPP-sanitized:** 427 tasks.

That is 961 tasks with 10 samples each. Each task was tagged by features of its reference solution and task text:

- topic labels (multi-label keyword rules);
- AST node count, in quartile buckets with cuts 30, 48 and 76;
- control flow;
- imports;
- helper functions.

Raw output is in `research/raw/sa_b_families.json`.

The metrics are:

- **solved@10:** any of the 10 samples passes.
- **pass@1:** the mean pass rate per task.

## Results (solved@10)

| axis | bucket | gate | HE+MBPP |
|---|---|---|---|
| set | all | 0.230 | HE 0.128 / MBPP 0.262 |
| size | q1 ≤30 nodes | **0.460** | **0.387** |
| size | q2 ≤48 | 0.300 | 0.201 |
| size | q3 ≤76 | 0.138 | 0.112 |
| size | q4 >76 | 0.176 | 0.102 |
| imports | none / some | 0.272 / **0.075** | 0.242 / 0.148 |
| helpers | single / helper funcs | 0.237 / 0.180 | 0.233 / **0.071** |
| flow | nested loop | 0.190 | **0.085** |
| flow | straight line | 0.208 | 0.269 |

### Topic × size

These rows pool all sets and keep only cells with n ≥ 15.

| Strongest cells | Weakest cells |
|---|---|
| list·q1 0.517 (n=118) | string·q3 0.087 |
| dict_set·q1 0.500 (n=44) | tuple·q3 0.043 |
| math·q1 0.433 (n=104) | |
| string·q1 0.333 (n=72) | |

Above about 48 nodes, every topic drops to 0.09–0.16.

### Failure mix

- **Gate:** the "fail" label lumps wrong answers and exceptions together, so 98% show as "wrong". That split is not informative.
- **HE+MBPP:** about a third of the failing samples raise an exception. The rest are wrong answers.
- **Timeouts and no-stop:** at most 3%. Stopping is not the problem.

## Candidate narrowed scope

**In scope:**

- a single function;
- no imports;
- reference size ≤ 48 AST nodes;
- topic in {list, dict_set, math, string}.

| scope | gate tasks | solved@10 | pass@1 | HE+MBPP tasks | solved@10 | pass@1 |
|---|---|---|---|---|---|---|
| ≤30 nodes | 35 / 370 | 0.600 | 0.254 | 133 / 591 | 0.436 | 0.167 |
| ≤48 nodes (proposed) | 103 / 370 | 0.427 | 0.158 | 255 / 591 | 0.337 | 0.125 |
| everything | 370 | 0.230 | 0.070 | 591 | 0.225 | 0.077 |

The single-function filter was not applied in these two scope rows. Only size, imports and topic were filtered. Helper functions are rare below 48 nodes, so it moves the numbers little.

The ≤48 cut keeps a usable pool (358 benchmark tasks). On that pool, solved@10 is about 0.35–0.43, roughly double the full set. That is the regime where SA-7 sample-and-filter or retry can turn solved@10 into a delivered pass.

## What adopting it would mean

1. **Dev gate.** Add a scope-filtered gate next to the 370-task clean gate. Report both and do not replace the clean gate. The filter is in-scope reference ≤48 nodes, no imports and one function, with the topic filter optional.
2. **Training data.** Weight SA-2/SA-4 rows toward the scope.
   - Do not modify data/sa2 or data/sa4. Build a filtered view under a new directory.
   - Optionally down-weight import-heavy and multi-helper rows.
3. **Product statement.** "Writes short single-function Python (list, dict, math, string) with tests." It is not a general-purpose coder.
4. **Out of scope until the model grows:**
   - imports and the stdlib API (itertools, re, heapq);
   - multi-helper programs;
   - nested-loop algorithms;
   - tasks above about 76 nodes.

## Caveats

- Topic labels are keyword rules. They are a weak signal, and the size and import axes are more reliable.
- Size comes from the reference solution, not the model's attempt. At inference time we would need a proxy, for example the length of the task description or a classifier, to route tasks out of scope.
- These numbers come from one checkpoint (sa4_scale075) with sampling at the SA-E settings.
- The narrowed pool is drawn from benchmark tasks. If we train toward it, we must keep using the decontaminated training pools and never use the `final` split.

## Decision needed

Approve one of the following:

- **(a)** Adopt the ≤48-node / no-import / single-function scope as the SA target. This adds a scoped gate and edits PLAN §11.
- **(b)** Keep the full scope and use these numbers only as diagnostics.
