# SA-7 runtime proposal (DRAFT for user review, 2026-10-27)

Status: §6 steps 1–3 BUILT on CPU (2026-10-27, HANDOFF 2026-10-27f): scripts/tool_grammar.py, scripts/sa7_runtime.py, scripts/sa7_agent.py, all dry runs PASS. Step 4 ran on GPU 2026-10-27 (user-approved): runtime 22.5 % on the agent task set; agent loop re-run with the schema grammar, results in §8. §7 had no answer, so the build uses the defaults: visible/hidden split, write_file included, N = 12 (k 8 + 4 retry). The pass@10 numbers in §0 were measured on 2026-10-27 (scratch/sa_e_passk10.cmd); every other number here is a plan.
Model: runs/sa4_scale075 (the incumbent). PLAN §11 SA-7 row: "constrained tool-call decoding, sample-k + test filter, retry-with-traceback; agent-loop eval", gate "tool validity 100 %; agent-loop success reported".

## 0. Why the runtime, and what E says

- Greedy pass@1 is 12 % combined (HE 14/164, MBPP 57/427). On the clean gate it is 7 % per sample and 15–18 % solved@4.
- Whatever a runtime adds is capped by pass@k: sample-k + filter can only return an answer the model can already produce. The E numbers (2026-10-27, T 0.8, top-p 0.95) measure that cap:

| Set | pass@1 | pass@5 | pass@10 (solved) |
|---|---|---|---|
| MBPP-sanitized (427) | 8.2 % | 19.4 % | **26.2 %** (112) |
| HumanEval (164) | 6.3 % | 11.1 % | **12.8 %** (21) |
| HE + MBPP combined (591) | — | — | **22.5 %** (133); PLAN target 25–35 % |
| Clean gate (370, k = 10) | 7.0 % | 17.8 % | **23.0 %** (85) |

  Sources: research/raw/sa_e_passk10.json and scratch/sa_e_gate_k10 (3,700 samples, 0 not_evaluated, peak VRAM 2.9 GiB).
- Clean-gate curve: pass@1 7.0, @2 11.1, @4 16.2, @8 21.3, @10 23.0. The gain from k = 8 to k = 10 is about 0.8 pt per extra sample. 29 of the 85 solved tasks pass in only 1 of 10 samples, so a visible-test filter will lose some of them.
- How to read this:
  - The ceiling is about 23 %, which sits between the two thresholds this proposal set (≥ 25 % means the runtime carries; well under 20 % means data is the lever).
  - A realistic runtime result on the clean gate is roughly 7 % → 15–20 %. That is a real 2–3× gain.
  - Still, 77 % of clean tasks are never solved in 10 samples. No runtime reaches those; only new data or training does (option C).

## 1. Components

The runtime is one library, `scripts/sa7_runtime.py`, with a thin CLI. It reuses code that already exists and is tested:

| Part | Reuses | New |
|---|---|---|
| Constrained tool JSON | `ToolCallGrammar` from scratch/sa4_format_check.py (MAX_STR 128, MAX_RUN 16, no duplicate keys, ≤ 8 calls); 146/146 valid on GPU | Move it to `scripts/tool_grammar.py` unchanged, so the runtime and the format check import one copy |
| Sandbox | `sa2_build.run_guarded` (timeout, 1 GiB child RSS cap) + the 4× timeout retry → NOT_EVALUATED rule from sa5_rft | None |
| Cutting completions | `bench_passk.truncate` (HE stop rule, MBPP top-level rule) | None |
| Retry prompt | The SA-4 `repair` shape (attempt → real outcome → retry), which the model was trained on | A renderer that builds that prompt from a live failing sample |

### 1a. Sample-k + test filter

1. Sample k completions at T 0.8, top-p 0.95. Reuse the batched generate from bench_passk.
2. Run each sample against the **visible** tests only (§2). Keep the ones that pass.
3. Return one of them. Several passers are deduplicated by AST, and the answer comes from the largest AST cluster, ties broken by shortest. This is a cheap majority vote.
4. If nothing passes the visible tests, go to 1b. If the task has no visible tests, return the largest-cluster sample (vote only).

### 1b. Retry-with-traceback

1. Pick the failing sample whose error is the most informative: an assertion failure beats an exception, which beats a timeout. Ties go to the largest cluster.
2. Render a `repair` prompt from that sample and its real outcome. The outcome is either the Traceback or "returned X but expected Y", truncated to 600 chars.
3. Sample k_r more completions and go back to the filter in 1a.
4. At most R retry rounds.
5. Budget: k + R·k_r samples per task in total. The default is k = 8, R = 1, k_r = 4, which makes 12. §3 explains how N is chosen.

### 1c. Constrained tool calls

When the prompt lists tools, generation runs under `ToolCallGrammar`. Validity is 100 % by construction; the gate checks that it stays 100 % on the agent loop.

## 2. Leak-free filtering (important)

If the runtime filters on the same tests it is scored on, the result is just pass@k. That is an oracle number, not a runtime number. So the plan is:

- **MBPP:** the first assert is visible and the remaining asserts are hidden. Most sanitized tasks have 3 asserts. Tasks with a single assert are vote-only.
- **HumanEval:** the visible tests are the `>>>` examples parsed from the docstring, turned into asserts. Tasks without examples are vote-only. The hidden tests are the official `check()`.
- **Clean gate** (sa5_rft rows): split the asserts the same way as MBPP.

Each run reports three numbers:

- greedy pass@1, the baseline;
- runtime pass, filtered on visible tests and scored on hidden tests, which is the real number;
- oracle pass@N, the ceiling.

A builder `--dry-run` checks the split on CPU: every reference solution must pass both its visible and its hidden tests.

## 3. Choosing N

- From the E curve, take the marginal gain of pass@k between k = 1, 5 and 10.
- Stop growing N where one more sample adds less than about 0.5 pt of solved tasks.
- Latency: a 135M model with batched sampling on GPU takes about 1 s per task for 10 samples. On CPU the cost is roughly k times greedy, so the CPU default is N ≤ 8.
- The runtime exposes `--k`, `--retries` and `--k-retry` so N can be changed without code changes.

## 4. Agent-loop eval (new)

There is no agent-loop eval today. The SA-4 tool training is single-turn xLAM function calling, so a multi-turn coding loop is out of distribution. Expect a low number; the gate only says "reported".

- **Tools:** `read_file(path)`, `write_file(path, content)`, `run_python(code)` and `run_tests()`. Each is described in the same tool-list format SA-4 trained on, and every call goes through the grammar.
- **Environment:** a temporary directory per task. Every tool runs through `run_guarded` with the same caps. Observations are truncated to 600 chars.
- **Loop:** the model emits calls, the runtime executes them and appends the observations, for at most 8 turns. The episode succeeds when `run_tests()` passes at the end. Hidden tests are run once after the loop, outside the model's view.
- **Tasks (about 120, all from val; `final` untouched):**
  - 60 "implement": a clean-gate task as `solution.py` with a stub, plus `test_solution.py` holding the visible asserts.
  - 60 "fix": a clean-gate fix-tier task where `solution.py` holds the buggy code and the test fails.
  - *As built:* the clean gate has no fix-tier tasks (SA-V.1 kept none), so each fix episode starts from a real failing sa4_scale075 sample from the E k=10 run (scratch/sa_e_gate_k10/samples.jsonl) on a clean-gate task. Ids: research/raw/sa7_agent_ids.json.
- **Report:**
  - success rate on hidden tests;
  - success on visible tests;
  - mean number of turns;
  - tool validity (must be 100 %);
  - the share of episodes that never call `run_tests`.
- **Baseline in the same report:** the non-agent runtime (1a + 1b) on the same 120 tasks. This shows whether the loop adds anything over plain sample-and-test.

## 5. Gates

| Check | Bar |
|---|---|
| Tool-call validity: format set + agent loop | 100 % |
| Runtime pass on the clean gate (hidden tests) | > greedy pass@1, with a paired bootstrap CI above 0 |
| Runtime pass on HE + MBPP (hidden tests) | reported against the oracle pass@N |
| Agent-loop success | reported (no bar; PLAN §11) |
| Reference dry-run of the visible/hidden split | 100 % pass |

## 6. Build order (CPU first, GPU user-gated)

1. **CPU:**
   - Move the grammar into `scripts/tool_grammar.py`, with the format-check import updated and its tests re-run.
   - Write `sa7_runtime.py` with a stub-model unit test (fixed fake samples) covering: filter, vote, retry prompt rendering, budget accounting.
2. **CPU:** build the visible/hidden split and run the reference dry-run.
3. **CPU:** build the agent-loop task set and environment. Dry-run it with a scripted "oracle agent" that writes the reference and calls `run_tests`; every task must succeed.
4. **GPU (user-gated, ~1 h estimate, cap = fresh free − 0.75 GiB):** the runtime on the clean gate + HE/MBPP, then the agent loop. One .cmd with one arm per step, following the long-eval pattern.

Nothing in this proposal trains a model or touches data/sa2, data/sa4 or `final`.

## 7. Open questions for the user

1. Is the visible/hidden split (§2) acceptable as the definition of "runtime pass"? The alternative is to report only the oracle pass@N.
2. Should the agent-loop tools include `write_file`? Without it, the loop is "run code and tests, then answer", which is closer to what SA-4 saw.
3. Default budget: the E curve is still rising at k = 10 (about 0.8 pt per sample), so N = 12 (8 + 4 retry) is the proposed default. Is that acceptable?

## 8. Result note: schema-constrained grammar (2026-10-27, HANDOFF 2026-10-27g)

- The first agent run failed on schema errors (the grammar was syntax-only). `tool_grammar` now has a `schema` mode: exact argument keys, string values, and path enums. `sa7_agent` turns it on by default.
- Re-run (research/raw/sa7_agent_schema.json): tool errors went from 6.14 to 0 per episode, and episodes that never ran tests dropped from 100 % to 51 %. Success did not move (hidden pass 0.8 %, full 0 %), while the plain runtime gets 22.5 % on the same tasks.
- What is left:
  - a semantic gap: the model writes junk such as "test_solution.py" as the file content;
  - 43 unclosed replies, caused by repeated `\n` escapes inside strings.
- Conclusion: §1c validity is solved for closed replies. The agent loop needs training data, not more decoding constraints.
- **SA-7 CLOSED (user decision, 2026-10-27).** The runtime is the product. Agent-loop success is reported as ≈ 0 %, and the unclosed-reply gap is recorded as a known limitation.
