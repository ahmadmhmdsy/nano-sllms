# Python code benchmarks: what exists, what we now have, and what a 25M model can measure

Date: 2026-10-10. Owner: path-2 (RL / evaluation) line. Trigger: user asked whether we have any
benchmark data and whether we can fetch Python-code benchmarks plus SOTA numbers.

## 1. What we had before today

Nothing public. Our eval surface was entirely internal:

- the target val split (979,559 tokens / 3.5 MB of bytes) used for val loss and bits-per-byte;
- the synthetic coding families with dev/final/confirm/rl splits (glm53_flash/aspects.py);
- the mini-eval battery (16 tasks / 32 tests) in runs/target/final/mini_eval_report.json, on which
  the 226M target scores pass@1 0.0 and test_credit 0.0;
- the RL pass@k harness (scripts/evaluate_passk.py, runs/eval, runs/regime).

PLAN.md section 10 lists HumanEval-style benchmarking as out of scope. This intake moves it into
scope at the user's request; PLAN.md itself is user-gated and has not been edited.

## 2. What we downloaded (data/bench/raw, recorded in data/bench/manifest.json)

| file | rows | what it is | licence |
|---|---|---|---|
| HumanEval.jsonl.gz | 164 | the canonical hand-written pass@1 benchmark | MIT |
| mbpp.jsonl | 974 | crowd-sourced problems (superset of the eval rows) | CC-BY-4.0 |
| sanitized-mbpp.json | 427 | MBPP with the auto-generated tests removed | CC-BY-4.0 |
| humanevalplus.jsonl | 164 | HumanEval with far more tests (strict version) | Apache-2.0 |
| mbppplus.parquet | 378 | MBPP+ with extra tests | Apache-2.0 |
| bigcodebench_v0.1.4.parquet | 1140 | tasks requiring real library calls | Apache-2.0 |
| ds1000.jsonl | 1000 | data-science problems over 7 libraries | research use only |

Confirmed available but deliberately not downloaded yet: LiveCodeBench (code_generation_lite,
test.jsonl..test6.jsonl), cruxeval-org/cruxeval, tianyang/repobench_python_v1.1.

Rebuild the manifest with scripts/bench_inventory.py (CPU-only, network-free).

## 3. SOTA anchors (as published by the cited source, not measured here)

| model | params | HumanEval | MBPP | HumanEval+ | MBPP+ | source |
|---|---|---|---|---|---|---|
| CodeLlama-7B (base) | 7B | 33.5 | 41.4 | 3.7 | 11.0 | arXiv:2308.12950 |
| StarCoder2-3B | 3B | 31.7 | - | - | - | bigcode model card |
| Qwen2.5-Coder-0.5B (base) | 0.5B | 28.0 | 52.9 | 23.8 | 47.1 | arXiv:2409.12186 |

The relevant reading: the smallest model with a published HumanEval score is 0.5B - twenty times
our 25M glm - and it saw trillions of tokens. Live leaderboards (evalplus, BigCodeBench,
LiveCodeBench) track the frontier at 90 percent-plus HumanEval; none of that is reachable at our
scale, and quoting it against a 25M model would be misleading. Our 226M target already scores 0.0
on our own much easier battery, so a HumanEval pass@1 of 0.0 for the E-117 arms is the expected
result, not a failure.

## 4. What is actually measurable at our scale

1. **Bits-per-byte on benchmark text** - the only tokenizer-agnostic scalar, so the byte arm, the
   BPE arm and the 226M target can be compared on the same axis. It is also a genuine
   cross-distribution probe: the target trained on code_search_net python, HumanEval is not in it.
   Measure on HumanEval prompt+solution, sanitized-MBPP, DS-1000 and BigCodeBench prompts.
2. **pass@1 with the official tests** as a floor and a plumbing check (HumanEval check(candidate),
   reusing the mini-eval pattern: temp file, subprocess, timeout). Report 0.0 honestly.
3. **Syntactic validity rate** - the fraction of completions that parse. Cheap and non-zero at 25M.
4. The synthetic RL families (E-118) remain the only signal we can actually move at this scale.

## 5. Proposed experiment E-119 (pre-register before its GPU run)

Arms: target baseline (runs/target/final, 226M, CodeLlama tokenizer) vs E-117 A-BYTE vs E-117
B-BPE. Metrics: BPB per benchmark text set; pass@1 on HumanEval and sanitized-MBPP; syntax rate.
Primary gate: A-BYTE BPB on HumanEval text below the target baseline BPB - a fair, reachable bar
that tests whether the glm byte strategy generalizes across distributions. Secondary: the BPE arm
should beat the byte arm on MBPP BPB if the larger vocabulary helps on code-like text.

## 6. Rules

- data/bench/raw is EVAL-ONLY. Never train on it, never mix it into the pretraining corpus, and
  never let an RL task generator read it - a contaminated benchmark measures nothing.
- DS-1000 is research-use-only; do not redistribute it.
- The GPU is single and E-117 is training, so E-119 waits for it.
## 7. Frontier context (fetched 2026-10-10, for scale calibration only)

- LiveCodeBench v6 (contamination-resistant, continuously updated): the leaderboard top sits at
  91.9-93.2 percent pass@1 (Qwen3.8-Flash-Next 91.90, Sakana Fugu-Ultra 93.2) - see
  https://www.llmboard.ai/benchmarks/livecodebench-v6-306 and https://benchlm.ai/benchmarks/livecodebench-v6.
- BigCodeBench: 1140 tasks, with a ~150-task Hard subset; the live board is
  https://bigcode-bench.github.io/ (paper arXiv:2406.15877).
- None of these numbers are comparable to a 25M model. They are recorded so that no future agent
  mistakes our BPB/pass@1 results for a frontier claim.

