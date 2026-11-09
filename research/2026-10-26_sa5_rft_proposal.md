# SA-5 RFT proposal (DRAFT for user review, 2026-10-26)

Status: proposal only. Nothing is built, sampled or launched.
Init: runs/sa4_scale075 (promoted SA-4). Plan v2 row: "RFT, 2–3 rounds: sample k=8 on train tasks, keep test-passing, SFT on them", ≤ 5 h total, gate "pass@1 up each round, no L0 regression".

## 1. Data (task pool), no download needed

- Pool: data/sa2/train.jsonl, which has 125,102 rows with asserts. All of them are already decontaminated against HE/MBPP.
  - Code shapes: opc 58.9k, oci 36.9k, selfoss/teacher about 1.7k.
  - Fix shapes: opc-fix 16.5k, oci-fix 10.6k, teacher-fix 0.5k.
- Tool shape: xlam, 14,854 rows. These rows have no asserts; they are scored by call match (see §2).
- Per round, draw a new disjoint slice of 4,000 tasks:
  - about 2,400 code (he/mbpp prompt shapes)
  - about 1,000 fix
  - about 600 tool
- Use only code/fix tasks with ≥ 3 asserts, so that weak tests let fewer wrong answers through.
- `final` is untouched. Gates use dev/val only.
- Caveat: SA-3 trained on SA-2 completions, so some tasks are memorised. Selection by solve rate (§2) drops those that are already solved.

## 2. Sampling and reward

- k = 8 samples per task, temperature 0.8, top-p 0.95.
  - max_new_tokens: 384 for code/fix, 256 for tool.
  - Runs on GPU with the VRAM cap set to fresh free − 0.75 GiB.
- Code/fix reward: binary. The sample passes only if every assert passes, checked with the existing sandbox `evaluate_source` (2 s timeout).
- Tool reward: the parsed JSON call list exactly matches the reference (names plus canonical args, order-insensitive). Sampling is **unconstrained**, so the model also learns the format and does not rely on the SA-7 grammar.
- Keep rule:
  - Keep tasks with 1–7 of 8 passing. Tasks at 8/8 are already solved; keep at most 10 % of those as anchors.
  - Keep ≤ 2 distinct passing samples per task, deduplicated by normalised AST, preferring the shorter one.
  - Drop samples that hit max tokens, or whose text after the function is not empty or EOS.
- Mix: kept samples plus 25 % SA-4 train replay (all shapes, including repair), to protect format and fix.
- Expected yield per round, if 40–60 % of tasks land in 1–7/8: about 3–5k rows, plus about 1–1.5k replay.

## 3. Training

- `sft.py` with a new configs/sft_sa5_r{1,2,3}.yaml, copied from sft_sa4.yaml. Same LoRA settings, 1 epoch, and only the data dir changes.
- Each round starts from the previous round's best merged model (expert iteration). Round 1 starts from sa4_scale075.
- After each round, do a LoRA-scale merge at 0.75 and at 1.0. Both are scored; the better one is kept. Scale 0.75 won at SA-3 and SA-4.

## 4. Budget (≤ 5 h total; round 1 is a stop/go point)

| Step | per round | note |
|---|---|---|
| Sampling 32k completions + sandbox | ≤ 45 min (unmeasured) | a 200-task probe measures it first; if it is slower, cut the slice to fit |
| SFT on ~5–6k rows | ~15–20 min | from SA-4: 61k rows took ~1.6 h |
| Merge 0.75/1.0 + gate eval | ~30 min | sa4_gate_eval battery (24 min) + constrained format (~3 min) |
| **Round total** | **~1.5 h** | 3 rounds ≈ 4.5 h; stop early when the gate fails |

Disk: 26 GB free now. Each round adds about 0.6 GB (two merged models plus about 100 MB of samples).

## 5. Gate (each round, against the previous round)

Up (all on dev/val, not `final`):

- **Primary:** sampled pass@1 (from k = 8) on SA-2 val code/fix rows. This is about 1.2k tasks, so it has much less noise than greedy HE.
- Greedy HE + MBPP pass@1, summed, is not down. Note: 1 HE task is 0.6 %.

No L0 regression:

- English BPB ≤ +3 % vs init.
- Constrained format 146/146.
- Unconstrained format not down.
- Ladder fix tier ≥ 27, which is the SA-4 value. Init was 35; recovering toward it is a target.
- Tool exact match ≥ 50/146.

Rules:

- **Stop rule:** the first round that fails ends SA-5. The previous round's model goes to SA-6.
- After the final round, a single `final`-split score goes into HANDOFF.

## 6. Build before any GPU launch

- scripts/sa5_rft.py: sample, score, filter, then write a data dir in sa4 `ds` format. It reuses the bench_passk generation and the sandbox.
- CPU dry run: score the reference completions of 500 tasks. All of them should pass; the tool references should match themselves.
- GPU probe, user-gated: 200 tasks × 8, to measure throughput and solve-rate distribution before round 1.

## 7. Points to decide or watch

1. The fix tier is below init (27 vs 35). RFT on fix rows is the main lever. If round 1 does not raise it, say so instead of burying it.
2. Memorisation: SA-2 tasks were SFT targets. If most tasks are 8/8 (the probe shows this), RFT signal is thin. The fallback is fresh OCI rows (A3 stream, about 200 MB), which needs your OK.
3. Weak tests let wrong code through. The ≥ 3-assert floor helps; evalplus is still eval-only.
