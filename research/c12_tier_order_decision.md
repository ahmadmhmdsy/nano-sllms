# Milestone E - Tier-order decision framework + Tier 2 V1 judge rubric

Written 2026-09-06 (CPU-safe window work beside live M3). The decision
itself belongs to the user at Tier 1 pass; this document fixes the
analysis, the recommendation, and the Tier 2 V1 judge contract.

## 1. The decision point

When C12 Tier 1 passes its eval gate (research/c12_runbook.md), choose the
order of the remaining distillation tiers:

- **Tier 3** - intra-ladder logit KD, P (runs/pilot/final) teaching S.
  Same tokenizer on both ends (CodeLlama 32k), so per-token KL is
  well-defined - no cross-tokenizer machinery needed.
- **Tier 2 V1** - on-policy alignment via the teacher-as-judge rerank
  (Qwen/Qwen3.5-0.8B, downloaded + verified 2026-09-06). Cross-tokenizer
  LOGIT KD is undefined (32k vs 248,320 vocab - research/c12_tier2_brief.md),
  so V1 uses verdicts, not logits.

## 2. Analysis

| Factor | Tier 3 first | Tier 2 V1 first |
|---|---|---|
| Dependencies | none (both endpoints local, same tokenizer) | judge harness must be built + frozen first |
| Cost | ~1/10 GPU-hour claim (c12_distillation_plan.md) | teacher inference pass + rerank cycles |
| Risk | tests KD plumbing end-to-end at low stakes | new failure surface: judge parsing/agreement |
| Side benefit | a KD'd S is an extra CONTROL for judging Tier 2 gains | alignment signal arrives sooner |

## 3. Recommendation: Tier 3 first

1. Risk-ordered: Tier 3 validates the logit-serving + KD-loss plumbing
   (the same machinery any future strong-to-weak transfer needs) at the
   lowest cost and with zero external dependencies.
2. While Tier 3 runs, the Tier 2 V1 judge harness (section 4) is built and
   dry-validated - the wall-clock cost of "Tier 2 later" is then near zero.
3. Science: a KD-improved S gives a second baseline, making the Tier 2
   rerank delta measurable instead of anecdotal.

**Reversal criteria** (pick Tier 2 V1 first if any holds at Tier 1 pass):
- Tier 3 preflight shows teacher-logit serving does not fit VRAM beside
  the student on 8 GB (measure, do not assume).
- The user explicitly prioritizes the on-policy alignment path.
- Tier 1 eval gate is marginal AND judge-based sample selection is the
  identified critical lever.

Decision owner: user. Status: RECOMMENDATION ONLY - nothing runs without
the Tier 1 pass plus the standing run gates.

## 4. Tier 2 V1 judge contract (frozen before the run)

### 4.1 Verdict schema (strict JSON, one object per sample)

    {"verdict": "pass|partial|fail", "score": 0-2, "reason": "<=200 chars"}

Judge prompt states the rubric: correctness first, then completeness,
then Python validity; V1 is a RERANK signal, not ground truth - the judge
never executes code.

### 4.2 Parse ladder (never crash a run on a malformed verdict)

1. strict json.loads on the stripped reply
2. strip markdown code fences, retry strict parse
3. extract the first balanced {...} block, retry strict parse
4. ast.literal_eval on the extracted block (tolerates single quotes)
5. all fail -> verdict "unparseable", the sample falls back to the
   ast-parse baseline (sft_data.parses_as_python) for that decision

### 4.3 Decoding + logging

- Temperature 0.2, top_p 0.9, max_new_tokens 120 (verdicts are short).
- Every verdict appended to runs/sft_t2_v1/judge_verdicts.jsonl with:
  instruction_id, verdict, score, parse_path (strict|fences|ast|failed),
  baseline_ast_pass (bool), agreement (bool).
- Per-batch agreement rate printed; final aggregate (judge-pass vs
  ast-parse-pass agreement, unparseable rate) written into the run's
  eval report.

### 4.4 V1 judge acceptance gates

- unparseable rate < 10% over the batch, else tune the prompt once and
  re-measure (never silently relax parsing).
- agreement rate reported every batch; a judge that agrees with the
  ast baseline less than chance is replaced, not averaged in.
- teacher inference runs STANDALONE (0.8B fp16 ~ 1.7 GB weights) - never
  co-run with any training job.

## 5. What this milestone does NOT do

- No run is started; Tier 2/3 remain user-gated (TASKS rows 5, 14).
- No teacher weights are moved; data/teacher/ stays gitignored.
