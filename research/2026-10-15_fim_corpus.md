# FIM corpus from the example-level EOS reference set (byte-string sentinels)

**Date:** 2026-10-15
**Author:** delegated research subagent (CPU-only; no model trained, no GPU job, nothing under `runs/` touched, source corpus unmodified)
**Status:** DONE — transform built and independently verified PASS.
**Companion docs:** [research/2026-10-15_special_tokens.md](2026-10-15_special_tokens.md) (section 6 item #2 — the decision this implements), [research/2026-10-15_eos_rebaseline.md](2026-10-15_eos_rebaseline.md)

## 1. Deliverables

| Artifact | Path |
|---|---|
| transform script | [scratch/fim_transform.py](../scratch/fim_transform.py) |
| verification script | [scratch/verify_fim.py](../scratch/verify_fim.py) |
| corpus (train.bin / val.bin / meta.json) | [data/reference/bytes_pg_replay_fim/](../data/reference/bytes_pg_replay_fim/) |
| source, read-only | [data/reference/bytes_pg_replay_eos/](../data/reference/bytes_pg_replay_eos/) |

The source has no paired `.txt`; examples were recovered by decoding the uint16 id stream (id - 4 = byte, split on eos_id = 2). The source is 16,800 train / 840 val examples, all ASCII (ids 2..129), no empty example.

## 2. Why byte-string sentinels (no vocabulary change)

A new special id is a new embedding row the corpus never exercises — exactly the failure that took E-122f from 11/71 to 0/71, and the reason the survey ranks real-id FIM as the riskier of its two routes. Byte-string sentinels reuse the existing byte vocabulary (ids 4..259), so `vocab_size` stays 260, no checkpoint needs an embedding / lm-head resize, and the generation mask in `glm-5.3-flash-from-scratch/glm53_flash/runtime.py` (~line 357, masks pad/bos/sep to -inf) needs no edit: the sentinel bytes are ordinary byte ids and remain emittable.

Sentinels (UTF-8 bytes, packed as id = 4 + byte):

- `fim_prefix` = `<|fim_prefix|>` (14 B) — emitted
- `fim_suffix` = `<|fim_suffix|>` (14 B) — emitted
- `fim_middle` = `<|fim_middle|>` (14 B) — emitted
- `fim_pad` = `<|fim_pad|>` (11 B) — reserved, never emitted (we do not left-pad; verified present 0 times in the output)

## 3. Exact emitted format

```
untouched : <example bytes> EOS
PSM       : PRE prefix SUF suffix MID middle EOS
SPM       : PRE suffix SUF prefix MID middle EOS
```

- PRE/SUF/MID are the sentinel byte strings above; EOS is id 2, exactly once, at the end of every example.
- Packing is byte-for-byte the source recipe: every byte -> id = 4 + byte (uint16 little-endian), then id 2.
- The three content pieces always satisfy `prefix + middle + suffix == source example bytes`. The split removes a middle span and the middle is generated last, immediately before EOS (composing with the terminator the model already learned).
- PSM/SPM choice here is the FIM paper / Code Llama suffix-prefix-middle ordering: SPM reorders the two context pieces so the prefix sits immediately before MID and remains visible context. It is NOT StarCoder SPMv2 (see section 8).

## 4. Transform parameters

- **fim_rate 0.5**: a per-example coin decides transformed vs untouched. Untouched examples are copied byte-identically and keep their single EOS.
- **PSM/SPM 50/50** per transformed example.
- **Byte-level split**: two distinct offsets sampled uniformly from {1..n-1}, sorted `i < j`; `prefix = raw[:i]`, `middle = raw[i:j]`, `suffix = raw[j:]`. All three pieces non-empty (source min example length is 93 B).
- **Example count and order preserved**, so output index k twins source index k; the val split is transformed with an independent seed and no example crosses the train/val boundary.
- **Determinism**: python `random.Random`, train seed 9319, val seed 9320. Re-running the builder reproduces the same sha256 values.

## 5. Outputs

| split | examples | transformed | untouched | PSM | SPM | ids | bytes | sha256 |
|---|---|---|---|---|---|---|---|---|
| train.bin | 16,800 | 8,460 | 8,340 | 4,176 | 4,284 | 3,231,813 | 6,463,626 | `00b3ef8e0c18636be958540240e6ab0038ff0e76050b5b2e60194e6cf33495f8` |
| val.bin | 840 | 403 | 437 | 180 | 223 | 160,868 | 321,736 | `6836024115d94bd1f0bb25d3be1aeabe0eff7875b566fcb0282d95e247a3897e` |

Source shards: train.bin 2,876,493 ids / 5,752,986 B (sha256 `ac9d0b6bf2374be2951218a8bf58984128caaf0e6996bb16e8bac92cb9460e30`), val.bin 143,942 ids / 287,884 B (sha256 `00862575f6f5f3670cb8b9a9ba6f8c4185855ee7c18460cbafb41dbb21d27e01`). Every transformed example adds 3 * 14 = 42 content ids, so the id delta is exactly 8,460 * 42 = 355,320 (train) and 403 * 42 = 16,926 (val). `meta.json` records source corpus, source sha256, fim_rate, PSM/SPM split, sentinel bytes, sentinel encoding, orders, byte_offset, append_eos, eos_id, per-split counts, and the sha256 of each output.

Exact commands (from the parent root, venv only, CPU):

```powershell
& .\.venv\Scripts\python.exe scratch\fim_transform.py   # exit 0
& .\.venv\Scripts\python.exe scratch\verify_fim.py      # exit 0
```

## 6. Independent verification (scratch/verify_fim.py)

The verifier does not import the transform and does not share its RNG; it re-derives every structure from the two byte streams. Result: **9 checks, 0 failed, exit 0 (PASS)**.

| Check | Status | Counts |
|---|---|---|
| A. twin, train | PASS | 8,340 untouched byte-identical; 8,460 transformed reassemble (PSM 4,176 / SPM 4,284, ambiguous 0), 0 failures |
| A. twin, val | PASS | 437 untouched byte-identical; 403 transformed reassemble (PSM 180 / SPM 223), 0 failures |
| B. one EOS per example, train | PASS | 0 interior EOS; EOS total 16,800 == examples; last id is EOS |
| B. one EOS per example, val | PASS | 0 interior EOS; EOS total 840 == examples; last id is EOS |
| C. sentinels absent from source, train | PASS | all 4 sentinels 0 occurrences in 16,800 source examples; in output prefix/suffix/middle 8,460 each, pad 0 |
| C. sentinels absent from source, val | PASS | all 4 sentinels 0 occurrences in 840 source examples; in output prefix/suffix/middle 403 each, pad 0 |
| D. round-trip >= 20 | PASS | 20/20 transformed examples parsed, middle span recovered at the expected offset, 0 failures |

Check A proves the twin relationship both ways: an untransformed output example is byte-identical to its source, and a transformed one differs only by sentinel insertion + piece reordering (content segment lengths sum to the source length and `prefix + middle + suffix` reproduces the source exactly). Check C proves the markers are genuinely distinctive in this corpus. Check D decodes transformed examples back to text and confirms the middle slice equals `source[len(prefix) : len(prefix)+len(middle)]`.

## 7. Held-out infilling eval (idea; deliberately NOT run)

**Items.** Build a held-out infilling battery from data disjoint from training: the `val` split of this corpus (840 examples) plus, for a cleaner task-level view, the playground `dev` / `final` rungs (never in the pretrain split). Each item = a source reference function with a chosen middle span. Report two subsets: **random-span** (same two-cut distribution as training) and **line-span** (the function body line, a realistic infill target). Two prompt orderings (PSM and SPM) per item.

**Prompt construction.** Feed exactly the training envelope up to MID: `PRE prefix SUF suffix MID` (PSM) or `PRE suffix SUF prefix MID` (SPM), through `ByteTokenizer.encode(prompt, bos=True)`, and generate greedily (temperature 0) until EOS or a byte budget (e.g. 160 B, the E-122g setting). The generation `g` is decoded up to the first EOS.

**Primary metric — exact middle match rate.** `score = (1 / N) * sum over items of [ g == reference_middle ]`, byte-for-byte. This is the direct "given prefix+suffix, does the model reproduce the middle?" question, and because the middle is generated immediately before EOS it also tests termination.

**Secondary metrics.**
- **Middle edit similarity**: `1 - Levenshtein(g, reference_middle) / max(len(g), len(reference_middle))`, reported as a mean; gives partial credit where exact match is 0.
- **Line-exact rate** for the line-span subset (the infilled line matches), the code-review-friendly number.
- **EOS compliance**: fraction of items where EOS is emitted before the byte budget (report alongside score, never merged — E-122g showed length and correctness are different signals).
- **Left-to-right regression guard**: the existing strict playground battery on unchanged AR prompts, FIM arm vs control. The FIM-for-free claim predicts no drop; a drop falsifies applying FIM at 25.7M.

**Controls and pre-registered threshold.** A/B against the identical corpus with fim_rate = 0 (i.e. `bytes_pg_replay_eos`), same init, same steps, same seeds, EOS present in both arms; greedy so CPU rows are deterministic. Pre-register the claim as: FIM arm beats the no-FIM control on exact middle match by a factor >= 2 (and >= 20% absolute) while its strict AR score is within noise of the control. A trivial "copy the last prefix bytes" baseline is the floor. If the FIM arm does not clear that, FIM did not transfer at this scale and the next data-format idea should be chosen instead. No training was launched here.

## 8. Limitations

- **SPM convention.** This corpus uses the FIM-paper suffix-prefix-middle ordering (Code Llama style), where the prefix stays visible context. StarCoder SPMv2 as implemented in the santacoder FIM benchmark (`<fim_prefix><fim_suffix>suffix<fim_middle>prefix`, the prefix being generated) was not used; the survey's "SPMv2" wording should not be read as this file's SPM. Either convention is valid, but a training run must evaluate with the same one it trained on.
- **Random spans can cut mid-line or mid-identifier** (byte-level, per the task), so many middles are not self-contained statements. The line-span eval subset in section 7 is the interpretable complement.
- **No amoeba-safe mid-codepoint splits here:** the corpus is pure ASCII (ids 2..129), so byte-level splits coincide with character-level splits; a non-ASCII source could split inside a UTF-8 sequence, which would still round-trip byte-for-byte but would not be valid UTF-8 per piece.
- **FIM at 25.7M remains untested** — this artifact supplies the data and the metric, not a result.
- Nothing under `runs/` was read or written; the source corpus was only read.

## 9. Held-out infilling harness (usage)

Implemented in [scratch/bench_infilling.py](../scratch/bench_infilling.py). It decodes the `val`
split of this corpus, keeps the **transformed** examples only (403 of 840), builds the prompt as
the stored envelope up to and including `<|fim_middle|>`, generates greedily with the existing
`GlmArm` / `HfArm` plumbing imported from
[scratch/bench_playground_batched.py](../scratch/bench_playground_batched.py) (git-clean, not
modified), and scores the generated middle against the reference middle. The decoder re-derives the
packed-id structure itself (`id = 4 + byte`, `EOS = 2`, byte-string sentinels) and does **not**
import `fim_transform.py`, so builder and scorer can disagree. Decode check on the real val
split: 840 examples -> 403 transformed, 437 untouched, 0 malformed, sha256
`6836024115d94bd1f0bb25d3be1aeabe0eff7875b566fcb0282d95e247a3897e` (matches `meta.json`).

PSM/SPM order is deliberately **not** inferred: it is not recoverable from the FIM stream alone,
and for both orders the generation prefix is the stored envelope up to MID, so the eval is
order-agnostic (each row carries `order: "unknown_from_fim_stream"` plus the two raw context spans
as `context_first` / `context_second`).

### Metrics and output

- **PRIMARY `exact_match_rate`**: generated bytes up to the first EOS equal the reference
  middle bytes, byte for byte (section 7).
- `edit_similarity_mean`: mean of `1 - Levenshtein(g, ref) / max(len(g), len(ref))`, byte level.
- `line_exact_rate`: generation and reference have identical newline-delimited byte lines
  (the interpretable complement when a middle spans whole lines; a trailing-newline difference
  counts as a miss).
- `eos_compliance`: of the rows whose stop reason is decidable, the fraction that stopped at
  EOS rather than the byte budget or the token cap. Reported separately from score, never merged
  (E-122g). Rows containing U+FFFD are undecidable (`eos_unknown`) because
  `decode(errors="replace")` loses the raw bytes.
- `completion_bytes`: mean / median / min / max generated middle length.
- Aggregates also split `exact_match_rate` by single-line vs multi-line reference middles.
- Every per-example row preserves `case_id`, `prompt`, `generation`, `reference_middle` and a
  pass/fail flag per metric; `--control NAME` additionally reports each arm's exact-match
  delta, factor and whether the section 7 bar (factor >= 2 **and** >= 20 percentage points) is met.

### Commands (venv only; `--device cpu` is the default so this is GPU-free by default)

```powershell
# mechanical self-test: no model, no GPU; exits non-zero on any failed check
& .\.venv\Scripts\python.exe scratch/bench_infilling.py --self-test

# inspect the 403 decoded cases (no model)
& .\.venv\Scripts\python.exe scratch/bench_infilling.py --dump-cases scratch/infill_cases.json

# FIM arm vs the no-FIM control -- the GPU-time step; checkpoints live under runs/
& .\.venv\Scripts\python.exe scratch/bench_infilling.py ^
    --arm fim=glm:glm-5.3-flash-from-scratch/runs/e122i_fim/final ^
    --arm base=glm:glm-5.3-flash-from-scratch/runs/e122i_nofim/final ^
    --control base --budget-bytes 160 --batch-size 8 --device cuda --out scratch/infill_e122i.json
```

Other flags: `--limit N` (debug), `--max-tokens N` (default 1000), `--budget-bytes N`
(default 160, the E-122g setting), `--batch-size N` (default 1), `--no-repeat-ngram N`
(0 = plain greedy, the section 7 regime), `--split val`, `--corpus <dir>`, `--out <json>`.
`--arm name=glm:<checkpoint-dir>` is repeatable; `hf:<model-dir>` is also accepted.

### Self-test result (2026-10-15, CPU, no model)

Command: `& .\.venv\Scripts\python.exe scratch/bench_infilling.py --self-test` -> **17 checks, 0
failed, exit 0 (PASS)**. It proves: Levenshtein sanity; feeding the reference middle back scores
exact-match 1.0 and edit similarity 1.0; a corrupted middle scores exact-match 0.0 with partial
edit credit; an exact match stopped at the byte budget is exact but eos-compliance false (the two
signals stay separate); `line_exact` catches a trailing-newline difference; the stop-reason
classifier separates EOS / budget / cap and marks U+FFFD output undecidable; synthetic PSM and SPM
packed examples decode to the right prompt (ending at MID) and reference; an untouched example is
skipped and a malformed envelope is rejected; aggregation math is exercised on a mixed synthetic
batch; and the real val split decodes to the `meta.json` counts (840 / 403 / 437, 0 malformed)
with the recorded sha256.

The model path was not exercised against a real checkpoint: the FIM / no-FIM checkpoints live
under `runs/`, which this task was told not to touch, and the single GPU was busy. A
load-error dry run (`--arm dead=glm:scratch/does_not_exist`) imported torch and the arm
plumbing, captured the error and wrote the report (exit 0), so the wiring above the generation
calls is exercised.
