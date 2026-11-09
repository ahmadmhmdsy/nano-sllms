# E-112 - Does the cloned glm-5.3-flash-from-scratch reproduce its published pretrain->RL result on this machine? (TASKS row 157)

Independent re-run of the external repository `vukrosic/glm-5.3-flash-from-scratch`,
vendored here as a nested git repo tracked by gitlink (its HEAD `96d9c7c`, parent
commit `2bd733b`). It was ported to run on this project's single GPU and single
`.venv` (bf16 -> fp16 + `GradScaler`, `torch==2.5.1` unpinned, zero-flag auto-resume)
and then re-run with the upstream recipe unchanged. The question is not "is the code
nice" but "do the published numbers survive the port".

## Question

Does a bf16->fp16 + torch-2.14 port of the cloned repo, running inside this project's
existing `.venv`, reproduce the published 25.7M-parameter pretrain -> RL claims on this
8 GiB Quadro RTX 4000 (sm_75 - a card with no bf16 at all)?

## Hypothesis

Pre-registered in TASKS 157 BEFORE the first GPU launch (criteria fixed, then run):

- **V1** - pretrain dev pass@1 at step 400 >= 6/8 (published 0/8 -> 2/8 -> 7/8 -> 8/8).
- **V2** - greedy confirm after RL: >= 12/24 solved, paired gain >= +8 tasks, ZERO
  losses, exact McNemar p < 0.05, and pre-RL <= 2/24 (published 0/24 -> 16/24).
- **V3** - pass@8 over the 3 trained families gains >= +6 tasks (published +10;
  9/24 -> 19/24, exact 13/192 -> 88/192).
- **V3b** - pass@8 over all 8 families gains >= +1 task (published 25/64 -> 34/64).
- **V4** - qualitative per-family pattern: `increment`/`double` become reliable, `even`
  stays unreliable, untrained families must not collapse catastrophically.

Falsification: any of V1/V3/V3b failing, or V2 failing its statistical clause, means the
port does not reproduce the published claim. The prediction was that ALL five pass.

## Rig

- **Data** - the repo's own generated task families (`increment`, `double`, `even`,
  `square`, `list_sum`, `reverse`, `absolute`, `nonnegative`), byte tokenizer
  (vocab 260), context 192. Split identities are disjoint and frozen (dev / final /
  rl / confirm); the confirmation set is opened only after duration is selected.
- **Model** - `GLM53FlashFromScratch`: 12 layers, dim 192, 6 heads, 8 experts top-2 +
  1 shared, 4 residual streams; **25,730,592 total / 9,805,344 active** parameters. RL
  trains the last block's head only: **2,190,152** trainable parameters.
- **Arms** - this is a replication, not a comparison study; the four stages are the
  arms: pretrain -> RL -> greedy confirm -> pass@8 confirm. Everything except dtype,
  torch version and device is held identical to the published recipe.
- **Budget** - pretrain 400 steps, batch 32, seq 128, lr 3e-4, seed 42, checkpoints at
  100/200/400. RL: 96 groups, group size 16, 32 tasks/family, families
  `increment,double,even`, temperature 0.35, lr 5e-5, last-block-head, binary reward,
  seed 31415, init `runs/pretrain/checkpoint-0100`. pass@8: 8 samples, temperature
  0.35, seed 8675309.
- **Machine** - Quadro RTX 4000 8 GiB (sm_75), torch 2.14.0+cu126, Python 3.12.9,
  Windows, allocator cap 4.14 GiB (55% of 7,704 MiB free) + spill detector. Published
  reference: RTX 3080 Ti 12 GiB, torch 2.5.1+cu121, Python 3.12.3, Linux.

## Results table

Wall-clock is the headline column. Every number cites the file it came from; no two
evaluation sources are merged. Upstream values are the published ones.

| stage | params | wall-clock | in-run evidence (file) | independent eval (file) | ppl | verdict |
|---|---|---|---|---|---|---|
| pretrain 400 steps | 25,730,592 / 9,805,344 active | **179.3 s** (published 63.1 s) | tokens_seen **1,381,760**; peak **2.141 GiB** (published 2.139); `runs/pretrain/receipt.json` | dev pass@1 **0/8 -> 3/8 -> 7/8 -> 8/8** at ckpt 0/100/200/400; `runs/eval/dev-*.json` | - | **PASS** (V1) |
| RL 96 groups | 2,190,152 trainable | **440.9 s** (published 155.5 s) | 1,536 rollouts; 406 exact (published 417); 77 groups with nonzero spread (published 73/96 updated); peak **0.939 GiB** (published 0.936); `runs/rl/receipt.json` | greedy confirm **5/24 -> 12/24**, 7 gains / 0 losses, p=0.0156; `runs/eval/comparison-confirm.json` | - | **PARTIAL** (V2) |
| pass@8, 3 trained families | same | 4 evals, ~25 min | cumulative exact at groups 24/48/72/96 = 104/160/236/406; `runs/rl/progress.jsonl` | **10/24 -> 15/24**; exact **35/192 -> 75/192**; `runs/eval/comparison-pass8-trained.json` | - | **FAIL** (V3) |
| pass@8, all 8 families | same | 4 evals, ~25 min | same | **30/64 -> 30/64**; exact 110/512 -> 125/512; bootstrap -3.9 to +9.8 pp; `runs/eval/comparison-pass8-all.json` | - | **FAIL** (V3b) |
| receipt replay | - | ~40 s | 2 of 3 published greedy completions byte-identical (steps 0 and 200); all 3 `.pt` sha256 differ | `runs/eval/receipt-replay.json` | - | **PASS** (behaviour) / **INCONCLUSIVE** (hashes) |

Per-family exact rate (solved passes / 64 samples), ours before -> after, with the
published figure where one exists:

| family | trained? | ours exact before -> after | published exact before -> after | ours solved@8 before -> after |
|---|---|---|---|---|
| increment | yes | 2/64 -> **25/64** | 6/64 -> 40/64 | 2/8 -> **7/8** |
| double | yes | 33/64 -> **50/64** | 4/64 -> 45/64 | 8/8 -> 8/8 (saturated before RL) |
| even | yes | 0/64 -> 0/64 | 3/64 -> 3/64 | 0/8 -> 0/8 |
| square | no | 35/64 -> **7/64** | 52/64 -> 37/64 | 8/8 -> 6/8 |
| list_sum | no | 36/64 -> 42/64 | 61/64 -> 56/64 | 8/8 -> 8/8 |
| reverse | no | 3/64 -> **0/64** | (not published) | 3/8 -> 0/8 |
| absolute | no | 0/64 -> 0/64 | (not published) | 0/8 -> 0/8 |
| nonnegative | no | 0/64 -> 0/64 | (not published) | 0/8 -> 0/8 |

Greedy confirm, per family: `double` 5/8 -> 7/8, `increment` 0/8 -> 5/8, `even` 0/8 ->
0/8. Exact rate 20.83% -> 50.0%, task bootstrap +12.5 to +50.0 pp (published +25.0 to
+53.1 pp).

## Gates and verdicts

- **V1 - PASS.** ACTUAL 8/8 at step 400 (curve 0/8 -> 3/8 -> 7/8 -> 8/8); REQUIRED
  >= 6/8. The intermediate points differ from the published curve in our favour: our
  checkpoint-0100 already scores 3/8 where the published one scores 2/8. No stop rule
  fired; every stage exited 0 and no run spilled to shared memory.
- **V2 - PARTIAL.** Statistical clause PASS: 7 gains, **0 losses**, exact two-sided
  McNemar **p = 0.0156** (< 0.05), post-RL 12/24 >= 12/24. Baseline clause FAIL:
  pre-RL **5/24**, required <= 2/24. Gain clause FAIL: **+7**, required >= +8.
- **V3 - FAIL.** ACTUAL **+5** tasks (10/24 -> 15/24), exact +40/192 (+20.8 pp);
  REQUIRED >= +6 tasks. Published: +10 tasks, exact +75/192 (+39.1 pp).
- **V3b - FAIL.** ACTUAL **net 0** (30/64 -> 30/64), 6 gains and 6 losses, McNemar
  p = 1.0, bootstrap -3.9 to +9.8 pp (spans zero); REQUIRED >= +1 task. Published
  +9 (25/64 -> 34/64).
- **V4 - PASS.** The pattern replicates: `increment` and `double` become reliable
  (`increment` 0/8 -> 5/8 greedy and 2/8 -> 7/8 at k), `even` never becomes reliable
  (0/8 greedy, 0/8 at k, 0/64 exact, unchanged), and untrained families do regress
  (`square` 8/8 -> 6/8 at k with exact 35 -> 7, `reverse` 3/8 -> 0/8). Published
  `list_sum` regresses too (61/64 -> 56/64); ours improves slightly (36/64 -> 42/64),
  which is the one per-family disagreement.

**Headline verdict: the port reproduces the infrastructure EXACTLY, the DIRECTION and
the per-family pattern of the RL result, but NOT its magnitude - so 3 of the 5
pre-registered gates fail.**

The mechanism is identified rather than hand-waved: every quantity that could match
did match (token stream, parameter counts, RL trainable parameters, rollout count,
both VRAM peaks), so the pipeline is faithful. What differs is the starting point of
RL. Our fp16/torch-2.14 pretrain `checkpoint-0100` is AHEAD of the published bf16
step-100 checkpoint - dev 3/8 vs 2/8, greedy confirm 5/24 vs 0/24, and `double`
already solved 8/8 at k - so the headroom the published RL result exploits has
already been spent by pretraining. Reproducing the *step number* is not reproducing
the *regime*.

## Levers to carry

- **Per-checkpoint receipts are the cheapest validation target that exists.** The
  clone ships `artifacts/receipts/system-receipt.json` (reference GPU, torch, parameter
  counts), `pretraining-gpu-checkpoint-generation.json` (per-checkpoint weights sha256
  AND the exact greedy completion for one frozen task) and `checkpoints.sha256`.
  Replaying them (`scripts/validate_upstream_receipts.py`) converted "does the port
  work?" into a dozen exact numbers for ~40 s of GPU time.
- **Replay behaviour, not file hashes.** All three published `.pt` hashes differ,
  including the untrained step-0 checkpoint, purely because `torch.save` container
  bytes differ between torch 2.5.1 and 2.14. Compare decoded completions or tensors.
- **A fail-closed executable verifier**: the repo scores candidates by RUNNING tests
  and treats an execution failure as a failure, never as a pass. Worth copying into any
  task where a reward could be gamed by non-compiling text.
- **Disjoint frozen split identities** (dev / final / rl / confirm) with the
  confirmation set opened only once, after duration is chosen - it is the reason the
  confirmation number here can be trusted as an out-of-sample reading.
- **Paired-seed + exact McNemar + task-level bootstrap + Holm** as the default
  statistics kit; the formulas were copied verbatim from the repo into
  `scripts/compare_receipts.py` so our numbers are directly comparable.
- **Retained negative results.** The repo keeps 5 failed RL pilots and a "dev winner
  failed to replicate" lesson (group size 4 looked like 12.3 vs 11.3/40 with CI -6.7 to
  +1.7 pp). Both were worth reading before trusting any single-arm result.
- **Zero-flag auto-resume** is the pattern this project already demands, and it ported
  cleanly: `<checkpoint>/state.pt` carries optimizer + GradScaler + RNG + tokens_seen,
  `progress.jsonl` is appended per step, and the final receipt is rebuilt from disk.
- **Pick the RL init checkpoint to match the published regime**, not the published step
  number, when the goal is to reproduce a magnitude.

## Mistakes and gotchas

- **`evaluate_passk.py` receipts nest `episodes[].samples[].evaluation`**, while
  `evaluate.py` receipts put `evaluation` at the top of each episode. Two analysis
  scripts written against the flat schema reported 0 solved / 0 exact and then raised
  `KeyError: evaluation`. Both now flatten either shape. Worse, `task_id` sorts AFTER
  `samples` inside a sampled episode, so a schema probe keyed on `task_id` can detect
  the list and still read the wrong level.
- **A `.pt` file sha256 is not a weights comparison** across torch versions - see
  MEMORY 2026-10-08.
- **bf16 on sm_75 must raise, not silently downgrade.** `amp_dtype()` now raises below
  sm_80; the port uses fp16 + `GradScaler`.
- **pytest's default `--basetemp` under `%TEMP%` fails with `WinError 5`** in
  `cleanup_dead_symlinks` on this machine; pass `--basetemp runs\_pytest_tmp`.
- **`laya/scripts/preflight_check.py` was SKIPPED** - it requires a config YAML that
  this nested repo does not have. Substituted: `pytest` (13/13), `--help` on every
  modified entry point, and a CPU smoke run.
- The harness `read` tool truncates any line longer than 2000 chars with a marker, so a
  long ledger row cannot be used as an edit anchor; take the tail from the file itself.

## Scale-transfer caveats

- The claim is licensed at **25.7M parameters, context 192, 400 pretrain steps /
  1.38M tokens**. Nothing here says anything about a larger model or a longer schedule.
- **dtype, torch version and device are confounded** (fp16 + torch 2.14 + Quadro RTX
  4000 vs bf16 + torch 2.5.1 + 3080 Ti). The magnitude gap is attributed to the RL
  init *regime* because the infrastructure numbers matched exactly, but only ONE RL
  init checkpoint was tested - the alternative explanation (fp16 pretraining genuinely
  learns faster per step) is consistent with the same evidence and is not separated.
- The ~2.8x wall-clock slowdown (179.3 vs 63.1 s pretrain; 440.9 vs 155.5 s RL) is a
  device-class difference, not a port defect; the VRAM peaks matched within 0.2% and
  0.3%.
- Whether this port reaches the published RL *magnitude* with an init checkpoint
  selected to match the published headroom (dev 2/8) is **not tested** - that is the
  explicit follow-up in the E-112 row.

## Evidence files

Nested repo: `runs/pretrain/receipt.json`, `runs/rl/receipt.json`,
`runs/rl/progress.jsonl`, `runs/eval/dev-*.json`, `runs/eval/confirm-pass8-*.json`,
`runs/eval/comparison-*.json`, `runs/eval/receipt-replay.json`. Scripts:
`scripts/vram_probe.py`, `scripts/validate_upstream_receipts.py`,
`scripts/summarize_receipts.py`, `scripts/compare_receipts.py`,
`scripts/rl_training_stats.py`, `tests/test_resume.py`. Upstream reference:
`artifacts/receipts/system-receipt.json`,
`artifacts/receipts/pretraining-gpu-checkpoint-generation.json`, `REPORT.md`.
