# EOS at example granularity — task-distribution SFT (E-122g)

**Date:** 2026-10-14
**Agent:** delegated subagent (EOS/termination hypothesis)
**Status:** IN PROGRESS — training launched

## Hypothesis

The byte model cannot terminate because its packed corpora contained only byte tokens
(ids >= 4; ByteTokenizer maps text bytes to 4..259 and reserves pad=0/bos=1/eos=2/sep=3),
so it never saw id 2. Its `decode()` stops only at eos_id=2, so an unterminated model
runs to the byte budget forever.

## Prior failed attempt (do NOT repeat)

E-122f: `runs/e122f_eos_cc` = 600 steps on `data/reference/corpus_codeparrot_v1`
(raw Python, EOS once per ~8.9 KB **whole file**) from `runs/e122_ref600/checkpoint-0330`.
Result: extracted pass 11/71 -> **0/71**, raw output collapsed into degenerate
repetition. Mechanism: EOS was an alien token with enormous loss; 600 steps of gradient
on one never-seen token collapsed the distribution, and raw-Python pretraining teaches
corpus imitation, not task-following. **This is the third time raw-Python pretraining
made task behaviour worse.**

## This experiment

Teach EOS at **example granularity** (one short playground function per example) on the
**task distribution**, not raw code. Corpus already built:
`data/reference/bytes_pg_replay_eos/` (uint16, meta: shape=reference-solutions,
append_eos=true, eos_id=2, byte_offset=4; train.bin 5,752,986 B = 2,876,493 ids /
16,800 examples; val.bin 287,884 B = 143,942 ids / 840 examples). Built from the 88-rung
playground, per-task-type 400 instances. EOS per example, ~171 bytes/example.

### Trainer choice (task 1)

`glm-5.3-flash-from-scratch/scripts/train_pretrain.py` with `--corpus ... --init-from ...`.

Why: it is the **only** script in the GLM line that consumes a packed `.bin` corpus
(`--corpus` appears in just `train_pretrain.py` and `evaluate_corpus.py` across
`scripts/`). `train_rft.py` is rejection-sampling (generates rollouts + executable
sandbox rewards, no `--corpus`); `train_aspect_lora.py` trains one LoRA per
`--aspect` from selected tasks. The E-121 "reference SFT" that moved the donor from
0/8 to 7/8 viable families was itself `train_pretrain.py --corpus` with
`--init-from` (it writes stage="fine_tune_initialization"; `--init-from` help text
says "E-121 fine-tune stage"). Corpus shape `reference-solutions` is exactly the
task-format corpus. `CorpusWindows` reads non-overlapping (seq_len+1)-id windows, so
the embedded EOS ids sit in-stream and are learned in context.

## Exact command (task 2)

Run from the parent root `E:\python projects\nano_SLMs`, background, logged under scratch/:

```powershell
$env:PYTHONUNBUFFERED=1
& .\.venv\Scripts\python.exe glm-5.3-flash-from-scratch\scripts\train_pretrain.py `
  --corpus data\reference\bytes_pg_replay_eos\train.bin `
  --val-corpus data\reference\bytes_pg_replay_eos\val.bin `
  --init-from glm-5.3-flash-from-scratch\runs\e122_ref600\checkpoint-0330 `
  --output glm-5.3-flash-from-scratch\runs\e122g_eos_ex `
  --steps 800 --batch-size 4 --grad-accum 8 --sequence-length 1024 `
  --learning-rate 3e-5 --lr-schedule cosine --warmup-steps 10 `
  --val-windows 32 --checkpoint-fraction 0.05 --vram-cap-gib 5.0 `
  --amp-dtype fp16 --device cuda --seed 42 2>&1 | Tee-Object -FilePath scratch\e122g_eos_ex_train.log
```

Hyperparameters cloned from the e122f receipt (bs 4 x accum 8 = effective 32, seq 1024,
cosine, warmup 10, seed 42, vram cap 5.0 GiB) except corpus (replay-EOE instead of
codeparrot) and output name. lr 3e-5 because 1e-4 measurably hurt in the E-122e A/B.
800 steps x 32 x 1024 = 26.2 M tokens ~= 1.5 epochs over the 2,876,493-token corpus
(2809 windows @ seq 1024; 800 x 4 = 3200 windows).

## Success signals (tasks 3-5)

- Extracted/strict pass on dev/core, per-rung 1, budget 160, max-tokens 256, batch 8.
- **Completion LENGTH must drop** (terminating before the 160-byte budget), not just pass count.
- Completions must be **valid Python / coherent**, not degenerate repetition
  (`self._context.context.context...`) — that was how E-122f was caught.

## Log

- 2026-10-14 ~15:5x — **LAUNCHED**, background job `pwsh-292`, cwd `E:\python projects\nano_SLMs`,
  log `scratch/e122g_eos_ex_train.log`, output `glm-5.3-flash-from-scratch/runs/e122g_eos_ex`.
  Start event confirmed: `initialized_from=glm-5.3-flash-from-scratch/runs/e122_ref600/checkpoint-0330`,
  `steps=800`, `effective_batch=32`, `vram_cap_gib=5.0`.
- **Early signal:** step 1 loss = **1.2257** (grad norm 1.99). Compare E-122f (same init, codeparrot
  whole-file EOS): step 1 loss = **2.6719**, grad norm 5.40. The replay-EOS corpus is on-distribution,
  so EOS is not an alien token this time; e122f's collapse came with a ~1.45-nat-higher initial loss.
- Steady pace ~4.8-6.7 s/step -> 800 steps ~= 70 min.

