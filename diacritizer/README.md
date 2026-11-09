# diacritizer — Arabic diacritization specialist (D-line)

Char-level bidirectional Transformer encoder + 15-class diacritic head.
Design: `../research/arabic_diacritization/DESIGN.md` (v1 APPROVED).
Plan: `../docs/plans/2026-09-11-arabic-diacritization-specialist.md`.
Current experiment sequence and the generalization path:
`../docs/plans/2026-10-05-diacritization-generalization-path.md`.
Verdict audit hub: `../research/verdict_audit.md`.

## Status

**Live truth is `../research/experiments.yaml` (the `diacritizer/E-xxx` rows) and
`../research/verdict_audit.md`.** This README states the line's shape, its stable
contract, and where to reproduce from — never volatile numbers, which belong in
the registry and the run directories.

- A0a (R50) DONE: `src/tokenizer.py` (char vocab), `src/labels.py` (15-class
  schema), `src/passthrough.py` (parse/reconstruct, D6 copy-by-construction),
  `scripts/selftest.py` gate runner.
- A0b/A0c and the D-smoke/D-pilot rows are long DONE and superseded; their
  verdicts live in `../research/EXPERIMENTS.md` (historical ledger) and the audit
  hub.

## Current best — shape of the recipe (as of 2026-10-06)

Read the *number* from the registry; reproduce from these structural elements.

| element | value |
|---|---|
| architecture | 18L / 64h / 256ffn / 8 heads / 2 KV, RoPE (≈ 1.08 M params) |
| data bed | `data/diac/v3ctx256` (ctx 256, PAD=0) |
| registered metric | mean DER over `GATE_EXTERNAL` = fadel_test + sadeed25 + wikinews2014, **lower is better** (`scripts/train.py`) |
| weights | `runs/diac/e70_c4b_ctx256/gate_probe/best_gate_weights.pt` (step 15000) |
| inference | `scripts/e67_p6_vote.py`, window 256 / stride 128 (majority voting) |
| verdict rows | `diacritizer/E-070` (weights), `diacritizer/E-071` (voting procedure), `diacritizer/E-072` (held-out corroboration) |

**The winning configuration is weights + procedure.** The sliding-window voting
transform is a free, deterministic improvement over a single forward pass and
ships *with* the checkpoint — quoting the checkpoint alone understates it.
`wikinews2024` is excluded from the registered metric (contaminated).

### Warm-start lineage (verified against the configs, not remembered)

    e67_p1_lm64/final/model.pt                  stage-1 char-LM, ctx 128
      -> e67_c1_ft64_fullv3                    ctx 128, v3 bed, AdamW
         (C4 warm-starts from C1's gate_probe/best_gate_weights.pt)
      -> e69_c4_ctx256                         ctx 256, v3ctx256, 60k steps
         (C4b warm-starts from C4's final/model.pt, i.e. step 60000)
      -> e70_c4b_ctx256                        60k further steps at ctx 256
         best = gate_probe/best_gate_weights.pt @ step 15000

Configs: `../runs/diac/e67_gen/c1_ft64_fullv3.yaml`, `c4_ctx256.yaml`,
`c4b_ctx256.yaml`. The two ctx-256 configs differ in exactly two keys (`phase`,
`pretrained_init`); each carries a fresh 60 000-step budget.

Note the lineage is intentionally not uniform: C4 inherited C1's *gate-best*
probe while C4b inherited C4's *final* weights. That asymmetry is recorded, not
tidied away — see the registry rows.

## Run

    & .\.venv\Scripts\python.exe diacritizer\scripts\selftest.py --phase all

A0 exit gate: `--phase all` == 3/3 PASS with a0b/a0c no longer SKIPPED.

## Guarantees proven by selftest a0a

- reconstruct(input) == input after removing exactly the class marks that
  follow an Arabic base letter; every other codepoint (emoji, latin, digits,
  RLM/ZWJ/ZWSP/BOM, tatweel, Quranic marks, lone marks) survives verbatim.
- Invalid mark combinations raise QuarantineError (sample quarantine path,
  never a silent guess).
- Base stream encodes to vocab with no UNK for the fixture corpus.
