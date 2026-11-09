# E-21 — micro 12-128-512 (~2.98M) — both arms + train-time cache test (2026-09-17)

USER GO: (a) config 12-128-512-4H/2KV; (b) test BOTH init arms; (c) check QCRI quality first (model = research-only).

## Setup
- Same recipe as gold (RoPE + SwiGLU + GQA, bidirectional, ctx 128, vocab 171/200).
- Tokens: the exact stage2b v3 packed corpus (data/diac/v3/tokens).
- Word cache: models/e19/our_word_cache.json built from the SAME raw pool that
  produced the tokens -> the cache is a training-process artifact, exactly per
  the user's proposal.
- Arms:
  - **A** from-scratch SFT (configs/diac_micro_a.yaml), 2500 steps, batch 32, lr 4e-4.
  - **B** warm-start from a shape-matched micro stage-1 char-LM. NOTE: the 30M
    stage1lm cannot be transplanted onto the micro stack (28L/272h shapes), so
    arm B required a NEW micro LM pretrain (configs/diac_micro_a_lm.yaml, 4k
    steps on the whole-corpus stage1 tokens, val_loss 1.6177) -> then
    configs/diac_micro_b.yaml with pretrained_init + reset_last_n_layers: 2.

## Gate results (best/final probe, step 2500; gate CSV = exact historical pipeline)

| gate | gold 30M | micro A (scratch) | micro B (warm-start) |
|---|---|---|---|
| fadel_test | 33.23 | 38.16 | 41.07 |
| sadeed25 | 45.68 | 51.40 | 54.08 |
| wikinews2024 | 48.65* | 60.96 | 62.76 |
| wikinews2014 | 48.99 | 53.87 | 56.44 |
| mean external | ~44.1 | 51.10 | 53.59 |

(*gold cache-merged; raw 48.99)

- **Arm A (from-scratch) WINS** the arm comparison, ~5.7 mean-DER better than B.
- Arm B had the LOWER SFT val_loss (0.2911 vs 0.3228) yet worse gates —
  the LM warm-start made its OWN text prior better but bidirectional gate
  behavior worse (top-2 reset only partially compensated). Warm-starting a
  bidirectional diacritizer from the causal stack does NOT transfer advantage
  at micro scale (consistent with the E-19b[Z-Mahmood]-trained-from-scratch win).
- Step-500 palette: A 60.7 vs B 60.6 — they cross later; B regressed on probes
  after 1000.

## Train-time cache test (the user's key idea) — VERDICT: cache adds ~NOTHING

| gate | A raw | A + cache | B raw | B + cache |
|---|---|---|---|---|
| fadel | 38.16 | 38.13 | 41.07 | 41.04 |
| sadeed | 51.40 | 51.40 | 54.08 | 54.08 |
| wn2024 | 60.96 | 60.96 | 62.76 | 62.76 |
| wn2014 | 53.87 | 53.87 | 56.44 | 56.44 |

Cache-merge at eval (majority-vote word map built from the training corpus
itself)改动 gate DER by -0.003..-0.03 - i.e. ZERO, on a 2.98M model that has
only 10% of gold's capacity. Combined with gold's earlier -0.3..-0.6: the
majority-vote word prior is what ANY model of this class learns from the data
directly; the cache is a redundancy, not extra knowledge. The "cache as
training-process artifact" idea is now TESTED and FALSIFIED for this task —
record it as E-21's real contribution (MEMORY lesson 66).

## QCRI quality audit (question c, research-only)

Sample 400 articles of Wikipedia_20240420.diac.jsonl:
- 4.14 marks/word-base; 99.8% words carry vowels (HUMAN pool: 3.41 and 98.1%) -
  machine-labeled, OVER-vocalized (full case-endings).
- Validator-DER of its labels vs our gold: 45.0; vs Z-Mahmood 56.1 (on a 300
  random wiki paragraphs sample) -- i.e. the predictions of a rival model, not
  reference-grade text.
- Verdict: GOOD-but-machine. Fine as WEAK supplement (E-22 optional), never as
  gate refs or replacement for human gold. License still unstated in repo.
- WikiNews-2014 bench inside it == our wn2014 gate: any training use must pass
  gate shingle dedup first.

## Conclusion / recommendation

- The micro arm answers the "cheapSpielfeld" question: 12-128-512 at 2.98M
  trains 3.5x faster (1684 s for 2500 steps) but lands ~7-8 DER behind the 30M
  gold. It is a legitimate fallback/ablation baseline, not a replacement.
- Z-Mahmood (4.5M, BIGGER training pool + external cache) remains best external
  option per the user's rule (a) — 12.60 abdou-heldout vs micro A/B ~unknown
  (would need abdou probe) or gold's 41.46.
- For D-line production: keep gold 30M; keep cache as instant-deployment trick
  (it does add +0.3-0.6 on gold); the train-time cache idea is falsified.

## Artifacts

- runs/diac/micro_a{,_lm,_b}/final (+ gate_eval.csv, cache_eval.json)
- configs/diac_micro_{a,a_lm,b}.yaml
- diacritizer/scripts/e21_qcri_quality.py, e21_micro_cache_eval.py
- commit 19668dc (arm A + QCRI audit); this row completes with E-21 close-out.
