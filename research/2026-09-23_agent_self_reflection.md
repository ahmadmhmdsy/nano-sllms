# Agent self-reflection — E-65 session (2026-09-23)

Trigger: user request "reflect on yourself and identify all the issues and suggest all the
solution". Scope: every failure I caused or missed during the E-65 (L3) session, with root
causes and the fixes now in place. Evidence-first; nothing softened.

## 1. Runtime failures I shipped (six resume events)

| # | Failure | Evidence | Root cause | Fix now in place |
|---|---|---|---|---|
| F1 | mask_tokens shape bug (RuntimeError [48,256] vs [176]) | pretrain startup crash | full-shape tensor assigned to a boolean-mask selection (selection is 1-D) | boolean-index rule in MEMORY; fixed pre-launch |
| F2 | CUDA OOM at micro_batch 48 | pretrain launch 1 | vocab logits (B×S×30522, fp32 in backward) never budgeted; repo's vram_probe discipline skipped for a new architecture | micro 32 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (peak 4,610 MiB); VRAM gate FAIL recorded honestly |
| F3 | probe AttributeError 'BertForMaskedLM' has no 'head' | first probe boundary, mid-pretrain | probe written for one model class, run against two (LayaDecisionModel vs raw MLM) | hasattr(model, "head") branch |
| F4 | probe device mismatch (type_emb cpu vs cuda) | probe boundary | shared-encoder wrapping does not move NEW modules to GPU | .to(device) on any fresh wrap |
| F5 | argparse NameError at ladder startup | exit before any training | validated with ast.parse only — syntax checks cannot catch undefined names | preflight C2 (pyflakes) + C3 (--help smoke) |
| F6 | random.Random.permutation at stage-B boundary | first stage-B batch | numpy API called on a stdlib Random; aug_item had zero test coverage | stdlib shuffle; preflight C5 runs aug through a real batch |
| F7 | smoke-as-written validated nothing (self-caught) | first preflight run "resumed epoch 5 update 5233" | run_stage reads cfg["out_dir"]; my _smoke suffix changed only a local var — the smoke resumed real checkpoints and ran ZERO training steps | cfg["out_dir"] override in smoke mode; R6 below |

F7 is the most instructive: a smoke that cannot fail is worse than no smoke. The first
green "PASS" hid an empty training loop; it was caught only by reading the smoke output
critically (a resumed checkpoint state is disconfirming evidence in a smoke context).

## 2. Silent near-misses and design misses (worse than crashes)

- **N1 — data-leak near-miss:** a `config_name=` kwarg bug silently disabled the
  typed-test AND PhishNChips exclusion loaders during corpus build. Caught only because
  the zero-overlap verification printed 0 for every source. **Fix:** build_corpus_l3.py
  now hard-fails (SystemExit) on any overlap > 0 — eval leakage is fatal, never a counter.
- **N2 — PyYAML duplicate keys:** one flat yaml carried micro_batch 48 (pretrain) and 8
  (ladder); PyYAML keeps the LAST — pretrain would have run at the wrong batch silently.
  **Fix:** preflight C1 duplicate-key detector (a custom loader that records repeats).
- **N3 — spec vs registration:** the user asked for ~50M; I registered 6L/H512 without a
  one-line param calculation and it measures 35.1M. Recorded honestly, but the delta
  belonged in the pre-register, not the post-mortem. **Rule R5:** param count is computed
  at pre-registration time.
- **N4 — augmentation/test mismatch:** option-order shuffle (repack) does not reproduce
  the block-surgery eval transform; G3 got WORSE with aug (0.185 vs 0.385 unaugmented).
  Position/option-order bias is documented ([NAACL 2024 findings](https://aclanthology.org/2024.findings-naacl.130.pdf));
  stronger mitigations are training on multiple permutations of the SAME item, or
  surface-form/PMI debiasing ([Holtzman et al. 2021](https://ar5iv.labs.arxiv.org/html/2104.08315)) — with the caveat that probability-mass
  fixes do not always transfer ([arXiv:2305.14596](https://ar5iv.labs.arxiv.org/html/2305.14596)); validate on held-out.
- **N5 — script datasets unsupported:** HF datasets ≥3.0 removed script loaders
  ([PR #7592](https://github.com/huggingface/datasets/pull/7592)); spamassassin and sroie died at build time. **Fix:** probe one item
  per manifest source before a big build; migrate sources to parquet.
- **N6 — replay sizing:** 15% uniform replay fully held retention here (−0.0003) but the
  same ratio forgot for L2 (−0.062) — replay strategy (balanced/reservoir sampling,
  buffer size) matters as much as the ratio ([arXiv:2203.10317](https://arxiv.org/pdf/2203.10317v1.pdf),
  [arXiv:2505.12512](https://arxiv.org/html/2505.12512)). Next ladder: per-source balanced replay.

## 3. How I worked (agent-process failures)

- **P1 — wrong number in the permanent ledger.** I wrote stage-A exit macro 0.6329 into
  EXPERIMENTS.md and TASKS.md from memory before reading train_summary.json; the actual
  value is 0.6625. Caught one turn later by luck (the full-curve read). **Rule R2: ledger
  metrics are parsed from source JSON in the same program that writes the ledger — never
  typed from memory.**
- **P2 — batch kills:** a no-op edit (old_string == new_string) and an edit-before-read
  rejection each killed an entire multi-call program. Preconditions are verified BEFORE
  batching; old==new is always a bug.
- **P3 — run_code parse errors (twice):** a `//` comment inside Python, and Python
  raw-string syntax inside a JS template literal. A parse error means NOTHING ran —
  every edit in those programs was a no-op. Windows paths in JS strings need escaped
  backslashes.
- **P4 — here-string quoting fights:** PowerShell multi-line appends replaced with the
  temp-file pattern (write tool → Get-Content | Add-Content).
- **P5 — walked away twice into startup crashes** (F5, F6) before any preflight existed.

## 4. What held (keep)

- Auto-resume: 6/6 resumes, zero progress lost — checkpoints save before every boundary.
- Eval isolation: 20,004 ban hashes, zero overlap in every source, now enforced by hard-fail.
- Honest reporting: VRAM gate FAIL, params 35.1M vs ~50M, 1/4 gates — all recorded as FAIL/PASS as measured.
- Scheduled bench monitoring: trend signal during 79 min of pretraining; absolute probe numbers correctly quarantined as monitoring-only.
- Replay correctness: the E-63 forgetting failure mode did NOT recur (−0.0003 vs −0.062).

## 5. Standing rules adopted

- **R1** No unattended long launch without `laya/scripts/preflight_check.py` (C1 dup-keys → C2 pyflakes → C3 --help → C4 data loads → C5 --smoke GPU 1-batch).
- **R2** Ledger metrics parsed from source JSON at write time.
- **R3** Corpus builds hard-fail on any eval-set overlap.
- **R4** New helpers get a 1-case test or a smoke path through the trainer.
- **R5** Param count computed at pre-registration; user-stated size deltas are registered risks.
- **R6** A smoke must demonstrably execute the training loop (updates > 0 in a fresh _smoke dir), not merely import and exit.
