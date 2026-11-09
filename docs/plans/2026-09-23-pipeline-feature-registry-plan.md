# Plan: D-line pipeline feature registry + always-on monitoring (2026-09-23, user-requested)

## Goal
Make the diacritizer codebase a small registry-based pipeline so that:
- features live in clean importable modules (features/, evals/, monitors/),
- experimental features are one-use opt-in (never silently active),
- validated features are promoted to the default pipeline and AUTO-apply even when the
  config/agent forgets to set them,
- every training process automatically runs smoke-check + full monitoring + internal
  multi-metric eval + (at close) external gate profile without anyone remembering.

## Codebase issues this fixes (from the project's own incident history)
| issue in history | registry answer |
|---|---|
| Eval instrument silently wrong (E-60a argmax-through-wrong-head; rebuild incidents; 272-wide warm-load mismatch) | one eval module, ONE head path, load-shape assertions, per-class always printed, multi-metric always |
| Detached train "deaths"; forgotten log teeing (pwsh-89 re-run) | monitor always on, tees jsonl + human tail, auto VRAM residency check |
| Newton-Schulz fp32 NaN discovered only by a ad-hoc probe (E-63) | trained-process divergence detector (auto stop-early + mark FAIL in monitor jsonl) - permanent |
| Label smoothing/logit-adjust compare trap (loss scales not comparable across arms) | eval module always emits comparable fields separately from loss (acc, per-class, der-lite) |
| Logit-adjust looked fine on slice but collapsed rare classes unseen | R1/R2: full-val + 4-gate multi-metric is the DEFAULT eval set |
| Config generator dropped warm-init silently | registry validates configs at load (required keys per feature, warm-init assertion, path existence) |
| temp scratch scripts vanish in git but carry lessons | permanent mechanics live in tests/ (selftest suite) run as smoke preflight, scratch never referenced by pipeline |
| One-use scripts re-implemented per experiment | features/ modules importable + registered by name |

## Layout (target, incremental)
diacritizer/src/pipeline/
  registry.py     # FeatureSpec: name, maturation (experimental/validated), stages, hooks
  events.py       # TrainEvent bus: on_start/on_step/on_eval/on_checkpoint/on_end
  monitor.py      # ALWAYS-ON monitor (jsonl : runs/<phase>/monitor/log.jsonl):
                  #  step, train_loss, val_loss/acc, lr, scaler_scale, grad_norm,
                  #  VRAM used/reserved + spill guard, pace s/step, NaN/inf auto-flag,
                  #  best tracking, elapsed, ETA; writes summary at on_end
  smoke.py        # runs config's 60-step smoke first (unless --skip-smoke with reason),
                  # isolated run dir (isolated dir deleted on PASS automatically)
  evals/internal.py  # full unseen val: loss, acc, der-lite, per-class (single pass)
  evals/external.py  # 4+1 gate bench + eval_der full set (bench.py wrapper)
  features/
    optimizer_muon.py (validated NS fix inside)
    losses.py (label_smoothing, logit_adjust)  # logit_adjust stays EXPERIMENTAL
    shared_kv.py, shared_layers.py (experimental, E-63 verdicts attached)
    block_dropout.py (NEW, experimental, for the smaller-model track M3)
  tests/ (promoted from scratch/skv_test.py + e63 selftest pieces: permanent preflight)

## Maturation policy (answers "one use vs permanent")
- experimental: applied ONLY when config lists it (pipeline.features: [name]).
- validated: applied by default in every train; config can only turn it OFF explicitly
  with an override flag (opt-out, never opt-in).
- promotion path = [ experimental ] -> [ experiment closes with recorded verdict + full
  multi-metric ] -> [ user confirms ] -> [ validated ] in registry.py; registry diff is
  the audit trail (add promotion line to EXPERIMENTS row on promote).
- validated-at-birth list (already confirmed by history): warm-init loader, GRADScaler
  skip-on-overflow, autosave checkpoints w/ rotation, zero-flag auto-resume, monitor,
  smoke gate, internal full-val eval, external gates profile at run close, early-stop
  (patience from config), NaN/divergence auto-stop.

## Multi-metric eval contract (what every eval emits)
loss / acc / der_lite / per-class vector / (external) DER, DER_nocase, WER,
text_preservation per gate; the trainers' printed line NEVER carries a single number.

## Rollout in stages (each independently shippable)
S1 monitor.py + events bus + validated default (auto on every train) - do first
S2 evals/internal.py as the single in-loop eval; delete duplicated eval logic
S3 smoke.py mandatory preflight (skip requires --skip-smoke "reason")
S4 external.py gates subprocess (light limit probes in-loop, full at close)
S5 registry + features/ import + config validation
S6 promote checkpoint-average + EMA + KD arm as validated features behind config knobs
