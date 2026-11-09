# Plan: per-layer Shared KV for the D-line stage2b2500 arch (E-62)

**Request (user, 2026-09-22):** take the stage2b2500 model arch and add to it
shared KV for each layer so each layer can (a) use the past KV, (b) create its
own KV, or (c) reindex/edit past KV; then smoke-train the model and report.

## Interpretation (recorded)

"Shared KV" = a per-forward cross-layer KV store within the bidirectional
diacritization encoder (the published E-54c shared-KV thread, applied here to
the D-line stage2b2500 arch). At each layer, config decides the mode:

| mode   | behaviour |
|--------|-----------|
| new    | layer projects its own K/V from its own input, WRITES them to the shared store, and attends (standard behaviour) |
| reuse  | layer computes only its Q; READS the last shared/past KV entry (pre-RoPE), re-ropes it at its own positions, attends. Creates no new KV of its own. |
| edit   | reuse + a learned SharedKVEditor transforms the past KV before use (learnable reindex + per-head residual scales). |

Config surface:`model.kv_pattern` = `new` (default; byte-identical to the
pre-extension arch), `cycle3` (new/reuse/edit repeating), `reuse_all`,
`edit_all`, or an explicit per-layer list. The first layer is always `new`
(there must be a writer before any reader).

## Why the cache is pre-RoPE

Consumers reindex AND re-rope the past KV at their own positions, so a stored
entry is only useful when un-rotated; every consumer jams its own position
frame onto whatever it reads. That makes reuse/edit layers POSITION-ADAPTIVE,
the actual "reindex/edit" semantics rather than blind copy.

SharedKVEditor (init == exact identity):

    x' = x + a0*(roll(x,-1)-x) + a1*(roll(x,+1)-x)   # tanh-gated residual reindex
    k' = k' * (1 + tanh(k_scale)); v' = v' * (1 + tanh(v_scale))  # per-kv-head

Both K and V reindex TOGETHER (a (k_i, v_i) entry is a paired slot; the
reindex must keep points coherent). reindex gates init 0 and scales init 0 =>
at init every edit layer is EXACTLY the reuse layer; the identity path is
proven, deviation from it is learned.

## What will NOT change

- Every existing diac_* config (no kv_pattern key) keeps its exact model,
  params, state-dict keys and forward numerics (proved by pre/post patch
  equivalence capture, scratch/skv_test.py).
- train.py contract unchanged: same smoke/gate plumbing; reset_last_n_layers
  now rebuilds reset blocks WITH their kv_mode (composition fix, no-op for
  every existing config).
- The gold runs/diac/stage2b2500 deployment artifacts are untouched; the
  smoke writes to a NEW phase dir runs/diac/stage2b2500_skv.

## Files

- diacritizer/src/model.py - SharedKVEditor; GQAAttention/Block cache modes;
  DiacritizerModel kv_modes; build_from_config resolver _resolve_kv_modes.
- diacritizer/scripts/train.py - one-line fix: reset-last-N honours kv_mode.
- configs/diac_stage2b2500_skv.yaml - smoke config (cycle3, 60 steps).
- scratch/skv_ref_capture.py + scratch/skv_test.py - validation.

## Smoke gates (pre-registered before the run)

1. All-19 validation suite PASS (default-path loss/logit/params/keys/shapes
   identical pre/post patch; every kv mode trains; editor grads flow; identity
   init proven; resolver contracts).
2. Warm-load: stage1lm init loads 255 tensors; missing == editor params only.
3. 60-step smoke: val_loss improves vs the step-1 baseline; no divergence;
   final export + checkpoints written to runs/diac/stage2b2500_skv.
4. Developer evidence not part of the verdict: VRAM stays dedicated-only.

This is an architecture-SMOKE, not a quality gate: no gate bench, no
DER claims, no deployment recommendation.

## Known limitations / scope

- The editor's reindex family is deliberately tiny (identity / roll+1 /
  roll-1 blend). Richer editors (per-head rotations, learned index maps,
  sparse edit masks) are follow-ups, gated on what cycle3 shows.
- reuse layers keep their (warm but unused) k/v projections so state-dict
  shapes stay matched to the Stage-1 init and every load path stays simple;
  they receive no gradients and cost no FLOPs.
- Cross-layer only: there is no cross-STEP (decode-step) KV store; the cache
  dies with the forward pass, so training semantics are untouched.
