# Smoke/reuse SOP — runtime profiles (2026-09-23)

## What this is

A self-contained facility that records EVERY profiling lesson from smoke
runs into one durable store, so future runs (and future agents) reuse the
BEST known knobs instantly instead of re-deriving them from scratch.

## Files (one home per concern)

``
diacritizer/src/pipeline/profiles/registry.json   # the data (entries with evidence)
diacritizer/src/pipeline/profiles/store.py        # ProfileStore: list/show/recommend/record/promote/retire + batch_for shortcut
diacritizer/src/pipeline/profiles/__init__.py     # public API: from pipeline.profiles import ProfileStore
diacritizer/scripts/profile_ctl.py                # least-step CLI (list|show|recommend|record|promote|retire)
scratch/_profiles_selftest.py                     # the unit smoke (also validates retire/exact-match logic)
``

## How a new profile gets born and promoted (the loop)

1. record  (status=experimental) right after the smoke sweep runs.
2. promote (status=validated)  once a second independent run reproduces it.
3. auto-run picks it up: scripts consult `ProfileStore().recommend()`.
4. ANY deviation later -> retire(id, reason) so the stale knob never silently misfires.

## Anatomy of an entry

```json
{ "id": "gate_bench/gate_bench.512.default",
  "process": "gate_bench",                   // what class of job this is (gate_bench | train_smoke | fullval_eval | ...) 
  "status": "validated",                     // validated > fallback > experimental > retired
  "match": { "device": "sm_75_6gib", "any_model": true },   // or "model_bucket": "16Lx272"
  "params": { "batch": 512 },                // the knobs that get auto-applied
  "evidence": { "wall_s_at_1": 249, "wall_s_at_512": 10, "peak_vram_gib_at_512": 1.5, ... },
  "notes": "..." }
``

Matching policy (hard-coded in store.py): exact model-bucket > any-model >
fallback; experimental/retired entries are NEVER auto-applied.

## The smoke process itself (what the sweep did, reusable recipe)

1. Parity fixture FIRST: run bs=1 through the부 legacy path on a fixed
   subset and keep the bytes (runs/_benchk/sadeed_b1.prd.txt).
2. Sweep: fixed source + --limit 5000 (fast enough to iterate, long enough
   to catch batch effects), batch keys [1, 64, 256, 512, 1024, 2048, 8192, 16384].
3. Record wall_s + peak_vram_gib + a DER cross-check for every point
   (bs batching must produce metric-identical DER, or document the tie-flip).
4. Choose by: throughput plateau that keeps peak VRAM under ~25% free and
   shows no allocator thrash cliff. 512 was the winner; 2048+ collapsed.
5. Wire: script asks the store (no manual re-derivation).

## Hard-learned gotchas baked into the code/comments

- SDPA masks: float mask = additive bias (scrambles scores!); use BOOL mask.
- Exist-check before every write; never bulk-delete run dirs (user data).
- vram_probe only valid when idle; NVML may hiccup under WDDM load.
- big batch on TWO fronts (model VRAM + padded batch T) — peak_vram reported
  in bench JSON line is the profiling tool.

## Where this plugs into the future (scalability hooks)

- New process type (e.g. `fullval_eval`): record a profile with the same
  recipe; train.py loops can consult it the same way bench.py does.
- Cross-machine/device: change "device" key per bucket; the store can hold
  per-device entries for the same process.
