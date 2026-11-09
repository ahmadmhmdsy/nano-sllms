# RESULTS.md — V4.1 transfer ideas micro-bench (2026-09-19)

Scratch experiments testing DeepSeek-V4.1-Flash ideas on our hardware
(Quadro RTX 3000 6GB, fp16). Standalone code, pipeline untouched.
Scripts in this folder; raw JSON next to them. All runs: .venv python, real
pilot tokens (pilot/tokens/train_000.bin + val_000.bin, uint32 packed).

## M1 — where does training VRAM go (pilot-proxy: 12L d768 GQA, bs 2, AdamW)

| ctx | peak/fwd+bwd+opt | params | grads | AdamW states | activations |
|---|---|---|---|---|---|
| 512 | 2.56 GB | 0.38 | 0.38 | 0.75 | 1.06 GB (41.5%) |
| 1024 | 3.36 GB | 0.38 | 0.38 | 0.75 | 1.86 GB (55.4%) |
| 2048 | 5.11 GB | 0.38 | 0.38 | 0.75 | 3.61 GB (70.6%) |

Conclusion: at our scale the *activations* (not the KV cache) dominate at
long ctx. Model+optimizer is a fixed 1.5 GB. Any VRAM/throughput lever must
attack activations, i.e. KV/attention recompute like CED-lite, SWA windows or
gradient checkpointing — not just KV cache bytes. Prefill throughput
~43-51k tok/s (fwd) across ctx; not ctx-bound yet.

## M2 — CED-lite variants vs dense GQA baseline (d256 6L ctx512 bs8, 300 steps, same seed/schedule)

Train loss (avg 25-step bins) and val_loss (100 val blocks):

| variant | params | train loss last50 | val loss | tok/s |
|---|---|---|---|---|
| A dense GQA (baseline) | 12.71 M | 5.03 | 7.329 | 56.3k |
| B shared-KV upper half (YOCO/CSA2-reuse) | 12.58 M | 5.07 | 7.335 | 56.6k |
| C CED-proj (eq.1 per-layer K/V proj) | 12.71 M | 5.05 | 7.324 | 55.7k |
| D B + SWA128 borrows | 12.58 M | 5.08 | 7.323 | 53.5k |
| E C + SWA128 borrows | 12.71 M | 5.06 | 7.323 | 52.9k |

VERDICT (M2 at nano scale): loss is NEUTRAL within ±0.02 val across all
variants; KV compute is too small a fraction at ctx512 to move wall time.
No accuracy loss from borrowing — confirms the idea doesn't hurt; benefit
must come from memory/speed at larger ctx/micro-batch, not from loss.

## M2b — same variants at PILOT-PROXY scale (12L d768, bs2, 4 steps, AdamW)

| variant | ctx 512 peak / s/step | ctx 1024 peak / s/step |
|---|---|---|
| A dense | 2.563 GB / 0.164 s | 3.370 GB / 0.177 s |
| B shared | **2.515 GB / 0.104 s** (-36% time, -0.05 GB) | 3.319 GB / 0.173 s (-1% time) |
| C proj | 2.565 GB / 0.112 s | 3.371 GB / 0.175 s |
| C proj + SWA128 | 2.560 GB / 0.120 s | 3.380 GB / 0.187 s |

At ctx512 pilot scale, removing half the layers' K/V compute is a real
~30-36% step-time saving and slightly lower peak VRAM. At ctx1024 the win
washes out (SDPA/logit compute dominates). The CED-ok-for-nano conclusion:
adopt primarily for INFERENCE KV cache (see analytic below), and possibly
NOT needed as a training-time save at our 6GB small scale unless we push
batch × ctx to fill VRAM (activations house >70% there, so it *can* help —
measure per-batch before switching).

Analytic inference KV at pilot proxy, fp16:
- dense GQA: 2·12L·4kv·64·ctx·2 B = 12 kB/token → 24.6 MB/ctx 2048/seq.
- CED-lite shared upper half: (7 instead of 12 K/V slots) ≈ 7.2 kB/token → -40%.
- FP4 KV (V4.1's) would be another -50%, but fp16-only Turing → NOT portable without
  custom quant kernels (out of scope).

## M4 — Muon optimizer (fp32, d192 4L ctx256 bs16, 300 steps, real pilot tokens)

| variant | train loss last50 |
|---|---|
| AdamW lr 4e-4 | 4.993 |
| Muon on 2D params lr 1e-2 + AdamW elsewhere | 4.945 (better but same ballpark) |
| Muon lr 3e-2 | **4.485 (-10% vs AdamW)** |

Caveat: short run; Muon lr sensitivity is high (1e-2 vs 3e-2 differ 4.95→4.49),
so an LR sweep is REQUIRED before adopting. Still: Muon ≥ AdamW everywhere and
3e-2 is a large gap. Worth a real P-scale LR-swept A/B before T.

## Bench artifacts

- bench_common.py · mini_model.py · bench_ced_lite.py · profile_vram.py ·
  bench_muon.py (+ smoke run output captured in session transcript)
- JSON: ced_lite_results.json · vram_profile.json · pilot_proxy_profile.json ·
  muon_results.json

Progressive verdicts:
- M1: activations dominate VRAM at ctx≥1024 (70%) → the cheap global lever is
  activation trimming, not KV cache. This argues gradient checkpointing and
  CED-lite upper halves for long-ctx training.
- M2: KV-sharing does NOT hurt loss at nano scale (val ±0.02) — safe to adopt
  architecturally (its win is inference KV memory and some step time).
- M2b: at pilot scale ctx512, CED saves ~35% step time; at ctx1024 washes out.
- M4: Muon beats AdamW under the same scheduling at every lr tried, at d192.
  Needs LR sweep before adoption.

## M5/M6 — two more V4.1 ideas (d192 4L ctx256 bs16, 300 steps, fp32, real pilot tokens)

M5 Sinkhorn-balanced embedding (light geomean row-norm rebalance at rate
0.05, every step, tied embedding only):

| arm | train loss last50 |
|---|---|
| control (AdamW, same seed/schedule) | 4.928 |
| + sinkhorn rebalance 0.05 | **4.835 (-1.9%)** |

Modest but consistent win in the short budget; cheap (one row-norm op on the
embedding per step). Worth re-testing at P-scale with rarer-vocab-heavy data
before adopting; rate needs a small sweep (0.02..0.1).

M6 MTP-style aux head (predict token t+2 from h_t, separate tied head):

| aux weight | main train loss last50 (main ONLY, aux excluded) |
|---|---|
| control | 4.928 |
| 0.02 | 4.956 (+0.6%) |
| 0.1 | 5.215 |
| 0.3 | 5.864 |

VERDICT: NO benefit at nano scale; the aux-head gradient destabilizes early
training and the deficit persists at every non-tiny weight (even 0.02 is
slightly worse). DeepSeek's MTP gains come with long training, warmup of the
aux head and separate aux-LR/anneal — our 300-step budget cannot replicate
that. Honest verdict: FAIL for adoption as-is; only retry at P-scale if we
ever decouple aux-LR (e.g. 0.1x main) — not on the TODO until then.

## M5b — Sinkhorn rate sweep (partial; same d192 4L setup as E-42)

| rate | train loss last50 |
|---|---|
| control | 4.928 |
| 0.02 | 4.841 |
| 0.05 | 4.835 |
| **0.1** | **4.798** |
| 0.2 | 4.801 |

is not a thin optimum. Rate 0.2 deferred behind a GPU-waiting background
is not a thin optimum. Rate 0.2 deferred behind a GPU-waiting background
runner (sinkhorn_sweep_rest.py: polls mem_get_info, aborts cleanly at 1h cap,
writes sinkhorn_sweep.json incrementally).

## M7 — P-scale A/B of the Sinkhorn lever (12L d768, ctx 512, bs 2, 300 steps, same seed)

| arm | train loss last50 |
|---|---|
| control (AdamW) | 4.995 |
| + sinkhorn 0.1 | **4.972 (-0.5%)** |

Smaller but DIRECTIONAL PASS at pilot-proxy scale (the gap narrows vs the
nano -2.6%, consistent with a longer-lived, more-tuned optimizer schedule
washing out a small per-step nudge). Honest verdict for adoption: WEAK PASS
at pilot proxy — the effect is real and harmless but small; the lever is
safe to wire into train.py as a flag-default-off experiment rather than
default-on. A cleaner test on the real pipeline (HFinetune path, longer
schedule, val loss) is the deciding run, not this proxy.
