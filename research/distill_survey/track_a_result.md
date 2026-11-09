# Track A — context eval probes: RESULT (2026-09-09)

Row 29. Eval-only Dynamic-NTK + StreamingLLM probes on the 226.5M model
(16L/d1024, 16Q/4KV, head_dim 64, CodeLlama tokenizer, TRAINED ctx 1024,
CSN val 978,944 tokens / 956 blocks @1024). No training code touched:
knobs live in `src/model.py` (eval-path helpers) + `scripts/eval.py` flags,
defaults OFF; `sanity_check` 4/4 PASS after the change; CPU selftest
(`scripts/ctx_probe.py --selftest`) 5/5 PASS. Artifacts:
`runs/ctx_probes/20260909T184503Z/` (full) + `quick_vram_probe*/` (VRAM
probes). GPU: RTX 3000 6 GB, fp32 eval, batch 8/4/1 by ctx; 0 OOM,
0 interruptions. Wall ~= 2.1 h total (one thermal throttle stretch,
e1 w1024s4remap 3501 s vs ~220 s typical - normal per AGENTS 4.5).

## Knobs implemented (defaults OFF, contract safe)

- A1 Dynamic-NTK: `apply_rope_scaling(model, "dynamic-ntk", factor=1)`
  (transformers 5.16.1 native `rope_parameters: {rope_type: dynamic,
  factor: F}`; base scales by `(F*seq/max_pos - (F-1))**(dim/(dim-2))`
  only beyond max_position_embeddings=1024, auto-resets below).
  eval.py: `--ctx N --rope-scaling dynamic-ntk [--rope-factor F]`.
- A2 StreamingLLM: `build_streaming_sink_mask(L, W, sink)` - query i
  attends j iff `j<=i` and (`j>=i-W+1` or `j<sink`) - plus OPTIONAL
  cache-relative positions `streaming_position_ids` (pos_shift, cap
  sink+W). eval.py: `--stream-window W [--sink-tokens K]
  [--stream-positions absolute|remapped]`.

## CSN val loss (full 978,944 tokens per point; delta vs same-ckpt @1024)

| point | target/final | delta | sft_v2_e1/final | delta |
|---|---|---|---|---|
| base_1024 (knobs off) | **1.8512** | - | **2.0466** | - |
| noop_2048 | 2.0796 | +12.3% | 2.3298 | +13.8% |
| noop_4096 | 2.8834 | +55.8% | 3.2255 | +57.6% |
| **ntk_2048** | **1.7740** | **-4.17%** | **1.9764** | **-3.43%** |
| **ntk_4096** | **1.8836** | **+1.75%** | 2.1263 | +3.90% |
| w1024s4abs_2048 | 1.8392 | -0.65% | 2.0520 | +0.26% |
| w1024s4remap_2048 | 3.2992 | +78.2% | 3.4897 | +70.5% |
| w512s4remap_2048 | 4.1027 | +121.6% | 4.2911 | +109.7% |
| w1024s0remap_2048 | 3.3084 | +78.7% | 3.4969 | +70.9% |
| w1024s4remap_4096 | 4.1752 | +125.5% | 4.3545 | +112.8% |

(abs = sink+window mask with ABSOLUTE positions; remap = StreamingLLM
pos_shift cache-relative positions. VRAM peak: 4.39 GB @1024/2048,
2.63 GB @4096 b1 - all far under 6 GB; KV cache confirmed non-binding:
~16 KiB/token, note 04.)

## Baseline provenance (IMPORTANT)

- `runs/target/final` was bit-exact-restored from checkpoint-4000 on
  2026-09-09 (row 19 gotcha 1). base_1024 today = **1.8512**, EXACTLY the
  trainer-recorded best eval_loss @4000 (1.8512). The on-disk
  `runs/target/final/eval_report.json` (1.8641) is STALE - it was computed
  2026-09-07 on the PRE-restore final (= end-of-run weights). New
  baselines must re-run eval; the driver stores both numbers.
- `sft_v2_e1/final` was never reweighted: base_1024 = **2.0466** matches
  its eval_report.json to 4 decimals (clean cross-check).

## Findings

1. **Dynamic-NTK is a free lunch at 2x context.** @2048 the eval-only
   knob IMPROVES val loss vs the trained-1024 baseline (-4.2% base /
   -3.4% e1) while the raw no-knob extension costs +12-14%. The extra
   conditioning context helps more than the rescaled positions hurt.
2. **@4096 eval-only is borderline.** Pure-LM surface +1.75% (inside the
   +/-2% gate); instruct surface +3.9% (outside). Raw extrapolation
   without NTK is destroyed (+56-58%) - positions, not context, break.
3. **StreamingLLM pos_shift is WRONG for this from-scratch model.** The
   cache-relative remap collapses at every width (W512 worse than W1024,
   sink 4 ~= sink 0, all +70-126%). The model relies on correct RELATIVE
   distances, not on attention sinks: it has no sink specialization
   (trained causal-full at 1024, unlike StreamingLLM's pretrained LLMs).
4. **The useful window variant is mask + absolute positions:** last-W
   tokens + 4 sinks keeps @2048 within -0.65%/+0.26% of baseline - a
   zero-training ~2x context lever for INFERENCE where you control the
   mask, but it caps usable context at ~W+sink for recall.

## DECISION for Track B (adoption_plan B gate)

The A gate fired: "if NTK @2048 already holds val loss within ~+2%, skip
straight to a 4096 target in B." It holds with margin (-4.2% / -3.4%),
and @4096 eval-only is +1.75% (base) / +3.9% (e1). Therefore:

- **Track B targets ctx 4096 with YaRN factor 4** (original_max 1024),
  PREREQ unchanged: vram_probe at ctx 4096 with the 8-bit-Adam default
  FIRST (6 GB card; if the training probe OOMs at 4096, fall back to the
  planned 2048/factor-2 target - eval-only NTK@4096 is already +1.75%,
  so the fallback loses little).
- B gates must check BOTH surfaces (base + e1) + the @1024 forgetting
  guard; e1@4096 is +3.9% eval-only, so a 4096 fine-tune must lift the
  instruct surface too.
- Do NOT adopt StreamingLLM pos_shift for this ladder (finding 3);
  window+sink masks with absolute positions remain an inference-time
  option only.

RESULT (2026-09-10, Track B executed - TASKS row 30): 4096 CONFIRMED by
the mandatory 8-bit-Adam vram_probe (@4096 peak 4.24 GiB alloc / 5.02
reserved, 6.6k tok/s; @2048 3.34/3.60 - fallback not needed). Fine-tune
1000 steps @4096 from runs/target/final (config-gated train.init_from;
HF-native yarn factor 4, attention temp 1.1386): **val @4096 1.7035 =
-8.0% vs the 1024 baseline (-9.6% vs the eval-only NTK@4096 1.8836)**,
forgetting guard @1024 +1.15% (1.8724), honest extrapolation @8192
2.1240 (+14.8% vs baseline but -1.7% vs eval-only NTK@8192 - coherent
2x). Instruct re-SFT on the extended base = row 41 (user-gated).

## 8k-16k extension (2026-09-09/10, row 29 follow-up)

Presets long/extreme added to scripts/ctx_probe.py. Same protocol (full
val stream per point; tail remainder dropped by packing: 974,848 tokens
@8192 / 970,752 @12288 / 966,656 @16384 = 99.5-98.7% of val). VRAM NOTE
(MEMORY 32): 12288/16384 exceed the 6 GiB card and Windows sysmem
fallback silently spills (peak_allocated 6.15 / 7.9 / 8.17 GiB measured,
never an OOM); pace collapses accordingly (wabs_16384 23 s/batch, 29 min
for one point).

| point | target/final | delta | sft_v2_e1/final | delta |
|---|---|---|---|---|
| noop_8192 | 3.9409 | +112.9% | 4.3577 | +112.9% |
| ntk_8192 | 2.1616 | +16.8% | 2.4586 | +20.2% |
| ntk_12288 | 2.4096 | +30.2% | 2.7336 | +33.6% |
| ntk_16384 | 2.6159 | +41.3% | 2.9597 | +44.6% |
| w1024s4abs_8192 | 1.8236 | **-1.50%** | 2.0444 | **-0.11%** |
| w1024s4abs_16384 | **1.8512** | **+0.00%** | not planned | - |

(e1 ntk_16384 + w1024s4abs_8192 were lost to two machine-sleep incidents;
RESCUED 2026-09-10 05:29-05:45 UTC on MUO4QK5 (RTX 4000 8 GB) under the
standby guard (standby-timeout-ac 0, was 60 min): ntk_16384 2.9597
(wall 270 s, peak 7.92 GiB - fits the 8 GB card, no spill) and
w1024s4abs_8192 2.0444 (wall 296 s, peak 4.45 GiB), zero interruptions.
Evidence: runs/ctx_probes/20260909T214837Z_e1_rescue (completed rescue);
20260909T224806Z = the earlier partial, recovered from stdout.)

Findings (extends the four above):

5. **Eval-only NTK is NOT free past ~4-6k.** The NTK curve bends up:
   1024 1.8512 -> 2048 1.7740 -> 4096 1.8836 -> 8192 2.1616 (+16.8%) ->
   12288 2.4096 (+30.2%) -> 16384 2.6159 (+41.3%). Raw noop at 8192 is
   already total collapse (+112.9%). The instruct surface bends the same
   way (e1 @16384 2.9597 = +44.6% vs its 2.0466 baseline; slightly above
   the +37-42% trend-band prediction - honest note). NTK holds the model
   COHERENT to 16k but nowhere near the gate.
6. **The window+sink absolute mask holds the baseline FLAT to 16k**:
   @8192 -1.5%, @16384 1.851194 vs baseline 1.851243 = +0.00%. This
   from-scratch model's CSN predictions are effectively LOCAL: capping
   each query to 4 sinks + the last 1024 tokens costs nothing measurable
   at 16x the trained context - but recall of anything older than the
   window is gone by construction (the flat loss measures the local
   surface, not long-range recall). e1 confirms it: w1024s4abs_8192
   2.0444 = -0.11% vs its 2.0466 baseline.

(Track B decision UNCHANGED by the extension: 4096 stays the fine-tune
 target; 8k+ eval-only is +17..+41% and the window mask is inference-only.)

## Reproduce

```powershell
& .\.venv\Scripts\python.exe scripts\ctx_probe.py --selftest
& .\.venv\Scripts\python.exe scripts\ctx_probe.py --points full
& .\.venv\Scripts\python.exe scripts\eval.py --config configs/target.yaml `
    --ckpt runs/target/final --ctx 2048 --rope-scaling dynamic-ntk `
    --skip-gen --report-out runs\ctx_probes\manual_ntk2048.json
```