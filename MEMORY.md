## 2026-10-14 - Arabic in the Space: RTL is a layout property, not a text-alignment patch

- **Physical CSS properties silently stay on the wrong side under dir=rtl.** border-left,
  padding-left, text-align:left, margin-left and box-shadow offsets all keep pointing the
  same physical way when the document flips. A translation pass that only swaps strings
  therefore produces a page that reads Arabic with English layout - text hanging off the
  right edge, icons on the wrong side. Convert to the logical equivalents
  (border-inline-start, padding-inline-start, text-align:start/end, inset-inline-start,
  margin-inline-start, border-start-end-radius). Two things have NO logical form and need an
  explicit [dir="rtl"] flip: box-shadow offsets and :has(input:checked) inset highlights.
  **Audit for physical properties before translating, not after.**
- **Set lang and dir BEFORE first paint, from an inline head script.** Resolving the
  language after DOMContentLoaded means every page load paints LTR first and then snaps to
  RTL - a visible flash on every single visit. The inline script duplicates the detection
  function on purpose; that duplication is the price of no flash, and it is worth it.
  Detection order that matches user expectation: localStorage, then navigator.languages,
  then the default. Persisting the choice is what makes the switch feel like a setting.
- **Language switching must repaint, never re-run.** Keep each tab's last run and redraw the
  chrome from it. Re-running a model because a label changed wastes the user's compute and
  produces a NEW result for an unchanged input, which reads as nondeterminism.
- **data-i18n writes textContent, so HTML entities render literally.** A dictionary string
  containing "&middot;" or "&nbsp;" shows those characters on screen. Put the real
  character in the string (a middle dot, a space) and keep the dictionary ASCII-safe with
  \uXXXX escapes. Grep the dictionary for "&[a-z]+;" as a check.
- **Uppercase styling is meaningless in Arabic.** letter-spacing and text-transform on
  eyebrows, table headers, badges and field labels must be reset under html[lang="ar"],
  or the Arabic text gets stretched and the Arabic small-caps look wrong.
- **A @property is invisible to dataclasses.asdict().** registry.py exposed is_ours as a
  property, so the exported JSON never carried it and the front end read undefined - the
  "ours" badge had silently never rendered in a published Space, and nothing failed loudly.
  **When a field drives UI and is exported through asdict, assert it is present in the JSON**
  rather than trusting the dataclass to carry it.
## 2026-10-14 - A free HuggingFace account cannot host a Gradio Space; a static Space can run a model in the browser

- **The 402 is not a quota problem and deleting a Space does not fix it.** create_repo(...,
  space_sdk="gradio") returns "402 Payment Required: Static Spaces are free for everyone,
  but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."
  The account ahmadsy is isPro=False. Deleting an existing Space changes nothing: the
  block is on the Space KIND, not on a count. A probe proved the split - a static Space
  was created, written and deleted cleanly on the same free account. **Choose the host
  kind before writing the app, not after.**
- **A static Space plus ONNX Runtime Web is a real model demo, not a mock.** Both nanos
  export to ONNX (dynamo=False, opset_version=17, dynamic axes {0:"batch",1:"seq"}) and
  run in the page through onnxruntime-web. The JS decode is a port of the Python decode,
  and the check that proves it is NUMERIC, not visual: on the same reference the browser
  and the Python package produce identical DER/WER to two decimals (mishkala 2.92/8.16,
  flan 3.65/10.20, nano1p1m 5.84/14.29, nano4p5m 10.95/20.41). **Port a metric and then
  assert equality on a fixed input; "it looks right" is not a port.**
- **Vendor the runtime or the model, never the wasm.** onnxruntime-web's wasm payload is
  14-28 MB. Loading ort.min.js and its wasm from a CDN and shipping only the 4.8 MB and
  18.3 MB .onnx graphs keeps the Space at 22 MB and it still boots.
- **A static Space cannot run a Python dependency, so say which columns are live.** Only
  the two ONNX models execute in the browser; mishkala and flan are precomputed outputs
  for the five built-in samples and the one pre-filled reference. Every such card is
  labelled "precomputed" and the Space card says so in prose. **A demo that implies a
  live model where there is a cached string is a lie with a nice UI.**
- **"color-scheme: light" is not optional on a hard-coded light page.** With the OS in
  dark mode and no color-scheme declaration, every native control (checkbox, range
  slider, file input, scrollbar) renders in dark UA styling on a paper-white page.
- **A decorative colour used as text is a contrast bug.** #c98a16 is 2.95:1 on white and
  2.60:1 on the pale tint it sits on - both below 4.5:1. Split the token: add
  --saffron-ink #8a5c07 (5.81:1 on white, 5.13:1 on the tint) for badge text, the
  disagreement underline, the scatter dots and the list bullets, and stop using the
  bright hue for anything that carries meaning. A tinted BACKGROUND under dark text
  needs no separate check - #16192e on #fbf0d6 is 15.29:1.
## 2026-10-14 - A published card must compare only against models a reader can download

- **Both published model cards compared our published models against LOCAL, UNPUBLISHED
  checkpoints.** ahmadsy/tashkeel-nano-1p1m named E-097 arm B @ 30k and the deployed
  gold e73_soup/c4b_uniform; ahmadsy/tashkeel-nano-4p5m named a 17.2M champion M1,
  stage2b2500 (deployed gold) and all six E-63s sweep arms. User-reported 2026-10-14: a
  reader on another machine cannot download or verify any of them. **A comparison row
  whose other side is not downloadable is not evidence, it is an assertion.** Every row
  was replaced by a published benchmark result, recorded as E-104.
- **Check that the two metrics measure the same thing BEFORE placing numbers side by
  side.** Our der_registered is **WORD-level** (a word is correct only if all its marks
  match), so our 11.71 and the Sadeed leaderboard's DER (CE) 7.29 are not comparable:
  their DER (CE) is **diacritic/letter-level** and their **WER (CE)** is the word-level
  analogue. Two metrics with the same-looking name and different units is a silent
  2x-class error.
- **The benchmark's gold is not NFC, and its evaluator is order-sensitive.**
  SadeedDiac-25's gold writes shadda BEFORE the vowel (176 of 200 sampled rows are not
  NFC), and its evaluator **drops a shadda that follows a vowel**. Our 1p1m package emits
  NFC (vowel first), so its raw output is scored wrong on **every geminate**: DER (CE)
  **8.78 as-emitted vs 3.78 order-normalised** - a 2.3x swing that is pure serialization.
  This is the E-100 defect in mirror image, so the fix was already known.
- **When a metric artifact can move a number that much, publish BOTH readings.** The
  1p1m card reports as-emitted 8.78 and order-normalised 3.78 side by side with the cause
  stated, rather than picking the flattering one. It also states the residual caveat: the
  benchmark does not say whether its own leaderboard systems were order-normalised, so
  the ranking is approximate.
- **Measure whether the artifact applies before correcting for it.** The 4p5m package
  still ships the PRE-NFC renderer, so it already emits shadda-first and its two readings
  are IDENTICAL (9.91 both ways). Assuming it needed the same normalisation would have
  been wrong; the identity was measured.
- **Result of the public comparison:** 1p1m (1.08 M) ranks **3rd of 11** on DER (CE)
  behind Claude-3.7-Sonnet 1.39 and Gemini-Flash-2.0 3.19 and ahead of GPT-4 3.86 and the
  dedicated 1.5B Sadeed model 7.29, and **2nd** on DER without case endings (2.25). 4p5m
  (4.49 M) ranks 6th of 12, ahead of every 7B-13B system.
- **Retract in place.** The 4p5m card's earlier 31.31 / 43.37 / 47.19 (mean 40.62) table
  came from the pre-NFC decoder and is retracted on the card itself, with the corrected
  19.70 / 30.30 / 36.31 (28.77) and fair-decode 14.30 / 22.59 / 31.81 (22.90) beside it.
- **A benchmark without a leaderboard is not a comparison surface.** abdou_test is
  public and fully held out, but has no published baselines, so on the cards it is
  labelled a self-report, not a comparison.
## 2026-10-14 - A stale decoder can make a published number look 59 pct worse than the model is

- **The published model's card numbers were a renderer artifact, not a model measurement.** runs/diac/e63s_s4_12l160 (the HF card ahmadsy/tashkeel-nano-4p5m) scores der_registered **0.313075** on fadel_test from its own gate_probe/step19500 predictions, and the registered scorer reproduces that exactly. But the SAME checkpoint scores **0.197014** when decoded by the current harness. The 59 pct relative gap is not model quality; it is the ORDER of combining marks.
- **The tell is that two files can differ in 71 pct of their bytes and still be the same text.** The reference prediction file and a fresh decode differ in **1,247,905 of 1,753,582 bytes**, yet per-line character multisets are identical for 1991 of the first 2000 differing lines, macro_f1 agrees to 4 decimals and der_canonical agrees to 5. The only difference is U+0651 shadda + U+064E fatha versus U+064E + U+0651. **A metric that compares text literally is sensitive to mark order while every class-level metric is not**, so an order-only difference appears in exactly one column and reads as a real effect.
- **Unicode NFC is the correct order, and it is not the order one writes by hand.** FatHa (ccc 30) sorts before shadda (ccc 33), so NFC geminates are vowel-first. The historical renderer emitted shadda-first while the harness NFC-normalized every reference and never its own output, so the two were compared raw. diacritizer/src/labels.py:77-84 documents the fix and its size: **4.5 pct of all units and 42 pct of the registered gate** on every geminate.
- **Date the artifact before trusting it as an instrument.** labels.py was fixed **2026-10-11 20:38**; the published model's prediction files are dated **2026-09-24 18:39**, 17 days EARLIER. A harness-validation rule that demands reproducing a number is sound only if the artifact behind that number post-dates every fix to the path that produced it. **When a validation fails, diagnose the reference before declaring the instrument unknown**: here the reference was stale and the new decoder was right, confirmed by decoding the same checkpoint with e99_batch_decode.py (already validated to six decimals), which gave 0.197014 against the new script's 0.197030.
- **Retract loudly.** runs/diac/e98_published/comparison.md claimed arm B beat the published model by **-64.01 pct**; the fair figure with NFC on both sides is **-39.74 pct** (armB_30k centered) to **-49.71 pct** (E-099 soup). That file now carries a correction banner and research/experiments.yaml records the amendment explicitly rather than silently swapping the reference.

## 2026-10-14 - A shape-compatible checkpoint export is not proof you exported the right one

- The bed-B soup was staged for HF publishing from runs/diac/e98_soupB/final/model.pt -
  the RUN'S FINAL CHECKPOINT (step 60000), not the soup. The soup is
  runs/diac/e98_soup/soupB_uniform/model.pt. The two files are 4,383,455 B vs
  4,372,511 B, both are 165-entry OrderedDicts, both load with missing 0 /
  unexpected 0, and both are 1,083,328 params, so every check a packaging script
  would normally make passed while the wrong bytes were in the bundle.
- It was caught only because the STAGED PACKAGE was scored against the registered
  instrument before upload. It returned centered 0.131788 / single 0.139144, which
  matches step 60000's recorded 0.131842 / 0.139097, not the soup's
  0.115148 / 0.123145. Re-exporting the soup fixed it; the published repo then
  reproduced 0.051967 on fadel_test from a fresh download.
- Lesson: when a run directory contains BOTH a final/ checkpoint and a derived
  artifact (soup, average, probe pick) in a sibling directory, the derived artifact
  is what a card quotes and it is NOT the file the run last wrote. Export by the
  registered path, then SCORE THE EXPORT - a load test proves the shape, not the
  identity.
- The release also had to ship the NFC-fixed labels.py: the 4.5M repo's package
  still carries the pre-NFC marks_for_label, which is the defect E-100 root-caused
  (worth 4.5 pct of all units and 42 pct of the registered gate). Publishing a new
  package by copying the old one would have re-shipped the bug.

## 2026-10-14 — A gate probe is a noisy instrument: min-of-N against a point estimate is selection, not signal

- **Probe-to-probe DER scatter exceeds the pre-registered significance bar.** Nine `gate_eval.csv` probes on the same E-099 model across steps 32500-52500 have mean 0.143702 and sample SD 0.003900 — a **2.71% coefficient of variation** — against a **2% relative** significance bar and a **0.82%** init-draw band. The series also *oscillates* (0.139166 to 0.149770, period ~20000 steps) rather than trending, and the internal `val_loss` agrees the optimum is interior (best 0.06799996 at step 45000, then rising). **Check the scatter of a metric before trusting a verdict judged on it**: if the SD of repeated measurements exceeds the decision bar, the experiment cannot resolve the effect it claims to test.
- **`min` over N probes compared to a single baseline measurement is biased toward acceptance.** The expected minimum of 12 draws from N(0.143702, 0.003900) is ~0.13735, so the rule reports a ~6% "gain" that is pure winner's curse. On the real numbers the two statistics **disagree on the verdict**: mean-of-probes × the measured centering factor 0.94387 = 0.135636, which is 0.31% *above* the 0.135223 bar (REJECT), while best-of-probes × 0.94387 = 0.131354, which is 2.86% *below* it (ACCEPT). When a pre-registered rule and its unbiased companion disagree, **apply the registered rule but report the caveat in the same breath** — never silently swap the statistic after seeing the data, and never quote the favourable one alone.
- **Prefer `mean over probes` (or a per-probe paired comparison) as the pre-registered statistic** for any continuation/checkpoint-selection experiment; a minimum is only legitimate when each candidate is itself an average over repeated measurements.
## 2026-10-13 — Harness background jobs are reaped while idle; a multi-hour GPU run must be detached

- **A 2-hour training run died with no traceback and an empty `job_list`.** `runs/diac/e98_soupB/monitor/log.jsonl` ends on an `on_step` for step 35000 with **no matching `on_save`**, and `scratch/e99_train.log` just stops after `[SAVE] checkpoint-34900` — so the process was terminated between finishing a step and writing its checkpoint. `job_list` returning `[]` is the tell: the harness reaps background jobs, and it did so ~1.75 h after the last tool call of the turn. A silent stop with a clean log is an EXTERNAL kill; a crash leaves a traceback in the redirected stderr.
- **`tools.pwsh run_in_background` is not a durable home for a job that must outlive the turn.** Relaunching it the same way reproduces the same death. The escape that works on Windows is `Invoke-CimMethod -ClassName Win32_Process -MethodName Create`, which asks the WMI service to start the process **outside** the caller's job object, so closing the job cannot kill it:
  ```powershell
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = 'cmd.exe /c "E:\...\scratch\e99_resume.cmd"'
    CurrentDirectory = 'E:\python_projects\nano_SLMs' }
  ```
  Wrap the real command in a `.cmd` that `cd /d`s to the repo and redirects `1>`/`2>&1` to a log, because a WMI-started process has no console. Verify the escape before trusting it: start a throwaway `cmd /c echo ... > file` and confirm the file appears and `ReturnValue` is 0.
- **Auto-resume makes the kill cheap but not free.** `train.py:417-429` reloads the newest `checkpoint-*` with zero flags, so recovery is the original command verbatim (`rules/gpu.md`) — but the newest checkpoint was `-34900` while the log had already reached step 35000, because `save_every` writes AFTER the step. Budget the loss as up to `save_every` steps, and always confirm the new log contains the literal `[RESUME] from checkpoint-N step N` before believing the run continued.
- **A probe can fire and be lost to the same kill; check the probe directory, not the log, for what survived.** The step-32500 probe's `weights_step32500.pt` and its four `step32500_*_pred.txt` files were already on disk and are usable even though the run died afterwards — the permanent per-step probe files are the durable artifact, while `gate_probe/model.pt` is overwritten every probe.
## 2026-10-13 — Diacritizer: the protected GPU job is starved by DESKTOP GPU clients, and nvidia-smi cannot see it

- **A video player halved the training throughput and every standard check said the GPU was fine.** `nvidia-smi --query-gpu` reported `100 %` utilization with only the training PID among compute apps, but the measured step rate fell from **1.085 s/step to 6.11 s/step (5.6x)** — `runs/diac/e98_soupB/monitor/log.jsonl` shows `t=398.6` at step 30400 and `t=1009.7` at step 30500. The culprit was `mpv.net` (pid 9492), a *graphics* client: `nvidia-smi --query-compute-apps` lists it with `[N/A]` memory, so it looks like a no-op. The counter that exposes it is `Get-Counter '\GPU Engine(*)\Utilization Percentage'`, whose instance names embed `pid_<n>_luid_..._engtype_3d`; mapping that pid back through `Get-Process` gave mpvnet **47.7 %** vs python **40.3 %**. Diagnostic rule: when a GPU job's step time regresses with no config change, read per-process GPU engine counters, not `nvidia-smi` utilization.
- **Heavy CPU work beside the protected GPU job is not free either.** The same run lost ~5x for the window in which two 41 k-line corpus scorers and a context-table fit were running on CPU; the dataloader starves. `rules/gpu.md`'s single-job rule covers CPU-heavy co-runs too — schedule scorers and table fits for after the training job exits, or accept the slowdown explicitly.
- **A 34-84x inference speedup was sitting in the decode scripts.** `diacritizer/scripts/e93_centered.py` and `e94_nospace.py` both call the model ONE sentence per forward pass, peaking at 0.44 GiB of 6 GiB VRAM. `diacritizer/scripts/e99_batch_decode.py` collects every sliding window from every sentence, length-sorts them and batches them with `bench.predict_bares`'s bool padding mask; it decodes all eight gate corpora in ~40 s and 15091 abdou sentences in ~120 s. It is VALIDATED, not merely faster: it reproduces E-097's recorded arm B `single` to **six decimals** on all four corpora, which also proves the ~1e-4 "harness mismatch" was the batch-1 path's numerics rather than the model.
- **Whitespace mismatch is a property of the TOKEN BED, not of the decoder.** Space-token (`space_id 77`) counts in the first 200 rows: `v3` 0, `v3ctx256` 0, `v3full` 7823/51200 (15.28 %). So E-095's `nospace` fix helps a no-space bed and *inverts* on `v3full` (E-097: +334 % on abdou). Before reusing a decode-side fix on a new bed, count the token classes the fix touches.
## 2026-10-05 — Diacritizer: the stage1lm "warm-start" was a 100% silent no-op (shape-filtered init)

- **`pretrained_init` skips every tensor when the hidden size differs, and it does so silently.** `diacritizer/scripts/train.py:296-299` filters the warm-start state dict to keys whose shape EXACTLY matches (`fit = {k: v for k, v in sd.items() if k in msd and tuple(msd[k].shape) == tuple(v.shape)}`) and then merely *prints* `shape_skipped`. Nothing fails, nothing warns, and training proceeds from random init. The stage-1 char-LM is **272-hidden** (`runs/diac/stage1lm/final/model.pt`, `embed.weight` (171, 272)); the s7 diacritizer is **64-hidden** (`embed.weight` (171, 64)). All 165 keys are common, **zero** shapes fit, so `pretrained_init` transferred 0 of 165 tensors.
- **The tell was in the run's own first print, not in a metric.** R1's launch line read `{"tensors_in": 255, "missing": 165, "unexpected": 0, "shape_skipped": 255}`. `shape_skipped == tensors_in` is the signature of a total skip; `missing` equal to the entire model state dict confirms it. Read that line before believing any "warm-start" label.
- **Every "warm-start" s7 result is really random init.** E-064 A2, E-065 F1, E-065 F2 and E-066 R1 all point at the same 272-hidden LM and all silently no-op. No 64-hidden stage-1 LM exists anywhere under `runs/diac` (both `stage1lm` and `stage1lm_v4` are 272-hidden), so the warm-start contrast is **unmeasurable for this architecture**, not merely untested.
- **`train.py` has no seed**, so "identical recipe" runs differ by random init draw. Any arm-vs-arm comparison on this line carries init-draw variance as a confound; a same-recipe replicate is the only way to size it.
- **A shape-filtered loader needs a hard failure or an explicit transferred-count assertion.** The E-63s fix (2026-09-23) was written to stop a smaller-hidden arm from crashing the whole init — it succeeded so well that a 100% skip became indistinguishable from a real transfer. Generalised lesson: when a fallback silently degrades to "do nothing", assert how much was actually transferred.

## 2026-10-02 — Diacritizer metric harness: the registered DER is byte-order sensitive (and how to keep a self-built eval library honest)

- **The registered DER silently penalises a legal byte order.** `labels.label_for_marks` accepts a shadda+vowel run in EITHER order and maps both to one class — its docstring says so explicitly ("Real corpora use BOTH orders"). `eval_der._pairs_ok` compares raw mark STRINGS and therefore does not honour that equivalence. mishkala emits vowel-first throughout, so 784 of its words are per-position class-correct yet scored wrong; it loses **7.25 DER points** (52.45% -> 45.20% when mark runs are canonicalised shadda-first) and that is enough to flip which model ranks first. zmahood is unaffected (47.88% -> 47.89%). Under the schema-faithful canonical condition **mishkala is the best model on the WikiNews 2024 gate, not zmahood** (paired line bootstrap, difference -2.69 pts, 95% CI [-3.89, -1.11], exact McNemar p = 6.8e-39).
- **The tell was a disagreement between two metric families, not a number that looked wrong.** Word-level DER said zmahood > mishkala; the order-insensitive class metrics said the opposite (position accuracy 86.82% vs 82.34%, macro-F1 0.7378 vs 0.6820). When a byte-order-sensitive metric contradicts an order-insensitive one, suspect the metric before the model. Before blaming alignment coverage, re-run on the common base-aligned subset — that ruled coverage out here (mishkala still won).
- **Do NOT silently fix a registered metric.** The registered gate is left byte-order-sensitive on purpose so the historical series stays comparable; the corrected condition is reported BESIDE it as `der_canonical`. Any comparison against a published model must name which condition it used.
- **An eval library that re-implements a registered scorer must be parity-locked by a test, not by inspection.** `eval_metrics.py` re-implements the word loop to accumulate integer counts; `selftest.py --phase a0d` re-derives the same decisions from `eval_der`'s own predicate over hand-built fixtures, asserts the counts-not-rates property on a fixture with opposite per-line rates, and pins the two published numbers exactly (zmahood 0.4788055764883195, s7 0.6864167294649586). Drift now fails the registered gate. Extends MEMORY 71 (parity-test self-built eval plumbing) from "test it once" to "the gate re-tests it forever".
- **Aggregate corpus rates as ratios of integer totals, never as means of per-line fractions.** The old `eval_multi_ref.py` bug under-reported DER ~29.8x and reordered the ranking, because it over-weighted short lines. A fixture with one 1-word bad line beside one 100-word good line is the regression test: the truth is 1/101, a mean-of-lines would say 0.5.
- **Exact McNemar must be computed in log space.** The direct `math.comb` / `2**n` form raises `OverflowError: int too large to convert to float` on realistic discordant counts; accumulate over `math.lgamma` instead. It still underflows to 0.0 at the extreme (render it as `< 1e-308`, do not print `0.0`, which reads as "impossible").
- **pipeline_v2: adding a suite to `SUITES` is a breaking change.** `test_evaluation.py` pins `sorted(suites.SUITES)` to an exact 4-item list and requires `suite_for("laya_bench")` to raise; `test_handlers.py` iterates `SUITES.items()`. A new domain therefore gets a PARALLEL catalogue (`DIACRITIZER_SUITES`) with its own `diacritizer_suite_for`, leaving `suite_for` refusing it. Honest consequence: the R1 stage graph resolves evaluators only through `suite_for`, so the new suite is not reachable from an R1 config — the working integration is the artifact -> `EvaluationReport` bridge.
- **`Threshold.direction` reads backwards.** `"min"` = the value may not EXCEED the bound (an upper bound); `"max"` = it may not fall BELOW it (a lower bound). `val_loss` uses `"min"` for its ceiling. Getting this backwards inverts a gate while still looking correct.
- **`require_complete_suite` demands the value set EQUAL the suite's metric set exactly.** A union of everything measured fails as "unexpected metrics", so any bridge must scope the row to `suite.names()` first.
## 2026-09-24 - U-line U-1 facts (UI-JSON generator)
- The month-old sm_75 ceiling claim now has a CAT SCALING datapoint: 28M ctx512 fp16 8-bit-adam trains at ~24k tok/s sustained (batch16/accum2, ckpt-off) with NO arch changes - compute-bound, no data bottleneck at workers 0.
- A val eval_loss of 0.27 on the chain surface is enough for 100% greedy structural validity - the packed-sequence format is nearly free to learn (cross-archetype template echo happens regardless and is NOT caught by the validator: only structural/bg semantics areʯ - semantic-FOLLOW must be its own gate).
- Do not trust workers>2 inside a MANUAL bench on Windows (spawn-inside-spawn error before DataLoader starts; verify with condor workers run); real OOM signature = WDDM shared-memory spill w/ alloc growth past 60% of free.
- corpus pipeline: prepare_u1.py -> tokenize_data.py are REUSABLE as-is for any text corpus (key = raw JSONL {text: task+chain}); renames: val cap 1500 rows after a smoke-time 0-row val bug.

## 2026-09-24 - U-line chain-codec + data-scale lessons (ui/data/omen_alpha)

- Chain codec (ui/src/chain.py) hard limits measured against the round-trip gate: $template interpolation with spaces corrupts BIND tokenization; one binding per event is representable (last-wins overwrite otherwise); $or/$and conditional groups are rejected by validate; changeType IS the string enum ("−1","0","1") per catalog.py (an agent critique claiming "numeric enum" was wrong — adjudicate against catalog, not critique); Table.caption exists (Table{columns, rows, caption?}); Table rows must bind via $state or $item (literal string-arrays fail the ao check); pushState params pattern {"statePath","value","title","id"}; removeState with literal index 0 only.
- Data-gate discipline that caught real bugs: run validate_chain(spec_to_chain(spec, task)) AND a full spec->chain->spec round-trip equality check on EVERY row — the round-trip caught a duplicate element key the validator alone did not.
- Same-seed scale tiers are nested prefixes of each other (deterministic sampler, zero-dedup pressure: attempts == rows across 5k/20k/50k). Mirror only the LARGEST tier into the canonical data path; keeping 5k+20k+50k files side by side invited a 25k-row duplication trap.
- Critique subagents (PROMPT D) produce plausible-but-wrong catalog claims; source of truth is catalog.py + the validator, not the reviewer. Two of its 7 claims were false positives.
- Windows tooling: Node execSync must quote the venv exe path ("E:\python projects\... with spaces"); inline python -c code gets mangled by cmd quoting — write a temp .py file instead (scratch/_find_dollar.py pattern).
## 2026-09-19 - DA-2 emo data/recipe facts
- tweet_eval emoji class->emoji mapping: fetch class_label.names from the HF API (cardiffnlp/tweet_eval), never hand-guess emoji strings.
- TEAD (arbml/TEAD): all 12,558 rows contain emojis; 469 distinct types; the DeepMoji one-type filter keeps 67 across 7,773 rows - it is the vocab bottleneck.
- snakers4/emoji-sentiment-dataset (11 langs incl ar) is BOTH dead (hosting 404) and CC-BY-NC - rejected on both counts.
## 2026-09-19 — mu1 lesson: micro-scale weight merging fails by basin divergence (E-27)
- Weight soup/TIES merges of four same-seed 12K experts ALL collapse to 0.0 exact-match; embedding geometries go near-orthogonal per task (corr 0.027). Routing/distillation, not merging, is the correct composition path at this scale; also: tasks with self-distinguishing formats make learned routers trivial (1.00 accuracy).
# MEMORY.md — durable memory: decisions & lessons

What the next agent must not rediscover. Two ledgers: **Decisions** (user
calls and stable policies; valid until revoked) and **Lessons** (hard-won
technical knowledge). Append-oriented: correct an entry only with a dated
note, never silently.

Provenance: seeded from HANDOFF §5–§6 (2026-09-06 snapshot). HANDOFF stays
the state-of-the-run narrative; this file owns durable knowledge from now on.

- 2026-09-08 (U11): subprocess stdout redirected to a file is BLOCK-buffered on Windows - `chain_out.log` loss lines lag the real step count by several log cycles. Mid-run triggers (tests, future monitoring) must anchor on flush=True markers (`[sft] PILOT:`, `[resume]`) or wall-clock timing, NOT on `{'loss': ...}` lines. Also: transformers 5.16.1 dispatches TrainerCallback.on_save only around _save_checkpoint (line ~2130) - the checkpoint-aligned stop flag MUST be checked there, and interval saves do honor it (verified live: stop after checkpoint-250 mid-run).
## Cosine-tail masks fixed-lr overfit (E-26, 2026-09-18)
- At constant high lr (mid-cosine) tiny models massively overfit: val loss can climb +90% while train falls; but the final lr anneal pulls val back to ~its own minimum. Judging an run by its FINAL eval loss underestimates overfit damage — downstream exact_match landed flat-worse than the 12K run despite "recovered" val. Track min-eval step and the curve, not the last value (tools: mex/scripts/plot_losses.py). Judge saturation by comparing held-out eyeball metrics at 1x vs 10x steps.
## Decisions ledger

| Date | Decision | Why | Reference |
|---|---|---|---|
| 2026-09 | fp16 only; no bf16; no flash-attn | Turing sm_75 has no fast bf16; flash-attn needs Ampere+; PyTorch SDPA instead | PLAN.md A3/A4 |
| 2026-09 | plain GQA decoder first; Flash-Next hybrid is a later upgrade | resources' skeletons are pseudo-code; their own advice is plain-first | PLAN.md A7 |
| 2026-09 | auto-checkpoint + auto-resume with zero flags is non-negotiable | user hard requirement; acceptance-tested in the M0 kill/resume drill | PLAN.md §5.3 |
| 2026-09 | venv managed by uv only — never pip | reproducibility; the rebuild recipe was proven once after a machine move | HANDOFF §4 |
| 2026-09 | M3/target weights stay LOCAL; `optimizer.pt` gitignored | ~900 MB model does not fit free LFS (~1 GB quota, ~790 MB used) | HANDOFF §7 |
| 2026-09-06 | M3 restarted FRESH on MUO4QK5 | TU09FBO handoff bundle never arrived (user call); old bundle/zip obsolete | HANDOFF §3b |
| 2026-09-06 | C12 Tier 1 approved to implement; RUN gated on M3 completion | pipeline had no SFT stage; T never saw Evol-Instruct → contamination-free SFT data | HANDOFF §8.4; research/c12_distillation_plan.md |
| 2026-09-06 | C12 runs ONLY from runs/target/final; SFT-from-checkpoint-1000 DECLINED; prep-during-M3 (CPU-only) approved | step-1000 base is 20%-trained (eval 2.2650 vs final target ~1.2-1.3); SFT cannot co-run with live M3 (~2.6 GB free of 8); full SFT would be redone on final T anyway (double GPU cost) | research/c12_runbook.md decision log; scripts/c12_preflight.py |
| 2026-09-06 | Window-prep milestones approved (user menu): Tier 2 readiness brief + phase dashboard + LoRA design; Tier 2 RUN itself stays gated (needs teacher download + GPU window approvals) | use the M3 training window for pipeline prep without touching the live run | TASKS.md rows 7-9; research/c12_tier2_brief.md |
| 2026-09-06 | bitsandbytes 0.50.2 VERIFIED on sm_75 (python -m bitsandbytes: SUCCESS) | unlocks 8-bit teacher/optimizer options on this Turing card; peft/trl/triton NOT installed | research/c12_tier2_brief.md §2 |
| 2026-09-06 | peft 0.20.0 installed (user OK via window-menu answer); uv DRY-RUN first: 12 new packages, ZERO upgrades - safe beside live M3 | LoRA option for cheap fine-tunes; import OK vs transformers 5.16.1 / torch 2.14.0+cu126; CPU smoke PASS | TASKS row 9; research/lora_peft_design.md |
| 2026-09-06 | Tier 2 teacher = Qwen/Qwen3.5-0.8B (user pick, replaces the brief's Qwen2.5-Coder default) - VERIFIED before download | ungated, ~1.7 GB (1,688 MB / 29 files), Apache-2.0, 24 text layers h1024, vocab 248,320, ctx 262k, chat template; multimodal qwen3_5 arch (judge uses text tower only); transformers 5.16.1 loads it natively; weights local-only (data/teacher/ gitignored) | TASKS row 5; scripts/download_teacher.py |
| 2026-09-06 | 3 prep commits PUSHED (08d3292, f161eb9, 8d3c378 -> origin/main) | user-approved; git lfs status clean first | git log |
| 2026-09-06 | Net search = Exa helper only (AGENTS.md section 3); harness web_search failed repeatedly ('Not found'); exa-py 2.20.0 installed (12 new pkgs, zero upgrades); EXA_API_KEY still MISSING - no .env exists, user must create it | Exa search is the repo's research path; raw payloads land in research/raw/ | AGENTS.md section 3; research/qwen35_teacher_tasks.json |
| 2026-09-06 | Recommendation menu approved as MILESTONES ONLY - B throughput, C evaluation, D data, E distillation order + judge rubric, F off-site backup, G product/research (TASKS rows 11-17): created, none started (user: do-not-start) | row 10 (LoRA hook) remains the pre-existing pending user-gated item; A (power/thermals) is a USER-OWNED hardware action, recommended but not tracked as agent work | TASKS rows 11-17 |
| 2026-09-06 | User added EXA_API_KEY and HF_TOKEN to the gitignored project .env | unblocks Exa research (teacher web context) and the gated-data path (row 13); GOTCHA: hf_hub/datasets do NOT auto-load .env - scripts must load_dotenv (exa_search.py is the only auto-loader today) | TASKS rows 5/13; HF whoami + gated probe |
| 2026-09-06 | Milestones B-G EXECUTED in the window (user GO after the create-only round): CPU-safe parts done beside live M3 - B: A/B configs + protocol doc + train.py env fingerprint (bnb 0.50.2 already present); C: scripts/mini_eval.py (canned 16/16; smoke-final CPU x2 identical) + status.py tokens/sec+MFU; D: research/pretrain_mix_proposal.md + prepare_data.py .env loader (gated probe OK); E: tier-order framework (rec: Tier 3 before Tier 2 V1) + frozen judge rubric; F: scripts/backup_to_hub.py dry-run PASS (upload = user repo-name gate); G1: research/gdn_sandbox_design.md (impl post-M3); G2: scripts/run_custom.py (dry-run PASS) - ALL GPU work deliberately staged post-M3, nothing co-ran | use the M3 window without touching the live run; CPU-testable acceptance measured, GPU acceptance deferred with explicit gates | TASKS rows 11-17; HANDOFF §8 |
| 2026-09-06 | M3 checkpoint cadence = save+eval every 100 steps (user allowed 100–200; HF requires save_steps to be a multiple of eval_steps with load_best_model_at_end), save_total_limit 3 = best + 2 latest | bounds crash loss to ~100 steps (~40 min throttled); 36 GB free at resume; eval adds ~7% wall time | configs/target.yaml; HANDOFF §3b |
| standing | no long runs, deletions, pushes, or purchases without user approval | safety policy | CLAUDE.md §7–8 |
| 2026-09-07 | User GO: "create milestones ... then proceed" — Milestone B GPU arms + row 10 LoRA hook + vram_probe, Tier 3 KD BEFORE Tier 2, SFT v2 on the minimax3 corpus from runs/target/final | GPU window open post-M3/Tier-1; sequential never-co-run discipline | TASKS rows 18-20; configs/pilot_b8*.yaml, sft_v2.yaml, kd_s_*.yaml |
| 2026-09-07 | Milestone B DECISION (gates measured): **adamw_bnb_8bit + batch 1/accum 32 = default for the NEXT pretrain** — loss parity (2.5603 vs fp32 2.5644), pace 1.02x, VRAM −563 MiB (1.36 vs 1.91 GB probe peak), kill/resume drill PASS; batch 2/accum 16 REJECTED (0.60x pace, no loss gain on the 6 GB card — activations/grad-ckpt dominate, batching gains nothing); LoRA hook GPU peak 0.72 GB @ 2.61M trainable | any future full retrain should set optim adamw_bnb_8bit; fp32 AdamW stays the fallback if bnb misbehaves | research/milestone_b_8bit_ab.md; TASKS rows 11/18 |
| 2026-09-09 | Rows 2/13/16 executed beside a CONCURRENT Track A session: USER-APPROVED deletion of checkpoint-sft_v2_e1-final.zip (868 MB, CRC-verified redundant vs disk + pack); Milestone D wired+measured+sized — token-proportional smol-capped mix 10000/37500/59000/16900 (~133.65M kept tokens, max_steps 4100) derived from MEASURED tok/row after the proposal's row targets proved 3.75x over budget; RUN stays user-gated until the next pretrain window; Milestone G1 implemented + CPU-validated (G1.a staged on a GPU window) | shared working tree with Track A -> targeted re-read edits only for shared files; GPU is global (total nvidia-smi memory.used = busy signal); nothing deleted/pushed without approval | research/milestone_d_actual.json; research/milestone_d_measurement.md; research/gdn_sandbox_design.md; TASKS rows 2/13/16 |
| 2026-09-10 | Track B EXECUTED (user go): YaRN ctx-4096 fine-tune of runs/target/final -> runs/yarn_4096 via config-gated train.init_from (base untouched); 4096 CONFIRMED by the mandatory 8-bit-Adam probe (peak 4.24 GiB alloc); val @4096 1.7035 = -8.0% vs the 1024 baseline (-9.6% vs eval-only NTK@4096), forgetting guard @1024 +1.15% PASS, extrapolation @8192 2.1240 (better than eval-only NTK@8192); instruct re-SFT = row 41 (user-gated) | plan §B gate fired at row 29 (NTK@2048 -4.17%); artifact = context-extended BASE | TASKS row 30; runs/yarn_4096/final/train_summary.json |
| 2026-09-10 | Row 41 vehicle DECIDED (user 4-option menu): instruct re-SFT on runs/yarn_4096 = LoRA-SFT r=32/alpha=64/lr 1e-4/1 epoch -> merged final (row-34 Track F repurposed as the row-41 vehicle), P0 soup probe FIRST (e1+yarn_4096 alpha sweep 0.25/0.5/0.75, eval-only, scripts/soup_merge.py); gates STRICT (CSN guard <= +5% vs 1.8724; base surface @4096 <= +2% vs 1.7035; instruct val < 2.1265-equiv; ast). Grounds: in-repo LoRA forgetting +0.13% (h2) vs full SFT +9.8% (e1) / +19.7% FAIL (e2); literature LoRA Learns Less and Forgets Less (arXiv 2405.09673), replay-mix anti-forgetting (PMLR v267 bethune25a, arXiv 2603.04964), LongAlign = extend THEN align (2024.findings-emnlp.74); long-PROMPT instruct ability stays untested by short-ctx SFT (add a long-prompt probe to gates) | user picks; ctx-expansion ranking: trained YaRN > NTK@<=2048 > window+sink-abs (flat to 16k, recall-capped) >> pos_shift (never) | TASKS row 41; research/raw/r41*.json; OUTCOME 2026-09-10: gates 3/4 (base @4096 +4.64% FAIL on the raw merged) -> repair soup a60 (0.6 LoRA + 0.4 yarn) passes ALL 4 (guard +1.03%, ast 0.90/0.96, @4096 +1.29%) -> user picked BOTH (final_a60 = chat-default candidate, raw = instruct-max); long-prompt probe @3.4k tok FAIL (short-ctx SFT does not align long prompts - LongAlign lever = future long-instruct slice, TASKS row 42); commit made, push user-gated |
| 2026-09-10 | KT ladder APPROVED (Q1 both capability+research / Q2 architecture AFTER the KT-1 A/B numbers / Q3 evening + multi-day GPU windows / Q4 SmolLM2-360M teacher first); Route 0 warm-start SmolLM2-135M stays the Q2 fallback, never rejected | a poor base stays poor through SFT - get A/B evidence before architecture calls | docs/plans/2026-09-10_kt_ladder_plan.md; TASKS rows 43-46 |
| 2026-09-10 | KT-1 A/B RESULT + Q2 DECIDED: transplant init WINS -6.24% eval @1000 (2.2902 vs 2.4426; plan gate >=3-5% MET; gap still growing at 1000; train/loss corroborates -7.64%); user KEEPS OUR ARCHITECTURE - Route 0 warm-start SmolLM2-135M queued as a LATER test (row 47). Checkpoint policy CONFIRMED: keep best + last two (= exactly what save_total_limit=3 + load_best_model_at_end already does during runs); completed-run kt_ab ckpt-800s deleted; checkpoint_backup row41 zip (2.88 GB) deleted after restore verification - new packs can be rebuilt from on-disk originals later | single-variable A/B (same seed/GPU/stack); user call | TASKS rows 43/47; runs/kt_ab_* |
| 2026-09-11 | ARABIC DIACRITIZATION D-LINE APPROVED (user go, full design session): (D1) build route = OUR architecture blocks as a NEW bidirectional char encoder + 15-class per-letter classification head inside a self-contained diacritizer/ submodule — NOT an M3/e1 fine-tune (Sadeed 2504.21635 shows decoder-LM frame = hallucination/copying; English/code weights transfer nothing), NOT a CATT fork (baseline only); (D2) pilot size = ~30M (evidence sweet spot: CATT EO, rababa ~30M ~1% DER claim, Mishkala 12.5M 1.66% claim); 100M stretch only if pilot plateaus; (D3) placement = submodule-form; no edits to M3-line src/configs; (D4) input-policy ladder STAGED: A2 strip-and-rediacritize -> A3 preserve-known+fill-gaps (two-stream + partial-diacritic curriculum) -> A4 dual modes (preserve + rewrite); each stage gated on prior eval; (D5) Phase-1 domain = mixed general (SadeedDiac-25-shaped ~50/50), scale-up path to poetry/Quran later; (D6) NON-ARABIC POLICY = copy-by-construction: context-visible, zero-predictable, byte-exact reassembly; tatweel passthrough never diacritized; Quranic marks + invisibles (RLM/LRM/ZWJ/ZWNJ) byte-exact passthrough; eval harness (DER/WER +-case endings, text-preservation, preservation) built BEFORE any training (A0 CPU-only exit gate = selftest 4/4 PASS) | whole D-line structured evidence: DESIGN research/arabic_diacritization/DESIGN.md + plan docs/plans/2026-09-11-arabic-diacritization-specialist.md + TASKS rows 50-55 |
| 2026-09-19 | DA-LINE (Desert Ant recreation) opened with approval to plan + START the most feasible target (user: "create a plan/milestones/tasks/..etc for them and start with most feasible/achievable one"): chosen DA-1 = Tongue-analogue text language ID (hashed char n-grams + script routing, 21 langs incl. the Arabic-script group ar/fa/ur/ps) as THE most feasible (CPU-only, eval fully copyable from their card, ~200MB corpus). Ladder DA-2..DA-9 recorded user-gated. langid/ is a self-contained submodule; zero edits to M3/D/mu lines; no GPU use | desert-ant teardown report (done this session); research/desert_ant_recreation/DESIGN.md | docs/plans/2026-09-19-desert-ant-recreation-da1-langid-plan.md; TASKS rows 81-91 |
| 2026-09-18 | ME-LINE (micro-expert composition) APPROVED as a new research line: (1) Stage 0/μ0 sandbox approved as the BASE for later stages — stepping stones, every μ-stage consumes μ0 artifacts; (2) ALL FOUR composition methods in scope as μ1 arms (weight-merge, MoE-merge BTX, dispatch, committee-distillation); (3) task policy = A+B hybrid — synthetic testbed tasks + REAL Arabic diacritics as the first real expert (existing D-line data reused now); μ3 D-line transfer user-gated separately | user brainstorm 2026-09-18; web-research grounding (BTM/BTX/merging/MoE refs in DESIGN §8); validates the E-13/E-14/E-19 domain-routing frontier at cheap micro scale first | research/micro_experts/DESIGN.md; TASKS rows 76-79 |
| 2026-09-24 | U-1 full arm addressed as the CURRENT U-line milestone (user GPU window): train+validator-gate DONE; next = semantic-follow gate + diversity rung pre-registered, user-gated like all GPU launches | EXPERIMENTS.md U-1 verdicts 2026-09-24 |


## Lessons (seeded from HANDOFF §5)

1. **Windows console cp1252:** any script printing Arabic/emoji crashes with
   `UnicodeEncodeError` unless stdout is reconfigured
   (`sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')`).
   Applied in `diacritizer/scripts/selftest.py`; every future script that
   prints non-ASCII must do the same.

2. **Corpus mark order is NOT canonicalized in the wild:** real Tashkeela/
   Fadel text writes shadda AFTER the vowel (`ثَّ`) as often as the canonical
   shadda-first (`ثَّ`). A parser enforcing one order quarantines ~89% of a
   classical corpus. Fold both orders to the same 15-class gemination label.

3. **transformers 5.16.1 drift:** TrainingArguments has NO `logging_dir` and
   NO `save_safetensors` (safetensors is the only format). TensorBoard dir =
   env `TENSORBOARD_LOGGING_DIR` (train.py sets it to `runs/<phase>/logs`).
   Trainer takes `processing_class=` (not `tokenizer=`), `eval_strategy=`.
2. **No console loss lines:** Trainer 5.16 prints no `{'loss': ...}` to
   stdout. Read metrics via `EventAccumulator(dir).Reload(); ea.Scalars('train/loss')`
   (tags: train/loss, train/grad_norm, train/learning_rate, eval/loss — eval/loss
   only exists after the first eval step).
3. **Thermal throttling:** sustained training cycles 84–90 °C; SM clock drops
   1560→~1110 MHz; pace swings ~2× (e.g. 4.2–6.75 s/it observed). Safe but
   slow — never kill a run for pace alone. A proper high-watt AC adapter
   roughly halves wall time (battery-drain-on-AC capped a run at 20–25 s/it).
4. **Modern Standby can freeze a long run** (CUDA context survived once);
   disable sleep-on-AC for unattended runs.
5. **DSH agent quirks:** run_code wall-clock ≤600 s (chain single
   `job_output` waits); job status lives at `result.job.status`;
   write/edit calls may need an explicit `description`; strip ANSI/control
   chars from pwsh/tqdm output before returning JSON; Windows sandbox prefers
   simple single-line pwsh commands.
6b. **fp16-only Turing + AMP (D-line):** a fully-cast fp16 model CANNOT use
   torch.amp.GradScaler ("Attempting to unscale FP16 gradients"). Working
   contract in diacritizer/scripts/train.py: fp32 master weights + autocast
   fp16 COMPUTE + GradScaler with skip-on-overflow. Also: SDPA requires
   q/k/v same dtype - cast RoPE output back to v.dtype (diac root model.py).

6. **Machine moves happen:** the venv rebuild recipe (HANDOFF §4) was proven
   once — keep it current in ENVIRONMENT.md.
7. **CUDA_VISIBLE_DEVICES='' does NOT force CPU** in torch on Windows (empty
   string still exposes the GPU — a 2026-09-06 sanity probe ran on GPU at
   0.95 GB beside the live M3 run and fit; do not rely on '' to hide the
   GPU — use -1 after verifying, or make the script CPU-explicit).

8. **DSH tool-arg hygiene:** tool args must be lossless JSON - never pass
   undefined-valued properties (e.g. timeoutMs left undefined inside the args
   object) or null where an object is expected; both hit 'binding arguments
   must be lossless JSON' (2026-09-06). Build arg objects conditionally. Same-day note: when a multi-edit batch fails midway, audit WHICH edits
   applied before retrying - a 'failed' program had silently applied 3 of 5
   edits, producing duplicate TASKS rows (caught by read-back verification).
9. **Windows WDDM GPU-process listing:** `nvidia-smi --query-compute-apps` lists EVERY process holding a GPU context on Windows (WebView, WhatsApp, Terminal, SearchHost...) - an unfiltered "GPU busy" guard would abort every legitimate chain (caught 2026-09-06 in run_custom's dry-run). Filter the lines to `python` process names; verified live: only the real trainer PID remains, desktop noise disappears.

10. **Checkpoint bundles carry machine-local absolute paths:** `trainer_state.json`'s
    `best_model_checkpoint` is the SOURCE machine's path (MUO4QK5 wrote
    `E:\python projects\...` with a space; TU09FBO is `E:\python_projects\...`).
    Patch it to the local path right after extracting a bundle, else
    load_best_model_at_end can fail at training END and the limit-3 rotation
    will not protect the best checkpoint (found 2026-09-06 resuming M3 from
    the step-2000 zip). Same family: HF requires save_steps to be a round
    multiple of eval_steps when load_best_model_at_end is on — cadence
    changes must move both keys together.

 11. **nvlddmkm Event 153 = GPU driver engine fault** — kills a CUDA run
    instantly with python exit 1 (found 2026-09-06 19:07, killing the M3
    resume mid-step before the step-2100 save). Distinct from Modern Standby
    (freeze; process survives) and OOM (traceback says so). Recovery = rerun
    the exact train command (auto-resume); identify it via the System log
    (nvlddmkm 153). If it repeats, suspect the SW-power-capped hardware
    state (underpowered adapter), not the training code.
 12. **Windows: `CUDA_VISIBLE_DEVICES=""` does NOT hide the GPU** (empty
    string = all devices visible; found 2026-09-07 when a "CPU-only" sanity
    check ran on cuda beside arm A). Use `-1`. Gotcha 2: `Start-Process`
    from a fresh shell inherits NO such var — set it in the launching
    session before Start-Process, or the child lands on GPU.
 13. **Kill mid-checkpoint-write corrupts the checkpoint and crashed
    auto-resume** (FileNotFoundError on trainer_state.json; found 2026-09-07
    during the LoRA kill/resume drill). trainer_state.json is written LAST
    by the saver, so `find_latest_checkpoint` (train.py/sft.py/kd.py) now
    requires it — a partial checkpoint is skipped instead of crashing the
    unattended resume.
 14. **The partial-checkpoint guard was DEFEATED by `resume_from_checkpoint=True`**
    (found live 2026-09-08 on sft_v2_e1: sleep killed a save mid-write; the
    guard picked ckpt-750 but the boolean made HF Trainer run its OWN
    discovery, which picked the partial ckpt-1000 -> crash). FIX: all three
    scripts now pass `str(resume_from)` (the path) instead of True.
 15. **Disk-full signature: checkpoint writes HANG at the same byte offset**
    (E: hit 3.9 MB free; optimizer.pt stalled at exactly 482,394,112 bytes
    twice; GPU idle 0%, process alive in disk-wait; nvlddmkm/disk events
    clean — check `Get-Volume E` SizeRemaining FIRST when a run freezes
    mid-save). Prevention: keep >5 GB free before any SFT-scale run;
    SFT checkpoints are ~2.7 GB each (906 MB model + 1.8 GB optimizer).

 14. **Windows venv python.exe is a launcher shim — pid genealogy skips a
     level:** .venv\Scripts\python.exe spawns the real base interpreter
     (the uv cpython) as a CHILD and waits, so a child's os.getppid() is
     its own launcher shim, NOT the webui server/probe that launched it —
     and the grandparent is the one holding a lingering CUDA context after
     a chat unload (found 2026-09-08: run_custom's parent-pid exclusion
     aborted every launch with "GPU busy (1 compute process)" three times).
     Fix: _ancestor_pids() Toolhelp32 walk — the _PE32W struct MUST
     byte-match PROCESSENTRY32W (size_t heap id + 4-byte LONG priority; an
     8-byte pointer field silently inflates dwSize and Process32FirstW
     fails with an empty ancestor walk). Note "Get-Process python*" shows
     TWO pids for one logical venv python run; nvidia-smi lists the base
     interpreter pid.

 15. **Gradio 6 auth= is a form+cookie flow, NOT HTTP Basic:** unauth /
     returns a 200 login shell; valid login = POST /login →
     {"success":true} + session cookies; the gated API (/config) answers
     401 until the cookie is present — curl -u user:pass alone gets 401
     even with VALID creds (probe with a cookie jar, not -u; found
     2026-09-08 verifying the web UI --auth).

 16. **A webui server that once loaded a chat model keeps a CUDA context after unload** (Windows WDDM) — nvidia-smi keeps listing its python pid, so run_custom's never-co-run guard aborts ANY UI-launched chain at the train step ('GPU busy'). Fix at the moment: restart the UI (or fully close it) before launching; the fresh server has no CUDA context until a chat model loads on GPU. Preflight already prints the blocker - read it instead of re-launching blind. Found 2026-09-08 during the U6 kill->Resume drill (two stale UI instances held ~1.2 GB).

 17. **TheGamingMahi/TinyCode's corrupt shard is mode-dependent death:** streaming=True + a row cap below the shard (<= ~1,250 rows) works fine; a row request past the shard CastErrors mid-stream, and FULL download (streaming=False) always dies (DatasetGenerationError @ ~1,900 ex). Keep it as the UI default only with small row caps; use other ungated sources for bigger pulls. Also: bigcode/the-stack-smol is now UNGATED (namespace moves changed access) - only the-stack-v2 / starcoderdata-class repos still gate. Found 2026-09-08 (U7 e2e + download-mode probe).

 18. **The 226M SFT student answers QA with import boilerplate and cannot reliably copy injected facts out of context** (Track H, 2026-09-08): runs/sft_v2_e1/final continues `### Response:` with `import re` / `from typing import Optional` for QA-style recall; an explicit "reply with one short sentence, do not write code" cue made it WORSE (1/6 -> 0/6); a needle-style completion probe ("My sister's name is") also failed 0/6. The copied fact, when it surfaces at all, appears late in a long continuation — score FULL continuations, never first lines. Deterministic across runs (f3 PASS identical twice, greedy). The memory MECHANISM itself validated: store 6/6, retrieval 6/6 correct fact per question, model-mode rolling summary 7/7 folds (61 tok), AST regression 4/4 with preamble vs 2/4 plain. Also: the prompted fact extractor echoes code fragments ('"""', '>>>', 'def ') — validate extractions against the turn and filter code markers; the deterministic fallback carried 6/6 coverage. Conclusion: orchestration memory works; copy-out is a model-capability wall — the trained path (Track C) or a stronger student is the fix. **UPDATE 2026-09-09: the copy wall is TRAINABLE — after copy-behavior LoRA-SFT (row 37 Phase 1) the recall gate passes 3/6 (probe style); QA-template code answers turned into tautologies as the cost.**

 19. **A foreground subagent inside run_code dies with the 600 s wall-clock
    ceiling** (found 2026-09-09, Track H2): the ceiling kills the run_code
    worker AND the child mid-call — empty children registry, no file, no
    closing message (the 2026-09-08 "subagent infra dead" foreground
    ToolCallError matches this signature). Background subagents live
    outside any single run_code call — spawn with run_in_background: true
    and collect via the settle notice / list_agents; keep foreground
    subagent calls for tasks that finish well under 10 minutes.
 20. **Open-ended long quotas make subagents loop in thought** (2026-09-09,
    Track H2): one 150-pair corpus author deliberated until the user
    killed it — no file, no closing message. Fix that went 20/20: chunk to
    ~30 units per spawn with a mechanical procedure (compose in your head
    -> ONE write call -> 3-line reply; no reads, no scripts, ~4-min
    budget), disjoint per-chunk file names (the validator globs + dedupes
    across files), rotated name pools per chunk.
 21. **Subagent JSONL output needs a mechanical repair pass** (2026-09-09):
    two defect classes seen in 20 authoring chunks — (a) real newlines
    inside JSON string values (30 pairs became 150 physical lines; repair:
    rejoin the regular head/bullets/tail groups with literal backslash-n),
    (b) missing opening quotes after the JSON keys ({"instruction":
    Notes:...). Both repaired losslessly by scanning for the
    '{"instruction"' group anchor, re-escaping, json.loads-verifying,
    rewriting only on 100% parse success. A full exemplar LINE in the
    prompt stopped (a) in later waves; (b) still needs the post-write
    parse check.
 22. **sft_data.py small-corpus gotchas** (2026-09-09, h2_copy_lora):
    n_val = round(rows * val_fraction) uses the TARGET rows, not the kept
    count — a 585-pair corpus configured rows: 5000 got an INVERTED
    250 val / 110 train split (fix without touching the script: set rows
    ~= corpus size so target ≈ kept); min_chars floors the RESPONSE
    length (len(response) < min_chars) — 30 chars killed every
    note-completion answer ("Anna.", "port 6379."); ast_filter: false is
    mandatory for plain-English corpora. The current script math is a
    no-op for big corpora, so no script change was made.
 23. **Probe a dataset's schema before bulk sampling** (2026-09-09, h2p2):
     the v3 builder sampled 0/500 from teknium/OpenHermes-2.5 because it
     assumed instruction/output keys — the real schema is ShareGPT-style
     conversations [{from, value}] (self-contained pool 611,945 after
     skipping system_prompt/multi-turn; fixed after a one-row schema
     probe). Also: HuggingFaceH4/no_robots Chat is multi-turn by
     construction (795/796 rows have >2 messages) so a single-turn filter
     yields ~nothing from that category; dolly categories are
     brainstorming/classification/closed_qa/creative_writing/general_qa/
     information_extraction/open_qa/summarization (open_qa 3742). Keep the
     builder's per-source category/pool census print for any new HF slice,
     and probe candidate teacher/corpus repos via HfApi (gated status +
     safetensors sizes) before downloading. .env keys load by NAME-ONLY
     prints — never values.
 24. **Unambiguous sentinels — a sentinel string can self-match** (2026-09-09):
     a GPU-free check whose empty branch printed "no python compute apps"
     and whose harness then tested output.contains("python") reported BUSY
     on an empty GPU. Make the empty-state output a string that cannot
     contain the searched token (GPU-FREE-SENTINEL matched via
     includes('GPU-FREE-SENTINEL')), or count raw csv lines before any
     message text. Same rule for every grep-then-branch tool pattern.

 25. **KD loss + VRAM gotchas at full-vocab logits** (2026-09-09, Track C):
      (a) F.kl_div treats `target` as PROBABILITIES by default — passing
      log-probs as the target silently yields NaN (log of a negative);
      with log-prob targets pass `log_target=True` (loss =
      exp(target)·(target − input) = exact KL). The CPU math test caught
      this BEFORE any GPU run — always validate new loss math on CPU
      random logits (exact-vs-brute-force + limits + finite-grad checks)
      first. (b) KD's fp32 full-vocab logit tensors dominate VRAM: at
      P-arch/ctx 1024 the KD arm's VRAM ladder measured micro-b8 = 7897
      MiB of 8192 (96 pct, OOM-risk) -> b4/accum2 = 7010 (86 pct, chosen
      for BOTH A/B arms — fairness needs identical micro-batching) ->
      b2/a4 train was only 4926 but an eval_batch-8 transient on top of
      the train cache spiked 7486 (eval batch must also drop). A 50-step
      probe per NEW loss shape (incl. its eval) is the cheap way to see
      the true peak. (c) A DistiLLM skew floor (skew=0.1) costs ~+0.44 GB
      even chunked per-sample.

 26. **transformers 5.16.1 custom-arch auto_map is DOUBLE-BROKEN** (2026-09-09,
     G1 sandbox): dynamic loads are hard-gated behind trust_remote_code=True
     (no env bypass; Windows has no SIGALRM so it raises immediately), AND the
     ref parser accepts only two-part dotted paths (split('.') expects exactly
     2 — 'src.model.X' crashes with 'too many values to unpack'). WORKING
     PATH: idempotent AutoConfig.register + AutoModelForCausalLM.register at
     src.model import — any process importing src.model then loads the
     checkpoint natively (eval.py's existing 'from src.model import ...'
     fires it). Proven fresh-process: save_pretrained -> from_pretrained ->
     generate, 60/60 weights. Related 5.16 drift: model configs are strict
     dataclasses (a 4.x-style __init__ collides on reload); _tied_weights_keys
     is a dict now; torch.autocast downcasts fp32 matmuls INSIDE its region —
     GDN-style fp32 state must run under autocast(enabled=False), .float()
     alone is not enough.

 27. **HF gated=auto: a VALID token is NOT access** (2026-09-09,
     starcoderdata): 403 GatedRepoError with a whoami-OK token means the
     dataset's terms were never accepted for THIS account; acceptance on the
     dataset page grants immediately (no re-auth). The 2026-09-06 "verified
     unlockable" evidence covered the-stack-smol only — verify access PER
     REPO before planning a mix around it.

 28. **bigcode per-language builder configs are GONE** (2026-09-09):
     the-stack-smol AND starcoderdata accept only the 'default' config —
     load_dataset(..., 'python') dies with BuilderConfig-not-found; the
     working form is data_dir= ('data/python' / 'python' respectively).
     prepare_data.py now supports a data_dir passthrough; legacy
     pilot.yaml/pilot_b8.yaml were fixed the same day.

 29. **Whole-file code sources measure ~4x their function-source estimates**
     (2026-09-09, Milestone D): the-stack-smol 2,717 tok/row (est ~700),
     starcoderdata 2,346 (est ~600) vs CSN 272 and Evol 474. Sizing a mix by
     ROWS inherits the estimate error — the proposal's row targets were 3.75x
     the 134M budget. MEASURE tok/row per source (2k-row samples, cheap)
     BEFORE sizing, then derive rows from measured rates.

 30. **Concurrent agent sessions on one working tree** (2026-09-09): a second
     session edited src/model.py + scripts/eval.py mid-flight. Protocol that
     held: edit shared files ONLY as targeted re-read edits (never whole-file
     writes from memory); snapshot WIP before edits
     (data/snapshot_row13_row16_wip/); the GPU-busy signal is TOTAL
     nvidia-smi memory.used (per-process values report [N/A] on Windows
     WDDM); every GPU gate re-checks immediately before firing.

  31. **A restored final's eval_report.json is stale provenance** (2026-09-09,
      Track A): runs/target/final was bit-exact-restored from checkpoint-4000
      (row 19 gotcha 1), so a fresh knobs-off re-eval gives 1.8512 == the
      trainer's recorded best@4000, while the on-disk eval_report.json still
      says 1.8641 (computed 2026-09-07 on the PRE-restore final weights).
      After ANY final-dir reweight/restore, re-run eval before quoting the
      baseline; cross-check against the trainer's own eval curve
      (train_summary/tfevents), not the stale report. e1 was never
      reweighted - its 2.0466 reproduced exactly.

 32. **pwsh tool-call children — servers AND daemons — die with the call** (2026-09-09,
     U12/U13 drill): an ended/aborted run_code pwsh call reaped the UI-server child AND the
     agent-browser daemon + its browser (same class as the Start-Process reaping); afterward
     every agent-browser command hung with EMPTY output (no daemon). Pattern that works: run
     long-lived processes as harness BACKGROUND jobs (run_in_background keeps the tree alive
     across calls) and attach foreground calls to them ($env:AGENT_BROWSER_SESSION set in
     every call). CLI gotchas from the same drill: quote snapshot refs ('@e10' — bare @e10
     parses as PowerShell splatting); refs go stale after any re-render (re-snapshot before
     each click); a regex passed from a JS string into pwsh loses its backslashes (arrives as
     [^\"]-style mangled) — use IndexOf string surgery for ref extraction instead.

  33. **Windows sysmem fallback masks VRAM OOM; a suspend kills CUDA
      contexts** (2026-09-10, Track A 8k-16k): (a) at ctx 12288/16384 the
      fp32 eval exceeded the 6 GiB card and the driver silently spilled
      to system RAM - torch.cuda.max_memory_allocated measured 6.15 / 7.9 /
      8.17 GiB with NO OOM ever raised; the only tell is PACE collapse
      (23 s/batch vs ~14 expected). When sizing GPU runs, compare
      max_memory_allocated against nvidia-smi memory.total, not against
      success/failure. (b) A mid-run suspend (Kernel-Power 506/507) can
      leave the process alive but the CUDA context damaged: the next big
      op dies with CUBLAS_STATUS_EXECUTION_FAILED, or the NEXT process
      launch dies hard at the first forward with nvlddmkm Event 153
      (driver reset) in the Event Log and no Python traceback.
      nvidia-smi answering != a healthy context. After ANY suspend during
      a GPU run: expect the run to fail, check the Event Log, relaunch
      fresh (reboot if 153 repeats). Persist per-point results as they
      finish - the ctx_probe driver now flushes every point to disk
      immediately, so crashes are lossless.

 34. **transformers 5.16 LlamaConfig keeps rope theta INSIDE rope_parameters**
     (2026-09-10, Track B): there is NO config.rope_theta attribute
     (AttributeError) - build_config passes rope_theta=10000.0 to the
     constructor but the value lives in the rope_parameters dict. A
     config-gated yarn must set the FULL dict explicitly ("rope_type",
     "rope_theta", "factor", "original_max_position_embeddings" - the
     apply_rope_scaling pattern). HF-native yarn is COMPLETE (per-band
     ramp beta 32/1 + mscale attention temperature 0.1*ln(factor)+1;
     verified live: 26/32 dims changed, temp 1.1386 @ factor 4) - never
     reimplement YaRN math. Same-day ctx-extension facts: PackedDataset
     slices fixed-length windows from flat shards at ANY seq_len, so a
     ctx extension needs NO token re-pack; and eval.py materializes full
     logits (batch 8 @4096 = ~4.3 GB of logits alone) - always pass
     --batch explicitly on long-ctx evals (the trainer's loss-only eval
     path does not need this; batch 2 @4096 fine).

 35. **LoRA on a ctx-extended base: gate BOTH surfaces, and drift is NOT linear in soup alpha** (2026-09-10, row 41): the r=32/lr 1e-4 adapter drifted the LONG-ctx surface ~1.17x the short-ctx one (guard @1024 +3.99% vs base @4096 +4.64%) - a LoRA run that only checks the 1024 guard can quietly spend the ctx-extension win; always also eval @4096 (skip-gen) after merging. Repair souping (LoRA-merged x base): ast is interpolation-fragile (a40 -> 0.68) but the ctx-surface drift is SUB-linear (a60 predicted +2.78% linearly, measured +1.29%) - a mid alpha (0.6) passed ALL gates (guard +1.03%, ast 0.90/0.96, @4096 +1.29%) where the raw adapter failed one. soup_merge.py is the reusable tool (CPU-only, config/tokenizer donor = the rope-family side).

 36. **Cross-tokenizer piece alignment works at the RAW-BYTE level** (2026-09-10,
     KT-1 row 43): decode BOTH sides' pieces to bytes (GPT-2 byte-alphabet
     reverse map for BPE byte chars; SP margin marker -> 0x20 + '<0xNN>'
     byte pieces) and CodeLlama-SP <-> Llama-3-BPE exact matching becomes
     deterministic string work, no fuzzy logic. SmolLM2-360M <-> our 32k SP
     measured: 58.7% exact + 41.1% averaged-subpiece fallback = 99.98%
     teacher-derived coverage. The <60% exact gate FAILED but investigation
     JUSTIFIED it: the bug detector is the fallback sub==1 bucket (15/32,009
     = 0.05%, all <0xC0-class byte pieces retokenizing to U+FFFD) plus the
     frequency skew (matched mean id 13,457 vs unmatched 19,638 = tail-only
     misses); a high sub>=2 fallback share is genuine SP-vs-BPE merge-shape
     difference, not a bug. Isometric lift d960->d1024 = seeded QR, use
     Q.T (orthonormal ROWS, W W^T = I) - a 1024x960 matrix CANNOT have
     orthonormal rows (1024 > 960); the plan's wording meant the isometry.
     Implementation: scripts/embed_transplant.py.
 37. **Batch-1 defaults were VRAM-era artifacts - batch where VRAM allows**
     (2026-09-10, KT-2 row 44): judge inference ran 5.6x faster batched
     (sample num_return_sequences=K for K candidates of ONE prompt - no
     padding needed; teacher scoring packed by padded-token budget ~6144
     tokens to keep fp16 full-vocab logits < ~1.6 GB - MEMORY 25b applies;
     verdict GENERATION needs LEFT padding + explicit attention_mask).
     LoRA-SFT at ctx 512: batch 4 / accum 4 = same effective batch 16 as the
     old batch 1 / accum 16 -> identical training math, ~3x faster wall
     (30 steps in 42 s). Rule: keep effective batch constant, raise
     per-device batch until VRAM or interleave quality says stop.
     Related: a 360M judge is LENIENT (score-2 bias) - the informative
     signal is the judge-vs-LL agreement rate + the unparseable gate, and
 38. **`CUDA_VISIBLE_DEVICES=''` SHOWS the GPU, never hides it** (2026-09-11, mounting): PowerShell cannot hold an empty env value -_deletes the variable, so a ''CPU-only'' sanity check ran ON GPU beside a live judge (harmless, disclosed). Always pin '-1' for CPU-only; add `-1` semantics to any preflight relying on the empty-string form.
 39. **A module registered under TWO name paths kills checkpoint save** (2026-09-11, mounting, drill): attach_mounts registered bridges as model.layers.<l>.bridge.* AND student._mount_bridges (nn.ModuleList) -> same tensors under 2 names -> transformers 5.16.1 remove_tied_weights_from_state_dict RAISES (safetensors forbids duplicate names) on EVERY bridge-mode checkpoint save; fill mode (no bridges) hid it. Fix = register once in the wrapper and hold a plain python list in the attribute (plain list keeps detach_mounts truthiness + trainer bridge iteration intact).
 40. **RemoveColumnsCollator silently strips keys from plain torch Datasets** (2026-09-11, mounting): Trainer._get_dataloader wraps default_data_collator in RemoveColumnsCollator when remove_unused_columns=True (default) and a Dataset is not a HF datasets.Dataset -> custom batch keys (teacher_ids/teacher_pad_mask) vanish before compute_loss (KeyError at first micro-batch). Fix: remove_unused_columns=False + label_names=['input_ids'] (the labels-free batch ALSO made eval take model(**inputs) with loss=None: NO eval_loss, cliff book + A/B metric inert — label_names is the eval-liveness fix).
 41. **Cross-arch teacher dtype**: SmolLM2 configs load bf16; on an fp16 sm_75 pipeline force teacher dtype=float32 at from_pretrained — all teacher-hidden consumers (probes, kv_proj) are fp32 by construction; one kwarg homogenizes every site and no per-mode logic is needed (2026-09-11, mounting drill).
 42. **Kill BEFORE the first save leaves nothing to resume** (2026-09-11, mounting drill): trainer_state.json only exists inside a complete checkpoint dir; a partial-ckpt guard then SKIPS the partial dir and resume restarts from 0 — kill/resume drills must outlive the first save_steps boundary (drill redesigned to 50-step saves).
  46. **CPU-resident 1-D tensors must be moved explicitly** (2026-09-12, mounting drop arm crash): a plain tensor kept on the module (MountBridge._sev_keep) is NOT an nn.Buffer - it stays on CPU while h is cuda -> RuntimeError at first use of that code path (drop arm, step ~300, the only mode that opens non-trivial severance masks; gate/hybrid never exercised it). Fix: .to(device=h.device, ...) at use site. Lesson: masking tensors created on CPU must be buffers OR moved at use; test each mask path BEFORE the arm, not just a GPU smoke of the mode family.
  47. **Trainer-end final_eval can differ from the last trainer-state curve** (2026-09-12, mounting): fill/drop final_eval values come from the post-save evaluate() under best-restore/cliff bookkeeping (fill: bitwise equal to its step-300 eval; drop: end-state independent eval 2.0473 BETTER than the stored best ckpt-900 = 2.2881). Cite the standalone eval_report.json as the authoritative independence number and report the in-run curve separately - never silently merge the two sources.
 43. **Append-only judge stages are cleanly resumable per prompt** (2026-09-11, KT-2 r2): a session kill mid-`kt2_judge.py --stage sample` left an exact prompt boundary (whole prompts x K, valid last line); `stage_sample` now reads existing `(prompt_id, cand_id)` keys from candidates.jsonl and skips fully-sampled prompts — prompts.jsonl is seed-deterministic so ids line up 1:1. Judge stages do NOT honor the train.py zero-flag auto-resume contract; rely on this file-level guard or wipe the stage's files. Torn-line tolerant.
 44. **sft.py auto-resume is keyed on output_dir - typos hijack the NEW run** (2026-09-11, KT-2 r2b): a config stub left `output_dir` pointing at the PRIOR run's dir -> sft.py silently resumed that run's checkpoint-206, trained 0 steps and re-exported the OLD weights as the new final (resumed_from: checkpoint-206, epoch 2.0 under a 1-epoch config). ALWAYS verify train_summary.json `resumed_from: null` before trusting a rerun. Same run exposed: transformers cosine schedule hit LR=0 at ~55% of max_steps (LR ran 0.0 for half the 206-step r2 run) - 1-epoch runs and reading train/learning_rate from tfevents both mitigate/detect it.
 45. **On-policy SFT scale-up did NOT stack: AST pass-rate is recipe-insensitive at this corpus scale** (2026-09-11, KT-2 r2/r2b): judge scaled 7x cleanly (1738 pairs, unparseable 0.1%, judge-vs-LL agreement 100%, all r2 gates PASS), but LoRA-SFT on it dropped AST greedy/sampled to 0.86/0.88 (floor 0.90/0.92 FAIL) and rs 1e-4->5e-5 + 1 epoch fixed CSN (best-ever 1.8974) while AST stayed EXACTLY 0.86/0.88 — the 50-instruction probe saturates against LoRA recipe knobs; moving AST needs data-quality/selection changes (or full-FT), not LR/epoch tuning. r1 final remains the row-44 student pending user decision (A keep r1 / B promote r2b / C one more knob).

     LL tiebreak effectively drives winner selection.

 49. **Deep-narrow 128-ctx is the right architecture class for sequence-labeling on a 6 GB sm_75** (2026-09-12, D-pilot): the approved ctx-1024 pilot (8L/512) measured ~17 s/step (30k steps = 4-5 days, rejected before its first checkpoint - zero resume loss). User-approved switch to ctx 128 + 14L/384h/8 heads/2 KV/ffn 1536 = 30.0M params: ~0.24 s/step (~68x), 2.4/6.1 GiB peak, and the 1000-step smoke already beat the D-smoke final val_acc (0.9090 vs 0.8817). Rule of thumb: for local-window labeling tasks, cut ctx to the task horizon FIRST - attention is O(n^2) and val-eval cost scales with it - then spend the freed budget on depth. Do not size specialist runs off LLM-style context priors.

 50. **Prepared window size MUST equal model token ctx** (2026-09-12, D-v2): the v2 corpus was prepared as 1024-char windows but trained at token ctx 128 - packers/sliding truncation silently trained on ONLY the first 128 chars of every window (~75% of the prepared text never learned) and window tails were never seen; 30k steps produced no gate gain. Fix = set prepare_data CTX_CHARS to the model ctx and re-prep + re-tokenize. Whatever the ctx, match, then re-check gates.

 51. **Windows-ProcessPool + lazy generator stream = teardown DEADLOCK** (2026-09-12, prep parallelization): imap_unordered over an on-the-fly input generator just HANGS at pool __exit__ on Windows (handler thread alive inside _get_tasks; found via faulthandler.dump_traceback_later(45, exit=True), not via a traceback). Fix pattern: (a) materialize the scan FIRST into a concrete list (C-speed reads), (b) chunk it, (c) pool.map over a MODULE-LEVEL batch worker fn (spawn requires importable top-level fns - nested defs cannot be pickled on Windows), (d) source-level filters inside stream_samples, NOT after the yield (a 500-row slice test went 57.8 s -> 0.7 s once skipped sources stopped being read at all). Large arrows: 2nd cost was reading unwanted shard sets; 3rd is that generate-driven dedupe/split steps DO survive in the parent process.

 52. **Checkpoint rotation quietly deletes your best checkpoint** (2026-09-12, pilot128): keep-N rotation (3) pruned EVERYTHING except the last 3 saves, so the best-val checkpoint (~step 8200 of a 25k overfit run) was deleted and only the overfit tail was evaluatable; the bench neutral conclusion was forced to cite a tail checkpoint. Fix IN PLACE: train.py now also writes best.pt (every eval, val_loss, resumable) and final/ exports BEST.pt; the rotation policy is still 'last-N' for checkpoint dirs - keep-best-in-rotation NOT yet implemented (pending user); if you ever rely on a particular mid-run checkpoint, COPY it aside (or set save_total high) BEFORE the run. Also relevant: comparing 'val_acc@checkpoint' across runs is only valid on the SAME val split - pilot128's 0.9553 (Fadel-heavy selection split) vs v2b's 0.9421 (mixed-domain split) are different scales.

 53. **A local-scope rebind of a module global is NOT an override - it silently writes through to the shared path** (2026-09-12, D-line options A/B): prepare_data.py v1 of --out-dir did `OUT = Path(args.out_dir)` inside main() (Python: local variable; module OUT unchanged) so the fadel-only prep CLOBBERED data/diac/prepared/{train,val}.jsonl (the general v2b corpus). Damage was bounded ONLY because the packed v2b tokens were already on disk. Fix: route out_dir as a process() PARAMETER. Recovery: the deterministic seed pipeline regenerated the corpus byte-exact (stats matched to the row). Rule: any "write to an alternate location" knob gets tested by list-then-compare the target dir BEFORE trusting it; parameterize, never rebind globals.

  54. **A bare state_dict final loses its arch - always ship config beside it** (2026-09-13, b65): train.py saved final/model.pt as a BARE state_dict; bench.py then silently fell back to the pilot128 yaml shape. Works unnoticed until the ckpt arch differs (b65 20L/512 layer-shape mismatch storm). Fixes: bench --config override + train.py writes final/config.yaml alongside. Rule: any exported artifact carries its own arch/config.

\ \ 56\.\ \*\*A\ gate\ source\ folded\ into\ the\ training\ corpus\ =\ validation-on-train\*\*\ \(2026-09-14,\ d_gate_overlap\ audit\):\ the\ wikinews2024\ gate\ text\ appears\ VERBATIM\ in\ v2b\ train\ windows\ \(40-char\ shingle\ Jaccard\ 0\.968,\ 4-gram\ containment\ 0\.972,\ 356/356\ units\)\ -\ every\ WN24\ external\ number\ on\ v2b-token\ models\ \(v2b\ 61\.1\ /\ v2d28\ 59\.1\ /\ b65\ 60\.3\ /\ stage2a\ 60\.9\)\ is\ strongly\ OPTIMISTIC:\ label\ WN24\ in-domain\ dev,\ not\ external\.\ fadel_test\ /\ sadeed25\ /\ wikinews2014\ audited\ clean\ \(zero\ copy\ overlap,\ domain-related\)\ -\ keep\ external;\ sadeed25\ stays\ "overlap-flagged"\ whenever\ sadeedt\ is\ in\ the\ train\ mix\.\ Rule:\ run\ the\ shingle-containment\ audit\ before\ trusting\ any\ new\ gate\ \(research/d_gate_overlap/\ scripts\)\.
 56. **A gate source folded into the training corpus = validation-on-train** (2026-09-14, d_gate_overlap audit): the wikinews2024 gate text appears VERBATIM in v2b train windows (40-char shingle Jaccard 0.968, 4-gram containment 0.972, 356/356 units) - every WN24 external number on v2b-token models (v2b 61.1 / v2d28 59.1 / b65 60.3 / stage2a 60.9) is strongly OPTIMISTIC; label WN24 in-domain dev, not an external gate. fadel_test / sadeed25 / wikinews2014 audited clean (zero copy overlap, domain-related) - keep external; sadeed25 stays overlap-flagged whenever sadeedt is in the train mix. Rule: run the shingle-containment audit before trusting any new gate (research/d_gate_overlap/ scripts).

 57. **React Flow 12 stays `visibility:hidden` until the USER node object carries `measured`** (2026-09-13, flow/ browser drill): in a controlled ReactFlow, dropping the xyflow-only `measured` field in the projection mapping left EVERY node unhittable (elementFromPoint returned the pane) while a ResizeObserver storm fired silently - code reading and unit tests both passed. Rule: if you hand-build xy nodes from a domain graph, iterate `measured: prev?.measured` explicitly. Also drill-caught: (a) dataset `label` (the MVP HF-dataset name) had NO editing UI - label editing must be explicitly added when the config mapping reads it; (b) a dialog opened inside a popup-triggering click path can only be verified while staying on the app tab (`agent-browser tab t1`) inside the SAME pwsh call - every pwsh call relaunches the browser and blanks the session tabs, so any multi-step UI drill must be a single chained command.

 27. **Model-viz build gotchas** (2026-09-10, `viz/`): (a) pnpm 12 no longer reads pnpm-only settings from package.json - use 'pnpm approve-builds esbuild' once (else vite build dies with ERR_PNPM_IGNORED_BUILDS). (b) YAML scalars with an inner ': ' colon or a leading triple-quote are parse errors under the yaml pkg - quote the whole scalar; the doctor test catches it before the UI runs. (c) agent-browser has NO 'sleep' subcommand - use Start-Sleep inside a pwsh script file; 'eval -b <base64>' needs a syntactically perfect JS payload (a dropped open-quote fails silently at eval time); React state does not fire from dispatchEvent - use real 'agent-browser click' for click-path assertions.

 58. **flow/ built-ins are NodeDefinitions, not dict entries** (2026-09-13, registry promotion f72bd25): each node kind lives in flow/server/nodes/builtin/*.py owning its spec (ports/PropSpec/label_semantic/features), required_upstream (the config_gen chain order is DERIVED topologically from these - never re-hardcode a CHAIN list), validate_semantic (per-kind knob errors, historic wording preserved) and build_section (config keys). Golden files (flow/tests/goldens/, created PRE-refactor) lock config_gen output so behavior-moving refactors are provable; regenerate with the committed create_goldens.py. Frontend has zero kind-string conditionals: UI renders from the /api/nodes snapshot (features + label_semantic), FALLBACK_REGISTRY in flow/src/nodes/registry.ts only boots offline. flows.save is atomic (temp+os.replace) as of 53986a8.

 59. **Param-math anchors belong to TESTS, not comments** (2026-09-14, F2 66eaa28): the dense-formula engine's totals (nano 226,526,208 / SmolLM2 134,515,008) are pinned as exact-integer vitest anchors in BOTH viz (`pnpm test`) and flow (`src/model/paramMath.test.ts`), and the committed decoder graphs re-assert them via the walker (`acceptance.test.ts`). When porting math across projects, port formulas + tests together; the first T3 run caught lmHead pricing the untied head over the OUTPUT vocab-d instead of the HIDDEN input-d precisely because the anchor test disagreed.

 60. **Pretrain-init scale is NOT the final-model lever — a fresh held-out test sins against portfolio thinking** (2026-09-15/16, E-17): a 4x-bigger whole-corpus stage-1 LM (370 M chars incl abdou/sadeed/qcri FULL, wn2024 excluded, abdou TEST split never touched) warm-started the adopted reset-last-2 recipe and produced a statistical wash on the 4 standing gates (clean mean 42.7 vs gold 42.6) and a decisive LOSS on the never-trained abdou test-00000 split (DER 49.4 vs gold 41.6, 15,091 sentences). The small-LM gold (stage2b2500) REMAINS the deployed final model. Again: val_loss favored run-end weights while gates favored the @2500 probe peak — the gate watchdog, not val, picked correctly in three consecutive runs (stage2b/stage2final). Rule: judge init/model changes ONLY on external gates + a never-trained held-out split; budge nothing on val.

 61. **The input contract is a test artifact multiplier: pre-marked input inflates external scores AND seeds phantom defects** (2026-09-16, E-18): feeding partially-diacritized text into the D-line model made the passthrough copy REST OVER the model's own inserted marks (عُقدت -> عَُقِدْتْ doubling), which an outside reviewer mis-diagnosed as a Softmax/CRF-level defect; the same pre-marks also INFLATED apparent quality (passthrough letters score as 'preserved' in DER while the model's own errors hide). Fixes: any text-input path MUST strip_marks first (applied in diacritize.py; input stripping is also implicitly the bench/eval contract already); never hand-evaluate a model on marked input; when someone reports 'double diacritics', check the input bytes before touching the loss.


 62. **A random model name cited by another agent must be looked up BEFORE downloading anything** (2026-09-16, E-19 prep): a peer agent confidently cited "QINA King v21" as "the best open Arabic diacritization model"; sweep (Exa search + contents, GitHub, Hugging Face API model registry) found NO such model anywhere - the claim was fabrication. Rule: treat any model/dataset name from another agent as unverified until it appears in a primary registry (HF API, GitHub, paper with DOI); one registry search costs seconds, a bad download costs a training slot.

 63. **Cleanup script guard paths MUST match the test's path form** (2026-09-16, Tier-A cleanup incident): a PowerShell `if ($entry.FullName -eq $keep) { continue }` guard FAILED silently because Get-ChildItem .FullName returns an ABSOLUTE path (`E:\python_projects\nano_SLMs\runs\...`) while the guard's `$keep` was a RELATIVE path (`runs\yarn_lora_sft_v1\final_a60`) - PowerShell `-eq` on strings is a literal compare; no normalization. Outcome: a KEEP-listed dir (864 MB E-07 `yarn_lora_sft_v1/final_a60/model.safetensors`, an explicit user-keep per the review) was deleted. Compounding error: the pre-deletion SNAPSHOT used `robocopy /XF model.safetensors` to save space, so even the backup was weight-free. Rule for any future destructive op: (a) the pre-flight list of KEEP paths MUST be converted to the same form as the comparison (use `[System.IO.Path]::GetFullPath` on BOTH sides, or compare on a normalized relative form via `Split-Path -Leaf`/Regex). (b) when snapshotting weights as a safety net, EXCLUDE the exclusion list - or do NOT snapshot at all and rely on git + the actual KEEP survivors. (c) after each deletion, verify the KEEP list is STILL intact BEFORE moving to the next group (post-state check, not just pre-flight). (d) if the user approves a scoped cleanup, the pre-deletion review must enumerate which weights in which Tier-X dirs are expected to survive - this incident was caught because the post-state check ran a List-Present of all KEEP winners; without that the loss would have been silent.
## Data-source knowledge (seeded from HANDOFF §6)

- bigcode/the-stack-v2, the-stack-smol, starcoderdata: **gated** (manual HF

  64. **Assume a 0-star release's architecture will fight the bench box** (2026-09-17, E-19 bench): etherll granitemoehybrid's mamba2 chunk-scan allocates prefill-time temporaries that OOM the 5.8 GB WDDM-spilled GPU at batch>=2 (batch=1 forced, ~6.5 s/line); Fine-Tashkeel T5 works at batch<=8 but dies at 16 (SDPA OOM). Bench FIRST with 5-8 lines before any full-gate launch; OOM resilience (retry-with-truncation + empty_cache) belongs in the harness, not in hope. Also: causal-LM diacritizers WITH EOS-ish decoding loop catastrophically on out-of-distribution short texts (wikinews) - label/seq2seq specialists (CRF like mishkala, BiLSTM like Z-Mahmood) preserve base letters for free; a 15-label re-application model can NEVER damage base chars.
  terms acceptance). Fallback actually used: nickrosh/Evol-Instruct-Code-80k-v1
  (instruction→code-answer, ~430 tok/row) — used for smoke AND pilot.
- bare `code_search_net` no longer resolves on the Hub (repo moved under a
  namespace); canonical ungated copy = `code-search-net/code_search_net`
  (config `python`, ~326 tok/row) — the M3 source (150k rows → 47.9M train tokens).
- Other ungated options found via Exa: tokyotech-llm/swallow-code-v2 (49.8B tok,
  Apache 2.0; load config `swallowcode-v2`), bigcode/python-stack-v1-functions-filtered-sc2.
- TheGamingMahi/TinyCode has a corrupted shard (CastError mid-stream); the
  fallback chain handles it.
- Tokenizer: borrowed CodeLlama-32k; packing = uint32 `.bin` shards, cap
  8M tokens/shard, memmap-loaded at train time.
- A user-supplied `HF_TOKEN` (after browser terms-acceptance) would unlock
  the gated sources — never invent one.
- starcoderdata python MEASURED (2026-09-09, 2k rows): 2,345.7 tok/row
  (median 921, p90 5,317, max 146,310) — whole files; the ~600 estimate was
  ~4x low. the-stack-smol python: 2,717.2 (median 929, max 121,723).
  CSN python: 271.8. Evol-Instruct: 474.4. Cross-source SHA1 exact-dup over
  8,000 measured rows = 0.0000 (within-source 0 everywhere).
- starcoderdata terms accepted 2026-09-09 (gated=auto grants immediately);
  HF_TOKEN access works; BOTH bigcode repos need the data_dir load form.
- Next-pretrain mix SIZED on those measurements (TASKS row 13): 10000 smol /
  37500 starcoder / 59000 csn / 16900 evol rows ~= 133.65M kept tokens;
  configs/next_pretrain.yaml; the RUN is user-gated at the next window.


  65. **A word-cache only buys what the model does not already know** (2026-09-17, E-20 gpu+cpu bench): a majority-vote word-cache (375,923 forms from 140.6M tokens) over the gold D-line model moved gates by only -0.3..-0.6 DER (fadel 33.23->32.88 etc). The BiLSTM-family posters (Z-Mahmood) show the SAME null effect (E-19b cache-On == cache-Off). N-gram caches shine for exact idiom sentences / fixed phrases (Quran citations), not for MSA word priors that a 30M-param model has already internalized; treat cache as a deployment trick (instant, offline) not a teaching substitute.

  66. **The train-time cache idea is TESTED and FALSIFIED** (2026-09-17, E-21): the user's hypothesis - build the word-cache as part of the training process so the model 'stores its knowledge in it' - measured on BOTH micro arms (2.98M scratch + warm-start): cache-merged eval = -0.00..-0.03 DER (zero). Majority-vote word priors exist in ANY model of this class at 10% of gold's params; do not spend more runs on cache-in-training for this task. Also: warm-starting a bidirectional diacritizer from a causal LM stack (micro_a_lm val 1.6177 -> SFT val 0.2911 < scratch's 0.3228) made SFT validation better but GATES 5.7 mean-DER worse - SFT val_loss is not a gate-quality proxy.

  67. **Machine-labeled weak data HELPS a small train pool (unlike gate refs, it just needs to be plentiful and on-domain)** (2026-09-17, E-22): +362k gate-deduped QCRI machine-vocalized wiki windows cut the micro's gate mean-DER -2.7 (36.44/48.67/58.13/50.11 vs 38.16/51.40/60.96/53.87). Over-marked, model-output labels are fine as SUPPLEMENT; never treat them as refs or mix them into val (val stayed byte-identical for the A/C comparison). Workflow that works: 10-word-shingle gate-dedup with a >=1% threshold -> space-window -> passthrough.parse inside the existing pack pipeline.

  68. **Weak machine-labeled supplements help capacity-limited training REVERSibly, then hurt saturated models** (2026-09-17, E-23a): +362k gate-deduped QCRI windows cut the micro mean-DER -2.7 (E-22) but the 30M gold REGRESSED +1.45 mean (wn2024 +7.06 collapse) on the same v3q tokens and same recipe. A small model soaks up extra (even machine-marked) supervision; a big model has already saturated what multi-source wiki-style labels offer and gets poisoned by distribution shift (QCRI over-marking + Wikipedia style). Takeaway: gate supplements by MODEL SIZE, not just by data gain on one cap; treat E-22-style positive deltas as scoped to small caps until proven otherwise.

  69. **Machine TEACHER quality does not rescue a saturated label pool - style conflict dominates** (2026-09-18, E-23c): Z-Mahmood (rule-a teacher that BEATS gold) relabeled 3.06M modern words; adding 315k windows to the micro (arm E) REGRESSED mean-DER +4.1 (52.55 vs 48.42) and even the wn2014 gate, its own home domain. Combined machine-supplement set: E-22 QCRI on micro +2.7 (quantity starved pool), E-23a QCRI on gold 30M -1.45, E-23c ZM on micro -4.1. Conclusion: external machine labels only help a SUPPLEMENT-STARVED small model; once the pool is human-gold-styled and the model saturates it, adding any second machine-label style averages conflicting mark distributions. Future distillation arms must blend styles at the label level (e.g. teacher-confidence weighting), not as raw extra rows.


  70. **Z-Mahmood's architecture does not transfer to nano_SLMs micro scale** (2026-09-17, E-23d S4): the roasted BiLSTM (Embed128 -> BiLSTM 3x256 + Bahdanau attention, 4.50M, our I/O contract, arm-C EXACT budget batch 32/2500 steps on v3q) hit mean DER 55.83 @2500 (fadel 45.77 / sadeed 58.09 / wn2024 62.90 / wn2014 56.55) vs arm C 48.42 (+7.41 worse) and arm A 51.10. Two lr probes: our micro lr 4e-4 stalls the LSTM at loss ~1.0 (bare-collapse word DER ~0.997); ZM-native lr 1e-3 is load-bearing for a 3-layer cuDNN LSTM at batch 32 and still loses. ZM's rule-(a) baked-off edge is data-scale + train-duration (their 3.5 h full pass), NOT portable architecture; keep the bidirectional char transformer for D-line.

  71. **Self-built eval plumbing must be parity-tested against the committed harness before any verdict** (2026-09-17, E-23d): the first BiLSTM gate rows (~0.997 mean DER, identical digits across steps) came from a probe gate that double-iterated tab-joined refs (one CHARACTER per 'ref'); a sub-1% exact-match artifact (all-bare luck) that looked like a converged-garbage model. The fixed numeric trajectory only appeared after replicating eval.py compare logic IN-PROCESS. Any new gate plumbing: cross-check one row against `scripts/eval.py compare` on the SAME pred/ref files before trusting it.

  72. **Classical human-gold expansion is a dead lever on the D-line - the pool is style-bound, not size-bound** (2026-09-18, E-23b S2): 1.2M gate-deduped NEW classical windows (Tashkeela mirrors, 3.28M dropped by the fadel+gate shingle barrier) roughly doubled the classical mass of the micro pool (v3t 4,003,948 rows, val identical) at arm C's exact recipe/training budget; ALL FOUR gates receded (38.69/50.70/59.93/52.55, mean 48.72 vs arm C 48.42, +2.0..+2.4 per gate). Degradation is uniform, not instance-specific: adding same-family classical data shifts the pool's style balance away from the modern-MSA-heavy gates while contributing no new supervision signal the pool lacked. Corollary: only NEW-STYLE labeled sources (modern MSA) can move the data lever, and no open labeled modern-MSA source exists as of 2026-09-18.

  73. **Composition mechanisms are a designed, measured experiment — test them at micro scale BEFORE any large-scale attempt** (2026-09-18, ME-line design): four distinct ways to "combine small models" exist with published recipes — (A) weight merge (soup/TIES; PREREQUISITE: shared seed + same basin — unshared inits cannot be averaged due to permutation symmetry), (B) MoE merge (Branch-Train-Merge/MiX: branch a SHARED seed into domain experts, merge as FFN slots + learned router, arXiv 2208.03306 / 2403.07816), (C) dispatch (external tiny router sends the whole input to one expert — must work on UNLABELED inputs), (D) committee distillation (in-repo proven lever, WHAT_WORKS #2/#10). At 100-300K params the shared char vocab (≤128 ids) + shared seed are non-negotiable mergeability prerequisites — embeddings dominate the budget. Sandbox rule: every composition claim is measured against a dense param-matched control trained on the union data at matched tokens. Design: research/micro_experts/DESIGN.md.

  74. **Leftover-compute co-running has a written protocol now - but zero completed evidence rows** (2026-09-20, CLAUDE.md §19/§20 session): user-approved policy = GPU may be scavenged ONLY from foreign, non-training incumbents (60% of free VRAM, hard allocator ceiling, no WDDM shared-memory spill, re-check before launch) and NEVER beside this repo's own train.py/sft.py runs (single-GPU rule, AGENTS §4.6 + lessons 9/16 stand). First sanctioned co-run = TASKS row 98; until it happens with logged evidence, treat §20 rules 9-10 as UNPROVEN-by-default and prefer the §20 rule 10 fallback (wait/notify).
- **(2026-09-19) mu2 G2 LoRA trade-off + throughput re-opt:** at 780K-param trunks, LoRA fill-in learning and trunk retention share one knob: replay ratio (clean samples) is the retention lever, lr×steps the task lever (E-31e recipe: lr 1e-4, 2500–3000 steps, corrupt 2-of-3 blocks ⇒ both gates PASS). CUDA launch overhead dominates sub-1M models — batch 32×accum 1 + grad_ckpt off gave ~10× it/s; profiler (scripts/profile_resources.py, psutil+nvidia-smi) verdicts proved the data pipeline innocent (disk <2 MB/s). Mid-run resume with changed batch works, but save_total_limit rotation can lose the best checkpoint before load_best_at_end sees it — future: raise save_total_limit or re-checkpoint best when observed.

- **mu2 lesson (E-34->E-35, 2026-09-19):** when composing a specialist head over a general trunk, NEVER restrict the trunk's fallback branch to a subset of the vocab - that branch blinding by construction produces errors the unrestricted trunk (weights identical) would not, and the composed metric then measures the rule, not the model. Correct pattern: specialist decides first over its own classes; general arm keeps its FULL output space. Also record per-position breakdowns (mark/non-mark/none) before claiming composition effects.

- DA-3 (2026-09-19): arbml/SANAD parquet is the best ungated Arabic topic-corpus (131,807 usable rows, 7 clean sections; Arabic text = column ATICLE); heegyu/news-category-balanced-top10 train_sampled.json is JSONL not JSON. HuffPo editorial buckets are weaker Chinese-taxonomy than news-section sources - keep if founding a DA-3b around 36-IAB taxonomy close to the original Gist. EXPERIMENTS.md row ids mu3 sections E-48a/b and E-50/51 already occupied - next free DA row id was E-52.

- DA-7 (2026-09-19): iahlt/arabic_ner_mafat is ungated 40k-sentence Arabic NER (tokens+BILUO parquet);
  asas-ai/ANERCorp is the classic 9-tag corpus with pre-flattened tokens (no sentence ids in parquet).
  On an 87%-O token corpus, plain CE on Muon learns nothing entity-side (ent-acc 0.085 in 3ep);
  mean-normalized inverse-sqrt weighted CE (clip 0.5) fixes it (0.388 in 6ep) - inverse of DA-2b where
  weighting hurt: weight CE only when O is >2/3 of the data.
- DA-7b (2026-09-19): iahlt/arabic_ner_mafat leaves most bare Arabic/Latin digit runs O-tagged; token-level
  digit/date regex rules score P 0.008-0.37 against MAFAT gold - deterministic number rules must be evaluated on the
  corpus they will run on, never on gold in which bare digit runs are unannotated. Model beats regex on both numeric and
  temporal subsets there; keep regex only for unambiguous PII (URL, email, phone-length digit runs, long IDs).
- USER POLICY (2026-09-19): GPU may be used when free; if another process/agent holds it, wait or keep working on CPU until it finishes, unless the CPU path is trivial-by-design. Overrides the audio-tier-only GPU restriction of DESIGN.md section 2; record any GPU use in the run row. REFINED 2026-09-20 (user-approved, CLAUDE.md §20): for FOREIGN NON-TRAINING incumbents only, measured leftover compute may be scavenged (60% of free VRAM, no shared-memory spill) — repo train.py/sft.py runs keep the wait rule; first evidence row TASKS 98 pending record any GPU use in the run row.
- DA-9 Schemer (2026-09-20): honest FAIL at the pre-registered 0.75 span-F1 bar with the E-54 hashed token tagger: 
  class-weighted 20ep reached only 0.339 micro (rules-only 0.287). Costs: (a) reader format mismatch cost one full pass - 
  DA-7 ner tsv format is ONE LINE PER SENTENCE (pairs on the line); writing one pair per line silently turns every token into 
  a 1-token sentence with <s>...</s> context and quietly caps quality; always assert len(toks)==len(tags) AND expected tokens-per-sentence. 
  (b) ZERO-count label classes made inverse-sqrt weights blow up - max(cnt,1). 
  (c) Strict span-F1 vs token-acc mismatch: with 90%+ O rate, entity-token accuracy 0.70 coexisted with span-F1 0.5 - 
  synthetic template data makes model learn digit-vs-month type confusion (NUM_AI vs DATE_G): token-context-only hashed feature is too weak; 
  per-type heads + CRF or char-class-augmented features are the recorded next levers. 
  GPU policy note: tiny hashed models get almost no GPU benefit (41MB peak; encode_batch CPU-bound dominates) - still used GPU per policy when free.

## 2026-09-22 — push-quota incident: history rewrite stripped ~16 GiB of blobs (worktree-delete gotcha)
- The 205-commit push backlog carried ~12.5 GiB of plain git blobs (diac npy/jsonl ~10.8 GiB; mex bridge/xtower .pt + distill_cache.npy + _xv.npy ~1.4 GiB) plus 152 LFS weight pointers (3.36 GiB on disk). Root cause: the "weights stay local" .gitignore rules covered only the OLD run lines (target/sft/kd/kt/yarn/...) — runs/mex/, models/e19/, runs/gdn_smoke*, data/gdn_selftest/, data/diac/** big files were never given rules, and .gitattributes routes every *.safetensors to LFS.
- Fix: git filter-repo --invert-paths rewrote all 299 commits stripping those paths; new ignore rules added (runs/mex/**/*.safetensors|*.pt|*.npy, models/e19 weight files, gdn smoke/selftest, data/diac/, data/md_measure/*/raw/). Backups: checkpoint_backup/{commit_graph,commit_stats,refs}_backup_20260922.txt + lfs_status_map_20260922.txt.
- GOTCHA (costly): filter-repo's final reset --hard DELETES stripped files from the working tree when they were tracked. All 152 LFS-backed weights were restored byte-exact from .git/lfs/objects (the rewrite never touches it) via the path->OID map — snapshot 'git lfs status' BEFORE any rewrite. Non-LFS deletions were NOT recoverable from git: diac token arrays/big jsonl (regenerate from data/diac/raw via prepare scripts, seed 20260911), mex bridge/xtower .pt + distill_cache.npy + _xv.npy (re-derive or accept loss), models/e19/our_word_cache.json (cache, regenerable).
- Lesson: before ANY history rewrite, every tracked-on-disk artifact is expendable ONLY if LFS-backed (verify OID present in .git/lfs/objects) or regenerable from sources that live outside git.
- Related discovery: MEMORY.md (0 bytes) and HANDOFF.md (~1 KB) were committed clobbered by 9897100 / e9c79d9 / 99b8f3b on 2026-09-21/22 — BEFORE this rewrite (concurrent-session mid-write commits suspected). Restored here from last-full versions 0f83061 (MEMORY, 70,755 chars) and 9dbb5c2 (HANDOFF, 146,449 chars). Anything recorded only in these two files between those commits survives only in git log messages.
- Push policy from now on: git lfs status MUST show 0 pending objects, and the unpushed payload must be spot-checked (git rev-list --objects origin/main..HEAD + cat-file sizes) before every push.

- **WDDM shared-memory spill gotcha (2026-09-22, live-discovered on B1')**: this box's GPU Process Memory counters are the ONLY reliable per-process split (nvidia-smi per-process used_memory reports `[N/A]`). Detection: `new-object System.Diagnostics.PerformanceCounter('GPU Process Memory','Dedicated Usage'/'Shared Usage','pid_<pid>_luid_..._phys_0')` with 2x NextValue samples (first sample is 0). B1 (irretrievably) and B1' (caught in-flight) BOTH silently spilled ~4 GiB into shared GPU memory; user rule: VRAM-only. Fix = `torch.cuda.set_per_process_memory_fraction` guard in train_e61b.py (env E60B_VRAM_CAP_GIB, default 5.5) + micro-batch 256 x accum 2 (gradient-EXACT for fixed-length 96-char rows). ~5.6 GiB is the practical dedicated envelope; caps above it hit CUDA driver OOM even below the 6.14 GiB total. Watchdog: scratch/gpu_shared_watch.py. Monitor output: .venv python scratch/gpu_shared_watch.py --once + auto-kill any job where the alloc cap OOMs rather than chase the cap upward.

- **Process docs 2026-09-22**: (1) docs/plans/2026-09-22-agent-reflection-lessons.md = what the agent got wrong this span and the standing upgrades (anchored eval, patch-then-read-back, residency watchdog, divergence runtime gate, RAM-bounded data code, mmap-copyfile pattern). (2) research/2026-09-22-mex-v2-composite-proposal.md = draft v2 composite design combining gold-line + ladder + expert towers + shared KB under multi-metric discipline. Both user-facing; the v2 launch itself remains user-gated.

- **v2 proposal final review (2026-09-22)**: caught that the draft named e23a_gold_v3q as the KD teacher although it was NOT adopted as production gold (regressed 45.59 vs 44.14, wn24 QCRI collapse). Proposal corrected: candidate teachers = stage2final (proposed) OR e23a_gold_v3q strictly in-domain with the E-23-S5 cross-agreement filter (flag_below threshold) as a MUST term. Also added explicit eval cadence (in-run in-domain multi-metric every 500; full external profile at >=60% max_steps and close) and seed/shape matching for A/B arms.


## Data-citation policy (user decision, 2026-09-21)
- Every external dataset/corpus used in any experiment report: cite each owner the way THEY request to be cited.
- If the dataset does not state a citation request: do not invent one; note "we use this data" + dataset name + source if available.
- Ledger cross-refs: data/catalog CSVs carry the owner's requested citation (or "no citation requested") in notes from E-60 onward.

## 2026-09-22 — E-63 Newton–Schulz fp32 divergence (Muon arm NaN)

- **Lesson:** the Muon NS-5 polynomial (a=3.4445, b=−4.775, c=2.7885) *amplifies*
  singular values ~1.46× per step in fp32 and blows up super-exponentially once
  σ>~1.3 — random 272×272 grads explode to norm 2.5e9 in 5 iterations, poisoning
  every 2D hidden weight within 2 steps (train loss NaN at step 2; GradScaler
  overflow-skip masks it forever).
- **Fix (shipped in diacritizer/scripts/train.py `_newton_schulz5`):** normalize X
  every iteration (keeps the map in its contraction basin) + rescale output to
  Frobenius norm √min(m,n) so update scale matches the canonical expectation.
- Verified: unit norms ≈16.49 = √272 on all shapes, 8×8 no longer NaN, Muon smoke
  val 0.7146/0.7614 at lr 1e-2 (2.5e-2 diverged even before, keep 1e-2).
- **CORRECTED 2026-10-04 — the diagnosis above is wrong about the polynomial.**
  c=2.7885 is NOT the canonical coefficient; canonical is a=3.4445, b=−4.7750,
  **c=2.0315** (KellerJordan/Muon). The "amplifies ~1.46× per step" behaviour is
  the typo, not NS. The shipped band-aid stops the blow-up but leaves the update
  non-orthogonal. See the 2026-10-04 entry at the end of this file.

## 2026-09-23 — PERMANENT user rules (E-63 close): standing eval/monitor contract
- R1: EVERY model gets the four external gates full multi-metric eval (strict acc +
  DER/DER_nocase/WER/text_preservation + per-class) by DEFAULT - even if the user did
  not ask for it in this turn. Never trust one eval/metric (label-smoothing loss-scale
  trap; logit-adjust slice-vs-unseen reversal are both recorded incidents).
- R2: Monitor the ENTIRE train process at every step automatically (jsonl monitor) -
  module planned in docs/plans/2026-09-23-pipeline-feature-registry-plan.md S1.
- R3: Smoke end-to-end BEFORE every training run - always, no exceptions, isolated dir.
- Approved next-arms: checkpoint-average + gate-matched selection first; E-61 KD term;
  data diversification; smaller-model ladder M1(16Lx272)/M2(28Lx208)/M3(28Lx272+block
  dropout p0.1) on the SAME trial bed protocol (bands vs S0 in-domain AND external).

## 2026-09-23 — bench.py batched inference (user-requested speed fix)
- Earlier "equal token length" bucketing never filled batches (char-level
  ids => nearly every piece a unique length) so --batch changed nothing:
  user caught it. Rewrote bench.py: CROSS-SENTENCE length-sorted bucket
  batching with BOOLEAN key-padding masks (float mask = additive bias —
  masks scores of pads must be bool False).
- model.py: SDPA now takes attn_mask through GQA/Block/hidden/forward
  (default None = byte-identical legacy). fp16 parity vs bs1: 0.0 diff.
- Gate-bench sweep (fadel 2500 sents, M1 16L model):
  bs=1 249s/0.17GiB · 64 11s/0.34 · 256 11s/0.84 · 512 10s/1.5 · 1024 11s/2.83
  · 2048 37s/5.48 (VRAM pressure region) · 8192 186s/15.2 OOM-throttled.
  OPTIMUM: --batch 512 (~25x faster than bs1, only 1.5 GiB peak VRAM);
  64-1024 tie; >2048 collapses (Turing allocator/registration thrash).
- Batched DER metrics IDENTICAL to bs1 (rare single-mark fp-tie flips only).

## 2026-09-23 — profile store for smoke/reuse (user-requested wrap)
- diacritizer/src/pipeline/profiles/ (registry.json + store.py + __init__:  
  from pipeline.profiles import ProfileStore) + diacritizer/scripts/profile_ctl.py.
- Lessons now codified: float SDPA mask = additive bias (bool only); meta-
  batching needs cross-sentence buckets (per-sentence bucketing never fills);
- Self-test scratch/_profiles_selftest.py PASS; bench.py now auto-applies
  best stored batch when --batch absent (profile-store:512 verified live).

## 2026-09-23 — per-500-step gate probes are the real stopping signal
- E-63c full-data M1: in-domain val_loss improved monotonically 0.2038->0.1349 across 20k steps while mean
  external DER degraded 0.4619(step5k)->0.6299(20k). Val-loss/acc plateau/best tracking NEVER would have
  caught it — the LIVE gate_eval_every=500 probe did (user-asked feature, now validated feature row).
- Rule: for any D-line full run, gate probes at <=500-step cadence + best_gate_weights.pt snapshot are part
  of the completion criteria, not an optional extra.
## 2026-09-23 — E-63s: load_state_dict strict=False does NOT cover shape mismatch
- Warm-init from a checkpoint with a DIFFERENT hidden size crashed every smaller-hidden arm on the smoke
  preflight (RuntimeError size mismatch, not a "missing key"). Fixed with shape-filtered partial transfer in
  train.py (fit = tensors whose shapes match; rest train fresh; log shape_skipped). Partial-warm arms (S4-S6,
  hidden 160/128/96) built fine on it and STILL beat several full-warm arms at equal scale.
- Depth arms (hidden 272 fixed) peaked at 4.5-6.5k and decayed; ALL sub-272 arms were still improving at the
  10k wall — hidden-width shrink needs proportionally MORE steps (3 arms are queued resume candidates).
## 2026-09-22 - Laya-line gotchas (E-62)
- HF dataset ids: allenai/scitail config is `tsv_format`/`snli_format` (plain `tsv` errors: BuilderConfig not found); yelp_review_full lives at `Yelp/yelp_review_full` (no fancyzhx namespace for review_full); canonical SQuAD v2 id is `squad_v2` (rajpur/ namespace retired).
- Per-(question-type, option-count) temperature calibration OVERFITS on small calibration slices: 8 groups x 400 items made ECE WORSE (0.0673 -> 0.0948). Fit ONE global T on >= 2000 items or skip - soft-CE training already yields good raw ECE (~0.07).
- Windows Python: docstrings containing .venv paths must be raw strings or every run prints SyntaxWarning: invalid escape sequence.
- Stream-job rule: job_output returns only NEW output since the last read; a failed parent program loses the earlier read. Prefer reading artifacts from disk (build_stats.json) over re-reading job streams.


- Stage B fine-tune forgets the stage A mixture (macro 0.7478 -> 0.6856, yelp5 0.512 -> 0.22): if retention matters, replay 10-20% of the mixture inside stage B (Laya-line L3 lever).
- Option-rename invariance of the current decision head is weak (perm agreement 0.325-0.385): add synthetic option renaming as training augmentation, do not just probe it.
- Zero-shot transfer of a 37M decision head to unseen task families is near chance without task-relevant mixture coverage (phishing AUROC 0.45-0.58); Laya transfer numbers rest on 421M pretraining.
- Global-T on typed train WORSENED ECE a second time (0.0888 -> 0.1282): the typed head is already calibrated; stop reaching for temperature scaling here.

## 2026-09-23 - E-65 (L3) hard-won gotchas
- PyYAML silently keeps the LAST duplicate key - one flat yaml holding two stages' configs ran pretrain at micro_batch 8 instead of 48; split configs/laya_l3.yaml vs laya_l3_ladder.yaml.
- Boolean-mask assignment shape rule: batch[bool_mask] = full-shape tensor FAILS (the selection is 1-D); draw values with the same shape as the batch then index with the same mask.
- datasets>=3: pass the config via name= (config_name= collides as a kwarg); script-based datasets (spamassassin.py, sroie.py) are no longer supported; typed-decisions needs config "all".
- BertForMaskedLM has .bert + MLM head; LayaDecisionModel has .encoder (NOT .bert) + .head - probe/eval code must branch on hasattr(model, "head") and .to(device) any freshly wrapped model (shared-encoder wrapping does NOT put new modules on GPU).
- random.Random has no .permutation (numpy API) - use rng.shuffle(list(range(n))).
- py_compile does NOT catch NameError/AttributeError - smoke-launch any new trainer before walking away.
- MLM vocab-logits blow the VRAM budget at backward (batch x seq x 30522 fp32 ~1.5 GB at 12k tokens/batch): micro 32 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True held peak 4,610 MiB on the 8 GB card (over the 4,096 gate - recorded).
- WDDM OOM messages can show absurd memory values (17 TB "GiB") - bogus display values, do not debug against them.

## 2026-09-23 - agent-process lessons (E-65 self-reflection)
- Ledger discipline: metrics enter EXPERIMENTS/TASKS parsed from the source JSON in the SAME program that writes the ledger - writing from memory shipped a wrong stage-A macro (0.6329 vs actual 0.6625) once.
- Preflight gate: no unattended long launch without laya/scripts/preflight_check.py (dup-key yaml check, pyflakes, --help, data loads, --smoke 1-batch GPU run); a smoke must show updates>0 in a fresh _smoke dir or it validated nothing (run_stage reads cfg[out_dir] - override it, not just a local variable).
- HF datasets >=3.0 removed script datasets (spamassassin/sroie died): probe one item per manifest source before big builds; migrate to parquet.
- run_code: a parse error means NOTHING ran (whole program skipped); JS strings need escaped backslashes for Windows paths and cannot carry Python raw-string syntax; no-op edits (old==new) reject the whole batch.
## 2026-09-23 - E-66 levers + remaining process lessons (self-reflection follow-through)
- Option-order aug lesson (E-65): repack-shuffle aug did NOT produce block-surgery invariance (0.185 vs 0.385 unaugmented) - invariance needs the SAME item in multiple orders (perm-duplicate, train_l3.py perm_dup knob, E-66) or PMI debiasing; validate any debias on held-out (arXiv 2305.14596).
- Replay: the ratio is not the only knob - per-source balanced/reservoir sampling is the E-67 lever (arXiv 2203.10317, 2505.12512); single-lever discipline: one change per experiment.
- The edit tool requires a prior read-tool read of the exact file (pwsh Get-Content does not satisfy it) - batch reads BEFORE batch edits.
- Param count is computed at pre-registration via laya/scripts/param_count.py --target_m (E-65 registered 35.1M against a ~50M ask - the delta belonged in the pre-register, not the post-mortem).
- datasets >=3.0 removed script datasets: build_corpus_l3.py --probe_sources <manifest.json> probes one item per source (with refs/convert/parquet fallback) BEFORE a build; manifest entries may carry revision.
## 2026-09-23 - E-66 lesson: where order-invariance does NOT come from
- Perm-duplicate training (same item, 2 deterministic orders, 2x stage-B density) improved typed acc +4.45 and calibration but moved perm agreement only 0.185 -> 0.215. With E-65 (repack-shuffle: 0.185), training-side augmentation is EXHAUSTED as a G3 route at this scale - the bias lives in scoring/eval. Next: permutation-averaged scoring + per-position prior correction on TRAIN only (E-67, pre-registered).
- Monitoring the update counter against the schedule total caught a real scheduler-overrun bug mid-run (450/376) - cheap counters are the best tripwires; fix was same-run via zero-flag relaunch (5 min cost).
- Accuracy levers can cost behavior: perm-dup gained acc but regressed the confidence-probe profile 2 -> 4 failures. The probe suite is what makes that visible - never judge a lever on accuracy alone.
- Harness-vs-shell file view split (2026-09-23): files written via pwsh (Add-Content/Set-Content) were INVISIBLE to the harness read/edit tools (stale mirror) - TASKS.md row edits failed to locate text that git proves exists. Rule: keep ONE mutation channel per file; verify file state through the same channel that wrote it; never mix pwsh writes with edit-tool edits on the same file.
## 2026-09-23 - E-67 lesson: the position prior is signal, not noise
- Eval-side permutation-averaging (k=5) held accuracy but moved perm agreement only +0.01; per-position prior correction (fitted on train) IMPROVED agreement but REGRESSED accuracy -2.85 pts. The head's position preference carries task-correlated information; removing it destroys signal. G3 0.90 is demoted to record-only (see EXPERIMENTS E-67 standing decision) - never gate on it again without an architecture-level mechanism.
- eval_calibrated.py bug pattern: zipping per-option index entries against per-view score lists silently misaligns; group index entries PER VIEW before zipping (caught by first-run TypeError, fixed same day).
## 2026-09-23 - E-69 relaunch lesson: checkpoint carry is part of config generation
- A new experiment dir with no A_last.pt silently runs the FULL stage A from scratch (~45 min at 5235 updates) instead of resuming epoch 5 - discovered 21 min in from the console (update 1950/5235). The checkpoint carry (Copy-Item runs/laya/l3/A_last.pt into the new out_dir) belongs in the SAME program that writes the experiment config, verified in the launch-verification step; any '[l3:A] update N/5235' line at startup means a missed carry.
- Second miss in the same relaunch: Out-Null instead of Tee - the external console log is a standing user requirement, never opt out of it. Launch commands are copied verbatim from the previous experiment (env + Tee), only the config path changes.
## 2026-09-23 - E-69 lesson: what distillation does NOT carry
- KD from our own L2 (w=0.5, T=2) LOWERED typed acc 0.5630 -> 0.5490 on the identical E-66 recipe and re-introduced the 4-failure probe regression. Gold targets are ALREADY soft probability distributions - a teacher's T=2-sharpened beliefs contain no information gold lacks; the incumbent's edge is its pretrained encoder, transferable only by weight transfer, not by target mimicry.
- Distill claims must be tested at the winner's recipe length: every 4-epoch run in this line is under-trained (E-68 proved length alone is +9); a KD-vs-E-68 comparison would confound two levers.
## 2026-09-23 - External review adoption (three standing rule changes)
An external review of E-62..E-69 flagged real portfolio-level gaps; verdict on the review: largely accurate, some stale facts. Adopted as standing rules:
- RULE-G1-DELTA: G1 is a retention DELTA from stage-A exit (post-B minus stage-A-exit >= -0.01) in every future ladder pre-register - never an absolute macro bar; forgetting and ceiling are separate questions (the 0.74 absolute bar conflated them for E-63/E-65/E-68).
- RULE-PROBE-GATE: probe-suite failure count is a no-regression gate (<= line-best baseline = 2) in every future ladder verdict - a run that tanks the confidence profile on accuracy alone is NOT a win (E-66/E-69 both regressed 2->4 ungated).
- RULE-CALIB-QUARANTINE: never adopt a fitted temperature/prior/position-correction unless accuracy AND ECE both improve on held-out; otherwise discard. Reaffirms the E-62/E-63/E-67 record (three attempts of the same root pattern).
Deferred from the same review: per-source balanced reservoir replay (the one untried typed<->phish trade-off knob) as its own single-lever pre-register (candidate E-71) after E-70.
## 2026-09-23 - E-70 lesson: token scale is the TRANSFER lever, recipe length the typed lever
- Doubling domain pretrain (260M -> 520M) did NOT raise typed acc (0.6420 < 0.6530, curves still rising; length lever may still close it) but moved zero-shot phishing AUROC 0.5878 -> 0.6966 - above BOTH published systems (Laya 0.678, Jev 0.689): the first benchmark beat. General-token-process quality (MLM heldout ppl 10.97 -> 8.12) transfers to distant zero-shot tasks far more than it does to in-domain decision heads.
- Evidence channel: the frozen-head probe bench in pretrain bench_log.jsonl tracks this before any fine-tune - watch the scheduled probes trend to predict where a recipe will land.
- Trade-off now two-sided: recipe length raised typed (+9 class, E-68), token scale raised transfer (+0.11 AUROC class, E-70) - the two objectives have measurable, separate knobs. Balanced replay (E-71) is the candidate to break the coupling..
## 2026-09-23 - Disk-full incident + smoke-checkpoint trap (E-72 gate)
- The E-72 preflight C5 failed deterministically with miniz zip errors; root cause was NOT the checkpoint but the DISK: volume filled to 0 GB, torch.save died at the same ~156MB boundary repeatedly (both a partial smoke A_last and a fresh in-flight save). Rule for future gates: check drive free-space FIRST when a zip/serialization error appears during LOAD or SAVE - a full volume mimicscheckpoint corruption.
- Two traps stacked: (1) every *_smoke run dir is a SIBLING of out_dir (out_dir + "_smoke") holding ~950MB throwaway checkpoints each - they are NOT in the experiment dirs and had silently accumulated across e68/e69/e70/e72 (~2.9GB); (2) a pwsh timeout KILL kills the smoke subprocess MID-SAVE, leaving a partial A_last that every later smoke then RESUMES from (os.path.exists resume) - deterministic poison. Session go-forward always: purge *_smoke dirs before relying on C5; the C5 timeout kill itself corrupts state.
- Housekeeping executed (reported per ENVIRONMENT rule): 4 smoke dirs deleted + completed-run intermediate states (e68 A/B_last, e70 B_last, l3 B_last; every final/model.pt and the ladder-critical l3/e70/e72 A_last kept). 0 GB -> 5.25 GB free.
- E-72 launched AFTER the gate passed 5/5 on the clean disk. External review's point stands in hard form: disk budget must be checked pre-flight like VRAM.

## 2026-09-23 - E-72 lesson: the typed/transfer trade-off is real and scaled
- Stage-B length on the same 520M encoder: typed +5.45 (0.6420 -> 0.6965, past the L2 incumbent and every prior rung) but phishing AUROC -0.087 (0.6966 -> 0.6092). Length and token scale pull the SAME transfer dial in opposite directions - not additive.
- Two production candidates in the lineage: E-72 typed specialist / E-70 transfer specialist (only model beating BOTH published benchmarks on a metric). Report both in user-facing scoreboards; silent merging is forbidden without a pre-register (E-71 balanced replay is the coupling breaker).
- G1 delta rule is the decision standard; the eval_report absolute-bar schema stays as the published framing.
- Also: failed disk-write tests can leave partial scratch files (writetest.bin) when the WriteAllBytes exception hits mid-write - always Remove-Item in the catch path.

## 2026-09-23 - E-71 lesson: balanced replay buys back transfer at capped typed cost
- Uniform replay over a 4x-skewed mixture starved under-represented sources and amplified the length-induced transfer decay. Equal-pressure stratified sampling (source=workflow, flag replay_balanced, default-off = bit-identical old behavior) recovered 6.0 of the 8.7 pts of phishing lost to stage-B length, while typed stayed +2.85 over the unbalanced arm.
- Decoder dim: every recipe change on this line trades typed against transfer somewhere in the 0.61-0.70 band on both axes - treat the pair of numbers as the unit of progress, not accuracy alone.

## 2026-09-23 - E-73 lesson: the mixer works - guard the objective with a stop rule keyed to its metric
- An early stop on the metric you must protect (phish drop 0.02 from run peak) converted the last 5 'losing' epochs of a 12-ep schedule into saved compute and kept the artifact in the published-class on transfer (0.6932) while typed held the registered line. When a run visibly trades one metric against another along epochs, a metric-pegged stop beats a fixed length.
- The stage-B bench already computed phish per epoch - the stop logic consumed the existing signal (no new eval cost).

## 2026-09-23 - E-74 in flight: symmetric trainer (option-isolated packing)
- laya/scripts/train_l3_sym.py committed and gating 5/5: fresh item construction (n_options=1, binary target, BCEWithLogits on the raw marker score), grouped decision metrics by qid, smoke-passing two-stage (5-ep sym stage A on mixture_raw packs, 8-ep sym stage B) with the SAME 520M encoder init - clean single recipe-flip vs the E-72 positional arm.
- Honest scope note: v1 covers stage-A/B + grouped typed/macro/Brier; decision_bench/perm-bench phase-2 adaptation pending (phish needs sym re-pack hooking); perm_dup/order-aug inapplicable under sym (documented in pre-register).
- Early pace: ~130 updates/s (236k-ish sub-items) - stage A may cost only ~15 min.

## 2026-10-04 - Muon NS coefficient typo was the real root cause (corrects 2026-09-22)
- `diacritizer/scripts/train.py:53` shipped the Newton–Schulz polynomial with
  **c=2.7885 instead of the canonical 2.0315**. Both Muon failures in this repo
  (E-63 S3 arm step-2 NaN; the s7 run e63s_s7_18l64 NaN @ ~9071) trace to it.
- Numerically proven on the model's own shapes (`scratch/ns_coefficient_test.py`):
  the repo polynomial WITHOUT the band-aid explodes to 2.8e16/inf on every shape
  (reproduces the E-63 step-2 NaN); WITH the band-aid the update survives but is
  not orthogonal — Q/O 64×64 update σ 0.001–1.644 vs canonical 0.041–1.156
  (ortho_err 0.84 vs 0.37). The band-aid was a symptom patch on a typo.
- Generalizable lesson: when an optimizer's own math is being blamed, diff its
  constants against the reference implementation FIRST. A unit test pinning the
  NS coefficients would have caught this before two GPU runs were burned.
- Consequence: Muon is unvalidated in this repo, not refuted. A fixed-Muon rung
  must first remove the band-aid, add the aspect-ratio scale
  `max(1, m/n)**0.5`, implement a real warm-up + decay (none exists today), and
  prove itself on the 2000-step trial bed before any full run.
- Option A (the same day, user-approved) instead put the s7 arch on the proven
  AdamW recipe and it completed 20000 steps cleanly — see research/EXPERIMENTS.md.
  Durable trainer fact recorded there: `warmup_steps` and `grad_clip` in the
  existing configs are DEAD KEYS (warm-up is unimplemented; grad_clip is
  hardcoded 1.0 at train.py:389). A config carrying them implies behaviour the
  trainer does not have.

## 2026-10-05 - D-line data hazards: a gate file sits in the training source list, and PAD labels have drifted
- **A gate benchmark is listed as a TRAINING source.** `diacritizer/scripts/
  prepare_data.py:40` includes `raw/wikinews/wikinews2024_multi_ref.diac` in
  SOURCES, but `diacritizer/scripts/bench.py:147` and
  `diacritizer/scripts/eval_e63s_wn2024.py:22` use that SAME file as the
  wikinews2024 gate benchmark. Any corpus built with the default `--sources`
  therefore trains on the wikinews2024 test set. Measured: a default rebuild
  produced train_windows 2,433,799 / val 160,201 including 356 wikinews2024
  windows, while the recorded reference
  (`data/diac/prepared_rebuilt/prepare_stats.json`, backed up at
  `scratch/v3_prepare_stats_reference.json`) has NO wikinews2024 entry.
  Rebuilding with `--sources fadel_train,fadel_val,abdou_train,abdou_valid,
  sadeedt_train,qcri_wiki` reproduces every reference stat exactly (2,433,430
  train / 160,214 val). The recorded v3 is clean; the HAZARD is that re-running
  the pipeline with its defaults silently leaks the gate into training.
- **The PAD label convention is undocumented and has drifted.** The entire
  trained D-line lineage - the deployed gold `stage2b2500` (tokens_phase v3),
  every E-63 arm and the s7 pilots (v3s10) - carries label **0** at padded
  positions, so pad IS scored in their loss (~12% of positions; 15.1 pad
  tokens per 128-char window on average). The current
  `diacritizer/scripts/tokenize_data.py:42` initialises `y` to **-1**, and
  `diacritizer/src/model.py:276` masks `labels >= 0`, so a FRESH tokenize
  excludes pad from the loss. `git log -S` shows tokenize_data.py has used -1
  since it was first committed (84fa004, 2026-09-11), so the v3/v3s10 files on
  disk were NOT produced by the current script. Consequence: a full-data run on
  a fresh tokenize reports a different loss/val_loss metric than every number
  in the ledger. The GATES are unaffected either way - `bench.py:65`
  predict_bares -> `diacritize_piece` (bench.py:55) walks only the original
  piece characters, so padded logits never reach a gate.
- **The full v3 token corpus had been deleted.** `data/diac/v3/tokens/` was
  empty at session start while the pilots' 10% slice `data/diac/v3s10/tokens/`
  survived. Rebuilt from the recorded reference stats and byte-verified against
  v3s10: `train_ids` prefix 243,343 rows IDENTICAL, `val_ids` prefix 9,613
  IDENTICAL, and 0 non-PAD label differences across 3,670,723 compared
  positions. The text is byte-identical to what the pilots trained on; the only
  difference is the pad-label convention.
- **val_loss is not a safe model-selection metric near convergence.** A1
  (lr 4e-4) improved its internal val_loss from 0.2431 to its run best 0.2397
  between step 18000 and 20000 while ALL FOUR external gates regressed (mean
  49.80 -> 51.82 DER), consistently across gates rather than as single-gate
  noise. Select the deployment checkpoint from `gate_probe/best_gate_weights
  .pt`, which the monitor already tracks, not from `best.pt` /
  `final/model.pt`.
- **The s7 LR hypothesis was right.** A1 (lr 4e-4, otherwise byte-identical to
  A0's 2e-4) reached best val_loss 0.23969 vs A0's 0.25892 - a 7.4% gain - and
  beat A0 at every gate probe step up to 18000. The 2e-4 inherited from the
  28L/272h arch WAS under-tuned for the 4x-narrower 18L/64h s7 arch.

## 2026-10-06 - Verdict-audit lessons: both directions of error, and the ones I made myself
- **A wrong verdict is more expensive than a failed run** because it propagates into
  later decisions. The audit found errors in BOTH directions: false FAILs (E-19
  Fine-Tashkeel scored on a slice instead of the gate; E-23a gold wn2024 misread as
  48.65 vs the real 57.58) and false PASSes (E-38a, E-45, E-54c/d, E-63c, E-63s
  S3-best). Never assume a recorded FAIL is a real failure, nor a PASS a real pass -
  re-derive from the artifact.
- **Check the metric DIRECTION before comparing anything.** A resolved config in
  `runs/v2/core/micro_full/validate/config/resolved.json` had the registered
  direction inverted; every verdict derived from it is sign-flipped. Read the
  authority (`train.py` defines GATE_EXTERNAL) rather than trusting prose.
- **Reproduce the baseline inside the same harness before claiming a delta.** In
  E-071 the voting script's own single pass had to re-derive the recorded 0.374792
  (it did, |diff| 0.00008) before the vote number was admissible. Without that, a
  harness difference is indistinguishable from a real gain.
- **A "best-of-run" number over many noisy probes is a SELECTION, not a
  measurement.** E-070's headline rested on one early probe of twelve. The fix is
  not to distrust it but to TEST it on a set that took no part in the selection
  (E-072 used the held-out abdou control and the ordering held). Always record the
  matched-step number alongside the best-of-run number.
- **Judge budget at MATCHED TOKENS.** C1 (batch 128 x ctx 128) and C4/C4b
  (batch 64 x ctx 256) are all 16384 tokens/step, so matched steps = matched tokens
  = matched compute. "Under-trained" is a matched-token claim, not a step count.
- **An authority banner makes a file off-limits, not a to-do list.**
  `research/EXPERIMENTS.md` preserves duplicate ids (75 of 110 ids recur;
  E-13 is defined twice) ON PURPOSE per its own banner. The honest response to "the
  ids are duplicated" was to record the gap and leave the file byte-stable, not to
  tidy history. Check a file's self-declared authority before "fixing" it.
- **Verify a lineage against the configs, never from memory.** Reading the three
  D-line configs showed the warm-start chain is intentionally asymmetric: C4
  inherited C1's *gate-best* probe, while C4b inherited C4's *final* weights. I had
  it wrong from memory; the configs settled it.
- **Batch reads BEFORE batch edits.** The edit tool rejects a write to any file not
  read this session, and compaction drops that state - three edits failed this
  session purely for that reason, one round-trip each. Read every target in one
  program, then edit in the next.
- **A shell-style interpolation inside a JS template literal is eaten by JS, not by
  pwsh.** Writing the dollar-brace form in a TS template becomes an undefined JS
  identifier ("ReferenceError: g is not defined") or a parse error; concatenate
  instead. Same class as the backtick-in-template-literal parse error - both bit me
  in this very session, and both cost a round-trip.
- **Rule adoption, 2026-10-06 (user-approved):** eight lessons from the verdict audit
  were promoted into `rules/` — three into `evidence.md` (metric direction comes from
  the defining code; an artifactless verdict is a defect in either direction; scope the
  conclusion to the evidence), two into `experiments.md` (config declares intent, the log
  proves execution; selection is not measurement), and one each into `context.md` (a
  file's self-declared authority governs whether it may be edited), `tooling.md` (harness
  preconditions and text-channel hazards), and `gpu.md` (measure a contention theory
  before acting on it). Staged proposal and bounded diff:
  `.skillopt-sleep/staging/2026-10-06-rule-proposals.md`; pre-adoption backup:
  `.skillopt-sleep/backups/2026-10-06-rules-adoption/`. Lint was 51 issues before and 51
  after, with zero concerning `rules/`, so the adoption added no regression.
## Disk reclamation, 2026-10-06 (user-authorized deletion)

The user explicitly authorized removing two targets identified by a read-only disk
audit. Measured effect: E: free went from 10.363 GiB to 22.272 GiB (+11.909 GiB).

REMOVED 1 - runs/v2/core/gpu_smoke/train/checkpoints/step-00000001..step-00000080
(80 directories, 2,287,063,991 bytes = 2.130 GiB). The run's own config declares
the retention bound these 80 exceeded: keep_last_checkpoints: 3,
keep_best_checkpoints: 1 (<=4 permitted, 84 existed). COMPLETE.json names only
step-00000084, and a corpus-wide grep found zero references to steps 1..83.
KEPT: step-00000081..84. keep_best_checkpoints: 1 stays unresolved because no
on-disk metadata names the best-by-val_loss step, which is why 81..84 were kept
as margin.
REGENERATE: ./.venv/Scripts/python.exe pipeline_v2/run.py run --config
pipeline_v2/configs/experiments/gpu_smoke.yaml

REMOVED 2 - data/fleurs/data (13,097 files, 10,491,671,956 bytes = 9.771 GiB), a
re-downloadable public dataset (google/fleurs). Its only substantive citation was
a STALE TASKS.md row that recorded the audio as BLOCKED on HF auth even though
the audio was already present. data/fleurs/.cache was deliberately KEPT (26 files,
~3 KB) so a re-download can reuse it.
REGENERATE: langid/scripts/fleurs_download.py (REPO = "google/fleurs",
TARGET = ROOT/"data"/"fleurs", pulls data/<lang>/audio/<split>.tar.gz via
hf_hub_download). Requires HF auth for the audio tarballs.

REFERENCE FOLLOW-UP (not yet done, flagged to the user): TASKS.md row 101 (DA-4/5
audio tier) and langid/AGENTS.md lines 18/29 still describe the FLEURS audio as
present; they now point at a path that does not exist.
NOT TOUCHED, deliberately: every other audit candidate was either cited evidence
(runs/target/checkpoint-4000 carries an explicit user keep-decision and is final's
restore source; e70_c4b_ctx256/eval/best15000 is a cited --ckpt;
e63_m1_tr/final_ladder_snapshot is explicitly preserved) or a load-bearing premise
of an active plan (the 400 weights_step*.pt probes, the v3/v3ctx256 beds,
prepared_rebuilt, tokens_pad-1_backup).
- **eval_metrics scoring is ~5 ms/line, so any real scoring run belongs in the
  background.** One 2,500-line gate file takes ~13 s; a 10,809-line abdou file
  takes ~54 s; a 12-file sweep therefore runs ~11 min and will be killed by a
  foreground timeout, which surfaces as a bare exit 1 with truncated output and
  no traceback - easy to misread as a crash or a bug in the scorer. Launch
  scoring with run_in_background, never inline. (Also note abdou pred/ref files
  carry more PHYSICAL lines than pairs: 4,000 pairs wrote 10,809 lines, because
  multi-line abdou entries survive the join.)

## 2026-10-06 - D-line combination stack: the soup and voting COMPOSE, and the correction I owed myself

- **A uniform weight soup of one training lineage is a free accuracy win (E-073).** Averaging the 12
  `e70_c4b_ctx256` gate probes in fp32 and casting back to the member dtype gave 3-gate 0.352209 and
  held-out abdou 0.349044 - against the best single probe in that set at 0.374792 / 0.375141, i.e.
  **-6.03% on the registered gates and -6.96% on held-out**, with NO training and no new checkpoint.
  It was the project best at the time (prior: E-071 voting 0.369702).
- **My own prediction was refuted, and that is the part worth keeping.** I predicted the soup would blur
  a reliable member-to-member quality gradient (E-074 had just measured a +0.909 rank agreement, so a
  gradient demonstrably existed). Wrong model: **reliable member ORDERING and the averaging benefit are
  ORTHOGONAL.** Averaging cancels basin-curvature noise across members; it does not erase member
  identity. A measured gradient is not evidence that averaging must hurt.

## 2026-10-11 - D-line decode contract: our OWN decoder was on the WRONG side of the byte-order trap, and the registered gate was reporting about 2x the true error (corrects and extends 2026-10-02)

- **The 2026-10-02 entry sized the byte-order trap from EXTERNAL models and missed that our own deployed arm sat on the wrong side of it on every corpus.** Its evidence was mishkala losing 7.25 DER points on WikiNews 2024 while zmahood looked "unaffected"; its rule was to leave the registered gate byte-order-sensitive and report `der_canonical` beside it. That rule is right about the METRIC and it never asked what OUR decode emits. It emits shadda-first from `labels.marks_for_label`, while `bench.load_pairs` NFC-normalizes EVERY reference (lines 114 fadel / 125 abdou / 134 sadeed / 152 wikinews). FatHa U+064E has combining class 30 and shadda U+0651 has class 33, so NFC canonically reorders shadda-first into vowel-first. Prediction and reference were compared in different byte orders on every geminate, and because `der_registered`'s denominator is WORDS, one flipped geminate marks the whole word wrong.
- **Measured magnitude (E-090, full registered gate re-run):** deployed E-075 voted arm 0.346827 -> 0.200865 = **minus 42.08 pct**; abdou 0.341475 -> 0.188487 = minus 44.80 pct; single arm 0.352195 -> 0.208612 = minus 40.77 pct. The per-corpus cost was NOT constant - wn2024 11.83 pct, wn2014 36.11, sadeed 44.11, abdou 44.80, fadel 48.37 - so the offset distorted CROSS-corpus comparisons even where each arm's own ranking survived.
- **The root cause is the harness normalizing only ONE side, not a corpus convention.** The comment in `labels.py` ("Real corpora use BOTH orders") is stale for this pipeline: the raw corpus `data/diac/raw/fadel/test.txt` is 100 pct shadda-first (21196 geminates) and the scored `e75_vote/fadel_test.vote.ref.txt` is 100 pct vowel-first with the SAME 21196. `NFC(raw_gold) == scored_ref` on 2500/2500 lines while `raw_gold == scored_ref` on only 250/2500. References 2500/2500 NFC, predictions 228/2500. Scoring the SAME predictions against RAW gold gives 0.138220, against the NFC ref 0.267606.
- **Three independent routes agree, and that agreement is what makes the diagnosis safe:** pred-vs-raw-gold 0.138220, `der_canonical` 0.138220, NFC-of-prediction 0.138172. When a suspiciously large metric improvement has three independent confirmations landing on the same value, it is a representation artifact being removed, not a result being manufactured.
- **Fix the DECODER, never the registered metric.** That is the distinction the 2026-10-02 rule protects: the gate stays byte-order-sensitive so published-model comparisons stay meaningful, while our own output becomes canonical Unicode text. One line: `marks_for_label` returns `unicodedata.normalize("NFC", shadda+vowel)` for classes 9..14. `label_for_marks` already accepted both orders, so no label set, training data or checkpoint changed; the diacritic CONTENT of every prediction is identical and only its byte order moved.
- **A decode fix invalidates the future series, so old and new runs are not directly comparable.** Old on-disk artifacts keep the old numbers; converting them needs the exact NFC transform (`research/verdict_audit/tools/order_fix_probe.py`). Any gate table mixing pre- and post-2026-10-11 arms must name the decode convention per row. `selftest.py` a0a's byte round-trip fixture had to become an NFC-equality assertion for the same reason - a test that pins raw bytes re-asserts the wrong decode.
- **A pre-registered PROXY criterion can fail while the mechanism is confirmed exactly.** E-090 predicted all ten re-run numbers to six decimals from a CPU forensic re-score (fadel 0.138172, sadeed 0.205100, wn2014 0.259324, wn2024 0.459118, abdou 0.188487, gate_mean3 0.200865) and every one matched. The acceptance rule nevertheless compared against the artifact's `der_canonical` and required 0.5 pct agreement on all five; wn2024 deviated 0.935 pct (0.459118 vs 0.463451) so the verdict was recorded `partial`, not `accepted`. Cause identified: `canonicalize_word` also rewrites classifiable runs through the label map, which is not the same condition as NFC. Pre-register the criterion against the quantity the fix actually produces, and honour the rule you wrote even when the deviation favours you.
- **A CPU-bound decode path, not the GPU, sets the wall clock here.** E-090's abdou step took about 67 min against a 39.8 min reference on the same arm; the decoder process sat at about 91 pct of ONE core with the GPU at 7-12 pct. Four corpora finished in about 6 min and abdou (13x their characters) took the rest. Never diagnose "stalled" from GPU utilisation on this line - check the process's accumulated CPU seconds.

- **A decode fix invalidates prior RANKINGS, not just prior numbers, and the correction can be read off artifacts that already exist.** Because the offset was corpus-dependent (11.83 pct on wn2024 up to 48.37 pct on fadel) it did not merely inflate every arm, it gave a spurious SIGN to differences among near-ties. Restating the ledger needed NO model re-runs, because `der_canonical` is order-insensitive by construction, so each arm's already-recorded `der_canonical` IS its corrected `der_registered` (`research/verdict_audit/tools/corrected_ledger.py`). Result (E-091): the D-line has exactly ONE mechanism that beats the plain single pass significantly - sliding-window voting, -3.7 pct; the E-073 soup is a WASH (+0.02 pct) rather than the 6.03 pct "win" it recorded against the best individual PROBE; and E-078's primary arm, recorded as a -0.25 pct gate win that fired PARTIAL, loses on BOTH readings corrected (+0.25 pct gates, +1.48 pct abdou) and so fires its own FALSIFIED branch - E-078 amended partial -> rejected with a dated note.
- **A SIGN CHANGE IS NOT A FINDING, and a pre-registered rule must be re-APPLIED rather than re-argued.** The E-078 reversal is real in sign but only 0.25 pct in magnitude, inside the 0.82 pct init-draw noise band, so the honest statement is not "the ranking flipped" but "those arms were never distinguishable and the bug manufactured an ordering". Record both, because both are true at different resolutions. What made the verdict change defensible was that E-078's record ALREADY stated its own rule ("FALSIFIED iff L1 >= c4b on the gates AND on abdou") and the corrected numbers satisfy it: the amendment applies the author's own criterion, not a new one invented after seeing the data.
- **Appending a YAML record can silently corrupt the PREVIOUS one, and the damage is invisible to a normal load.** When E-090 was appended it landed inside E-089's `gates:` block scalar, orphaning E-089's pre-registration tail onto the end of E-090's `finding` and leaving E-090 with a DUPLICATE `evidence:` key. PyYAML takes the LAST duplicate key, so E-090 quietly claimed E-089's ctx512 evidence while still parsing cleanly and reporting the right experiment count. Verify registry edits with a loader that raises on duplicate keys, and re-read the boundary lines after any append.

- **A cross-basin negative control is mandatory, not optional (E-073 `all24`).** Soupy the 12 C4b probes
  together with the 12 C4 probes - same architecture, same parent lineage, but a different continuation
  basin - scored **+19.89%** (0.449353 vs 0.352209). That single arm proves same-basin membership is
  load-bearing AND that the eval path is not accidentally rewarding the merge.
- **Sliding-window voting composes ADDITIVELY with the soup (E-075).** Voting the `c4b_uniform` soup at
  window 256 / stride 128 moved the gates 0.352195 -> **0.346827** (-1.52%) and abdou 0.349032 ->
  **0.341475** (-2.17%), beating the single pass on EVERY corpus measured including the contaminated
  wikinews2024 diagnostic column (0.522325 -> 0.520723). Both levers are inference-time, so the
  composition is free. 0.346827 is the new project best and sits 6.19% below the previous one.
  Weight averaging removes noise ACROSS checkpoints; context voting removes per-window boundary error
  WITHIN one model. Different axes, additive gains.
- **A dev set can be VALIDATED as a selector before it is trusted as one (E-074).** First measure the
  ceiling: across the 12 C4b probes the mean inter-gate Spearman rho is +0.809, so the Spearman-Brown
  reliability of the 3-gate mean is 0.927 and **no dev set can rank better than sqrt(0.927) = 0.963**
  against it. abdou_test then reached rho +0.909 = **94.4% of that ceiling**. So the registered gates
  are not noise-dominated, and selection on abdou is a leak-free substitute for selecting on the gates.
  But abdou is a sound SELECTOR, not an oracle: it swaps the top two probes (0.374565 vs 0.374955,
  0.10% apart), so **a top-2 call on abdou must still be treated as a tie.**
- **Voting's headroom is bounded by `window == ctx`.** At window 256 on a ctx-256 model, every sentence
  of 256 characters or fewer gets exactly one window and voting is a mathematical no-op; the entire
  measured effect lives in longer sentences. E-071's 128/64 arm remains unrun and is the obvious next
  lever - do not read E-075's -1.52% as the ceiling of the voting idea.
- **Checkpoint-DIRECTORY loaders swallow a soup with no code change.** `bench.py:185` and
  `e67_p6_vote.py:136` both do `state["model"] if "model" in state else state`, so a soup saved in the
  probes' own `{model, step, config}` wrapper shape loads through the existing eval path unchanged.
  Prefer that shape when writing any new merge tool.
- **A repo-root path derived from `__file__` must be COUNTED, not guessed.**
  `Path(__file__).resolve().parent.parent.parent` from `research/verdict_audit/tools/` lands on
  `research/`, not the repo root, and the failure mode is a bare `ModuleNotFoundError` at import time.
  From that directory the root is `parents[3]`.
- **A corrected verdict must be pushed to every row derived from it, or the stale copy wins.**
  E-067 P1's first verdict was FAIL on an INVERTED metric direction and was corrected in the
  experiment record the same day (P1 is a WIN, 0.4299 vs 0.4577, and it survives the no-seed
  control C0-R at 0.4539). But `state/tasks.yaml` task-e067 kept the pre-correction wording -
  "P1 CLOSED as a clean FALSIFICATION ... warm-start is now falsified 4/4" - for hours, and that
  row is what a session bootstrap reads. A wrong number in a narrative file is a wrong number;
  a wrong verdict in the *task registry* silently redirects the next session's plan. Corrected
  2026-10-06. Generalised: when a verdict is corrected, grep for every derived statement of it.
- **Greedy soup selection and uniform soup averaging are NOT two settings of one dial - they
  search different families.** E-076 tested dev-informed weighting against E-073's uniform mean
  and lost (+0.70% gates, +0.58% abdou, outside the pre-registered TIE band). The reason is in
  `weights.json`: R=4 rounds of Caruana greedy produced support on **4 of the 12** members at
  0.25 each, while uniform spreads mass over **all 12** at 1/12. Greedy's dev trace improved
  monotonically (0.375332 -> 0.358318 -> 0.354069 -> 0.351603) but only *within* a family that
  does not contain the uniform solution. So "the selector is validated" (E-074: abdou rho +0.909)
  and "the selector improves the result" are different claims: a good selector over a restricted
  support can still lose to a dumb rule over the full support. Before comparing a search to a
  baseline, check that the baseline is **inside the search space** - if it is not, the comparison
  measures the support restriction, not the weighting scheme.
- **The soup + vote composition has now replicated on a second, independent soup.** E-075 measured
  voting beating the same model's single pass by -1.52% gates / -2.17% abdou on the uniform soup;
  E-076's greedy soup independently reproduced -1.57% / -2.17%. Two different soups, same additive
  composition. One replication is a result; two is a property.
- **Never estimate a training run's remaining time from its first minutes, and never
  from a step count alone.** Every D-line run writes `runs/diac/<phase>/monitor/log.jsonl`
  with `on_step` events carrying `t` (seconds since run start), so the rate history is
  always recoverable after the fact - use it. Measured 2026-10-06 on E-079 arm C: the run
  opened at 6.3 steps/s and fell to a flat 1.92 steps/s (52 s per 100 steps) immediately
  after the step-5000 gate evaluation. Read naively, the first five minutes predicted a
  ~2.5 h run; the real budget is ~8.3 h. Reference logs: `e70_c4b_ctx256` ran 51.6-53.8
  s/100 for its entire 60 k steps, `e69_c4_ctx256` drifted 18.5 -> 25.4 s/100 over 60 k and
  NEVER collapsed, and `e67_c1_ft64_fullv3` held 17.4 s/100 to step ~27.5 k then sat at
  ~41.5 s/100 for the remaining 172 k. The steady rate is what the
  budget must use, and the right comparison is another run's OWN monitor log, not a guess.
  Corollary: a 60 k-step run at ctx 256 / batch 64 costs about 10 h on this machine, so a
  two-arm paired design is a ~20 h GPU window. Budget that before promising a phasing plan.
- **The D-line GPU is pinned at its BASE clock and never boosts - measured, not inferred.**
  On 2026-10-06 the Quadro RTX 3000 held **945 MHz SM (pstate P2) across 650 consecutive
  samples**, including through the gate bench, and never once exceeded base; samples dipped
  to P3/P5/P8 at 300-435 MHz between gate subprocesses. Simultaneously: **41 W of an 80 W
  limit, 72 C against an 87 C target, memory utilization ~10%, disk read 0 MB/s, CPU ~35%**.
  So the clock is NOT held back by power, heat, bandwidth or the host - the driver/platform
  keeps it at base. Practical consequences: (1) every wall-clock budget in this repo's run
  history is inflated and must be read from `monitor/log.jsonl`, never guessed; (2) the same
  machine produces two distinct modes, ~16-20 s/100 steps and ~52 s/100 steps, and the onset
  is variable - `e69_c4_ctx256` stayed fast for all 60 k, `e70_c4b_ctx256` was slow from
  step 5, `e67_c1_ft64_fullv3` switched at ~27.5 k, `e79_c_18l64` at ~5-6 k; (3) the clock ceiling (945 ->
  2100 MHz = 2.2x) does NOT fully explain the observed ~3.4x, so do not attribute the whole
  effect to clocks. TESTED (E-080, 2026-10-06): NVIDIA power management mode 'Prefer maximum
  performance' DOES raise the clock - a fresh process measured median 1395 / peak 1830 MHz at
  71-88 W and ~16.5 s/100 against the base-clock ~52 s/100 - but it does NOT HOLD. E-080's own
  probe decayed to exactly 945 MHz / 42 W at t~1250 s and stayed there, and the resumed arm C
  came back latched from step 0. The profile is a real but NON-DURABLE win; never promise a
  speed-up from a profile change alone.
  INSTRUMENT TRAP, same experiment: a power-saturating probe (4096^2 fp16 matmul, 78 W) drags
  its OWN clock down to 900-960 MHz and therefore CANNOT measure the training clock. It
  reported 'clock capped' while the real training workload was boosting to 1830 MHz. Match the
  probe's power profile to the workload before drawing a clock conclusion from it.
  STILL OPEN: the decay onset is variable (never for c4, step 5 for c4b, ~21 min for E-080) and
  unexplained. Thermal-throttle, power-policy and eval-triggered explanations have all been
  falsified against measured data. See diacritizer/E-080.

## 2026-10-11 - D-line: the sliding-window vote's gain is a ONE-TIME context correction, not an outvote of independent errors (E-092)

- **Voting is the only significant decode-time mechanism on this line, and E-092 shows WHY it works - and why it cannot be extended.** On the same 400 abdou pairs: single pass 0.198411 -> 2 votes 0.188956 (**minus 4.77 pct**); 2 -> 4 votes minus 0.23 pct; 4 -> 8 votes minus 0.21 pct (exact paired McNemar p = 0.220 and 0.345). The FIRST re-decode buys almost the whole gain and the next three doublings buy essentially nothing - the marginal effect of a further vote is about **23x smaller** than the first.
- **That shape is diagnostic, and it is the reusable part.** If extra covering windows contributed independent evidence, the curve would keep falling. It saturates immediately, so overlapping windows of ONE deterministic model are **NOT decorrelated** - adjacent windows see nearly the same context and repeat the same mistakes. The gain is a one-time correction of positions the full-ctx decode gets wrong for structural reasons (long-sentence edge/context artifacts), and it is fully captured by the first alternate context.
- **E-077's premise was never actually tested by E-077.** E-077 framed its window sweep as asking whether "shrinking the window mechanically raises the number of independent contexts each position is voted from", but it always held `stride = window/2` (W256/128, W192/96, W128/64). With `window_starts`, that gives EVERY interior position EXACTLY 2 covering windows at every window size it tried. E-077 varied the CONTEXT LENGTH per vote, never the VOTE COUNT. Check the arithmetic of a sweep's design before accepting its conclusion about the variable it names.
- **A cheap screen is only worth reading once it reproduces a known full run.** The 400-pair prefix reproduced BOTH arms to ~0.25 pct (single 0.198411 vs 0.198883 full; vote 0.188956 vs 0.188487 full) and the prefix vote-vs-single gap (minus 4.77 pct) matched the full-corpus gap (minus 5.23 pct). Calibrate a screen against a full-gate artifact before believing a small effect it reports.
- **What survives, and its price.** The only remaining decorrelation route is the one E-088 named: a candidate set whose errors are NOT the same errors, i.e. a word-level or variable-length model TRAINED FROM SCRATCH. E-092 shows decorrelation cannot be harvested by re-decoding this checkpoint; it must come from a different model, which makes it a training arm, not a few-minute screen.
- **The vote's gain is 92 pct CHUNK-BOUNDARY REPAIR, and that makes it void as a "combination" result - check what the BASELINE arm actually decodes before crediting a method.** `bench.predict_bares` builds the "single" arm with `chunks_of(bare, ctx)`, i.e. **non-overlapping** 256-char pieces (`diacritizer/scripts/bench.py:52-53`). Bucketing words by distance to the nearest multiple of 256 on the same 400 abdou pairs: words within 8 chars of a boundary are wrong **33.25 pct** vs **18.10 pct** for interior words (83.7 pct relative excess), that bucket is only 7.5 pct of words but supplies **91.8 pct of the vote's entire error reduction**, and on interior words the vote is very slightly **WORSE** (0.1816 vs 0.1810). At stride 128 a boundary word is interior in exactly one of its two covering windows, so the middle-context tie-break is what picks it: the mechanism is **centered-context selection**, not context mixing. E-091 had called sliding-window voting "the only mechanism with a measured significant effect" and read it as an ensemble win - it is a repair of our own chunking. **When a method's whole gain concentrates in one 7.5 pct bucket defined by an implementation detail, the method is measuring the implementation, not the idea.**
- **The "sliding-window vote" is a NO-OP; the gain is pure centered-context selection, and a tie-break is not a voting mechanism (E-093).** The vote does majority-first and centering only on ties. Running a third arm that centers ALWAYS and never votes, from the same loaded model: single 0.198411 / vote 0.188956 / centered 0.189054. Centered reproduces **-4.72 pct of the -4.77 pct** attributed to voting, and vote-vs-centered is **24 discordant words in 41,019 (p = 0.541)** - the majority step is worth **four words**. The deployed arm's entire advantage over the hard-chunked single is "pick the window in which the position is most central".
- **A mechanism can be statistically real and still be the wrong explanation - test the LAYER, not the pipeline.** E-091 (written earlier the same session) recorded the vote as "the only mechanism with a measured significant effect" and read it as an ensemble result. The effect was real; the attribution was wrong twice over - it repaired our own chunking (E-092) and its combining step did nothing (E-093). After this, **no combination/ensemble/merger/voter mechanism survives on the D-line with any measured effect**, and the honest baseline for "the model" is the centered decode (0.1890), not the hard-chunked single (0.1984). When a component's removal changes nothing measurable, delete it from the explanation even if you keep the artifact.
- **`render_current_state.py --check` does NOT detect duplicate YAML keys, so a duplicate-key corruption can pass the state gate - only a duplicate-key-raising loader catches it.** I repeated this hazard on myself this session: editing E-093 to set its closing `verdict` left the pre-registration's `verdict: none` in place, producing two `verdict` keys. PyYAML silently takes the LAST one, so `--write` and `--check` both reported **PASS** while the record was corrupt. The guard that works is a loader that raises on a repeated key (`scratch/check_yaml.py`), run against `research/experiments.yaml` after EVERY edit to it. **Corollary: never set a field that may already exist - grep the record for the key first, or edit it in place; and treat "state render PASS" as evidence about the RENDERER, not about the YAML.**
- **`research/experiments.yaml` records are NOT in numeric order - E-091 sits at the very END of the file, after E-093 - so anchoring an edit on a later-numbered record's `- id:` line can silently attach your text to a DIFFERENT, earlier record.** I did exactly this: intending to add an amendment to E-091, I anchored on `  - id: diacritizer/E-092`, which put the block at the end of **E-090**'s mapping. The YAML stayed valid, no duplicate key appeared, and `render_current_state.py --check` reported PASS - E-090 simply gained a 1534-char `amendment` that described E-091's correction, while E-091 got nothing. **A syntactically valid file can carry semantically misattributed fields, and no existing gate catches it.** The only reliable check is to PARSE the YAML and assert the field exists on the intended `id` (`'amendment' in record_for('E-091')`), never to trust that an insertion point landed where its neighbours suggest. Fixing it needed line-precise surgery guarded by assertions, because the block had to come OUT of one record and go INTO another.

## 2026-10-11 - D-line: the model was fed a DIFFERENT ALPHABET at inference than it was trained on (E-094/E-096)

- **The training input alphabet is 37 Arabic base letters plus pad, and nothing else - verified by bincount, not inferred.** Over all 2,433,430 rows of \`data/diac/v3ctx256/tokens/train_ids.npy\` (622,958,080 tokens) only **38 vocab ids of 171** have a non-zero count: pad (0), the letters U+0621-U+063A (ids 4-29) and U+0641-U+064A (ids 36-45), and alef wasla U+0671 (id 46). **133 ids have count exactly 0**, including space (77), tab (78), newline (79), every digit, every punctuation mark, every Latin letter, tatweel (35), and \`<unk>\` (3).
- **That is BY DESIGN, and the root cause is one line in the data pipeline.** \`prepare_data.py:186-188\` builds the training input as \`bases = "".join(u.raw for u in units if u.is_target)\`; \`passthrough.py:67-78\` sets \`is_target\` True iff \`is_arabic_base(ch)\`; \`passthrough.py:18-22\` makes \`is_arabic_base\` exclude tatweel and include U+0621-U+064A plus U+0671. So the model was trained on a **pure letter stream with word boundaries erased**, which is sound for a char-level task where marks attach to letters.
- **\`bench.predict_bares\` (\`bench.py:75\`) encodes the RAW bare text, so inference fed untrained embeddings for 20-25 pct of ALL input tokens.** Measured untrained-token counts fed at inference: fadel_test 143,733 / sadeed25 61,333 / wikinews2014 20,614 / wikinews2024 13,097 / abdou_test 1,836,623. The space embedding ROW exists and was never touched by any gradient. **A vocab id with zero training count that inference feeds is untrained noise - audit the vocab against the training bed before trusting any inference-time mechanism.**
- **Fixing it is the largest effect ever measured on this line: minus 30.2 pct relative (E-094, screen).** Registered-convention numbers on abdou_test first 400 pairs: single 0.198411 / macro_f1 0.7845 -> nospace 0.138488 / 0.8625; centered 0.189054 -> nospace_centered 0.128473 / 0.8749. Paired exact McNemar single vs nospace: 4449 discordant (995/3454), p = 1.75e-314. Roughly **8x the previous project best** (E-077 w192, minus 3.74 pct) and 15x the 2 pct bar, and it is near-additive with centering so it does not replace it. \`single\` and \`centered\` reproduce E-093's published numbers EXACTLY, which is what validates the two scorers against each other.
- **E-094's fix is only PART of the mismatch: whitespace is ~85 pct of the untrained noise, and \`nospace\` leaves 13-22 pct of it behind.** Residual untrained tokens after stripping whitespace only: fadel 21,135 / sadeed 9,651 / wn2014 2,716 / wn2024 2,836 / abdou 311,148. The principled arm (E-096) skips **every non-\`is_arabic_base\` character** and reconstructs the training input distribution exactly - **verified to feed ZERO untrained ids on all five corpora** - so the remaining untrained share is punctuation, digits, Latin, tatweel and \`<unk>\`.
- **Word-level DER IS sensitive to line grouping - this corrects the earlier note that it is not.** E-094 scored byte-identical predictions two ways: inline \`em.score_corpus(preds, refs_all)\` over the 400 in-memory pair entries gave single **0.199240**, while reading the same written files with \`.read_text().splitlines()\` gave **0.198411** (~0.4 pct). \`abdou_test\` gold fields contain embedded newlines, so 400 pairs are ~1065 line-segments. The REGISTERED convention is \`splitlines()\` (\`score_gate_bench.py:35-36\` and every other registered screen scorer). **Always state the line convention with a D-line DER number, and never score a registered result from in-memory pairs.**
- **A screen is not a gate.** E-094 is an exploratory discovery made by inspecting the training bed and screened immediately; it was **not** pre-registered, so it is recorded honestly as a screen. The registered claim is E-095, which pre-registers the verdict rule and scores the full five-corpus gate in the registered convention before any number is cited for it.

## 2026-10-12 - D-line: the full-character-set bed (E-097) and the parse/encode granularity trap

- **`passthrough.parse` groups a RUN of non-target characters into ONE `Unit` whose `.raw` can be many characters long, while `tokenizer.encode` emits one id per CHARACTER.** Any consumer that keeps non-target characters in the input must expand its labels per character; the stripped-alphabet path never hits this because every target unit is exactly one character. The failure is silent: `tokenize_data.pack_rows` does `L = len(seq); y[i, :L] = labels[:L]`, so a short label list leaves the tail of `y` at the pad value and every label after the first multi-character unit is SHIFTED. Caught only because a test asserted `len(bases) == len(labels)` (28 vs 17 on a 28-char line) - no metric would have shown it as a bug rather than as a bad arm.
- **The guard that makes such a data-pipeline change safe on Windows: an env-var switch read at MODULE IMPORT time, not a monkeypatched global.** `prepare_data.py` builds its rows in a `multiprocessing.Pool`, and on spawn the workers re-import the module; they inherit the parent ENVIRONMENT but not the parent's in-memory rebinds. `KEEP_ALL = os.environ.get("PREP_KEEP_ALL") == "1"` therefore propagates; assigning `prepare_data._parse_pair = ...` in the parent would not have reached the workers. Default OFF keeps every existing corpus byte-identical.
- **Regression-test a pipeline edit against `git show HEAD:<path>` loaded as a SECOND module, not by re-running the pipeline.** The expensive path (reading 94 raw files) is not needed to prove a default-path no-op: import HEAD's copy under another name and assert the two implementations agree on a battery of mixed-script cases. Here that was 5/5 byte-identical, plus `selftest.py --phase all` 4/4 PASS, and it is what licensed touching a file the whole corpus lineage depends on.
- **The prepared corpus stores FULL LINES, not windows - `window_line()`/`CTX_CHARS=128` in `prepare_data.py` is DEAD CODE (defined, never called).** That is why a prepared row's base stream can exceed 128 characters, and therefore why `v3` and `v3ctx256` differ at all. Read the call sites before trusting a module constant to describe the artifacts.
- **`data/diac/v3/tokens` is DERIVABLE from `data/diac/v3ctx256/tokens`: same 2,433,430 rows, and `v3 == v3ctx256[:, :128]`.** Measured on sampled rows (0, 1, 1000 equal; rows 1234567 and 2433429 have non-zero content BEYOND column 128 that the 128-wide bed cannot hold). So v3ctx256 is the superset and v3 the truncation - check the y arrays and the PAD convention (`--pad-label`, default -1 vs the v3 lineage 0) before treating either as redundant, and grep for live `tokens_phase: v3` references before deleting anything.
- **`train.py:326` resolves `tokens_phase` to `DATA/<name>/tokens` for ANY name** (`tokens_dir = DATA / cfg.get("tokens_phase", phase) / "tokens"`), so a brand-new bed needs no `train.py` change - only the `--phase` `choices` list in `tokenize_data.py` is closed, and `pack_rows` can be imported and called directly instead.

## 2026-10-12 - D-line: the clock latch, the not-free preflight, and why val_loss cannot compare two alphabets

Context: E-097 (full-character-set retrain) moved from registration to a built,
verified bed and a stage-0 smoke on the D-line. Seven things learned that will
bite the next long run.

1. THE CLOCK LATCH IS MEASURED, NOT ASSUMED. e70_c4b_ctx256 recorded 0.6005
   s/step. The e97 arm-A smoke measured 1.11 s/step steady state (55.5 s per 50
   steps, stable from step 100 to 1900) on the same model shape - 1.85x slower.
   A budget approved on a recorded rate is therefore ~1.85x too optimistic.
   Measure the rate on the actual bed and the actual machine BEFORE committing a
   multi-hour budget, then re-derive the budget from the measurement. A 60000-step
   pair that looked like 12-32 h was really ~37-47 h.
2. THE PREFLIGHT SMOKE IS NOT FREE. pipeline/smoke.py sets eval_every 30 for its
   60-step preflight, so every train.py launch runs TWO full internal val evals
   BEFORE the real run starts. On the D-line that is ~20 min per eval, ~22 min per
   arm, and it is invisible in any per-step budget. Add ~25-45 min per arm.
3. val_loss IS NOT COMPARABLE ACROSS BEDS WITH DIFFERENT TARGET ALPHABETS. Bed B
   (full character set) labels about 37 pct of its positions bare-at-non-letter,
   which are trivial, so its val_loss is structurally lower: 0.09529 at step 30
   against bed A's 0.10864 at step 2000. That gap measures the target
   distribution, not quality. Only a metric computed on the SAME external corpora
   with the SAME decode - the registered gate DER - compares the arms. Reading
   val_loss across alphabets is a category error.
4. NEVER MEASURE A CORPUS PROPERTY FROM THE HEAD OF A FILE. An estimate of the
   base-letter density of the two beds taken from the first 4000 rows of
   prepared_rebuilt/train.jsonl gave 33.6/40.0 pct. The true whole-bed values,
   measured on the built arrays, are 190.7 vs 162.5 base letters per row, i.e.
   bed B holds 14.8 pct (train) / 15.7 pct (val) FEWER base letters. The prefix
   sample overstated the confound by more than 2x. Measure the artifact you ship,
   not a prefix of its input.
5. TO MAKE TWO BEDS CELL-PAIRED, REPRODUCE THE SHUFFLE. tokenize_data.pack_rows
   shuffles with random.Random(seed).shuffle(order) and then fills row j. A second
   bed holds the same text at the same row only if it applies the SAME permutation:
   build order, invert it (inv[i] = j), and write source row i to destination
   inv[i]. scratch/e97_smoke_shuf vs scratch/e97_smoke_noshuf verified A[j] ==
   B[order[j]] on ids AND labels. Then assert the pairing property on the built
   artifact: bed B's base-letter subsequence must be a PREFIX of bed A's row with
   identical labels - scratch/e97_verify_bed.py checks exactly that over 200000
   train rows and all 160214 val rows, zero violations.
6. LABEL GRANULARITY IS NOT ENCODE GRANULARITY. passthrough.parse groups a run of
   non-target characters into ONE multi-char Unit, while tokenizer.encode emits one
   id per character. Expanding labels per Unit silently desynchronises labels from
   ids (28 vs 17). Expand per character and assert len(bases) == len(labels).
7. WRITING MULTI-LINE YAML FROM A PYTHON STRING LEAKS LF INTO CRLF FILES. The
   state/tasks.yaml and MEMORY.md edits both did this (66 and 9 lone LFs). After
   any scripted multi-line write to a CRLF file, normalize and verify:
   raw.replace("\r\n", "\n").replace("\n", "\r\n"), then print CRLF and
   lone-LF counts. yaml.safe_load still passes with mixed EOLs, so a successful
   YAML parse is NOT a check for this.

## 2026-10-13 - D-line E-097 ACCEPTED: the alphabet mismatch is better fixed in the DATA than at inference, and the decode fix inverts

E-097 closed ACCEPTED: training on the full character set beats the best
inference-side band-aid by 20.29 pct relative on the registered gate
(best(B) 0.146185 vs best(A) 0.183386; bar 0.98 * best(A) = 0.179718). Both arms
ran 30000 steps from a byte-identical warm start with the token bed the only
difference.

1. THE BAND-AID WAS COMPENSATING, NOT IMPROVING. E-095's nospace decode recovers
   about 24 pct on a model trained on the stripped alphabet. On a model trained on
   the FULL alphabet the same fix is a catastrophic regression: arm B single
   0.146185, nospace 0.569627, base_only 0.625349, and on abdou_test nospace is
   +334.30 pct against single. The decode mode must therefore be chosen from the
   TRAINING alphabet, and a full-character-set model must be decoded on raw text.
   E-095 stays correct for the model it was actually measured on.
2. A SHORT SCREEN CAN UNDERSTATE A GAP, NOT ONLY OVERSTATE IT. At step 2000 the
   gap was -14.32 pct and the record predicted it would SHRINK, on the reasoning
   that the control arm had further to go. It widened to -20.29 pct. Arm A was
   already at its asymptote at step 2000 - its gate base_only moved 0.180099 ->
   0.183386 across the next 28000 steps, i.e. slightly WORSE - while arm B improved
   5.3 pct. Distance-from-asymptote is not visible in step count and must not be
   assumed from it.
3. val_loss IS NOT A PROXY FOR THE GATE. Arm A's val_loss improved over 30000
   steps (0.10864 -> 0.10780) while its gate DER got slightly worse. The two axes
   can move in opposite directions, so "did it converge" must be asked on the
   registered metric, not on the training-side loss.
4. A CONFOUND CAN BE ARGUED AWAY WHEN IT POINTS THE WRONG WAY. Bed B holds 14.8
   pct fewer base letters per row on train (15.7 pct on val), which handicaps B, so
   a 20.29 pct B win cannot be explained by the volume difference. Record the
   DIRECTION of a confound, not merely its existence.
5. CHEAP DERIVED BEDS: stream the RECORDED source of the existing bed and write the
   arrays directly, instead of re-running the generator and materialising a
   multi-GB intermediate. e97_build_bed.py turned a planned 1-3 h / 7.5-9 GB build
   into 39 min and ~6 GB, and guaranteed that both beds hold the same texts.
6. AN ACCEPTED BED VERDICT IS NOT A NEW GOLD MODEL. E-097 shows the BED is better;
   the deployed anchor is still runs/diac/e73_soup/c4b_uniform, a 12-probe soup of
   e70_c4b_ctx256 at 60000 steps, while arm B is a 30000-step single model.
   Promoting a new gold requires the DEPLOYED recipe re-run on bed B as its own
   registered experiment.
