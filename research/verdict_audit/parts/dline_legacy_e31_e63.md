# Verdict audit — D-line legacy E-31..E-55 (mu2/mu3/dia2) and E-61..E-63

Scope: recorded verdicts in \`research/EXPERIMENTS.md\` for the mu2/mu3/dia2 arm
(E-31..E-55) plus E-61 and the diacritizer E-63, re-checked against artifacts
under \`runs/\` and the registry rows \`research/experiments.yaml\`
\`diacritizer/E-061\` (line 121) and \`diacritizer/E-063\` (line 36).

Read-only audit. No file in the repository was modified except this report.

## Method and metric-direction table

Recheck vocabulary used: CONFIRMED / INVERTED / UNSUPPORTED / NO_ARTIFACT /
MISCITED. "CONFIRMED" means the cited artifact exists and carries the claimed
number within rounding; a verdict whose *direction* is right but whose numeric
evidence is prose-only is not CONFIRMED.

Directions (confirmed against the selector code / anchors, not assumed):

| metric | direction | proof in this repo |
|---|---|---|
| DER / WER / DER-lite / CDER | lower better | \`research/EXPERIMENTS.md:738\` "DER lower=better"; \`diacritizer/scripts/train.py:215\` keeps new best only when \`mean_der < prev\` |
| CE / mixed CE / retention CE / val_loss / NLL / BERT | lower better | \`runs/mex/mu3_tall/final/gates_e50.json\` gate \`mixed CE <= 2.2753\`; E-45 gate (c) "<= 3.49" |
| accuracy / markpos / allpos / fill acc / mark F1 / router acc / exact | higher better | E-32 "fill acc ... PASS" 0.6891 > 0.4063; E-48a bar "0.80" |
| contrastive_lift | higher better | \`research/EXPERIMENTS.md:754\`, external lift bar >= 0.20 |
| per-class floor | higher better | E-61 v21c "NO class below attempt-3's recorded floor" |

No metric-direction inversion (the E-067/P1 bug class) was found anywhere in
E-31..E-63. Every FAIL/PASS below is direction-consistent with the table above.
The defects found are evidence-provenance defects (prose-only numbers,
same-path artifact collisions, denominator-zero probes) and two stale
"best overall" rankings in E-63 appendices.

## Per-verdict sections

### E-31e — recorded PASS (fill/retention) — UNSUPPORTED
Compared: E-31e (lr 1e-4, 3000 steps, hold_every 3) vs either-guess 0.4063 and
the retention guard 0.7431. Metrics direction: fill acc higher-better,
retention CE lower-better. Recorded 0.6765 / 0.7407 (lines 93, 97).
Cited artifact: \`configs/mu2_g2.yaml\`, weights \`runs/mex/mu2_g2/final\`.
Recheck: the sweep gate values 0.6765/0.7407 appear in no metrics artifact.
A tfevents tag scan of the \`runs/mex/mu2_g2*\` arms shows only
\`train/loss\`, \`eval/loss\`, \`eval/runtime\`, \`samples_per_second\`,
\`steps_per_second\` — no fill-acc / retention-CE field. Verdict direction is
correct (0.6765 > 0.4063; 0.7407 <= 0.7431) but the numbers are unsupported.
Corrected: PASS plausible, numbers not artifact-backed.

### E-32 — recorded PASS (widen) — CONFIRMED (cross-artifact anchor)
Compared: widened trunk (hidden 160->320) + LoRA settle vs either-guess 0.4063
and guard 0.7431. Recorded fill 0.6891, retention 0.7395 (lines 113-116).
Recheck: \`runs/mex/mu2_g3/final\` has no gate JSON, but the two gate numbers
survive as anchors in later artifacts: \`runs/mex/mu2_g4b/train_summary.json\`
\`gate_trunk_only: 0.6891\` and \`runs/mex/mu2_g4/train_summary.json\`
\`gate_retention_anchor: 0.7395\`. Direction: 0.7395 < the prior 0.7407 is an
improvement (lower-better) — the ledger uses it in that sense, correctly.
Corrected: CONFIRMED, with the anchor provenance noted.

### E-33 — recorded PASS (mark head + DER-lite) — CONFIRMED
Recorded mark acc 0.8518, n=101,930, Wilson95 [0.8496, 0.8539] (lines 126-131).
Recheck: \`runs/mex/mu2_g4/train_summary.json\` = \`mark_acc 0.85177\`,
\`mark_n 101930\`, \`wilson95 [0.84958, 0.85394]\`, \`gate_either_guess 0.4063\`.
Direction higher-better > 0.4063. Corrected: CONFIRMED.

### E-34 — recorded FAIL (composed decode) — CONFIRMED
Recorded composed 0.2796 vs gate 0.6891 (line 149). Recheck:
\`runs/mex/mu2_g4b/train_summary.json\` \`composed_acc 0.27964\`. Direction
higher-better; 0.27964 < 0.6891 -> FAIL correct. The read-only diagnostics
0.6896 (composed marks) and 0.5464 (composed non-mark) are prose-only
(no artifact carries them). Corrected: CONFIRMED; diagnostics UNSUPPORTED.

### E-35 — recorded PASS (corrected rule) — CONFIRMED
Recorded all 0.6196, mark 0.7556, non-mark 0.5418, trunk-only 0.7515
(lines 162-170). Recheck: \`runs/mex/mu2_g4b2/eval_rule.json\` = \`all 0.6196\`,
\`markpos_head 0.7556\`, \`markpos_trunk_only 0.7515\`, \`nonmark 0.5418\`, both
gates PASS. Direction higher-better. Corrected: CONFIRMED.

### E-36 — recorded FAIL(a), PASS(b)(c) — CONFIRMED
Recorded bridged markpos 0.7632 (83,154/108,956) vs target 0.7660, identity
\`0.00e+00\` (lines 186-193). Recheck: \`runs/mex/mu2_g5/summary.json\` =
\`bridged_mark_acc 0.76319\`, \`trunk_only_mark_acc 0.75148\`,
\`gate_a_target 0.766\`, \`gate_a "FAIL"\`, \`identity_max_dlogit 0.0\`.
Direction higher-better < 0.766 -> FAIL correct. Corrected: CONFIRMED.

### E-37a — recorded PASS (two-layer bridge) — CONFIRMED
Recorded markpos 0.7846 >= 0.7660 (line 215). Recheck:
\`runs/mex/mu2_g37a/summary.json\` \`bridged_mark_acc 0.78456\`,
\`gate_a "PASS"\`. Corrected: CONFIRMED.

### E-38a — recorded PASS (bridges live) — MISCITED (false-positive risk)
Recorded bridges-ON composed 0.6485 all / 0.7862 mark / 0.5698 non-mark, and
bridges-OFF identity 0.6196/0.7556 (lines 230-241). Cited artifact
\`runs/mex/mu2_e38a/summary.json\`. Recheck: the file contains the **OFF**
values — \`all 0.6196\`, \`markpos 0.7556\`, \`nonmark 0.5418\` — with the PASS
flags. The ON headline numbers exist nowhere in the artifact tree. Root cause
in the writer: \`mex/scripts/eval_mu2_e38a.py:96\` emits
\`json.dumps(res_on | res_off | gates)\`, a dict-union in which \`res_off\`
silently overwrites \`res_on\`; the ON results were console-only. This is the
single strongest false-positive risk in E-31..E-55: the recorded PASS rests on
numbers the cited artifact does not contain. Corrected: verdict MISCITED; the
bridges-ON PASS is not reproducible from the artifact (OFF values confirmed).

### E-39a — recorded PASS (bridge-aware head) — CONFIRMED
Recorded markpos 0.7984, all 0.6533, nonmark 0.5703 (lines 252-260). Recheck:
\`runs/mex/mu2_e39a/train_summary.json\` = \`all 0.6533\`, \`markpos 0.7984\`,
\`nonmark 0.5703\`, \`g_a PASS\`, \`g_b PASS\`. Direction higher-better.
Corrected: CONFIRMED.

### E-41 — recorded PASS (x-tower on frozen mu2 trunk) — CONFIRMED
Recorded x-CE 6.1823 -> 4.2513, rel -31.2%, identity 0, hygiene exact
(lines 282-290). Recheck: \`runs/mex/mu2_e41/summary.json\` =
\`ce_trunk_off 6.18231\`, \`ce_xtower 4.25130\`, \`rel_drop 0.3123\`,
\`identity_max_dlogit 0\`, \`gate_a "PASS"\`; \`runs/mex/mu2_e41/hygiene.json\`
reproduces 0.6533/0.7984. Direction CE lower-better. Corrected: CONFIRMED.

### E-42 — recorded PASS (Net2Net widen x2 + remount) — CONFIRMED
Recorded max dlogit 1.14e-05, remount markpos 0.7984 EXACT / all 0.6534
(lines 303-307). Recheck: \`runs/mex/mu2_g4_init/verify.json\`. Direction:
the 0.6534 vs canonical 0.6533 is a one-position fp rounding drift, inside the
pre-registered 1e-4 bar, and the ledger documents it. Corrected: CONFIRMED.

### E-43 — recorded PASS (mu3 G-settle) — NO_ARTIFACT
Recorded retention CE 0.7391, fill 0.6972 (75966/108956), mount 0.6564 all /
0.8012 mark, trunk-only 0.6242/0.7620 (lines 327-337). Recheck: none of these
numbers is in any artifact. \`runs/mex/mu3_g4/final\` holds only
\`train_summary.json\` (\`best_eval_loss 0.74684\`) plus weights. The values are
prose-only, and the 0.8012 anchor is additionally a hardcoded literal in
\`mex/scripts/eval_mu3_e44_cohab.py:97\` (\`"anchor_E43":{"all":0.6564,"markpos":0.8012}\`).
\`mex/scripts/eval_mu2_e43_mounts.py\` prints its results and writes no JSON
(only \`print(...)\` at lines 80/86). Corrected: NO_ARTIFACT; verdict direction
is plausible but unverifiable from the cited evidence.

### E-44 — recorded (a)(b) PASS / (c) FAIL — MISCITED
Recorded x-CE 4.2467 -> 3.3229 (-21.8%), identity 0, cohabitation 0.6459 mark
vs 0.8012 mu2-only, bar >= 0.78 (lines 361-370). Recheck:
\`runs/mex/mu3_xtower/summary.json\` = \`ce_trunk_off 4.2467\`,
\`ce_xtower 3.3229\`, \`rel_drop 0.2175\`, \`identity_max_dlogit 0\`,
gates a/b PASS — so (a)/(b) are CONFIRMED. But gate (c)'s 0.6459 is
console-only: \`mex/scripts/eval_mu3_e44_cohab.py\` writes no JSON (prints at
lines 91/96). The recorded FAIL(c) — which drove the whole E-45 redesign — has
no artifact. Corrected: (a)/(b) CONFIRMED, (c) UNSUPPORTED.

### E-45 — recorded ALL PASS (joint two-tower co-train) — MISCITED
Recorded markpos 0.7943 (bar 0.78), mixed CE 2.2753 (bar <= 3.49), all 0.6462
(bar 0.6202), identity 0 (lines 390-401). Cited artifact
\`runs/mex/mu3_joint/gates.json\`. Recheck: the file now holds
\`{"gates":{"gate_b":true,"gate_c":true,"gate_d":true},
"composed":{"all":0.6461,"markpos":0.7941},"mixed_ce":2.2752}\` — these are the
**E-54b Muon** numbers, not E-45's 0.7943/0.6462/2.2753. Cause:
\`mex/scripts/eval_mu3_e45_gates.py:130\` and
\`mex/scripts/eval_mu3_dia2_gates.py:131\` both \`write_text\` the same path, so
the later dia2 eval overwrote E-45's output. Gate thresholds are otherwise
identical between the two scripts (\`markpos>=0.78\`, \`ce<=3.49\`,
\`all>=0.6202\`). Corrected: MISCITED; E-45's own numbers are prose-only and
were destroyed by a path collision (the surviving file is E-54b Muon).

### E-46 — recorded capability boundary (first-char fill) — CONFIRMED
Recorded x1 7/50, x2 1/50, x3 0, x4 0; trunk 0/50 (lines 440-446). Recheck:
\`runs/mex/mu3_joint/e46_fill.json\` = \`fill_composed {x1:[7,50], x2:[1,50],
x3:[0,50], x4:[0,50]}\`, \`fill_trunkonly\` all \`[0,50]\`. Corrected: CONFIRMED.

### E-47 — recorded router PASS, (c)/(d) FAIL — CONFIRMED
Recorded router 1.0000, (c) 0.36 FAIL, (d) 0.7708 FAIL (lines 478-484).
Recheck: \`runs/mex/mu3_router/gates.json\` = \`router_acc 1.0\`,
\`x2_digit_acc 0.36\`, \`x3_acc 0.77083\`, \`gates {gate_b true, gate_c false,
gate_d false}\`. All higher-better; all failures direction-correct. Note the
ledger's (c)/(d) labels correspond to the artifact's \`x2_digit_acc\`/\`x3_acc\`.
Corrected: CONFIRMED.

### E-48a — recorded both magnitude gates FAIL — CONFIRMED
Recorded x2 first-char 0.4200 (21/50), x3 0.8000 (40/50) (lines 511-520).
Recheck: \`runs/mex/mu3_router/gates_e48a.json\` = \`x2_acc 0.42\`,
\`x3_acc 0.8\`, gates false. Higher-better vs bars 0.80/0.85. Corrected: CONFIRMED.

### E-48b — recorded both exact-match gates FAIL (MoE readout) — CONFIRMED
Recorded x2 EXACT 0.00 (0/50), x4 EXACT 0.00 (0/50) (lines 534-543). Recheck:
\`runs/mex/mu3_router/gates_e48b.json\` = \`x2_exact 0.0\`, \`x4_exact 0.0\`,
\`counts {x2:[0,50], x4:[0,50]}\`, both gates false. Higher-better. Corrected:
CONFIRMED. (The claimed \`expert_x2.pt\`/\`expert_x4.pt\` weights are secondary;
the verdict is backed by the gates file.)

### E-49 — recorded read-only demo PASS — CONFIRMED
Recorded demo.md with >= 5 live rows (lines 552-559). Recheck:
\`runs/mex/mu3_router/demo.md\` exists. Documentation/read-only rung, no gate
number. Corrected: CONFIRMED.

### E-50 — recorded FAIL (t1/t2), mixed CE PASS — MISCITED (denominator zero)
Recorded: (id) 0.0 PASS; (t3) mixed CE 1.8757 PASS; x1 0.1121 FAIL; x2 fill
0/50 FAIL; x3 25/50 "direction PASS" (lines 573-585). Recheck:
\`runs/mex/mu3_tall/final/summary.json\` = \`{"x2_fill":[0,0],
"mixed_xce":1.8756782237221212}\` — mixed CE is CONFIRMED (lower-better,
1.8757 <= 2.2753), but the x2 gate artifact is **[0,0]: a zero denominator**,
not "0/50". The ledger's "0/50" implies 0 correct of 50; the artifact says 0
probes ran/qualified, so that FAIL is not supported as stated.
\`runs/mex/mu3_tall/final/gates_e50.json\` holds only the calibration values
\`x1_composed_characc 0.37828\`, \`x2_fill [18,50]\`, \`x3_acc [20,50]\`,
\`x1_n 571\` (E-45 standing); the taller-trunk's x1 0.1121 and x3 25/50 are
absent. Corrected: mixed-CE gate CONFIRMED; x1/x3 MISCITED; x2 UNSUPPORTED
(denominator 0).

### E-51 — recorded every gate FAIL — NO_ARTIFACT
Recorded x1 0.1191, x2 0/50, x3 29/50, mixed CE 5.6833 (lines 596-607).
Recheck: none of 0.1191 / 5.6833 / 29/50 appears in any file under \`runs/\`
(the only "0.0076"/"0.68xx" hits are unrelated soup/laya logs). Prose-only.
Corrected: NO_ARTIFACT; FAIL plausible but unverifiable.

### E-52 — recorded every gate FAIL (trunk growth + family LoRA) — NO_ARTIFACT
Recorded B1 0.2890, B2 0.0076, x2 6/50, x3 20/50, LoRA x2 trajectory to 18/50
(lines 619-639). Recheck: no artifact carries these numbers, and the ledger
claims a save at \`runs/mex/mu3_lora/x2\` while \`runs/mex/mu3_lora/\` contains
**0 files**. Prose-only. Corrected: NO_ARTIFACT; the claimed archived weights
are not present.

### E-53 — recorded DONE / PASS (documentation rung) — CONFIRMED
Recorded capability card at \`runs/mex/mu3_joint/capability_card.md\` (line 645).
Recheck: file exists; it restates the measured gates. Caveat: it also restates
the unbacked E-43/E-45/E-54 numbers, so it is a secondary (derived) source, not
gate evidence. Corrected: CONFIRMED as a documentation rung.

### E-54a — recorded PASS (zero-loss widen 640->1280) — NO_ARTIFACT
Recorded w2 markpos 0.7940 vs 0.7943, w3 mixed CE 2.2752 vs 2.2753, w1 in-script
(lines 669-678). Recheck: \`runs/mex/dia2_init\` has no gate JSON. The only
surviving file with these metrics is \`runs/mex/mu3_joint/gates.json\`, which
holds E-54b Muon's 0.7941/2.2752 (see E-45). w1 is an in-script assertion, not
persisted. Corrected: NO_ARTIFACT (verdict plausible; numbers near-miss and not
artifact-backed).

### E-54b — recorded Muon m1 FAIL, AdamW m2 PASS — MISCITED
Recorded AdamW 0.74688 / 0.7987 / 0.6504 / 2.2798 and Muon 0.74674 / 0.7941 /
0.6461 / 2.2752 (lines 680-693). Recheck: \`runs/mex/dia2b_adamw/final/train_summary.json\`
and \`runs/mex/dia2b_muon/final/train_summary.json\` confirm only
\`best_eval_loss 0.74688\` / \`0.74674\`. The composed numbers survive only for
the **losing Muon arm** in \`runs/mex/mu3_joint/gates.json\`
(0.7941/0.6461/2.2752); the **winning AdamW arm's** 0.7987/0.6504/2.2798 are
prose-only. Direction higher-better: 0.7941 < 0.7987 -> m1 FAIL is
direction-correct. Corrected: MISCITED; the deciding AdamW composed numbers are
not artifact-backed (same output-path collision as E-45).

### E-54c — recorded c1 FAIL, c2 PASS, c3 FAIL — NO_ARTIFACT
Recorded c1 0.7920/2.5265 vs 0.7987, c2 0.7947/0.6460, c3 shared-KV 0.5419,
control 0.6000/0.4751 (lines 707-716). Recheck: none of these numbers appears
under \`runs/\`. \`runs/mex/dia2c_ctx192/final/train_summary.json\` holds only
\`best_eval_loss 0.75350\`. The script \`mex/scripts/eval_dia2c.py:132\` writes
\`runs/mex/gates_dia2c.json\`, but that file now contains E-59 dia2i's
\`composed {all 0.6002, markpos 0.7534}, mixed_ce 4.0083\` — again overwritten.
Corrected: NO_ARTIFACT; verdict direction plausible, evidence destroyed by a
path collision.

### E-54d — recorded PASS (replay-scale settle) — NO_ARTIFACT
Recorded markpos 0.8134 (> 0.7987), all 0.6672, mixed CE 2.3829 (lines 726-734).
Recheck: \`runs/mex/dia2d_scale/final/train_summary.json\` = \`best_eval_loss
0.74715\` only. 0.8134/0.6672/2.3829 appear only in
\`runs/mex/mu3_joint/capability_card.md\` (a restatement of the same prose).
Direction higher-better / CE lower-better; plausible, not artifact-backed.
Corrected: NO_ARTIFACT.

### E-55 / E-55b / E-55c — recorded honest external loss — CONFIRMED (one MISCITED mean)
Recorded dia2d mini mean DER 0.858 vs gold 0.456 / ZM 0.525 / BiLSTM 0.558
(lines 744-754), and the E-55c profile means (lines 779-789). Recheck against
\`runs/mex/e55_bench/*.metrics.json\`:
- dia2d der_collapse 0.8192 / 0.9373 / 0.8251 / 0.8500 -> mean 0.8579 = 0.858;
  gold_v3q 0.3355 / 0.4533 / 0.5569 / 0.4773 -> mean 0.4558 = 0.456;
  e23c_zm mean 0.5249 = 0.525. All CONFIRMED.
- E-55c means: dia2d mark_acc 0.472, word_ax 0.040, word_partial50 0.470,
  lift 0.035, F1 0.487, der_collapse 0.858 all recompute correctly from the
  metrics.json files. Gold: mark_acc 0.826, word_ax 0.489, word_partial50 0.958,
  lift 0.437, der_collapse 0.456 all correct.
- One MISCITED value: E-55c gold **mark F1 mean** is printed as 0.822, but the
  artifact values (fadel 0.8686, sadeed 0.8406, wn24 0.7894, wn14 0.8199) give
  **0.8296**. e23c_zm's 0.789 recomputes correctly.
- E-55's "nocase" parentheticals (0.7461/0.8639/0.7677/0.7821) and the strict
  DER column "0.8191" are prose-only; no gate_eval/DER-nocase CSV exists under
  \`runs/mex/e55_bench\` (only \`*.metrics.json\`, \`*_perline.csv\`, pred/ref
  text). The lenient value 0.8192 in metrics.json is the artifact-backed one.
Corrected: CONFIRMED (mean DER loss is real); gold mark-F1 mean MISCITED;
nocase parentheticals UNSUPPORTED.

### E-61 (V2-1 KD arm) — recorded MIXED (v21a PASS, v21b PASS, v21c FAIL) — partial
Registry: \`research/experiments.yaml:121-150\` (status closed, verdict partial).
- **v21a PASS (in-domain markpos 0.6824 vs 0.6496)** — UNSUPPORTED. The run dir
  \`runs/mex/dia2j_v21_final\` contains only \`final/config.json\`,
  \`final/generation_config.json\`, \`final/head15.safetensors\`,
  \`final/model.safetensors\` — no train_summary, no metrics, no gate JSON. The
  value 0.6824 appears nowhere under \`runs/\` or \`scratch/\`. Prose-only.
- **v21b PASS (external mark-acc mean 0.3652 vs 0.207)** — CONFIRMED with a
  rounding MISCITE: \`scratch/prof_v21/{fadel,sadeed,wn24,wn14}.metrics.json\`
  give mark_accuracy 0.3487 / 0.3703 / 0.3730 / 0.3696 -> mean **0.3654**, not
  the recorded 0.3652 (delta 0.0002). Per-gate values match the prose; the
  abdou contrastive_lift -0.0242 is present.
- **v21c FAIL (per-class floor 0.00/0.01)** — NO_ARTIFACT. The 15-value
  per-class vector is prose-only; no artifact carries it.
Corrected: v21b CONFIRMED (mean MISCITED by 0.0002); v21a/v21c prose-only, so
the "MIXED verdict" rests on one artifact-backed gate out of three. The
registry verdict \`partial\` is conservative and honest.

### E-63 (diacritizer trial-bed ladder) — recorded partial — CONFIRMED
Registry: \`research/experiments.yaml:36-63\` (status closed, verdict partial).
Trial-bed/full-data numbers all match artifacts:
- \`runs/e63_arms.log:30/63/96/130/162/195/228\` -> S0 0.2057/0.9317,
  S1 0.2168/0.9281, S2 0.3609/0.8796, S3 0.3976/0.8655, S4 0.7178/0.9327,
  S5 0.2125/0.9297, C1 0.7245/0.9299 (all match lines 889-895).
- \`runs/e63_finalists_eval.log\` -> full-val S0 acc 0.93237 / DER-lite 0.06763,
  S4 0.93329/0.06671, S5 0.91491/0.08509, C1 0.91493/0.08507 (all match).
- \`runs/diac/stage3best_ls/gate_eval.csv\` -> 0.50643/0.63089/0.64817/0.60560/
  0.56636 (matches line 899), text_preservation 1.0 except abdou 0.9846.
- \`runs/diac/stage2b2500/gate_eval.csv\` -> 0.33234/0.45679/0.57583/0.48994
  (matches line 900): every external gate regressed vs the deployed gold while
  in-domain val rose. Corrected: CONFIRMED; verdict partial is accurate.

### E-63c appendix (full-data M1) — recorded "new overall best on every recorded gate" — MISCITED (false claim)
Recorded champion step 5000 fadel 33.08 / sadeed 45.44 / wn24 57.17 / wn14 49.07
/ abdou 41.19 vs gold 33.23 / 45.68 / 57.58 / 48.99 (report
\`research/e63_overfit_report_2026-09-23.md:117-124\`). Recheck against
\`runs/diac/e63_m1_full/gate_eval.csv\` step 5000: 0.33077 / 0.45447 / 0.57168 /
0.49070. The claim "all gates strictly better or equal vs gold" is **false** for
wikinews2014: champion 0.49070 > gold 0.48994 (higher DER = worse) by 0.00076.
The 4-gate mean 0.4619 < gold 0.4637 is correct, so "new overall best" on the
mean holds; only the per-gate absolute claim is wrong. The selection metric
\`runs/diac/e63_m1_full/gate_probe/best_gate.json\` = \`{"step":5000,
"mean_der":0.42531}\` is a **3-gate** mean (fadel+sadeed+wn2014)/3 — it excludes
wn2024 and abdou, so "every recorded gate" overstates the selector. Champion
weights \`gate_probe/weights_step5000.pt\` and \`best_gate_weights.pt\` both exist.
Corrected: MISCITED/false-positive on the per-gate claim; mean-best CONFIRMED.

### E-63s appendix (smallest-model sweep) — recorded "BEST OVERALL = S3 (8.64M) at 40.77" — MISCITED (false-positive ranking)
Recorded table (report lines 137-152): S3 8Lx272 best 40.77; S4 12Lx160 "best
step 10000, tracker 41.30, wall-hit"; S4/S5/S6 listed as auto-resume candidates.
Recheck:
- \`runs/e63s_s3.log:97\` [GATE-BEST] step 5500 mean_external_DER 40.77 — matches.
- \`runs/e63s_s4.log:327\` MONITOR-END done_steps **20000** (not 10000), and
  \`runs/e63s_s4.log:278\` [GATE-BEST] step 16500 mean_external_DER **39.96**
  (\`runs/diac/e63s_s4_12l160/gate_probe/best_gate.json\` =
  \`{"step":16500,"mean_der":0.39962}\`; \`gate_eval.csv\` step 16500 =
  fadel 0.30685 / sadeed 0.42744 / wn24 0.56132 / wn14 0.46456).
  39.96 < 40.77, so **S4 is the artifact-backed best**, and the recorded
  S3-best claim is a false positive; S4's row is understated (a false negative).
The other rows match their logs (S1 43.60, S2 42.91, S5 44.16, S6 46.45).
Corrected: "BEST OVERALL = S3" MISCITED/false-positive; S4 (4.5M) is the best
recorded state.

### E-63 report internal inconsistencies (same artifact set) — MISCITED
- \`research/e63_overfit_report_2026-09-23.md:91\` lists raw stage3best_ls as
  \`0.5064 / 0.6163 / 0.6482 / 0.6056 / 0.5978\`; the artifact
  \`runs/diac/stage3best_ls/gate_eval.csv\` gives sadeed **0.63089** (not 0.6163)
  and abdou **0.56636** (not 0.5978). The line-899 ledger values (0.6309 /
  0.5664) are the correct ones.
- Line 121's "raw baseline" row repeats the avg3 row's values
  (47.94/60.49/63.28/59.18) and prints 0.5978, which is not their mean (0.5772).
  The label/values are inconsistent.

### Documentation hazard: duplicate experiment ids — MISCITED
\`research/EXPERIMENTS.md\` reuses E-40..E-59 ids across two different lines
(mu-line and the DA/langid line). Examples: line 313 puts a DA-3 topic-tagging
result under the heading "## E-43 (PRE-REGISTERED, mu3 G-settle …)"; lines 301,
308-322 interleave DA-1/DA-2/DA-3/DA-9 rows among the mu-line E-42..E-55
blocks; lines 916-922 reuse E-63/E-64 for the Laya line. Any audit or
"authority" lookup that keys only on the id is ambiguous. \`research/experiments.yaml\`
sidesteps this by migrating only \`diacritizer/E-061\` (line 5-10 migration note)
and registering E-063 from the diacritizer line.

## Final table

| id | recorded verdict | metric + direction | cited artifact | recheck | corrected one-line verdict |
|---|---|---|---|---|---|
| E-31e | PASS | fill acc ↑ 0.6765 > 0.4063; retention CE ↓ 0.7407 <= 0.7431 | configs/mu2_g2.yaml; runs/mex/mu2_g2/final (weights only) | UNSUPPORTED | direction correct; gate numbers not in any artifact (tfevents loss only) |
| E-32 | PASS | fill ↑ 0.6891; retention CE ↓ 0.7395 | runs/mex/mu2_g3/final (no gate JSON) | CONFIRMED | corroborated by anchors in runs/mex/mu2_g4{,b}/train_summary.json |
| E-33 | PASS | mark acc ↑ 0.85177, n=101930 | runs/mex/mu2_g4/train_summary.json | CONFIRMED | exact match incl. Wilson CI |
| E-34 | FAIL | composed ↑ 0.27964 < 0.6891 | runs/mex/mu2_g4b/train_summary.json | CONFIRMED | match; diagnostics 0.6896/0.5464 prose-only |
| E-35 | PASS | all ↑ 0.6196; mark ↑ 0.7556 >= 0.7515 | runs/mex/mu2_g4b2/eval_rule.json | CONFIRMED | match, both gates PASS |
| E-36 | FAIL(a) | bridged markpos ↑ 0.76319 < 0.7660 | runs/mex/mu2_g5/summary.json | CONFIRMED | match; identity 0 |
| E-37a | PASS | bridged markpos ↑ 0.78456 >= 0.7660 | runs/mex/mu2_g37a/summary.json | CONFIRMED | match |
| E-38a | PASS | ON all ↑ 0.6485 / mark ↑ 0.7862 | runs/mex/mu2_e38a/summary.json | MISCITED | artifact holds OFF 0.6196/0.7556; ON headline console-only (eval_mu2_e38a.py:96 union overwrite) — false-positive risk |
| E-39a | PASS | mark ↑ 0.7984; all ↑ 0.6533 | runs/mex/mu2_e39a/train_summary.json | CONFIRMED | match |
| E-41 | PASS | mixed CE ↓ 6.18231→4.25130 (-31.2%) | runs/mex/mu2_e41/summary.json + hygiene.json | CONFIRMED | match |
| E-42 | PASS | widen dlogit <1e-2; remount 0.6534/0.7984 | runs/mex/mu2_g4_init/verify.json | CONFIRMED | 0.6533 vs 0.6534 is documented fp rounding inside 1e-4 |
| E-43 | PASS | retention CE ↓ 0.7391; fill ↑ 0.6972; mount 0.6564/0.8012 | runs/mex/mu3_g4/final (no gate JSON) | NO_ARTIFACT | numbers prose-only; 0.8012 hardcoded in eval_mu3_e44_cohab.py:97 |
| E-44 | (a)(b) PASS / (c) FAIL | x-CE ↓ 4.2467→3.3229; cohab mark ↑ 0.6459 < 0.78 | runs/mex/mu3_xtower/summary.json | MISCITED | (a)(b) confirmed; (c) 0.6459 console-only -> FAIL(c) unsupported |
| E-45 | ALL PASS | mark ↑ 0.7943; all ↑ 0.6462; CE ↓ 2.2753 | runs/mex/mu3_joint/gates.json | MISCITED | artifact now holds E-54b Muon 0.7941/0.6461/2.2752 (same output path as eval_mu3_dia2_gates.py:131) |
| E-46 | boundary | first-char fill ↑ 7/1/0/0 of 50 | runs/mex/mu3_joint/e46_fill.json | CONFIRMED | match |
| E-47 | router PASS; (c)(d) FAIL | router ↑ 1.0; x2 ↑ 0.36; x3 ↑ 0.7708 | runs/mex/mu3_router/gates.json | CONFIRMED | match (ledger (c)/(d) = artifact x2/x3) |
| E-48a | FAIL a+b | x2 ↑ 0.42; x3 ↑ 0.80 | runs/mex/mu3_router/gates_e48a.json | CONFIRMED | match |
| E-48b | FAIL a+b (MoE readout) | x2 exact 0/50; x4 exact 0/50 ↑ | runs/mex/mu3_router/gates_e48b.json | CONFIRMED | match |
| E-49 | PASS | demo.md (read-only) | runs/mex/mu3_router/demo.md | CONFIRMED | file exists |
| E-50 | FAIL t1/t2; t3 PASS | mixed CE ↓ 1.8757; x2 0/50; x1 0.1121; x3 25/50 | runs/mex/mu3_tall/final/{summary,gates_e50}.json | MISCITED | x2 artifact is [0,0] (zero denominator); x1 0.1121 & x3 25/50 absent (gates_e50 holds E-45 calibration 0.3783/18/20) |
| E-51 | all FAIL | x1 ↑ 0.1191; x2 0/50; x3 29/50; CE ↓ 5.6833 | (none) | NO_ARTIFACT | numbers appear nowhere under runs/ |
| E-52 | all FAIL | B1 ↑ 0.2890; B2 0.0076; x2 6/50; x3 20/50; x2 LoRA 18/50 | (none); runs/mex/mu3_lora/ empty | NO_ARTIFACT | claimed saved runs/mex/mu3_lora/x2 does not exist |
| E-53 | DONE/PASS | documentation rung | runs/mex/mu3_joint/capability_card.md | CONFIRMED | file exists; restates unbacked E-43/E-45/E-54 numbers |
| E-54a | PASS | mark ↑ 0.7940; CE ↓ 2.2752 | runs/mex/dia2_init (no gate JSON) | NO_ARTIFACT | only gates.json (E-54b Muon) survives; w1 in-script only |
| E-54b | m1 FAIL / m2 PASS | mark ↑ Muon 0.7941 < AdamW 0.7987 | runs/mex/dia2{b_adamw,b_muon}/final/train_summary.json + mu3_joint/gates.json | MISCITED | eval-loss 0.74688/0.74674 confirmed; winner AdamW composed 0.7987/0.6504/2.2798 prose-only |
| E-54c | c1 FAIL, c2 PASS, c3 FAIL | mark ↑ 0.7920 < 0.7987; shared-KV 0.5419 | runs/mex/dia2c_ctx192 (train_summary only) | NO_ARTIFACT | all probes console-only; gates_dia2c.json overwritten by E-59 (0.6002/0.7534/4.0083) |
| E-54d | PASS | mark ↑ 0.8134; all ↑ 0.6672; CE ↓ 2.3829 | runs/mex/dia2d_scale/final/train_summary.json | NO_ARTIFACT | best_eval_loss 0.74715 confirmed; gate numbers only restated in capability_card.md |
| E-55 / b / c | honest external loss | mean DER ↓ 0.858 vs gold 0.456 | runs/mex/e55_bench/*.metrics.json | CONFIRMED | per-gate metrics back it; gold mark-F1 mean 0.822 vs artifact 0.830 (MISCITED); nocase parentheticals prose-only |
| E-61 v21a | PASS | in-domain markpos ↑ 0.6824 vs 0.6496 | runs/mex/dia2j_v21_final | UNSUPPORTED | run dir has weights only; 0.6824 absent from runs/ and scratch/ |
| E-61 v21b | PASS | external mark-acc ↑ mean 0.3652 vs 0.207 | scratch/prof_v21/*.metrics.json | CONFIRMED | mean 0.3654 vs 0.3652 (MISCITED 0.0002); per-gate values match |
| E-61 v21c | FAIL | per-class floor 0.00/0.01 | (none) | NO_ARTIFACT | 15-value per-class vector prose-only |
| E-63 (diac) | partial | S4 0.7178/0.9327; full-val 0.93329/0.06671; external regressed | runs/e63_arms.log; runs/e63_finalists_eval.log; runs/diac/stage3best_ls/gate_eval.csv; runs/diac/stage2b2500/gate_eval.csv | CONFIRMED | all trial/full/external numbers match; partial verdict accurate |
| E-63c appendix | "new best on every recorded gate" | DER ↓ champion vs gold | runs/diac/e63_m1_full/gate_eval.csv; gate_probe/best_gate.json | MISCITED | wn2014 0.49070 > gold 0.48994 (claim false); selector mean_der is 3-gate |
| E-63s appendix | "BEST OVERALL = S3 40.77" | tracker mean DER ↓ | runs/diac/e63s_s4_12l160/gate_probe/best_gate.json; gate_eval.csv; runs/e63s_s4.log | MISCITED | S4 artifact best 39.96@16500 < S3 40.77 -> S4 is best; S3-best stale (false positive), S4 row understated |
| report line 91 | raw stage3best_ls 0.6163/0.5978 | DER ↓ | runs/diac/stage3best_ls/gate_eval.csv | MISCITED | artifact sadeed 0.63089, abdou 0.56636 |

Counts by recheck result (36 table rows, E-48b/E-61 counted per gate row):
CONFIRMED 17, MISCITED 9, NO_ARTIFACT 7, UNSUPPORTED 3.
INVERTED (metric direction read backwards): **0** — no E-067/P1-class
direction inversion exists in E-31..E-63.

## Unverifiable items and why

- E-31 sweep gate values (fill acc / retention CE) — no metrics artifact; the
  tfevents carry only loss/runtime tags.
- E-34 diagnostics 0.6896 (composed marks) and 0.5464 (composed non-mark) —
  prose-only.
- E-38a bridges-ON 0.6485/0.7862/0.5698 — overwritten by \`res_off\` in
  \`runs/mex/mu2_e38a/summary.json\` (dict-union at eval_mu2_e38a.py:96).
- E-43 all four gate values — no JSON writer; \`eval_mu2_e43_mounts.py\` prints only.
- E-44 gate (c) 0.6459 — \`eval_mu3_e44_cohab.py\` prints only.
- E-45 0.7943/0.6462/2.2753 — \`runs/mex/mu3_joint/gates.json\` overwritten by the
  E-54b dia2 eval (eval_mu3_dia2_gates.py:131 vs eval_mu3_e45_gates.py:130).
- E-47 gate (a)/(e) "voucher" text and E-48a "clean-trunk head 0.36" reference —
  not persisted.
- E-50 x1 0.1121, x3 25/50, and x2 as "0/50" — artifact says \`[0,0]\`.
- E-51 all numbers — absent from \`runs/\`.
- E-52 all numbers and the claimed \`runs/mex/mu3_lora/x2\` save — absent.
- E-54a w2/w3 and w1 — no persisted gate JSON; w1 in-script assertion only.
- E-54b winning AdamW composed 0.7987/0.6504/2.2798 — prose-only; surviving
  gates.json holds the losing Muon arm.
- E-54c all probes — console-only; \`runs/mex/gates_dia2c.json\` overwritten by E-59.
- E-54d gate numbers — only restated in \`capability_card.md\`.
- E-55 nocase parentheticals (0.7461/0.8639/0.7677/0.7821) and strict DER 0.8191
  — no DER-nocase CSV under \`runs/mex/e55_bench\`.
- E-61 v21a 0.6824 and v21c per-class vector — no artifact.
- E-63 \`runs/e63s_s7_*\` arm directories exist under \`runs/diac/\` with no
  matching verdict rows in the audited ledger and no \`*s7*.log\` names; their
  results are outside this audit's recorded-verdict set.
