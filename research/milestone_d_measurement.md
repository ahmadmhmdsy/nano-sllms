# Milestone D measurement — actuals, rev 2 (2026-09-09)

TASKS row 13 · spec research/pretrain_mix_proposal.md §5 items 4 + 11.
Rev 2: starcoderdata REMEASURED after the user accepted its Hub terms mid-session
(rev 1 recorded FAIL/unmeasured; gate-incident audit in §3). All four sources now
measured at 2,000 rows each. Method: per-source legacy-path scratch configs
(rows 2000, NO target_rows) under data/md_measure/<tag>/ → scripts/prepare_data.py
→ scripts/tokenize_data.py (CodeLlama-32k, min_chars 200, SHA1 dedupe, vf 0.02).
Full numbers + sizing arithmetic JSON: research/milestone_d_actual.json.

## 1) Per-source measurement (2,000 rows requested each)

| Source | Access form | requested | rows_seen | kept | too_short | dups (within) | tok/row | median | p90 | max | sample tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| the-stack-smol (python) | data_dir: data/python | 2000 | 2091 | 2000 | 91 | 0 | **2717.2** | 929 | 5609 | 121,723 | 5,434,305 |
| starcoderdata (python) | data_dir: python | 2000 | 2111 | 2000 | 111 | 0 | **2345.7** | 921 | 5317 | 146,310 | 4,691,480 |
| code_search_net (python) | config: python | 2000 | 2087 | 2000 | 87 | 0 | **271.8** | 172 | 505 | 7,341 | 543,524 |
| Evol-Instruct-Code-80k-v1 | (no config) | 2000 | 2165 | 2000 | 165 | 0 | **474.4** | 417 | 845 | 2,272 | 948,877 |

tok/row = packed tokens (CodeLlama, bos+eos per row) ÷ kept rows. tok/char:
smol 0.3264 · starcoder 0.3260 · csn 0.2977 · evol 0.2916.
Distribution shape: smol and starcoderdata are BOTH whole-file sources with
near-identical medians (929 vs 921) and heavy tails (>120k tok max); the 600 tok/row
estimate for starcoderdata was ~4x low for the same reason as the 700 for smol.
kept/seen = 0.958 (smol) · 0.947 (starcoder) · 0.958 (csn) · 0.924 (evol) at
min_chars 200 → request ~5% more rows than the kept target.

### vs proposal §2.2 estimates (±25% gate)

| Source | est | measured | delta | flag |
|---|---:|---:|---:|---|
| the-stack-smol | ~700 | 2717.2 | **+288%** | **FAIL** |
| starcoderdata | ~600 | 2345.7 | **+291%** | **FAIL** |
| CSN | ~326 | 271.8 | −16.6% | PASS (consistent with the M3 ~319 tok/row) |
| Evol | ~430 | 474.4 | +10.3% | PASS |

## 2) Sizing arithmetic (USER DECISION pending — run-gated)

Measured tok/row: smol 2717.2 · starcoder 2345.7 · csn 271.8 · evol 474.4.

**As-configured target_rows (10000/200000/15000/5000) = 502,761,000 tokens =
3.75x the 134M budget** (starcoder alone 469.1M = 93.3% token share). The
configured rows CANNOT be fetched as-is without busting the budget 3.75x.

| Option | starcoder rows | mix shape (tok) | total tokens | steps @32,768 |
|---|---:|---|---:|---:|
| A — token-proportional 30/55/10/5 at 134M, smol capped at 10k rows (27.2M = 20.3%) | **35,778** | smol 27.17M · starcoder 83.92M · csn 15.26M · evol 7.63M (rows 10k/35,778/56,150/16,091) | **133.99M** | 4,090 |
| B — keep §2.2 rows except starcoder, resize starcoder to fill 134M | **42,795** | smol 27.17M · starcoder 100.38M (74.9%) · csn 4.08M · evol 2.37M (rows 10k/42,795/15k/5k) | **134.01M** | 4,090 |
| C — pure 55% share for starcoder (73.7M) | 31,411 | total only 107.32M (under budget; smol share then over 25%) | 107.30M | 3,275 |

Option A derivation: smol physically caps at 10,000 python rows (27,172,000 tok);
remaining 106,828,000 split 55:10:5 → starcoder 83,936,000/2345.7 = 35,778 rows;
csn 15,261,000/271.8 = 56,150; evol 7,631,000/474.4 = 16,091.
Option B derivation: (134,000,000 − 33,621,000)/2345.7 = 42,795 rows.

## 3) Gate incident audit (starcoderdata)

- 2026-09-09 ~18:25 UTC: content reads 403 GatedRepoError ("restricted and you
  are not in the authorized list") despite a VALID HF_TOKEN (whoami → ahmadsy);
  metadata was public (dataset_info gated=auto, 865 siblings; ls → 59 parquet
  files in python/; stat train-00000 = 399,133,539 B). 2/2 retries used, recorded
  FAIL in rev 1. The 2026-09-06 "verified unlockable" evidence had covered
  the-stack-smol only.
- ~19:00 UTC: user accepted the terms on the Hub page; re-probe returned a live
  row (keys content/id/max_stars_count/max_stars_repo_name/max_stars_repo_path);
  the full 2,000-row prepare then completed in under 2 minutes (EXIT=0,
  1960 train + 40 val; log data/md_measure/starcoderdata/prepare.log).
- LESSON: gated=auto grants only AFTER the per-repo accept click — a valid token
  proves identity, not access; run a 1-row content probe per source on prep day.
- ALSO FOUND (both probes): bigcode/starcoderdata has NO "python" builder config
  (only default) — same as the-stack-smol. The working form is data_dir: python;
  next_pretrain.yaml now uses it for BOTH sources (config: python in the proposal
  §3.2 recipe and in pilot.yaml/pilot_b8.yaml is dead for these two repos).

## 4) Cross-source exact-dup (SHA1, proposal §5 item 11) — all four sources

- 8,000 rows scanned (4 x 2k): **0 exact dups** — within-source and all six
  pairwise intersections 0 (first-wins global pass 0). Overall measured rate: 0.0000.
- The §5.11 CSN↔starcoderdata overlap risk was NOT OBSERVED at 2k-row scale
  (0 intersections); the shared-SHA1 machinery in mix mode re-checks at full scale.

## 5) Wiring evidence (backward compat + mix mode)

| Check | Result |
|---|---|
| Legacy 5-row regression, PRE vs POST edit | BYTE-IDENTICAL outputs (sha256 source.txt 06EEF883…, train.jsonl 6D7AD77E…, val.jsonl 6625EB0F…); only additive source_stats.json |
| Legacy fallthrough (fake dataset first) | falls to TinyCode, 4+1 rows, exit 0; both candidates in stats |
| Mix, zero-network (2 local candidates, same file) | shared SHA1 set proven: source 2 dropped 3 cross-dups, kept only the 4th distinct row; totals kept 4 / seen 11 / dups 5 / rate 0.454545 (hand-computed match) |
| Mix refusal guard | partial target_rows → SystemExit "requires target_rows on EVERY entry" (exit 1) |
| Mix live (4 real candidates, tiny targets, pre-acceptance) | smol 29+1 · starcoder 403 recorded, run continued (exit 0) · csn 29+1 · evol 19+1; 80 kept / 86 seen / 0 dups |
| next_pretrain.yaml validator (CPU) | ALL PASS incl. data_dir forms for smol+starcoder; dims == pilot @ctx1024; 100.7M params; 4100×32,768 = 134,348,800; save%eval == 0; adamw_bnb_8bit |

## 6) Disk + method notes

- Free disk before/after: E: 36.35 → 36.30 GB, C: 31.21 → 31.13 GB (HF cache on C:).
- Streaming + take(2000) only; nothing cached beyond the HF default. Total network
  well under the 30-min cap; the starcoderdata pull needed no 500-row fallback.
- Proposal §5 item 3 (200-row smoke-config prep) NOT RUN as written: it would
  overwrite the existing data/smoke/raw/ artifacts; the same path is proven by
  the measurement runs above.
- Scratch retained for audit: data/md_measure/ (configs + raw + tokens + stats +
  prepare.log) and data/legacy_check/ (regression outputs + prepare_data_milestone_d.diff).
- next_pretrain.yaml now carries a measured-values warning at its starcoderdata
  target_rows: 200,000 rows = 469.1M tokens (3.75x budget) — sizing is a user
  decision (§2 options) before the real prep.
