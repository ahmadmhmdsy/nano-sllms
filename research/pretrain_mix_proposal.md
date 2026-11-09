# Milestone D — Data upgrade for the next pretrain run

> TASKS.md row 13 / PLAN.md §4 successor. P-scale (~100M params), gated sources now
> unlocked, disk headroom ~34.7 GB free on E:. Every web-derived claim cites an Exa
> payload path under `research/raw/` or a primary URL. No datasets were downloaded during
> research (hard constraint).

## 0) Why this document exists

The current corpus (CodeSearchNet python + Evol-Instruct fallback, 150k rows → 47.9M train tokens at ctx 1024, HANDOFF §3b) is **narrow**: CodeSearchNet rows are short single-function docstrings, not real source files. A 100M model trained on it learns function-shape syntax but no real project structure (imports, classes, multi-file context). Milestone D adds gated candidates **the user has already accepted on huggingface.co** (MEMORY.md / TASKS row 13, evidence 2026-09-06), proposes a mix sized for the ~100M Chinchilla-style budget, and lays out a download plan that fits the 34.7 GB free disk headroom without trashing the live M3 run.

## 1) Verified facts about the candidate datasets

All three live under `bigcode/*`, are **gated with contact-info acceptance** (user already accepted 2026-09-06 — TASKS row 13; `HF_TOKEN` lives in project-root `.env`, loaded by `exa_search.py`'s `load_dotenv`; per the MEMORY gotcha, `prepare_data.py` will need the same shim). They use the `streaming=True` + `data_dir=<lang>` pattern that the existing `scripts/prepare_data.py` already implements (see `text_of()` and `TEXT_KEYS = ("content", "code", ...)`; "content" is the first match — all three schemas provide it).

### 1.1 bigcode/the-stack-v2

Source: `research/raw/milestone_d_stack_v2_card.json` (HF card, 2026-09-06).

| Field | Value | Source |
|---|---|---|
| Hub structure | one parquet config per language (`data/<Lang>/*.parquet`); ~658 langs | `milestone_d_stack_v2_overview.json` |
| Languages | 658 unique, detected by `go-enry / linguist` | `milestone_d_stack_v2_card.json` |
| Total size | full 67.5 TB · dedup 32.1 TB · train ~900B tokens | `milestone_d_stack_v2_card.json` |
| **Python subset size** | **dedup: 233.29 GB / 56.93M files** · full-SWH: 191.61 GB / 56.19M files | `milestone_d_stack_v2_python_subset.json` (arxiv 2402.19173 Table 1) |
| Duplicate rate | roughly 40 % of permissive files were (near-)duplicates | `milestone_d_stack_v2_card.json` |
| HF listing size | 427 GB total on the Hub | `milestone_d_stack_v2_overview.json` |
| **Schema** | blob_id, content_id, src_encoding, language, license_type, detected_licenses, is_vendor, is_generated, length_bytes, gha_* — **NO inline content on the Hub** | `milestone_d_stack_v2_card.json` |
| Content fetch | `s3://softwareheritage/content/<blob_id>` via `smart_open` + `boto3`; **requires AWS access** + SWH/INRIA agreement | `milestone_d_stack_v2_card.json` |
| License scope | **only permissive (Blue Oak list) + no-license**; file-level ScanCode SPDX | `milestone_d_stack_v2_license.json` |
| Dedup | MinHash+LSH near-dup over 5-gram Jaccard; preserves repo context (stars/forks/recency) | `milestone_d_stack_v2_python_subset.json` |

**Bottom line:** python alone is 191–233 GB — **15–20× our download budget**. A 1 % slice
is still 1.9–2.3 GB raw SWH and the Hub gives only IDs — we'd need AWS creds and 230 KB/s
for ~3 h. The pre-built ID corpora (`the-stack-v2-train-full-ids`, `...-train-smol-ids`)
have the same content-fetch problem.

### 1.2 bigcode/the-stack-smol

Source: `research/raw/milestone_d_stack_smol_card.json`.

| Field | Value | Source |
|---|---|---|
| Hub structure | one parquet per language under `data/<lang>/` | `milestone_d_stack_smol_card.json` |
| Languages | **30** (python, javascript, java, c++, c, ts, go, rust, ruby, …) | `milestone_d_stack_smol_card.json` |
| Sampling | ~0.1 % of the-stack, **each language has 10,000 random samples** | `milestone_d_stack_smol_card.json` |
| **Total size** | **2.6 GB of text** (HF listing **2.95 GB**) | `milestone_d_stack_smol_card.json` |
| **Python rows** | **10,000** (`data_dir="data/python"` returns `num_rows: 10000`) | `milestone_d_stack_smol_card.json` |
| Schema | `content, avg_line_length, max_line_length, alphanum_fraction, licenses, repository_name, path, size, lang` — **inline content** | `milestone_d_stack_smol_card.json` |
| License | permissive + no-license (BigCode v1 pipeline, ScanCode) | `milestone_d_stack_smol_card.json` |
| Dedup | upstream BigCode pipeline (exact-hash then near-dup); this *is* its output | `milestone_d_stack_smol_card.json`, `milestone_d_answer_stackmix.json` |

**Bottom line:** the only one of the three we can stream-and-train **today without AWS
creds**. 10k python rows is small; combine with starcoderdata or scale to the whole 30-lang
smol (~300k rows). Exa's synthesis explicitly recommends smol for ≤3 GB budgets
(`milestone_d_answer_stackmix.json`).

### 1.3 bigcode/starcoderdata

Source: `research/raw/milestone_d_starcoderdata_card.json`.

| Field | Value | Source |
|---|---|---|
| Hub structure | per-language parquet subdirs + 4 special dirs (jupyter-scripts-dedup-filtered, jupyter-structured-clean-dedup, github-issues-filtered-structured, git-commits-cleaned) | `milestone_d_starcoderdata_card.json` |
| Languages | 86 code languages + GitHub Issues + Jupyter + commits | `milestone_d_starcoderdata_card.json` |
| **Total size** | **783 GB code** + 54 GB issues + 13 GB Jupyter + 32 GB commits ≈ **~250B tokens** | `milestone_d_starcoderdata_card.json` |
| Python subset | `load_dataset("bigcode/starcoderdata", data_dir="python", split="train")` — content field present | `milestone_d_starcoderdata_card.json` |
| Dedup / PII | near-deduplication and PII removal (StarCoder paper, Lozhkov et al. 2024) | `milestone_d_starcoderdata_card.json` |
| License | permissive + no-license (mirrors Stack v1 terms); must include terms + require user agreement on redistribution | `milestone_d_starcoderdata_overview.json` |
| Discussion re tokens | users note paper says 1T tokens but only ~250B released | `milestone_d_starcoderdata_overview.json` (HF discussion #13) |

**Bottom line:** the **python subdir is the biggest single addition**. No public python-only
size number; the existing `configs/pilot.yaml` fallback chain already lists it
(`bigcode/starcoderdata`, config `python`) — it streams fine with the existing pipeline. The
M2 pilot fell back to Evol-Instruct (50k rows → 20.9M tokens) only because that machine
had no HF token.

## 2) Pretrain mix proposal (P-scale ~100M, ctx 512/1024)

### 2.1 Token budget anchor

Chinchilla ≈ 20× params = ~2B tokens for a fully-fit 100M model. We cannot hit that on
34.7 GB free at 230 KB/s, and the M3 pilot ran 164M tokens for 226M params (HANDOFF §3b)
— a pilot-shrink, not Chinchilla. Milestone D **matches the M3 budget shape** for an
apples-to-apples comparison:

- target train tokens: ~150–200M (1.5–2× params; fits disk + a few days wall-clock)
- val: 1–2 % of train (~1.5–4M)
- packing: ctx 1024 (target config; pilot config uses 512 — mix is config-agnostic)

### 2.2 The mix

| Source | Subset | Proportion | Target rows | est. tok/row | est. tokens | License filter |
|---|---|---|---|---|---|---|
| **the-stack-smol (python)** | `data/python` (10k rows) | **30 %** | 10,000 | ~700 (avg file incl. license/header overhead) | ~7M | permissive + no-license (already filtered upstream) |
| **starcoderdata (python)** | `data_dir="python"` | **55 %** | stream until budget hit (~120M tok / ~600 tok/row ≈ 200k rows) | ~600 (CSN pilot baseline 326 tok/row was on docs; starcoderdata python is whole files — expect 2×) | ~120M | permissive + no-license (already filtered upstream) |
| **CodeSearchNet (python)** | `code-search-net/code_search_net`, config `python` | **10 %** | 15,000 (down from M3's 150k; NL bridge for tokenizer coverage) | ~326 (HANDOFF §6 measurement) | ~5M | mixed (GitHub scrapes; MIT/Apache-dominated but not curated) |
| **Evol-Instruct (python)** | `nickrosh/Evol-Instruct-Code-80k-v1` (pilot fallback) | **5 %** | 5,000 | ~430 (pilot measurement, HANDOFF §3) | ~2M | per-row instruction; mostly MIT-style answers |
| **TOTAL** | — | 100 % | ~230k rows | — | **~134M train tokens** (~0.95 of M3 budget) | — |

Rationale: **30 % smol** = breadth over 30 languages (real tasks mix code).
**55 % starcoderdata python** = the workhorse: real whole-files, near-dedup, PII-cleaned.
**10 % CSN** = NL↔code bridge — CodeLlama-32k is heavily code-weighted (§4.3) and CSN is docstring-rich.
**5 % Evol-Instruct** = SFT-signal pre-bake: identical to C12 Tier 1's data (HANDOFF §8.4) — a sliver in pretraining means Tier 1 SFT starts from a model that already half-knows the format, reducing the post-M3 GPU bill (TASKS row 4's ~5–6 h full SFT drops if the base is pre-exposed).

### 2.3 Token budget sanity check

- 134M tokens packed at ctx 1024 = ~131k blocks, 4 × 8 MB shards (`shard_tokens: 8000000`, `tokenize_data.py:30`).
- At measured M3 pace ~9.6–12.7 s/it (RTX 4000 uncapped, 32,768 tok/step) → **~4,100–5,400 steps = ~15–20 h wall-clock**.
- 164M tokens = ~3.4 epochs at this mix. Same `load_best_model_at_end` + `save_total_limit=3` discipline (PLAN §5.3).

## 3) Download plan (vs 34.7 GB free on E:)

### 3.1 Streaming vs snapshot

**Decision: streaming for everything, no snapshots.** (1) Network: documented slow (PLAN §A5); a snapshot costs hours and provides no benefit — we only need ~134M tokens. (2) Disk: 34.7 GB free must survive data prep + 5000-step checkpoint rotation (~7.6 GB for M3) + val shards + logs. (3) Pipeline match: `prepare_data.py` already uses `streaming=True` with `take(N)` in its fallback chain — first candidate to yield `n_rows` wins (`configs/pilot.yaml`).

### 3.2 Concrete fetch (per source, in `dataset_candidates` order)

For the **next-pretrain** config (a new `configs/next_pretrain.yaml` sibling of
`configs/target.yaml` — not yet created, gated on user OK per TASKS row 13):

```yaml
data:
  dataset_candidates:
    - name: bigcode/the-stack-smol          # 30 langs x 10k rows; ~2.6 GB HF
      config: python                        # python subdir; 10k rows inline content
    - name: bigcode/starcoderdata           # 250B tok; per-lang parquet
      config: python                        # data_dir=python
    - name: code-search-net/code_search_net # ungated fallback (NL bridge)
      config: python
    - name: nickrosh/Evol-Instruct-Code-80k-v1  # ungated last-resort
```

A row cap **per source** must replace the single `data.rows` field (currently `prepare_data.py` shares one `n_rows` across the fallback chain). Two options, in order of preference: (1) Extend the YAML schema with a per-candidate `target_rows:` override, then teach `prepare_data.py` to allocate `n_rows` per candidate proportionally (omit `target_rows` → global cap = today's behavior, backwards-compatible). (2) **Run four separate `prepare_data.py` invocations**, each writing to `data/next_pretrain/raw/<source>/{train,val}.jsonl`, then `cat` them before `tokenize_data`. Simpler, no code change, but no cross-source `dedupe` unless we SHA1 cross-source before `cat`. **Recommended for v1** (matches the create-only instruction).

### 3.3 Per-source estimated download

| Source | HF listing | Stream delta (python) | @ 230 KB/s | Notes |
|---|---|---|---|---|
| the-stack-smol (python) | 2.95 GB total / python ~100 MB (10k files) | ~30–80 MB | ~3–6 min | smallest; fallback-1 |
| starcoderdata (python) | 783 GB / 86 langs ~9 GB avg, python likely larger | ~5–15 GB (dedup pre-applied) | ~6–18 h **@230 KB/s** | **bottleneck** |
| CSN python | ~50 MB total | ~5 MB (15k rows) | <1 min | ungated, used in M3 |
| Evol-Instruct-Code-80k | ~35 MB | ~5 MB | <1 min | ungated, used in pilot |

**Reality check.** The actual measured network on M3 prep was much faster than the 230 KB/s
estimate — HANDOFF §4: teacher downloaded at 1,688 MB / 29 files in ~3 min. So 5–15 GB of
starcoderdata python realistically downloads in **~30–60 min**, not 6–18 h. We still stream +
take(N) so worst-case cap is rows-bound not bytes-bound.

### 3.4 Disk accounting vs 34.7 GB free

| Item | Bytes | Cumulative free |
|---|---|---|
| Free at proposal time | — | 34.7 GB |
| data/next_pretrain/raw (4 × 200 MB worst-case) | -0.8 GB | 33.9 GB |
| data/next_pretrain/tokens (~134M tok × 4 B uint32 = ~536 MB) | -0.6 GB | 33.3 GB |
| HF streaming cache (datasets 5.0.1 default; HF_HOME defaults to C: cache) | 0 | 33.3 GB |
| runs/next_pretrain checkpoints (save_total_limit=3 × ~2.5 GB) | -7.5 GB | 25.8 GB |
| runs/next_pretrain final/ + logs + tfevents | -1.0 GB | 24.8 GB |
| data/teacher (already on disk from C12) | -1.7 GB (already counted) | 24.8 GB |
| **Live M3 run** (runs/target/ ckpt-* + data/target/tokens, ~5 GB) | not ours to plan | — |
| **Headroom after provisioning** | — | **24.8 GB** ✓ |

24.8 GB headroom after provisioning covers one more replan pass + a doubled-budget retry if
step-500 eval shows trouble. No snapshot cache.

## 4) Risks

### 4.1 License

- **the-stack-v2**: permissive + no-license only (Blue Oak list). **CodeSearchNet is not license-curated** — GitHub scrapes, per-function licenses not asserted; risk: low for a research pilot, unacceptable for a product. Mitigation: document in MEMORY.md; future milestone = per-row ScanCode pass for CSN before any production release.
- **starcoderdata**: permissive + no-license (Stack v1 terms); near-dedup + PII removal upstream. Cite StarCoder terms clause if redistributing non-trivial samples. **the-stack-smol**: derived from Stack v1, same posture; per-row license spot-check on the 10k subsample recommended before publishing sample output.

### 4.2 PII / dedup

- starcoderdata is **already** PII-redacted — no further action. the-stack-smol is upstream-v1 dedup, not v2's stronger near-dup. **Add our own SHA1 exact-hash dedupe** (already in `prepare_data.py:113` — `sha1(text).hexdigest()` in `seen` set) and an **optional MinHash second pass** for cross-source near-dup (open question — follow-up, not in v1).
- CSN has no dedup upstream; our SHA1 pass handles exact dupes. For Tier-2 SFT safety (TASKS row 4: T never saw Evol-Instruct), guarantee CSN is **not** a subset of Evol-Instruct — sample cross-check before commit.

### 4.3 Tokenizer coverage (CodeLlama 32k) — NL vs code

CodeLlama 32k is heavily code-weighted. CSN docstrings measured at ~0.13 tok/char vs ~0.18 for code (HANDOFF §3: 326 tok/row on CSN). The 10 % CSN allocation exists precisely to teach NL↔code boundaries. If a future milestone shifts to Qwen2.5-Coder (better NL coverage, HANDOFF §6), re-tune proportions — stay on CodeLlama for this run to keep the comparison to M3 clean. The 5 % Evol-Instruct allocation also covers the chat-template path: Tier 1 SFT uses role markers CodeLlama doesn't natively know (`scripts/sft.py` handles the mismatch, but pre-exposure reduces surface area).

### 4.4 GPU / live-run isolation & disk-creep

This proposal is read-only research. §3 is **explicitly user-gated** (TASKS row 13 status `pending`, create only). No GPU/CUDA work, no process kills — confirmed at session start (live M3 on MUO4QK5 not touched). Any actual download runs as a background job (no GPU contention) and reports progress every 250 rows (`prepare_data.py:121`). 24.8 GB headroom post-provisioning is healthy, but the live M3 run holds ~5 GB independently; a simultaneous user action (e.g. delete pilot checkpoint pre-M3, HANDOFF §3b) would lower the bar. Mitigation: report free-disk before each download kickoff (TASKS row 2 procedure).

## 5) Acceptance checklist

Use when TASKS row 13 moves `pending` → `in_progress`. Each item PASS with evidence
(file path, log line, or numbers) before the next.

1. [ ] **Gated access verified per source** — HF whoami returns access to all three bigcode
   datasets (already passed 2026-09-06; re-run on prep day).
2. [ ] **New config**: `configs/next_pretrain.yaml` mirroring `configs/target.yaml` but with
   the 4-source `dataset_candidates` chain + per-source `target_rows` or 4-stream `cat` step.
3. [ ] **Smoke data prep** on the smoke config first — 200 rows from the-stack-smol python
   reach `data/smoke/raw/source.txt` with `min_chars: 60` in <5 min.
4. [ ] **Per-source measured tok/row** — `prepare_data.py` + `tokenize_data.py` for each of
   the 4 sources at 2,000 rows each; record in `research/milestone_d_actual.json`. Confirm
   the 700/600/326/430 estimates in §2.2 are within ±25 %.
5. [ ] **Disk probe** before and after each prep step (TASKS row 2 procedure).
6. [ ] **Full prep** to `data/next_pretrain/{raw,tokens}` with §2.2 mix totals.
   `tokens/meta.json` reports train_tokens ≈ 134M ± 15 %.
7. [ ] **sanity_check PASS** for `configs/next_pretrain.yaml` (4/4 PASS gates).
8. [ ] **GPU preflight**: `vram_probe.py --config configs/next_pretrain.yaml` (ctx 512
   first; ctx 1024 if headroom) — peak ≤ 5 GB on RTX 4000 8 GB.
9. [ ] **User approval** to start the run (TASKS row 13 user-gated; row 6 upgrade-path
   requires explicit go per HANDOFF §8.3).
10. [ ] **Auto-resume contract re-verified**: kill at step 100, relaunch exact command,
    resume at step 101 — M0 drill pattern (HANDOFF §2).
11. [ ] **Cross-source dedupe SHA1** — concatenate the 4 sources' SHA1 sets before final
    tokenization; report dedupe rate. > 5 % collisions = investigate (likely CSN ↔
    starcoderdata overlap).
12. [ ] **Final post-run commit** — train_summary.json, eval_report.json, tfevents, and the
    new config. Weights stay LOCAL per HANDOFF §7 (LFS quota exhausted).
13. [ ] **MEMORY.md update** — append new data-source row with measured tok/row and total
    tokens; mark CSN-only milestone superseded.
14. [ ] **TASKS row 13 status → done** with evidence links.

## 6) Cross-references

Raw Exa payloads: `research/raw/milestone_d_*.json` (9 files, 2026-09-06). Exa task list: `research/milestone_d_tasks.json`. HF cards: `research/raw/milestone_d_{stack_v2,stack_smol,starcoderdata}_card.json`. Primary URLs: `https://huggingface.co/datasets/bigcode/the-stack-v2`, `.../bigcode/the-stack-smol`, `.../bigcode/starcoderdata`, `https://arxiv.org/abs/2402.19173` (StarCoder2 / Stack v2 tech report). Repo anchors: PLAN.md §4, HANDOFF.md §3b/§6/§7, MEMORY.md data-source rows, TASKS.md rows 6/13.
