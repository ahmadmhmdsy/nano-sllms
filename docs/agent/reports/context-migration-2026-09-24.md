# Context migration report - Task 7 (2026-09-24)

Scope: repair the task/experiment **views** by migrating records from the
historical legacy ledgers (TASKS.md, research/EXPERIMENTS.md) into the
canonical registries (state/tasks.yaml, research/experiments.yaml), via an
evidence-backed, conservative audit. This report is the migration audit; the
legacy ledgers were given an authority banner pointing here, and their rows
were NOT mass-rewritten. No source code, tests, GPU work, network calls, or
process inspection was performed. Validation labels are the repository
vocabulary: PASS / FAIL / SKIPPED / BLOCKED / NEEDS_USER_DECISION.

**Authority statement.** TASKS.md and research/EXPERIMENTS.md are historical
ledgers from this point on. Current truth: state/tasks.yaml,
research/experiments.yaml, and the generated CURRENT_STATE.md /
state/current.json (rendered only by scripts/render_current_state.py). The
banner edits are visible at the top of both ledger files.

Line references below are **pre-migration** references (HEAD 5f75682).
Post-banner TASKS.md line numbers shift by the banner insert (+24);
EXPERIMENTS.md shifts by its own banner insert (+10 approx).

## 1. Duplicate legacy task row IDs (TASKS.md)

| row id | occurrences (pre-migration lines) | contents | treatment |
|---|---|---|---|
| 61 | 58, 122 | placeholder row; E-19 external-model bake-off (done) | historical exception |
| 62 | 59, 73 | E-20 cache research (done); Tier-A cleanup DATA LOSS (blocked) | row 73 data-loss migrated to core/task-tier-a-data-loss; E-20 stays historical |
| 63 | 14, 60, 74, 181 | E-62 shared-KV smoke task; E-21 micro arms (done); TBD Tier-A remaining (pending); E-63 feature ladder (done) | row 74 folded into core/task-tier-a-data-loss; E-63 covered by canonical diacritizer/task-e63 / diacritizer/E-063; E-62 left historical (closed in EXPERIMENTS.md; no duplicate canonical record created) |
| 64 | 61, 183 | E-22 micro arm C (done); Registry S1 monitor (done) | both folded into diacritizer/task-registry-s2s4 (S1 closed inside the migrated record; ledger done rows historical) |
| 65 | 62, 184 | E-23 Z-Mahmood leverage plan (done); Registry S2-S4 (in progress) | folded into diacritizer/task-registry-s2s4 |
| 66 | 63, 185 | E-23a gold+v3q SFT (done); Registry S5-S6 (in progress) | in-progress registry work migrated to diacritizer/task-registry-s5s6; E-23a stays historical |
| 67 | 64, 186 | E-23e cross-agreement validator (done); E-63 remediation averaging (done) | historical exceptions (both closed, no migration) |
| 68 | 65, 187 | E-23b classical S2 (done); M1/M2/M3 smaller-model ladder (in progress, called running, pwsh-105) | ladder row left historical: running-process claim unverifiable this session, no closure artifact found |
| 70 | 72 (vs 75 at 71) | superseded-closure duplicate of row 75 (same E-23d verdict) | historical exception |
| 49 | 107, 108 | full duplicate row (in-progress arms on 2026-09-11 vs pending user-gated) | conflicting stale claims kept as history; nothing migrated |
| 47 | 109, 110 | KT Route-0 ladder row (pending); interactive architecture visual (done) | historical exceptions |
| 40 | 99, 100 | Web UI U13 Model tab (done); Track A ctx-extension (done) | historical exceptions |
| 97 | inside malformed row 172 and again at 174 | U-line U-1 corpus build vs DA-8 title ranker | historical exceptions |
| 91 | 166 vs 172 | DA-3..9 rungs (pending) vs dia2 rung pair (done) | historical exceptions |

The row-number space overlap across sections (63/64/65/66/67/68 reused
between the D-line rows and the 2026-09-23 Registry rows) is the root
cause: the legacy one-row-per-task scheme has no stable IDs. The namespaced
canonical registry repairs exactly that.

## 2. Duplicate legacy experiment IDs (research/EXPERIMENTS.md)

| ID | occurrences (approx., pre-migration) | conflict | treatment |
|---|---|---|---|
| E-13 | row 31 (stem, truncated mid-cell) vs full row 36 | truncated malformed duplicate vs the full depth-ablation record | full row 36 is the historical record; row 31 stem is a malformed duplicate. Historical; not migrated (2026-09-13 D-line) |
| E-41 | 231 (Muon vs AdamW nano scratch bench), 295 (Muon lever applied to DA-3), section E-41 PRE-REGISTERED mu3-A x-task bridge tower | three distinct experiments share E-41 | historical exception; nothing renamed, no namespace invented |
| E-42 | 282 (Sinkhorn-balanced expansion) vs section E-42 PRE-REGISTERED mu3-B Net2Net widen x2 | two distinct experiments | historical exception |
| E-43 | 289 AND 290 (two near-identical DA-1 rows) | duplicate row | historical exception |
| E-44 | 291/292 (DA-2 Emo-analogue; one truncated row) vs sections E-44 PRE-REGISTERED mu3 x-tower remount and E-44 closed DA-2 | mu3-line E-44 collides with DA-2-line E-44 | historical exception |
| E-47 | 293 (DA-2b row) vs closed DA-2b section and mu3 task-router pre-register section | collision | historical exception |
| E-52 | 294 (DA-3 Gist-analogue) vs mu3 trunk-side growth pre-register section | collision | historical exception |
| E-53 | 295 (Muon lever on DA-3) vs documentation-rung section | collision | historical exception |
| E-57 | 299 AND 303 (two DA-9 Schemer rows) plus per-E-57-pattern citations | duplicate row | historical exception |
| E-58a | 302: row with ellipsis placeholder columns, CLOSED PASS | malformed row | historical exception |
| E-61 | rows 850-852 (teacher / launch1 harness artifact / v21 IN PROGRESS) plus pre-register section ~858 and CLOSED section ~873 (post-banner lines) | uniquely scoped once pre-registers landed; the internal profile rows collapse under one closed section | the single closed V2-1 KD arm record is unambiguous and current for the dia2 line, so it migrated as diacritizer/E-061; the other E-collisions were NOT migrated |
| date-only rows | 40-48 (Milestone B/C/D, SFT v2, Tier-3 KD, Track A/H, H2 verdict menu, project-wide analysis) | id column holds a date, not an ID (malformed schema) | historical exceptions preserved verbatim; not migrated |

## 3. Malformed rows and broken links flagged by the linter (known exceptions, expected to persist)

- TASKS.md:36 (pre-migration; post-banner TASKS.md:60) - corrupted row 21 text, an empty-evidence-cell artifact; not repaired here.
- research/EXPERIMENTS.md:6 - broken link (the experiment-template link).
- research/EXPERIMENTS.md:9 - broken link (the WHAT_WORKS link).
These linter findings were reviewed and left in place as historical ledger
damage; repairing them is a separate, user-gated ledger-rewrite scope.

## 4. Stale live-process / status claims superseded by later evidence

| claim (pre-migration ref) | later artifact that supersedes it |
|---|---|
| TASKS.md:14 (row 63): 60-step smoke running, close row later | research/EXPERIMENTS.md E-62 row (~37 pre-banner) records SMOKE PASS closed with warm load 255/27/0 and loss/val curve values |
| TASKS.md:181 (row 63): stage3best_ls smoke PASS, in flight | research/EXPERIMENTS.md E-63 CLOSED (2026-09-23) section records the closed profile (~893 pre-banner) |
| research/EXPERIMENTS.md table row: v21 in progress (~852) | the CLOSED section (~873 pre-banner) records the closed mixed verdict with citations |
| E-60a/b pre-registers marked awaiting user go (~872-879) | the attempt-verdict sections record Rung A attempt-3 + Rung A CLOSED and B1/B1' honest-FAIL structural |
| TASKS.md:67 (row 71): full run launched | TASKS.md:68 (row 72) records the labeling DONE, 142,370 paras with packing done |
| TASKS.md:69 (row 73): job running | TASKS.md:70 (row 74) records the e23c_zm run done with gates and a NOT-adopted verdict |

Items deliberately left UNKNOWN / NEEDS_USER_DECISION (no current status
inferred from stale prose):

- TASKS.md:187 (row 68): bench bench called running, pwsh-105. No closing
  artifact found this session; the row was left untouched.
- All pwsh-N / pid / UI-restarted mentions are dated narrative, never read
  as current liveness; current live_state is deliberately UNKNOWN in
  state/current.json (machine, VRAM, and process deliberately UNKNOWN).

## 5. Conflicting machine/environment claims (historical)

- TASKS.md:15 (row 1): M3 COMPLETED 2026-09-07 on TU09FBO, while the
  narrative bullet TASKS.md:93-94 documents an M3 restart on MUO4QK5 with
  the earlier TU09FBO handoff bundle never arriving (user call: prior
  partial run abandoned). The completed run and the abandoned partial run
  conflict as side-by-side machine claims; both are kept as dated history.
- TASKS.md:47 (row 31): Move-back-to-MUO4QK5 note, RTX 4000 8 GB driver
  595.97, side by side with 6 GB-card and 5.8 GB WDDM-GPU references
  (research/EXPERIMENTS.md E-19 row) and MEMORY 31/32 sysmem-fallback
  notes on the 6 GB TU09FBO card. These are dated facts from different
  sessions on genuinely different hardware; not read as current state.
  Current live_state machine/VRAM is deliberately UNKNOWN in
  state/current.json.

## 6. Migrated records (old IDs and namespaced aliases)

Five task records (state/tasks.yaml), one experiment record
(research/experiments.yaml). Each: one status, one next_action, exact
evidence paths (all verified to exist), owner and last_verified at UNKNOWN
(unverified), and user-gated items carried as pending or blocked plus
explicit user wording, never an unsupported status.

| canonical id | aliases | status | next_action | evidence (exists=PASS) |
|---|---|---|---|---|
| diacritizer/task-registry-s2s4 | 64, 65 | in_progress | finish the S4 external-gates subprocess refactor | plans doc + TASKS.md |
| diacritizer/task-registry-s5s6 | 66 | in_progress | refactor muon and losses into features/ | plans doc + TASKS.md |
| diacritizer/task-dia-ladder-plan | 35b, E-60 | pending (launch user-gated) | await explicit user go | 2026-09-21 dia-ladder plan + EXPERIMENTS.md |
| mex/task-mu1-arm-b | 77-B, 77 | pending (user-gated MoE block decision) | obtain user decision on the new MoE block | MU1_REPORT.md + EXPERIMENTS.md |
| core/task-tier-a-data-loss | 62, 63 | blocked (user must decide recovery) | user incident-recovery decision | checkpoint-review REVIEW.md + tierA-cleanup final_a60 config.json + TASKS.md |
| diacritizer/E-061 | E-61, 061 | closed (experiment) | none; V2-2 follow-up user-gated, outside this record | EXPERIMENTS.md + runs/mex/dia2j_v21_final + scratch/prof_v21 + runs/diac/stage2b2500/gate_probe + configs/dia2j_v21_kd.yaml |

Evidence-path existence for the NEW records: every path above PASS
(verified 2026-09-24; exact linter output recorded in the Task 7
implementation report).

**Alias-not-globally-unique caveat (review finding, added 2026-09-24).**
Because the legacy row/ID collisions are deliberately preserved, legacy
aliases are NOT globally unique across the canonical registries. In
particular, the legacy alias `63` is contextually ambiguous across THREE
records that carry it: core/task-tier-a-data-loss (aliases 62, 63),
diacritizer/task-e63 (aliases E-63, 63), and diacritizer/E-063 (aliases
E-63, 63, 063). An alias therefore resolves only together with its owning
record/scope (core/, diacritizer/, ...); never resolve a bare legacy number
against the registries without that context.

## 7. Deliberately NOT migrated (historical exceptions)

- The ambiguous/stale backlog: e63-small-style naming ambiguities,
  Milestone-F (row 15, blocked on user), the umbrella row 102 (Laya
  ladder), the broad pending Track-D/E/F/G backlog bucket (rows 32-35),
  and other stale rows whose closure evidence is not exact enough.
- The duplicate row ids 40/47/49/61/70/91/97, plus duplicate rows
  63/64/65/66/67/68 whose duplicate contents are done-era historical
  records, not current open work; they were folded into canonical records
  above only where this report marks it explicitly (62, 63, 64, 65, 66).
- E-13/E-41/E-42/E-43/E-44/E-47/E-52/E-53/E-57 experiment-ID collisions:
  kept as historical ledger rows; not renamed and not namespaced.
- research/EXPERIMENTS.md table rows 40-48 (date-only) kept verbatim.
- The two broken research/EXPERIMENTS.md markdown links and the TASKS.md
  malformed row are documented as historical exceptions and left untouched.

Validations and the commit record live in
.superpowers/sdd/task-7-context-report.md (the Task 7 implementation
report). Validations cover the exact linter/renderer output, the focused
and full context tests, and the commit hash.
