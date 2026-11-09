# Micro-Expert Composition Line (ME-line) — Design

> Status: DESIGN v1 — user-approved 2026-09-18. Stage 0 (μ0) approved **as the base
> for all later stages**; μ1–μ3 approved as direction, each launch re-gated.
> Origin: user brainstorm 2026-09-18 — "train many tiny (100–300K) single-task
> models, then combine them into a 1–10M model; start small, grow step by step."
> Companion records: TASKS rows 76–79; MEMORY decision 2026-09-18 + lesson 73;
> HANDOFF 2026-09-18 entry. Web research behind §4/§8 was done live in the same session.
> User decisions recorded: (1) all four composition methods in scope; (2) μ0 = base
> (stepping stones); (3) task policy = synthetic research testbed + REAL Arabic
> diacritics as the first real expert (option A + B hybrid; the D-line data on disk
> is decent, so we focus on it now). NOTHING is implemented yet — this is the spec.

## 1) Purpose and the falsifiable question

**Primary question:** at matched TOTAL parameters and matched total training
tokens, does a routed committee of micro-specialists beat ONE dense multi-task
model on held-out per-task evals and mixed-input evals?

**Secondary goals:**
- Build a cheap sandbox where composition mechanisms can be A/B'd in minutes —
  an experiment class the repo cannot afford at the S/P/T (12M–226M) scale.
- Transfer any winning recipe to the real lines (D-line first). The repo's own
  verdicts (E-13 "per-domain routing", E-14 "specialization beats transfer",
  E-19 rule-(b) domain-specialist pass) independently point at the same frontier;
  this line is the direct, cheap test of that frontier.

## 2) Stage ladder — stepping stones (answers the user's question)

μ0 is the base: every later stage consumes μ0 artifacts and retrains nothing it
can reuse. Artifact flow:

    μ0  shared vocab + task generators + eval harness
        ├── 4 trained micro-experts (same seed, mergeable)
        └── dense multi-task CONTROL (honest baseline)
              │
    μ1        composition bake-off — 4 arms, SAME experts reused per arm:
              A weight-merge · B MoE-merge · C dispatch · D committee-distill
              │  (pros/cons/gaps recorded per arm — user requirement)
    μ2        grow: 8–16 experts → 1–10M combined band, scaling curves
              │
    μ3        D-line transfer (real diacritics) — user-gated

If a μ1 arm fails, the experts still serve the surviving arms and μ2 — a failed
composition arm costs only its composition code, never the base.

| Stage | Status | Consumes | Produces |
|---|---|---|---|
| μ0 sandbox base | APPROVED — next | D-line slice, existing scripts/train.py + src/model.py | shared char vocab, 4 seeded CPU task generators, 4 micro-experts, per-task eval harness, dense control, param-math acceptance test |
| μ1 bake-off | planned (gate: μ0) | μ0 artifacts only | 4 arm results + pros/cons/gaps + winning recipe |
| μ2 grow | planned (gate: μ1 winner) | μ1 recipe | 8–16 experts, combined 1–10M, scaling curve |
| μ3 D-line transfer | planned (USER-GATED) | μ1 recipe + full D-line corpora | committee vs stage2b2500 4-gate + abdou held-out |

## 3) Design decisions (ME-line scope)

| id | Decision | Why |
|---|---|---|
| ME-D1 | Shared character vocab ≤128 ids across ALL experts AND the control | At 100–300K params embeddings dominate the budget; per-expert vocabs would make experts mutually unintelligible (composition dies first). Char-level matches the D-line precedent. |
| ME-D2 | All experts start from the SAME random seed/init | Mergeability (arm A) and BTM-style MoE merge (arm B) both need a shared basin; at micro scale a shared seed costs nothing. Naive averaging across different inits fails (permutation symmetry — Git Re-Basin). |
| ME-D3 | Expert shape ≈ 2 layers, d≈96–128, ffn 4×, 4 GQA heads, tied embeddings, 200–300K params | Param math at design estimate ≈235K (vocab ≤128 × d96 ≈ 12K emb; ~111K/layer). Exact shape + count locked at build and pinned by an acceptance test (MEMORY lesson 59). |
| ME-D4 | μ0 task set (user option A+B hybrid): X1 diacritics-wordlist (REAL, D-line slice) + X2 arithmetic + X3 structure + X4 string-ops (all synthetic, seeded, CPU-generated) | Real anchor early (user: "short/easy to deliver result"); synthetic tasks are exactly measurable — the setting where ≤300K models demonstrably learn (grokking / arithmetic-at-tiny-scale literature). |
| ME-D5 | Honest control: one dense multi-task model at matched TOTAL params (all experts summed) trained on the union of task data at matched TOTAL tokens | Without this control no composition claim is falsifiable. Equal-compute framing: sequential expert wall-clock is recorded and is part of the trade-off story. Param matching is PER STAGE: μ0/μ1 compares at ~1M totals; the 1–10M band is reached in μ2. |
| ME-D6 | Pre-registration discipline: the E-24 ledger row opens at μ0 launch with thresholds frozen BEFORE any results (E-15/E-16 discipline) | Honest reporting over optimistic claims; negatives are reported as negatives. |
| ME-D7 | Operations: experts trained SEQUENTIALLY on the single GPU, never concurrent with any live train job (single-GPU rule); generators + harness dev are CPU-only | AGENTS.md §4; one consumer card. |

## 4) The four composition methods — to validate in μ1 with pros/cons/gaps

| Arm | Mechanism | Expected pros | Expected cons | Gaps to measure |
|---|---|---|---|---|
| A | Weight merge (soup first, TIES/DARE fallback) | Zero inference cost; single deployable dense model; in-repo soup tooling exists (E-07/E-10) | Needs ME-D2 shared seed; task vectors may conflict; forgetting risk | Per-task accuracy drop vs experts; soup vs TIES sign-election; is a shared-seed "universal expert" possible at micro scale |
| B | MoE merge, BTM/BTX-style (experts become FFN slots + learned token-level router) | Literature-proven at scale; keeps experts intact; token-level routing; grows naturally in μ2 | Needs new MoE block in our arch (user-gated change; flow/ also deferred MoE bodies); router needs training | Router accuracy vs task-label ceiling; expert interference; top-1 vs top-2 |
| C | Dispatch ("vice" idea — router sends the WHOLE input to one expert) | Simplest; zero arch changes; pure modularity; failure = pure router failure | No knowledge sharing; router must infer the task from UNLABELED user input (labels exist only in training) | Router accuracy on unlabeled mixed inputs; OOD-input handling; boundary brittleness |
| D | Committee distillation (experts teach one dense student; loss = 0.5 KL + 0.5 CE) | Proven in-repo levers #2/#10 (E-03 −6.71%, E-16 P→S pass); produces one deployable dense model; student can exceed the committee on mixed inputs | Teacher-ensemble forward cost at train time; inherits committee blind spots; loses the "grow by adding experts" property | Retention: how much committee signal a param-matched student keeps |

**μ1 metric definitions (thresholds frozen in the μ1 pre-registration):**
- per-task accuracy (exact match; X1: held-out word exact-match + DER-lite),
- routing accuracy (C arm; unlabeled setting is the headline number),
- **composition premium** = best arm − dense control, per task and mixed,
- equal-tokens control bookkeeping per ME-D5.

## 5) μ0 scope (approved — build plan comes next)

**Experts (≈235K each, ME-D3):**
- **X1 diacritics-wordlist (real):** word-level bare→vocalized restoration sampled
  from the EXISTING data/diac corpora (no new downloads); fixed held-out wordlist.
- **X2 arithmetic:** a+b and a−b, operands 0–999, exact-answer eval.
- **X3 structure:** balanced-brackets / JSON-ish validity + next-tag, exact eval.
- **X4 string-ops:** copy / reverse / sort-letters of short char strings, exact eval.

**Also in μ0:** dense multi-task control (≈ same total params as the 4 experts,
union data, matched tokens); eval harness (exact-match + routing eval on
unlabeled mixed inputs); generators are seeded + deterministic + CPU-only.

**μ0 gate (pre-registered at launch in E-24):** every expert must beat its
trivial baseline (identity / most-frequent-answer) by a frozen margin. A task a
≤300K model cannot learn is ALSO a result — the fix (shrink task or grow expert)
is explicitly a user-gated menu item, never a silent workaround.

## 6) Honesty rules

E-24 pre-registered before results (ME-D6); every μ1 arm gets its own row before
its results; failed arms are reported with their pros/cons/gaps, not dropped
silently; wall-clock + params + tokens reported for every arm; user gates at
stage boundaries (μ2 launch, μ3 anything — D-line remains separately governed).

## 7) Risks

| Risk | Mitigation |
|---|---|
| ≤300K experts fail even the "tiny" tasks | That IS the μ0 feasibility answer; menu back to user (shrink task / raise per-expert budget), thresholds unchanged post-hoc |
| Merge arms (A/B) all fail | Expected possible outcome; C and D do not depend on mergeability; report as measured |
| GPU contention with other live jobs | ME-D7 sequential discipline; CPU-only until a free GPU window |
| Disk headroom swings (~10 GB) | Micro artifacts are tiny (KB–MB scale); nothing large added |
| Other agents active in the same tree (rows 69–75, E-23 lineage) | Touch only ME-line paths; no edits to M3/D-line src/configs |

## 8) Web references (retrieved 2026-09-18)

- Branch-Train-Merge (experts from a shared seed, merged, routed): https://arxiv.org/abs/2208.03306
- Branch-Train-MiX (expert-LLM → MoE with learned routing): https://doi.org/10.48550/arxiv.2403.07816
- Scaling Laws for Fine-Grained MoE (many small experts): https://arxiv.org/html/2402.07871v1
- Model merging survey (task arithmetic / TIES / merger-friendly fine-tuning): https://arxiv.org/html/2408.07666v5
- Git Re-Basin (permutation symmetry — why unshared inits cannot be averaged): https://github.com/samuela/git-re-basin
- Modular Deep Learning survey (frozen modules + routing): https://doi.org/10.48550/arxiv.2302.11529
- Router upcycling in MoE: https://doi.org/10.48550/arxiv.2509.00679
- TinyStories (what 1M-scale LMs can and cannot do): https://arxiv.org/pdf/2305.07759
- Grokking modular arithmetic: https://arxiv.org/html/2301.02679
- Teaching Arithmetic to Small Transformers: https://arxiv.org/pdf/2307.03381

In-repo levers reused: WHAT_WORKS #1 (warm-start init), #2/#10 (KD 0.5KL+0.5CE),
#5 soup as post-hoc knob, E-20 micro param math. Related ledger: E-13/E-14/E-19
(domain-specialist + routing verdicts), E-21 (micro 12-128-512 recipe exists).
