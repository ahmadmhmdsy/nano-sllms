# U-chain dataset adequacy report (R2, 2026-09-27, written while the
# progressive-averaging arm trains on the GPU)

Corpus under analysis: data/u1/full/raw/{train,val,test}.jsonl, the exact
sources the R2 arm experiments regenerated from (the 4000-row paired arm
slices are seeded subsets of train.jsonl; the pilot scale below is the
pre-registered full set). All numbers are read from the artifacts of
scratch/dataset_adequacy_r2.py / _structure.py / _components.py (throwaway
probes; the report is the durable record).

## Volume and tokens

| split | rows | unique texts | tokens | mean tok/row | p95 tok/row | rows > ctx 512 |
|---|---|---|---|---|---|---|
| train | 40000 | 40000 (no dup) | 15,585,563 | 389.6 | 598 | 5006 |
| val | 1500 | 1500 | 681,306 | 454.2 | 613 | 562 |

Tokenizer: data/target/tokenizer/tokenizer.json, context 512; zero <unk>
overflow tokens in both splits: the vocabulary is complete for this domain.

## Structure (chain contract)

Every sampled row (6000/6000) reverses through ui.src.chain.chain_to_spec
with zero validator issues: the corpus is 100% structurally valid against
the CURRENT validator. Line grammar per row: UI1/TASK/ROOT/STAX + EL/PROP/
BIND lines; component inventory over the full 40000 rows:

Card 65000, Button 25000, Stack 20000, Input 20000, Badge 13032,
Switch 13014, Grid 10000, Metric 10000, Alert 10000, Select 5000,
Heading 5000, Text 5000, Pagination 5000, Table 5000, Progress 3429,
LineGraph 1803, BarGraph 1673 — 17 component types at healthy frequency
bands; bind targets bindState/state/item; 8662 unique ROOT ids in an 8k
sample (unique card ids row to row).

## Diversity (the weak rung)

- Task-side: only 36 unique task phrasings across 40000 train rows (top
  task repeated ~313 times). The pairing task->spec inside one row is
  injective, but the PROMPT distribution is thin: a model can overfit
  task phrasing memorization instead of task->spec semantics.
- Spec-side: 19 unique titles across 40000 rows (Account settings, Todo
  list, ... each exactly 5000): spec values are template-generated, not
  sampled from a wide natural distribution. Diversity of LAYOUT (roots,
  component mixes) is high; diversity of surface VALUES (titles, STAX
  numbers) is low and repetitive.

## Adequacy vs the task (near-SOTA claim surface)

For the pre-registered task (38-bit-style chain JSON spec generation as
sdpa f16 8L/256H model, valid@4 + pass@4 on the same distribution):

- ENOUGH for: home-archetype component/graph coverage (17 types), chain
  grammar mastery, low-perplexity task->spec mapping (u2h/u2i/eos arms
  reached internal val_loss 4.07 and pass@4 40/40 at 600 steps on an
  8L/256H model with weight averaging).
- NOT ENOUGH for: SOTA-grade task->spec generalization CLAIMED AS
  GENERAL. 36 unique task phrasings means any pass@k on a val split drawn
  by the same generator (ui.src.generator) is错过了 family-fit, not
  natural-language-fit: a near-SOTA-looking pass@4 = 1.00 CANNOT be read
  as the model generalizing to phrases outside the 36 templates.
- v_length: 5006/40000 train rows exceed the 512 context (tail-truncated
  by packing with the mask contract); acceptable for 600-step signal but
  a pilot scale run should either raise ctx or drop the policy formally.

## Recommendation

The corpus is sufficient for the registered R2 question (paired escalation
levers on the current contract). For a defensible near-SOTA claim, the ui
line needs a DIVERSITY escalation: randomize STAX numeric values, expand
task phrasings beyond the 36 templates (the ui/src/generator templates),
and report pass@k on a NOVEL-phrase test slice — recorded as the
pre-registered next step for the pilot scale, before any full SOTA claim.


## UPDATE (u3 diversity escalation, 2026-09-27, real regeneration)

Deterministic generator diversity items landed in ui/src/generator.py
(value randomization for STAX money/labels/quarters/chart titles/section
heads; compositional task phrasing with register rolls; 2-3 metric tile
root variants). Regenerated corpus: data/u3/raw/u3_diverse_50k.jsonl
(builder: build_corpus --per-arch 5000 --seed 7; 40000 rows, 0 fails,
round-trip + idempotence clean per builder contract).

Measured change (probe scratch/probe_u3_titles2.py, artifact values):

| metric | u1 corpus | u3 corpus |
|---|---|---|
| unique task phrasings | 36 | 870 |
| distinct STAX heads (first 120 chars) | ~1 per archetype constant | 12,915 |
| distinct money tokens | ~1-4 | 611 |
| distinct title props | 19 | 27 |
| metric label vocab | 2 | 26 |

LLM-authored novel-phrase slices (ONLY opencode3/space-bunny-free
subagents, 4 parallel agents): data/u3/novel/phrasings_{a,b,c,d}.jsonl =
120+120+180+160 = 580 phrases. Held-out NOVEL gate slices built with
spec_to_chain from a disjoint seed window (50000/60000 offsets):
data/u3/novel/gate_novel_a.jsonl + gate_novel_b.jsonl (40+40 rows,
validate_chain 0 invalid / 80 checked).

Pre-registration reminder: a 600-step arm run on the u3 corpus
(pipeline_v2/configs/experiments/u3_arm_eos_avg.yaml, adopted recipe:
eos seams + final_n=2 averaging) measures family-fit on the NEW family;
the defensible near-SOTA column remains the novel-phrase gate, which
requires a trained checkpoint then scratch/gate_u2_arms.py on the novel
rows.
