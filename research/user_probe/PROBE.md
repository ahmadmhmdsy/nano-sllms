## User syntax-trap probe set (registered 2026-09-16)

Purpose: the user's hand-built adversarial texts stress the model's weakest
surface - long-distance syntax (Nawaasikh enna/kana, passive verbs, Arabic
feminine plural kasra, the five nouns, ma'saddan). Kept as ONE of the standing
probe sets: the text is DEEPLY domain-tied to Tashkeela-style formal Arabic
(the model has seen similar text), so numbers here are ADVISORY, not gates.

Rooms:
1. MSA news/politics (foreign names, passive 'uqirrat).
2. Grammar traps (feminine plural kasra, five nouns, skyn-merging 'lam yakuni').
3. Classical/philosophical (Tashkeela-style rhetoric).

File layout:
- probe1_bare.txt + probe1_ref.txt (bare = marks stripped from the user text;
  ref = the user's semantically-correct gold, saved NFC).
- (2, 3) same per paragraph.
- probe_ref.txt = the user's ORIGINAL text (partially diacritized; NO fully
  vocalized gold exists for these) - so there is NO legal DER here. These are
  QUALITATIVE probes: read gold_pred vs your own native-speaker judgement.
- gold_pred_probe{1,2,3}.txt = stage2b2500 predictions generated WITH the
  input-stripping path active (bare input guaranteed).

Important honest note discovered while scoring: against the fully-clean bare
inputs the gold model's errors are MORE visible than in the user's original
double-run (where partially-marked inputs quietly inflated apparent quality
via passthrough). The E-18 lesson (input contract) is even more relevant.

Findings registered as E-18 (input-contract + postproc A/B): see
research/EXPERIMENTS.md row E-18 and MEMORY lesson 61.
