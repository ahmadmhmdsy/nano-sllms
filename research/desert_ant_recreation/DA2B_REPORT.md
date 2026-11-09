# DA-2b (arms a/b/c) report - 2026-09-19 - E-47

Registered BEFORE results (research/EXPERIMENTS.md, E-47 row). Fixed bars from
the E-44 pre-registration, re-measured on the new (bigger) dataset prior:
prior top-1 = 0.2467 -> bars: 2x prior = 0.4934, prior+0.05 = 0.2967.

## Arms and results (all CPU-only, 4 concurrent bag arms then transformer)

Dataset (arm a - new Arabic data): TEAD (7,773 ar one-type rows, 67 emojis)
+ **amgadhasan/arabic_tweets_dialects (34,514 one-type rows mined from 147,725
Arabic tweets, 435 distinct emojis)** + tweet_eval/emoji (en, 20 emojis).
New splits: 73,164 / 9,076 / 9,304; 160 emoji classes (147 gold emojis on ar
test, 20 on en).

| arm | recipe | best val | TEST top-1 ar | TEST top-1 en | prior (0.2467) |
|---|---|---|---|---|---|
| A0 (a) | bag plain, 160 classes | 0.2723 | **0.3207** | 0.2013 | ar +7.4 pp / en -4.5 pp |
| A1 (a+c) | bag + weighted CE + LS 0.1 | 0.1206 | not evaluated (diverged) | - | FAIL recipe |
| A2 (a+b) | tiny transformer (1 layer d32 ff64 over token buckets) | 0.3168 | **0.4063** | 0.2267 | ar +16.0 pp |
| A3 (a+b+c) | transformer + weighted CE + LS | 0.3191 | 0.4061 | 0.2267 | identical (top-3 ar 0.484 / en 0.434) |

top-3: A2 ar 0.4952 / en 0.4252.

## Gates (fixed bars, read honestly)

- D2 vs 2x prior (0.4934): **FAIL** in every arm (best ar 0.4063 = 82% of the
  bar).
- D2 vs prior+0.05 (0.2967): **ar PASS in A2/A3** (0.4063, +11 pp over bar);
  **en FAIL** (0.2267 < 0.2967 and even below the 0.2467 prior - the en head
  degraded when ar dominated training; honest regression vs E-44 en 0.2335).
- D4 int8 transformer: **1.686 MiB**, 1.75 ms/text pure-numpy per-text
  forward, int8 vs fp32 agreement ar 1000/1000, en 998/1000 - **PASS**.
- D5: no single-emoji collapse (95+ predicted classes on ar for A0; A2 is
  top-heavy but the top-1 set spans the heavy-tail head, ties 1-4).

## Honest closing verdict

- Option (a) bigger Arabic corpus: **worked as data lever** - ar test rows
  5x, ar top-1 0.2255 -> 0.3207 (bag) / 0.4063 (transformer).
- Option (b) tiny transformer head: **the real skill lever** - +8.6 pp ar
  over the bag on the same data; cheapest single change.
- Option (c) weighted CE + label smoothing: **FAIL, recorded honestly**.
  Median-normalized inverse-sqrt weights diverged the bag arm (loss ~99);
  mean-normalized at lr 0.05 still diverged (loss ~80); lr 0.01 + grad clip
  1.0 stopped the blow-up but the arm got STUCK (loss plateau ~7.6,
  val 0.12 - worse than prior): with ~100 singleton classes in a 160-way
  head, rare-class upweighting floods the head. On the transformer, (c) is
  a no-op at best (0.3191 vs 0.3168 val; test identical to 4 decimals).
- Overall: D2's 2x-prior bar remains **unmet** (the 20x head class still
  owns 18k of 73k train rows); the prior+0.05 bar is now met for
  Arabic (top-1 0.406) but not for English in this training mix.

## Next-step candidates (user-gated, NOT done)

1. En-preserving multi-task: keep the E-44 20-class en head and train ar
   separately (two small heads, shared trunk) - en regression risk removed.
2. Bigger transformers still fit D4 (d=64, 2 layers ~= 3.2 MiB int8).
3. A 2x-prior-class bar may simply be wrong for 812-way emoji suggestion
   their card reports "top-1 ~0.5 in 812-way produkts with a full
   transformer"; without their reported per-class numbers a head-to-head
   is still not possible.

Artifacts: runs/langid_da2b/{emo_bag,emo_transformer,emo_transformer_w_ls}.pt,
emo_tf_int8.npz (+ .report.json), *.log.json, *.eval.json.
