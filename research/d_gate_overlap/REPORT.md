# Gate ↔ Train Data-Domain Overlap Audit (CPU-only)

Method: for each train source (reservoir sample, cap 2000 windows, seed 20250101,
text = `text` field of prepared windows) and each gate (bench.py load_pairs,
raw files only), text is mark-stripped (U+064B-0652, 0670, 0653-065F, 06D6-06ED)
+ NFC. Metrics: 40-char shingle Jaccard; token 4-gram containment
(|gate∩train|/|gate|); char-bigram cosine. Full numbers in
`overlap_metrics.json`. Scripts: `scratch/g_overlap_sample.py`,
`scratch/g_overlap_metrics.py`.

## Verdicts

| Gate | Nearest train source | shJ | 4g-contain | bigCos | Verdict |
|---|---|---|---|---|---|
| fadel_test | v2b::fadel_train (+fadel_val) | 0.0002–0.0005 | 0.011 | **0.9994** | **RELATED (+)** same domain, no verbatim overlap → fair-ish DER, mildly favorable because char stats are near-identical |
| sadeed25 | sadeedt::sadeedt_train | 0.0001 | 0.005 | 0.984 | **RELATED (+)** classical domain covered by Sadeed_Tashkeela train; no copy overlap → fair DER if sadeedt is trained on |
| wikinews2024 | v2b::wikinews2024 | **0.968** | **0.972** | 1.000 | **CLOSE — LEAKAGE: the gate appears verbatim inside training windows (356/356 units)** → DER on wn2024 is effectively validation-on-train, drastically OPTIMISTIC |
| wikinews2014 | v2b::qcri_wiki | 0.0000 | 0.001 | 0.959 | **RELATED (near-baseline)** same wiki-1D-epoch texture via QCRI wiki train; zero copy overlap → honest but slightly favorable read |

## Gate file sanity (read-only checks)

| Gate path | exists | bytes | lines | empty | usable pairs (mark-stripped) |
|---|---|---|---|---|---|
| data/diac/raw/fadel/test.txt | yes | 1,747,544 | 2,500 | 0 | 2,500 |
| data/diac/raw/sadeed_25/sadeed25.parquet | yes | 678,514 | 1,200 rows (cols: filename,input,output) | 0 | 1,612 line-level pairs |
| data/diac/raw/wikinews/wikinews2024_multi_ref.diac | yes | 200,310 | 431 | 75 | 356 |
| data/diac/raw/wikinews/wikinews2014_multi_ref.diac | yes | 310,342 | 400 | 0 | 393 (after #-header skips) |

Train source sizes (windows seen in prepared corpora): abdou_train 1,443,972;
abdou_valid 29,633; fadel_train 50,000; fadel_val 2,500; qcri_wiki 32,818;
wikinews2024 356; sadeedt_train 1,042,678.

## Read

1. **wikinews2024 gate is contaminated** — the v2b train corpus folds in a source literally called `wikinews2024` with exactly 356 windows, and 97% of the gate's 40-char shingles and 4-grams match it. Any DER reported on wn2024 for models trained on v2b is optimistic and should be treated as in-domain dev, not an external gate.
2. fadel_test and sadeed25 are *domain-matched without leakage* — identical bigram texture (cos ≈ 0.98–1.00) but essentially zero seq copy overlap. DER there is trustworthy (neither inflated by memorization nor deflated by domain shift).
3. wikinews2014 is the most "external" wiki gate: zero verbatim overlap with every train source; nearest is qcri_wiki by texture (0.959), so a mild favorability remains only at the char-stat level, not content level.
4. Baseline sanity: empty-line counts are zero except 75 separator blanks in the wn2024 multi-ref file (expected format, skipped by bench.py).
