# Diacritizer full-metric evaluation

Generated 2026-10-02 13:14:52 by `diacritizer/scripts/eval_report.py`.

Reference: `models/e19/wn2024.gold.raw.pred.ref.txt` (356 lines, 10616 words, sha256 `8353777058e51525`).

Command: `diacritizer\scripts\eval_report.py --ref models\e19\wn2024.gold.raw.pred.ref.txt --manifest research\wn2024_models.json --out-json research\wn2024_full_metrics.json --out-md research\wn2024_full_metrics.md --baseline mishkala --rank-by der_canonical`

## Conditions

The WikiNews 2024 reference carries `/` alternatives inside a word, so
"DER" is ambiguous unless the comparison condition is named. Every table
below states which one it uses.

| condition | meaning |
|---|---|
| registered | whole `/` token, marks vs the single raw reference line (eval_der) |
| anyalt | whole `/` token, marks vs ANY alternative (registered gate / eval_multi_ref) |
| primary | the model's FIRST alternative vs the reference's FIRST alternative |
| canonical | as anyalt, but each mark run is rewritten to shadda-first order first |

### Shadda/vowel byte order

labels.label_for_marks accepts a shadda+vowel run in EITHER byte order and
maps both to a single class; its own docstring records that real corpora use
both. eval_der compares raw mark STRINGS, so a vowel-first prediction is
scored wrong against a shadda-first reference even though the schema calls
the two identical. The canonical condition canonicalizes the order first, so
the delta below is exactly the cost of that byte order - and it is large
enough to change which model ranks first.

| model | der_anyalt | der_canonical | delta |
|---|---|---|---|
| mishkala | 52.45% | 45.20% | 7.25% |
| zmahood | 47.88% | 47.89% | -0.01% |
| e63s_s4_12l160 | 55.90% | 51.33% | 4.57% |
| stage2b2500 | 57.66% | 52.80% | 4.86% |
| e63s_s3_8l272 | 58.36% | 53.51% | 4.84% |
| e63s_s2_10l272 | 59.98% | 54.90% | 5.09% |
| e63s_s5_10l128 | 59.23% | 54.97% | 4.26% |
| e63s_s1_12l272 | 60.51% | 56.16% | 4.35% |
| e63s_s6_8l96 | 60.68% | 56.99% | 3.69% |
| e63s_s7_18l64 | 68.64% | 67.19% | 1.45% |

## Headline metrics

| model | der_anyalt | der_canonical | der_primary | der_primary_nocase | ser | cder | mark_f1 | macro_f1 | position_accuracy | cohen_kappa | worst_class_recall | density_ratio | case_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mishkala | 52.45% | 45.20% | 50.94% | 42.68% | 0.00% | 28.85% | 74.43% | 73.78% | 86.82% | 0.835 | 33.33% | 0.939 | 86.10% |
| zmahood | 47.88% | 47.89% | 47.32% | 39.99% | 0.00% | 21.30% | 84.74% | 68.20% | 82.34% | 0.778 | 25.00% | 0.876 | 81.84% |
| e63s_s4_12l160 | 55.90% | 51.33% | 53.82% | 42.28% | 0.00% | 32.12% | 71.53% | 57.21% | 82.97% | 0.785 | 3.85% | 0.917 | 75.20% |
| stage2b2500 | 57.66% | 52.80% | 55.84% | 43.88% | 0.00% | 32.79% | 70.64% | 60.82% | 82.57% | 0.781 | 11.54% | 0.930 | 74.97% |
| e63s_s3_8l272 | 58.36% | 53.51% | 56.25% | 42.56% | 0.00% | 32.75% | 70.81% | 59.37% | 82.66% | 0.781 | 0.00% | 0.925 | 72.61% |
| e63s_s2_10l272 | 59.98% | 54.90% | 58.23% | 43.86% | 0.00% | 34.90% | 68.74% | 60.07% | 80.91% | 0.758 | 0.00% | 0.921 | 68.54% |
| e63s_s5_10l128 | 59.23% | 54.97% | 57.15% | 45.54% | 0.00% | 34.37% | 69.37% | 56.93% | 81.14% | 0.763 | 0.00% | 0.927 | 73.76% |
| e63s_s1_12l272 | 60.51% | 56.16% | 59.39% | 44.98% | 0.00% | 35.29% | 68.47% | 57.63% | 80.44% | 0.753 | 0.00% | 0.918 | 67.65% |
| e63s_s6_8l96 | 60.68% | 56.99% | 59.11% | 48.17% | 0.00% | 35.70% | 68.32% | 51.03% | 79.37% | 0.740 | 0.00% | 0.920 | 72.98% |
| e63s_s7_18l64 | 68.64% | 67.19% | 68.14% | 57.75% | 0.00% | 45.96% | 59.18% | 36.75% | 68.75% | 0.600 | 0.00% | 0.901 | 58.87% |

## Confidence intervals (line bootstrap, 2000 resamples)

Lines are the independent unit: a sentence is annotated as a whole, so the
bootstrap resamples lines and re-weights by each line's own word count.

Ranking and the paired test below use der_canonical.

| model | metric | point | 95% CI |
|---|---|---|---|
| mishkala | der_anyalt | 0.5245 | [0.5093, 0.5414] |
| mishkala | der_canonical | 0.4520 | [0.4359, 0.4701] |
| mishkala | der_primary | 0.5094 | [0.4945, 0.5263] |
| mishkala | der_registered | 0.5167 | [0.5017, 0.5335] |
| mishkala | ser | 0.0000 | [0.0000, 0.0000] |
| mishkala | cder | 0.2885 | [0.2820, 0.2953] |
| zmahood | der_anyalt | 0.4788 | [0.4671, 0.4918] |
| zmahood | der_canonical | 0.4789 | [0.4672, 0.4919] |
| zmahood | der_primary | 0.4732 | [0.4615, 0.4866] |
| zmahood | der_registered | 0.4786 | [0.4667, 0.4916] |
| zmahood | ser | 0.0000 | [0.0000, 0.0000] |
| zmahood | cder | 0.2130 | [0.2050, 0.2218] |
| e63s_s4_12l160 | der_anyalt | 0.5590 | [0.5487, 0.5695] |
| e63s_s4_12l160 | der_canonical | 0.5133 | [0.5033, 0.5243] |
| e63s_s4_12l160 | der_primary | 0.5382 | [0.5282, 0.5487] |
| e63s_s4_12l160 | der_registered | 0.5576 | [0.5473, 0.5681] |
| e63s_s4_12l160 | ser | 0.0000 | [0.0000, 0.0000] |
| e63s_s4_12l160 | cder | 0.3212 | [0.3152, 0.3273] |
| stage2b2500 | der_anyalt | 0.5766 | [0.5652, 0.5880] |
| stage2b2500 | der_canonical | 0.5280 | [0.5168, 0.5392] |
| stage2b2500 | der_primary | 0.5584 | [0.5474, 0.5692] |
| stage2b2500 | der_registered | 0.5757 | [0.5645, 0.5871] |
| stage2b2500 | ser | 0.0000 | [0.0000, 0.0000] |
| stage2b2500 | cder | 0.3279 | [0.3220, 0.3337] |
| e63s_s3_8l272 | der_anyalt | 0.5836 | [0.5735, 0.5943] |
| e63s_s3_8l272 | der_canonical | 0.5351 | [0.5249, 0.5461] |
| e63s_s3_8l272 | der_primary | 0.5625 | [0.5526, 0.5729] |
| e63s_s3_8l272 | der_registered | 0.5820 | [0.5719, 0.5926] |
| e63s_s3_8l272 | ser | 0.0000 | [0.0000, 0.0000] |
| e63s_s3_8l272 | cder | 0.3275 | [0.3217, 0.3334] |
| e63s_s2_10l272 | der_anyalt | 0.5998 | [0.5897, 0.6112] |
| e63s_s2_10l272 | der_canonical | 0.5490 | [0.5385, 0.5603] |
| e63s_s2_10l272 | der_primary | 0.5823 | [0.5724, 0.5933] |
| e63s_s2_10l272 | der_registered | 0.5970 | [0.5869, 0.6083] |
| e63s_s2_10l272 | ser | 0.0000 | [0.0000, 0.0000] |
| e63s_s2_10l272 | cder | 0.3490 | [0.3431, 0.3550] |
| e63s_s5_10l128 | der_anyalt | 0.5923 | [0.5820, 0.6032] |
| e63s_s5_10l128 | der_canonical | 0.5497 | [0.5395, 0.5605] |
| e63s_s5_10l128 | der_primary | 0.5715 | [0.5618, 0.5822] |
| e63s_s5_10l128 | der_registered | 0.5901 | [0.5799, 0.6009] |
| e63s_s5_10l128 | ser | 0.0000 | [0.0000, 0.0000] |
| e63s_s5_10l128 | cder | 0.3437 | [0.3378, 0.3500] |
| e63s_s1_12l272 | der_anyalt | 0.6051 | [0.5945, 0.6163] |
| e63s_s1_12l272 | der_canonical | 0.5616 | [0.5502, 0.5732] |
| e63s_s1_12l272 | der_primary | 0.5939 | [0.5832, 0.6051] |
| e63s_s1_12l272 | der_registered | 0.6039 | [0.5934, 0.6152] |
| e63s_s1_12l272 | ser | 0.0000 | [0.0000, 0.0000] |
| e63s_s1_12l272 | cder | 0.3529 | [0.3472, 0.3587] |
| e63s_s6_8l96 | der_anyalt | 0.6068 | [0.5965, 0.6176] |
| e63s_s6_8l96 | der_canonical | 0.5699 | [0.5600, 0.5811] |
| e63s_s6_8l96 | der_primary | 0.5911 | [0.5811, 0.6018] |
| e63s_s6_8l96 | der_registered | 0.6046 | [0.5943, 0.6154] |
| e63s_s6_8l96 | ser | 0.0000 | [0.0000, 0.0000] |
| e63s_s6_8l96 | cder | 0.3570 | [0.3505, 0.3636] |
| e63s_s7_18l64 | der_anyalt | 0.6864 | [0.6762, 0.6971] |
| e63s_s7_18l64 | der_canonical | 0.6719 | [0.6617, 0.6828] |
| e63s_s7_18l64 | der_primary | 0.6814 | [0.6713, 0.6921] |
| e63s_s7_18l64 | der_registered | 0.6860 | [0.6759, 0.6968] |
| e63s_s7_18l64 | ser | 0.0000 | [0.0000, 0.0000] |
| e63s_s7_18l64 | cder | 0.4596 | [0.4538, 0.4655] |

## Pairwise significance vs `mishkala` on `der_canonical` (paired, per word)

Exact two-sided McNemar on discordant words, plus a paired line bootstrap of
the difference. Both use the same sentences for both systems, which is what
makes them valid where comparing two independent intervals would not be.

| model | diff der_canonical | 95% CI | excludes 0 | McNemar p | significant |
|---|---|---|---|---|---|
| zmahood | -0.0269 | [-0.0389, -0.0111] | yes | 4.09e-17 | yes |
| e63s_s4_12l160 | -0.0613 | [-0.0758, -0.0440] | yes | 4.44e-56 | yes |
| stage2b2500 | -0.0760 | [-0.0901, -0.0585] | yes | 3.38e-81 | yes |
| e63s_s3_8l272 | -0.0832 | [-0.0972, -0.0656] | yes | 1.15e-95 | yes |
| e63s_s2_10l272 | -0.0970 | [-0.1116, -0.0799] | yes | 9.88e-128 | yes |
| e63s_s5_10l128 | -0.0978 | [-0.1118, -0.0811] | yes | 4.15e-134 | yes |
| e63s_s1_12l272 | -0.1096 | [-0.1243, -0.0923] | yes | 9.82e-155 | yes |
| e63s_s6_8l96 | -0.1179 | [-0.1321, -0.1012] | yes | 2.33e-183 | yes |
| e63s_s7_18l64 | -0.2200 | [-0.2350, -0.2034] | yes | < 1e-308 | yes |

## Per-class recall (primary condition)

Support is the reference count of each class; a recall of 0.00% on a class
with support is a collapsed class that an accuracy figure would hide.

Support is taken as the largest count any model aligned, i.e. the
reference count. It is NOT one model's denominator: class metrics are
computed over each model's ALIGNED subset and alignment coverage differs
(see alignment_coverage in the headline table), so a model that fails to
align a word also drops it from that class's denominator. Reading one
model's support beside another model's recall silently compares two
different denominators.

| class | support | mishkala | zmahood | e63s_s4_12l160 | stage2b2500 | e63s_s3_8l272 | e63s_s2_10l272 | e63s_s5_10l128 | e63s_s1_12l272 | e63s_s6_8l96 | e63s_s7_18l64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bare` | 9993 | 97.97% | 98.52% | 98.34% | 97.93% | 98.23% | 98.12% | 97.58% | 98.03% | 97.38% | 93.60% |
| `fatha` | 15366 | 77.93% | 75.97% | 76.45% | 75.26% | 77.70% | 76.73% | 74.69% | 75.88% | 72.85% | 69.24% |
| `damma` | 3013 | 82.67% | 76.71% | 78.03% | 70.23% | 74.68% | 72.65% | 70.06% | 72.35% | 66.61% | 47.46% |
| `kasra` | 8814 | 90.32% | 83.52% | 84.98% | 84.32% | 80.28% | 75.60% | 81.13% | 75.52% | 80.81% | 66.49% |
| `sukun` | 6270 | 88.97% | 81.59% | 83.94% | 86.83% | 84.70% | 81.00% | 83.86% | 81.90% | 82.62% | 58.93% |
| `fathatan` | 275 | 82.26% | 81.09% | 42.55% | 71.27% | 60.36% | 80.73% | 61.45% | 49.45% | 46.91% | 22.18% |
| `dammatan` | 103 | 74.00% | 52.43% | 54.37% | 47.57% | 48.54% | 35.92% | 58.25% | 49.51% | 41.75% | 14.56% |
| `kasratan` | 493 | 76.65% | 70.39% | 31.44% | 41.99% | 47.67% | 39.96% | 43.00% | 44.42% | 26.57% | 16.43% |
| `shadda` | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `shadda_fatha` | 1890 | 91.86% | 75.40% | 85.34% | 86.56% | 87.04% | 88.52% | 84.44% | 86.03% | 80.21% | 58.57% |
| `shadda_damma` | 222 | 77.78% | 59.46% | 60.81% | 57.21% | 63.06% | 55.41% | 61.71% | 56.76% | 40.54% | 26.13% |
| `shadda_kasra` | 629 | 78.95% | 47.69% | 58.35% | 56.44% | 53.26% | 54.69% | 51.19% | 53.10% | 44.99% | 20.03% |
| `shadda_fathatan` | 44 | 97.62% | 70.45% | 13.64% | 36.36% | 36.36% | 72.73% | 40.91% | 34.09% | 11.36% | 0.00% |
| `shadda_dammatan` | 24 | 33.33% | 25.00% | 12.50% | 37.50% | 16.67% | 20.83% | 16.67% | 25.00% | 8.33% | 0.00% |
| `shadda_kasratan` | 26 | 54.17% | 34.62% | 3.85% | 11.54% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Collapsed classes

| model | classes present | collapsed (recall 0.00%) | worst class recall |
|---|---|---|---|
| mishkala | 14 | none | 33.33% |
| zmahood | 14 | none | 25.00% |
| e63s_s4_12l160 | 14 | none | 3.85% |
| stage2b2500 | 14 | none | 11.54% |
| e63s_s3_8l272 | 14 | `shadda_kasratan` | 0.00% |
| e63s_s2_10l272 | 14 | `shadda_kasratan` | 0.00% |
| e63s_s5_10l128 | 14 | `shadda_kasratan` | 0.00% |
| e63s_s1_12l272 | 14 | `shadda_kasratan` | 0.00% |
| e63s_s6_8l96 | 14 | `shadda_kasratan` | 0.00% |
| e63s_s7_18l64 | 14 | `shadda_dammatan`, `shadda_fathatan`, `shadda_kasratan` | 0.00% |

## Confusion matrix - `mishkala` (gold rows x predicted columns)

| gold \ pred | bare | fatha | damma | kasra | sukun | fathatan | dammatan | kasratan | shadda | shadda_fa | shadda_da | shadda_ka | shadda_fa | shadda_da | shadda_ka |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bare` | 9460 | 31 | 6 | 4 | 120 | 1 | 0 | 0 | 0 | 4 | 1 | 10 | 0 | 8 | 11 |
| `fatha` | 2428 | 11582 | 284 | 249 | 113 | 48 | 20 | 40 | 0 | 64 | 1 | 3 | 28 | 1 | 1 |
| `damma` | 11 | 327 | 2399 | 95 | 8 | 9 | 35 | 3 | 0 | 1 | 12 | 2 | 0 | 0 | 0 |
| `kasra` | 112 | 352 | 176 | 7692 | 24 | 2 | 11 | 133 | 0 | 2 | 3 | 9 | 0 | 0 | 0 |
| `sukun` | 22 | 228 | 132 | 185 | 5390 | 0 | 11 | 64 | 0 | 10 | 5 | 8 | 0 | 0 | 3 |
| `fathatan` | 0 | 6 | 3 | 2 | 0 | 218 | 20 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `dammatan` | 0 | 1 | 5 | 1 | 0 | 15 | 74 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `kasratan` | 0 | 16 | 5 | 54 | 1 | 10 | 24 | 361 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `shadda` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `shadda_fatha` | 7 | 47 | 1 | 10 | 10 | 0 | 0 | 0 | 0 | 1682 | 26 | 48 | 0 | 0 | 0 |
| `shadda_damma` | 0 | 1 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 19 | 168 | 17 | 0 | 0 | 1 |
| `shadda_kasra` | 5 | 8 | 0 | 38 | 3 | 0 | 0 | 0 | 0 | 45 | 29 | 480 | 0 | 0 | 0 |
| `shadda_fathatan` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 41 | 0 | 0 |
| `shadda_dammatan` | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 0 | 0 | 8 | 3 |
| `shadda_kasratan` | 7 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 13 |

## Metric definitions

| metric | definition |
|---|---|
| `der_registered` | word error rate, whole '/' token, marks vs the single raw reference line (eval_der parity) |
| `der_nocase_registered` | as der_registered with the trailing case-ending run dropped |
| `wer_registered` | word error rate on exact token equality against the raw reference line |
| `preserve_registered` | share of words whose base letters equal the raw reference's |
| `der_anyalt` | word error rate, marks compared against ANY '/' alternative (registered gate semantics) |
| `der_anyalt_nocase` | as der_anyalt with the trailing case-ending run dropped |
| `der_canonical` | as der_anyalt but with each mark run rewritten to shadda-first order first, so shadda+vowel and vowel+shadda compare equal (the equivalence labels.py declares) |
| `der_canonical_primary` | as der_primary but with shadda/vowel byte order canonicalized |
| `der_primary` | word error rate of the model's FIRST alternative against the reference's FIRST alternative |
| `der_primary_nocase` | as der_primary with the trailing case-ending run dropped |
| `preserve_primary` | share of words whose first-alternative base letters equal the reference's first alternative |
| `word_accuracy_primary` | 1 - der_primary |
| `ser` | sentence exact-match rate: share of lines where EVERY word is correct under der_anyalt |
| `der_ambiguous` | word error rate restricted to words carrying '/' alternatives |
| `der_unambiguous` | word error rate restricted to words with a single reference form |
| `case_accuracy` | share of words whose FINAL base letter's class matches the reference (i'rab/case ending) |
| `case_error_rate` | 1 - case_accuracy |
| `case_contribution_to_der` | der_primary - der_primary_nocase, the case ending's share of word errors |
| `cder` | character-level diacritic error rate: (substitutions+deletions+insertions)/gold diacritic count |
| `cder_per_position` | mark edit operations divided by base-letter positions |
| `mark_precision` | correct marks / marks emitted |
| `mark_recall` | correct marks / marks in the reference |
| `mark_f1` | harmonic mean of mark_precision and mark_recall |
| `mark_substitutions` | diacritics emitted but wrong |
| `mark_deletions` | reference diacritics omitted |
| `mark_insertions` | diacritics emitted where the reference has none |
| `mark_correct` | diacritics emitted at the right position with the right class |
| `gold_marks` | classifiable diacritics in the reference (the CDER and recall denominator) |
| `pred_marks` | classifiable diacritics the model emitted (the precision and density denominator) |
| `density_pred` | marks emitted per base-letter position |
| `density_gold` | marks present in the reference per base-letter position |
| `density_ratio` | density_pred / density_gold; below 1 means systematic under-diacritization |
| `over_diacritization` | share of positions where the model marked a bare reference letter |
| `under_diacritization` | share of positions where the model left a marked reference letter bare |
| `bare_emission_rate` | share of positions the model left unmarked (a silent model scores 1.0 here) |
| `macro_f1` | unweighted mean F1 over the classes present in the reference |
| `weighted_f1` | support-weighted mean F1 over the classes present in the reference |
| `micro_f1` | equals position_accuracy for single-label multiclass |
| `position_accuracy` | share of aligned base positions whose 15-class label is correct |
| `cohen_kappa` | agreement corrected for chance agreement |
| `matthews_cc` | multiclass Matthews correlation coefficient |
| `mean_class_recall` | unweighted mean per-class recall |
| `worst_class_recall` | lowest per-class recall among classes present; catches rare-class collapse |
| `collapsed_classes` | classes present in the reference with recall exactly 0.0 |
| `alignment_coverage` | share of words whose base letters aligned, i.e. the class metrics' denominator |
| `unclassified_units` | base letters carrying a non-classifiable mark (passthrough); excluded from the confusion matrix |

## Families implemented but not computed here

- **top_k_accuracy** - implemented as eval_metrics.topk_accuracy; needs a per-position logits dump that no script produces yet (CPU-sufficient)
- **calibration_ece_nll_brier** - implemented as eval_metrics.expected_calibration_error / negative_log_likelihood / brier_score; needs the same missing logits dump (CPU-sufficient)
- **auc_one_vs_rest** - implemented as eval_metrics.auc_ovr; needs the same missing logits dump (CPU-sufficient)
- **latency_throughput** - needs the model loaded; not derivable from decoded text
- **oov_conditioned_der** - needs the training vocabulary; requires tokenizer + train split
- **noise_robustness** - needs re-inference on perturbed input
- **idempotence** - needs re-inference on the model's own output
- **human_rating** - out of scope for an automated harness
