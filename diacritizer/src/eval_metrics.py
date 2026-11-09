"""Extended diacritization metric suite for the D-line (pure stdlib).

One reusable implementation of every metric the D-line reports, so a metric
is added in one place and every caller -- scripts, the pipeline_v2 adapter,
the selftest -- sees the same number. No third-party imports.

WHY CONDITIONS ARE DECLARED
---------------------------
The WikiNews benchmark references carry "/" alternatives inside a word
(one token = several valid diacritizations). "DER" is therefore ambiguous
unless the comparison condition is named. Three are implemented:

  registered  the whole "/"-containing token, marks compared against any
              single alternative. Byte-identical to eval_multi_ref.py and
              therefore to the registered gate; this is the parity anchor.
  primary     the model's FIRST alternative against the reference's FIRST
              alternative, aligned base letter by base letter. Every
              class/character metric uses this condition: it scores the
              choice the model actually made and is not distorted by
              whether the model re-diacritized the other alternatives.
  nocase      as primary with the trailing case-ending run dropped.

AGGREGATION
-----------
Word and character metrics are accumulated as integer COUNTS (correct,
total) and divided once at the end. A corpus rate is never the mean of
per-line fractions -- that is the eval_multi_ref.py bug (it under-reported
by ~words-per-line and flipped the model ranking), and counts make the
class of error unrepresentable.

CLASS METRICS AND UNCLASSIFIABLE MARKS
--------------------------------------
Only the eight consumable marks (labels._CONSUMABLE) form the 15 classes.
A base letter carrying a non-consumable mark (Quranic annotation,
superscript alef, hamza combinant) is counted in unclassified_units and
excluded from the confusion matrix, exactly as labels.py routes it to
passthrough. It is still counted by the character metrics, which compare
raw mark strings and need no class.
"""
from __future__ import annotations

import math
import random

import eval_der
import labels as L

__all__ = [
    "CLASS_NAMES", "N_CLASSES", "CONDITIONS",
    "split_alts", "units", "class_of", "align_units", "edit_ops",
    "score_line", "score_corpus", "summarize",
    "bootstrap_ci", "mcnemar", "paired_bootstrap", "word_outcomes",
    "softmax", "topk_accuracy", "negative_log_likelihood", "brier_score",
    "expected_calibration_error", "auc_ovr", "score_dump_report",
]

CLASS_NAMES = L.CLASS_NAMES
N_CLASSES = L.N_CLASSES
CONDITIONS = ("registered", "primary", "nocase")

#: The marks that can form a class (labels.py private _CONSUMABLE, rebuilt
#: here from public constants so this module never imports a private name).
CONSUMABLE = frozenset({
    L.FATHA, L.DAMMA, L.KASRA,
    L.FATHATAN, L.DAMMATAN, L.KASRATAN,
    L.SUKUN, L.SHADDA,
})


# --- word decomposition ----------------------------------------------------

def split_alts(word):
    """The "/"-separated alternatives of one gold or predicted token."""
    return word.split("/")


def units(word):
    """[(base_char, mark_run), ...] -- one entry per base letter.

    A mark that appears before any base letter attaches to a synthetic
    empty base so no character is ever dropped.
    """
    out = []
    for ch in word:
        if eval_der.is_mark(ch):
            if out:
                base, marks = out[-1]
                out[-1] = (base, marks + ch)
            else:
                out.append(("", ch))
        else:
            out.append((ch, ""))
    return out


def class_of(mark_run):
    """Class id for a mark run, or None when it is not classifiable."""
    if not mark_run:
        return L.C_BARE
    for ch in mark_run:
        if ch not in CONSUMABLE:
            return None
    try:
        return L.label_for_marks(tuple(mark_run))
    except ValueError:
        return None


def align_units(pred_word, gold_word):
    """Position-wise (gold_class, pred_class) pairs, or None if bases differ.

    Alignment is by base character, so a word whose base letters drifted is
    refused rather than silently mis-aligned; the caller counts it as
    unaligned. Classes may be None (unclassifiable) and are then excluded
    from the confusion matrix by the caller.
    """
    pu = units(pred_word)
    gu = units(gold_word)
    if [b for b, _ in pu] != [b for b, _ in gu]:
        return None
    return [(class_of(g), class_of(p)) for (_, g), (_, p) in zip(gu, pu)]


def canonicalize_word(word):
    """Rewrite every mark run in the schema's canonical (shadda-first) order.

    labels.label_for_marks accepts shadda+vowel in EITHER byte order and maps
    both to the same class -- its own docstring says "Real corpora use BOTH
    orders: shadda+vowel (canonical) and vowel+shadda (raw Tashkeela order).
    Both are the same gemination class." eval_der, however, compares raw mark
    STRINGS, so a vowel-first prediction is scored wrong against a
    shadda-first reference even though the schema calls the two identical.

    Measured effect on WikiNews 2024: mishkala is vowel-first throughout and
    loses 7.25 DER points to this alone, which is enough to change which model
    ranks first. The canonical comparison is therefore reported beside the
    registered one rather than replacing it: the registered gate stays the
    historical anchor, and this condition says what the schema means.
    """
    out = []
    for base, marks in units(word):
        if marks:
            cls = class_of(marks)
            if cls is not None:
                marks = L.marks_for_label(cls)
        out.append(base + marks)
    return "".join(out)


def marks_equal_canonical(a, b):
    """Mark-string equality after canonicalizing shadda/vowel order."""
    return eval_der.marks_of(canonicalize_word(a)) == eval_der.marks_of(canonicalize_word(b))


def edit_ops(a, b):
    """(correct, sub, dele, ins) turning mark string a into b, minimum ops."""
    la, lb = len(a), len(b)
    dp = [[None] * (lb + 1) for _ in range(la + 1)]
    dp[0][0] = (0, 0, 0)
    for j in range(1, lb + 1):
        dp[0][j] = (0, 0, j)
    for i in range(1, la + 1):
        dp[i][0] = (0, i, 0)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                best = dp[i - 1][j - 1]
            else:
                s, d, n = dp[i - 1][j - 1]
                best = (s + 1, d, n)
            cand = dp[i - 1][j]
            cand = (cand[0], cand[1] + 1, cand[2])
            if sum(cand) < sum(best):
                best = cand
            cand = dp[i][j - 1]
            cand = (cand[0], cand[1], cand[2] + 1)
            if sum(cand) < sum(best):
                best = cand
            dp[i][j] = best
    sub, dele, ins = dp[la][lb]
    return (la - sub - dele, sub, dele, ins)


# --- per-line scoring ------------------------------------------------------

def _blank_record():
    return {
        # word-level denominators and numerators
        "words": 0,
        "registered_correct": 0, "registered_nocase_correct": 0,
        "registered_wer_correct": 0, "preserve_correct": 0,
        "anyalt_correct": 0, "anyalt_nocase_correct": 0,
        "canonical_correct": 0, "canonical_primary_correct": 0,
        "primary_correct": 0, "primary_nocase_correct": 0,
        "preserve_primary_correct": 0,
        "amb_words": 0, "amb_correct": 0,
        "unamb_words": 0, "unamb_correct": 0,
        "case_words": 0, "case_correct": 0,
        "aligned_words": 0, "unaligned_words": 0,
        "exact_lines": 0,
        # character-level
        "positions": 0, "gold_marks": 0, "pred_marks": 0,
        "mark_correct": 0, "mark_sub": 0, "mark_del": 0, "mark_ins": 0,
        "pos_over": 0, "pos_under": 0, "pos_bare_pred": 0, "pos_bare_gold": 0,
        "unclassified_units": 0,
        # additive derived counters (bootstrap resamples lines, so every
        # interval target must be a per-line sum of a numerator and a
        # denominator, never a pre-divided rate)
        "line_unit": 0, "anyalt_wrong": 0, "primary_wrong": 0,
        "registered_wrong": 0, "mark_errors": 0, "canonical_wrong": 0,
    }


def score_line(pred_text, ref_text):
    """Every metric for one aligned (prediction line, reference line) pair.

    Returns (record, confusion) where confusion is a 15x15 nested list of
    integer counts indexed [gold_class][pred_class].
    """
    rec = _blank_record()
    confusion = [[0] * N_CLASSES for _ in range(N_CLASSES)]

    pw = pred_text.split()
    gw = ref_text.split()
    n = max(len(pw), len(gw), 1)
    rec["words"] = n
    rec["exact_lines"] = 1

    reg = eval_der.evaluate(pred_text, [ref_text], nocase=False)
    reg_nc = eval_der.evaluate(pred_text, [ref_text], nocase=True)
    rec["registered_correct"] = round((1.0 - reg["der"]) * reg["words"])
    rec["registered_nocase_correct"] = round((1.0 - reg_nc["der"]) * reg_nc["words"])
    rec["registered_wer_correct"] = round((1.0 - reg["wer"]) * reg["words"])
    rec["preserve_correct"] = round(reg["preserve"] * reg["words"])

    for i in range(n):
        p = pw[i] if i < len(pw) else None
        g = gw[i] if i < len(gw) else None
        if p is None or g is None:
            rec["exact_lines"] = 0
            continue

        gold_alts = split_alts(g)
        pred_alts = split_alts(p)

        anyalt = any(eval_der._pairs_ok(p, alt, nocase=False) for alt in gold_alts)
        anyalt_nc = any(eval_der._pairs_ok(p, alt, nocase=True) for alt in gold_alts)
        rec["anyalt_correct"] += int(anyalt)
        rec["anyalt_nocase_correct"] += int(anyalt_nc)
        rec["canonical_correct"] += int(
            any(marks_equal_canonical(p, alt) for alt in gold_alts))

        ambiguous = len(gold_alts) > 1
        if ambiguous:
            rec["amb_words"] += 1
            rec["amb_correct"] += int(anyalt)
        else:
            rec["unamb_words"] += 1
            rec["unamb_correct"] += int(anyalt)

        if not anyalt:
            rec["exact_lines"] = 0

        # primary condition: the model's first choice vs the first reference
        pa, ga = pred_alts[0], gold_alts[0]
        prim = eval_der._pairs_ok(pa, ga, nocase=False)
        prim_nc = eval_der._pairs_ok(pa, ga, nocase=True)
        rec["primary_correct"] += int(prim)
        rec["primary_nocase_correct"] += int(prim_nc)
        rec["canonical_primary_correct"] += int(marks_equal_canonical(pa, ga))
        rec["preserve_primary_correct"] += int(eval_der.base_of(pa) == eval_der.base_of(ga))

        pairs = align_units(pa, ga)
        if pairs is None:
            rec["unaligned_words"] += 1
            continue
        rec["aligned_words"] += 1

        pu = units(pa)
        gu = units(ga)
        rec["positions"] += len(pairs)

        for idx, (gcls, pcls) in enumerate(pairs):
            gmarks = gu[idx][1]
            pmarks = pu[idx][1]
            c, s, d, ins = edit_ops(gmarks, pmarks)
            rec["mark_correct"] += c
            rec["mark_sub"] += s
            rec["mark_del"] += d
            rec["mark_ins"] += ins
            rec["gold_marks"] += len(gmarks)
            rec["pred_marks"] += len(pmarks)
            if gmarks and not pmarks:
                rec["pos_under"] += 1
            if pmarks and not gmarks:
                rec["pos_over"] += 1
            if not pmarks:
                rec["pos_bare_pred"] += 1
            if not gmarks:
                rec["pos_bare_gold"] += 1
            if gcls is None or pcls is None:
                rec["unclassified_units"] += 1
            else:
                confusion[gcls][pcls] += 1

        # case ending = the final base letter's mark run. A token whose first
        # alternative is empty (a leading "/") aligns to zero units and has no
        # final letter, so it contributes no case observation.
        if pairs:
            rec["case_words"] += 1
            if pairs[-1][0] is not None and pairs[-1][0] == pairs[-1][1]:
                rec["case_correct"] += 1

    rec["line_unit"] = 1
    rec["anyalt_wrong"] = rec["words"] - rec["anyalt_correct"]
    rec["primary_wrong"] = rec["words"] - rec["primary_correct"]
    rec["registered_wrong"] = rec["words"] - rec["registered_correct"]
    rec["mark_errors"] = rec["mark_sub"] + rec["mark_del"] + rec["mark_ins"]
    rec["canonical_wrong"] = rec["words"] - rec["canonical_correct"]
    return rec, confusion


# --- corpus aggregation ----------------------------------------------------

def _rate(num, den):
    return num / den if den else 0.0


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def score_corpus(pred_lines, ref_lines):
    """Corpus metrics over line-aligned prediction/reference text lists."""
    total = _blank_record()
    confusion = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    line_records = []

    n_lines = max(len(pred_lines), len(ref_lines))
    for i in range(n_lines):
        p = pred_lines[i] if i < len(pred_lines) else ""
        g = ref_lines[i] if i < len(ref_lines) else ""
        rec, conf = score_line(p, g)
        line_records.append(rec)
        for k in total:
            total[k] += rec[k]
        for a in range(N_CLASSES):
            for b in range(N_CLASSES):
                confusion[a][b] += conf[a][b]

    return {"record": total, "confusion": confusion, "lines": line_records,
            "n_lines": n_lines}


def summarize(scored):
    """Flatten score_corpus output into the reported metric mapping."""
    r = scored["record"]
    conf = scored["confusion"]
    n_lines = scored["n_lines"]

    out = {}
    # --- word level -------------------------------------------------------
    out["lines"] = n_lines
    out["words"] = r["words"]
    out["der_registered"] = _rate(r["words"] - r["registered_correct"], r["words"])
    out["der_nocase_registered"] = _rate(r["words"] - r["registered_nocase_correct"], r["words"])
    out["wer_registered"] = _rate(r["words"] - r["registered_wer_correct"], r["words"])
    out["preserve_registered"] = _rate(r["preserve_correct"], r["words"])
    out["der_anyalt"] = _rate(r["words"] - r["anyalt_correct"], r["words"])
    out["der_anyalt_nocase"] = _rate(r["words"] - r["anyalt_nocase_correct"], r["words"])
    out["der_canonical"] = _rate(r["words"] - r["canonical_correct"], r["words"])
    out["der_canonical_primary"] = _rate(r["words"] - r["canonical_primary_correct"], r["words"])
    out["der_primary"] = _rate(r["words"] - r["primary_correct"], r["words"])
    out["der_primary_nocase"] = _rate(r["words"] - r["primary_nocase_correct"], r["words"])
    out["preserve_primary"] = _rate(r["preserve_primary_correct"], r["words"])
    out["word_accuracy_primary"] = _rate(r["primary_correct"], r["words"])
    out["ser"] = _rate(r["exact_lines"], n_lines)
    out["der_ambiguous"] = _rate(r["amb_words"] - r["amb_correct"], r["amb_words"])
    out["der_unambiguous"] = _rate(r["unamb_words"] - r["unamb_correct"], r["unamb_words"])
    out["ambiguous_words"] = r["amb_words"]
    out["case_accuracy"] = _rate(r["case_correct"], r["case_words"])
    out["case_error_rate"] = 1.0 - out["case_accuracy"]
    out["case_contribution_to_der"] = out["der_primary"] - out["der_primary_nocase"]

    # --- character / mark level ------------------------------------------
    out["positions"] = r["positions"]
    out["aligned_words"] = r["aligned_words"]
    out["unaligned_words"] = r["unaligned_words"]
    out["alignment_coverage"] = _rate(r["aligned_words"], r["words"])
    errs = r["mark_sub"] + r["mark_del"] + r["mark_ins"]
    out["cder"] = _rate(errs, r["gold_marks"])
    out["cder_per_position"] = _rate(errs, r["positions"])
    out["mark_precision"] = _rate(r["mark_correct"], r["pred_marks"])
    out["mark_recall"] = _rate(r["mark_correct"], r["gold_marks"])
    out["mark_f1"] = _f1(out["mark_precision"], out["mark_recall"])
    out["mark_substitutions"] = r["mark_sub"]
    out["mark_deletions"] = r["mark_del"]
    out["mark_insertions"] = r["mark_ins"]
    out["mark_correct"] = r["mark_correct"]
    # emit the two denominators as well: without them a consumer cannot tell a
    # small CDER caused by few gold marks from one caused by few errors, and the
    # density ratios cannot be recomputed from the artifact alone.
    out["gold_marks"] = r["gold_marks"]
    out["pred_marks"] = r["pred_marks"]
    out["density_pred"] = _rate(r["pred_marks"], r["positions"])
    out["density_gold"] = _rate(r["gold_marks"], r["positions"])
    out["density_ratio"] = _rate(r["pred_marks"], r["gold_marks"])
    out["over_diacritization"] = _rate(r["pos_over"], r["positions"])
    out["under_diacritization"] = _rate(r["pos_under"], r["positions"])
    out["bare_emission_rate"] = _rate(r["pos_bare_pred"], r["positions"])
    out["bare_gold_rate"] = _rate(r["pos_bare_gold"], r["positions"])
    out["unclassified_units"] = r["unclassified_units"]

    # --- 15-class confusion-derived --------------------------------------
    total_c = sum(sum(row) for row in conf)
    gold_sup = [sum(conf[a]) for a in range(N_CLASSES)]
    pred_sup = [sum(conf[a][b] for a in range(N_CLASSES)) for b in range(N_CLASSES)]
    diag = sum(conf[c][c] for c in range(N_CLASSES))

    per_class = []
    for c in range(N_CLASSES):
        tp = conf[c][c]
        fp = pred_sup[c] - tp
        fn = gold_sup[c] - tp
        prec = _rate(tp, tp + fp)
        rec_ = _rate(tp, tp + fn)
        per_class.append({
            "class": CLASS_NAMES[c], "index": c,
            "precision": prec, "recall": rec_, "f1": _f1(prec, rec_),
            "support": gold_sup[c], "predicted": pred_sup[c],
        })

    present = [pc for pc in per_class if pc["support"] > 0]
    out["confusion_total"] = total_c
    out["position_accuracy"] = _rate(diag, total_c)
    out["macro_f1"] = _rate(sum(pc["f1"] for pc in present), len(present))
    out["weighted_f1"] = _rate(
        sum(pc["f1"] * pc["support"] for pc in present), sum(pc["support"] for pc in present))
    out["micro_f1"] = out["position_accuracy"]
    out["classes_present"] = len(present)
    out["mean_class_recall"] = _rate(sum(pc["recall"] for pc in present), len(present))
    out["worst_class_recall"] = min((pc["recall"] for pc in present), default=0.0)
    out["worst_class"] = min(present, key=lambda pc: pc["recall"])["class"] if present else None
    out["collapsed_classes"] = sorted(
        pc["class"] for pc in present if pc["recall"] == 0.0)

    if total_c:
        pe = sum(gold_sup[c] * pred_sup[c] for c in range(N_CLASSES)) / (total_c * total_c)
        out["cohen_kappa"] = (out["position_accuracy"] - pe) / (1.0 - pe) if pe < 1.0 else 0.0
        num = diag * total_c - sum(pred_sup[c] * gold_sup[c] for c in range(N_CLASSES))
        d1 = total_c * total_c - sum(v * v for v in pred_sup)
        d2 = total_c * total_c - sum(v * v for v in gold_sup)
        out["matthews_cc"] = num / math.sqrt(d1 * d2) if d1 > 0 and d2 > 0 else 0.0
    else:
        out["cohen_kappa"] = 0.0
        out["matthews_cc"] = 0.0

    out["per_class"] = per_class
    out["confusion"] = conf
    return out


# --- uncertainty and paired tests -----------------------------------------

def bootstrap_ci(line_records, num_key, den_key, n_boot=2000, seed=42, alpha=0.05):
    """Percentile CI for a ratio, resampling LINES with replacement.

    Lines are the independent unit (a sentence is annotated as a whole), so
    the bootstrap resamples lines and re-weights by that line's own
    denominator rather than resampling words, which would understate the
    interval by ignoring within-sentence correlation.
    """
    rng = random.Random(seed)
    usable = [r for r in line_records if r[den_key]]
    if not usable:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": 0, "lines": 0}
    point = _rate(sum(r[num_key] for r in usable), sum(r[den_key] for r in usable))
    m = len(usable)
    stats = []
    for _ in range(n_boot):
        num = 0
        den = 0
        for _ in range(m):
            r = usable[rng.randrange(m)]
            num += r[num_key]
            den += r[den_key]
        stats.append(num / den if den else 0.0)
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * n_boot) - 1)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot, "lines": m}


def word_outcomes(pred_text, ref_text, nocase=False, canonical=False):
    """Per-word correctness bits for one line (any-alternative condition).

    canonical=True uses the same decision as the der_canonical condition
    (shadda/vowel byte order canonicalized first). It exists so a paired test
    can be run under the SAME condition as the difference it is reported
    beside: pairing a canonical difference with a raw-anyalt contingency
    silently reports a p-value for a different comparison than the one the
    table claims.
    """
    pw = pred_text.split()
    gw = ref_text.split()
    n = max(len(pw), len(gw))
    bits = []
    for i in range(n):
        p = pw[i] if i < len(pw) else None
        g = gw[i] if i < len(gw) else None
        if p is None or g is None:
            bits.append(False)
            continue
        if canonical:
            bits.append(any(marks_equal_canonical(p, alt) for alt in split_alts(g)))
        else:
            bits.append(any(eval_der._pairs_ok(p, alt, nocase=nocase) for alt in split_alts(g)))
    return bits


def mcnemar(pred_lines_a, pred_lines_b, ref_lines, nocase=False, canonical=False):
    """Exact two-sided McNemar over paired per-word outcomes of two systems.

    b = A correct while B is wrong; c = the reverse. The p-value is the exact
    binomial on min(b, c) out of b + c. The chi-square approximation is not
    used: at the discordant counts a real model comparison produces it is
    unreliable, and pairing is what makes the test valid -- comparing two
    independent intervals instead would ignore that both systems saw the
    same sentences.
    """
    b = c = both = neither = 0
    for i in range(max(len(pred_lines_a), len(pred_lines_b), len(ref_lines))):
        pa = pred_lines_a[i] if i < len(pred_lines_a) else ""
        pb = pred_lines_b[i] if i < len(pred_lines_b) else ""
        g = ref_lines[i] if i < len(ref_lines) else ""
        oa = word_outcomes(pa, g, nocase=nocase, canonical=canonical)
        ob = word_outcomes(pb, g, nocase=nocase, canonical=canonical)
        m = max(len(oa), len(ob))
        oa = oa + [False] * (m - len(oa))
        ob = ob + [False] * (m - len(ob))
        for x, y in zip(oa, ob):
            if x and y:
                both += 1
            elif x and not y:
                b += 1
            elif y and not x:
                c += 1
            else:
                neither += 1
    n = b + c
    if n == 0:
        p = 1.0
    else:
        # 2 * P(X <= min(b, c)) for X ~ Binomial(n, 1/2), accumulated in log
        # space. The direct form (sum of math.comb over 2**n) overflows the
        # float conversion as soon as the discordant count reaches ~1000,
        # which every real model comparison here does.
        k = min(b, c)
        lg = math.lgamma
        logs = [lg(n + 1) - lg(i + 1) - lg(n - i + 1) - n * math.log(2.0)
                for i in range(k + 1)]
        top = max(logs)
        p = min(1.0, 2.0 * math.exp(top) * sum(math.exp(v - top) for v in logs))
    return {"b_a_correct_only": b, "c_b_correct_only": c,
            "both_correct": both, "neither_correct": neither,
            "discordant": n, "p_value_two_sided": p,
            "significant_at_0.05": bool(p < 0.05)}


def paired_bootstrap(records_a, records_b, num_key, den_key, n_boot=2000, seed=42, alpha=0.05):
    """Paired line bootstrap of the metric DIFFERENCE between two systems.

    The same resampled line indices are used for both systems, which is what
    makes the interval a paired one: shared sentence difficulty cancels
    instead of inflating the difference's variance.
    """
    rng = random.Random(seed)
    usable = [(ra, rb) for ra, rb in zip(records_a, records_b) if ra[den_key]]
    if not usable:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": 0, "lines": 0,
                "excludes_zero": False}

    def _diff(pairs):
        na = sum(ra[num_key] for ra, _ in pairs)
        nb = sum(rb[num_key] for _, rb in pairs)
        den = sum(ra[den_key] for ra, _ in pairs)
        return (na - nb) / den if den else 0.0

    point = _diff(usable)
    m = len(usable)
    stats = []
    for _ in range(n_boot):
        sample = [usable[rng.randrange(m)] for _ in range(m)]
        stats.append(_diff(sample))
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * n_boot) - 1)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot, "lines": m,
            "excludes_zero": (lo > 0.0) or (hi < 0.0)}

# --- metrics that need model scores, not just decoded text ----------------
#
# Everything above is computable from decoded text alone. The families below
# need the per-position class SCORES the model produced, because a hard
# argmax discards exactly what they measure: how confident the model was and
# whether the runner-up was close. They are implemented here so the harness
# is complete and reusable; producing the score dump needs one forward pass
# over the evaluation split, which is GPU work.
#
# Dump format (one JSON object per line, written by an inference script):
#   {"gold": [int, ...], "logits": [[float, ... 15], ...]}

def softmax(row):
    """Numerically stable softmax of one score row."""
    if not row:
        return []
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [v / s for v in ex] if s else [0.0] * len(row)


def topk_accuracy(logit_rows, gold, ks=(1, 2, 3)):
    """Fraction of positions whose gold class is in the model's top k."""
    hits = {k: 0 for k in ks}
    n = 0
    for row, g in zip(logit_rows, gold):
        if g is None or g < 0 or not row:
            continue
        n += 1
        order = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
        for k in ks:
            if g in order[:k]:
                hits[k] += 1
    return {"n": n, **{"top%d" % k: (hits[k] / n if n else 0.0) for k in ks}}


def negative_log_likelihood(logit_rows, gold):
    """Mean -log p(gold): the proper scoring rule the model is trained on."""
    tot = 0.0
    n = 0
    for row, g in zip(logit_rows, gold):
        if g is None or g < 0 or not row:
            continue
        p = softmax(row)[g]
        tot += -math.log(max(p, 1e-12))
        n += 1
    return tot / n if n else 0.0


def brier_score(logit_rows, gold):
    """Mean multiclass Brier: sum over classes of (p - onehot)^2."""
    tot = 0.0
    n = 0
    for row, g in zip(logit_rows, gold):
        if g is None or g < 0 or not row:
            continue
        p = softmax(row)
        tot += sum((p[c] - (1.0 if c == g else 0.0)) ** 2 for c in range(len(p)))
        n += 1
    return tot / n if n else 0.0


def expected_calibration_error(logit_rows, gold, bins=15):
    """ECE over confidence bins: |accuracy - mean confidence|, weighted by bin.

    Reported alongside the raw reliability curve because a single ECE number
    hides the direction of the miscalibration (over- vs under-confident).
    """
    buckets = [[0, 0.0, 0] for _ in range(bins)]  # count, sum_conf, hits
    n = 0
    for row, g in zip(logit_rows, gold):
        if g is None or g < 0 or not row:
            continue
        p = softmax(row)
        conf = max(p)
        pred = p.index(conf)
        b = min(bins - 1, int(conf * bins))
        buckets[b][0] += 1
        buckets[b][1] += conf
        buckets[b][2] += int(pred == g)
        n += 1
    ece = 0.0
    curve = []
    for i, (cnt, sc, hit) in enumerate(buckets):
        if not cnt:
            curve.append({"bin": i, "n": 0})
            continue
        acc = hit / cnt
        avg = sc / cnt
        ece += (cnt / n) * abs(acc - avg) if n else 0.0
        curve.append({"bin": i, "n": cnt, "accuracy": acc, "confidence": avg,
                      "gap": acc - avg})
    return {"ece": ece, "n": n, "curve": curve}


def auc_ovr(logit_rows, gold, cls):
    """One-vs-rest ROC-AUC for one class, rank based (ties averaged).

    Rank based rather than threshold swept so the value does not depend on a
    chosen operating point, which is the point of reporting AUC at all.
    """
    pos, neg = [], []
    for row, g in zip(logit_rows, gold):
        if g is None or g < 0 or not row:
            continue
        (pos if g == cls else neg).append(row[cls])
    if not pos or not neg:
        return None
    allv = sorted(pos + neg)
    rank_of = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        rank_of[allv[i]] = avg
        i = j + 1
    rsum = sum(rank_of[v] for v in pos)
    return (rsum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def score_dump_report(path, bins=15):
    """Full score-dependent report from a JSONL logits dump (see header)."""
    import json as _json
    rows, golds = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            rows.extend(obj["logits"])
            golds.extend(obj["gold"])
    out = {
        "positions": len(golds),
        "topk": topk_accuracy(rows, golds),
        "nll": negative_log_likelihood(rows, golds),
        "brier": brier_score(rows, golds),
        "auc_ovr": {CLASS_NAMES[c]: auc_ovr(rows, golds, c) for c in range(N_CLASSES)},
    }
    out["calibration"] = expected_calibration_error(rows, golds, bins=bins)
    return out

