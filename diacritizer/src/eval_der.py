"""DER + WER (+- case endings) + text-preservation hard gate (R52, DESIGN 5).

QCRI-EvalDiac-compatible semantics, position-wise word alignment:
  - case-ending marks = the trailing mark run of the final letter of a word
  - a word counts CORRECT for DER if its mark string matches ANY reference
    (multi-reference support for WikiNews/SadeedDiac benchmark files)
  - DER without case endings drops that trailing run on both sides first
  - WER counts a word pair as error when base letters differ (base drift is a
    hallucination and separately caught by the preservation gate) or when the
    full diacritic string differs from the PRIMARY reference
  - text_preservation = share of words with identical base letters (hard gate;
    ~1.0 by construction in our pipeline; any drop must fail the report)

Equal-length alignment: extra/missing words count as errors, never dropped.
Line-length mismatch between prediction and reference is reported via the
returned alignment length (n_words difference), never silently ignored.
Marks are handled as ASCII-range ints (Arabic diacritic block 0x064B-0x0652).
"""
MARK_LO, MARK_HI = 0x064B, 0x0652  # fathatan .. sukun (the 8 label marks)


def is_mark(ch):
    return MARK_LO <= ord(ch) <= MARK_HI


def base_of(word):
    return "".join(c for c in word if not is_mark(c))


def marks_of(word):
    return "".join(c for c in word if is_mark(c))


def strip_case_marks(word):
    """Remove the ENTIRE trailing mark run (final letter's full label)."""
    end = len(word)
    while end > 0 and is_mark(word[end - 1]):
        end -= 1
    return word[:end]


def _pairs_ok(pred_word, ref_word, nocase):
    if pred_word is None or ref_word is None:
        return False
    if nocase:
        return strip_case_marks(pred_word) == strip_case_marks(ref_word)
    return marks_of(pred_word) == marks_of(ref_word)


def evaluate(pred_text, refs, nocase=False):
    """refs: list of gold texts (>= 1; extra = multi-reference).

    Returns dict: DER	error-rate semantics scaled as fractions in [0,1].
    """
    pw = pred_text.split()
    rows = []
    max_len = max([len(pw)] + [len(r.split()) for r in refs])
    for i in range(max_len):
        p = pw[i] if i < len(pw) else None
        cand = [(r.split()[i] if i < len(r.split()) else None) for r in refs]
        der_ok = any(_pairs_ok(p, c, nocase) for c in cand)
        g = cand[0]
        base_ok = (p is not None and g is not None and base_of(p) == base_of(g))
        wer_ok = (p is not None and g is not None and p == g)
        rows.append((int(der_ok), int(base_ok), int(wer_ok)))
    n = max(len(rows), 1)
    der = 1.0 - sum(r[0] for r in rows) / n
    preservation = sum(r[1] for r in rows) / n
    wer = 1.0 - sum(r[2] for r in rows) / n
    return {"der": der, "preserve": preservation, "wer": wer, "words": n,
            "len_pred": len(pw)}


def score_line(pred_text, refs):
    """Both flavors at once -> (with-case, without-case) metric dicts."""
    a = evaluate(pred_text, refs, nocase=False)
    b = evaluate(pred_text, refs, nocase=True)
    return {"DER": a["der"], "DER_nocase": b["der"], "WER": a["wer"],
            "text_preservation": a["preserve"], "words": a["words"],
            "length_mismatch": a["len_pred"] != (len(refs[0].split()) if refs else 0)}


def aggregate(list_of):
    """Word-weighted mean of metric dicts (the benchmark-pack report format)."""
    total = {"DER": 0.0, "DER_nocase": 0.0, "WER": 0.0, "text_preservation": 0.0}
    m = {"DER": "DER", "DER_nocase": "DER_nocase", "WER": "WER",
         "text_preservation": "text_preservation"}
    ntotal = 0
    for s in list_of:
        w = s["words"]
        for k in total:
            total[m[k]] += s[m[k]] * w
        ntotal += w
    ntotal = max(ntotal, 1)
    return {k: v / ntotal for k, v in total.items()} | {"words": ntotal}
