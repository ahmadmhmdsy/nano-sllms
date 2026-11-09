
"""mex/scripts/mark_metrics.py - E-55b/c: full per-prompt metric profile.

Standard bench metric set (run for ANY model line on the diacritizer gates; the
user approved always reporting the full profile alongside strict DER/WER):

  position-level (LCS fair-aligned per word):
      mark hit / wrong / missed / extra
      mark_accuracy = recall over ref marks; mark_precision counts spurious;
      mark_f1
  word-level (over words that carry >=1 ref mark):
      word_ax        = mark-exact words share (strict DER restricted to marked words)
      word_partial50 = words with >=50% of ref marks hit (allowing <=25% spurious)
      der_collapse   = word DER after stripping shadda+voiced compounds and
                       marks outside 0x064B..0x0652 (labels lenience)
      contrastive_lift = zero-mark-baseline DER minus der_collapse(the model) on
                       the same lines: lift over "emit bare text"
  buckets (mark accuracy by host position of the ref marks in the word):
      first / middle / last base char of the word

Per-prompt CSV rows: line, words, marked_words, word_ax, word_p50, hit, wrong,
missed, extra, der_collapse_ok, der_words + separate strict CSV from eval.py.
CPU-only. Usage: python -X utf8 mex/scripts/mark_metrics.py <pred.txt> <gate>
"""
import csv, json, sys
import difflib
from pathlib import Path

REPO = Path("E:/python_projects/nano_SLMs")
MARK_RANGES = ((0x064B, 0x0652), (0x0670, 0x0670),
               (0x0653, 0x0653), (0x0654, 0x0654), (0x0655, 0x0655),
               (0x0656, 0x0656), (0x0657, 0x0657), (0x0658, 0x0658),
               (0x0659, 0x0659), (0x065F, 0x065F), (0x06D6, 0x06ED))

def is_mark(ch):
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in MARK_RANGES)

def base_of(w):
    return "".join(c for c in w if not is_mark(c))

def marks_of(w):
    return [c for c in w if is_mark(c)]

def lcs_len(a, b):
    return sum(blk.size for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())

def collapse_marks(w):
    out = []
    i = 0
    while i < len(w):
        c = w[i]
        if ord(c) == 0x0651 and i + 1 < len(w) and 0x064B <= ord(w[i + 1]) <= 0x0652:
            i += 1
        elif (not is_mark(c)) or 0x064B <= ord(c) <= 0x0652:
            out.append(c)
        i += 1
    return "".join(out)

def base_count(w):
    return sum(1 for c in w if not is_mark(c))

def host_bucket(g, mark_host):
    """Relate the base char index of the ref marks' host to the word length."""
    host_rel = 0
    if g and any(not is_mark(c) for c in g):
        idx = 0
        for k, c in enumerate(g):
            if not is_mark(c):
                idx += 1
                if k == mark_host:
                    host_rel = idx
    # if mark_host == 0, host_rel stays 0 -> treated as first
    last_first_len = host_rel
    return host_rel

def main():
    pred_p = Path(sys.argv[1])
    ref_p = pred_p.with_suffix(".ref.txt")
    gate = sys.argv[2] if len(sys.argv) > 2 else pred_p.stem
    preds = pred_p.read_text(encoding="utf-8").splitlines()
    refs = ref_p.read_text(encoding="utf-8").splitlines()
    T = dict(hit=0, wrong=0, missed=0, extra=0, ref_marks=0,
             w_marked=0, w_ax=0, w_p50=0, wc_ok=0, wc_n=0,
             b_first=[0, 0], b_mid=[0, 0], b_last=[0, 0])
    csvp = pred_p.with_name(pred_p.stem + "_perline.csv")
    brown = csv.writer
    with csvp.open("w", encoding="utf-8", newline="") as fh:
        w = brown(fh)
        w.writerow(["line", "words", "marked_words", "word_ax", "word_p50",
                    "mark_hit", "mark_wrong", "mark_missed", "mark_extra",
                    "der_collapse_ok", "der_words"])
        for ln, line in enumerate(preds):
            pred = line.rstrip("\t")
            if not pred.strip():
                continue
            ref = refs[ln].split("\t")[0]
            pp, rp = pred.split(), ref.split()
            nax = max(len(pp), len(rp))
            lh = lw = lmiss = lex = lrefm = wax = wp50 = marked = dok = 0
            for i in range(nax):
                p = pp[i] if i < len(pp) else None
                g = rp[i] if i < len(rp) else None
                if p is None or g is None or base_of(p) != base_of(g):
                    continue
                pm, gm = marks_of(p), marks_of(g)
                h = lcs_len(pm, gm)
                sub = max(0, min(len(pm), len(gm)) - h)
                ins = max(0, len(pm) - len(gm) - sub)
                de = max(0, len(gm) - len(pm) - sub)
                lh += h; lw += sub; lmiss += de; lex += ins
                if gm:
                    marked += 1
                    lrefm += len(gm)
                    frac = h / len(gm)
                    if h == len(gm) and ins == 0:
                        wax += 1
                    if frac >= 0.5 and ins <= max(1, len(gm) // 4):
                        wp50 += 1
                    # bucket: host char index of the marks inside the word
                    host_idx = 0
                    for k, c in enumerate(g):
                        if not is_mark(c):
                            host_idx = k
                    tb = base_count(g)
                    rel = 0 if tb <= 1 or host_idx == 0 else (
                        2 if host_idx == tb - 1 else 1)
                    if rel == 0:
                        T["b_first"][0] += h; T["b_first"][1] += len(gm)
                    elif rel == 1:
                        T["b_mid"][0] += h; T["b_mid"][1] += len(gm)
                    else:
                        T["b_last"][0] += h; T["b_last"][1] += len(gm)
                if collapse_marks(p) == collapse_marks(g):
                    dok += 1
            T["hit"] += lh; T["wrong"] += lw; T["missed"] += lmiss; T["extra"] += lex
            T["ref_marks"] += lrefm; T["w_marked"] += marked; T["w_ax"] += wax
            T["w_p50"] += wp50; T["wc_n"] += nax if False else len(pp); T["wc_ok"] += dok
            w.writerow([ln, len(pp), marked, wax, wp50, lh, lw, lmiss, lex, dok, len(pp)])
    hit, wrong, missed, extra, tm = T["hit"], T["wrong"], T["missed"], T["extra"], T["ref_marks"]
    prec = hit / max(hit + wrong + extra, 1)
    rec = hit / max(hit + wrong + missed, 1)
    # contrastive lift: baseline = predict no marks at all. A word is "correct"
    # under the baseline iff the ref word carries no mark. denon of wc_n covers
    # all aligned words; ref-marked words are exactly the baseline errors.
    words_total = T["wc_n"]
    baseline_der = T["w_marked"] / max(words_total, 1)
    model_der_c = 1 - T["wc_ok"] / max(words_total, 1)
    rep = {"pred": str(pred_p), "gate": gate, "words": words_total,
           "ref_marks": tm, "marked_words": T["w_marked"],
           "mark_hit": hit, "mark_wrong": wrong, "mark_missed": missed, "mark_extra": extra,
           "mark_accuracy": round(rec, 4), "mark_precision": round(prec, 4),
           "mark_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
           "word_ax": round(T["w_ax"] / max(T["w_marked"], 1), 4),
           "word_partial50": round(T["w_p50"] / max(T["w_marked"], 1), 4),
           "mark_acc_first": round(T["b_first"][0] / max(T["b_first"][1], 1), 4),
           "mark_acc_mid": round(T["b_mid"][0] / max(T["b_mid"][1], 1), 4),
           "mark_acc_last": round(T["b_last"][0] / max(T["b_last"][1], 1), 4),
           "der_collapse_1minus": round(model_der_c, 4),
           "zero_mark_baseline_der": round(baseline_der, 4),
           "contrastive_lift": round(baseline_der - model_der_c, 4)}
    (pred_p.with_suffix(".metrics.json")).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=True))

main()
