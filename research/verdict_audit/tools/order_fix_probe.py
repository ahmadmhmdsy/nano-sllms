"""Decisive probe: does re-rendering our detokenizer vowel-first turn the
REGISTERED metric into the canonical one?

mark_order_census.py found gold 100 pct vowel-first and our output 100 pct
shadda-first on every registered corpus. marks_for_label() renders shadda-first
(labels.py:74), and the head is 15-class - it CANNOT express byte order. So the
whole convention cost should be recoverable by post-processing alone, with no
retraining. This probe re-renders every predicted mark run vowel-first and
re-scores with the UNCHANGED registered metric. If der_registered lands on
der_canonical, the claim is confirmed.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402
import labels as L  # noqa: E402

SHADDA = "\u0651"


def vowel_first_line(line):
    out = []
    for word in line.split():
        w = []
        for base, marks in em.units(word):
            cls = em.class_of(marks)
            if cls is not None:
                s = L.marks_for_label(cls)
                if s.startswith(SHADDA) and len(s) > 1:
                    s = s[1:] + s[0]          # shadda-first -> vowel-first
                marks = s
            w.append(base + marks)
        out.append("".join(w))
    return " ".join(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    d = REPO / "runs" / "diac" / "e75_vote"
    for corpus in ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024",
                   "abdou_test"]:
        pl = (d / (corpus + ".vote.txt")).read_text(encoding="utf-8").splitlines()
        rl = (d / (corpus + ".vote.ref.txt")).read_text(encoding="utf-8").splitlines()
        n = min(len(pl), len(rl))
        pl, rl = pl[:n], rl[:n]
        base = em.summarize(em.score_corpus(pl, rl))
        vf = [vowel_first_line(x) for x in pl]
        fixed = em.summarize(em.score_corpus(vf, rl))
        print(json_line(corpus, base["der_registered"], fixed["der_registered"],
                        base["der_canonical"], base["der_anyalt"]), flush=True)
        if corpus == "fadel_test":
            for i, (p, r) in enumerate(zip(pl, rl)):
                if SHADDA in p and SHADDA in r:
                    print("  RAW gold :", [hex(ord(c)) for c in r[:60]
                                           if ord(c) > 0x600][:12])
                    print("  RAW pred :", [hex(ord(c)) for c in p[:60]
                                           if ord(c) > 0x600][:12])
                    print("  FIX pred :", [hex(ord(c)) for c in vf[i][:60]
                                           if ord(c) > 0x600][:12])
                    break


def json_line(corpus, base, fixed, canon, anyalt):
    rel = (fixed - base) / base * 100.0
    return ("%-14s registered=%.6f  order_fixed=%.6f  (rel %+.2f pct)  "
            "canonical=%.6f  anyalt=%.6f" % (corpus, base, fixed, rel, canon,
                                             anyalt))


if __name__ == "__main__":
    main()
