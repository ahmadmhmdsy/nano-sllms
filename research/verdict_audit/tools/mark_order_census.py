"""Which shadda/vowel BYTE ORDER does each side actually emit?

WHY. error_decomposition.py found that ~4.5 pct of ALL units are class-correct but
byte-different, and that this convention cost is remarkably CONSTANT across corpora
(0.045-0.047) while the genuine class error varies 2x. That raises a decisive
question: is the order disagreement SYSTEMATIC (our decoder emits one order, the gold
uses the other) or INCONSISTENT (our decoder emits both)? If our own output is
inconsistent, re-encoding every mark run through its class is a FREE
post-processing win; if it is systematic, the cost is a fixed representational
convention we cannot fix without oracle knowledge of the test corpus.

SHADDA U+0651; vowels are U+064B..U+0650. A run is "shadda-first" when the shadda
byte precedes every vowel byte in it.
"""
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402
try:
    import eval_der as _ed
    _split = _ed.split_alts
except Exception:                                        # noqa: BLE001
    def _split(w):
        return [w]

SHADDA = "\u0651"
VOWELS = set("\u064b\u064c\u064d\u064e\u064f\u0650")


def order_of(run):
    if SHADDA not in run:
        return None
    vi = [i for i, ch in enumerate(run) if ch in VOWELS]
    if not vi:
        return None
    return "shadda_first" if run.index(SHADDA) < min(vi) else "vowel_first"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    d = REPO / "runs" / "diac" / "e75_vote"
    for corpus in ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024",
                   "abdou_test"]:
        pl = (d / (corpus + ".vote.txt")).read_text(encoding="utf-8").splitlines()
        rl = (d / (corpus + ".vote.ref.txt")).read_text(encoding="utf-8").splitlines()
        n = min(len(pl), len(rl))
        gold_order = Counter()
        pred_order = Counter()
        mismatch = Counter()
        n_conv = 0
        for li in range(n):
            pw = pl[li].split()
            gw = rl[li].split()
            for i, p in enumerate(pw):
                if i >= len(gw):
                    break
                gold = _split(gw[i])[0]
                pu = em.units(p)
                gu = em.units(gold)
                if [b for b, _ in pu] != [b for b, _ in gu]:
                    continue
                for (_, gm), (_, pm) in zip(gu, pu):
                    go, po = order_of(gm), order_of(pm)
                    if go:
                        gold_order[go] += 1
                    if po:
                        pred_order[po] += 1
                    if gm != pm and em.class_of(gm) == em.class_of(pm) \
                            and em.class_of(gm) is not None:
                        n_conv += 1
                        mismatch[(go, po)] += 1
        tot_g = sum(gold_order.values())
        tot_p = sum(pred_order.values())
        print(corpus, "gold_geminates=%d" % tot_g,
              {k: round(v / max(tot_g, 1), 3) for k, v in gold_order.items()},
              "| pred_geminates=%d" % tot_p,
              {k: round(v / max(tot_p, 1), 3) for k, v in pred_order.items()},
              "| convention_units=%d" % n_conv,
              {str(k): v for k, v in mismatch.most_common()}, flush=True)


if __name__ == "__main__":
    main()
