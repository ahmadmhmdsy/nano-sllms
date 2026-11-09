"""END-TO-END BOUND for the hybrid: model + word-form table override on CERTAIN keys.

The previous script compared the table's accuracy on unambiguous keys against the
model's accuracy on ALL keys - apples to oranges. This measures the actual
end-to-end effect of the real candidate arm:

  decode with the model, then for every word whose (stripped) form is
  UNAMBIGUOUS in the table and seen >= K times, overwrite the FINAL unit's mark
  with the table's majority mark.

Table is built on the TEST ref (deliberate leakage), so every number here is an
UPPER bound on what a train-derived table could do. If the leaked version does not
clear the 2 pct bar, the route is priced out and no train-split work is justified.

Baseline: the centered arm, der_registered 0.189054, word_err 7435/40532.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402

DIR = REPO / "runs" / "diac" / "e93_centered"
MARKS = set("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670")


def key_of(w):
    return "".join(c for c in w if c not in MARKS)


def rebuild(units, last_mark):
    if not units:
        return ""
    out = [b + m for b, m in units[:-1]]
    out.append(units[-1][0] + last_mark)
    return "".join(out)


def main():
    ref = (DIR / "abdou_test.ref.txt").read_text(encoding="utf-8").splitlines()
    pred = (DIR / "abdou_test.centered.txt").read_text(encoding="utf-8").splitlines()

    tab = defaultdict(Counter)
    for rl in ref:
        for w in rl.split():
            g = w.split("|")[0]
            k = key_of(g)
            u = em.units(g)
            if k and u:
                tab[k][u[-1][1]] += 1

    base = em.summarize(em.score_corpus(pred, ref))
    print(f"baseline der_registered {base.get('der_registered')}")

    results = {}
    for K in (1, 2, 3, 5):
        maj = {k: c.most_common(1)[0][0]
               for k, c in tab.items() if len(c) == 1 and sum(c.values()) >= K}
        overridden = 0
        newpred = []
        for pl, rl in zip(pred, ref):
            pw = pl.split()
            outw = []
            for p in pw:
                k = key_of(p)
                if k in maj:
                    u = em.units(p)
                    if u:
                        p2 = rebuild(u, maj[k])
                        if p2 != p:
                            overridden += 1
                        outw.append(p2)
                        continue
                outw.append(p)
            newpred.append(" ".join(outw))
        sm = em.summarize(em.score_corpus(newpred, ref))
        der = sm.get("der_registered")
        rel = 100.0 * (der - base.get("der_registered")) / base.get("der_registered")
        results[K] = {
            "keys_in_table": len(maj), "words_overridden": overridden,
            "der_registered": der, "rel_pct": round(rel, 3),
            "passes_2pct_bar": rel <= -2.0,
        }
        print(f"K={K}: keys {len(maj):5d} overridden {overridden:5d} "
              f"der {der:.6f} rel {rel:+.3f} pct  bar={'PASS' if rel <= -2.0 else 'FAIL'}")

    (DIR / "wordform_hybrid_bound.json").write_text(
        json.dumps({"baseline_der": base.get("der_registered"), "variants": results},
                   indent=2), encoding="utf-8")


main()
