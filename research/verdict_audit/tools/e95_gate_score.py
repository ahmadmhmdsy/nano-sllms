"""E-095 - score E-094-style arms in the REGISTERED convention, any corpus.

score_gate_bench.py:35-36 (and every other registered screen scorer) reads preds
and refs with .read_text().splitlines(). That is THE registered convention; the
number of line-SEGMENTS is not the number of pairs whenever a gold field contains
an embedded newline. E-094's inline em.score_corpus(preds, refs_all) call passed
the in-memory pair entries instead and shifted der_registered by ~0.4 pct on
BYTE-IDENTICAL predictions, so it must never be used for a registered number.

Usage:
  python e95_gate_score.py --dir runs/diac/e95_gate/fadel_test \
                           --corpus fadel_test --arms single,nospace
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402


def lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def mcnemar(pa, pb, ref):
    disc = a_only = b_only = 0
    for x, y, r in zip(pa, pb, ref):
        bx, by = em.word_outcomes(x, r), em.word_outcomes(y, r)
        if len(bx) != len(by):
            continue
        for u, v in zip(bx, by):
            if u != v:
                disc += 1
                if u and not v:
                    a_only += 1
                else:
                    b_only += 1
    n = a_only + b_only
    if n == 0:
        return {"discordant": 0, "p": 1.0}
    k = min(a_only, b_only)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    return {"discordant": disc, "a_only": a_only, "b_only": b_only,
            "p": min(1.0, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arms", default="single,nospace")
    args = ap.parse_args()

    d = Path(args.dir)
    ref = lines(d / ("%s.ref.txt" % args.corpus))
    data = {}
    for a in args.arms.split(","):
        f = d / ("%s.%s.txt" % (args.corpus, a))
        if not f.exists():
            continue
        pred = lines(f)
        if len(pred) != len(ref):
            print("  !! %s: %d pred lines vs %d ref lines" % (a, len(pred), len(ref)))
            continue
        sm = em.summarize(em.score_corpus(pred, ref))
        data[a] = {"der_registered": sm["der_registered"],
                   "der_anyalt": sm.get("der_anyalt"),
                   "der_canonical": sm.get("der_canonical"),
                   "macro_f1": sm.get("macro_f1"), "pred": pred}
    base = data.get("single", {}).get("der_registered")
    print("REGISTERED CONVENTION (%s: %d line-segments)" % (args.corpus, len(ref)))
    for a, dd in data.items():
        rel = ""
        if base:
            rel = "  rel %+.2f pct" % (100.0 * (dd["der_registered"] - base) / base)
        print("  %-10s der_registered %.6f  macro_f1 %.4f%s"
              % (a, dd["der_registered"], dd["macro_f1"], rel))
    out = {"corpus": args.corpus, "line_segments": len(ref), "arms": {}}
    for a, dd in data.items():
        out["arms"][a] = {k: v for k, v in dd.items() if k != "pred"}
    out["mcnemar"] = {}
    for x, y in (("single", "nospace"),):
        if x in data and y in data:
            out["mcnemar"]["%s_vs_%s" % (x, y)] = mcnemar(
                data[x]["pred"], data[y]["pred"], ref)
    (d / "score_registered.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["mcnemar"], indent=2))


main()
