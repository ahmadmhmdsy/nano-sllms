"""Score the E-094 arms in the REGISTERED convention.

score_gate_bench.py:35-36 reads preds and refs with .splitlines(), which for
abdou_test yields 1065 line-SEGMENTS from 400 pairs (the gold fields contain
embedded newlines). E-094's inline scoring passed the 400 in-memory entries
instead, which shifts der_registered by ~0.4 pct on byte-identical predictions.
This re-scores the SAME written files the registered way so the numbers are
comparable to every other D-line number.
"""
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402

DIR = REPO / "runs" / "diac" / "e94_nospace"
ARMS = ["single", "nospace", "centered", "nospace_centered"]


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
    ref = lines(DIR / "abdou_test.ref.txt")
    data = {}
    for a in ARMS:
        f = DIR / f"abdou_test.{a}.txt"
        if not f.exists():
            continue
        pred = lines(f)
        if len(pred) != len(ref):
            print(f"  !! {a}: {len(pred)} pred lines vs {len(ref)} ref lines")
            continue
        sm = em.summarize(em.score_corpus(pred, ref))
        data[a] = {"der_registered": sm["der_registered"],
                   "der_anyalt": sm.get("der_anyalt"),
                   "der_canonical": sm.get("der_canonical"),
                   "macro_f1": sm.get("macro_f1"), "pred": pred}
    base = data.get("single", {}).get("der_registered")
    print("REGISTERED CONVENTION (splitlines, %d line-segments)" % len(ref))
    for a, d in data.items():
        rel = ""
        if base:
            rel = "  rel %+.2f pct" % (100.0 * (d["der_registered"] - base) / base)
        print("  %-18s der_registered %.6f  macro_f1 %.4f%s"
              % (a, d["der_registered"], d["macro_f1"], rel))
    out = {"line_segments": len(ref), "arms": {}}
    for a, d in data.items():
        out["arms"][a] = {k: v for k, v in d.items() if k != "pred"}
    out["mcnemar"] = {}
    for x, y in (("single", "nospace"), ("centered", "nospace_centered"),
                 ("single", "centered"), ("nospace", "nospace_centered")):
        if x in data and y in data:
            out["mcnemar"][f"{x}_vs_{y}"] = mcnemar(
                data[x]["pred"], data[y]["pred"], ref)
    (DIR / "e94_score_registered.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["mcnemar"], indent=2))


main()
