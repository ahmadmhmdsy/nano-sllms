"""BOUND on the WORD-FORM LOOKUP route to the word-final axis (CPU, no GPU).

The oracle bound (final_unit_oracle.py) says a PERFECT word-final fixer would cut
word error 36.4 pct relative. No such fixer exists. The cheapest REAL method is a
word-form -> final-class lookup table. This bounds that route.

The table is built on the TEST data itself and scored on the same tokens. That is
DELIBERATE LEAKAGE: it is the best case any table could reach, so it is a true
UPPER bound for a train-derived table, not an estimate of one.

The bar it must clear is the model's own final-unit accuracy (1 - final_err_rate,
about 0.904 on the centered arm). A table that cannot beat that on leaked data
cannot help on clean data.
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


def final_class(w):
    u = em.units(w)
    if not u:
        return None
    return em.class_of(u[-1][1])


def main():
    ref = (DIR / "abdou_test.ref.txt").read_text(encoding="utf-8").splitlines()
    # build the leaked table: key -> Counter of gold final classes
    tab = defaultdict(Counter)
    toks = []
    for rl in ref:
        for w in rl.split():
            g = w.split("|")[0]
            k = key_of(g)
            fc = final_class(g)
            if not k or fc is None:
                continue
            tab[k][fc] += 1
            toks.append((k, fc))
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    ambig = sum(1 for c in tab.values() if len(c) > 1)

    hit = sum(1 for k, fc in toks if maj.get(k) == fc)
    # how good is the majority table restricted to keys that are UNAMBIGUOUS
    unamb = [(k, fc) for k, fc in toks if len(tab[k]) == 1]
    hit_u = sum(1 for k, fc in unamb if maj.get(k) == fc)

    # model's own final-unit accuracy on the centered arm
    pred = (DIR / "abdou_test.centered.txt").read_text(encoding="utf-8").splitlines()
    n = ok = 0
    for pl, rl in zip(pred, ref):
        for p, g in zip(pl.split(), rl.split()):
            g0 = g.split("|")[0]
            if [b for b, _ in em.units(p)] != [b for b, _ in em.units(g0)]:
                continue
            gc, pc = final_class(g0), final_class(p)
            if gc is None:
                continue
            n += 1
            ok += (gc == pc)
    model_acc = ok / max(n, 1)

    out = {
        "tokens": len(toks), "distinct_keys": len(tab),
        "ambiguous_keys": ambig,
        "ambiguous_key_share": round(ambig / max(len(tab), 1), 4),
        "leaked_table_final_acc": round(hit / max(len(toks), 1), 4),
        "unambiguous_token_share": round(len(unamb) / max(len(toks), 1), 4),
        "leaked_table_final_acc_on_unambiguous": round(hit_u / max(len(unamb), 1), 4),
        "model_final_acc": round(model_acc, 4),
        "model_final_tokens": n,
    }
    out["table_minus_model_pp"] = round(
        100 * (out["leaked_table_final_acc"] - model_acc), 2)
    out["table_unamb_minus_model_pp"] = round(
        100 * (out["leaked_table_final_acc_on_unambiguous"] - model_acc), 2)
    (DIR / "wordform_table_bound.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(f"{k}: {v}")


main()
