"""POST-HOC ORACLE BOUND on the WORD-FINAL axis (no GPU, existing artifacts).

WHY. E-093 showed the decode axis is exhausted: centering captures the whole gain
and the majority step is a no-op. The error decomposition (E-075 voted arm) named
the largest UNTOUCHED error mass: the word-final unit is 20-28 pct of units but
26-48 pct of errors. Before building any word-final fixer, bound its ceiling the
way E-088 bounded the reranker: an ORACLE that repairs every word-final class
error and nothing else.

WORD-LEVEL, because der_registered counts WORDS: a word is wrong if ANY unit is
wrong, so the oracle must be scored as "words whose ONLY class errors are final".

Arms: the three E-093 artifacts (single / vote / centered) on abdou_test, 400 pairs.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402

DIR = REPO / "runs" / "diac" / "e93_centered"


def _split_alts(w):
    return w.split("|")


def bound(pred, ref):
    W = Wwrong = only_final = inner_any = final_any = 0
    units = units_err = 0
    fin_units = fin_err = 0
    for pl, rl in zip(pred, ref):
        pw, gw = pl.split(), rl.split()
        for i, p in enumerate(pw):
            if i >= len(gw):
                break
            gold = _split_alts(gw[i])[0]
            pu, gu = em.units(p), em.units(gold)
            if [b for b, _ in pu] != [b for b, _ in gu]:
                continue                      # unaligned, excluded (same as E-075 tool)
            W += 1
            w_fin = w_inner = w_any = False
            for ui, ((_, gm), (_, pm)) in enumerate(zip(gu, pu)):
                units += 1
                gc, pc = em.class_of(gm), em.class_of(pm)
                if gc is not None and pc is not None and gc != pc:
                    units_err += 1
                    w_any = True
                    if ui == len(gu) - 1:
                        w_fin = True
                    else:
                        w_inner = True
                if ui == len(gu) - 1:
                    fin_units += 1
                    if gc is not None and pc is not None and gc != pc:
                        fin_err += 1
            if w_any:
                Wwrong += 1
                if w_fin:
                    final_any += 1
                if w_inner:
                    inner_any += 1
                if w_fin and not w_inner:
                    only_final += 1
    orc = Wwrong - only_final
    return {
        "words": W,
        "word_err": Wwrong, "word_err_rate": round(Wwrong / max(W, 1), 4),
        "units": units, "unit_err": units_err,
        "final_units": fin_units, "final_unit_err": fin_err,
        "final_units_share": round(fin_units / max(units, 1), 4),
        "final_err_share_of_errors": round(fin_err / max(units_err, 1), 4),
        "final_err_rate": round(fin_err / max(fin_units, 1), 4),
        "inner_err_rate": round((units_err - fin_err) / max(units - fin_units, 1), 4),
        "words_wrong_with_final_err": final_any,
        "words_wrong_with_inner_err": inner_any,
        "words_wrong_ONLY_final": only_final,
        "oracle_word_err": orc,
        "oracle_word_err_rate": round(orc / max(W, 1), 4),
        "oracle_rel_reduction": round(100.0 * only_final / max(Wwrong, 1), 2),
    }


def main():
    ref = (DIR / "abdou_test.ref.txt").read_text(encoding="utf-8").splitlines()
    out = {}
    for arm in ("single", "vote", "centered"):
        pred = (DIR / f"abdou_test.{arm}.txt").read_text(encoding="utf-8").splitlines()
        assert len(pred) == len(ref), (arm, len(pred), len(ref))
        b = bound(pred, ref)
        sm = em.summarize(em.score_corpus(pred, ref))
        b["der_registered"] = sm.get("der_registered")
        b["der_canonical"] = sm.get("der_canonical")
        out[arm] = b
    (DIR / "final_unit_oracle.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    for arm, b in out.items():
        print(f"--- {arm} ---")
        print(f"  der_registered {b['der_registered']}  der_canonical {b['der_canonical']}")
        print(f"  words {b['words']}  word_err {b['word_err']} ({b['word_err_rate']})")
        print(f"  final units {b['final_units']} ({b['final_units_share']} of units)"
              f"  final err {b['final_unit_err']} ({b['final_err_share_of_errors']} of errors)")
        print(f"  final_err_rate {b['final_err_rate']} vs inner {b['inner_err_rate']}")
        print(f"  words wrong ONLY-final {b['words_wrong_ONLY_final']}"
              f"  (also-inner {b['words_wrong_with_inner_err']})")
        print(f"  ORACLE word_err {b['oracle_word_err']} ({b['oracle_word_err_rate']})"
              f"  = {b['oracle_rel_reduction']} pct relative reduction")


main()
