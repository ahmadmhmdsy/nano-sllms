"""Error decomposition of the DEPLOYED E-075 voted arm.

WHY. E-088 proved that 71-80 pct of all errors sit on words where all seven
E-083 schemes unanimously agree, so the remaining error is SYSTEMATIC, not a
matter of disagreement. Combination, window, context and decoding are all closed
by measurement. The open question is therefore WHAT the model is systematically
getting wrong, because that decides which training lever (capacity, data, or
objective) is worth a real GPU run.

WHAT IT REPORTS, per corpus, over the deployed voted arm on disk:
  1. registered DER (eval_metrics, the house metric) + anyalt word error rate;
  2. alpha. error split by POSITION IN WORD - the word-final unit (the case
     ending) versus every inner unit;
  3. beta.  error split by GOLD mark family (bare / vowel / tanwin / sukun /
     shadda-only / shadda+vowel / shadda+tanwin);
  4. gamma. the confusion matrix over (gold class -> predicted class), top rows;
  5. delta. error split by sentence length (<= ctx 256 versus > 256).

README: CPU ONLY, no GPU, no training - it reads
runs/diac/e75_vote/<corpus>.vote.txt and .vote.ref.txt, which are the deployed
artifacts. Unit-level alignment uses the FIRST gold alternative; the anyalt
word-level rate is reported beside it so the alternative-set effect is visible.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402

try:
    import eval_der as _ed  # noqa: E402
    _split_alts = _ed.split_alts
except Exception:                                        # noqa: BLE001
    def _split_alts(w):
        return [w]


FALLBACK = ["bare", "fatha", "damma", "kasra", "sukun", "fathatan", "dammatan",
            "kasratan", "shadda", "shadda_fatha", "shadda_damma", "shadda_kasra",
            "shadda_fathatan", "shadda_dammatan", "shadda_kasratan"]
try:
    import labels as L  # noqa: E402
    NAMES = None
    for attr in ("LABELS", "CLASSES", "NAMES", "LABEL_NAMES"):
        v = getattr(L, attr, None)
        if isinstance(v, (list, tuple)) and len(v) == 15:
            NAMES = [str(x) for x in v]
            break
except Exception:                                        # noqa: BLE001
    NAMES = None


def cname(i):
    if i is None:
        return "UNCLASSIFIABLE"
    if NAMES:
        return NAMES[i]
    return FALLBACK[i] if 0 <= i < 15 else ("cls%d" % i)


FAM = {0: "bare", 1: "vowel", 2: "vowel", 3: "vowel", 4: "sukun", 5: "tanwin",
       6: "tanwin", 7: "tanwin", 8: "shadda_only", 9: "shadda_vowel",
       10: "shadda_vowel", 11: "shadda_vowel", 12: "shadda_tanwin",
       13: "shadda_tanwin", 14: "shadda_tanwin"}


def load(d, corpus, suffix):
    p = Path(d) / ("%s.%s.txt" % (corpus, suffix))
    r = Path(d) / ("%s.%s.ref.txt" % (corpus, suffix))
    pl = p.read_text(encoding="utf-8").splitlines()
    rl = r.read_text(encoding="utf-8").splitlines()
    n = min(len(pl), len(rl))
    return pl[:n], rl[:n]


def analyse(pred, ref, ctx):
    sm = em.summarize(em.score_corpus(pred, ref))
    conf = Counter()
    pos_err, pos_tot = Counter(), Counter()
    fam_err, fam_tot = Counter(), Counter()
    len_err, len_tot = Counter(), Counter()
    wbits_n = wbits_err = 0
    aligned_words = unaligned_words = 0
    units_n = units_err = 0
    conven_err = unclass_err = 0
    for pl, rl in zip(pred, ref):
        bits = em.word_outcomes(pl, rl)
        wbits_n += len(bits)
        wbits_err += sum(1 for b in bits if not b)
        pw = pl.split()
        gw = rl.split()
        alts = None
        for i, p in enumerate(pw):
            if i >= len(gw):
                break
            gold = _split_alts(gw[i])[0]
            pu = em.units(p)
            gu = em.units(gold)
            if [b for b, _ in pu] != [b for b, _ in gu]:
                unaligned_words += 1
                continue
            aligned_words += 1
            for ui, ((_, gm), (_, pm)) in enumerate(zip(gu, pu)):
                units_n += 1
                bucket = "final" if (ui == len(gu) - 1) else "inner"
                pos_tot[bucket] += 1
                gc = em.class_of(gm)
                pc = em.class_of(pm)
                if gc is not None and 0 <= gc < 15:
                    fam_tot[FAM[gc]] += 1
                if gm == pm:
                    continue                  # byte-identical under BOTH metrics
                if gc == pc and gc is not None:
                    # SAME CLASS, different raw mark bytes: the shadda/vowel byte
                    # order. The registered STRING metric counts this WRONG; the
                    # canonical metric counts it RIGHT. It is a representation
                    # convention cost, not a modelling error.
                    conven_err += 1
                    continue
                units_err += 1                # genuine class error
                pos_err[bucket] += 1
                if gc is not None and 0 <= gc < 15:
                    fam_err[FAM[gc]] += 1
                if gc is None or pc is None:
                    unclass_err += 1
                else:
                    conf[(cname(gc), cname(pc))] += 1
        g0 = " ".join(_split_alts(w)[0] for w in gw)   # the whole gold sentence
        b = "le256" if len(g0) <= ctx else "gt256"
        len_tot[b] += 1
        if not all(bits):
            len_err[b] += 1
    return {
        "der_registered": sm.get("der_registered"),
        "der_anyalt": sm.get("der_anyalt"),
        "der_canonical": sm.get("der_canonical"),
        "lines": len(pred),
        "words": wbits_n, "word_err_rate": round(wbits_err / max(wbits_n, 1), 4),
        "units": units_n,
        "class_err_rate": round(units_err / max(units_n, 1), 4),
        "convention_err_rate": round(conven_err / max(units_n, 1), 4),
        "unclassifiable_err": unclass_err,
        "unit_err_rate": round((units_err + conven_err) / max(units_n, 1), 4),
        "class_err_share": round(units_err / max(units_err + conven_err, 1), 4),
        "aligned_words": aligned_words, "unaligned_words": unaligned_words,
        "position": {k: {"units": pos_tot[k], "err": pos_err[k],
                         "share_of_units": round(pos_tot[k] / max(units_n, 1), 4),
                         "share_of_errors": round(pos_err[k] / max(units_err, 1), 4),
                         "err_rate": round(pos_err[k] / max(pos_tot[k], 1), 4)}
                     for k in ("final", "inner")},
        "family": {k: {"units": fam_tot[k], "err": fam_err[k],
                       "share_of_errors": round(fam_err[k] / max(units_err, 1), 4),
                       "err_rate": round(fam_err[k] / max(fam_tot[k], 1), 4)}
                   for k in sorted(fam_tot, key=lambda x: -fam_err[x])},
        "sent_len": {k: {"lines": len_tot[k], "word_err_lines": len_err[k],
                         "line_err_rate": round(len_err[k] / max(len_tot[k], 1), 4)}
                     for k in ("le256", "gt256")},
        "top_confusions": [{"gold": g, "pred": p, "n": n}
                           for (g, p), n in conf.most_common(12)],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/diac/e75_vote")
    ap.add_argument("--suffix", default="vote")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--out", default="runs/diac/_screen_out/error_decomposition.json")
    args = ap.parse_args()
    corpora = ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024",
               "abdou_test"]
    res = {"status": "EXACT-CPU-ANALYSIS", "arm": "E-075 voted soup (" + args.dir +
                                                  " ." + args.suffix + ")",
           "corpora": {}}
    for c in corpora:
        try:
            pred, ref = load(args.dir, c, args.suffix)
        except FileNotFoundError as exc:
            print(json.dumps({"SKIP": c, "why": str(exc)[-90:]}), flush=True)
            continue
        r = analyse(pred, ref, args.ctx)
        res["corpora"][c] = r
        p = r["position"]
        print(json.dumps({c: {
            "der": r["der_registered"], "der_canon": r["der_canonical"],
            "class_err_rate": r["class_err_rate"],
            "convention_err_rate": r["convention_err_rate"],
            "class_share_of_raw_err": r["class_err_share"],
            "word_err": r["word_err_rate"],
            "final": {"share_units": p["final"]["share_of_units"],
                      "share_errors": p["final"]["share_of_errors"],
                      "err_rate": p["final"]["err_rate"]},
            "inner": {"share_units": p["inner"]["share_of_units"],
                      "share_errors": p["inner"]["share_of_errors"],
                      "err_rate": p["inner"]["err_rate"]},
            "top_families_by_error": {k: v["share_of_errors"]
                                      for k, v in list(r["family"].items())[:5]},
            "top_confusions": r["top_confusions"][:6],
            "gt256": r["sent_len"]["gt256"],
        }}, ensure_ascii=False), flush=True)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote " + str(out), flush=True)


if __name__ == "__main__":
    main()
