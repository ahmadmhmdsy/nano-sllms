"""E-084 companion probe A-prime: positional/structural validity census.

CPU only, no GPU, reads existing prediction artifacts.

WHY THIS EXISTS. E-18 tested dedup + legality_repair and got BYTE-IDENTICAL
output, and the reason is structural, not empirical: the 15-class head
(labels.label_for_marks) only ever emits () / one vowel / one tanwin / sukun /
shadda+{vowel,tanwin}. Two vowels on one letter is UNREPRESENTABLE. So the
literature's constrained-decoding win (Amr et al., arXiv:2607.18946) cannot
transfer to WITHIN-POSITION legality - there is nothing left to fix there.

That leaves the POSITIONAL rules, which ARE fully representable by the head and
were never tested:
  R1 tanween only on the final letter of a word,
  R2 sukun not on the first letter of a word,
  R3 the definite article (alef+lam) carries no diacritic,
  R4 shadda carried alone, with no vowel or tanween.

The decision this probe makes is whether the constrained-decoding axis has ANY
headroom BEFORE a GPU second is spent. The binding constraint is not what OUR
model does but what the GOLD does: if the gold text itself breaks a rule,
constraining our decoder to obey it can only add errors. So every rule is
counted on BOTH sides and the headroom is an UPPER BOUND.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))

import eval_metrics as em      # noqa: E402
import labels as L             # noqa: E402
import eval_der                # noqa: E402

TANWIN = {L.C_FATHATAN, L.C_DAMMATAN, L.C_KASRATAN,
          L.C_SHADDA_FATHATAN, L.C_SHADDA_DAMMATAN, L.C_SHADDA_KASRATAN}
RULES = ("R0_unclassifiable", "R1_tanween_nonfinal", "R2_sukun_first",
         "R3_defart_marked", "R4_shadda_alone")
ALEF, LAM = "\u0627", "\u0644"


def word_violations(units):
    """(violations, classes) for one word's [(base, mark_run), ...]."""
    viol = set()
    classes = []
    for i, (base, marks) in enumerate(units):
        cls = em.class_of(marks)
        classes.append(cls)
        if cls is None:
            viol.add("R0_unclassifiable")
            continue
        if cls in TANWIN and i < len(units) - 1:
            viol.add("R1_tanween_nonfinal")
        if i == 0 and cls == L.C_SUKUN:
            viol.add("R2_sukun_first")
        if cls == L.C_SHADDA:
            viol.add("R4_shadda_alone")
    if len(units) >= 2 and units[0][0] == ALEF and units[1][0] == LAM:
        if any(em.class_of(m) != L.C_BARE for _, m in units[:2]):
            viol.add("R3_defart_marked")
    return viol, classes


def census(pred_path, ref_path):
    preds = Path(pred_path).read_text(encoding="utf-8").splitlines()
    refs = Path(ref_path).read_text(encoding="utf-8").splitlines()
    stat = {"lines": min(len(preds), len(refs)), "lines_skipped_len": 0,
            "units": 0, "units_wrong": 0, "words": 0, "words_wrong": 0,
            "fixable_upper": 0, "pred": {r: 0 for r in RULES},
            "gold": {r: 0 for r in RULES}}
    for pl, rl in zip(preds, refs):
        pw = pl.split()
        alt_texts = rl.split("\t")
        rw = alt_texts[0].split()
        if len(pw) != len(rw):
            stat["lines_skipped_len"] += 1
            continue
        for p_word, g_field in zip(pw, rw):
            g_alts = g_field.split("/")
            pu = em.units(p_word)
            stat["units"] += len(pu)
            stat["words"] += 1
            pv, pcls = word_violations(pu)
            for r in pv:
                stat["pred"][r] += 1
            g_units = [em.units(g) for g in g_alts]
            if not any([b for b, _ in gu] == [b for b, _ in pu] for gu in g_units):
                continue                       # base drift: not comparable
            gv_union = None
            gcls_sets = []
            for gu in g_units:
                gv, gcls = word_violations(gu)
                gv_union = set(gv) if gv_union is None else (gv_union & set(gv))
                gcls_sets.append(gcls)
            for r in gv_union:
                stat["gold"][r] += 1
            if not any(eval_der.marks_of(p_word) == eval_der.marks_of(g)
                       for g in g_alts):
                stat["words_wrong"] += 1
            bad = 0
            for i, pc in enumerate(pcls):
                if not any(i < len(gc) and gc[i] == pc for gc in gcls_sets):
                    bad += 1
            stat["units_wrong"] += bad
            if pv and not gv_union and bad:
                stat["fixable_upper"] += bad
    u = max(stat["units"], 1)
    stat["pct_units_wrong"] = round(100.0 * stat["units_wrong"] / u, 3)
    stat["pct_fixable_upper"] = round(100.0 * stat["fixable_upper"] / u, 3)
    stat["pred_per_10k_units"] = {r: round(10000.0 * stat["pred"][r] / u, 2)
                                 for r in RULES}
    stat["gold_per_10k_units"] = {r: round(10000.0 * stat["gold"][r] / u, 2)
                                 for r in RULES}
    return stat


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.scheme_dir)
    res = {"status": "SCREEN-NOT-FULL-TRAIN-companion",
           "note": "CPU census of EXISTING artifacts; no training, no screen verdict.",
           "scheme_dir": str(d), "corpora": {}}
    for pf in sorted(d.glob("*.single.txt")):
        corpus = pf.name[:-len(".single.txt")]
        rf = d / (corpus + ".single.ref.txt")
        if not rf.exists():
            continue
        res["corpora"][corpus] = census(pf, rf)
        c = res["corpora"][corpus]
        print(json.dumps({corpus: {k: c[k] for k in
              ("units", "pct_units_wrong", "pct_fixable_upper",
               "pred_per_10k_units", "gold_per_10k_units")}},
              ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
