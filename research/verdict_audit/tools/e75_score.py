"""E-075: does sliding-window voting compose with the E-073 weight soup?"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("diacritizer/src")))
import eval_metrics as em

CORPORA = ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024", "abdou_test"]
EXT = ["fadel_test", "sadeed25", "wikinews2014"]
BASE = Path("runs/diac/e75_vote")
REF = {"soup_single_gate": 0.352209, "soup_single_abdou": 0.349044,
       "e071_vote_gate": 0.369702, "best_single_probe_gate": 0.374792}


def lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


out = {}
for kind in ("single", "vote"):
    row = {}
    for c in CORPORA:
        pf = BASE / (c + "." + kind + ".txt")
        rf = pf.with_suffix(".ref.txt")
        if not (pf.exists() and rf.exists()):
            row[c] = {"missing": True}; continue
        sm = em.summarize(em.score_corpus(lines(pf), lines(rf)))
        row[c] = {k: sm.get(k) for k in
                  ("der_registered", "der_canonical", "macro_f1", "der_anyalt")}
    got = [row[c]["der_registered"] for c in EXT if "der_registered" in row.get(c, {})]
    row["gate_mean3"] = sum(got) / len(got) if got else None
    out[kind] = row

BASE.mkdir(parents=True, exist_ok=True)
(BASE / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print("REFERENCES (from artifacts):")
for k, v in REF.items():
    print("  %-24s %.6f" % (k, v))
print()
print("%-6s %-11s %-11s %-11s %-11s %-11s %-11s" % ("kind", "fadel", "sadeed", "wn2014", "wn2024", "abdou", "gate_mean3"))
for kind in ("single", "vote"):
    r = out[kind]
    def g(c):
        v = r.get(c, {})
        return "%.6f" % v["der_registered"] if "der_registered" in v else "MISSING"
    gm = r["gate_mean3"]
    print("%-6s %-11s %-11s %-11s %-11s %-11s %-11s" % (
        kind, g("fadel_test"), g("sadeed25"), g("wikinews2014"),
        g("wikinews2024"), g("abdou_test"), "%.6f" % gm if gm else "NA"))
print()
s, v = out["single"], out["vote"]
if s["gate_mean3"] and v["gate_mean3"] and "der_registered" in s.get("abdou_test", {}) and "der_registered" in v.get("abdou_test", {}):
    sg, vg = s["gate_mean3"], v["gate_mean3"]
    sa, va = s["abdou_test"]["der_registered"], v["abdou_test"]["der_registered"]
    print("gate:  single %.6f -> vote %.6f  (%+.2f%%)" % (sg, vg, 100*(vg-sg)/sg))
    print("abdou: single %.6f -> vote %.6f  (%+.2f%%)" % (sa, va, 100*(va-sa)/sa))
    print("vote vs E-071 single-model voting (0.369702): %+.2f%%" % (100*(vg-REF["e071_vote_gate"])/REF["e071_vote_gate"]))
    print("vote vs best single probe (0.374792):        %+.2f%%" % (100*(vg-REF["best_single_probe_gate"])/REF["best_single_probe_gate"]))
    print()
    if vg < sg and va < sa and vg < REF["e071_vote_gate"]:
        print("E-075 VERDICT: WIN")
    elif vg < sg and va < sa:
        print("E-075 VERDICT: COMPOSES")
    elif vg >= sg and va >= sa:
        print("E-075 VERDICT: FALSIFIED")
    else:
        print("E-075 VERDICT: MIXED")
    bad = [c for c in CORPORA if out["vote"].get(c, {}).get("der_anyalt")
           and abs(out["vote"][c]["der_anyalt"] - 0.7554) < 0.01]
    if bad:
        print("  WARNING degenerate-mode signature on: " + ", ".join(bad) + " -> INSTRUMENT-UNKNOWN")
else:
    print("incomplete - vote arm not finished")
