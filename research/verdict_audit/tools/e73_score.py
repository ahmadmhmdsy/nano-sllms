"""Score the E-073 soup arms. CPU-only, reads the pred/.ref.txt pairs bench.py wrote."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("diacritizer/src")))
import eval_metrics as em

ARMS = ["c4b_uniform", "c4_uniform", "all24_uniform"]
CORPORA = ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024", "abdou_test"]
EXT = ["fadel_test", "sadeed25", "wikinews2014"]
BASE = Path("runs/diac/e73_soup")
REF = {"c4b_gate_mean": 0.388148, "c4b_gate_min": 0.374792,
       "c4_gate_mean": 0.398088, "c4_gate_min": 0.387232,
       "abdou_heldout_best": 0.375141}


def lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


out = {}
for a in ARMS:
    row = {}
    for c in CORPORA:
        pf = BASE / a / "pred" / (c + ".txt")
        rf = pf.with_suffix(".ref.txt")
        if not (pf.exists() and rf.exists()):
            row[c] = {"missing": True}
            continue
        sm = em.summarize(em.score_corpus(lines(pf), lines(rf)))
        row[c] = {k: sm.get(k) for k in
                  ("der_registered", "der_canonical", "macro_f1", "der_anyalt")}
    got = [row[c]["der_registered"] for c in EXT if c in row and "der_registered" in row[c]]
    row["gate_mean3"] = sum(got) / len(got) if got else None
    out[a] = row

BASE.mkdir(parents=True, exist_ok=True)
(BASE / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print("REFERENCE (recomputed from probe prediction files):")
for k, v in REF.items():
    print("  %-20s %.6f" % (k, v))
print()
print("%-14s %-11s %-11s %-11s %-11s %-11s" % ("arm", "fadel", "sadeed", "wn2014", "wn2024", "abdou"))
for a in ARMS:
    r = out[a]
    print("%-14s %-11s %-11s %-11s %-11s %-11s" % (
        a,
        "%.6f" % r["fadel_test"]["der_registered"] if "der_registered" in r.get("fadel_test", {}) else "MISSING",
        "%.6f" % r["sadeed25"]["der_registered"] if "der_registered" in r.get("sadeed25", {}) else "MISSING",
        "%.6f" % r["wikinews2014"]["der_registered"] if "der_registered" in r.get("wikinews2014", {}) else "MISSING",
        "%.6f" % r["wikinews2024"]["der_registered"] if "der_registered" in r.get("wikinews2024", {}) else "MISSING",
        "%.6f" % r["abdou_test"]["der_registered"] if "der_registered" in r.get("abdou_test", {}) else "MISSING"))
print()
for a in ARMS:
    gm = out[a]["gate_mean3"]
    ab = out[a].get("abdou_test", {}).get("der_registered")
    if gm is None or ab is None:
        print("%-14s incomplete" % a); continue
    print("%-14s gate_mean3=%.6f  abdou=%.6f  vs c4b_min=%+.2f%%  vs c4b_mean=%+.2f%%  vs abdou_best=%+.2f%%" % (
        a, gm, ab, 100 * (gm - REF["c4b_gate_min"]) / REF["c4b_gate_min"],
        100 * (gm - REF["c4b_gate_mean"]) / REF["c4b_gate_mean"],
        100 * (ab - REF["abdou_heldout_best"]) / REF["abdou_heldout_best"]))
print()
r = out["c4b_uniform"]
gm, ab = r["gate_mean3"], r["abdou_test"]["der_registered"]
if gm < REF["c4b_gate_min"] and ab < REF["abdou_heldout_best"]:
    print("E-073 VERDICT: WIN")
elif gm < REF["c4b_gate_mean"] and ab < REF["abdou_heldout_best"]:
    print("E-073 VERDICT: PARTIAL")
elif gm >= REF["c4b_gate_mean"] and ab >= REF["abdou_heldout_best"]:
    print("E-073 VERDICT: FALSIFIED")
else:
    print("E-073 VERDICT: MIXED")
for c in CORPORA:
    v = r.get(c, {})
    if v.get("der_canonical") and abs(v["der_canonical"] - 0.75) < 1e-6:
        print("  WARNING non-degeneracy: der_canonical ~0.75 on", c)
