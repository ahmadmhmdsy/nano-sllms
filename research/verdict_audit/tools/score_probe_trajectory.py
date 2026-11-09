"""Score every probe step's existing gate predictions -> trajectory table.

Pure CPU, read-only on the run dirs. Source artifacts are the per-step
prediction files written by the gate probe; no metric is taken from prose.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path("diacritizer/src")))
import eval_metrics as em

GATES = ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024"]
EXT = ["fadel_test", "sadeed25", "wikinews2014"]
RUNS = ["e70_c4b_ctx256", "e69_c4_ctx256"]


def lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


out = {}
for run in RUNS:
    gp = Path("runs/diac") / run / "gate_probe"
    steps = sorted({int(m.group(1)) for f in gp.glob("step*_*_pred.txt")
                    if (m := re.match(r"step(\d+)_", f.name))})
    rows = {}
    for s in steps:
        row = {}
        for g in GATES:
            pf, rf = gp / f"step{s}_{g}_pred.txt", gp / f"step{s}_{g}_pred.ref.txt"
            if not (pf.exists() and rf.exists()):
                continue
            sm = em.summarize(em.score_corpus(lines(pf), lines(rf)))
            row[g] = {k: sm[k] for k in ("der_registered", "der_canonical", "macro_f1") if k in sm}
        ext = [row[g]["der_registered"] for g in EXT if g in row]
        if ext:
            row["EXT_MEAN_der_registered"] = sum(ext) / len(ext)
        rows[str(s)] = row
    out[run] = {"steps": steps, "rows": rows}

print("API:", [n for n in dir(em) if not n.startswith("_")][:24])
for run in RUNS:
    print("\n=== " + run + " ===")
    print("step   EXT_MEAN   fadel    sadeed   wn2014   wn2024   canon(EXTmean)")
    for s in out[run]["steps"]:
        r = out[run]["rows"][str(s)]
        if "EXT_MEAN_der_registered" not in r:
            continue
        can = [r[g]["der_canonical"] for g in EXT if g in r and "der_canonical" in r[g]]
        print("%-6d %.6f   %.6f %.6f %.6f %.6f   %.6f" % (
            s, r["EXT_MEAN_der_registered"],
            r.get("fadel_test", {}).get("der_registered", float("nan")),
            r.get("sadeed25", {}).get("der_registered", float("nan")),
            r.get("wikinews2014", {}).get("der_registered", float("nan")),
            r.get("wikinews2024", {}).get("der_registered", float("nan")),
            (sum(can) / len(can)) if can else float("nan")))
Path("research/verdict_audit/probe_trajectory.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("\nwrote research/verdict_audit/probe_trajectory.json")
