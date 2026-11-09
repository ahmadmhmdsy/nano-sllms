"""E-095 gate aggregation: apply the PRE-REGISTERED verdict rule.

Rule (fixed in the E-095 record BEFORE the run):
  nospace is ACCEPTED iff
    (a) gate_mean3(nospace) <= 0.98 * gate_mean3(single)   [2 pct relative], AND
    (b) no individual registered gate corpus is worse than its single under the
        0.82 pct C0-R init-draw noise band.
  PARTIAL if (a) holds but (b) fails on exactly one corpus; REJECTED if (a) fails.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIR = REPO / "runs" / "diac" / "e95_gate"
GATE = ["fadel_test", "sadeed25", "wikinews2014"]
OTHER = ["wikinews2024", "abdou_test"]
BAR = 0.98
NOISE = 0.0082


def load(c):
    f = DIR / c / "score_registered.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    got = {c: load(c) for c in GATE + OTHER}
    print("corpora present:", [c for c, d in got.items() if d])
    print()
    print("%-14s %12s %12s %10s" % ("corpus", "single", "nospace", "rel"))
    means = {"single": [], "nospace": []}
    for c in GATE + OTHER:
        d = got[c]
        if not d:
            print("%-14s %12s" % (c, "(missing)"))
            continue
        a = d["arms"]
        s = a.get("single", {}).get("der_registered")
        n = a.get("nospace", {}).get("der_registered")
        rel = (100.0 * (n - s) / s) if (s and n) else float("nan")
        tag = "" if c in GATE else "  (unregistered)"
        print("%-14s %12.6f %12.6f %9.2f%%%s" % (c, s, n, rel, tag))
        if c in GATE and s and n:
            means["single"].append(s)
            means["nospace"].append(n)

    if len(means["single"]) < len(GATE):
        print()
        print("GATE INCOMPLETE: %d/%d registered corpora scored - no verdict yet."
              % (len(means["single"]), len(GATE)))
        return
    gs = sum(means["single"]) / len(means["single"])
    gn = sum(means["nospace"]) / len(means["nospace"])
    print()
    print("gate_mean3 single  = %.6f" % gs)
    print("gate_mean3 nospace = %.6f" % gn)
    rel = 100.0 * (gn - gs) / gs
    print("relative change    = %+.2f pct   (bar: <= -2.00 pct)" % rel)

    # per-corpus noise-band check
    worse = []
    for c in GATE:
        a = got[c]["arms"]
        s = a["single"]["der_registered"]
        n = a["nospace"]["der_registered"]
        if n > s * (1.0 + NOISE):
            worse.append((c, 100.0 * (n - s) / s))
    print("corpora worse than single beyond the 0.82 pct band:", worse or "none")

    cond_a = gn <= BAR * gs
    if cond_a and not worse:
        v = "ACCEPTED"
    elif cond_a and len(worse) == 1:
        v = "PARTIAL"
    elif cond_a:
        v = "REJECTED (multiple gate corpora worse)"
    else:
        v = "REJECTED"
    print()
    print("PRE-REGISTERED VERDICT:", v)
    out = {"gate_mean3": {"single": gs, "nospace": gn}, "relative_pct": rel,
           "worse_beyond_noise": worse, "verdict": v}
    (DIR / "e95_verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


main()
