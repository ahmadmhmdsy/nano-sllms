"""E-086 scorer: score a voting arm against the on-disk E-075 reference.

Reads <dir>/<corpus>.<suffix>.txt + .ref.txt (the shape e67_p6_vote.py writes)
and the same-shaped E-075 artifact, scores both through the registered metric,
and reports the paired delta plus a McNemar test on the identical line list.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402

CORPORA = ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024", "abdou_test"]
GATES = ["fadel_test", "sadeed25", "wikinews2014"]


def load(d, corpus, suffix):
    p = Path(d) / ("%s.%s.txt" % (corpus, suffix))
    r = Path(d) / ("%s.%s.ref.txt" % (corpus, suffix))
    if not p.exists() or not r.exists():
        return None
    return (p.read_text(encoding="utf-8").splitlines(),
            r.read_text(encoding="utf-8").splitlines())


def der(pred, ref):
    return em.summarize(em.score_corpus(pred, ref)).get("der_registered")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--suffix", default="vote")
    ap.add_argument("--baseline-dir", default="runs/diac/e75_vote")
    ap.add_argument("--baseline-suffix", default="vote")
    ap.add_argument("--arm", default="vote_long_window")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"status": "SCREEN-NOT-FULL-TRAIN" if "screen" in args.dir else "EXACT",
           "arm": args.arm, "dir": args.dir, "suffix": args.suffix,
           "baseline": "%s (%s)" % (args.baseline_dir, args.baseline_suffix),
           "corpora": {}}
    for c in CORPORA:
        a = load(args.dir, c, args.suffix)
        b = load(args.baseline_dir, c, args.baseline_suffix)
        if not a or not b:
            print(json.dumps({"SKIP": c, "arm_found": bool(a),
                              "base_found": bool(b)}), flush=True)
            continue
        pa, ra = a
        pb, rb = b
        if len(pa) != len(pb) or len(ra) != len(rb):
            print(json.dumps({"SKIP": c, "reason": "line-count mismatch",
                              "arm": [len(pa), len(ra)],
                              "base": [len(pb), len(rb)]}), flush=True)
            continue
        da, db = der(pa, ra), der(pb, rb)
        rec = {"der_arm": da, "der_baseline": db, "lines": len(pa),
               "rel": round((da - db) / db, 5)}
        try:
            rec["mcnemar"] = em.mcnemar(pa, pb, ra)
        except Exception as exc:
            rec["mcnemar"] = "unavailable: %s" % exc
        res["corpora"][c] = rec
        print(json.dumps({c: {"der_arm": da, "der_base": db,
                              "rel_pct": round(100.0 * rec["rel"], 3)}},
                         ensure_ascii=False), flush=True)
    g = [c for c in GATES if c in res["corpora"]]
    if len(g) == 3:
        ma = sum(res["corpora"][c]["der_arm"] for c in g) / 3.0
        mb = sum(res["corpora"][c]["der_baseline"] for c in g) / 3.0
        res["gate_mean3_arm"] = ma
        res["gate_mean3_baseline"] = mb
        res["gate_mean3_rel"] = round((ma - mb) / mb, 5)
        print(json.dumps({"GATE_MEAN3": {"arm": ma, "baseline": mb,
                                         "rel_pct": round(100.0 * (ma - mb) / mb, 3)}},
                         ensure_ascii=False), flush=True)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote " + str(out), flush=True)


if __name__ == "__main__":
    main()
