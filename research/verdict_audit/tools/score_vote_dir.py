r"""Score a directory of single/vote prediction pairs and print a corpus table.

Reusable form of e75_score.py: point --base at any directory holding
<corpus>.single.txt and <corpus>.vote.txt (each with a .ref.txt beside it),
and optionally pass --ref name=value to print deltas against known numbers.

Usage:
  & .\.venv\Scripts\python.exe research\verdict_audit\tools\score_vote_dir.py \
      --base runs/diac/e76_greedy/pred \
      --ref uniform_voted_gate=0.346827 --ref uniform_voted_abdou=0.341475
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402

CORPORA = ["fadel_test", "sadeed25", "wikinews2014", "wikinews2024", "abdou_test"]
EXT = ["fadel_test", "sadeed25", "wikinews2014"]


def lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", action="append", default=[],
                    help="name=value reference number, repeatable")
    ap.add_argument("--title", default="")
    ap.add_argument("--kinds", default="single,vote",
                    help="comma-separated arms to score: single, vote, or both")
    args = ap.parse_args()
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in ("single", "vote"):
            ap.error("--kinds entries must be single or vote, got %r" % k)

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    base = REPO / args.base
    refs = {}
    for item in args.ref:
        k, _, v = item.partition("=")
        refs[k] = float(v)

    out = {}
    for kind in kinds:
        row = {}
        for c in CORPORA:
            pf = base / (c + "." + kind + ".txt")
            rf = pf.with_suffix(".ref.txt")
            if not (pf.exists() and rf.exists()):
                row[c] = {"missing": True}
                continue
            sm = em.summarize(em.score_corpus(lines(pf), lines(rf)))
            row[c] = {k: sm.get(k) for k in
                      ("der_registered", "der_canonical", "macro_f1", "der_anyalt")}
        got = [row[c]["der_registered"] for c in EXT
               if "der_registered" in row.get(c, {})]
        row["gate_mean3"] = sum(got) / len(got) if got else None
        out[kind] = row

    outp = Path(args.out) if args.out else (base / "verdict.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.title:
        print(args.title)
    if refs:
        print("REFERENCES:")
        for k, v in refs.items():
            print("  %-26s %.6f" % (k, v))
    print()
    print("%-7s %-11s %-11s %-11s %-11s %-11s %-11s"
          % ("kind", "fadel", "sadeed", "wn2014", "wn2024", "abdou", "gate_mean3"))
    for kind in kinds:
        r = out[kind]

        def g(c):
            v = r.get(c, {})
            return "%.6f" % v["der_registered"] if "der_registered" in v else "MISSING"

        gm = r["gate_mean3"]
        print("%-7s %-11s %-11s %-11s %-11s %-11s %-11s" % (
            kind, g("fadel_test"), g("sadeed25"), g("wikinews2014"),
            g("wikinews2024"), g("abdou_test"), "%.6f" % gm if gm else "NA"))
    print()
    if not ("single" in out and "vote" in out):
        return
    s, v = out["single"], out["vote"]
    ok = (s["gate_mean3"] and v["gate_mean3"]
          and "der_registered" in s.get("abdou_test", {})
          and "der_registered" in v.get("abdou_test", {}))
    if ok:
        sg, vg = s["gate_mean3"], v["gate_mean3"]
        sa, va = s["abdou_test"]["der_registered"], v["abdou_test"]["der_registered"]
        print("gate:  single %.6f -> vote %.6f  (%+.2f%%)" % (sg, vg, 100 * (vg - sg) / sg))
        print("abdou: single %.6f -> vote %.6f  (%+.2f%%)" % (sa, va, 100 * (va - sa) / sa))
        for k, rv in refs.items():
            tgt = vg if k.endswith("gate") else va
            print("vote %-24s %.6f -> %+.2f%%" % (k, rv, 100 * (tgt - rv) / rv))
        print()
        print("der_canonical / macro_f1 (vote):")
        for c in CORPORA:
            r = v.get(c, {})
            if "der_canonical" in r:
                print("  %-14s der_canon %.6f   macro_f1 %.4f   der_anyalt %.6f"
                      % (c, r["der_canonical"], r["macro_f1"], r["der_anyalt"]))
        bad = [c for c in CORPORA
               if v.get(c, {}).get("der_anyalt")
               and abs(v[c]["der_anyalt"] - 0.7554) < 0.01]
        if bad:
            print("  WARNING degenerate-mode signature on: " + ", ".join(bad)
                  + " -> INSTRUMENT-UNKNOWN")
    else:
        print("incomplete - one arm missing")


if __name__ == "__main__":
    main()
