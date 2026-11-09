"""Score a run's clean-input gate_bench predictions with the registered metric set.

CPU only, no bootstrap: uses diacritizer/src/eval_metrics.py directly so the
numbers match eval_report.py's der_* family. Part of the 2026-10-06 verdict
audit (E-063 clean-instrument cross-check).

Usage:
    .venv\\Scripts\\python.exe research\\verdict_audit\\tools\\score_gate_bench.py \
        runs/diac/stage3best_ls runs/diac/e63_m1_tr --gates fadel_test,sadeed25,wikinews2014
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as EM  # noqa: E402

GATE_FILES = {
    "fadel_test": "fadel_test",
    "sadeed25": "sadeed25",
    "wikinews2014": "wikinews2014",
    "wikinews2024": "wikinews2024",
    "abdou_test": "abdou_test",
}


def score_one(gate_dir: Path, gate: str, tag: str):
    stem = GATE_FILES[gate]
    ref = gate_dir / f"{stem}.prd.ref.txt"
    pred = gate_dir / f"{stem}.prd.txt"
    if not ref.exists() or not pred.exists():
        return None
    ref_lines = ref.read_text(encoding="utf-8").splitlines()
    pred_lines = pred.read_text(encoding="utf-8").splitlines()
    s = EM.summarize(EM.score_corpus(pred_lines, ref_lines))
    keep = {k: s[k] for k in s if k.startswith(("der_", "cder", "ser", "mark_f1", "preserve"))}
    keep["gate"] = gate
    keep["tag"] = tag
    keep["lines"] = len(ref_lines)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs (a gate_bench/ subdir is appended)")
    ap.add_argument("--gates", default="fadel_test,sadeed25,wikinews2014")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    rows = []
    print(f"%-22s %-14s %8s %8s %8s %8s" % ("run", "gate", "der_reg", "der_any", "der_canon", "mark_f1"))
    for run in args.runs:
        d = Path(run)
        tag = d.name
        gb = d / "gate_bench"
        if not gb.exists():
            print(f"{tag:<22} NO gate_bench")
            continue
        for g in gates:
            r = score_one(gb, g, tag)
            if r is None:
                print(f"{tag:<22} {g:<14} MISSING")
                continue
            rows.append(r)
            print("%-22s %-14s %8.4f %8.4f %8.4f %8.4f" % (
                tag, g, r.get("der_registered", float("nan")),
                r.get("der_anyalt", float("nan")), r.get("der_canonical", float("nan")),
                r.get("mark_f1", float("nan"))))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
