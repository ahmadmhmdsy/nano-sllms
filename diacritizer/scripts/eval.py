"""Eval harness CLI (R52). Mode: compare --pred FILE --ref FILE [--ref FILE]

Line-per-line positional alignment across prediction and every reference
(multi-ref; WikiNews/SadeedDiac benchmark files are 1 line = 1 sentence).
Empty/reference-missing lines count as length-mismatch errors, never skipped
silently. Report: stdout JSON (the 4 metrics, never rounded silently).
CPU-only.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_der  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compare"])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", action="append", required=True)
    args = ap.parse_args()
    pred_lines = Path(args.pred).read_text(encoding="utf-8").splitlines()
    ref_sets = [Path(r).read_text(encoding="utf-8").splitlines() for r in args.ref]
    n = max([len(pred_lines)] + [len(r) for r in ref_sets])
    metrics = []
    for i in range(n):
        p = pred_lines[i] if i < len(pred_lines) else ""
        # multi-ref support: one line may carry alternatives tab-separated
        refs = [(rl[i].split("\t") if i < len(rl) else [""]) for rl in ref_sets]
        refs = [r for alt in refs for r in alt]
        metrics.append(eval_der.score_line(p, refs))
    agg = eval_der.aggregate(metrics)
    report = {"pred": args.pred, "refs": args.ref,
              "lines": len(metrics), "aggregate": agg}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
