#!/usr/bin/env python
"""Score a train-time gate_probe step against its written .ref.txt siblings.

Independent re-instrumentation for the D-line verdict audit: the registered
gate_eval.csv numbers come from the in-training probe path, and this tool
re-scores the exact written prediction/ref pairs with eval_metrics.score_corpus
so a recorded DER can be reproduced without re-running any model.

Usage:
  python research/verdict_audit/tools/score_probe.py <run_dir> <step> [--out FILE]
  python research/verdict_audit/tools/score_probe.py --list <run_dir>

Read-only: it only reads runs/diac/<run>/gate_probe/step<step>_<gate>_pred{,.ref}.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as EM  # noqa: E402

EXT = ["fadel_test", "sadeed25", "wikinews2014"]
ALL = EXT + ["wikinews2024"]
GATES = ALL + ["abdou_test"]


def probe_steps(run_dir: Path) -> list[int]:
    steps = set()
    for p in (run_dir / "gate_probe").glob("step*_*_pred.txt"):
        if p.name.endswith(".ref.txt"):
            continue
        try:
            steps.add(int(p.name.split("_")[0][4:]))
        except ValueError:
            continue
    return sorted(steps)


def score(run_dir: Path, step: int) -> dict:
    per: dict[str, float] = {}
    for gate in GATES:
        pred = run_dir / "gate_probe" / f"step{step}_{gate}_pred.txt"
        ref = run_dir / "gate_probe" / f"step{step}_{gate}_pred.ref.txt"
        if not pred.exists():
            continue
        refs = ref.read_text(encoding="utf-8").splitlines()
        preds = pred.read_text(encoding="utf-8").splitlines()
        m = EM.summarize(EM.score_corpus(preds, refs))
        per[gate] = m["der_registered"]
    out = {"run": run_dir.name, "step": step, "per_gate": per}
    if all(g in per for g in EXT):
        out["mean3"] = sum(per[g] for g in EXT) / 3
    if all(g in per for g in ALL):
        out["mean4"] = sum(per[g] for g in ALL) / 4
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="run name under runs/diac or a path")
    ap.add_argument("step", nargs="?", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not a.run_dir:
        ap.error("run_dir is required")
    rd = Path(a.run_dir)
    if not rd.is_absolute():
        cand = REPO / "runs" / "diac" / a.run_dir
        rd = cand if cand.exists() else (REPO / a.run_dir)
    if a.list or a.step is None:
        print(json.dumps({"run": rd.name, "steps": probe_steps(rd)}))
        return 0
    res = score(rd, a.step)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if a.out:
        (REPO / a.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("%s @%d: %s" % (
        res["run"], res["step"],
        " ".join("%s=%.4f" % (g, v) for g, v in res["per_gate"].items()),
    ))
    if "mean3" in res:
        print("   mean3=%.4f" % res["mean3"], end="")
    if "mean4" in res:
        print("  mean4=%.4f" % res["mean4"], end="")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
