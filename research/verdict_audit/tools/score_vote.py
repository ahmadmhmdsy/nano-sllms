#!/usr/bin/env python
"""Paired scorer: compare two diacritizer prediction files on the same refs.

Independent re-instrumentation (rules/evidence.md): reuses eval_metrics, the
same scorer the training loop and the gate bench use, so a DER claim is an
artifact-backed number and never a console-only one.

Usage:
  python research/verdict_audit/tools/score_vote.py --ref REF --a A --b B \
      [--label-a single] [--label-b vote] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as EM  # noqa: E402


def load(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def report(preds: list[str], refs: list[str]) -> dict:
    sc = EM.score_corpus(preds, refs)
    s = EM.summarize(sc)
    reg = EM.bootstrap_ci(sc["lines"], "registered_correct", "words")
    out = {
        "lines": len(preds),
        "der_registered": s["der_registered"],
        "der_canonical": s["der_canonical"],
        "der_anyalt": s["der_anyalt"],
        "macro_f1": s["macro_f1"],
        # NB: registered_correct/words is word ACCURACY (= 1 - DER), so this
        # interval bounds accuracy, not DER. Labelled accordingly.
        "ci_word_accuracy": [reg["lo"], reg["hi"]],
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="single")
    ap.add_argument("--label-b", default="vote")
    ap.add_argument("--mc", action="store_true", help="paired McNemar a vs b")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    refs = load(Path(a.ref))
    pa, pb = load(Path(a.a)), load(Path(a.b))
    if not (len(refs) == len(pa) == len(pb)):
        print(json.dumps({"error": "length mismatch",
                          "ref": len(refs), "a": len(pa), "b": len(pb)},
                         indent=2))
        return 2
    ra, rb = report(pa, refs), report(pb, refs)
    out = {"ref": a.ref, "label_a": a.label_a, "label_b": a.label_b,
           a.label_a: ra, a.label_b: rb,
           "delta_der_registered": rb["der_registered"] - ra["der_registered"],
           "rel_change_pct": 100.0 * (rb["der_registered"] - ra["der_registered"])
           / ra["der_registered"] if ra["der_registered"] else None}
    if a.mc:
        for canon in (False, True):
            m = EM.mcnemar(pa, pb, refs, canonical=canon)
            out["mcnemar_canonical" if canon else "mcnemar_nocase"] = m
    txt = json.dumps(out, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
