#!/usr/bin/env python
"""E-092 vote-density screen scorer: PAIRED analysis across strides.

*** FEW-MINUTE SCREEN, NOT A FULL GATE RUN. ***

Reads runs/diac/e92_stride/<stride>/abdou_test.vote.{txt,ref.txt} for strides
128 / 64 / 32, scores each with the registered der_registered, and compares each
denser stride against S128 with an EXACT PAIRED McNemar over per-word outcomes on
the SAME lines.

Pairing is the point: a 400-line selector cannot resolve a 2 pct effect from two
unpaired aggregates, but the same sentences under both systems cancel shared
sentence difficulty, and the test uses only the discordant words.

Applies the pre-registered rule recorded in E-092. CPU only.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402

BASE = REPO / "runs" / "diac" / "e92_stride"
STRIDES = [128, 64, 32]
CONTROL = 128
SIG_PCT = 2.0
P_BAR = 0.05


def lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    data = {}
    for s in STRIDES:
        pf = BASE / str(s) / "abdou_test.vote.txt"
        rf = BASE / str(s) / "abdou_test.vote.ref.txt"
        if not (pf.exists() and rf.exists()):
            print("MISSING", pf)
            return 2
        pred, ref = lines(pf), lines(rf)
        sm = em.summarize(em.score_corpus(pred, ref))
        data[s] = {
            "pred": pred, "ref": ref,
            "der_registered": sm["der_registered"],
            "der_canonical": sm["der_canonical"],
            "macro_f1": sm["macro_f1"],
            "der_anyalt": sm["der_anyalt"],
            "lines": len(pred),
        }

    if len({d["lines"] for d in data.values()}) != 1:
        print("LINE COUNT MISMATCH - not a paired comparison")
        return 2
    ref = data[CONTROL]["ref"]
    for s in STRIDES:
        if data[s]["ref"] != ref:
            print("REF MISMATCH at stride %d - arms did not see the same gold" % s)
            return 2

    base_der = data[CONTROL]["der_registered"]
    # NOTE: these are line SEGMENTS, not source sentences - abdou_test gold fields
    # contain embedded newlines and this tool reads with splitlines() (same caveat
    # E-088 recorded). Word-level DER is unaffected by line grouping.
    print("*** FEW-MINUTE SCREEN (abdou_test first 400 pairs -> %d scored line-segments),"
          " NOT A FULL GATE RUN ***" % data[CONTROL]["lines"])
    print("window FIXED at 256; only stride (vote density) varies\n")
    print("%-8s %-7s %-12s %-10s %-10s" % ("stride", "votes", "der_reg", "rel%", "macro_f1"))
    for s in STRIDES:
        d = data[s]
        rel = (d["der_registered"] - base_der) / base_der * 100.0
        print("%-8d %-7d %-12.6f %+9.2f%% %-10.4f"
              % (s, 256 // s, d["der_registered"], rel, d["macro_f1"]))

    print("\nPAIRED exact McNemar, each denser stride vs S128 (same %d lines):"
          % data[CONTROL]["lines"])
    results = {}
    for s in STRIDES:
        if s == CONTROL:
            continue
        d = data[s]
        rel = (d["der_registered"] - base_der) / base_der * 100.0
        mc = em.mcnemar(data[CONTROL]["pred"], d["pred"], ref)
        if (rel <= -SIG_PCT and mc["p_value_two_sided"] < P_BAR
                and mc["c_b_correct_only"] > mc["b_a_correct_only"]):
            verdict = "WIN"
        elif rel >= SIG_PCT:
            verdict = "FALSIFIED"
        else:
            verdict = "INCONCLUSIVE"
        results[s] = {"der_registered": d["der_registered"], "rel_pct": rel,
                      "mcnemar": mc, "verdict": verdict}
        print("  S%d vs S128: rel %+.2f%%  discordant=%d (S128-only %d, S%d-only %d)  "
              "p=%.3g  -> %s"
              % (s, rel, mc["discordant"], mc["b_a_correct_only"], s,
                 mc["c_b_correct_only"], mc["p_value_two_sided"], verdict))

    nondeg = all(abs(data[s]["der_anyalt"] - 0.7554) > 0.05 for s in STRIDES)
    print("\nnon-degeneracy (der_anyalt far from the 0.7554 collapse signature): %s"
          % ("PASS" if nondeg else "FAIL"))
    out = {
        "status": "SCREEN-ONLY",
        "note": "few-minute screen on abdou_test first N lines; NOT a full gate run",
        "window": 256, "control_stride": CONTROL,
        "sig_pct": SIG_PCT, "p_bar": P_BAR,
        "per_stride": {str(s): {k: v for k, v in data[s].items() if k not in ("pred", "ref")}
                       for s in STRIDES},
        "paired_vs_control": {str(s): results[s] for s in results},
        "non_degeneracy_pass": nondeg,
    }
    (BASE / "stride_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", (BASE / "stride_sweep.json").relative_to(REPO).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
