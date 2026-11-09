#!/usr/bin/env python
"""E-093 scorer: does the majority step of the sliding-window vote earn its place?

*** FEW-MINUTE SCREEN, NOT A FULL GATE RUN. ***

Scores three arms emitted from ONE loaded model - single (hard-chunked), vote
(majority, ties -> most central) and centered (NO vote, most central always) - on
the same 400 abdou pairs, and runs exact paired McNemar between them.

Applies the pre-registered rule recorded in E-093. CPU only.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
import eval_metrics as em  # noqa: E402

BASE = REPO / "runs" / "diac" / "e93_centered"
SRC = "abdou_test"
ARMS = ["single", "vote", "centered"]
SIG_PCT = 2.0
P_BAR = 0.05


def lines(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    ref = lines(BASE / ("%s.ref.txt" % SRC))
    data = {}
    for a in ARMS:
        pf = BASE / ("%s.%s.txt" % (SRC, a))
        if not pf.exists():
            print("MISSING", pf)
            return 2
        pred = lines(pf)
        if len(pred) != len(ref):
            print("LINE MISMATCH %s: %d pred vs %d ref - not a paired comparison"
                  % (a, len(pred), len(ref)))
            return 2
        sm = em.summarize(em.score_corpus(pred, ref))
        data[a] = {"pred": pred, "der_registered": sm["der_registered"],
                   "der_canonical": sm["der_canonical"],
                   "macro_f1": sm["macro_f1"], "der_anyalt": sm["der_anyalt"]}

    print("*** FEW-MINUTE SCREEN (abdou_test first 400 pairs -> %d line-segments),"
          " NOT A FULL GATE RUN ***" % len(ref))
    print("ONE loaded model, three mark-selection rules; window 256 / stride 128\n")
    for a in ARMS:
        print("  %-9s der_registered %.6f   macro_f1 %.4f   der_anyalt %.4f"
              % (a, data[a]["der_registered"], data[a]["macro_f1"], data[a]["der_anyalt"]))

    print("\nPAIRED exact McNemar (a=first, b=second; b_* = a correct & b wrong):")
    pairs = [("single", "vote"), ("single", "centered"), ("vote", "centered")]
    res = {}
    for a, b in pairs:
        mc = em.mcnemar(data[a]["pred"], data[b]["pred"], ref)
        rel = (data[b]["der_registered"] - data[a]["der_registered"]) / data[a]["der_registered"] * 100.0
        res["%s_vs_%s" % (a, b)] = {"rel_pct_b_minus_a": rel, "mcnemar": mc}
        print("  %-9s -> %-9s rel %+7.2f%%  discordant=%4d (%s-only %4d, %s-only %4d)  p=%.3g"
              % (a, b, rel, mc["discordant"], a, mc["b_a_correct_only"], b,
                 mc["c_b_correct_only"], mc["p_value_two_sided"]))

    mv = res["vote_vs_centered"]
    rel = mv["rel_pct_b_minus_a"]
    p = mv["mcnemar"]["p_value_two_sided"]
    if rel <= SIG_PCT and not (p < P_BAR and rel >= SIG_PCT):
        # centered is not worse than vote by >= 2 pct
        if rel < SIG_PCT:
            verdict = "VOTE-REDUNDANT"
        else:
            verdict = "INCONCLUSIVE"
    if rel >= SIG_PCT and p < P_BAR:
        verdict = "VOTE-EARNS"
    elif rel < SIG_PCT:
        verdict = "VOTE-REDUNDANT"
    else:
        verdict = "INCONCLUSIVE"
    print("\nPRE-REGISTERED RULE: centered vs vote rel %+.2f%%, p=%.3g -> %s" % (rel, p, verdict))

    nondeg = all(abs(data[a]["der_anyalt"] - 0.7554) > 0.05 for a in ARMS)
    print("non-degeneracy: %s" % ("PASS" if nondeg else "FAIL"))
    out = {"status": "SCREEN-ONLY", "window": 256, "stride": 128,
           "sig_pct": SIG_PCT, "p_bar": P_BAR, "verdict": verdict,
           "per_arm": {a: {k: v for k, v in data[a].items() if k != "pred"} for a in ARMS},
           "paired": res, "non_degeneracy_pass": nondeg}
    (BASE / "centered_screen.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", (BASE / "centered_screen.json").relative_to(REPO).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
