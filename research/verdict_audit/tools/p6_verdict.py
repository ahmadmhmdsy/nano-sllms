import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("research/verdict_audit/tools").resolve()))
sys.path.insert(0, str(Path("diacritizer/src").resolve()))
import eval_metrics as EM
sys.stdout.reconfigure(encoding="utf-8")
EXT = ["fadel_test", "sadeed25", "wikinews2014"]
D = Path("runs/diac/e70_c4b_ctx256/p6_vote")
RECORDED = {"fadel_test": 0.2995331659978577, "sadeed25": 0.39254526896760294,
            "wikinews2014": 0.4322979328447993}
def rep(pred):
    ref = D / (pred.stem.replace("_single", "_single").replace("_vote", "_single") + ".ref.txt")
    ref = D / (pred.name.split("_single")[0].split("_vote")[0] + "_single.ref.txt")
    p = pred.read_text(encoding="utf-8").splitlines()
    r = ref.read_text(encoding="utf-8").splitlines()
    sc = EM.score_corpus(p, r)
    s = EM.summarize(sc)
    return s, sc, p, r
rows = {}
for g in EXT:
    s_single, sc_s, ps, rs = rep(D / (g + "_single.txt"))
    s_vote, sc_v, pv, rv = rep(D / (g + "_vote.txt"))
    mc = EM.mcnemar(ps, pv, rs, canonical=True)
    rows[g] = dict(single=s_single["der_registered"], vote=s_vote["der_registered"],
                   single_can=s_single["der_canonical"], vote_can=s_vote["der_canonical"],
                   single_f1=s_single["macro_f1"], vote_f1=s_vote["macro_f1"],
                   recorded=RECORDED[g], mc=mc, n=len(ps))
for g in EXT:
    r = rows[g]
    print(f"{g:14s} n={r['n']:5d}  single {r['single']:.6f} (recorded {r['recorded']:.6f})  "
          f"vote {r['vote']:.6f}  {(r['vote']-r['single'])/r['single']*100:+.3f}%")
    print(f"{'':14s}        canonical {r['single_can']:.6f} -> {r['vote_can']:.6f} "
          f"({(r['vote_can']-r['single_can'])/r['single_can']*100:+.3f}%)  "
          f"macro_f1 {r['single_f1']:.4f} -> {r['vote_f1']:.4f}  "
          f"McNemar b={r['mc']['b_a_correct_only']} c={r['mc']['c_b_correct_only']} p={r['mc']['p_value_two_sided']:.2e}")
ms = sum(rows[g]["single"] for g in EXT)/3
mv = sum(rows[g]["vote"] for g in EXT)/3
mr = sum(rows[g]["recorded"] for g in EXT)/3
print(f"\n3-GATE MEAN  single {ms:.6f}   (recorded probe {mr:.6f} = E-070 headline 0.374811)")
print(f"             vote   {mv:.6f}   rel {(mv-ms)/ms*100:+.4f}%")
print(f"BASELINE REPRODUCTION: |single - recorded| = {abs(ms-mr):.6f} -> "
      f"{'PASS' if abs(ms-mr) < 0.002 else 'CHECK'}")
print(f"PRE-REGISTERED RULE: WIN needs <= -1.00% -> "
      f"{'WIN' if (mv-ms)/ms*100 <= -1.0 else ('PARTIAL' if mv < ms else 'FALSIFIED')}")
deg = any(abs(rows[g]['vote'] - rows[g]['vote_can']) < 1e-12 and rows[g]['vote_f1'] < 0.2 for g in EXT)
print(f"DEGENERACY CHECK: {'DEGENERATE' if deg else 'non-degenerate (der_anyalt != der_canonical, macro_f1 healthy)'}")
json.dump({g: {k: v for k, v in rows[g].items()} for g in EXT} |
          {"mean_single": ms, "mean_vote": mv, "mean_recorded": mr},
          open("runs/diac/e70_c4b_ctx256/p6_vote/verdict.json", "w", encoding="utf-8"), indent=2)
