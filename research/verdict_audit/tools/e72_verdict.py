import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("diacritizer/src").resolve()))
import eval_metrics as EM
sys.stdout.reconfigure(encoding="utf-8")
D = Path("runs/diac/e70_c4b_ctx256/abdou")
ref = (D / "step15000_pred.ref.txt").read_text(encoding="utf-8").splitlines()
steps = ["step15000", "step40000", "step60000"]
res = {}
for s in steps:
    p = (D / (s + "_pred.txt")).read_text(encoding="utf-8").splitlines()
    sc = EM.score_corpus(p, ref); su = EM.summarize(sc)
    res[s] = dict(pred=p, der=su["der_registered"], can=su["der_canonical"],
                  f1=su["macro_f1"], n=len(p))
    print(f"{s:10s} n={res[s]['n']:6d}  der_registered {res[s]['der']:.6f}  "
          f"der_canonical {res[s]['can']:.6f}  macro_f1 {res[s]['f1']:.4f}")
print()
best = min(steps, key=lambda s: res[s]["der"])
print("abdou_test ordering (lower better):", " < ".join(sorted(steps, key=lambda s: res[s]['der'])))
print("best on abdou:", best, "| gate-best was step15000")
for other in ["step40000", "step60000"]:
    mc = EM.mcnemar(res["step15000"]["pred"], res[other]["pred"], ref, canonical=False)
    mc2 = EM.mcnemar(res["step15000"]["pred"], res[other]["pred"], ref, canonical=True)
    print(f"  step15000 vs {other}: d_der {res['step15000']['der']-res[other]['der']:+.6f} "
          f"({(res['step15000']['der']-res[other]['der'])/res[other]['der']*100:+.3f}%)  "
          f"McNemar nocase b={mc['b_a_correct_only']} c={mc['c_b_correct_only']} p={mc['p_value_two_sided']:.3g} | "
          f"canonical p={mc2['p_value_two_sided']:.3g}")
print()
print("PRE-REGISTERED RULE:")
print("  CORROBORATED iff step15000 beats BOTH step40000 and step60000 on abdou DER")
print("  NOISE-SUSPECT iff the abdou ordering disagrees with the gate ordering")
print("  INCONCLUSIVE if ordering disagrees but is within the paired CI")
for other in ["step40000", "step60000"]:
    mc = EM.mcnemar(res["step15000"]["pred"], res[other]["pred"], ref, canonical=False)
    sig = mc["p_value_two_sided"] < 0.05
    print(f"  vs {other}: step15000 {'BEATS' if res['step15000']['der']<res[other]['der'] else 'LOSES TO'} it, "
          f"McNemar {'significant' if sig else 'NOT significant'} (p={mc['p_value_two_sided']:.3g})")
json.dump({s: {k: v for k, v in res[s].items() if k != "pred"} for s in steps},
          open(D / "verdict.json", "w", encoding="utf-8"), indent=2)
