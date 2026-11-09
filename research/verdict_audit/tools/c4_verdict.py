import json, os, csv, collections, sys
sys.stdout.reconfigure(encoding="utf-8")
EXT = ["fadel_test", "sadeed25", "wikinews2014"]
def gates(run):
    p = f"runs/diac/{run}/gate_eval.csv"
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        by[int(r["step"])][r["gate"]] = float(r["DER"])
    return {s: sum(d[g] for g in EXT)/3 for s, d in by.items() if all(g in d for g in EXT)}
def valacc(run):
    out = {}
    for l in open(f"runs/diac/{run}/monitor/log.jsonl", encoding="utf-8"):
        try: o = json.loads(l)
        except: continue
        if o.get("event") == "on_eval" and o.get("val_acc") is not None: out[int(o["step"])] = o["val_acc"]
    return out
R4, R1 = "e69_c4_ctx256", "e67_c1_ft64_fullv3"
g4, g1 = gates(R4), gates(R1)
v4, v1 = valacc(R4), valacc(R1)
b4 = min(g4.items(), key=lambda kv: kv[1])
print(f"C4 (ctx256) : probes {len(g4)}  BEST {b4[1]:.6f} @{b4[0]}   final {max(g4)}:{g4[max(g4)]:.4f}")
print(f"C1 (ctx128) : probes {len(g1)}  BEST {min(g1.values()):.6f} @{min(g1,key=g1.get)}")
print(f"C4 val_acc max {max(v4.values()):.6f}   C1 val_acc max {max(v1.values()):.6f}   delta {max(v4.values())-max(v1.values()):+.6f}")
c4 = min(g4.values()); c1 = min(g1.values())
print(f"\nFORMAL (pre-registered) RULE: best {c4:.6f} vs C1 {c1:.6f} = {(c4-c1)/c1*100:+.3f}% (negative = C4 better)")
print(f"  WIN bar <=0.37485 -> {'MET' if c4<=0.37485 else 'NOT MET'}")
print(f"  PARTIAL if <=0.3825 -> {'YES' if c4<=c1 else 'NO - C4 does not even beat C1'}")
print(f"  verdict: {'WIN' if c4<=0.37485 else ('PARTIAL' if c4<=c1 else 'FALSIFIED')}")
common = sorted(set(g4) & set(g1))
print(f"\nMATCHED-STEP RANGE (steps {min(common)}..{max(common)}, n={len(common)}):")
m4 = min(((g4[s], s) for s in common)); m1 = min(((g1[s], s) for s in common))
m4 = (m4[1], m4[0]); m1 = (m1[1], m1[0])
print(f"  C4 {m4[1]:.6f} @{m4[0]}   C1 {m1[1]:.6f} @{m1[0]}   = {(m4[1]-m1[1])/m1[1]*100:+.3f}%")
print("  C1 curve in range:", " ".join(f"{s}:{g1[s]:.4f}" for s in common))
print("  C4 curve in range:", " ".join(f"{s}:{g4[s]:.4f}" for s in common))
print("\nclosure: summary", os.path.exists(f"runs/diac/{R4}/monitor/summary.json"), "final", os.path.exists(f"runs/diac/{R4}/final/model.pt"))
print(open(f"runs/diac/{R4}/monitor/summary.json").read().replace(chr(10)," ") if os.path.exists(f"runs/diac/{R4}/monitor/summary.json") else "")
print("best_gate:", open(f"runs/diac/{R4}/gate_probe/best_gate.json").read())
