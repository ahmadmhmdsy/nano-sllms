import json, os, csv, collections, sys
sys.stdout.reconfigure(encoding="utf-8")
EXT = ["fadel_test", "sadeed25", "wikinews2014"]
def gates(run):
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"runs/diac/{run}/gate_eval.csv", encoding="utf-8")):
        by[int(r["step"])][r["gate"]] = float(r["DER"])
    return {s: sum(d[g] for g in EXT)/3 for s, d in by.items() if all(g in d for g in EXT)}
def valacc(run):
    o = {}
    for l in open(f"runs/diac/{run}/monitor/log.jsonl", encoding="utf-8"):
        try: r = json.loads(l)
        except: continue
        if r.get("event") == "on_eval" and r.get("val_acc") is not None: o[int(r["step"])] = r["val_acc"]
    return o
R, R1, R4 = "e70_c4b_ctx256", "e67_c1_ft64_fullv3", "e69_c4_ctx256"
g, g1, g4 = gates(R), gates(R1), gates(R4)
ranked = sorted(g.items(), key=lambda kv: kv[1])
b = ranked[0]
print(f"C4b (ctx256)  : {len(g)} probes  BEST {b[1]:.6f} @{b[0]}")
print(f"  second-best  : {ranked[1][1]:.6f} @{ranked[1][0]}   (+{(ranked[1][1]-b[1])/b[1]*100:.2f}%)")
print(f"  full curve   : " + " ".join(f"{s}:{v:.4f}" for s, v in sorted(g.items())))
print(f"  last probe   : {max(g)}:{g[max(g)]:.4f}")
v = valacc(R)
print(f"  val_acc max {max(v.values()):.6f} vs C1 {max(valacc(R1).values()):.6f} (delta {max(v.values())-max(valacc(R1).values()):+.6f})")
c1 = min(g1.values()); c4 = min(g4.values())
print(f"\nE-070 PRE-REGISTERED RULE: best {b[1]:.6f}")
print(f"  WIN <=0.3825          -> {'MET' if b[1] <= 0.3825 else 'no'}")
print(f"  PARTIAL <=0.387245    -> {'(would apply)' if 0.3825 < b[1] <= 0.387245 else 'n/a'}")
print(f"  FALSIFIED >0.387245   -> {'yes' if b[1] > 0.387245 else 'no'}")
print(f"  => {'WIN - NEW PROJECT BEST' if b[1] <= 0.3825 else ('PARTIAL' if b[1] <= 0.387245 else 'FALSIFIED')}")
print(f"  vs C1 {c1:.6f}: {(b[1]-c1)/c1*100:+.4f}%   vs C4 {c4:.6f}: {(b[1]-c4)/c4*100:+.4f}%")
print(f"  stricter 2% bar <=0.37485 -> {'MET by ' + format(0.37485-b[1], '.6f') if b[1] <= 0.37485 else 'NOT met'}")
common = sorted(set(g) & set(g1))
m_b = min(((g[s], s) for s in common)); m_1 = min(((g1[s], s) for s in common))
print(f"\nMATCHED TOKENS vs C1 (steps {min(common)}..{max(common)}, n={len(common)}):")
print(f"  C4b {m_b[0]:.6f} @{m_b[1]}   C1 {m_1[0]:.6f} @{m_1[1]}   = {(m_b[0]-m_1[0])/m_1[0]*100:+.3f}%")
print(f"  C4b better than C1 at same step in {sum(1 for s in common if g[s] < g1[s])}/{len(common)} steps")
common4 = sorted(set(g) & set(g4))
print(f"\nvs C4 at matched steps (n={len(common4)}): C4b " + str(round(min(g[s] for s in common4), 6)) + " vs C4 " + str(round(min(g4[s] for s in common4), 6)))
print("\nclosure: summary", os.path.exists(f"runs/diac/{R}/monitor/summary.json"), "final", os.path.exists(f"runs/diac/{R}/final/model.pt"))
print(open(f"runs/diac/{R}/monitor/summary.json").read().replace(chr(10), " ") if os.path.exists(f"runs/diac/{R}/monitor/summary.json") else "")
print("best_gate:", open(f"runs/diac/{R}/gate_probe/best_gate.json").read())
