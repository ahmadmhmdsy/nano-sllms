import json, os, csv, collections, sys, datetime
sys.stdout.reconfigure(encoding="utf-8")
R = "runs/diac/e69_c4_ctx256"
rs = [json.loads(l) for l in open(f"{R}/monitor/log.jsonl", encoding="utf-8") if l.strip()]
steps = [r for r in rs if r.get("event") == "on_step"]
evals = [r for r in rs if r.get("event") == "on_eval"]
cur = steps[-1]["step"]; t = steps[-1]["t"]
total = 60000
rate = (steps[-1]["t"] - steps[0]["t"]) / (steps[-1]["step"] - steps[0]["step"])
eta = (total - cur) * rate
print(f"step {cur:,} / {total:,}  ({100*cur/total:.1f}%)   elapsed {t/3600:.2f} h   rate {rate:.4f} s/step")
print(f"ETA {eta/3600:.2f} h  (finish ~{(datetime.datetime.now()+datetime.timedelta(seconds=eta)).strftime('%H:%M')})")
losses = [s["train_loss"] for s in steps if s.get("train_loss") is not None]
print(f"train_loss last {losses[-1]:.4f}   min {min(losses):.4f}")
if evals:
    print("val:", " ".join(f"{e['step']}:loss{e.get('val_loss',float('nan')):.4f}/acc{e.get('val_acc',float('nan')):.4f}" for e in evals[-4:]))
    print("best val_acc so far:", max(e.get("val_acc", 0) for e in evals))
else:
    print("no eval yet (first at step 5000)")
p = f"{R}/gate_eval.csv"
if os.path.exists(p):
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        by[int(r["step"])][r["gate"]] = float(r["DER"])
    EXT = ["fadel_test", "sadeed25", "wikinews2014"]
    print("--- gate curve (3-gate mean, lower better) ---")
    best = None
    for s in sorted(by):
        d = by[s]
        if all(g in d for g in EXT):
            m = sum(d[g] for g in EXT) / 3
            if best is None or m < best[1]: best = (s, m)
            print(f"  {s:>6}: mean3 {m:.4f}  fadel {d['fadel_test']:.4f} sadeed {d['sadeed25']:.4f} wn2014 {d['wikinews2014']:.4f}")
    if best: print(f"  BEST SO FAR: {best[1]:.6f} @{best[0]}   (WIN bar <=0.37485, C1 0.3825)")
else:
    print("no gate_eval.csv yet (first gate at step 5000)")
print("artifacts: summary", os.path.exists(f"{R}/monitor/summary.json"), "| final", os.path.exists(f"{R}/final/model.pt"),
      "| best_gate", os.path.exists(f"{R}/gate_probe/best_gate.json"))
