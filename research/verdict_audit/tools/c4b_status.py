import json, os, csv, collections, sys, datetime
sys.stdout.reconfigure(encoding="utf-8")
R = "runs/diac/e70_c4b_ctx256"
EXT = ["fadel_test", "sadeed25", "wikinews2014"]
rs = [json.loads(l) for l in open(f"{R}/monitor/log.jsonl", encoding="utf-8") if l.strip()]
st = [r for r in rs if r.get("event") == "on_step" and r.get("train_loss") is not None]
ev = [r for r in rs if r.get("event") == "on_eval" and r.get("val_acc") is not None]
cur, t = st[-1]["step"], st[-1]["t"]
w = st[-10:] if len(st) >= 10 else st
rate = (w[-1]["t"] - w[0]["t"]) / (w[-1]["step"] - w[0]["step"])
eta = (60000 - cur) * rate
print(f"C4b step {cur:,}/60,000 ({100*cur/60000:.1f}%)  elapsed {t/3600:.2f} h  recent rate {rate:.4f} s/step")
print(f"ETA {eta/3600:.2f} h  (finish ~{(datetime.datetime.now()+datetime.timedelta(seconds=eta)).strftime('%H:%M')})")
print(f"train_loss {st[-1]['train_loss']:.4f}")
if ev:
    print("val_acc:", " ".join(f"{e['step']}:{e['val_acc']:.4f}" for e in ev[-5:]), "| best", max(e["val_acc"] for e in ev))
else:
    print("no eval yet")
p = f"{R}/gate_eval.csv"
if os.path.exists(p):
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        by[int(r["step"])][r["gate"]] = float(r["DER"])
    print("--- gates (3-gate mean, lower better) ---")
    bs = None
    for s in sorted(by):
        d = by[s]
        if all(g in d for g in EXT):
            m = sum(d[g] for g in EXT)/3
            if bs is None or m < bs[1]: bs = (s, m)
            print(f"  {s:>6}: {m:.4f}  (fadel {d['fadel_test']:.2f} sadeed {d['sadeed25']:.2f} wn2014 {d['wikinews2014']:.2f})")
    print(f"  BEST {bs[1]:.6f} @{bs[0]}   WIN<=0.3825 PARTIAL<=0.387245  (C4 0.387245, C1 0.382541)")
else:
    print("no gate probes yet (first at step 5000)")
print("closed:", os.path.exists(f"{R}/monitor/summary.json"), "| final:", os.path.exists(f"{R}/final/model.pt"))
