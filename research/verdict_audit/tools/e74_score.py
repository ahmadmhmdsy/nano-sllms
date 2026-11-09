"""E-074: does abdou_test rank the 12 C4b probes the way the 3-gate mean does?"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path("diacritizer/src")))
import eval_metrics as em

EXT = ["fadel_test", "sadeed25", "wikinews2014"]
STEPS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
CEIL = 0.963


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((a[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((b[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else float("nan")


def spearman(a, b):
    return pearson(rank(a), rank(b))


def kendall(a, b):
    n = len(a); conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0: conc += 1
            elif s < 0: disc += 1
    tot = n * (n - 1) / 2
    return (conc - disc) / tot if tot else float("nan")


traj = json.loads(Path("research/verdict_audit/probe_trajectory.json").read_text(encoding="utf-8"))
rows = traj["e70_c4b_ctx256"]["rows"]
gate = {}
for s in STEPS:
    r = rows.get(str(s))
    if r and all(g in r for g in EXT):
        gate[s] = sum(r[g]["der_registered"] for g in EXT) / 3

abd = {}
for s in STEPS:
    pf = Path("runs/diac/e74_devcheck/pred/step%d.txt" % s)
    rf = pf.with_suffix(".ref.txt")
    if pf.exists() and rf.exists():
        sm = em.summarize(em.score_corpus(
            pf.read_text(encoding="utf-8").splitlines(),
            rf.read_text(encoding="utf-8").splitlines()))
        abd[s] = sm["der_registered"]

have = [s for s in STEPS if s in gate and s in abd]
print("probes with both readings: %d" % len(have))
if len(have) < 4:
    print("not enough probes scored yet - run e74_run.ps1 first"); sys.exit(0)

gv = [gate[s] for s in have]
av = [abd[s] for s in have]
rho = spearman(gv, av)
tau = kendall(gv, av)
print("\n%-8s %-12s %-12s" % ("step", "gate_mean3", "abdou_der"))
for s in sorted(have, key=lambda x: gate[x]):
    print("%-8d %-12.6f %-12.6f" % (s, gate[s], abd[s]))

random.seed(42)
boot = []
n = len(have)
for _ in range(2000):
    idx = [random.randrange(n) for _ in range(n)]
    r = spearman([gv[i] for i in idx], [av[i] for i in idx])
    if r == r:
        boot.append(r)
boot.sort()
lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
print("\nSpearman rho = %+.3f  95%% CI [%+.3f, %+.3f]  (ceiling %.3f)" % (rho, lo, hi, CEIL))
print("Kendall tau  = %+.3f" % tau)
print("fraction of ceiling reached = %.1f%%" % (100 * rho / CEIL))

gate_best = min(have, key=lambda s: gate[s])
abd_best = min(have, key=lambda s: abd[s])
gate_top3 = sorted(have, key=lambda s: gate[s])[:3]
print("\ngate argmin = step %d ; abdou argmin = step %d ; gate top3 = %s" % (
    gate_best, abd_best, gate_top3))

if rho >= 0.70 and abd_best in gate_top3:
    v = "USABLE"
elif rho >= 0.40:
    v = "WEAK"
else:
    v = "UNUSABLE"
print("\nE-074 VERDICT: %s" % v)
Path("runs/diac/e74_devcheck").mkdir(parents=True, exist_ok=True)
Path("runs/diac/e74_devcheck/verdict.json").write_text(json.dumps({
    "rho": rho, "rho_ci": [lo, hi], "kendall_tau": tau, "ceiling": CEIL,
    "gate_argmin": gate_best, "abdou_argmin": abd_best, "gate_top3": gate_top3,
    "verdict": v, "probes": have,
    "gate_mean3": {str(s): gate[s] for s in have},
    "abdou_der": {str(s): abd[s] for s in have}}, indent=2), encoding="utf-8")
print("wrote runs/diac/e74_devcheck/verdict.json")
