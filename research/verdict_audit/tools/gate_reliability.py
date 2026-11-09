"""How reliable is the 3-gate mean as a ranking target?

Uses only the already-scored probe trajectory (free, CPU). If the gates
disagree with each other across probes, then no dev set - however good - can
correlate strongly with their mean; this estimates that ceiling.
"""
import json
from pathlib import Path

EXT = ["fadel_test", "sadeed25", "wikinews2014"]


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


data = json.loads(Path("research/verdict_audit/probe_trajectory.json").read_text(encoding="utf-8"))
for run, blob in data.items():
    steps = [s for s in blob["steps"] if all(g in blob["rows"][str(s)] for g in EXT)]
    cols = {g: [blob["rows"][str(s)][g]["der_registered"] for s in steps] for g in EXT}
    means = [sum(cols[g][i] for g in EXT) / 3 for i in range(len(steps))]
    print("== %s  (%d probes with all 3 gates)" % (run, len(steps)))
    print("   3-gate mean: min=%.6f  mean=%.6f  max=%.6f  band=%.2f%% of min" % (
        min(means), sum(means) / len(means), max(means),
        100 * (max(means) - min(means)) / min(means)))
    pairs = [(EXT[i], EXT[j]) for i in range(3) for j in range(i + 1, 3)]
    rs = []
    for a, b in pairs:
        r = spearman(cols[a], cols[b])
        rs.append(r)
        print("   rho(%-12s, %-12s) = %+.3f" % (a, b, r))
    rbar = sum(rs) / len(rs)
    sb3 = 3 * rbar / (1 + 2 * rbar) if (1 + 2 * rbar) != 0 else float("nan")
    print("   mean inter-gate rho = %+.3f" % rbar)
    print("   Spearman-Brown reliability of the 3-gate mean = %.3f" % sb3)
    print("   -> ceiling on |rho| any dev set can reach vs this mean = %.3f" % (sb3 ** 0.5))
    # leave-one-gate-out: how well does 1 gate predict the other 2's mean?
    for g in EXT:
        others = [sum(cols[h][i] for h in EXT if h != g) / 2 for i in range(len(steps))]
        print("   rho(%-12s vs mean of other 2) = %+.3f" % (g, spearman(cols[g], others)))
    print()
