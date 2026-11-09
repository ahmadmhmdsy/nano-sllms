import yaml, sys
sys.stdout.reconfigure(encoding="utf-8")
c1 = yaml.safe_load(open("runs/diac/e67_gen/c1_ft64_fullv3.yaml", encoding="utf-8"))
c4 = yaml.safe_load(open("runs/diac/e67_gen/c4_ctx256.yaml", encoding="utf-8"))
keys = sorted(set(c1) | set(c4))
diff = {k: (c1.get(k, "<absent>"), c4.get(k, "<absent>")) for k in keys if c1.get(k) != c4.get(k)}
print("changed keys:")
for k, (a, b) in diff.items():
    print("  ", k, ":", a, "->", b)
expect = {"phase", "model", "pretrained_init", "batch_size", "total_steps", "tokens_phase"}
actual = set(diff)
print("expected:", sorted(expect))
print("UNEXPECTED EXTRA DIFFS:", sorted(actual - expect) or "none")
print("MISSING EXPECTED DIFFS:", sorted(expect - actual) or "none")
print("model diff:", {k: (c1["model"].get(k), c4["model"].get(k)) for k in set(c1["model"]) | set(c4["model"]) if c1["model"].get(k) != c4["model"].get(k)})
print("DIFF CHECK:", "PASS" if actual == expect else "REVIEW")
