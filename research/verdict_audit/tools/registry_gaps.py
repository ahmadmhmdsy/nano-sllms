import re, sys, yaml
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")
d = yaml.safe_load(open("research/experiments.yaml", encoding="utf-8"))
reg = [e["id"] for e in d["experiments"]]
print("experiments.yaml ids:", reg)
txt = open("research/EXPERIMENTS.md", encoding="utf-8").read()
ids = re.findall(r"\bE-(\d+)([a-z]?)\b", txt)
c = Counter(f"E-{n}{s}" for n, s in ids)
print("\nEXPERIMENTS.md distinct ids:", len(c))
dupes = {k: v for k, v in c.items() if v > 1}
print("appearing MORE THAN ONCE:", len(dupes))
for k, v in sorted(dupes.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {k}: {v}")
missing = sorted((f"E-{n}" for n in range(19, 76) if f"E-{n}" not in c), key=lambda s: int(s[2:]))
print("\nE-19..E-75 absent from EXPERIMENTS.md:", missing)
inreg = set(reg)
notreg = [k for k in sorted(c, key=lambda s: int(re.sub(r"\D", "", s))) if "diacritizer/" + k not in inreg]
print(f"\nEXPERIMENTS.md ids with NO experiments.yaml row ({len(notreg)}):", notreg)
