"""Stage the 12 C4b probes as checkpoint DIRS for bench.py (which takes a dir)."""
import shutil
from pathlib import Path

src = Path("runs/diac/e70_c4b_ctx256/gate_probe")
dst = Path("runs/diac/e74_devcheck")
steps = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
made = 0
for s in steps:
    f = src / ("weights_step%d.pt" % s)
    if not f.exists():
        print("MISSING", f); continue
    d = dst / ("step%d" % s)
    d.mkdir(parents=True, exist_ok=True)
    t = d / "model.pt"
    if not t.exists():
        shutil.copy2(f, t)
        made += 1
print("staged %d probe dirs under %s (copied %d)" % (len(steps), dst, made))
