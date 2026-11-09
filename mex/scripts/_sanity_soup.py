
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from mex.scripts.merge_soup import load_experts, uniform_merge, save_soup, EXPERTS
import mex.scripts.merge_soup as ms
states = load_experts()  # keys are task names
dup = {k: v.clone() for k, v in states["x1"].items()}
save_soup(uniform_merge({t: dup for t in EXPERTS}), Path("runs/mex/soup/_sanity"))

