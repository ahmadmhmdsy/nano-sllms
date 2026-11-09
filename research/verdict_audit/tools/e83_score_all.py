"""E-083 - score every scheme directory in parallel (CPU-only).

Reuses the tested score_vote_dir.py per scheme, one subprocess each, so the seven
scheme tables are produced concurrently instead of serially. Reads nothing else.
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "runs" / "diac" / "e83_weighted"
SCHEMES = ["base24", "c4b12", "c412", "b2", "b4", "b8", "ramp_b1"]
REFS = [
    "L1_gate=0.351324", "L1_abdou=0.350703",
    "c4b_soup_gate=0.352209", "c4b_soup_abdou=0.349044",
    "L2_gate=0.355179", "L2_abdou=0.354905",
    "all24_weight_gate=0.449353",
]


def one(s):
    d = ROOT / s
    cmd = [str(REPO / ".venv" / "Scripts" / "python.exe"), "-u",
           "research/verdict_audit/tools/score_vote_dir.py",
           "--base", str(d.relative_to(REPO)), "--kinds", "single",
           "--out", str(d / "verdict.json"),
           "--title", "E-083 scheme %s" % s]
    for r in REFS:
        cmd += ["--ref", r]
    log = ROOT / (s + ".score.log")
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT).returncode
    return s, rc, log


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=len(SCHEMES)) as ex:
        res = list(ex.map(one, SCHEMES))
    for s, rc, log in res:
        print("===== scheme %s rc=%d =====" % (s, rc))
        print(log.read_text(encoding="utf-8", errors="replace"))
    print("E83 SCORE ALL DONE")
