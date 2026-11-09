"""E-082 cadence-restart throughput probe: orchestrator.

PRE-REGISTERED 2026-10-11. See runs/diac/e82_gen/cadence.yaml for the design.

WHAT IT DOES: launches the real arm-C-workload trainer, lets it run past a step
target, STOPS it, verifies the newest periodic checkpoint actually loads, then
relaunches the exact original command with zero flags (auto-resume). Three
cycles, two restarts, 12,000 steps total.

WHY: E-081 (closed, accepted) observed that one stop/relaunch cleared the
945 MHz clock latch and restored the full fast mode (~23 s/100) for about 3,600
steps. n=1. This probe tests reproducibility of that window on a cadence.

SAFETY: strictly sequential. The next launch happens only after the previous
process is confirmed gone AND no train.py process remains, so there is never a
GPU co-run. Only processes this script started are stopped.

PRE-REGISTERED RULE (fixed before launch):
  Per-cycle rate r_i = seconds per 100 steps over the probe-free range of cycle
  i, read from monitor/log.jsonl. t RESETS on each resume, so only the final
  monotonically-increasing segment of each process is used.
    CADENCE WORKS iff every restart cycle has r <= 30 s/100
    CADENCE FAILS iff any restart cycle is >= 60 s/100 for the majority of steps
    PARTIAL       otherwise
  SECONDARY: fast-window length in steps per cycle, and the clock state.
  This is a THROUGHPUT calibration, not a model result.
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = REPO / ".venv" / "Scripts" / "python.exe"
CONFIG = "runs/diac/e82_gen/cadence.yaml"
PHASE = "e82_cadence"
RUN_DIR = REPO / "runs" / "diac" / PHASE
LOGF = RUN_DIR / "monitor" / "log.jsonl"
GEN = REPO / "runs" / "diac" / "e82_gen"
ORCH = GEN / "orchestrator.log"
TARGETS = [4000, 8000, 12000]
MARGIN = 200
MAX_WAIT = 5 * 3600


def log(msg):
    line = time.strftime("%H:%M:%S") + "  " + msg
    print(line, flush=True)
    with ORCH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def latest_step():
    if not LOGF.exists():
        return 0
    step = 0
    for ln in LOGF.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("event") == "on_step":
            try:
                step = max(step, int(e.get("step", 0)))
            except Exception:
                pass
    return step


def train_procs():
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*train.py*' } | "
           "Select-Object -ExpandProperty ProcessId")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def launch(cycle):
    out_path = GEN / ("cycle" + str(cycle) + ".log")
    f = out_path.open("a", encoding="utf-8")
    p = subprocess.Popen([str(PY), "diacritizer/scripts/train.py", "--config", CONFIG],
                         cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    log("cycle " + str(cycle) + ": launched pid " + str(p.pid))
    return p


def newest_ckpt():
    cks = []
    for d in RUN_DIR.glob("checkpoint-*"):
        try:
            cks.append((int(d.name.split("-")[1]), d))
        except Exception:
            pass
    if not cks:
        return None
    return sorted(cks)[-1][1]


def check_load(path):
    code = ("import torch,sys;"
            "c=torch.load(sys.argv[1],map_location='cpu',weights_only=False);"
            "print('step=' + str(c.get('step')))")
    out = subprocess.run([str(PY), "-c", code, str(path)],
                         capture_output=True, text=True, cwd=str(REPO))
    return out.returncode, ((out.stdout or "") + (out.stderr or "")).strip()


def stop_and_verify(cycle):
    left = train_procs()
    log("cycle " + str(cycle) + ": killing " + str(len(left)) + " train proc(s): " + str(left))
    for pid in left:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Stop-Process -Id " + str(pid) + " -Force"],
                       capture_output=True, text=True)
    for _ in range(40):
        if not train_procs():
            break
        time.sleep(2)
    left = train_procs()
    log("cycle " + str(cycle) + ": leftover train procs after stop = " + str(left))
    time.sleep(8)
    d = newest_ckpt()
    if d is None:
        log("cycle " + str(cycle) + ": NO CHECKPOINT FOUND - aborting")
        return False
    rc, val = check_load(d / "state.pt")
    log("cycle " + str(cycle) + ": checkpoint " + d.name + " load rc=" + str(rc) + " " + val)
    if rc != 0:
        log("cycle " + str(cycle) + ": checkpoint did NOT load - aborting before relaunch")
        return False
    return True


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    log("=== E-082 cadence-restart probe start; targets " + str(TARGETS) + " ===")
    for cycle, target in enumerate(TARGETS, start=1):
        p = launch(cycle)
        goal = target + MARGIN
        t0 = time.time()
        while True:
            if p.poll() is not None:
                log("cycle " + str(cycle) + ": process exited rc=" + str(p.returncode)
                    + " at step " + str(latest_step()))
                break
            st = latest_step()
            if st >= goal:
                log("cycle " + str(cycle) + ": reached step " + str(st) + " (goal " + str(goal) + ")")
                break
            if time.time() - t0 > MAX_WAIT:
                log("cycle " + str(cycle) + ": TIMEOUT waiting for step " + str(goal)
                    + " (at " + str(st) + ")")
                break
            time.sleep(5)
        if cycle < len(TARGETS):
            if not stop_and_verify(cycle):
                log("ABORT: refusing to relaunch")
                return
            time.sleep(5)
    log("=== ALL CYCLES DONE ===")


if __name__ == "__main__":
    main()
