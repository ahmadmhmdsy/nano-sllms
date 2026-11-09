"""E-080 clock / power-state calibration probe (pre-registered 2026-10-06).

QUESTION: does forcing the NVIDIA power-management profile to "Prefer maximum
performance" lift the D-line training SM clock above the measured 945 MHz base
pin, and what step-rate change does it buy?

Background (MEMORY, 2026-10-06): 650 consecutive nvidia-smi samples taken while
runs/diac/e79_c_18l64 trained showed clocks.sm pinned at 945 MHz (never above),
pstate P2, 38-41 W of an 80 W limit, 70-73 C against an 87 C target, 9-11 %
memory util, 0.00 MB/s disk read. The same run's own monitor log shows a
TWO-MODE rate: 15.2-16.2 s/100 for steps 100-5000, then 52.0-54.3 s/100 for
steps 6000-18400 -- a 3.3x swing. 945 -> 2100 MHz is only 2.2x, so the clock
ceiling alone CANNOT explain the swing; this probe isolates the clock half.

STAGE 1 (burn): ~90 s sustained fp16 matmul burn + 1 Hz clock sampling.
  The decisive, cheapest test of whether the profile change bound at all.
STAGE 2 (train): only if stage 1 lifts the clock. Runs the REAL trainer on the
  E-079 arm C shape (18L/64h/256ffn/8H/2KV, batch 64, ctx 256, v3ctx256 bed)
  for a bounded 6000 steps in its OWN phase dir, with the same 1 Hz sampling.

Subcommands:
  burn     [--seconds 90] [--out DIR]
  train    [--config YAML] [--out DIR] [--run-dir DIR]
  analyze  --csv CSV [--run-dir DIR] [--out DIR]

Artifacts (default DIR = runs/diac/e80_gen):
  clock_samples.csv  probe_run.log  timing.json
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASE_SM_MHZ = 945.0
QUERY = ("timestamp,pstate,clocks.sm,clocks.mem,utilization.gpu,"
         "utilization.memory,power.draw,temperature.gpu,memory.used")
FIELDS = ["ts", "pstate", "clocks_sm", "clocks_mem", "util_gpu", "util_mem",
          "power_w", "temp_c", "mem_used_mib"]


def nvidia_smi():
    for cand in ("nvidia-smi", "C:/Windows/System32/nvidia-smi.exe"):
        try:
            r = subprocess.run([cand, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                return cand
        except Exception:
            continue
    raise SystemExit("[FATAL] nvidia-smi not found")


def _num(tok):
    tok = tok.strip()
    if tok in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


class Sampler(threading.Thread):
    """1 Hz nvidia-smi sampler writing rows to a CSV."""

    def __init__(self, smi, csv_path, period=1.0):
        super().__init__(daemon=True)
        self.smi, self.csv_path, self.period = smi, csv_path, period
        self.stop_evt = threading.Event()
        self.rows = []
        self.errors = 0

    def run(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            fh.flush()
            while not self.stop_evt.is_set():
                t0 = time.time()
                try:
                    r = subprocess.run(
                        [self.smi, "--query-gpu=" + QUERY,
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=20)
                    line = (r.stdout or "").strip().splitlines()
                    if r.returncode == 0 and line:
                        parts = [p.strip() for p in line[0].split(",")]
                        if len(parts) == len(FIELDS):
                            row = [parts[0], parts[1]] + [_num(p) for p in parts[2:]]
                            w.writerow(row)
                            fh.flush()
                            self.rows.append(dict(zip(FIELDS, row)))
                        else:
                            self.errors += 1
                    else:
                        self.errors += 1
                except Exception:
                    self.errors += 1
                dt = self.period - (time.time() - t0)
                if dt > 0:
                    self.stop_evt.wait(dt)

    def halt(self):
        self.stop_evt.set()
        self.join(timeout=30)


def clock_stats(rows):
    def vals(key, pred=None):
        out = []
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            if pred is not None and not pred(r):
                continue
            out.append(float(v))
        return out

    sm = vals("clocks_sm")
    loaded = vals("clocks_sm", lambda r: (r.get("util_gpu") or 0) >= 50)
    idle = vals("clocks_sm", lambda r: (r.get("util_gpu") or 0) < 10)
    pw = vals("power_w")
    tp = vals("temp_c")
    ut = vals("util_gpu")
    if not sm:
        return {"n_samples": 0, "instrument": "no_data"}
    return {
        "n_samples": len(rows),
        "n_with_sm": len(sm),
        "n_loaded": len(loaded),
        "n_idle": len(idle),
        "ambient_sm_median": statistics.median(idle) if idle else None,
        "max_sm": max(sm),
        "median_sm": statistics.median(sm),
        "median_loaded_sm": statistics.median(loaded) if loaded else None,
        "p90_loaded_sm": (sorted(loaded)[int(0.9 * (len(loaded) - 1))] if loaded else None),
        "max_power_w": max(pw) if pw else None,
        "median_power_w": statistics.median(pw) if pw else None,
        "max_temp_c": max(tp) if tp else None,
        "max_util_gpu": max(ut) if ut else None,
        "pstate_set": sorted({str(r.get("pstate")) for r in rows if r.get("pstate")}),
    }


def clock_verdict(st, base=BASE_SM_MHZ):
    if st.get("instrument") == "no_data" or st.get("n_loaded", 0) < 10:
        return "INSTRUMENT_UNKNOWN"
    if st["max_sm"] > base:
        return "CLOCK_LIFTED"
    return "CLOCK_PINNED"


def rate_windows(log_path):
    """s/100 windows from a TrainMonitor log.jsonl (deduped on_step points)."""
    if not log_path.exists():
        return {}
    seen = {}
    for ln in log_path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("event") == "on_step" and d.get("t") is not None:
            seen.setdefault(int(d["step"]), float(d["t"]))
    pts = sorted(seen.items())
    out = {"points": len(pts), "last_step": pts[-1][0] if pts else None,
           "last_t_s": pts[-1][1] if pts else None, "windows": {}}
    prev = pts[0] if pts else None
    for s, t in pts[1:]:
        if prev and s % 1000 == 0 and s > prev[0]:
            out["windows"][str(s)] = round((t - prev[1]) / (s - prev[0]) * 100, 3)
            prev = (s, t)
    return out


def write_timing(out_dir, payload):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "timing.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[TIMING]", json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return p


def cmd_burn(a):
    import torch
    smi = nvidia_smi()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Sampler(smi, out / "clock_samples.csv")
    s.start()
    time.sleep(3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[BURN] device", dev, flush=True)
    x = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    y = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    t0 = time.time()
    iters = 0
    while time.time() - t0 < a.seconds:
        for _ in range(20):
            z = torch.matmul(x, y)
            x = z.mul_(0.001).add_(0.5).to(torch.float16)
        torch.cuda.synchronize()
        iters += 20
    dur = time.time() - t0
    s.halt()
    st = clock_stats(s.rows)
    st["burn_seconds"] = round(dur, 1)
    st["burn_iters"] = iters
    st["sampler_errors"] = s.errors
    payload = {"experiment": "E-080", "stage": "burn", "verdict": clock_verdict(st),
               "base_sm_mhz": BASE_SM_MHZ, "clock": st}
    write_timing(out, payload)
    return 0


def cmd_train(a):
    smi = nvidia_smi()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    py = REPO / ".venv" / "Scripts" / "python.exe"
    cmd = [str(py), str(REPO / "diacritizer" / "scripts" / "train.py"),
           "--config", str(a.config)]
    print("[TRAIN]", " ".join(cmd), flush=True)
    s = Sampler(smi, out / "clock_samples.csv")
    s.start()
    t0 = time.time()
    with open(out / "probe_run.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    dur = time.time() - t0
    s.halt()
    st = clock_stats(s.rows)
    rw = rate_windows(Path(a.run_dir) / "monitor" / "log.jsonl") if a.run_dir else {}
    payload = {"experiment": "E-080", "stage": "train", "verdict": clock_verdict(st),
               "base_sm_mhz": BASE_SM_MHZ, "config": str(a.config),
               "run_dir": str(a.run_dir) if a.run_dir else None,
               "train_returncode": rc, "wall_seconds": round(dur, 1),
               "sampler_errors": s.errors, "clock": st, "rate": rw}
    write_timing(out, payload)
    return 0


def cmd_analyze(a):
    rows = []
    with open(a.csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (_num(v) if k != "ts" and k != "pstate" else v)
                         for k, v in r.items()})
    st = clock_stats(rows)
    rw = rate_windows(Path(a.run_dir) / "monitor" / "log.jsonl") if a.run_dir else {}
    payload = {"experiment": "E-080", "stage": "analyze", "csv": a.csv,
               "verdict": clock_verdict(st), "base_sm_mhz": BASE_SM_MHZ,
               "clock": st, "rate": rw}
    if a.out:
        write_timing(Path(a.out), payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description="E-080 clock/power-state probe")
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("burn")
    b.add_argument("--seconds", type=float, default=90.0)
    b.add_argument("--out", default="runs/diac/e80_gen")
    b.set_defaults(fn=cmd_burn)
    t = sub.add_parser("train")
    t.add_argument("--config", default="runs/diac/e80_gen/train_probe.yaml")
    t.add_argument("--out", default="runs/diac/e80_gen")
    t.add_argument("--run-dir", default="runs/diac/e80_train")
    t.set_defaults(fn=cmd_train)
    z = sub.add_parser("analyze")
    z.add_argument("--csv", required=True)
    z.add_argument("--run-dir", default=None)
    z.add_argument("--out", default=None)
    z.set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
