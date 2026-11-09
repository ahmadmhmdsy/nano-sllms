"""E-081 clock-state watcher: 1 Hz GPU sampling with fast/base transition logging.

READ-ONLY ON THE GPU. This tool only issues nvidia-smi queries; it never launches a
GPU kernel. It is therefore NOT a co-run under rules/gpu.md and is safe to run
alongside the single protected training job.

It also tails the trainer's monitor/log.jsonl so every clock sample carries the
exact step and run-seconds it belongs to, which makes clock-vs-rate correlation
exact instead of approximate.

Usage:
  python e81_clock_watch.py sample  --out runs/diac/e81_gen/clock_samples.csv [--seconds 18000] [--monitor runs/diac/e79_a_26l64/monitor/log.jsonl]
  python e81_clock_watch.py analyze --csv runs/diac/e81_gen/clock_samples.csv
"""
import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

BASE_SM_MHZ = 945.0
FAST_SM_MHZ = 1200.0

QUERY = [
    "nvidia-smi",
    "--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used",
    "--format=csv,noheader,nounits",
]
FIELDS = ["ts", "epoch", "pstate", "clocks_sm", "clocks_mem", "power_w", "temp_c",
          "util_gpu", "util_mem", "mem_used_mib", "step", "run_t"]


def read_gpu():
    out = subprocess.run(QUERY, capture_output=True, text=True, timeout=15)
    line = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    row = {"ts": parts[0], "pstate": parts[1]}
    for k, v in zip(FIELDS[3:], parts[2:]):
        try:
            row[k] = float(v)
        except ValueError:
            row[k] = ""
    return row


def state_of(sm):
    if sm in ("", None):
        return "unknown"
    if sm <= BASE_SM_MHZ + 30:
        return "base"
    if sm >= FAST_SM_MHZ:
        return "fast"
    return "mid"


class LogTail:
    """Incremental reader for the trainer monitor log; yields the latest step/run_t."""

    def __init__(self, path):
        self.path = Path(path)
        self.off = 0
        self.step = None
        self.run_t = None

    def poll(self):
        try:
            with self.path.open("r", encoding="utf-8") as f:
                f.seek(self.off)
                data = f.read()
                self.off = f.tell()
        except FileNotFoundError:
            return
        for ln in data.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("event") == "on_step":
                self.step = e.get("step")
                self.run_t = e.get("t")


def cmd_sample(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    f = out.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    tail = LogTail(args.monitor) if args.monitor else None
    print("[watch] interval " + str(args.interval) + "s -> " + str(out), flush=True)
    if tail:
        print("[watch] tailing " + str(tail.path), flush=True)
    t0 = time.time()
    cur = None
    while True:
        if args.seconds and time.time() - t0 >= args.seconds:
            break
        try:
            row = read_gpu()
        except Exception as e:
            print("[watch] read failed: " + repr(e), flush=True)
            time.sleep(args.interval)
            continue
        row["epoch"] = round(time.time(), 3)
        if tail:
            tail.poll()
            row["step"] = tail.step if tail.step is not None else ""
            row["run_t"] = tail.run_t if tail.run_t is not None else ""
        w.writerow(row)
        f.flush()
        st = state_of(row["clocks_sm"])
        if st != cur:
            print("[watch] " + str(row["ts"]) + "  STATE " + str(cur) + " -> " + st
                  + "  sm=" + str(row["clocks_sm"]) + "  pw=" + str(row["power_w"])
                  + "  temp=" + str(row["temp_c"]) + "  util=" + str(row["util_gpu"])
                  + "  step=" + str(row.get("step")), flush=True)
            cur = st
        time.sleep(args.interval)
    f.close()
    print("[watch] done", flush=True)


def cmd_analyze(args):
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))

    def num(v):
        try:
            return float(v)
        except Exception:
            return None

    pts = []
    for r in rows:
        sm, st = num(r.get("clocks_sm")), num(r.get("step"))
        if sm is None or st is None:
            continue
        pts.append((st, sm, num(r.get("power_w")), num(r.get("temp_c")), num(r.get("util_gpu"))))
    print("total samples " + str(len(rows)) + ", aligned to a step " + str(len(pts)))
    if not pts:
        print("no aligned samples yet - the monitor log may not have been tailed")
        return
    buckets = {}
    for st, sm, pw, tp, ut in pts:
        buckets.setdefault(int(st // 1000) * 1000, []).append((sm, pw, tp, ut))
    print("")
    print("window   n   sm_med  sm_max  pw_med  temp_max  util_med")
    for b in sorted(buckets):
        v = buckets[b]
        sms = [x[0] for x in v]
        pws = [x[1] for x in v if x[1] is not None]
        tps = [x[2] for x in v if x[2] is not None]
        uts = [x[3] for x in v if x[3] is not None]
        print(format(b, "6d"), format(len(v), "4d"), format(statistics.median(sms), "7.0f"),
              format(max(sms), "7.0f"),
              format(statistics.median(pws) if pws else float("nan"), "7.1f"),
              format(max(tps) if tps else float("nan"), "8.0f"),
              format(statistics.median(uts) if uts else float("nan"), "8.0f"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--seconds", type=int, default=0)
    s.add_argument("--interval", type=float, default=1.0)
    s.add_argument("--monitor", default="")
    s.set_defaults(func=cmd_sample)
    a = sub.add_parser("analyze")
    a.add_argument("--csv", required=True)
    a.set_defaults(func=cmd_analyze)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
