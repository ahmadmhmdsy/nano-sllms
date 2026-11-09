"""scripts/profile_resources.py — GPU/CPU/RAM/disk sampler for a training run.

Run IN PARALLEL with scripts/train.py (or any phase train job); READ-ONLY:
never touches runs/ or the training process. Purpose: during a smoke or
production run, find WHERE the GPU starves — dataloader disk stalls show as
GPU-idle + CPU-idle + high disk-read; CPU-bound preprocessing shows GPU-idle
+ CPU-high; VRAM near cap at eval/save windows shows OOM-pressure risk.

Usage (venv-only):
  & .\.venv\Scripts\python.exe scripts\profile_resources.py \
      --out research/profiles/g2.csv --interval 2 --minutes 0
minutes=0 runs until killed (background job). CSV is flushed every sample, so
partial data is always usable; a rolling MD summary is rewritten every 60
samples and on exit.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import psutil


def gpu_sample():
    """via nvidia-smi CLI (present on this box; pynvml NOT installed)."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vals = [float(x.strip()) for x in out.stdout.splitlines()[0].split(",")]
        return vals[0], vals[1], vals[2]
    except Exception:
        return (float("nan"), float("nan"), float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="0 = run until killed")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.minutes * 60 if args.minutes > 0 else None
    last_disk = psutil.disk_io_counters()
    last_t = time.time()
    rows = []
    with out.open("w", encoding="utf-8", newline="") as csvf:
        w = csv.writer(csvf)
        w.writerow(["time", "gpu_util", "gpu_vram_mb", "gpu_temp",
                    "cpu_pct", "ram_pct", "disk_read_mb_s", "disk_write_mb_s"])
        while deadline is None or time.time() < deadline:
            time.sleep(args.interval)
            now = time.time()
            dt = max(now - last_t, 1e-6)
            ioc = psutil.disk_io_counters()
            r_bs = (ioc.read_bytes - last_disk.read_bytes) / dt / 1e6
            w_bs = (ioc.write_bytes - last_disk.write_bytes) / dt / 1e6
            last_disk, last_t = ioc, now
            gu, gvm, gt = gpu_sample()
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            w.writerow([time.strftime("%H:%M:%S"), f"{gu:.0f}", gvm, gt,
                        f"{cpu:.0f}", f"{ram:.0f}", f"{r_bs:.1f}", f"{w_bs:.1f}"])
            csvf.flush()
            rows.append((gu, cpu, r_bs + w_bs))
            if len(rows) >= 60:
                _emit_summary(args, rows)
                rows = rows[-30:]
        _emit_summary(args, rows)
    print(f"[profile] stopped; {out}")


def _emit_summary(args, rows) -> None:
    if not rows:
        return
    gu = [max(0.0, r[0]) for r in rows]
    cpu = [r[1] for r in rows]
    disk = [r[2] for r in rows]
    n = len(rows)
    idle_g = sum(1 for g in gu if g < 10) / n * 100
    lines = [
        f"- samples {n} (interval {args.interval}s, recent window)",
        f"- GPU util mean {sum(gu) / n:.0f}% peak {max(gu):.0f}% idle<10%: {idle_g:.0f}%",
        f"- CPU mean {sum(cpu) / n:.0f}% peak {max(cpu):.0f}%",
        f"- disk mean {sum(disk) / n:.0f} MB/s peak {max(disk):.0f} MB/s",
    ]
    gm = sum(gu) / n
    cpu_busy = sum(1 for c in cpu if c > 50) / n
    disk_busy = sum(1 for d in disk if d > 50) / n
    if gm >= 85:
        verdict = "GPU-BOUND (healthy)"
    elif gm < 60 and cpu_busy > 0.4:
        verdict = "CPU-BOUND (GPU starving; reduce dataloader python work)"
    elif gm < 60 and disk_busy > 0.3:
        verdict = "DATA-STARVED (GPU+CPU idle, disk active; prefetch in RAM)"
    elif gm < 60:
        verdict = "IDLE-ish (pacing, accum, eval/save windows)"
    else:
        verdict = "mixed"
    lines.append(f"- VERDICT: {verdict}")
    tgt = (Path(args.summary) if args.summary
           else Path(args.out).with_suffix(".summary.md"))
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_text("\n".join(["# resource profile — " +
                               time.strftime("%F %T")] + lines + [""]),
                   encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
