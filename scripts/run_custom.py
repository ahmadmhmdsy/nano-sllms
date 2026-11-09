"""One-command custom pipeline (Milestone G2): sanity -> prepare -> tokenize
-> train -> eval, stop-on-fail.

Chains the existing repo scripts in order against ONE config; the first
non-zero exit stops the chain so no half-built state is ever silently
consumed. The train (and eval) steps obey the never-co-run rule: they
abort if the GPU already has a compute process unless --allow_gpu_share
is passed explicitly.

Individual steps are skippable for resume flows (--from tokenize); the
train step itself relies on the auto-resume contract (re-run = resume,
zero flags).

Report: runs/<phase>/custom_chain.json + console summary.

Run:  .venv/Scripts/python scripts/run_custom.py --config configs/custom_example.yaml --dry-run
      .venv/Scripts/python scripts/run_custom.py --config configs/custom_example.yaml
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["sanity_check", "prepare_data", "tokenize_data", "train", "eval"]
GPU_STEPS = {"train", "eval"}
# U10: fine-tune chain (Train tab "SFT from checkpoint"). sft_data is CPU +
# network only (co-run safe); sft obeys the never-co-run rule like train.
SFT_STEPS = ["sft_data", "sft"]
SFT_GPU_STEPS = {"sft"}


def _pid_alive(pid):
    """False only when no live process owns the pid.

    Windows WDDM can keep a just-exited CUDA process listed in
    query-compute-apps for a while; such a stale entry can never train.
    Conservative on any doubt: probe errors / access-denied count as alive
    so the never-co-run guard is never weakened.
    """
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if h:
            k32.CloseHandle(h)
            return True
        return ctypes.get_last_error() != 87  # 87 = no such process
    except Exception:  # noqa: BLE001
        return True


def gpu_compute_pids():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        # Windows WDDM quirk: query-compute-apps lists EVERY process with a
        # GPU context (desktop apps, WebView...). Only python compute
        # processes can be a co-running trainer - filter to those, and drop
        # pids whose process is already gone (stale WDDM entries).
        hits = []
        for line in out.splitlines():
            line = line.strip()
            if line and "," in line and "python" in line.partition(",")[2].lower():
                if _pid_alive(line.partition(",")[0]):
                    hits.append(line)
        return hits
    except Exception:  # noqa: BLE001 - never block the chain on a probe error
        return []


def _ancestor_pids():
    r"""Pids of every ancestor of this process (empty on probe errors).

    .venv\Scripts\python.exe on Windows is a launcher that spawns the real
    base interpreter as a child and waits for it, so this chain process's
    getppid() is the chain's OWN launcher shim - NOT the webui server (or
    probe) that launched the whole thing. The server, which may hold a
    lingering CUDA context after a chat unload, sits one launcher level
    higher. Walk the full ancestor chain via the Toolhelp32 snapshot so
    every shim between here and the chain launcher is covered: an ancestor
    can never be a co-running trainer (it spawned this chain and waits).
    """
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _PE32W(ctypes.Structure):
            # Must byte-match PROCESSENTRY32W exactly (the API validates
            # dwSize): size_t for the heap id, 4-byte LONG for the priority.
            _fields_ = [("dwSize", wintypes.DWORD),
                        ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", wintypes.LONG),
                        ("dwFlags", wintypes.DWORD),
                        ("szExeFile", wintypes.WCHAR * 260)]

        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k32.Process32FirstW.restype = wintypes.BOOL
        k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PE32W)]
        k32.Process32NextW.restype = wintypes.BOOL
        k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PE32W)]

        snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if not snap or snap == wintypes.HANDLE(-1).value:
            return set()
        parent_of, entry = {}, _PE32W()
        entry.dwSize = ctypes.sizeof(_PE32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            parent_of[entry.th32ProcessID] = entry.th32ParentProcessID
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
        k32.CloseHandle(snap)
        anc, pid, hops = set(), os.getpid(), 0
        while pid in parent_of and hops < 32:
            pid = parent_of[pid]
            if not pid:
                break
            anc.add(pid)
            hops += 1
        return anc
    except Exception:  # noqa: BLE001 - degrade to own/parent exclusion
        return set()


def gpu_busy_others():
    """GPU python compute pids that are NOT this chain or its ancestors.

    The chain cannot be training when its own train step starts, and an
    ancestor (webui server / probe / shell) only ever holds a lingering
    CUDA context from a chat unload - neither is a co-running trainer.
    Anything else still aborts the chain (never-co-run rule).
    """
    own = {os.getpid(), os.getppid()} | _ancestor_pids()
    return [h for h in gpu_compute_pids()
            if not h.startswith(tuple(f"{p}," for p in own))]


def main() -> None:
    ap = argparse.ArgumentParser(description="one-command custom pipeline")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--from", dest="from_step", default=STEPS[0],
                    choices=STEPS, help="first step to run (default all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned commands and exit")
    ap.add_argument("--allow_gpu_share", action="store_true",
                    help="permit train/eval while another GPU process runs")
    ap.add_argument("--sft", action="store_true",
                    help="fine-tune chain (sft_data -> sft) instead of the "
                         "pretrain pipeline")
    ap.add_argument("--pilot", action="store_true",
                    help="passed to sft.py: first sft.pilot_rows pairs, 1 epoch")
    args = ap.parse_args()

    import yaml
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    phase = cfg.get("name", cfg_path.stem)
    cfg_arg = str(cfg_path if cfg_path.is_absolute() else
                  Path("configs") / cfg_path.name)

    steps_all = SFT_STEPS if args.sft else STEPS
    gpu_steps = SFT_GPU_STEPS if args.sft else GPU_STEPS
    start_i = steps_all.index(args.from_step) \
        if args.from_step in steps_all else 0
    plan = [s for s in steps_all[start_i:]]
    if args.sft:
        sft_cmd = [sys.executable, str(ROOT / "scripts" / "sft.py"),
                   "--config", cfg_arg] + (["--pilot"] if args.pilot else [])
        commands = {"sft_data": [sys.executable,
                                 str(ROOT / "scripts" / "sft_data.py"),
                                 "--config", cfg_arg],
                    "sft": sft_cmd}
    else:
        commands = {s: [sys.executable, str(ROOT / "scripts" / f"{s}.py"),
                        "--config", cfg_arg] for s in plan}

    print(f"[run_custom] phase={phase} steps={plan} dry_run={args.dry_run}",
          flush=True)
    if args.dry_run:
        busy = gpu_compute_pids()
        print(f"[run_custom] gpu python compute processes now: "
              f"{busy or 'none'}", flush=True)
        for s in plan:
            print("  " + s + ": " + " ".join(commands[s]), flush=True)
        print("[run_custom] dry-run complete - nothing was executed",
              flush=True)
        return

    chain = {"config": cfg_arg, "phase": phase, "steps": [],
             "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out_dir = ROOT / "runs" / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in plan:
        if s in gpu_steps and not args.allow_gpu_share:
            busy = gpu_busy_others()
            if busy:
                msg = (f"GPU busy ({len(busy)} compute process/es: {busy}) - "
                       "never-co-run rule; pass --allow_gpu_share to override")
                print(f"[run_custom] ABORT before '{s}': {msg}", flush=True)
                chain["status"] = "aborted_gpu_busy"
                (out_dir / "custom_chain.json").write_text(
                    json.dumps(chain, indent=2), encoding="utf-8")
                sys.exit(3)
        t0 = time.time()
        print(f"[run_custom] === step {s} ===", flush=True)
        proc = subprocess.run(commands[s], cwd=str(ROOT))
        dt = round(time.time() - t0, 1)
        chain["steps"].append({"step": s, "exit": proc.returncode,
                               "seconds": dt})
        if proc.returncode != 0:
            chain["status"] = f"failed_at_{s}"
            (out_dir / "custom_chain.json").write_text(
                json.dumps(chain, indent=2), encoding="utf-8")
            print(f"[run_custom] step '{s}' FAILED (exit {proc.returncode}) "
                  f"after {dt}s - chain stopped, report: "
                  f"{out_dir / 'custom_chain.json'}", flush=True)
            sys.exit(proc.returncode)
        print(f"[run_custom] step '{s}' ok ({dt}s)", flush=True)
    chain["status"] = "completed"
    (out_dir / "custom_chain.json").write_text(json.dumps(chain, indent=2),
                                               encoding="utf-8")
    print(f"[run_custom] chain COMPLETED -> {out_dir / 'custom_chain.json'}",
          flush=True)


if __name__ == "__main__":
    main()
