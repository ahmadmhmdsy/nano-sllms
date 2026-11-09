"""Smoke preflight (MEMORY 2026-09-23 rule R4, plan S3).

preflight_smoke() runs a 60-step end-to-end smoke of the SAME config (identical
except: phase suffixed "_smoke", tiny step budget) BEFORE the real train run,
then deletes the smoke run dir on PASS. Skipped automatically when:
- the phase is itself a smoke ("smoke" in phase), or
- the real run dir already holds a checkpoint (auto-resume of a started run).
Bypass flag: --skip-smoke (reason required in the flag value; recorded in log).
"""
import json
import subprocess
import sys
import shutil
from pathlib import Path


def preflight_smoke(args, cfg, run_dir, repo):
    phase = cfg["phase"]
    if "smoke" in phase:
        print("[SMOKE] smoke phase, no re-smoke", flush=True)
        return "skipped-self"
    ck = sorted((Path(run_dir)).glob("checkpoint-*")) if Path(run_dir).exists() else []
    if ck:
        print("[SMOKE] resume of started run, no pre-smoke", flush=True)
        return "skipped-resume"
    if getattr(args, "skip_smoke", None):
        print("[SMOKE-BYPASS] reason:", args.skip_smoke, flush=True)
        return f"bypassed: {args.skip_smoke}"
    smoke_cfg = dict(cfg)
    smoke_cfg["phase"] = phase + "_smoke"
    smoke_cfg["total_steps"] = 60
    smoke_cfg["eval_every"] = 30
    smoke_cfg["save_every"] = 60
    smoke_cfg["log_every"] = 20
    smoke_cfg["early_stop_patience"] = 100
    smoke_cfg.pop("gate_eval_every", None)
    smoke_path = repo / "configs" / (phase + "_smoke_auto.yaml")
    import yaml
    smoke_path.write_text(yaml.safe_dump(smoke_cfg, allow_unicode=True),
                          encoding="utf-8")
    smoke_run = repo / "runs" / "diac" / (phase + "_smoke")
    print("[SMOKE] running preflight 60-step smoke:", smoke_path, flush=True)
    try:
        r = subprocess.run([sys.executable,
                            str(repo / "diacritizer" / "scripts" / "train.py"),
                            "--config", str(smoke_path)],
                           capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "SMOKE_FAILED.log").write_text(
                (r.stdout or "")[-8000:] + "\n=== STDERR ===\n" + (r.stderr or "")[-8000:],
                encoding="utf-8")
            print("[SMOKE-FAIL] NONZERO EXIT -> refusing to start the real run.",
                  "Log: SMOKE_FAILED.log", flush=True)
            sys.exit(3)
        passes = "[MONITOR-END]" in (r.stdout or "")
        if not passes:
            print("[SMOKE-FAIL] no completion marker in smoke output", flush=True)
            sys.exit(3)
        print("[SMOKE] PASS -> cleaning smoke dir, starting real run", flush=True)
    finally:
        try:
            if smoke_run.exists():
                shutil.rmtree(smoke_run)
        except OSError:
            pass
    return "pass"
