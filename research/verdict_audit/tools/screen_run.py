"""E-084 triage bed - single-arm few-minute SCREEN driver.

*** THIS IS A SCREEN, NOT A FULL TRAINING RUN. ***

Runs ONE arm at a tiny fixed step budget and (optionally) scores it against a
same-seed control screen. It is the thing that decides "is this arm worth a real
run", never "is this arm better".

Why the pairing matters: before E-084 the trainer seeded its data order
(default_rng(777)) but NOT its weight init, so every arm carried an unmeasured
~0.82 percent init draw and two arms were never comparable. This driver passes
--seed, so the control and the arm share init and data order and init drift
cancels; the only remaining difference is the arm's own mechanism.

Guarantees the driver enforces:
  * the exact command is written to launch.json BEFORE it runs (launch gate 2);
  * the measured GPU gate note is recorded with it (launch gate 4);
  * a run dir holding checkpoints is REFUSED - train.py auto-resumes, so a
    silent second "fresh" screen would really be a continuation;
  * every artifact carries the SCREEN-NOT-FULL-TRAIN stamp.
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
STAMP = "SCREEN-NOT-FULL-TRAIN"
SMOKE_BYPASS = ("E-084 triage bed: this run IS a smoke - tiny fixed step budget, "
                "checkpoints land and loss advances, and the 60-step unseeded "
                "pre-smoke would double the GPU cost of a screen")
KEEP = ("model", "optimizer", "lr", "weight_decay", "batch_size", "accum",
        "tokens_phase", "pretrained_init", "loss", "muon_lr", "seed")


def coerce(v):
    lv = v.lower()
    if lv in ("true", "false"):
        return lv == "true"
    if lv in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def set_dotted(cfg, key, value):
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise SystemExit("[SCREEN-FATAL] cannot descend into " + key)
    node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--config", required=True, help="base arm config (shape + data source)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--init", default=None, help="warm-start checkpoint for BOTH arm and control")
    ap.add_argument("--drop-init", action="store_true", help="force from-scratch")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="0 (default) = NEVER run the val split. pipeline.evals."
                         "evaluate_unseen scans the WHOLE val set (~160k windows), "
                         "about 190 s per eval - more than the screen itself. The "
                         "selector DER trajectory is the diagnostic instead.")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--anchor", default=None,
                    help="also score this checkpoint as step 0 (no training) - the "
                         "pre-screen reference point, so a control trajectory can be "
                         "read against the model it started from")
    ap.add_argument("--baseline", default=None,
                    help="explicit control screen JSON path (same seed+steps). "
                         "Validated to exist BEFORE the GPU time is spent.")
    ap.add_argument("--baseline-arm", default=None,
                    help="SAFER: the control ARM name; the path is derived as "
                         "screen_<arm>_s<seed>.screen.json, which is exactly what "
                         "screen_score writes. E-084 r1 passed a hand-built "
                         "'<arm>_s1.screen.json' that never existed, so both paired "
                         "arms trained on the GPU and then died in the scorer.")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--diag-limit", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gate-note", default="", help="measured GPU gate numbers, recorded")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit("[SCREEN-FATAL] --set needs K=V, got " + kv)
        k, v = kv.split("=", 1)
        set_dotted(base, k, coerce(v))
    if args.init:
        base["pretrained_init"] = args.init
    if args.drop_init:
        base.pop("pretrained_init", None)

    py = sys.executable
    train_py = str(REPO / "diacritizer" / "scripts" / "train.py")
    score_py = str(REPO / "research" / "verdict_audit" / "tools" / "screen_score.py")
    cfg_dir = REPO / "configs" / "screen"
    out_dir = REPO / "runs" / "diac" / "_screen_out"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in args.seeds:
        cfg = copy.deepcopy(base)
        phase = "screen_%s_s%d" % (args.arm, seed)
        cfg["phase"] = phase
        cfg["total_steps"] = args.steps
        cfg["save_every"] = args.save_every
        # 0 means NEVER: a value above total_steps means the modulo never hits.
        cfg["eval_every"] = (args.eval_every if args.eval_every > 0
                             else args.steps * 10 + 7)
        cfg["log_every"] = max(10, min(50, args.steps // 4))
        cfg["early_stop_patience"] = 99
        cfg.pop("gate_eval_every", None)
        cfg["seed"] = seed
        cfgp = cfg_dir / (phase + ".yaml")
        cfgp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=True),
                        encoding="utf-8")
        rundir = REPO / "runs" / "diac" / phase
        stale = sorted(rundir.glob("checkpoint-*")) if rundir.exists() else []
        if stale:
            raise SystemExit(
                "[SCREEN-FATAL] %s already holds %d checkpoint(s) -> train.py would "
                "AUTO-RESUME instead of starting a fresh screen. Pick another arm "
                "name or seed." % (rundir, len(stale)))
        out_json = out_dir / ("%s.screen.json" % phase)
        train_cmd = [py, train_py, "--config", str(cfgp), "--seed", str(seed),
                     "--skip-smoke", SMOKE_BYPASS]
        score_cmd = [py, score_py, "--rundir", str(rundir), "--config", str(cfgp),
                     "--out", str(out_json), "--arm", args.arm, "--seed", str(seed),
                     "--stride", str(args.stride), "--limit", str(args.limit),
                     "--diag-limit", str(args.diag_limit), "--batch", str(args.batch)]
        if args.anchor:
            score_cmd += ["--model", args.anchor]
        baseline_path = args.baseline
        if args.baseline_arm:
            baseline_path = str(out_dir / ("screen_%s_s%d.screen.json"
                                           % (args.baseline_arm, seed)))
        if baseline_path and not Path(baseline_path).exists():
            raise SystemExit(
                "[SCREEN-FATAL] baseline %s does not exist; refusing to launch a "
                "screen whose paired comparison would crash only AFTER the GPU "
                "time was spent. screen_score writes screen_<arm>_s<seed>"
                ".screen.json." % baseline_path)
        if baseline_path:
            score_cmd += ["--baseline", baseline_path]
        jobs.append({"phase": phase, "seed": seed, "config": str(cfgp),
                     "rundir": str(rundir), "out_json": str(out_json),
                     "train_cmd": train_cmd, "score_cmd": score_cmd})

    launch = {"status": STAMP, "arm": args.arm, "steps": args.steps,
              "base_config": str(args.config),
              "init": base.get("pretrained_init"),
              "overrides": args.set,
              "gate_note": args.gate_note,
              "baseline": args.baseline or args.baseline_arm,
              "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "jobs": jobs,
              "disclaimer": ("Few-minute screen, NOT a full training run. Cannot rank "
                             "sub-2-percent effects; anything inside the band is "
                             "INCONCLUSIVE.")}
    (out_dir / ("%s.launch.json" % args.arm)).write_text(
        json.dumps(launch, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"launch_record":
                      str(out_dir / ("%s.launch.json" % args.arm)),
                      "jobs": len(jobs)}, ensure_ascii=False), flush=True)
    for j in jobs:
        print("TRAIN: " + " ".join(j["train_cmd"]), flush=True)
    if args.dry_run:
        print("[DRY-RUN] wrote configs + launch record only", flush=True)
        return

    for j in jobs:
        logp = out_dir / ("%s.train.log" % j["phase"])
        t0 = time.time()
        with logp.open("w", encoding="utf-8", errors="backslashreplace") as fh:
            fh.write("# %s\n# %s\n\n" % (STAMP, " ".join(j["train_cmd"])))
            fh.flush()
            r = subprocess.run(j["train_cmd"], stdout=fh,
                               stderr=subprocess.STDOUT, text=True)
        j["train_seconds"] = round(time.time() - t0, 1)
        j["train_exit"] = r.returncode
        print({"phase": j["phase"], "train_exit": r.returncode,
               "train_seconds": j["train_seconds"], "log": str(logp)}, flush=True)
        if r.returncode != 0:
            print("[SCREEN-TRAIN-FAIL] %s nonzero exit; scoring skipped"
                  % j["phase"], flush=True)
            launch["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            launch["failed_phase"] = j["phase"]
            (out_dir / ("%s.launch.json" % args.arm)).write_text(
                json.dumps(launch, indent=1, ensure_ascii=False), encoding="utf-8")
            raise SystemExit("[SCREEN-FATAL] train exit %d for %s; see %s"
                             % (r.returncode, j["phase"], logp))
        logp2 = out_dir / ("%s.score.log" % j["phase"])
        t1 = time.time()
        with logp2.open("w", encoding="utf-8", errors="backslashreplace") as fh:
            fh.write("# %s\n# %s\n\n" % (STAMP, " ".join(j["score_cmd"])))
            fh.flush()
            r2 = subprocess.run(j["score_cmd"], stdout=fh,
                                stderr=subprocess.STDOUT, text=True)
        j["score_seconds"] = round(time.time() - t1, 1)
        j["score_exit"] = r2.returncode
        print({"phase": j["phase"], "score_exit": r2.returncode,
               "score_seconds": j["score_seconds"], "log": str(logp2)}, flush=True)
        # A nonzero scorer exit used to be printed and then IGNORED: the r1
        # calibration reported 'done' for two arms whose verdict never existed.
        # A screen with no verdict is a failure, not a result.
        if r2.returncode != 0:
            launch["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            launch["failed_phase"] = j["phase"]
            (out_dir / ("%s.launch.json" % args.arm)).write_text(
                json.dumps(launch, indent=1, ensure_ascii=False), encoding="utf-8")
            raise SystemExit("[SCREEN-FATAL] scorer exit %d for %s; see %s"
                             % (r2.returncode, j["phase"], logp2))
        if Path(j["out_json"]).exists():
            doc = json.loads(Path(j["out_json"]).read_text(encoding="utf-8"))
            print({"PHASE": j["phase"], "status": doc["status"],
                   "verdict": doc["verdict"]["label"],
                   "selector": [p["selector"]["der_registered"] for p in doc["points"]]},
                  flush=True)

    launch["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / ("%s.launch.json" % args.arm)).write_text(
        json.dumps(launch, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"done": args.arm, "jobs": len(jobs)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
