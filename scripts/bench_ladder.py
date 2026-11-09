"""Stepped (ladder) playground battery: climb from the 71-task core battery toward 1,056 tasks.

Each level is one scratch/bench_hf.py run on the dev split. An arm climbs to the next level only
while its strict pass rate on the current level is >= --gate, so a weak checkpoint costs ~71-88
tasks instead of 1,056. Larger levels reuse the same 88 sub-skills with fresh entry points,
descriptions and generated cases (per-rung instances), so L2-L4 measure ROBUSTNESS. B1/W2/W4 are
PLAN.md section 11 stage SA-B: 192 NEW sub-skills (breadth ladders + a fix-from-traceback tier + a
strict tool-call tier) that live outside the original 88, so L0-L4 stay byte-identical.

  L0  core x1   =   71   (the historical battery, comparable with every earlier result)
  L1  all  x1   =   88   (+ loops and attributes tiers)
  L2  all  x3   =  264
  L3  all  x6   =  528
  L4  all  x12  = 1056
  B1  breadth x1 =  192   (new sub-skills only: 150 write + 22 fix + 20 tool)
  W2  wide x2   =  560   (88 + 192 = 280 sub-skills)
  W4  wide x4   = 1120   (the 1,000+ task / 200+ sub-skill battery)

Usage:
  & .\\.venv\\Scripts\\python.exe scripts\\bench_ladder.py --arm cptlora=hf:runs/sft_smollm2_cpt_pg/final `
      --device cuda --up-to L4 --out-dir research/raw/ladder_cptlora
  & .\\.venv\\Scripts\\python.exe scripts\\bench_ladder.py --dry-run     # task counts only, no model
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scratch" / "bench_hf.py"
NEST = ROOT / "glm-5.3-flash-from-scratch"
LADDER = [("L0", "core", 1), ("L1", "all", 1), ("L2", "all", 3), ("L3", "all", 6), ("L4", "all", 12),
          ("B1", "breadth", 1), ("W2", "wide", 2), ("W4", "wide", 4)]
NAMES = [name for name, _, _ in LADDER]


def level_size(tier: str, per_rung: int) -> int:
    sys.path.insert(0, str(NEST))
    from glm53_flash.playground import playground_pool
    return len(playground_pool(tier)) * per_rung


def run_level(name, tier, per_rung, arms, args, out_dir: Path) -> dict:
    out = out_dir / f"{name}_{tier}_x{per_rung}.json"
    if not (args.reuse and out.is_file()):
        cmd = [sys.executable, str(BENCH), "--split", args.split, "--tier", tier,
               "--per-rung", str(per_rung), "--budget-bytes", str(args.budget_bytes),
               "--max-tokens", str(args.max_tokens), "--batch-size", str(args.batch_size),
               "--device", args.device, "--out", str(out)]
        for arm in arms:
            cmd += ["--arm", arm]
        log = out.with_suffix(".log")
        print(f"[{name}] {tier} x{per_rung}: {len(arms)} arm(s) -> {out}", flush=True)
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not out.is_file():  # an exit code is not a result: require the report
            raise SystemExit(f"[{name}] bench failed (rc={rc}); see {log}")
    return json.loads(out.read_text(encoding="utf-8"))["arms"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], help="NAME=hf:PATH, as scratch/bench_hf.py")
    ap.add_argument("--up-to", choices=NAMES, default="L4")
    ap.add_argument("--start", choices=NAMES, default="L0")
    ap.add_argument("--gate", type=float, default=0.90, help="strict pass rate needed to climb")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--budget-bytes", type=int, default=160)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)  # every baseline ladder ran at 32; ~6x faster than 4
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=Path("research/raw/ladder"))
    ap.add_argument("--reuse", action="store_true", help="reuse a level report already on disk")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 0.0 <= args.gate <= 1.0:
        raise SystemExit("--gate must be in [0, 1]")
    levels = LADDER[NAMES.index(args.start):NAMES.index(args.up_to) + 1]
    if args.dry_run:
        for name, tier, per_rung in levels:
            print(f"{name}  {tier:<7} x{per_rung:<3} = {level_size(tier, per_rung):>5} tasks")
        return 0
    if not args.arm:
        raise SystemExit("at least one --arm is required (or --dry-run)")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    active = {spec.split("=", 1)[0]: spec for spec in args.arm}
    summary = {"gate": args.gate, "split": args.split, "levels": {}}
    for name, tier, per_rung in levels:
        if not active:
            break
        arms = run_level(name, tier, per_rung, list(active.values()), args, out_dir)
        row = {}
        missing = sorted(set(active) - set(arms))
        if missing:  # a --reuse report built with other arms
            raise SystemExit(f"[{name}] report has no arm(s) {missing}; drop --reuse or fix --arm")
        for arm_name in list(active):
            rep = arms[arm_name]
            rate = rep["solved_strict"] / rep["tasks"]
            row[arm_name] = {"strict": rep["solved_strict"], "tasks": rep["tasks"],
                             "rate": round(rate, 4), "climbs": rate >= args.gate}
            if rate < args.gate:
                del active[arm_name]
        summary["levels"][name] = {"tier": tier, "per_rung": per_rung, "arms": row}
        print(f"[{name}] " + json.dumps(row), flush=True)
    (out_dir / "ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
