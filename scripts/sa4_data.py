"""SA-4 data: agent-format rows (fix, repair loop, strict tool calls) plus replay, written as JSONL for sa3_data.py.

Shapes in data/sa4/{train,val}.jsonl:
  fix     SA-2 fix rows unchanged: the buggy body is shown commented out with its real sandbox outcome.
  repair  the other half of the SA-2 fix rows, re-rendered as a retry turn: the attempt is shown as live
          code, then its real outcome (the SA-2 witness run, a Traceback or "returned X but expected Y"),
          then REPAIR_ASK, then the header. This is the SA-7 retry-with-traceback prompt (PLAN.md SA-C).
  tool    SA-2 xLAM rows unchanged (one JSON list of {"name", "arguments"} calls).
  replay  playground reference solutions on the held-out "pretrain" split (SA-3 lesson: SFT without replay
          dropped ladder B1 53 -> 38). Only write-kind sub-skills: the fix and tool ladder tiers stay unseen
          by skill, so the SA-4 fix/format gates are not measured on replayed skills.
  he/mbpp SA-2 instruction rows, a small sample against HumanEval/MBPP forgetting.

Fix rows are split between fix and repair by a hash of the row id, so no buggy example appears in both.
Every completion was already executed against its tests by sa2_build (fix/repair/he/mbpp) or by the
playground's own reference execution (replay); this script only re-renders and samples.

  .venv/Scripts/python.exe -I scripts/sa4_data.py
  .venv/Scripts/python.exe -I scripts/sa3_data.py --src data/sa4 --shapes fix,repair,tool,replay,he,mbpp --out data/sa4/ds
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "glm-5.3-flash-from-scratch"))

from glm53_flash.playground import make_playground_task, playground_pool  # noqa: E402

REPAIR_HEAD = "# Complete this Python function."
REPAIR_ASK = "# That attempt is wrong. Write the fixed function."


def bucket(row_id: str) -> float:
    return int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def to_repair(row: dict) -> dict:
    """Re-render one SA-2 fix prompt as attempt -> real outcome -> retry; the completion is unchanged."""
    lines = row["prompt"].split("\n")
    start = lines.index("# Buggy version:") + 1
    call = next(i for i in range(start, len(lines)) if lines[i].startswith("# >>> "))
    task = next(i for i in range(call, len(lines)) if lines[i].startswith("# Task: "))
    header = lines[-1]
    # The attempt keeps its own header line: some mutations land in a default argument.
    if task != len(lines) - 2 or not header.startswith("def ") or not lines[start].startswith("# def "):
        raise ValueError(f"{row['id']}: unexpected fix prompt layout")
    attempt = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines[start:call]]
    prompt = "\n".join([REPAIR_HEAD, lines[task], *attempt, *lines[call:task], REPAIR_ASK, header])
    return {**row, "id": row["id"].replace("/fix", "/repair"), "shape": "repair", "prompt": prompt}


def replay_rows(first: int, count: int) -> list[dict]:
    rows = []
    for skill in playground_pool("wide"):
        if skill.kind != "write":
            continue
        for index in range(first, first + count):
            task = make_playground_task(skill, split="pretrain", index=index)
            rows.append({"id": f"playground/{task.task_id}", "source": "playground", "shape": "replay",
                         "entry_point": task.entry_point, "prompt": task.prompt,
                         "completion": task.reference_completion, "tests": []})
    return rows


def build(src: Path, split: str, args, rng: random.Random) -> tuple[list[dict], collections.Counter]:
    rows = [json.loads(line) for line in (src / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    out, plain = [], collections.defaultdict(list)
    for row in rows:
        if row["shape"] == "fix":
            out.append(to_repair(row) if bucket(row["id"]) < args.repair_frac else row)
        elif row["shape"] == "tool":
            out.append(row)
        elif row["shape"] in ("he", "mbpp"):
            plain[row["shape"]].append(row)
    for shape in sorted(plain):
        pool = plain[shape]
        out += rng.sample(pool, round(len(pool) * args.plain_frac))
    if split == "train":
        out += replay_rows(0, args.replay_per_skill)
    else:
        out += replay_rows(args.replay_per_skill, args.replay_val_per_skill)
    rng.shuffle(out)
    return out, collections.Counter(r["shape"] for r in out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the SA-4 agent-format mix from data/sa2 + playground replay.")
    ap.add_argument("--src", default="data/sa2")
    ap.add_argument("--out", default="data/sa4")
    ap.add_argument("--repair-frac", type=float, default=0.5, help="share of SA-2 fix rows rendered as repair turns")
    ap.add_argument("--plain-frac", type=float, default=0.1, help="share of SA-2 he/mbpp rows kept as replay")
    ap.add_argument("--replay-per-skill", type=int, default=40, help="playground pretrain instances per write skill")
    ap.add_argument("--replay-val-per-skill", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = ROOT / args.src, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"src": args.src, "args": vars(args), "repair_head": REPAIR_HEAD, "repair_ask": REPAIR_ASK, "splits": {}}
    for split in ("train", "val"):
        rows, counts = build(src, split, args, random.Random(f"{args.seed}:{split}"))
        with open(out / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split] = {"rows": len(rows), "shapes": dict(sorted(counts.items()))}
        print(json.dumps({"split": split, **report["splits"][split]}), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("DONE", out)


if __name__ == "__main__":
    main()
