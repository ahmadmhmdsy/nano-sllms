"""E2 step 1 (2026-10-29): verifier labels. Sample a generator k times on data/sa2 train tasks and grade every sample.

Task pool = the clean-gate rule on data/sa2/train.jsonl: shape he|mbpp, >= 3 asserts, no xlam rows (tool calls),
no prompt shared with data/sa2/val.jsonl. A seeded random --tasks of them, split and prompted exactly as
sa7_runtime.load_gate (first assert visible, the rest hidden). Each sample gets its visible kind (sa7_runtime.check)
and its full-test grade (sa7_runtime.grade), so the dump reads like an sa7_runtime --retries 0 dump (load_dump).
The generator saw these tasks in SA-2..SA-4 training: labels are in-distribution for it, not held-out evidence.
Chunks of --chunk tasks are appended as they finish; a re-run skips the task ids already in the dump.

  .venv/Scripts/python.exe -I scripts/e2_verifier_data.py --model runs/sa4_scale075 --k 16 --vram-cap-gib 6.4 \
      --out research/raw/e2_train_sa4
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from sa2_build import CHILD_RSS_CAP, MIN_FREE_RAM, parallel, ram_guard  # noqa: E402
from sa7_runtime import HFModel, check, first_sample, grade, make_task  # noqa: E402

EXCLUDED_SOURCES = ("xlam",)


def eligible(row: dict, val_prompts: set[str]) -> bool:
    return (row["source"] not in EXCLUDED_SOURCES and row["shape"] in ("he", "mbpp") and len(row["tests"]) >= 3
            and row["prompt"] not in val_prompts)


def load_tasks(n: int, seed: int) -> list[dict]:
    with (ROOT / "data" / "sa2" / "val.jsonl").open(encoding="utf-8") as f:
        val_prompts = {json.loads(line)["prompt"] for line in f}
    path = ROOT / "data" / "sa2" / "train.jsonl"
    with path.open(encoding="utf-8") as f:  # two passes: keep line numbers, not 140k rows
        keep = [i for i, line in enumerate(f) if eligible(json.loads(line), val_prompts)]
    chosen = set(random.Random(seed).sample(keep, min(n, len(keep))))
    tasks = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in chosen:
                continue
            row = json.loads(line)
            tests = row["tests"]
            task = make_task(row["id"], "train", "full" if row["shape"] == "mbpp" else "body", row["prompt"],
                             row["entry_point"], row["completion"], tests[:1], "\n".join(tests[1:]))
            task["source"] = row["source"]
            tasks.append(task)
    print(f"[tasks] {len(keep)} eligible, {len(tasks)} chosen: {dict(Counter(t['source'] for t in tasks))}",
          flush=True)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", type=int, default=1500)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=100, help="tasks per append (the resume unit)")
    ap.add_argument("--limit", type=int, default=0, help="first N chosen tasks only (dry runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0, help="required on cuda: fresh free - 0.75 GiB")
    ap.add_argument("--batch", type=int, default=128, help="generated rows per step")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="stem: <out>.tasks.jsonl, .samples.jsonl, .json")
    args = ap.parse_args()
    if args.device == "cuda" and args.vram_cap_gib <= 0:
        ap.error("--vram-cap-gib is required on cuda (fresh free VRAM - 0.75 GiB)")
    ram_guard(MIN_FREE_RAM + args.workers * CHILD_RSS_CAP)

    t0 = time.perf_counter()
    tasks = load_tasks(args.tasks, args.seed)
    if args.limit:
        tasks = tasks[:args.limit]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks_path, samples_path = out.with_name(out.name + ".tasks.jsonl"), out.with_name(out.name + ".samples.jsonl")
    with tasks_path.open("w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    done = set()
    if samples_path.exists():
        with samples_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [t for t in tasks if t["id"] not in done]
    print(f"[resume] {len(done)} tasks in the dump, {len(todo)} to sample", flush=True)
    model = HFModel(args) if todo else None
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        items = []
        for task, outs in zip(chunk, model.sample([t["prompt"] for t in chunk], args.k)):
            task["samples"] = [first_sample(task, raw, eos) for raw, eos in outs]
            items += [(task, s) for s in task["samples"]]
        check(items, args.workers, "visible")
        for (task, s), (passed, why) in parallel(items, lambda ts: (False, "nostop") if ts[1]["nostop"]
                                                 else grade(ts[0], ts[1]["code"]), args.workers, "grade"):
            s["passed"], s["why"] = passed, why
        with samples_path.open("a", encoding="utf-8", newline="\n") as f:  # whole tasks only: resume by id
            for t in chunk:
                for i, s in enumerate(t.pop("samples")):
                    f.write(json.dumps({"id": t["id"], "i": i, **s}) + "\n")
        print(f"[chunk] {start + len(chunk)}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)

    per_task: dict[str, list[bool]] = {}
    with samples_path.open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            per_task.setdefault(s["id"], []).append(s["passed"])
    n_pass = Counter("none" if not any(v) else "all" if all(v) else "mixed" for v in per_task.values())
    source = {t["id"]: t["source"] for t in tasks}
    report = {"model": args.model, "k": args.k, "tasks": len(per_task), "outcome": dict(n_pass),
              "mixed_by_source": dict(Counter(source[tid] for tid, v in per_task.items() if any(v) and not all(v))),
              "sample_pass": round(sum(map(sum, per_task.values())) / max(1, sum(map(len, per_task.values()))), 4),
              "seconds": round(time.perf_counter() - t0, 1), "args": {k: str(v) for k, v in vars(args).items()}}
    out.with_name(out.name + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
