# mex/scripts/build_mixed_eval.py — mu1 mixed eval/train sets (CPU, deterministic).
r"""Build the mu1 mixed sets from the SAME held-out distributions eval_mex.py
uses (mu1 plan "Mixed eval set"; research/micro_experts/DESIGN.md #4, arm C).

Outputs (data/mex/mixed/):
  val.jsonl   200 items = deduped 50 per task drawn from each task's HELD-OUT
              (val/test) mix exactly the way eval_mex.py builds its test
              prompts: x1 = the wordlist test slice, x2/x3/x4 = the generators'
              test split (seed=42). Each line:
                  {"prompt": <input text, no task label>,
                   "target": <expected continuation>, "task": "x1|x2|x3|x4"}
  train.jsonl 8000 items = 2000 per task from the generators'/wordlist TRAIN
              pools (the pools the experts trained on) with seed 42; x1 uses
              the wordlist train slice. Router training data (train_router.py).

Determinism: the only randomness is the module-level mixed-rng seed (42), so
re-running reproduces byte-identical files (mex/tests/test_mixed_build.py).

Usage:
    & .\.venv\Scripts\python.exe mex\scripts\build_mixed_eval.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mex.src.tasks import arith, structure, strops  # noqa: E402  (same seeds as pack)

SEED = 42
MIXED_DIR = ROOT / "data" / "mex" / "mixed"
VAL_PER_TASK = 50
TRAIN_PER_TASK = 2000
TASKS = ["x1", "x2", "x3", "x4"]


def task_prompt_target(line: str, task: str) -> tuple[str, str]:
    """eval_mex.py's exact prompt/target split for one raw task line.

    The prompt keeps its in-format terminator ('|', or the x3 newline); the
    target is the line-level continuation with the trailing newline stripped
    (exact-match scoring is whitespace-insensitive at the edges and the
    models emit the terminator themselves).
    """
    if task == "x3":
        seq, label = line.rstrip("\n").split("\n")
        return seq + "\n", label
    prompt, target = line.split("|", 1)
    return prompt + "|", target.rstrip("\n")


def heldout_lines(task: str) -> dict[str, list[str]]:
    """Held-out ('train'/'val'/'test') raw lines, built the eval_mex.py way.

    x1: the wordlist files build_x1_words.py wrote (train 60000 / val 1000 /
    test 2000 lines). x2/x3/x4: the seeded generators' splits, seed=42 with
    eval_mex.py's exact n_val/n_test.
    """
    if task == "x1":
        out = {}
        for split in ("train", "val", "test"):
            f = ROOT / "data" / "mex" / "x1" / f"{split}.txt"
            out[split] = f.read_text(encoding="utf-8").splitlines()
        return out
    gen = {"x2": arith, "x3": structure, "x4": strops}[task]
    d = gen(seed=42, n_val=200, n_test=500)
    return {"train": d["train"], "val": d["val"], "test": d["test"]}


def build_task_items(task: str, per_task: int, rng: random.Random,
                     split: str) -> list[dict]:
    """Deduped sample of per_task items from one task's pool of lines.

    Prompt/target pairs (not raw lines) are deduped: x1 lines can repeat a
    prompt with a different vocalization, and the mixed set must be
    unambiguous for exact-match scoring. Draw order is the single source of
    mixed-set randomness (module-level rng, seed 42).
    """
    lines = heldout_lines(task)[split]
    seen: set[tuple[str, str]] = set()
    pool: list[tuple[str, str]] = []
    for ln in lines:
        pt = task_prompt_target(ln, task)
        if pt not in seen:
            seen.add(pt)
            pool.append(pt)
    if len(pool) < per_task:
        raise ValueError(f"{task}: only {len(pool)} unique {split} items "
                         f"(need {per_task})")
    picked = rng.sample(pool, per_task)
    return [{"prompt": p, "target": t, "task": task} for p, t in picked]


def interleave(per_task_blocks: list[list[dict]], per_task: int) -> list[dict]:
    """Round-robin one item per task (mu1 plan seed order) so the files mix
    tasks instead of four contiguous per-task blocks."""
    return [per_task_blocks[j][i] for i in range(per_task)
            for j in range(len(TASKS))]


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps({"prompt": item["prompt"], "target": item["target"],
                                "task": item["task"]},
                               ensure_ascii=False) + "\n")


def build_items() -> tuple[list[dict], list[dict]]:
    """Build the full (val, train) item lists — the whole mixed-set build.

    Same parameters every call => identical lists (mex/tests/test_mixed_build.py).
    """
    rng = random.Random(SEED)
    val_blocks: list[list[dict]] = []
    train_blocks: list[list[dict]] = []
    for t in TASKS:
        val_blocks.append(build_task_items(t, VAL_PER_TASK, rng, "test"))
        train_blocks.append(build_task_items(t, TRAIN_PER_TASK, rng, "train"))
    return interleave(val_blocks, VAL_PER_TASK), interleave(train_blocks,
                                                            TRAIN_PER_TASK)


def main() -> None:
    val, train = build_items()
    write_jsonl(MIXED_DIR / "val.jsonl", val)
    write_jsonl(MIXED_DIR / "train.jsonl", train)
    print(f"wrote {MIXED_DIR / 'val.jsonl'}: {len(val)} items "
          f"({dict(sorted(Counter(i['task'] for i in val).items()))})")
    print(f"wrote {MIXED_DIR / 'train.jsonl'}: {len(train)} items "
          f"({dict(sorted(Counter(i['task'] for i in train).items()))})")


if __name__ == "__main__":
    main()
