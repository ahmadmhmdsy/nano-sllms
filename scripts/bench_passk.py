"""Greedy pass@1 on HumanEval and MBPP-sanitized: generate a completion, then execute it.

Generation: greedy, left-padded batches, cut at the first eos, then truncated to the
function body with the standard HumanEval stop rule (the first top-level line). MBPP
prompts use the EvalPlus form: a docstring holding the task text and its first test
assert. The model writes the whole function, and the text is cut at the first
top-level line that is not part of a def/class/import.

Execution: one subprocess per task (`python -I`, a fresh temp dir, timeout), run behind a
guard prelude that disables the destructive os/shutil/subprocess calls (the human-eval
reliability_guard idea). It is NOT a security sandbox. It only stops accidents from a
small model's code. A task passes iff the process exits 0 and prints the marker.

    --model canned   harness self-test: scores the reference solutions (must be ~1.0)
    --arm NAME=PATH  real run (HF checkpoint dir, merged weights); repeatable

--samples N (N > 1) switches to temperature sampling with N completions per task and reports
the unbiased pass@k estimate 1 - C(n-c, k)/C(n, k) for k in {1, 5, N}. It also reports the mean
number of distinct correct solutions per solved task, deduplicated by AST, which measures
diversity and not only whether the task was solved. The default N = 1 keeps the greedy report.

Generations are cached in the report, so --reuse re-executes without the GPU.
Run: .venv/Scripts/python scripts/bench_passk.py --model canned
     .venv/Scripts/python scripts/bench_passk.py --arm base=models/hf/SmolLM2-135M --device cuda
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "bench" / "raw"
MARKER = "PASSK_OK"
MIN_FREE_RAM = 8 << 30
GUARD = """\
import builtins, os, shutil, subprocess, sys
for _n in ("system", "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace",
           "kill", "killpg", "fork", "forkpty", "putenv", "chmod", "chown", "truncate",
           "execv", "execve", "execvp", "execvpe", "spawnv", "spawnve", "popen", "startfile"):
    if hasattr(os, _n):
        setattr(os, _n, None)
shutil.rmtree = shutil.move = shutil.chown = None
subprocess.Popen = subprocess.run = subprocess.call = subprocess.check_output = None
builtins.exit = builtins.quit = None
"""
MBPP_EXTRA_TOP = ("def ", "class ", "import ", "from ", "@")


def load_humaneval() -> list[dict]:
    with gzip.open(RAW / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [{"task_id": r["task_id"], "set": "humaneval", "prompt": r["prompt"],
             "canned": r["canonical_solution"],
             "test": f"{r['test']}\ncheck({r['entry_point']})\n"} for r in rows]


def load_mbpp_sanitized() -> list[dict]:
    rows = json.loads((RAW / "sanitized-mbpp.json").read_text(encoding="utf-8"))
    out = []
    for r in rows:
        tests = "\n".join(r["test_imports"] + r["test_list"]) + "\n"
        prompt = f'"""\n{r["prompt"]}\n{r["test_list"][0]}\n"""\n'
        out.append({"task_id": f"mbpp-san/{r['task_id']}", "set": "mbpp-sanitized", "prompt": prompt,
                    "canned": r["code"], "test": tests, "imports": "\n".join(r["test_imports"])})
    return out


def truncate(task: dict, text: str) -> str:
    """Cut a raw continuation down to the code that belongs to the answer."""
    lines = text.split("\n")
    keep = []
    if task["set"] == "humaneval":
        # Body continuation: stop at the first non-blank top-level line (HumanEval's stop rule).
        for line in lines:
            if line.strip() and not line[0].isspace():
                break
            keep.append(line)
    else:
        seen_def = False
        for line in lines:
            top = line.strip() and not line[0].isspace()
            if top and seen_def and not line.startswith(MBPP_EXTRA_TOP):
                break
            if top and line.startswith("```"):
                break
            seen_def = seen_def or line.startswith("def ")
            keep.append(line)
    return "\n".join(keep).rstrip() + "\n"


def model_prompt(task: dict) -> str:
    # Trailing whitespace is stripped (bigcode-evaluation-harness does the same): SmolLM2's
    # tokenizer merges "\n" with the indent that follows, so a prompt ending in a bare "\n"
    # token reads as end-of-file and greedy decoding emits eos at once.
    return task["prompt"].rstrip()


def program(task: dict, completion: str) -> str:
    # The guard prelude comes first, so a generated `from __future__` line would be a
    # SyntaxError; such imports are dropped (no-ops on Python 3.12 apart from annotations).
    completion = "\n".join(l for l in completion.split("\n") if not l.startswith("from __future__"))
    if task["set"] == "humaneval":
        code = model_prompt(task) + completion
    else:
        code = (task["imports"] + "\n" if task["imports"] else "") + completion
    return f"{GUARD}\n{code}\n\n{task['test']}\nprint({MARKER!r})\n"


def execute(code: str, timeout: float) -> tuple[bool, str | None]:
    with tempfile.TemporaryDirectory(prefix="passk_") as td:
        path = Path(td) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=td, capture_output=True,
                                  text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode == 0 and MARKER in proc.stdout:
            return True, None
        return False, (proc.stderr or proc.stdout or "")[-300:]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021) from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def shape_key(task: dict, completion: str) -> str:
    """AST fingerprint of a solution, so reformatted copies count as one shape."""
    code = model_prompt(task) + completion if task["set"] == "humaneval" else completion
    try:
        return ast.dump(ast.parse(code))
    except SyntaxError:
        return code.strip()


def generate(path: str, tasks: list[dict], device: str, batch: int, max_tokens: int,
             samples: int = 1, temperature: float = 0.8, top_p: float = 0.95, seed: int = 0) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    # batch counts generated rows, so the prompts per step shrink as samples grow.
    batch = max(1, batch // samples)
    sampling = ({"do_sample": True, "temperature": temperature, "top_p": top_p, "num_return_sequences": samples}
                if samples > 1 else {"do_sample": False})
    tok = AutoTokenizer.from_pretrained(path)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    out = []
    # Sort by prompt length so batches pad little; results are re-keyed by task_id.
    order = sorted(range(len(tasks)), key=lambda i: len(tasks[i]["prompt"]))
    for start in range(0, len(order), batch):
        chunk = [tasks[i] for i in order[start:start + batch]]
        enc = [tok(model_prompt(t))["input_ids"] for t in chunk]
        width = max(len(e) for e in enc)
        x = torch.full((len(enc), width), pad, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        for slot, ids in enumerate(enc):
            x[slot, width - len(ids):] = torch.tensor(ids)
            mask[slot, width - len(ids):] = 1
        with torch.no_grad():
            gen = model.generate(x.to(device), attention_mask=mask.to(device), max_new_tokens=max_tokens,
                                 eos_token_id=eos, pad_token_id=pad, **sampling)
        for row in range(gen.shape[0]):
            # num_return_sequences keeps each prompt's samples adjacent: row = slot * samples + s.
            task, sample = chunk[row // samples], row % samples
            new = gen[row, width:].tolist()
            emitted = eos in new
            if emitted:
                new = new[:new.index(eos)]
            raw = tok.decode(new, skip_special_tokens=True)
            key = task["task_id"] if samples == 1 else f"{task['task_id']}#{sample}"
            out.append({"key": key, "task_id": task["task_id"], "sample": sample, "raw": raw, "eos": emitted})
        if psutil.virtual_memory().available < MIN_FREE_RAM:
            raise SystemExit(f"[abort] free RAM below {MIN_FREE_RAM >> 30} GiB; nothing written")
        print(f"[gen] {min(start + batch, len(order))}/{len(order)}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def score(tasks: list[dict], gens: dict[str, dict], timeout: float, workers: int) -> dict:
    """gens is keyed by generation (task_id, or task_id#s when sampling); each row names its task_id."""
    by_id = {t["task_id"]: t for t in tasks}

    def one(key: str) -> tuple[str, bool, str | None]:
        task = by_id[gens[key].get("task_id", key)]
        ok, err = execute(program(task, gens[key]["completion"]), timeout)
        return key, ok, err

    per_task: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, ok, err in pool.map(one, list(gens)):
            gens[key]["passed"] = ok
            gens[key]["error"] = err
            per_task.setdefault(gens[key].get("task_id", key), []).append(gens[key])
    per_set: dict[str, dict] = {}
    for task_id, rows in per_task.items():
        task = by_id[task_id]
        n, c = len(rows), sum(r["passed"] for r in rows)
        row = per_set.setdefault(task["set"], {"tasks": 0, "passed": 0, "eos": 0, "generations": 0,
                                               "solved": 0, "distinct_correct": 0, "_pass": {}})
        row["tasks"] += 1
        row["generations"] += n
        row["passed"] += c if n == 1 else 0
        row["eos"] += sum(bool(r.get("eos")) for r in rows)
        row["solved"] += c > 0
        row["distinct_correct"] += len({shape_key(task, r["completion"]) for r in rows if r["passed"]})
        for k in sorted({1, 5, n}):
            if k <= n:
                row["_pass"][k] = row["_pass"].get(k, 0.0) + pass_at_k(n, c, k)
    for row in per_set.values():
        pass_k = row.pop("_pass")
        if row["generations"] > row["tasks"]:
            row["passed"] = row["solved"]  # sampled run: "passed" means solved by any sample
        row["pass_at_1"] = round(pass_k[1] / row["tasks"], 4)
        row["pass_at_k"] = {str(k): round(v / row["tasks"], 4) for k, v in pass_k.items()}
        row["eos_rate"] = round(row["eos"] / row["generations"], 4)
        row["distinct_per_solved"] = round(row["distinct_correct"] / row["solved"], 3) if row["solved"] else 0.0
    return per_set


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("llm", "canned"), default="llm")
    ap.add_argument("--arm", action="append", default=[], help="NAME=PATH (HF checkpoint dir)")
    ap.add_argument("--sets", default="humaneval,mbpp-sanitized")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks per set (probe)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=2, help="parallel execution subprocesses")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0, help="hard CUDA allocator ceiling (0 = none)")
    ap.add_argument("--samples", type=int, default=1, help="completions per task; >1 samples, reports pass@k + diversity")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", help="re-execute the cached generations in --out")
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "raw" / "smollm2_passk.json")
    args = ap.parse_args()

    loaders = {"humaneval": load_humaneval, "mbpp-sanitized": load_mbpp_sanitized}
    tasks = []
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        if name not in loaders:
            raise SystemExit(f"unknown set {name!r}; known: {sorted(loaders)}")
        rows = loaders[name]()
        tasks += rows[:args.limit] if args.limit else rows

    if args.device == "cuda" and args.vram_cap_gib > 0:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gib * (1 << 30) / total), 0)

    prior = json.loads(args.out.read_text(encoding="utf-8")) if args.reuse and args.out.is_file() else {}
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    sampled = args.samples > 1
    report = {"metric": f"sampled pass@k, n={args.samples} (execution)" if sampled else "greedy pass@1 (execution)",
              "sets": args.sets, "limit": args.limit or None,
              "max_tokens": args.max_tokens, "timeout_s": args.timeout, "arms": {}}
    if sampled:
        report["sampling"] = {"samples": args.samples, "temperature": args.temperature, "top_p": args.top_p,
                              "seed": args.seed}
    if args.model == "canned":
        arms = {"canned": None}
    elif args.reuse:
        arms = {name: None for name in prior.get("arms", {})}
    else:
        if not args.arm:
            raise SystemExit("--arm NAME=PATH is required for --model llm")
        arms = dict(spec.split("=", 1) for spec in args.arm)
    for name, path in arms.items():
        t0 = time.time()
        if args.model == "canned":
            gens = {t["task_id"]: {"completion": t["prompt"][len(model_prompt(t)):] + t["canned"]
                                   if t["set"] == "humaneval" else t["canned"] + "\n"} for t in tasks}
        elif args.reuse:
            gens = {g.get("key", g["task_id"]): g for g in prior["arms"][name]["generations"]}
            path = prior["arms"][name]["path"]
        else:
            raw = generate(path, tasks, args.device, args.batch, args.max_tokens,
                           args.samples, args.temperature, args.top_p, args.seed)
            by_id = {t["task_id"]: t for t in tasks}
            gens = {g.pop("key"): {**g, "completion": truncate(by_id[g["task_id"]], g["raw"])} for g in raw}
        gen_s = time.time() - t0
        per_set = score(tasks, gens, args.timeout, args.workers)
        report["arms"][name] = {"path": path, "per_set": per_set, "gen_seconds": round(gen_s, 1),
                                "total_seconds": round(time.time() - t0, 1),
                                "generations": [{"key": k, "task_id": k, **v} for k, v in gens.items()]}
        for set_name, row in per_set.items():
            extra = ""
            if row["generations"] > row["tasks"]:
                extra = (" " + " ".join(f"pass@{k}={v:.4f}" for k, v in row["pass_at_k"].items() if k != "1")
                         + f" distinct/solved={row['distinct_per_solved']:.2f}")
            print(f"[result] {name:<12} {set_name:<15} pass@1={row['pass_at_1']:.4f} "
                  f"({row['passed']}/{row['tasks']}) eos={row['eos_rate']:.3f}{extra}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
