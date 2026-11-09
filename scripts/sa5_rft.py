"""SA-5 RFT data: sample k completions per train task, keep the ones that pass, write a JSONL mix for sa3_data.py.

One round (research/2026-10-26_sa5_rft_proposal.md):
  pool    code = SA-2 he/mbpp rows (data/sa2/train.jsonl); fix = SA-4 fix + repair rows; tool = SA-4 xLAM rows
          (data/sa4/train.jsonl, so prompts match what SA-4 trained on). Code/fix rows need >= --min-asserts.
          Each pool is shuffled once by seed; round r takes slice r, so rounds never reuse a task.
  sample  k completions per task at temperature T. Code/fix text is cut like bench_passk (HumanEval stop rule
          for body shapes, top-level rule for mbpp); a sample that never stops (no EOS, nothing cut) is dropped.
  reward  code/fix: every assert passes under sa2_build.run_guarded (GUARD prelude, timeout, child RSS cap).
          tool: the reply parses as the SA-4 call list and matches the reference exactly (order-insensitive).
  keep    tasks solved by 1..k-1 samples, plus at most --anchor-frac of that count from fully solved tasks;
          <= --keep-per-task distinct passing samples per task (AST dedup, shortest first).
  mix     kept rows + SA-4 train replay (all shapes) so replay is --replay-frac of the train file;
          val = data/sa4/val.jsonl unchanged, so eval_loss stays comparable with SA-4.

A sandbox timeout or RSS kill is retried once with RETRY_TIMEOUT: if the retry passes, the kill was host
load, not the program, so the sample is NOT_EVALUATED (left out of pass rates and RL groups); otherwise the
retry's verdict stands. --task-ids (with --split val) restricts the code pool to a JSON id list, e.g. the
SA-V.1 clean gate research/raw/sa_v_clean_gate.json; without it the pools are unchanged.

Samples are appended to OUT/samples.jsonl task by task, so a stopped run resumes where it left off.
--dry-run scores each task's reference completion once (no model, CPU): every one should pass. It also
writes the data dir, treating every passing task as kept, so the writer and sa3_data.py can be checked.

  .venv/Scripts/python.exe -I scripts/sa5_rft.py --dry-run --code 300 --fix 125 --tool 75 --out scratch/sa5_dry
  .venv/Scripts/python.exe -I scripts/sa5_rft.py --model runs/sa4_scale075 --round 1 --out data/sa5_r1 --vram-cap-gib <free - 0.75>
  .venv/Scripts/python.exe -I scripts/sa3_data.py --src data/sa5_r1 --shapes he,mbpp,fix,repair,tool,replay --out data/sa5_r1/ds
"""
import argparse
import ast
import collections
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from bench_passk import truncate  # noqa: E402
from sa2_build import MIN_FREE_RAM, TIMEOUT, program, ram_guard, run_guarded  # noqa: E402

TOOL_PREFIX = '# {"name": '
CODE_SHAPES = ("he", "mbpp")
FIX_SHAPES = ("fix", "repair")
KILLED = ("timeout", "rss")
RETRY_TIMEOUT = 4 * TIMEOUT
NOT_EVALUATED = "not_evaluated"


def load_task_ids(path: str) -> set[str]:
    data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    ids = data["ids"] if isinstance(data, dict) else data
    if not ids or not all(isinstance(i, str) for i in ids):
        raise SystemExit(f"[sa5] {path}: expected a non-empty id list or {{'ids': [...]}}")
    return set(ids)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pools(args) -> dict[str, list[dict]]:
    sa2, sa4 = read_jsonl(ROOT / args.sa2 / f"{args.split}.jsonl"), read_jsonl(ROOT / args.sa4 / f"{args.split}.jsonl")
    out = {"code": [r for r in sa2 if r["shape"] in CODE_SHAPES and len(r["tests"]) >= args.min_asserts],
           "fix": [r for r in sa4 if r["shape"] in FIX_SHAPES and len(r["tests"]) >= args.min_asserts],
           "tool": [r for r in sa4 if r["shape"] == "tool"]}
    if getattr(args, "task_ids", None):
        keep = load_task_ids(args.task_ids)
        missing = keep - {r["id"] for r in out["code"]}
        if missing:
            raise SystemExit(f"[sa5] {len(missing)} --task-ids not in the {args.split} code pool, e.g. {sorted(missing)[:3]}")
        out["code"] = [r for r in out["code"] if r["id"] in keep]
        if len(out["code"]) != len(keep):
            raise SystemExit(f"[sa5] --task-ids match {len(out['code'])} rows for {len(keep)} ids: some id names 2+ rows")
    for name, rows in out.items():
        rows.sort(key=lambda r: r["id"])
        random.Random(f"{args.seed}:{name}").shuffle(rows)
    return out


def round_slice(pool: dict[str, list[dict]], args) -> list[dict]:
    tasks = []
    for name in ("code", "fix", "tool"):
        n = getattr(args, name)
        rows = pool[name][(args.round - 1) * n: args.round * n]
        if len(rows) < n:
            raise SystemExit(f"[sa5] pool {name} has {len(pool[name])} rows: round {args.round} needs {args.round * n}")
        tasks += rows
    return tasks


def tool_names(prompt: str) -> set[str]:
    names = set()
    for line in prompt.split("\n"):
        if line.startswith(TOOL_PREFIX):
            try:
                names.add(json.loads(line[2:])["name"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return names


def judge_tool(row: dict, raw: str) -> tuple[bool, str]:
    """Exact call-set match against the reference, the SA-4 format check's 'exact' verdict."""
    try:
        calls = json.loads(raw.strip())
    except json.JSONDecodeError:
        return False, "json"
    if not (isinstance(calls, list) and calls and all(
            isinstance(c, dict) and set(c) == {"name", "arguments"} and isinstance(c["arguments"], dict)
            for c in calls)):
        return False, "schema"
    if not all(c["name"] in tool_names(row["prompt"]) for c in calls):
        return False, "unknown_tool"
    key = lambda cs: sorted(json.dumps(c, sort_keys=True) for c in cs)
    return (True, "pass") if key(calls) == key(json.loads(row["completion"])) else (False, "mismatch")


def completion_of(row: dict, raw: str) -> str:
    if row["shape"] == "tool":
        return "\n" + raw.strip() + "\n"  # the SA-4 reference layout
    return truncate({"set": "mbpp" if row["shape"] == "mbpp" else "humaneval"}, raw)


def score(row: dict, raw: str, eos: bool) -> dict:
    completion = completion_of(row, raw)
    if row["shape"] == "tool":
        passed, why = judge_tool(row, raw) if eos else (False, "nostop")
    elif not eos and completion.rstrip() == raw.rstrip():
        passed, why = False, "nostop"  # hit max tokens inside the function
    else:
        code = program({**row, "completion": completion})
        passed, why, _ = run_guarded(code)
        if why in KILLED:
            passed, why, _ = run_guarded(code, timeout=RETRY_TIMEOUT)
            if passed:
                passed, why = False, NOT_EVALUATED
    return {"completion": completion, "passed": passed, "why": why}


def shape_key(row: dict, completion: str) -> str:
    code = completion if row["shape"] in ("mbpp", "tool") else row["prompt"] + completion
    try:
        return ast.dump(ast.parse(code))
    except SyntaxError:
        return code.strip()


def sampler(args):
    """Yields (task, [(raw, eos), ...]) per task: k samples from the model, or the reference once in --dry-run."""
    if args.dry_run:
        def run(tasks):
            for row in tasks:
                yield row, [(row["completion"], True)]
        return run

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.vram_cap_gib > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gib * (1 << 30) / total), 0)
    torch.manual_seed(args.seed + args.round)
    tok = AutoTokenizer.from_pretrained(ROOT / args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ROOT / args.model, torch_dtype=torch.float16).cuda().eval()
    per_step = max(1, args.batch // args.k)  # batch counts generated rows

    def run(tasks):
        for shape_tool in (False, True):
            group = sorted((t for t in tasks if (t["shape"] == "tool") == shape_tool), key=lambda t: len(t["prompt"]))
            max_new = args.max_new_tool if shape_tool else args.max_new_code
            for start in range(0, len(group), per_step):
                chunk = group[start:start + per_step]
                enc = tok([t["prompt"] for t in chunk], return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                         num_return_sequences=args.k, max_new_tokens=max_new,
                                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                seqs = out[:, enc["input_ids"].shape[1]:].tolist()
                for slot, row in enumerate(chunk):
                    samples = []
                    for seq in seqs[slot * args.k:(slot + 1) * args.k]:  # num_return_sequences keeps them adjacent
                        eos = tok.eos_token_id in seq
                        if eos:
                            seq = seq[:seq.index(tok.eos_token_id)]
                        samples.append((tok.decode(seq, skip_special_tokens=True), eos))
                    yield row, samples
    return run


def select(tasks: list[dict], done: dict[str, list[dict]], args) -> tuple[list[dict], dict]:
    rng = random.Random(f"{args.seed}:{args.round}:select")
    learn, full = [], []
    for row in tasks:
        good = [s for s in done[row["id"]] if s["passed"]]
        if not good:
            continue
        seen, rows = set(), []
        for s in sorted(good, key=lambda s: len(s["completion"])):
            key = shape_key(row, s["completion"])
            if key not in seen and len(rows) < args.keep_per_task:
                seen.add(key)
                rows.append({**row, "id": f"{row['id']}/rft{s['sample']}", "source": f"rft-{row['source']}",
                             "completion": s["completion"]})
        evaluated = sum(s["why"] != NOT_EVALUATED for s in done[row["id"]])
        (full if len(good) == evaluated and not args.dry_run else learn).append(rows)
    anchors = rng.sample(full, min(len(full), int(args.anchor_frac * len(learn))))
    kept = [r for rows in learn + anchors for r in rows]
    return kept, {"learn_tasks": len(learn), "full_tasks": len(full), "anchor_tasks": len(anchors), "rows": len(kept)}


def main() -> None:
    ap = argparse.ArgumentParser(description="SA-5 rejection-sampling fine-tuning data for one round.")
    ap.add_argument("--model", default="runs/sa4_scale075", help="merged HF dir to sample from")
    ap.add_argument("--round", type=int, default=1, help="1-based; picks the task slice")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=("train", "val"), default="train",
                    help="val = gate metric (sampled pass@1): sample + score only, no train/val files")
    ap.add_argument("--sa2", default="data/sa2")
    ap.add_argument("--sa4", default="data/sa4")
    ap.add_argument("--code", type=int, default=2400, help="he/mbpp tasks per round")
    ap.add_argument("--task-ids", default=None,
                    help="--split val only: JSON id list (or {'ids': [...]}) restricting the code pool")
    ap.add_argument("--fix", type=int, default=1000, help="fix/repair tasks per round")
    ap.add_argument("--tool", type=int, default=600, help="tool-call tasks per round")
    ap.add_argument("--min-asserts", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-code", type=int, default=384)
    ap.add_argument("--max-new-tool", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64, help="generated rows per step (prompts = batch // k)")
    ap.add_argument("--workers", type=int, default=4, help="sandbox subprocesses (1 GiB RSS cap each)")
    ap.add_argument("--keep-per-task", type=int, default=2)
    ap.add_argument("--anchor-frac", type=float, default=0.1, help="fully solved tasks kept, per partly solved task")
    ap.add_argument("--replay-frac", type=float, default=0.25, help="SA-4 replay share of the train file")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="score reference completions, no model")
    args = ap.parse_args()
    if args.dry_run:
        args.k = 1
    if args.round < 1 or args.k < 1:
        raise SystemExit("[sa5] --round and --k must be >= 1")
    if args.task_ids and args.split != "val":
        raise SystemExit("[sa5] --task-ids is a gate option: use it with --split val")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ram_guard(MIN_FREE_RAM)
    tasks = round_slice(pools(args), args)
    by_id = {t["id"]: t for t in tasks}
    log = out / "samples.jsonl"
    done: dict[str, list[dict]] = collections.defaultdict(list)
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line from a stopped run
            if s["id"] in by_id:
                done[s["id"]].append(s)
    done = collections.defaultdict(list, {i: v for i, v in done.items() if len(v) == args.k})
    todo = [t for t in tasks if t["id"] not in done]
    print(f"[sa5] round {args.round}: {len(tasks)} tasks, {len(done)} already sampled, {len(todo)} to go", flush=True)

    t0, gen_s, score_s = time.perf_counter(), 0.0, 0.0
    with ThreadPoolExecutor(args.workers) as pool, log.open("a", encoding="utf-8", newline="\n") as f:
        tick = time.perf_counter()
        for n, (row, samples) in enumerate(sampler(args)(todo), 1):
            mid = time.perf_counter()
            gen_s += mid - tick
            scored = list(pool.map(lambda s: score(row, *s), samples))
            for i, ((raw, eos), s) in enumerate(zip(samples, scored)):
                rec = {"id": row["id"], "shape": row["shape"], "sample": i, "eos": eos, "raw": raw, **s}
                done[row["id"]].append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            tick = time.perf_counter()
            score_s += tick - mid
            if n % 50 == 0 or n == len(todo):
                ram_guard(MIN_FREE_RAM)
                print(f"[sa5] {n}/{len(todo)} tasks  gen {gen_s:.0f}s  score {score_s:.0f}s", flush=True)

    per_shape: dict[str, dict] = {}
    for row in tasks:
        rows = done[row["id"]]
        c = sum(s["passed"] for s in rows)
        st = per_shape.setdefault(row["shape"], {"tasks": 0, "samples": 0, "passed": 0, "not_evaluated": 0,
                                                 "solved_hist": [0] * (args.k + 1), "why": collections.Counter()})
        st["tasks"] += 1
        st["samples"] += len(rows)
        st["passed"] += c
        st["not_evaluated"] += sum(s["why"] == NOT_EVALUATED for s in rows)
        st["solved_hist"][c] += 1
        st["why"].update(s["why"] for s in rows if not s["passed"])
    for st in per_shape.values():
        evaluated = st["samples"] - st["not_evaluated"]
        st["pass_rate"] = round(st["passed"] / evaluated, 4) if evaluated else None
        st["why"] = dict(st["why"].most_common())

    report = {"args": vars(args), "seconds": round(time.perf_counter() - t0, 1), "gen_seconds": round(gen_s, 1),
              "score_seconds": round(score_s, 1), "per_shape": per_shape}
    if args.split == "train":
        kept, selection = select(tasks, done, args)
        slice_ids = set(by_id)
        replay_pool = [r for r in read_jsonl(ROOT / args.sa4 / "train.jsonl") if r["id"] not in slice_ids]
        n_replay = round(len(kept) * args.replay_frac / (1 - args.replay_frac))
        rng = random.Random(f"{args.seed}:{args.round}:replay")
        train = kept + [{**r, "id": f"{r['id']}/sa5replay", "source": f"replay-{r['source']}"}
                        for r in rng.sample(replay_pool, min(n_replay, len(replay_pool)))]
        rng.shuffle(train)
        for split, rows in (("train", train), ("val", read_jsonl(ROOT / args.sa4 / "val.jsonl"))):
            with open(out / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report.update(selection=selection, train_rows=len(train),
                      train_shapes=dict(collections.Counter(r["shape"] for r in train)),
                      replay_rows=len(train) - len(kept))
    if "torch" in sys.modules and not args.dry_run:
        report["peak_vram_gib"] = round(sys.modules["torch"].cuda.max_memory_allocated() / (1 << 30), 2)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, indent=1))
    print("DONE", out)


if __name__ == "__main__":
    main()
