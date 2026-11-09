"""Never-seen bench eval (2026-10-29). EVAL ONLY. Every set/category is its own bench: scored, reported and compared
separately, never pooled (user rule: focused and general tasks are measured apart).

Per item: greedy pass@1, and the sa7-style runtime at k samples (visible filter, then vote_key majority among passers;
no passer or no visible test -> majority over all samples). "pass" = every test (visible + hidden), as in sa7.

Two tiers, each compared only with itself: full (every item, k=16) for a candidate worth keeping, and --quick (a fixed
QUICK_ITEMS per set, k=8, max-new 256) after every experiment. Quick reports are named <name>_quick and catch big moves
only: with 50 items a set's noise is about +-12 points.

  eval     .venv/Scripts/python.exe -I scripts/bench_ns_eval.py --model models/hf/SmolLM2-135M --name base
               --vram-cap-gib <fresh free - 0.75> [--quick] [--sets A_cruxeval,B_fix] [--k 16]
  compare  .venv/Scripts/python.exe -I scripts/bench_ns_eval.py --compare base,<name>
           .venv/Scripts/python.exe -I scripts/bench_ns_eval.py --compare base_quick,<name>_quick
Writes research/raw/bench_ns_<name>.json; compare prints per-set deltas with paired bootstrap 95% CIs.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from bench_ns import NS_DIR, code_of, cut, grade, load, vote_key  # noqa: E402
from sa2_build import CHILD_RSS_CAP, MIN_FREE_RAM, parallel, ram_guard  # noqa: E402
from sa7_runtime import HFModel, bootstrap_ci  # noqa: E402

OUT_DIR = ROOT / "research" / "raw"
QUICK_ITEMS = 50


def out_path(name: str) -> Path:
    return OUT_DIR / f"bench_ns_{name}.json"


def pick(item: dict, completions: list[str], visible: list[bool]) -> int:
    pool = [i for i, v in enumerate(visible) if v] or list(range(len(completions)))
    keys = [vote_key(item, completions[i]) for i in pool]
    top = Counter(keys).most_common(1)[0][0]
    return pool[keys.index(top)]


def quick_subset(items: list[dict]) -> list[dict]:
    """Fixed by id hash, so no seed or list to keep, and sets stay comparable across models."""
    return sorted(items, key=lambda it: hashlib.sha256(it["id"].encode("utf-8")).hexdigest())[:QUICK_ITEMS]


def grade_answers(items: list[dict], answers: list[list[str]], workers: int, label: str) -> tuple[dict, dict]:
    """(visible, all) verdicts by (item index, completion). Identical programs run once, and "all" contains the
    visible tests, so only visible passers need the second run."""
    jobs = sorted({(i, c) for i, cs in enumerate(answers) for c in cs})

    def verdicts(todo: list, which: str):
        run = lambda job: grade(items[job[0]], code_of(items[job[0]], job[1]), which)  # noqa: E731
        return {job: passed for job, (passed, _) in parallel(todo, run, workers, f"{which} {label}")}

    visible = verdicts(jobs, "visible")
    return visible, dict.fromkeys(jobs, False) | verdicts([job for job in jobs if visible[job]], "all")


def eval_set(path: Path, model, args) -> dict:
    items = load([path])
    if args.quick:
        items = quick_subset(items)
    if args.limit:
        items = items[:args.limit]
    # bench_passk.model_prompt: the tokenizer merges "\n" with the next indent, so a bare final "\n" reads as "done"
    prompts = [it["prompt"].rstrip() for it in items]
    t0 = time.perf_counter()
    greedy = [cut(it, raw) for it, (raw, _) in zip(items, model.greedy(prompts))]
    samples = [[cut(it, raw) for raw, _ in outs] for it, outs in zip(items, model.sample(prompts, args.k))]
    t1 = time.perf_counter()
    visible, full = grade_answers(items, [[g, *cs] for g, cs in zip(greedy, samples)], args.workers, path.stem)
    t2 = time.perf_counter()
    per_item = []
    for i, it in enumerate(items):
        vis = [visible[(i, c)] for c in samples[i]]
        passed = [full[(i, c)] for c in samples[i]]
        a = pick(it, samples[i], vis)
        per_item.append({"id": it["id"], "greedy": full[(i, greedy[i])], "runtime": passed[a],
                         "oracle": any(passed), "visible_pass": sum(vis), "greedy_completion": greedy[i],
                         "answer_completion": samples[i][a]})
    n = len(per_item)
    return {"set": path.stem, "items": n, "k": args.k, "generate_seconds": round(t1 - t0, 1),
            "grade_seconds": round(t2 - t1, 1), "unique_programs": len(visible),
            "greedy_pass1": round(sum(r["greedy"] for r in per_item) / n, 4),
            "runtime_pass": round(sum(r["runtime"] for r in per_item) / n, 4),
            f"oracle_pass@{args.k}": round(sum(r["oracle"] for r in per_item) / n, 4),
            "empty_answers": sum(not r["answer_completion"].strip() for r in per_item),
            "per_item": per_item}


def compare(names: list[str]) -> None:
    base, *others = [json.loads(out_path(n).read_text(encoding="utf-8")) for n in names]
    for other, name in zip(others, names[1:]):
        if any(other["args"][key] != base["args"][key] for key in ("k", "max_new")):
            raise SystemExit(f"{name} and {names[0]} used different k or max_new: compare a tier only with itself")
        print(f"== {name} vs {names[0]} (per set, never pooled)")
        for set_name, b in base["sets"].items():
            o = other["sets"].get(set_name)
            if o is None:
                print(f"  {set_name:16s} missing in {name}")
                continue
            bi = {r["id"]: r for r in b["per_item"]}
            shared = [r for r in o["per_item"] if r["id"] in bi]
            for metric in ("greedy", "runtime"):
                diffs = [float(r[metric]) - float(bi[r["id"]][metric]) for r in shared]
                mean = sum(diffs) / len(diffs)
                print(f"  {set_name:16s} {metric:8s} n={len(shared):4d} {names[0]} "
                      f"{sum(bi[r['id']][metric] for r in shared) / len(shared):.3f} -> {name} "
                      f"{sum(r[metric] for r in shared) / len(shared):.3f}  d={mean:+.3f} CI {bootstrap_ci(diffs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model dir")
    ap.add_argument("--name", help="report name: research/raw/bench_ns_<name>.json")
    ap.add_argument("--compare", help="comma list of report names; the first is the baseline")
    ap.add_argument("--sets", default="", help="comma list of file stems (default: every A_/B_ set file)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help=f"fixed {QUICK_ITEMS} items per set, k=8, max-new 256")
    ap.add_argument("--limit", type=int, default=0, help="first N items per set only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0, help="required on cuda: fresh free - 0.75 GiB")
    ap.add_argument("--batch", type=int, default=64, help="generated rows per step")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare.split(","))
        return
    if not args.model or not args.name:
        ap.error("--model and --name are required (or use --compare)")
    if args.device == "cuda" and args.vram_cap_gib <= 0:
        ap.error("--vram-cap-gib is required on cuda (fresh free VRAM - 0.75 GiB)")
    if args.quick:
        args.k, args.max_new, args.name = 8, 256, f"{args.name}_quick"
    ram_guard(MIN_FREE_RAM + args.workers * CHILD_RSS_CAP)
    paths = [NS_DIR / f"{s}.jsonl" for s in args.sets.split(",")] if args.sets \
        else sorted(p for p in NS_DIR.glob("[AB]_*.jsonl") if p.stem != "B_raw")
    model = HFModel(args)
    out = out_path(args.name)
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"sets": {}}
    report.update(model=args.model, args={k: str(v) for k, v in vars(args).items()})
    for path in paths:
        if path.stem in report["sets"]:  # finished sets survive a stop; delete the report to redo them
            print(f"[skip] {path.stem}", flush=True)
            continue
        t0 = time.perf_counter()
        r = eval_set(path, model, args)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        report["sets"][path.stem] = r
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")  # per set: a crash keeps finished sets
        print(json.dumps({k: v for k, v in r.items() if k != "per_item"}), flush=True)


if __name__ == "__main__":
    main()
