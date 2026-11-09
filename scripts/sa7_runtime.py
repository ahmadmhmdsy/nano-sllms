"""SA-7 runtime: sample-k + visible-test filter + retry-with-traceback (research/2026-10-27_sa7_runtime_proposal.md).

Per task the model draws k samples. Each runs against the task's *visible* tests only; among the passers the
largest AST cluster wins (ties: shortest code, then first). If none passes, the most informative failure (wrong
value > exception) is rendered as an SA-4 `repair` prompt (attempt -> real outcome -> retry, the shape SA-4 trained
on) and k_r more samples are drawn, for at most R rounds: k + R*k_r samples per task at most. A task with no
visible tests is vote-only: the largest cluster among samples that load without error, and no retry.

Every number is scored on the full tests (visible + hidden), which the filter never sees in full:
  greedy pass@1 (baseline) | runtime pass (the answer picked) | oracle pass@N (any of the N samples passes).
Splits (proposal §2): clean-gate rows (data/sa2/val.jsonl, research/raw/sa_v_clean_gate.json) and MBPP-sanitized
keep the first assert visible and the rest hidden; one assert means vote-only. HumanEval turns the docstring
`>>> call` / result pairs into visible asserts (pairs the canonical solution fails are dropped and counted);
check() is hidden. Sandbox and the 4x-timeout retry -> not_evaluated rule are sa2_build/sa5_rft's.

--dry-run needs no model: it checks the split (every reference must pass visible, hidden and full) and runs the
pipeline end to end with the reference as the only sample (k 1, no retry).

  .venv/Scripts/python.exe -I scripts/sa7_runtime.py --set gate --dry-run --out research/raw/sa7_dry_gate.json
  .venv/Scripts/python.exe -I scripts/sa7_runtime.py --set gate --model runs/sa4_scale075 --vram-cap-gib 6.4 \
      --out research/raw/sa7_gate.json
"""
import argparse
import ast
import json
import random
import re
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from bench_passk import RAW, load_humaneval, pass_at_k, truncate  # noqa: E402
from sa2_build import (CHILD_RSS_CAP, GUARD, MARKER, MIN_FREE_RAM, WITNESS, is_main_guard, parallel,  # noqa: E402
                       ram_guard, run_guarded, simple_example, strip_doc)
from sa4_data import REPAIR_ASK, REPAIR_HEAD  # noqa: E402
from sa5_rft import KILLED, NOT_EVALUATED, RETRY_TIMEOUT, load_task_ids  # noqa: E402

OUTCOME_CAP = 600
KEEP = "SA7_KEEP "
# Failure kinds, most informative first. Only "wrong" and "error" carry a witness, so only they seed a retry.
RANK = {"wrong": 0, "error": 1, "crash": 2, "timeout": 3, "rss": 3, "nostop": 3}


# ---------------------------------------------------------------- tasks

def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    return textwrap.shorten(re.split(r"(?<=[.!?])\s", text)[0], 200, placeholder="...") if text else ""


def description(prompt: str, kind: str, entry: str) -> str:
    """The first sentence of the task text: the module docstring (full) or the entry docstring (body)."""
    try:
        tree = ast.parse(prompt if kind == "full" else prompt.rstrip() + "\n    pass\n")
    except SyntaxError:
        return ""
    if kind == "full":
        doc = ast.get_docstring(tree)
    else:
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry), None)
        doc = ast.get_docstring(func) if func else None
    return first_sentence((doc or "").split(">>>")[0].split("\nassert ")[0])


def make_task(task_id: str, set_name: str, kind: str, prompt: str, entry: str, reference_raw: str,
              visible: list[str], hidden: str, preamble: str = "") -> dict:
    """kind "body": code = prompt + completion; "full": code = preamble + completion."""
    task = {"id": task_id, "set": set_name, "kind": kind, "prompt": prompt, "entry_point": entry,
            "preamble": preamble, "visible": visible, "hidden": hidden, "reference_raw": reference_raw,
            "description": description(prompt, kind, entry)}
    task["reference_code"] = code_of(task, reference_raw)
    return task


def load_gate(ids_path: str) -> list[dict]:
    ids = load_task_ids(ids_path)
    tasks = []
    with (ROOT / "data" / "sa2" / "val.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] not in ids:
                continue
            tests = row["tests"]
            visible, hidden = (tests[:1], tests[1:]) if len(tests) > 1 else ([], tests)
            # The prompt stays as sa2 stored it: that is what SA-2..SA-6 trained and sampled on.
            tasks.append(make_task(row["id"], "gate", "full" if row["shape"] == "mbpp" else "body", row["prompt"],
                                   row["entry_point"], row["completion"], visible, "\n".join(hidden)))
    if len(tasks) != len(ids):
        raise SystemExit(f"ABORT: {len(ids)} gate ids but {len(tasks)} rows in data/sa2/val.jsonl")
    return tasks


def mbpp_entry(code: str, test: str) -> str:
    defined = [n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    for node in ast.walk(ast.parse(test)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in defined:
            return node.func.id
    return defined[-1]


def load_mbpp() -> list[dict]:
    tasks = []
    for r in json.loads((RAW / "sanitized-mbpp.json").read_text(encoding="utf-8")):
        tl = r["test_list"]
        visible, hidden = (tl[:1], tl[1:]) if len(tl) > 1 else ([], tl)
        prompt = f'"""\n{r["prompt"]}\n{tl[0]}\n"""\n'.rstrip()  # bench_passk.model_prompt
        tasks.append(make_task(f"mbpp-san/{r['task_id']}", "mbpp", "full", prompt, mbpp_entry(r["code"], tl[0]),
                               "\n" + r["code"], visible, "\n".join(hidden), "\n".join(r["test_imports"])))
    return tasks


def he_examples(prompt: str) -> list[str]:
    """Docstring `>>> expr` followed by a one-line result -> 'assert expr == result'."""
    lines = [line.strip() for line in prompt.split("\n")]
    out = []
    for i, line in enumerate(lines[:-1]):
        if not line.startswith(">>> "):
            continue
        expr, expected = line[4:].strip(), lines[i + 1]
        after = lines[i + 2] if i + 2 < len(lines) else ""
        if not expected or expected.startswith((">>>", '"""', "'''")):
            continue
        if after and not after.startswith((">>>", '"""', "'''")):
            continue  # a multi-line result or prose
        try:
            ast.parse(expr, mode="eval")
            ast.parse(expected, mode="eval")
            stmt = f"assert {expr} == {expected}"
            ast.parse(stmt)
        except SyntaxError:
            continue
        out.append(stmt)
    return out


WIDE_ARROW = re.compile(r"\s*(?:==>|=>|➞|->|→|==|=)\s*")
WIDE_WORDS = {"true": "True", "false": "False", "none": "None", "null": "None"}


def he_examples_wide(prompt: str, entry: str) -> list[str]:
    """Prose examples `entry(args) => result` (also ==>, ➞, ->, →, ==, =) -> 'assert entry(args) == result'.
    Opt-in (SA-M step 4b): only for tasks without `>>>` examples; the reference filter in load_he still applies."""
    out = []
    for line in prompt.split("\n"):
        start = line.find(entry + "(")
        if start < 0:
            continue
        text = line[start:].strip()
        for m in WIDE_ARROW.finditer(text):
            expr, expected = text[:m.start()].strip().rstrip("#").strip(), text[m.end():].strip().rstrip(".,;")
            expected = WIDE_WORDS.get(expected.lower(), expected)
            try:
                call = ast.parse(expr, mode="eval").body
                ast.parse(expected, mode="eval")
                stmt = f"assert {expr} == {expected}"
                ast.parse(stmt)
            except SyntaxError:
                continue
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == entry:
                out.append(stmt)
                break
    return list(dict.fromkeys(out))


def keep_program(code: str, tests: list[str]) -> str:
    parts = [GUARD, code]
    for i, test in enumerate(tests):
        parts.append(f"try:\n{textwrap.indent(test, '    ')}\n    print({KEEP + str(i)!r})\n"
                     f"except Exception:\n    pass")
    parts.append(f"print({MARKER!r})")
    return "\n".join(parts) + "\n"


def load_he(workers: int, wide: bool = False) -> tuple[list[dict], dict]:
    tasks = []
    for t in load_humaneval():
        entry = re.search(r"check\((\w+)\)\s*$", t["test"]).group(1)
        prompt = t["prompt"].rstrip()  # bench_passk.model_prompt
        visible = he_examples(t["prompt"])
        from_prose = wide and not visible
        if from_prose:
            visible = he_examples_wide(t["prompt"], entry)
        tasks.append(make_task(t["task_id"], "he", "body", prompt, entry, t["prompt"][len(prompt):] + t["canned"],
                               visible, t["test"]))
        tasks[-1]["wide"] = from_prose and bool(visible)
    # Drop the docstring examples the canonical solution itself fails (float reprs, typos in the docstring).
    stats = {"parsed": sum(len(t["visible"]) for t in tasks), "dropped": 0}
    todo = [t for t in tasks if t["visible"]]
    for task, (_, _, stdout) in parallel(todo, lambda t: run_guarded(keep_program(t["reference_code"], t["visible"])),
                                         workers, "he examples"):
        kept = {int(line[len(KEEP):]) for line in stdout.splitlines() if line.startswith(KEEP)}
        stats["dropped"] += len(task["visible"]) - len(kept)
        task["visible"] = [test for i, test in enumerate(task["visible"]) if i in kept]
    stats["tasks_with_visible"] = sum(bool(t["visible"]) for t in tasks)
    if wide:
        stats["wide_tasks"] = sum(t["wide"] and bool(t["visible"]) for t in tasks)
    return tasks, stats


# ---------------------------------------------------------------- code, sandbox programs

def code_of(task: dict, completion: str) -> str:
    # The guard prelude comes first, so `from __future__` lines would be a SyntaxError (bench_passk.program).
    completion = "\n".join(line for line in completion.split("\n") if not line.startswith("from __future__"))
    body = task["prompt"] + completion if task["kind"] == "body" else completion
    return (task["preamble"] + "\n" if task["preamble"] else "") + body


def safe_example(test: str, entry: str) -> tuple[str, str] | None:
    try:
        return simple_example(test, entry)
    except (SyntaxError, AttributeError, IndexError):
        return None


def visible_program(task: dict, code: str) -> str:
    """sa2_build.witness_program over the visible tests, plus non-simple asserts run as written."""
    parts = [GUARD, "import json as _sa7_json", code]
    for test in task["visible"]:
        example = safe_example(test, task["entry_point"])
        if example is None:
            parts.append(
                f"try:\n{textwrap.indent(test, '    ')}\nexcept Exception as _sa7_error:\n"
                f"    print({WITNESS!r} + _sa7_json.dumps({{'call': {test!r}, "
                f"'error': type(_sa7_error).__name__ + ': ' + str(_sa7_error)}}))\n    raise SystemExit")
            continue
        call, expected = example
        parts.append(
            f"try:\n    _sa7_got = {call}\nexcept Exception as _sa7_error:\n"
            f"    print({WITNESS!r} + _sa7_json.dumps({{'call': {call!r}, "
            f"'error': type(_sa7_error).__name__ + ': ' + str(_sa7_error)}}))\n    raise SystemExit\n"
            f"if not (_sa7_got == ({expected})):\n"
            f"    print({WITNESS!r} + _sa7_json.dumps({{'call': {call!r}, 'got': repr(_sa7_got), "
            f"'expected': repr({expected})}}))\n    raise SystemExit")
    parts.append(f"print({MARKER!r})")
    return "\n".join(parts) + "\n"


def run_visible(task: dict, code: str) -> dict:
    passed, why, stdout = run_guarded(visible_program(task, code))
    if passed:
        return {"kind": "pass"}
    if why in KILLED:
        return {"kind": why}
    found = [line[len(WITNESS):] for line in stdout.splitlines() if line.startswith(WITNESS)]
    if not found:
        return {"kind": "crash"}  # syntax error, or it died while loading
    witness = json.loads(found[0])
    wrong = "got" in witness or witness["error"].startswith("AssertionError")
    return {"kind": "wrong" if wrong else "error", "witness": witness}


def grade(task: dict, code: str, tests: str = "full") -> tuple[bool, str]:
    parts = {"visible": "\n".join(task["visible"]), "hidden": task["hidden"]}
    body = "\n".join(task["visible"] + [task["hidden"]]) if tests == "full" else parts[tests]
    prog = f"{GUARD}\n{code}\n\n{body}\nprint({MARKER!r})\n"
    passed, why, _ = run_guarded(prog)
    if why in KILLED:
        passed, why, _ = run_guarded(prog, timeout=RETRY_TIMEOUT)
        if passed:
            passed, why = False, NOT_EVALUATED
    return passed, why


def shape(code: str) -> str:
    try:
        return ast.dump(ast.parse(code))
    except SyntaxError:
        return code.strip()


# ---------------------------------------------------------------- samples, retry prompt

def first_sample(task: dict, raw: str, eos: bool) -> dict:
    completion = truncate({"set": "humaneval" if task["kind"] == "body" else "mbpp"}, raw)
    return {"origin": "sample", "raw": raw, "eos": eos, "completion": completion, "code": code_of(task, completion),
            "nostop": not eos and completion.rstrip() == raw.rstrip()}


def attempt_parts(code: str, entry: str) -> tuple[str, str] | None:
    """(context, attempt): the other top-level code, and the entry function without its docstring."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry), None)
    if func is None or func.decorator_list:
        return None
    context = [ast.unparse(n) for n in tree.body if n is not func and not is_main_guard(n)
               and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    return "\n".join(context), ast.unparse(strip_doc(func))


def cap(line: str) -> str:
    return line if len(line) <= OUTCOME_CAP else line[:OUTCOME_CAP - 3] + "..."


def repair_prompt(task: dict, attempt: str, witness: dict) -> str:
    """The SA-4 repair layout (sa4_data.to_repair) built from a live failure."""
    if "error" in witness:
        error = witness["error"].split("\n")[0].rstrip()
        outcome = ["Traceback (most recent call last):", error[:-1] if error.endswith(":") else error]
    else:
        outcome = [f"returned {witness['got']} but expected {witness['expected']}"]
    call = " ".join(witness["call"].split())
    lines = [REPAIR_HEAD, f"# Task: {task['description']}", *attempt.split("\n"), cap(f"# >>> {call}"),
             *[cap(f"# {line}") for line in outcome], REPAIR_ASK, attempt.split("\n")[0]]
    return "\n".join(lines)


def retry_seed(task: dict) -> int | None:
    samples = task["samples"]
    sizes = Counter(shape(s["code"]) for s in samples)
    cands = [i for i, s in enumerate(samples) if s["visible"]["kind"] in ("wrong", "error")
             and attempt_parts(s["code"], task["entry_point"])]
    if not cands:
        return None
    return min(cands, key=lambda i: (RANK[samples[i]["visible"]["kind"]], -sizes[shape(samples[i]["code"])], i))


def vote(samples: list[dict], pool: list[int]) -> int:
    groups: dict[str, list[int]] = {}
    for i in pool:
        groups.setdefault(shape(samples[i]["code"]), []).append(i)
    best = min(groups.values(), key=lambda g: (-len(g), min(len(samples[i]["code"]) for i in g), g[0]))
    return best[0]


def check(items: list[tuple[dict, dict]], workers: int, label: str) -> None:
    todo = []
    for task, s in items:
        if s["nostop"]:
            s["visible"] = {"kind": "nostop"}
        else:
            todo.append((task, s))
    for (task, s), result in parallel(todo, lambda ts: run_visible(ts[0], ts[1]["code"]), workers, label):
        s["visible"] = result


def solve(tasks: list[dict], model, k: int, retries: int, k_retry: int, workers: int) -> None:
    """Fills task["samples"], task["answer"] (an index) and task["source"]."""
    new = []
    for task, outs in zip(tasks, model.sample([t["prompt"] for t in tasks], k)):
        task["samples"] = [first_sample(task, raw, eos) for raw, eos in outs]
        new += [(task, s) for s in task["samples"]]
    check(new, workers, "visible")
    for r in range(1, retries + 1):
        todo = []
        for task in tasks:
            if not task["visible"] or any(s["visible"]["kind"] == "pass" for s in task["samples"]):
                continue
            seed = retry_seed(task)
            if seed is None:
                continue
            context, attempt = attempt_parts(task["samples"][seed]["code"], task["entry_point"])
            prompt = repair_prompt(task, attempt, task["samples"][seed]["visible"]["witness"])
            todo.append((task, seed, context, attempt.split("\n")[0], prompt))
        if not todo:
            break
        new = []
        for (task, seed, context, header, prompt), outs in zip(todo, model.sample([t[4] for t in todo], k_retry)):
            for raw, eos in outs:
                completion = truncate({"set": "humaneval"}, raw)  # a body under the attempt's own header
                code = (context + "\n" if context else "") + header + completion
                s = {"origin": f"retry{r}", "seed": seed, "prompt": prompt, "raw": raw, "eos": eos,
                     "completion": completion, "code": code, "nostop": not eos and completion.rstrip() == raw.rstrip()}
                task["samples"].append(s)
                new.append((task, s))
        check(new, workers, f"retry {r}")
    for task in tasks:
        samples = task["samples"]
        assert len(samples) <= k + retries * k_retry, f"{task['id']}: budget overrun"
        passers = [i for i, s in enumerate(samples) if s["visible"]["kind"] == "pass"]
        if passers:
            task["answer"] = vote(samples, passers)
            origin = samples[task["answer"]]["origin"]
            task["source"] = "vote" if not task["visible"] else ("filter" if origin == "sample" else "retry")
        else:
            task["answer"], task["source"] = vote(samples, list(range(len(samples)))), "none"


def grade_all(tasks: list[dict], workers: int) -> None:
    items = [(t, s) for t in tasks for s in t["samples"] + [t["greedy"]]]
    for (task, s), (passed, why) in parallel(items, lambda ts: (False, "nostop") if ts[1]["nostop"]
                                             else grade(ts[0], ts[1]["code"]), workers, "grade"):
        s["passed"], s["why"] = passed, why


def bootstrap_ci(diffs: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def summarize(tasks: list[dict], k: int) -> dict:
    n = len(tasks)
    runtime = [t["samples"][t["answer"]]["passed"] for t in tasks]
    greedy = [t["greedy"]["passed"] for t in tasks]
    oracle = [any(s["passed"] for s in t["samples"]) for t in tasks]
    first = [[s for s in t["samples"] if s["origin"] == "sample"] for t in tasks]
    sources = Counter(t["source"] for t in tasks)
    return {
        "tasks": n, "greedy_pass1": sum(greedy) / n, "runtime_pass": sum(runtime) / n,
        "oracle_passN": sum(oracle) / n,
        "sample_pass1": sum(pass_at_k(len(f), sum(s["passed"] for s in f), 1) for f in first) / n,
        f"pass_at_{k}": sum(pass_at_k(len(f), sum(s["passed"] for s in f), k) for f in first if len(f) >= k) / n,
        "ci95_runtime_minus_greedy": bootstrap_ci([float(r) - float(g) for r, g in zip(runtime, greedy)]),
        "source": dict(sources), "source_passed": dict(Counter(t["source"] for t, r in zip(tasks, runtime) if r)),
        "vote_only_tasks": sum(not t["visible"] for t in tasks),
        "retried_tasks": sum(any(s["origin"] != "sample" for s in t["samples"]) for t in tasks),
        "samples_mean": sum(len(t["samples"]) for t in tasks) / n,
        "not_evaluated": sum(s["why"] == NOT_EVALUATED for t in tasks for s in t["samples"] + [t["greedy"]]),
        "visible_kinds": dict(Counter(s["visible"]["kind"] for t in tasks for s in t["samples"])),
    }


# ---------------------------------------------------------------- models

class ReferenceModel:
    """--dry-run: every prompt is answered by its task's reference; anything else gets an empty reply."""

    def __init__(self, tasks: list[dict]):
        self.answers = {t["prompt"]: t["reference_raw"] for t in tasks}

    def sample(self, prompts: list[str], n: int) -> list[list[tuple[str, bool]]]:
        return [[(self.answers.get(p, ""), True)] * n for p in prompts]

    def greedy(self, prompts: list[str]) -> list[tuple[str, bool]]:
        return [outs[0] for outs in self.sample(prompts, 1)]


class HFModel:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.args = torch, args
        if args.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gib * (1 << 30) / total), 0)
        torch.manual_seed(args.seed)
        self.tok = AutoTokenizer.from_pretrained(ROOT / args.model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(ROOT / args.model, torch_dtype=dtype).to(args.device).eval()

    def generate(self, prompts: list[str], n: int, sample: bool) -> list[list[tuple[str, bool]]]:
        tok, args = self.tok, self.args
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        per_step = max(1, args.batch // n)  # batch counts generated rows
        out: list = [None] * len(prompts)
        extra = {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p} if sample \
            else {"do_sample": False}
        for start in range(0, len(order), per_step):
            idx = order[start:start + per_step]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(args.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, num_return_sequences=n, max_new_tokens=args.max_new,
                                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, **extra)
            seqs = gen[:, enc["input_ids"].shape[1]:].tolist()
            for slot, i in enumerate(idx):
                out[i] = []
                for seq in seqs[slot * n:(slot + 1) * n]:  # num_return_sequences keeps them adjacent
                    eos = tok.eos_token_id in seq
                    if eos:
                        seq = seq[:seq.index(tok.eos_token_id)]
                    out[i].append((tok.decode(seq, skip_special_tokens=True), eos))
            print(f"[generate n={n}] {min(start + per_step, len(order))}/{len(order)}", flush=True)
        return out

    def sample(self, prompts: list[str], n: int) -> list[list[tuple[str, bool]]]:
        return self.generate(prompts, n, True)

    def greedy(self, prompts: list[str]) -> list[tuple[str, bool]]:
        return [outs[0] for outs in self.generate(prompts, 1, False)]


# ---------------------------------------------------------------- main

def run(tasks: list[dict], model, args) -> dict:
    solve(tasks, model, args.k, args.retries, args.k_retry, args.workers)
    for task, (raw, eos) in zip(tasks, model.greedy([t["prompt"] for t in tasks])):
        task["greedy"] = first_sample(task, raw, eos)
    grade_all(tasks, args.workers)
    return summarize(tasks, args.k)


def split_check(tasks: list[dict], workers: int) -> dict:
    report = {"visible_pass": 0, "hidden_pass": 0, "full_pass": 0, "failures": []}
    items = [(t, part) for t in tasks for part in ("visible", "hidden", "full")]
    for (task, part), (passed, why) in parallel(items, lambda tp: grade(tp[0], tp[0]["reference_code"], tp[1]),
                                                workers, "split"):
        report[f"{part}_pass"] += passed
        if not passed:
            report["failures"].append({"id": task["id"], "tests": part, "why": why})
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("gate", "he", "mbpp"), required=True)
    ap.add_argument("--model", help="HF model dir (not needed with --dry-run)")
    ap.add_argument("--task-ids", default="research/raw/sa_v_clean_gate.json", help="gate id list")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--k-retry", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="reference answers, no model: checks the split")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only")
    ap.add_argument("--workers", type=int, default=4, help="sandbox subprocesses (1 GiB RSS cap each)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0, help="required on cuda: fresh free - 0.75 GiB")
    ap.add_argument("--batch", type=int, default=64, help="generated rows per step")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.k < 1 or args.retries < 0 or args.k_retry < 1 or args.workers < 1:
        ap.error("--k, --k-retry and --workers must be >= 1; --retries >= 0")
    if not args.dry_run and not args.model:
        ap.error("--model is required without --dry-run")
    if not args.dry_run and args.device == "cuda" and args.vram_cap_gib <= 0:
        ap.error("--vram-cap-gib is required on cuda (fresh free VRAM - 0.75 GiB)")
    ram_guard(MIN_FREE_RAM + args.workers * CHILD_RSS_CAP)

    t0 = time.perf_counter()
    he_stats = None
    if args.set == "gate":
        tasks = load_gate(args.task_ids)
    elif args.set == "mbpp":
        tasks = load_mbpp()
    else:
        tasks, he_stats = load_he(args.workers)
    if args.limit:
        tasks = tasks[:args.limit]
    report = {"set": args.set, "tasks": len(tasks), "dry_run": args.dry_run, "he_examples": he_stats,
              "args": {k: str(v) for k, v in vars(args).items()}}
    if args.dry_run:
        report["split"] = split_check(tasks, args.workers)
        args.k, args.retries = 1, 0  # the reference is deterministic: one sample says everything
        report["end_to_end"] = run(tasks, ReferenceModel(tasks), args)
    else:
        report["summary"] = run(tasks, HFModel(args), args)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    report["per_task"] = [{"id": t["id"], "source": t["source"], "answer": t["answer"],
                           "runtime": t["samples"][t["answer"]]["passed"], "greedy": t["greedy"]["passed"],
                           "oracle": any(s["passed"] for s in t["samples"]), "samples": len(t["samples"])}
                          for t in tasks]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with out.with_suffix(".samples.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            for i, s in enumerate(t["samples"]):
                f.write(json.dumps({"id": t["id"], "i": i, **{k: v for k, v in s.items() if k != "prompt"},
                                    "retry_prompt": s.get("prompt")}) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_task", "args")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
