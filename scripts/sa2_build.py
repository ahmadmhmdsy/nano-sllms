"""Build the SA-2 SFT set (research/2026-10-22_sa_plan_v2.md §2; user approved A,B,D,E,F, not C, 2026-10-22).

Stages, all on CPU:
1. normalize - opc-sft-stage2 educational_instruct, OpenCodeInstruct (average_test_score 1.0),
               self-oss-instruct and data/sa2/teacher_candidates.jsonl (when present) become
               {instruction, code, entry_point, tests}: stdlib-only, no I/O calls, no classes, 1-40-line bodies.
2. shape     - HumanEval shape (signature + docstring with a >>> example -> body) when the function allows
               it, else MBPP shape ('\"\"\"' instruction + first assert '\"\"\"' -> code); both match bench_passk.
3. verify    - every shaped row runs the way bench_passk runs a completion (GUARD prelude, python -I, temp
               dir) with a timeout and a 1 GiB RSS cap on the child; failures are dropped, never repaired.
4. fix       - own generator: one AST mutation of a verified function, its real failure on the first test it
               breaks, in the playground fix-prompt format; the target is the original body.
5. tool      - xLAM rows as a strict JSON tool-call shape, checked against their own tool schemas.
6. decontam  - 13-gram coverage (bench_decontam rules) against HumanEval(+), MBPP, MBPP sanitized(+) and the
               playground dev/final `wide` tasks (B1/W2/W4), plus exact function-signature matches.

Outputs in data/sa2/: train.jsonl, val.jsonl (1 % by id hash), decontam_dropped.jsonl, report.json, and
verify_cache.jsonl (pass/fail verdicts by program hash, reused by the next build; --no-cache skips it).
Downloaded data is read in-process only; candidate code runs in child processes.

  .venv/Scripts/python.exe -I scripts/sa2_build.py --limit 300      # probe
  .venv/Scripts/python.exe -I scripts/sa2_build.py                  # full build
"""
import argparse
import ast
import copy
import gzip
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
NEST = ROOT / "glm-5.3-flash-from-scratch"
sys.path[:0] = [str(ROOT / "scripts"), str(NEST)]  # -I leaves the script directory off sys.path
from bench_decontam import RAW, load_tasks, ngrams  # noqa: E402
from bench_passk import GUARD, MARKER  # noqa: E402

SRC = ROOT / "data" / "hf" / "sa2"
OUT = ROOT / "data" / "sa2"
MIN_FREE_RAM = 4 << 30  # user decision 2026-10-22: 4 GiB floor; main() fits workers x CHILD_RSS_CAP above it
CHILD_RSS_CAP = 1 << 30
TIMEOUT = 6.0
CHUNK = 512
# Anything that changes a verdict for the same program text; GUARD and MARKER are inside the program.
CACHE_SALT = f"v1|timeout={TIMEOUT}|rss={CHILD_RSS_CAP}|{sys.version}"
N = 13
COVERAGE_DROP = 0.5
SIG_COVERAGE = 0.2
# Candidates run without a filesystem/process sandbox beyond GUARD, so modules that reach the OS,
# the clock or randomness (flaky tests) are rejected outright.
BANNED_MODULES = {"os", "sys", "subprocess", "shutil", "socket", "pathlib", "tempfile", "glob", "signal",
                  "threading", "multiprocessing", "concurrent", "asyncio", "ctypes", "urllib", "http", "ftplib",
                  "smtplib", "webbrowser", "pickle", "shelve", "sqlite3", "io", "builtins", "importlib", "gc",
                  "resource", "time", "random", "secrets", "uuid", "logging", "inspect", "platform", "atexit"}
ALLOWED_MODULES = set(sys.stdlib_module_names) - BANNED_MODULES
BANNED_CALL = re.compile(r"\b(input|open|print|eval|exec|compile|__import__|globals|locals|setattr|delattr|"
                         r"breakpoint|getattr)\s*\(")
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
SIG_RE = re.compile(r"def\s+(\w+)\s*\(([^)]*)\)")
CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add}
WITNESS = "SA2_WITNESS "


# ---------------------------------------------------------------- sources

def parquet_rows(path: Path, columns: list[str]):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
        yield from batch.to_pylist()


def load_opc():
    path = SRC / "OpenCoder-LLM__opc-sft-stage2" / "educational_instruct" / "train-00000-of-00001.parquet"
    for i, row in enumerate(parquet_rows(path, ["instruction", "code", "entry_point", "testcase"])):
        yield {"id": f"opc/{i}", "source": "opc", "instruction": row["instruction"], "code": row["code"],
               "entry_point": row["entry_point"], "tests": list(row["testcase"] or [])}


def load_oci():
    for shard in sorted((SRC / "nvidia__OpenCodeInstruct" / "data").glob("train-*.parquet")):
        tag = shard.stem.split("-")[1]
        for i, row in enumerate(parquet_rows(shard, ["input", "output", "unit_tests", "average_test_score"])):
            try:
                if float(row["average_test_score"]) < 1.0:
                    continue
                tests = json.loads(row["unit_tests"])
            except (TypeError, ValueError):
                continue
            block = CODE_BLOCK.search(row["output"] or "")
            if block and isinstance(tests, list):
                yield {"id": f"oci/{tag}/{i}", "source": "oci", "instruction": row["input"], "code": block.group(1),
                       "entry_point": None, "tests": tests}


def load_selfoss():
    path = SRC / "bigcode__self-oss-instruct-sc2-exec-filter-50k" / "data" / "train-00000-of-00001.parquet"
    for i, row in enumerate(parquet_rows(path, ["instruction", "response"])):
        blocks = CODE_BLOCK.findall(row["response"] or "")
        if len(blocks) >= 2:
            tests = [line for block in blocks[1:] for line in block.split("\n") if line.startswith("assert ")]
            yield {"id": f"selfoss/{i}", "source": "selfoss", "instruction": row["instruction"], "code": blocks[0],
                   "entry_point": None, "tests": tests}


def load_teacher():
    path = OUT / "teacher_candidates.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    yield {**row, "source": "teacher"}


LOADERS = [("opc", load_opc), ("oci", load_oci), ("selfoss", load_selfoss), ("teacher", load_teacher)]


# ---------------------------------------------------------------- normalize + shape

def body_stmts(func: ast.FunctionDef) -> list[ast.stmt]:
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def is_main_guard(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)


def simple_example(test: str, entry: str) -> tuple[str, str] | None:
    """'assert f(x) == y' -> ('f(x)', 'y'); None for any other assert."""
    check = ast.parse(test).body[0].test
    if isinstance(check, ast.Compare) and len(check.ops) == 1 and isinstance(check.ops[0], ast.Eq) \
            and isinstance(check.left, ast.Call) and isinstance(check.left.func, ast.Name) \
            and check.left.func.id == entry:
        return ast.get_source_segment(test, check.left), ast.get_source_segment(test, check.comparators[0])
    return None


def clean(raw: dict) -> tuple[dict | None, str]:
    # OpenCodeInstruct/self-oss statements trail off into markdown Input/Output/Sample sections; the
    # first assert already shows the I/O, so keep the statement only.
    instruction = re.split(r"\n\s*\*\*|```", raw.get("instruction") or "")[0]
    instruction = " ".join(instruction.split())
    if not 20 <= len(instruction) <= 600 or "```" in instruction:
        return None, "instruction"
    code = (raw.get("code") or "").replace("\r\n", "\n")
    code = "\n".join(line for line in code.split("\n") if not line.startswith("from __future__"))
    if not code.strip() or len(code) > 2500 or "\t" in code:
        return None, "size"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None, "syntax"
    imports, defs, assigns = [], [], 0
    for node in tree.body:
        segment = ast.get_source_segment(code, node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                return None, "import"
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(m.split(".")[0] not in ALLOWED_MODULES for m in modules):
                return None, "import"
            imports.append(segment)
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                return None, "decorator"
            defs.append(segment)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            defs.append(segment)
            assigns += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) or is_main_guard(node):
            continue
        elif isinstance(node, ast.ClassDef):
            return None, "class"
        else:
            return None, "toplevel"
    code = "\n".join(imports) + ("\n\n\n" if imports and defs else "") + "\n\n\n".join(defs) + "\n"
    if BANNED_CALL.search(code):
        return None, "io_call"
    funcs = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None, "nofunc"
    tests = []
    for test in raw.get("tests") or []:
        test = test.strip() if isinstance(test, str) else ""
        if not test or "\n" in test or len(test) > 300:
            continue
        try:
            parsed = ast.parse(test).body
        except (SyntaxError, ValueError):
            continue
        if len(parsed) == 1 and isinstance(parsed[0], ast.Assert):
            tests.append(test)
    names = {f.name: f for f in funcs}
    entry = raw.get("entry_point")
    if entry not in names:
        entry = next((f.name for f in reversed(funcs) if any(f"{f.name}(" in t for t in tests)), None)
    tests = [t for t in tests if f"{entry}(" in t][:8]
    if entry is None or not tests:
        return None, "tests"
    if any(f.body[0].col_offset != 4 for f in funcs):
        return None, "indent"
    func = names[entry]
    stmts = body_stmts(func)
    if not stmts or not 1 <= func.end_lineno - stmts[0].lineno + 1 <= 40:
        return None, "length"
    return {"id": raw["id"], "source": raw["source"], "instruction": instruction, "code": code, "imports": imports,
            "n_funcs": len(funcs), "assigns": assigns, "entry_point": entry, "tests": tests}, "ok"


def entry_func(c: dict) -> ast.FunctionDef:
    # Re-parsed on demand: holding ~150k function ASTs at once would cost several GiB of RAM.
    return next(n for n in ast.parse(c["code"]).body if isinstance(n, ast.FunctionDef) and n.name == c["entry_point"])


def shape(c: dict, rng: random.Random, he_frac: float) -> dict:
    lines = c["code"].split("\n")
    func = entry_func(c)
    stmts = body_stmts(func)
    header = lines[func.lineno - 1].rstrip()
    example = next(filter(None, (simple_example(t, c["entry_point"]) for t in c["tests"])), None)
    he_ok = (c["n_funcs"] == 1 and not c["assigns"] and example is not None and header.startswith("def ")
             and header.endswith(":") and stmts[0].lineno > func.lineno and '"""' not in c["instruction"]
             and "\\" not in c["instruction"] and len(example[0]) + len(example[1]) <= 160)
    row = {"id": c["id"], "source": c["source"], "entry_point": c["entry_point"], "tests": c["tests"]}
    if he_ok and rng.random() < he_frac:
        doc = textwrap.fill(c["instruction"], width=76, initial_indent="    ", subsequent_indent="    ").lstrip()
        prefix = "\n".join(c["imports"]) + "\n\n\n" if c["imports"] else ""
        # Ends at the closing quotes, like bench_passk.model_prompt (SmolLM2 merges "\n" with the indent).
        row["prompt"] = f'{prefix}{header}\n    """{doc}\n    >>> {example[0]}\n    {example[1]}\n    """'
        row["completion"] = "\n" + "\n".join(lines[stmts[0].lineno - 1:func.end_lineno]).rstrip() + "\n"
        row["shape"] = "he"
    else:
        row["prompt"] = f'"""\n{c["instruction"]}\n{c["tests"][0]}\n"""'
        row["completion"] = "\n" + c["code"].strip() + "\n"
        row["shape"] = "mbpp"
    return row


# ---------------------------------------------------------------- execution

def run_guarded(code: str, timeout: float = TIMEOUT) -> tuple[bool, str, str]:
    """bench_passk.execute plus an RSS cap on the child: (passed, reason, stdout)."""
    with tempfile.TemporaryDirectory(prefix="sa2_") as td:
        path = Path(td) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        out_path = Path(td) / "out.txt"
        reason = ""
        with out_path.open("wb") as out:
            # The base interpreter: candidates need only the stdlib, and a venv launcher would put
            # the real process one level down, out of reach of the RSS check.
            proc = subprocess.Popen([getattr(sys, "_base_executable", sys.executable), "-I", str(path)], cwd=td,
                                    stdout=out, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    reason = "timeout"
                else:
                    try:
                        if psutil.Process(proc.pid).memory_info().rss > CHILD_RSS_CAP:
                            reason = "rss"
                    except psutil.Error:
                        pass
                if reason:
                    proc.kill()
                    proc.wait()
                    break
                time.sleep(0.01)
        stdout = out_path.read_text(encoding="utf-8", errors="replace")
    if reason:
        return False, reason, stdout
    if proc.returncode == 0 and MARKER in stdout:
        return True, "pass", stdout
    return False, "fail", stdout


class VerifyCache:
    """run_guarded memoised on disk by program hash, so a rebuild re-runs only new or changed programs.

    Timeout and RSS kills are never cached: they depend on host load, not only on the program.
    """

    def __init__(self, path: Path | None):
        self.path, self.hits, self.misses = path, 0, 0
        self.done: dict[str, tuple[bool, str, str]] = {}
        self.lock = threading.Lock()
        if path and path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from an interrupted build
                    self.done[e["k"]] = (e["p"], e["r"], e["o"])

    def __call__(self, code: str) -> tuple[bool, str, str]:
        key = hashlib.sha256(f"{CACHE_SALT}\n{code}".encode("utf-8")).hexdigest()
        with self.lock:
            hit = self.done.get(key)
            if hit:
                self.hits += 1
                return hit
            self.misses += 1
        result = run_guarded(code)
        if self.path and result[1] in ("pass", "fail"):
            with self.lock:
                self.done[key] = result
                with self.path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps({"k": key, "p": result[0], "r": result[1], "o": result[2]}) + "\n")
        return result


def program(row: dict, tests: list[str] | None = None) -> str:
    body = row["completion"] if row["shape"] == "mbpp" else row["prompt"] + row["completion"]
    return f"{GUARD}\n{body}\n\n" + "\n".join(tests or row["tests"]) + f"\nprint({MARKER!r})\n"


def ram_guard(floor: int) -> None:
    free = psutil.virtual_memory().available
    if free < floor:
        raise SystemExit(f"ABORT: host free RAM {free / 2**30:.1f} GiB < {floor / 2**30:.1f} GiB")


def parallel(items: list, fn, workers: int, label: str, floor: int = MIN_FREE_RAM):
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(items), CHUNK):
            ram_guard(floor)
            chunk = items[start:start + CHUNK]
            yield from zip(chunk, pool.map(fn, chunk))
            print(f"[{label}] {min(start + CHUNK, len(items))}/{len(items)}", flush=True)


# ---------------------------------------------------------------- own generator: bug -> failure -> fix

def strip_doc(func: ast.FunctionDef) -> ast.FunctionDef:
    func = copy.deepcopy(func)
    func.body = body_stmts(func) or [ast.Pass()]
    return func


def mutate(func: ast.FunctionDef, rng: random.Random) -> ast.FunctionDef | None:
    """One wrong operator, off-by-one constant, flipped bool or missing return value."""
    sites = []
    for i, node in enumerate(ast.walk(func)):
        if isinstance(node, ast.Compare) and type(node.ops[0]) in CMP_SWAP:
            sites.append((i, "cmp"))
        elif isinstance(node, ast.BinOp) and type(node.op) in BIN_SWAP:
            sites.append((i, "bin"))
        elif isinstance(node, ast.BoolOp):
            sites.append((i, "bool"))
        elif isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 10:
            sites.append((i, "int"))
        elif isinstance(node, ast.Constant) and type(node.value) is bool:
            sites.append((i, "flag"))
        elif isinstance(node, ast.Return) and node.value is not None:
            sites.append((i, "return"))
    if not sites:
        return None
    index, kind = rng.choice(sites)
    clone = copy.deepcopy(func)
    node = list(ast.walk(clone))[index]  # same walk order on an identical tree
    if kind == "cmp":
        node.ops[0] = CMP_SWAP[type(node.ops[0])]()
    elif kind == "bin":
        node.op = BIN_SWAP[type(node.op)]()
    elif kind == "bool":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "int":
        node.value += rng.choice((-1, 1))
    elif kind == "flag":
        node.value = not node.value
    else:
        node.value = None
    return clone


def witness_program(buggy: str, c: dict) -> str:
    """Run the simple asserts one by one and report the first wrong result or exception."""
    parts = [GUARD, "import json as _sa2_json", buggy]
    for test in c["tests"]:
        example = simple_example(test, c["entry_point"])
        if example is None:
            continue
        call, expected = example
        parts.append(
            f"try:\n    _sa2_got = {call}\nexcept Exception as _sa2_error:\n"
            f"    print({WITNESS!r} + _sa2_json.dumps({{'call': {call!r}, "
            f"'error': type(_sa2_error).__name__ + ': ' + str(_sa2_error)}}))\n    raise SystemExit\n"
            f"if not (_sa2_got == ({expected})):\n"
            f"    print({WITNESS!r} + _sa2_json.dumps({{'call': {call!r}, 'got': repr(_sa2_got), "
            f"'expected': repr({expected})}}))\n    raise SystemExit")
    parts.append(f"print({MARKER!r})")
    return "\n".join(parts) + "\n"


def fix_eligible(c: dict) -> bool:
    return (c["n_funcs"] == 1 and not c["imports"] and not c["assigns"]
            and any(simple_example(t, c["entry_point"]) for t in c["tests"]))


def make_fix(c: dict, seed: int, run=run_guarded) -> tuple[dict | None, str]:
    rng = random.Random(f"{seed}:{c['id']}")
    func = entry_func(c)
    good = ast.unparse(strip_doc(func))
    header, *good_body = good.split("\n")
    for _ in range(3):
        mutant = mutate(func, rng)
        if mutant is None:
            return None, "no_site"
        buggy = ast.unparse(strip_doc(mutant))
        if buggy == good:
            continue
        _, _, stdout = run(witness_program(buggy, c))
        found = [line[len(WITNESS):] for line in stdout.splitlines() if line.startswith(WITNESS)]
        if not found:
            continue  # the tests miss this mutation, or it hangs
        witness = json.loads(found[0])
        if "error" in witness:
            outcome = ["Traceback (most recent call last):", witness["error"].split("\n")[0]]
        else:
            outcome = [f"returned {witness['got']} but expected {witness['expected']}"]
        if any(len(line) > 200 for line in outcome):
            continue
        description = textwrap.shorten(re.split(r"(?<=[.!?])\s", c["instruction"])[0], 200, placeholder="...")
        lines = ["# Fix the bug in this Python function.", "# Buggy version:"]
        lines += [f"# {line}" for line in buggy.split("\n")]
        lines.append(f"# >>> {witness['call']}")
        lines += [f"# {line}" for line in outcome]
        lines += [f"# Task: {description}", header]
        row = {"id": f"{c['id']}/fix", "source": f"{c['source']}-fix", "shape": "fix", "entry_point": c["entry_point"],
               "tests": c["tests"], "prompt": "\n".join(lines), "completion": "\n" + "\n".join(good_body) + "\n"}
        passed, reason, _ = run(program(row))
        return (row, "ok") if passed else (None, f"verify_{reason}")
    return None, "no_witness"


# ---------------------------------------------------------------- tool calls (xLAM)

def tool_rows(limit: int, cap: int, rng: random.Random) -> tuple[list[dict], Counter]:
    stats = Counter()
    data = json.loads((SRC / "Salesforce__xlam-function-calling-60k" / "xlam_function_calling_60k.json")
                      .read_text(encoding="utf-8"))
    rng.shuffle(data)
    rows = []
    for raw in data[:limit] if limit else data:
        stats["read"] += 1
        try:
            tools, calls = json.loads(raw["tools"]), json.loads(raw["answers"])
            params = {t["name"]: set((t.get("parameters") or {}).keys()) for t in tools}
            valid = bool(calls) and all(call["name"] in params and set(call["arguments"]) <= params[call["name"]]
                                        for call in calls)
        except (TypeError, KeyError, ValueError, AttributeError):
            valid = False
        if not valid:
            stats["reject_schema"] += 1
            continue
        lines = ["# Tools:"] + [f"# {json.dumps(t, ensure_ascii=False)}" for t in tools]
        lines += [f"# Request: {' '.join(raw['query'].split())}", "# Respond with the tool calls as one JSON list."]
        prompt = "\n".join(lines)
        if len(prompt) > 2500:
            stats["reject_size"] += 1
            continue
        answer = [{"name": call["name"], "arguments": call["arguments"]} for call in calls]
        rows.append({"id": f"xlam/{raw['id']}", "source": "xlam", "shape": "tool", "entry_point": None, "tests": [],
                     "prompt": prompt, "completion": "\n" + json.dumps(answer, ensure_ascii=False) + "\n"})
        stats["kept"] += 1
        if len(rows) >= cap:
            break
    return rows, stats


# ---------------------------------------------------------------- decontamination

def signatures(text: str) -> set[tuple[str, tuple[str, ...]]]:
    out = set()
    for name, args in SIG_RE.findall(text):
        names = tuple(a.split(":")[0].split("=")[0].strip().lstrip("*") for a in args.split(",") if a.strip())
        out.add((name, names))
    return out


def eval_docs() -> list[tuple[str, str, str]]:
    """(set, task_id, text) for every eval task - EVAL ONLY data, used here for overlap checks."""
    docs = [(name, tid, text) for name in ("humaneval", "mbpp", "mbpp-sanitized") for tid, text in load_tasks(name)]
    with (RAW / "humanevalplus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                docs.append(("humaneval+", r["task_id"], r["prompt"] + "\n" + r["canonical_solution"]))
    for r in pq.read_table(RAW / "mbppplus.parquet", columns=["task_id", "prompt", "code"]).to_pylist():
        docs.append(("mbpp+", str(r["task_id"]), r["prompt"] + "\n" + r["code"]))
    from glm53_flash.playground import PLAYGROUND_SEEDS, select_playground_tasks
    for split in PLAYGROUND_SEEDS:
        if split != "train":
            for task in select_playground_tasks(split, per_skill=4, tier="wide"):
                docs.append((f"playground-{split}", task.task_id, task.prompt + task.reference_completion))
    return docs


def decontaminate(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    docs = eval_docs()
    grams = [ngrams(text, N) for _, _, text in docs]
    freq = Counter(g for task in grams for g in task)
    owner, sizes = {}, []
    for k, task in enumerate(grams):
        unique = [g for g in task if freq[g] == 1]  # n-grams shared by 2+ tasks are boilerplate
        sizes.append(len(unique))
        owner.update((g, k) for g in unique)
    bench_sigs = {sig for name, _, text in docs if not name.startswith("playground") for sig in signatures(text)}
    kept, dropped = [], []
    stats = {"eval_tasks": Counter(name for name, _, _ in docs), "dropped_by_set": Counter(),
             "dropped_by_source": Counter(), "rows_with_any_hit": 0, "signature_only_kept": 0}
    for row in rows:
        text = row["prompt"] + row["completion"]
        hits = Counter(owner[g] for g in ngrams(text, N) if g in owner)
        stats["rows_with_any_hit"] += bool(hits)
        worst = max(((h / sizes[k], k) for k, h in hits.items()), default=(0.0, -1))
        sig = signatures(text) & bench_sigs if row["entry_point"] else set()
        # A shared generic signature (is_prime(n), fibonacci(n)) alone is a common concept, not a leak;
        # it drops the row only together with partial n-gram overlap.
        stats["signature_only_kept"] += bool(sig) and worst[0] < SIG_COVERAGE
        sig = sig if worst[0] >= SIG_COVERAGE else set()
        if worst[0] >= COVERAGE_DROP or sig:
            name = docs[worst[1]][0] if worst[0] >= COVERAGE_DROP else "signature"
            stats["dropped_by_set"][name] += 1
            stats["dropped_by_source"][row["source"]] += 1
            dropped.append({"id": row["id"], "source": row["source"], "reason": name,
                            "coverage": round(worst[0], 3),
                            "task": docs[worst[1]][1] if worst[1] >= 0 else None, "signature": sorted(map(str, sig))})
        else:
            kept.append(row)
    stats["eval_tasks"] = dict(stats["eval_tasks"])
    return kept, dropped, stats


# ---------------------------------------------------------------- main

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the SA-2 SFT set.")
    ap.add_argument("--limit", type=int, default=0, help="raw rows read per source (probe); 0 = all")
    ap.add_argument("--cap", type=int, default=60000, help="clean candidates kept per source")
    ap.add_argument("--tool-cap", type=int, default=15000)
    ap.add_argument("--fix-frac", type=float, default=0.35, help="share of eligible verified functions mutated")
    ap.add_argument("--he-frac", type=float, default=0.6, help="share of HE-eligible rows given the HE shape")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel child runs; lowered at start so workers x child RSS cap fits above the RAM floor")
    ap.add_argument("--min-free-ram-gib", type=float, default=MIN_FREE_RAM / 2**30,
                    help="abort below this host free RAM (GiB)")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write <out>/verify_cache.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # invalid escapes in scraped code
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    began = time.time()
    floor = int(args.min_free_ram_gib * 2**30)
    ram_guard(floor)
    fit = max(1, (psutil.virtual_memory().available - floor) // CHILD_RSS_CAP)
    if args.workers > fit:
        print(f"[ram] workers {args.workers} -> {fit} (worst case {fit} x 1 GiB children above the floor)", flush=True)
        args.workers = fit
    cache = VerifyCache(None if args.no_cache else args.out / "verify_cache.jsonl")
    print(f"[cache] {len(cache.done)} cached verdicts", flush=True)
    report = {"args": {k: str(v) for k, v in vars(args).items()}, "sources": {}}

    rows, items, seen = [], {}, set()
    for name, loader in LOADERS:
        stats = report["sources"].setdefault(name, Counter())
        for raw in loader():
            if args.limit and stats["read"] >= args.limit or stats["clean"] >= args.cap:
                break
            stats["read"] += 1
            c, why = clean(raw)
            if c is None:
                stats[f"reject_{why}"] += 1
                continue
            key = hashlib.sha1(" ".join(c["code"].split()).encode("utf-8")).hexdigest()
            if key in seen:
                stats["reject_duplicate"] += 1
                continue
            seen.add(key)
            stats["clean"] += 1
            row = shape(c, rng, args.he_frac)
            stats[f"shape_{row['shape']}"] += 1
            rows.append(row)
            items[row["id"]] = c
        print(f"[normalize] {name}: {dict(stats)}", flush=True)

    verified = []
    for row, (passed, reason, _) in parallel(rows, lambda r: cache(program(r)), args.workers, "verify", floor):
        report["sources"][row["source"]][f"verify_{reason}"] += 1
        if passed:
            verified.append(row)

    fix_rows = []
    candidates = [items[r["id"]] for r in verified if fix_eligible(items[r["id"]]) and rng.random() < args.fix_frac]
    for c, (row, reason) in parallel(candidates, lambda c: make_fix(c, args.seed, cache), args.workers, "fix",
                                    floor):
        report["sources"].setdefault(f"{c['source']}-fix", Counter())[reason] += 1
        if row:
            fix_rows.append(row)

    tools, tool_stats = tool_rows(args.limit, args.tool_cap, rng)
    report["sources"]["xlam"] = tool_stats

    kept, dropped, report["decontam"] = decontaminate(verified + fix_rows + tools)
    rng.shuffle(kept)
    val = [r for r in kept if int(hashlib.sha1(r["id"].encode("utf-8")).hexdigest(), 16) % 100 == 0]
    train = [r for r in kept if int(hashlib.sha1(r["id"].encode("utf-8")).hexdigest(), 16) % 100 != 0]
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)
    write_jsonl(args.out / "decontam_dropped.jsonl", dropped)

    shapes = Counter(r["shape"] for r in kept)
    lines = {s: statistics.median(len([l for l in r["completion"].split("\n") if l.strip()])
                                  for r in kept if r["shape"] == s) for s in shapes}
    code_rows = sum(n for s, n in shapes.items() if s != "tool")
    report.update({
        "final": {"train": len(train), "val": len(val), "by_shape": dict(shapes),
                  "by_source": dict(Counter(r["source"] for r in kept)), "median_completion_lines": lines},
        "gate": {"verified_rows": len(kept), "code_rows": code_rows, "need": 30000, "pass": len(kept) >= 30000,
                 "decontam_dropped": len(dropped)},
        "seconds": round(time.time() - began),
        "cache": {"hits": cache.hits, "misses": cache.misses, "workers": args.workers},
    })
    report["sources"] = {k: dict(v) for k, v in report["sources"].items()}
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=dict), encoding="utf-8")
    print(json.dumps({"final": report["final"], "gate": report["gate"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
