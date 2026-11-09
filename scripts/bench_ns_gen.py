"""Never-seen bench, half B (2026-10-29): our own items, written by a free Gemini model, checked in the sandbox.

Six categories, --per-category items each (default 150), every one in a bench_ns kind:
  write_fn   description + one example assert in a module docstring -> the whole function     kind full
  body       signature + docstring -> body (HumanEval shape)                                   kind body
  general    signature + docstring, stdlib/idiom topics (collections, itertools, re, ...)      kind body
  fix        buggy function + the error one call raises -> the corrected function             kind full
  predict    def f + an input -> the output literal (CRUXEval-O shape)                         kind predict
  tool_call  tool stubs + a user request -> a handler body that calls the tools               kind body
Nothing Gemini claims is trusted: references must pass and a stub must fail; predict outputs are computed by
running f; the fix error is the one the buggy code really raises; tool-call expectations are the calls the
reference really makes (recorded with defaults applied). Stdlib-only, deterministic code only.

  phase gen   ask Gemini (free allowlist, own cache data/teacher/bench_ns_cache.jsonl), append to B_raw.jsonl;
              resumable, stops at --over x per-category raw items or when every free model is spent
  phase build validate B_raw.jsonl -> data/bench/ns/B_<category>.jsonl (seeded, capped at --per-category)

  .venv/Scripts/python.exe -I scripts/bench_ns_gen.py --phase gen
  .venv/Scripts/python.exe -I scripts/bench_ns_gen.py --phase build
"""
import argparse
import ast
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from bench_ns import NS_DIR, code_of, grade  # noqa: E402
from bench_passk import GUARD, MARKER  # noqa: E402
from sa2_build import CHILD_RSS_CAP, MIN_FREE_RAM, parallel, ram_guard, run_guarded  # noqa: E402
from teacher_client import GEMINI_FREE_MODELS, ModelExhausted, Teacher, TeacherAbort  # noqa: E402

RAW_PATH = NS_DIR / "B_raw.jsonl"
CACHE = ROOT / "data" / "teacher" / "bench_ns_cache.jsonl"
CATEGORIES = ("write_fn", "body", "general", "fix", "predict", "tool_call")
PER_CALL = 5
NONDETERMINISTIC = ("random", "time", "datetime.now", "uuid", "os", "sys", "threading", "subprocess", "socket",
                    "input(", "open(", "id(", "hash(")
DOMAINS = ("inventory records", "text statistics", "calendar arithmetic", "grid games", "bank ledgers",
           "sensor readings", "student grades", "shopping carts", "train timetables", "chess boards",
           "recipe scaling", "log lines", "URL paths", "version strings", "playlists", "weather data",
           "matrix math", "string encoding", "set operations", "interval merging", "graph neighbours",
           "parking lots", "library loans", "temperature units", "polynomials", "time durations",
           "word puzzles", "budget splitting", "elevator moves", "file sizes", "colour codes", "queues",
           "tournament scores", "geometry", "binary numbers", "postal codes", "seat booking", "tax brackets",
           "run-length data", "nested dictionaries")
STDLIB_TOPICS = ("collections.Counter", "collections.defaultdict", "collections.deque", "itertools", "functools",
                 "re", "string methods", "f-strings and format specs", "sorted with key", "dict/set comprehensions",
                 "zip and enumerate", "heapq", "bisect", "math", "fractions", "decimal", "dataclasses",
                 "textwrap", "json", "statistics", "operator", "enum", "generators", "exceptions")
COMMON = ("Rules: original problems (no FizzBuzz, Fibonacci, palindrome, prime, anagram or other textbook "
          "classics); Python 3 standard library only; deterministic (no randomness, time, I/O, input()); "
          "each solution 4-25 lines and solvable by a careful beginner; distinct snake_case function names. "
          "Reply with JSON only: {\"items\": [...]} holding exactly %d objects." % PER_CALL)
ASKS = {
    "write_fn": ("Write %d Python function tasks about: {topics}. Each object: \"description\" (2-4 plain "
                 "sentences naming the function and its parameters), \"function\" (the full solution), "
                 "\"tests\" (5 assert lines that call it, covering edge cases)."),
    "body": ("Write %d HumanEval-style Python tasks about: {topics}. Each object: \"function\" (complete "
             "solution: typed signature, a docstring explaining the task with one >>> example, then the body), "
             "\"tests\" (5 assert lines, covering edge cases)."),
    "general": ("Write %d Python tasks that each need idiomatic use of: {topics}. Each object: \"function\" "
                "(complete solution: typed signature, a docstring explaining the task with one >>> example, "
                "then the body; any import is a body line AFTER the docstring), \"tests\" (5 assert lines)."),
    "fix": ("Write %d Python debugging tasks about: {topics}. Each object: \"fixed\" (a correct function), "
            "\"buggy\" (the same function with ONE realistic bug: off-by-one, wrong operator, wrong variable, "
            "missing case, wrong return), \"tests\" (5 assert lines that the fixed version passes and at "
            "least one of which the buggy version fails or raises on)."),
    "predict": ("Write %d short Python functions named f about: {topics}, for an output-prediction quiz. Each "
                "object: \"code\" (def f(...) with 3-10 lines of loops/branches/string or list operations, "
                "no prints, returns a value whose repr is under 80 characters), \"input\" (exactly the text that "
                "goes between the parentheses of the call f(...), e.g. \"[3, 1, 2], 2\" for two arguments; do "
                "not wrap the arguments in an extra list)."),
    "tool_call": ("Write %d tool-use tasks about: {topics}. Each object: \"tools\" (Python source for 2-3 "
                  "stub functions with typed parameters, some with defaults, a one-line docstring, body `...`), "
                  "\"request\" (one sentence a user would say; every argument value must be stated in it "
                  "literally), \"solution\" (the body of def handle_request(): 1-3 lines that call the tools "
                  "with the right arguments, indented 4 spaces)."),
}


def ask(teachers: list, state: dict, category: str, rng: random.Random) -> list[dict] | None:
    pool = STDLIB_TOPICS if category == "general" else DOMAINS
    topics = ", ".join(rng.sample(pool, 3))
    prompt = (ASKS[category] % PER_CALL).format(topics=topics) + "\n" + COMMON
    while True:
        teacher = teachers[state["current"]]
        try:
            reply = teacher.chat([{"role": "user", "content": prompt}], max_tokens=6000, temperature=0.9,
                                 json_mode=True)
            break
        except ModelExhausted as error:
            print(f"NEXT MODEL: {error}", flush=True)
            state["current"] += 1
            if state["current"] == len(teachers):
                raise TeacherAbort("every free model is exhausted for today") from None
    try:
        items = json.loads(reply)["items"]
    except (ValueError, KeyError, TypeError):
        return None
    return [{"category": category, "topics": topics, "model": teacher.model, **it} for it in items
            if isinstance(it, dict)]


def gen(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    have = Counter()
    if RAW_PATH.exists():
        with RAW_PATH.open(encoding="utf-8") as f:
            have = Counter(json.loads(line)["category"] for line in f if line.strip())
    teachers = [Teacher(m, cache_path=CACHE) for m in GEMINI_FREE_MODELS]
    state = {"current": 0}
    target = int(args.per_category * args.over)
    with RAW_PATH.open("a", encoding="utf-8", newline="\n") as out:
        try:
            for category in CATEGORIES:
                rng = random.Random(f"{args.seed}/{category}/{have[category]}")
                fails = 0
                while have[category] < target and fails < 10:
                    items = ask(teachers, state, category, rng)
                    if not items:
                        fails += 1
                        continue
                    for it in items:
                        out.write(json.dumps(it, ensure_ascii=False) + "\n")
                    out.flush()
                    have[category] += len(items)
                    print(f"[gen] {category} {have[category]}/{target}", flush=True)
        except TeacherAbort as error:
            print(f"STOP: {error}", flush=True)
    print(json.dumps({"raw": dict(have), "requests": {t.model: t.calls for t in teachers if t.calls}}), flush=True)


# ---------------------------------------------------------------- build: sandbox-checked items

def as_source(value) -> str:
    """Gemini sometimes returns a list of definitions where one source string was asked for."""
    return "\n\n\n".join(map(str, value)) if isinstance(value, list) else str(value)


def stdlib_only(src: str) -> bool:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return False
    mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    mods += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    return all(m.split(".")[0] in sys.stdlib_module_names for m in mods)


def deterministic(src: str) -> bool:
    return not any(re.search(rf"\b{re.escape(w)}", src) for w in NONDETERMINISTIC)


def one_function(src: str) -> ast.FunctionDef | None:
    try:
        fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)]
    except (SyntaxError, ValueError):
        return None
    return fns[0] if len(fns) == 1 else None


def asserts(tests) -> list[str]:
    if not isinstance(tests, list):
        return []
    keep = []
    for t in tests:
        try:
            if isinstance(t, str) and isinstance(ast.parse(t.strip()).body[0], ast.Assert):
                keep.append(t.strip())
        except (SyntaxError, ValueError, IndexError):
            continue
    return keep


def capture(src: str) -> str | None:
    """stdout of a guarded run, or None if it failed."""
    passed, _, stdout = run_guarded(f"{GUARD}\n{src}\nprint({MARKER!r})\n")
    return stdout if passed else None


def new_item(raw: dict, idx: int, kind: str, prompt: str, entry: str, reference: str, visible: list, hidden: list,
             support: str = "") -> dict:
    cat = raw["category"]
    return {"id": f"B/{cat}/{idx}", "half": "B", "set": cat, "category": cat, "kind": kind, "prompt": prompt,
            "entry_point": entry, "reference": reference, "visible": visible, "hidden": hidden,
            "support": support, "meta": {"topics": raw.get("topics"), "model": raw.get("model")}}


def build_code(raw: dict, idx: int) -> tuple[dict | None, str]:
    src = raw.get("function", "")
    fn = one_function(src)
    tests = asserts(raw.get("tests"))
    if fn is None or len(tests) < 3:
        return None, "shape"
    lines = src.rstrip().split("\n")
    if raw["category"] == "write_fn":
        desc = str(raw.get("description", "")).strip()
        if not desc:
            return None, "shape"
        prompt = f'"""\n{desc}\n\nFor example:\n{tests[0]}\n"""\n'
        reference = src.rstrip() + "\n"
        kind = "full"
    else:
        doc = fn.body[0] if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(
            getattr(fn.body[0], "value", None), ast.Constant) and isinstance(fn.body[0].value.value, str) else None
        if doc is None or len(fn.body) < 2:
            return None, "no-docstring"
        prompt = "\n".join(lines[:doc.end_lineno]) + "\n"
        reference = "\n".join(lines[doc.end_lineno:]) + "\n"
        kind = "body"
    return new_item(raw, idx, kind, prompt, fn.name, reference, tests[:1], tests[1:]), ""


def build_fix(raw: dict, idx: int) -> tuple[dict | None, str]:
    fixed, buggy = raw.get("fixed", ""), raw.get("buggy", "")
    f1, f2 = one_function(fixed), one_function(buggy)
    tests = asserts(raw.get("tests"))
    if f1 is None or f2 is None or f1.name != f2.name or len(tests) < 3 or fixed.strip() == buggy.strip():
        return None, "shape"
    probe = "\n".join(f"try:\n    {t}\nexcept Exception as _e:\n    print('FAIL', {i}, type(_e).__name__, "
                      f"str(_e)[:120].replace(chr(10), ' '))\n    print({MARKER!r})\n    raise SystemExit(0)"
                      for i, t in enumerate(tests))  # capture() needs MARKER, so print it before the early exit
    out = capture(f"{buggy}\n{probe}")
    m = re.search(r"^FAIL (\d+) (\w+) ?(.*)$", out or "", flags=re.M)
    if not m:
        return None, "buggy-passes"
    i, err, msg = int(m.group(1)), m.group(2), m.group(3).strip()
    error = f"{err}: {msg}" if msg else err
    prompt = (f"# This function has a bug. This call fails:\n#   {tests[i]}\n#   -> {error}\n"
              f"{buggy.rstrip()}\n\n\n# Corrected version:\n")
    return new_item(raw, idx, "full", prompt, f1.name, fixed.rstrip() + "\n", [tests[i]],
                    tests[:i] + tests[i + 1:]), ""


def build_predict(raw: dict, idx: int) -> tuple[dict | None, str]:
    code, inp = str(raw.get("code", "")).rstrip(), str(raw.get("input", "")).strip()
    fn = one_function(code)
    if fn is None or not inp:
        return None, "shape"
    if fn.name != "f":  # the quiz always calls f
        code = re.sub(rf"\b{fn.name}\(", "f(", code)
        fn = one_function(code)
    try:
        ast.parse(f"f({inp})")
    except SyntaxError:
        return None, "shape"
    out = capture(f"{code}\nprint('OUT=' + repr(f({inp})))")
    m = re.search(r"^OUT=(.*)$", out or "", flags=re.M)
    if not m or len(m.group(1)) > 120:
        return None, "no-output"
    try:
        ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None, "non-literal"
    return new_item(raw, idx, "predict", f"{code}\nassert f({inp}) ==", "f", m.group(1), [], []), ""


RECORDER = '''
import inspect as _ns_inspect
_NS_CALLS = []
def _ns_record(_ns_name, _ns_stub):
    _ns_sig = _ns_inspect.signature(_ns_stub)
    def _ns_tool(*a, **k):
        _ns_b = _ns_sig.bind(*a, **k)
        _ns_b.apply_defaults()
        _NS_CALLS.append((_ns_name, dict(_ns_b.arguments)))
    return _ns_tool
for _ns_name in {names!r}:
    globals()[_ns_name] = _ns_record(_ns_name, globals()[_ns_name])
def _ns_calls():
    _NS_CALLS.clear()
    handle_request()
    return list(_NS_CALLS)
'''


def build_tool(raw: dict, idx: int) -> tuple[dict | None, str]:
    tools, request, solution = (as_source(raw.get(k, "")) for k in ("tools", "request", "solution"))
    try:
        names = [n.name for n in ast.parse(tools).body if isinstance(n, ast.FunctionDef)]
    except (SyntaxError, ValueError):
        return None, "shape"
    if not 2 <= len(names) <= 4 or not request.strip() or not solution.strip():
        return None, "shape"
    body = "\n".join("    " + line.strip() if line.strip() and not line.startswith("    ") else line
                     for line in solution.rstrip().split("\n")) + "\n"
    prompt = f'{tools.rstrip()}\n\n\ndef handle_request():\n    """User: {request.strip()}"""\n'
    support = RECORDER.format(names=names)
    out = capture(f"{prompt}{body}\n{support}\nprint('CALLS=' + repr(_ns_calls()))")
    m = re.search(r"^CALLS=(.*)$", out or "", flags=re.M)
    if not m or m.group(1) == "[]":
        return None, "no-calls"
    return new_item(raw, idx, "body", prompt, "handle_request", body, [],
                    [f"assert _ns_calls() == {m.group(1)}"], support=support), ""


BUILDERS = {"write_fn": build_code, "body": build_code, "general": build_code, "fix": build_fix,
            "predict": build_predict, "tool_call": build_tool}


def check(pair) -> tuple[dict | None, str]:
    idx, raw = pair
    src = "\n\n".join(as_source(raw.get(k, "")) for k in ("function", "fixed", "buggy", "code", "tools", "solution"))
    if not stdlib_only(src.replace("...", "pass")) and raw["category"] != "tool_call":
        return None, "non-stdlib"
    if not deterministic(src):
        return None, "nondeterministic"
    it, why = BUILDERS[raw["category"]](raw, idx)
    if it is None:
        return None, why
    if it["kind"] == "predict":
        ok = grade(it, code_of(it, it["reference"]))[0]
        bad = grade(it, code_of(it, "object()"))[0]
    else:
        ok = grade(it, code_of(it, it["reference"]))[0]
        stub = "    pass\n" if it["kind"] == "body" else f"def {it['entry_point']}(*args, **kwargs):\n    pass\n"
        bad = grade(it, code_of(it, stub))[0]
    return (it, "ok") if ok and not bad else (None, "ref-fail" if not ok else "stub-pass")


def build(args) -> None:
    ram_guard(MIN_FREE_RAM + args.workers * CHILD_RSS_CAP)
    with RAW_PATH.open(encoding="utf-8") as f:
        raws = [json.loads(line) for line in f if line.strip()]
    summary = {}
    for category in CATEGORIES:
        pairs = [(i, r) for i, r in enumerate(raws) if r["category"] == category]
        why, keep, seen = Counter(), [], set()
        for _, (it, verdict) in parallel(pairs, check, args.workers, f"check {category}"):
            if it is not None and it["prompt"] in seen:
                verdict, it = "duplicate", None
            why[verdict] += 1
            if it is not None:
                seen.add(it["prompt"])
                keep.append(it)
        random.Random(f"{args.seed}/{category}").shuffle(keep)
        keep = sorted(keep[:args.per_category], key=lambda it: int(it["id"].rsplit("/", 1)[1]))
        with (NS_DIR / f"B_{category}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for it in keep:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        summary[category] = {"raw": len(pairs), "kept": len(keep), "verdicts": dict(why)}
        print(f"[{category}] {summary[category]}", flush=True)
    (NS_DIR / "B_build.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                     "sets": summary}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("gen", "build"), required=True)
    ap.add_argument("--per-category", type=int, default=150)
    ap.add_argument("--over", type=float, default=1.6, help="raw items to request per kept item (gen)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    NS_DIR.mkdir(parents=True, exist_ok=True)
    gen(args) if args.phase == "gen" else build(args)


if __name__ == "__main__":
    main()
