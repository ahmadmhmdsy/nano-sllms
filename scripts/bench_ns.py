"""Never-seen bench (NS, 2026-10-29): item schema, completion -> program, and grading. EVAL ONLY.

One jsonl row per item in data/bench/ns/<half>_<set>.jsonl:
  id, half ("A" web | "B" ours), set, category, kind, prompt, entry_point, reference, visible, hidden, support, meta
`kind` decides how a raw model continuation becomes a program:
  body      prompt + completion, cut at the first top-level line (HumanEval rule); tests are assert strings
  full      prompt + completion, cut MBPP-style (further defs allowed); tests are assert strings
  leetcode  prompt + completion inside `class Solution` (body cut); tests are {"input", "output"} JSON lines
  predict   completion is an output literal (first line only); prompt ends in "assert f(...) =="; no tests
  stdin     completion is a whole program; tests are {"input", "output"}; stdout compared token-wise
`support` is grader-only code (setup statements, a renamed reference), placed after the candidate.
`visible` drives the runtime's filter (sa7: first assert / public cases); `hidden` is everything else.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from bench_passk import GUARD, MARKER, truncate  # noqa: E402
from sa2_build import TIMEOUT, run_guarded  # noqa: E402

NS_DIR = ROOT / "data" / "bench" / "ns"
EXCLUDED = NS_DIR / "decontam_excluded.json"  # {"ids": [...], ...} written from the bench_decontam report
KINDS = ("body", "full", "leetcode", "predict", "stdin")
IO_TIMEOUT = 2 * TIMEOUT  # stdin/leetcode programs run every case in one process

_IO_HELPERS = '''
import io as _ns_io, json as _ns_json, math as _ns_math, sys as _ns_sys
def _ns_same_tok(a, b):
    if a == b:
        return True
    try:
        return _ns_math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except ValueError:
        return False
def _ns_same_text(got, want):
    g, w = got.split(), want.split()
    return len(g) == len(w) and all(_ns_same_tok(x, y) for x, y in zip(g, w))
def _ns_same_value(got, want):
    if isinstance(want, float) or isinstance(got, float):
        try:
            return _ns_math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6)
        except TypeError:
            return False
    if isinstance(want, list) and isinstance(got, (list, tuple)):
        return len(got) == len(want) and all(_ns_same_value(x, y) for x, y in zip(got, want))
    return got == want
'''

_STDIN_RUN = '''
_ns_code = compile(_NS_SRC, "candidate", "exec")
for _ns_inp, _ns_want in _NS_CASES:
    _ns_sys.stdin = _ns_io.TextIOWrapper(_ns_io.BytesIO(_ns_inp.encode("utf-8")), encoding="utf-8")
    _ns_out, _ns_real = _ns_io.StringIO(), _ns_sys.stdout
    _ns_sys.stdout = _ns_out
    try:
        exec(_ns_code, {"__name__": "__main__"})
    except SystemExit:
        pass
    finally:
        _ns_sys.stdout = _ns_real
    assert _ns_same_text(_ns_out.getvalue(), _ns_want), "wrong output"
'''

_LEET_RUN = '''
for _ns_inp, _ns_want in _NS_CASES:
    _ns_args = [_ns_json.loads(_l) for _l in _ns_inp.split("\\n") if _l.strip()]
    assert _ns_same_value(Solution().{entry}(*_ns_args), _ns_json.loads(_ns_want)), "wrong answer"
'''


def load(paths: list[Path] | None = None, keep_flagged: bool = False) -> list[dict]:
    """Bench items; ids in EXCLUDED (13-gram hits on our training corpora) are dropped so a rebuild cannot
    bring them back."""
    excluded = set() if keep_flagged or not EXCLUDED.exists() else \
        set(json.loads(EXCLUDED.read_text(encoding="utf-8"))["ids"])
    items = []
    for path in paths or sorted(p for p in NS_DIR.glob("[AB]_*.jsonl") if p.stem != "B_raw"):
        with path.open(encoding="utf-8") as f:
            items += [row for row in map(json.loads, filter(str.strip, f)) if row["id"] not in excluded]
    return items


def cut(item: dict, raw: str) -> str:
    """Raw continuation -> the completion that belongs to the answer."""
    if item["kind"] == "predict":
        return raw.strip().split("\n", 1)[0].strip()
    if item["kind"] == "stdin":  # a whole script: top-level code after a def is the point, stop at a fence only
        lines = raw.split("\n")
        end = next((i for i, line in enumerate(lines) if line.startswith(("```", '"""'))), len(lines))
        return "\n".join(lines[:end]).rstrip() + "\n"
    return truncate({"set": "humaneval" if item["kind"] in ("body", "leetcode") else "mbpp"}, raw)


def code_of(item: dict, completion: str) -> str:
    if item["kind"] == "stdin":
        return completion
    if item["kind"] == "predict":
        return item["prompt"].rstrip() + " " + completion
    return item["prompt"] + completion


def program(item: dict, code: str, tests: list) -> str:
    kind = item["kind"]
    if kind in ("stdin", "leetcode"):
        cases = [(t["input"], t["output"]) for t in tests]
        head = f"{_IO_HELPERS}\n_NS_CASES = {cases!r}\n"
        if kind == "stdin":
            return f"{GUARD}\n{head}_NS_SRC = {code!r}\n{_STDIN_RUN}\nprint({MARKER!r})\n"
        return (f"{GUARD}\n{code}\n\n{head}{item.get('support', '')}\n"
                f"{_LEET_RUN.format(entry=item['entry_point'])}\nprint({MARKER!r})\n")
    return f"{GUARD}\n{code}\n\n{item.get('support', '')}\n" + "\n".join(tests) + f"\nprint({MARKER!r})\n"


def grade(item: dict, code: str, which: str = "all") -> tuple[bool, str]:
    """(passed, reason) of code against the visible, hidden or all tests."""
    if which == "visible" and item["kind"] == "predict":
        return True, "no-visible"
    tests = {"visible": item["visible"], "hidden": item["hidden"], "all": item["visible"] + item["hidden"]}[which]
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False, "syntax"
    timeout = IO_TIMEOUT if item["kind"] in ("stdin", "leetcode") else TIMEOUT
    passed, reason, _ = run_guarded(program(item, code, tests), timeout)
    return passed, reason


def vote_key(item: dict, completion: str) -> str:
    """Equivalence class for the runtime vote: the output value for predict, the AST otherwise."""
    if item["kind"] == "predict":
        try:
            return repr(ast.literal_eval(completion))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return completion
    try:
        return ast.dump(ast.parse(completion))
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return completion
