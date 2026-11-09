"""SA-7 agent loop eval (research/2026-10-27_sa7_runtime_proposal.md §3 step 3): tool calls in a sandboxed workspace.

The workspace holds solution.py (writable) and test_solution.py (the task's visible asserts, read-only). Tools, in
the SA-4 tool-prompt layout so the model sees the format it was trained on:
  read_file(path) | write_file(path, content) solution.py only | run_python(code) runs solution.py then code |
  run_tests() runs every visible assert against solution.py and prints PASS/FAIL lines and "k/N passed".
Each turn the model writes one JSON list of calls under the tool_grammar decoder (greedy; a reply that never closes
the list within --max-new tokens ends the episode as "invalid", its tail kept as invalid_tail in the samples). With
--schema (default) the grammar also fixes argument keys, string values and the read/write paths (PATHS). Calls run in order; their outputs (tail-capped)
come back as `# Result:` lines in the next prompt, which keeps the last HISTORY turns. The episode ends when
run_tests reports N/N ("tests_pass"; a later write_file in the same reply clears it) or after --max-turns.

Two modes over clean-gate tasks that have visible tests (research/raw/sa_v_clean_gate.json, shuffled with seed 0):
  fix       solution.py starts as the lowest-index k=10 sample of the incumbent (scratch/sa_e_gate_k10) that
            fails the visible tests with a wrong value or an exception: a real failure, not a synthetic bug.
  implement solution.py starts as the signature (and docstring) with `raise NotImplementedError`.
The final solution.py is graded on visible, hidden and full tests (sa7_runtime.grade). The chosen ids go to
--ids-out and are reused when that file exists. --oracle needs no model: it writes the reference and runs the
tests, so every episode must end tests_pass in one turn with full tests passing.

  .venv/Scripts/python.exe -I scripts/sa7_agent.py --oracle --out research/raw/sa7_agent_dry.json
  .venv/Scripts/python.exe -I scripts/sa7_agent.py --model runs/sa4_scale075 --vram-cap-gib 6.4 \
      --out research/raw/sa7_agent.json
"""
import argparse
import ast
import copy
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
import sa7_runtime as rt  # noqa: E402
from sa2_build import CHILD_RSS_CAP, GUARD, MARKER, MIN_FREE_RAM, ram_guard, run_guarded  # noqa: E402
from sa5_rft import KILLED  # noqa: E402

TOOLS = [
    {"name": "read_file", "description": "Read a file from the workspace.",
     "parameters": {"path": {"description": "File name, e.g. solution.py.", "type": "str"}}},
    {"name": "write_file", "description": "Replace the whole content of solution.py.",
     "parameters": {"path": {"description": "Must be solution.py.", "type": "str"},
                    "content": {"description": "The complete new file content.", "type": "str"}}},
    {"name": "run_python", "description": "Run solution.py, then this code in the same namespace; returns the output.",
     "parameters": {"code": {"description": "Python code to run after solution.py.", "type": "str"}}},
    {"name": "run_tests", "description": "Run the tests in test_solution.py against solution.py.", "parameters": {}},
]
PARAMS = {t["name"]: set(t["parameters"]) for t in TOOLS}
# --schema: the grammar also fixes each tool's argument keys, string values and these paths
PATHS = {("read_file", "path"): ("solution.py", "test_solution.py"), ("write_file", "path"): ("solution.py",)}
HEADER = "\n".join(["# Tools:"] + [f"# {json.dumps(t)}" for t in TOOLS])
RESPOND = "# Respond with the tool calls as one JSON list."
HISTORY = 3  # turns kept in the prompt
FILES = ("solution.py", "test_solution.py")
LOADED = "_sa7_ns = {'__name__': 'solution'}"


# ---------------------------------------------------------------- workspace and tools

def start_episode(task: dict, mode: str, start: str) -> dict:
    return {"task": task, "mode": mode, "start": start,
            "files": {"solution.py": start, "test_solution.py": "\n".join(task["visible"]) + "\n"},
            "turns": [], "stop": None, "tool_calls": 0, "tool_errors": 0, "ran_tests": False, "green": False}


def run_output(prog: str) -> str:
    _, why, stdout = run_guarded(prog)
    if why in KILLED:
        return f"{why}: the run was stopped ({'time limit' if why == 'timeout' else 'memory limit'})"
    lines = [line for line in stdout.rstrip("\n").split("\n") if line != MARKER]
    out = "\n".join(lines).strip("\n")
    return out[-rt.OUTCOME_CAP:] if len(out) > rt.OUTCOME_CAP else out


def python_program(solution: str, code: str) -> str:
    return (f"{GUARD}\nimport traceback as _sa7_tb\n{LOADED}\n"
            f"try:\n    exec(compile({solution!r}, 'solution.py', 'exec'), _sa7_ns)\n"
            f"    exec(compile({code!r}, '<run_python>', 'exec'), _sa7_ns)\n"
            f"except BaseException as _sa7_e:\n    print('Traceback (most recent call last):')\n"
            f"    for _sa7_f in _sa7_tb.extract_tb(_sa7_e.__traceback__):\n"
            f"        if _sa7_f.filename in ('solution.py', '<run_python>'):\n"
            f"            print(f'  File \"{{_sa7_f.filename}}\", line {{_sa7_f.lineno}}, in {{_sa7_f.name}}')\n"
            f"    print(type(_sa7_e).__name__ + ': ' + str(_sa7_e))\nprint({MARKER!r})\n")


def tests_program(task: dict, solution: str) -> str:
    n = len(task["visible"])
    parts = [GUARD, "class _Sa7Wrong(Exception):\n    pass", LOADED, "_sa7_ns['_Sa7Wrong'] = _Sa7Wrong", "_sa7_k = 0",
             f"try:\n    exec(compile({solution!r}, 'solution.py', 'exec'), _sa7_ns)\n    _sa7_ok = True\n"
             f"except BaseException as _sa7_e:\n    _sa7_ok = False\n"
             f"    print('solution.py failed to load: ' + type(_sa7_e).__name__ + ': ' + str(_sa7_e))"]
    for test in task["visible"]:
        example = rt.safe_example(test, task["entry_point"])
        if example is None:
            snippet = test
        else:  # a simple `assert call == value` reports the value it got
            call, expected = example
            snippet = (f"_sa7_got = {call}\nif not (_sa7_got == ({expected})):\n"
                       f"    raise _Sa7Wrong(repr(_sa7_got), repr({expected}))")
        parts.append(f"if _sa7_ok:\n    try:\n        exec(compile({snippet!r}, 'test_solution.py', 'exec'), _sa7_ns)\n"
                     f"        print('PASS: ' + {test!r})\n        _sa7_k += 1\n"
                     f"    except _Sa7Wrong as _sa7_w:\n"
                     f"        print('FAIL: ' + {test!r} + ' -> returned ' + _sa7_w.args[0] + ' but expected ' "
                     f"+ _sa7_w.args[1])\n"
                     f"    except BaseException as _sa7_e:\n"
                     f"        print('FAIL: ' + {test!r} + ' -> ' + type(_sa7_e).__name__ + ': ' + str(_sa7_e))")
    parts.append(f"print(f'{{_sa7_k}}/{n} passed')\nprint({MARKER!r})")
    return "\n".join(parts) + "\n"


def call_tool(ep: dict, name: str, args: dict) -> str:
    """One tool call; returns its output. Bad arguments are reported, not raised."""
    if name not in PARAMS:
        return f"error: unknown tool {name}"
    if set(args) != PARAMS[name] or not all(isinstance(v, str) for v in args.values()):
        want = ", ".join(sorted(PARAMS[name])) or "no arguments"
        return f"error: {name} takes {want} (all strings)"
    files = ep["files"]
    if name == "read_file":
        return files[args["path"]] if args["path"] in files else f"error: no such file: {args['path']}"
    if name == "write_file":
        if args["path"] != "solution.py":
            return "error: only solution.py is writable"
        files["solution.py"] = args["content"]
        ep["green"] = False
        return f"wrote {len(args['content'])} chars to solution.py"
    if name == "run_python":
        return run_output(python_program(files["solution.py"], args["code"])) or "(no output)"
    ep["ran_tests"] = True
    out = run_output(tests_program(ep["task"], files["solution.py"]))
    n = len(ep["task"]["visible"])
    ep["green"] = out.split("\n")[-1] == f"{n}/{n} passed"
    return out


def step(ep: dict, reply: str) -> None:
    """Run one reply (a grammar-valid JSON list of calls) and record the turn."""
    observations = []
    for call in json.loads(reply):
        out = call_tool(ep, call["name"], call["arguments"])
        ep["tool_calls"] += 1
        ep["tool_errors"] += out.startswith("error: ")
        observations.append(f"[{call['name']}] {out}")
    ep["turns"].append({"reply": reply, "observations": observations})
    if ep["green"]:
        ep["stop"] = "tests_pass"


def prompt(ep: dict) -> str:
    task = ep["task"]
    if ep["mode"] == "fix":
        request = "Fix solution.py so the tests in test_solution.py pass."
    else:
        request = f"Implement {task['entry_point']} in solution.py so the tests in test_solution.py pass."
    lines = [HEADER, f"# Request: {request} Task: {task['description']} Files: {', '.join(FILES)}."]
    omitted = len(ep["turns"]) - HISTORY
    if omitted > 0:
        lines.append(f"# ({omitted} earlier turns omitted)")
    for turn in ep["turns"][-HISTORY:]:
        lines += [rt.cap(f"# Called: {turn['reply']}"), "# Result:"]
        lines += [rt.cap(f"# {line}") for obs in turn["observations"] for line in obs.split("\n")]
    lines.append(RESPOND)
    return "\n".join(lines)


def run_episodes(episodes: list[dict], agent, max_turns: int, workers: int) -> None:
    """Lockstep: one batched agent call per turn over the still-active episodes, then their tool calls in parallel."""
    for turn in range(max_turns):
        active = [ep for ep in episodes if not ep["stop"]]
        if not active:
            break
        replies = agent.act(active)
        todo = []
        for ep, reply in zip(active, replies):
            if reply is None:
                if not ep["stop"]:
                    ep["stop"] = "invalid"
                    ep["invalid_tail"] = getattr(agent, "unclosed", {}).get(id(ep))
            else:
                todo.append((ep, reply))
        for _ in rt.parallel(todo, lambda er: step(*er), workers, f"turn {turn + 1}"):
            pass
    for ep in episodes:
        ep["stop"] = ep["stop"] or "turns"


# ---------------------------------------------------------------- agents

class OracleAgent:
    """--oracle: writes the reference, then runs the tests."""

    def act(self, episodes: list[dict]) -> list[str]:
        return [json.dumps([{"name": "write_file", "arguments": {"path": "solution.py",
                                                                 "content": ep["task"]["reference_code"]}},
                            {"name": "run_tests", "arguments": {}}]) for ep in episodes]


class HFAgent:
    """Greedy decoding under the tool-call grammar (tool_grammar.ToolCallGrammar), --batch prompts per step."""

    def __init__(self, args):
        from tool_grammar import token_pieces

        self.lm, self.args = rt.HFModel(args), args
        self.pieces = token_pieces(self.lm.tok, self.lm.model.config.vocab_size)
        self.limit = self.lm.model.config.max_position_embeddings - args.max_new
        self.steps = self.interventions = 0
        self.unclosed: dict[int, str] = {}  # id(episode) -> tail of a reply that never closed (this turn)

    def act(self, episodes: list[dict]) -> list[str | None]:
        from transformers import LogitsProcessorList

        from tool_grammar import ToolCallGrammar

        tok, args = self.lm.tok, self.args
        prompts = [prompt(ep) for ep in episodes]
        out: list = [None] * len(prompts)
        self.unclosed = {}
        fit = []
        for i, p in enumerate(prompts):
            if len(tok(p)["input_ids"]) > self.limit:
                episodes[i]["stop"] = "context"
            else:
                fit.append(i)
        fit.sort(key=lambda i: len(prompts[i]))
        for start in range(0, len(fit), args.batch):
            idx = fit[start:start + args.batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(args.device)
            proc = ToolCallGrammar(self.pieces, tok.eos_token_id, [HEADER] * len(idx), args.max_str,
                                   schema=args.schema, values=PATHS if args.schema else None)
            with self.lm.torch.no_grad():
                self.lm.model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                       logits_processor=LogitsProcessorList([proc]))
            for slot, i in enumerate(idx):
                text = "".join(proc.texts[slot])
                if proc.states[slot][1][0] == "post":
                    out[i] = text.strip()
                else:
                    self.unclosed[id(episodes[i])] = f"[{len(text)} chars] ...{text[-300:]}"
            self.steps, self.interventions = self.steps + proc.steps, self.interventions + proc.interventions
        return out


# ---------------------------------------------------------------- tasks

def impl_stub(task: dict) -> str | None:
    """Signature and docstring with `raise NotImplementedError`; imports of the reference are kept (full shape)."""
    if task["kind"] == "body":
        return rt.code_of(task, "\n    raise NotImplementedError\n")
    tree = ast.parse(task["reference_raw"])
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == task["entry_point"]), None)
    if func is None:
        return None
    func = copy.deepcopy(func)
    func.body, func.decorator_list = [ast.Raise(exc=ast.Name("NotImplementedError"), cause=None)], []
    imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return rt.code_of(task, "\n".join([task["prompt"], *imports, ast.unparse(func)]) + "\n")


def k10_samples(path: str) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    with (ROOT / path).open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            by_id.setdefault(s["id"], []).append(s)
    return {i: sorted(ss, key=lambda s: s["sample"]) for i, ss in by_id.items()}


def select(tasks: list[dict], args) -> dict:
    """Fix seeds (task id, k10 sample index) and implement ids; see the module docstring."""
    path = ROOT / args.ids_out
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    samples = k10_samples(args.k10_samples)
    order = sorted(t["id"] for t in tasks if t["visible"])
    random.Random(0).shuffle(order)
    by_id = {t["id"]: t for t in tasks}
    fix, pos = [], 0
    while len(fix) < args.n_fix and pos < len(order):
        chunk, pos = order[pos:pos + 40], pos + 40
        items = [(by_id[i], s) for i in chunk for s in samples.get(i, []) if not s["passed"]]
        kinds = {(t["id"], s["sample"]): v["kind"] for (t, s), v in
                 rt.parallel(items, lambda ts: rt.run_visible(ts[0], rt.code_of(ts[0], ts[1]["completion"])),
                             args.workers, "fix seeds")}
        for i in chunk:
            seed = next((s["sample"] for s in samples.get(i, []) if kinds.get((i, s["sample"])) in ("wrong", "error")),
                        None)
            if seed is not None and len(fix) < args.n_fix:
                fix.append([i, seed])
    impl = [i for i in order[pos:] if impl_stub(by_id[i]) is not None][:args.n_impl]
    chosen = {"ids": [i for i, _ in fix] + impl, "fix": fix, "implement": impl, "fix_seed": "k10 lowest-index "
              "sample failing visible tests with kind wrong|error; tasks shuffled with random.Random(0)"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    return chosen


def build_episodes(tasks: list[dict], chosen: dict, args) -> list[dict]:
    by_id = {t["id"]: t for t in tasks}
    samples = k10_samples(args.k10_samples) if chosen["fix"] else {}
    episodes = []
    for task_id, seed in chosen["fix"]:
        task = by_id[task_id]
        episodes.append(start_episode(task, "fix", rt.code_of(task, samples[task_id][seed]["completion"])))
    for task_id in chosen["implement"]:
        episodes.append(start_episode(by_id[task_id], "implement", impl_stub(by_id[task_id])))
    return episodes


# ---------------------------------------------------------------- grading, report

def grade_episodes(episodes: list[dict], workers: int) -> None:
    items = [(ep, part) for ep in episodes for part in ("visible", "hidden", "full", "start")]

    def one(item):
        ep, part = item
        if part == "start":
            return rt.grade(ep["task"], ep["start"], "full")
        return rt.grade(ep["task"], ep["files"]["solution.py"], part)

    for (ep, part), (passed, _) in rt.parallel(items, one, workers, "grade"):
        ep[f"{part}_pass"] = passed


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    if not n:
        return {"episodes": 0}
    rate = lambda key: round(sum(bool(ep[key]) for ep in episodes) / n, 3)
    mean = lambda key: round(sum(ep[key] for ep in episodes) / n, 2)
    green = [ep for ep in episodes if ep["stop"] == "tests_pass"]
    return {"episodes": n, "full_pass": rate("full_pass"), "hidden_pass": rate("hidden_pass"),
            "visible_pass": rate("visible_pass"), "start_full_pass": rate("start_pass"),
            "tests_pass_stop": round(len(green) / n, 3),
            "hidden_pass_given_tests_pass": round(sum(ep["hidden_pass"] for ep in green) / len(green), 3)
            if green else None,
            "mean_turns": round(sum(len(ep["turns"]) for ep in episodes) / n, 2), "mean_tool_calls": mean("tool_calls"),
            "mean_tool_errors": mean("tool_errors"), "no_run_tests": round(sum(not ep["ran_tests"] for ep in episodes)
                                                                           / n, 3),
            "stop": dict(Counter(ep["stop"] for ep in episodes))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model dir (not needed with --oracle)")
    ap.add_argument("--oracle", action="store_true", help="reference answers, no model: checks the loop")
    ap.add_argument("--task-ids", default="research/raw/sa_v_clean_gate.json", help="gate id list")
    ap.add_argument("--k10-samples", default="scratch/sa_e_gate_k10/samples.jsonl", help="fix-mode start states")
    ap.add_argument("--ids-out", default="research/raw/sa7_agent_ids.json", help="chosen ids; reused if present")
    ap.add_argument("--n-fix", type=int, default=60)
    ap.add_argument("--n-impl", type=int, default=60)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-str", type=int, default=2048, help="chars per JSON string (write_file carries a file)")
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--schema", action=argparse.BooleanOptionalAction, default=True,
                    help="grammar enforces argument keys, string values and PATHS (--no-schema: syntax only)")
    ap.add_argument("--batch", type=int, default=16, help="prompts per generate call")
    ap.add_argument("--workers", type=int, default=4, help="sandbox subprocesses (1 GiB RSS cap each)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--vram-cap-gib", type=float, default=0.0, help="required on cuda: fresh free - 0.75 GiB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if min(args.max_turns, args.max_str, args.max_new, args.batch, args.workers) < 1 or min(args.n_fix,
                                                                                            args.n_impl) < 0:
        ap.error("--max-turns, --max-str, --max-new, --batch and --workers must be >= 1; --n-* >= 0")
    if not args.oracle and not args.model:
        ap.error("--model is required without --oracle")
    if not args.oracle and args.device == "cuda" and args.vram_cap_gib <= 0:
        ap.error("--vram-cap-gib is required on cuda (fresh free VRAM - 0.75 GiB)")
    ram_guard(MIN_FREE_RAM + args.workers * CHILD_RSS_CAP)

    t0 = time.perf_counter()
    tasks = rt.load_gate(args.task_ids)
    chosen = select(tasks, args)
    episodes = build_episodes(tasks, chosen, args)
    agent = OracleAgent() if args.oracle else HFAgent(args)
    run_episodes(episodes, agent, args.max_turns, args.workers)
    grade_episodes(episodes, args.workers)
    report = {"oracle": args.oracle, "ids": args.ids_out, "args": {k: str(v) for k, v in vars(args).items()},
              "overall": summarize(episodes),
              "per_mode": {m: summarize([ep for ep in episodes if ep["mode"] == m]) for m in ("fix", "implement")}}
    if not args.oracle:
        report["grammar"] = {"steps": agent.steps, "interventions": agent.interventions,
                             "intervention_rate": round(agent.interventions / max(agent.steps, 1), 4)}
    report["seconds"] = round(time.perf_counter() - t0, 1)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with out.with_suffix(".samples.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for ep in episodes:
            f.write(json.dumps({"id": ep["task"]["id"], "mode": ep["mode"], "stop": ep["stop"], "start": ep["start"],
                                "final": ep["files"]["solution.py"], "turns": ep["turns"],
                                **{k: ep[k] for k in ("tool_calls", "tool_errors", "ran_tests", "visible_pass",
                                                      "hidden_pass", "full_pass", "start_pass")},
                                "invalid_tail": ep.get("invalid_tail")}) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
