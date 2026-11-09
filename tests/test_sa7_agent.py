"""SA-7 agent loop: tools, prompt layout, stubs, episode stops, oracle, grammar fit (no model)."""
import json

from _common import run

import sa7_agent as ag
import sa7_runtime as rt
from tool_grammar import START, gfeed, grammar, tool_names

PROMPT = 'def add(a, b):\n    """Add two numbers. Then return.\n    >>> add(1, 2)\n    3\n    """'
GOOD = "\n    return a + b\n"
WRONG = "\n    return a - b\n"


def task():
    return rt.make_task("t/1", "gate", "body", PROMPT, "add", GOOD, ["assert add(1, 2) == 3"],
                        "assert add(2, 2) == 4\nassert add(0, 5) == 5")


def episode(start=WRONG, mode="fix"):
    t = task()
    return ag.start_episode(t, mode, rt.code_of(t, start))


def calls(*cs):
    return json.dumps([{"name": n, "arguments": a} for n, a in cs])


class Replies:
    """Answers every active episode with the same fixed reply (None = invalid)."""

    def __init__(self, reply):
        self.reply = reply

    def act(self, episodes):
        return [self.reply for _ in episodes]


def t_files_and_errors():
    ep = episode()
    assert ag.call_tool(ep, "read_file", {"path": "test_solution.py"}) == "assert add(1, 2) == 3\n"
    assert ag.call_tool(ep, "read_file", {"path": "x.py"}) == "error: no such file: x.py"
    assert ag.call_tool(ep, "write_file", {"path": "test_solution.py", "content": ""}).startswith("error: only")
    assert ag.call_tool(ep, "read_file", {}).startswith("error: read_file takes path")
    assert ag.call_tool(ep, "run_tests", {"x": "1"}).startswith("error: run_tests takes no arguments")


def t_run_python():
    ep = episode()
    assert ag.call_tool(ep, "run_python", {"code": "print(add(2, 3))"}) == "-1"
    out = ag.call_tool(ep, "run_python", {"code": "print(1 / 0)"})
    assert out.split("\n")[0] == "Traceback (most recent call last):" and "candidate" not in out, out
    assert out.endswith("ZeroDivisionError: division by zero"), out
    assert ag.call_tool(ep, "run_python", {"code": "x = 1"}) == "(no output)"
    assert ag.call_tool(ep, "run_python", {"code": "while True: pass"}).startswith("timeout")


def t_run_tests_red_green():
    ep = episode()
    out = ag.call_tool(ep, "run_tests", {})
    assert out == "FAIL: assert add(1, 2) == 3 -> returned -1 but expected 3\n0/1 passed", out
    assert not ep["green"] and ep["ran_tests"]
    ag.call_tool(ep, "write_file", {"path": "solution.py", "content": rt.code_of(ep["task"], GOOD)})
    assert ag.call_tool(ep, "run_tests", {}) == "PASS: assert add(1, 2) == 3\n1/1 passed" and ep["green"]
    ag.call_tool(ep, "write_file", {"path": "solution.py", "content": "x"})
    assert not ep["green"]  # a write after a green run needs a new run


def t_load_failure():
    ep = episode()
    ag.call_tool(ep, "write_file", {"path": "solution.py", "content": "def add(a, b):\n    return (\n"})
    out = ag.call_tool(ep, "run_tests", {})
    assert out.startswith("solution.py failed to load: SyntaxError") and out.endswith("0/1 passed"), out


def t_prompt_layout_and_history():
    ep = episode()
    p = ag.prompt(ep).split("\n")
    assert p[0] == "# Tools:" and p[-1] == ag.RESPOND and len(p) == 7
    assert p[5].startswith("# Request: Fix solution.py") and "Task: Add two numbers." in p[5]
    for i in range(5):
        ep["turns"].append({"reply": f"[{i}]", "observations": ["[run_tests] a\nb" + "x" * 1000]})
    p = ag.prompt(ep).split("\n")
    assert "# (2 earlier turns omitted)" in p and "# Called: [2]" in p and "# Called: [1]" not in p
    assert "# b" + "x" * 50 in "\n".join(p) and max(len(line) for line in p[1:]) <= rt.OUTCOME_CAP
    impl = ag.prompt(episode(mode="implement"))
    assert "# Request: Implement add in solution.py" in impl


def t_stubs():
    t = task()
    stub = ag.impl_stub(t)
    assert stub.endswith("raise NotImplementedError\n") and stub.startswith(PROMPT)
    full = rt.make_task("m/1", "gate", "full", '"""\nWrite a function to add.\nassert add(1, 2) == 3\n"""', "add",
                        "\nimport math\n@cache\ndef add(a, b):\n    return a + b\n\ndef helper():\n    pass\n",
                        ["assert add(1, 2) == 3"], "")
    stub = ag.impl_stub(full)
    assert "import math" in stub and "@cache" not in stub and "helper" not in stub and "return a + b" not in stub
    assert stub.rstrip().endswith("raise NotImplementedError"), stub
    assert not rt.grade(full, stub, "full")[0]


def t_oracle_one_turn():
    eps = [episode(), episode(mode="implement")]
    ag.run_episodes(eps, ag.OracleAgent(), max_turns=3, workers=2)
    ag.grade_episodes(eps, workers=2)
    for ep in eps:
        assert ep["stop"] == "tests_pass" and len(ep["turns"]) == 1 and ep["full_pass"] and not ep["start_pass"]
    s = ag.summarize(eps)
    assert s["full_pass"] == 1.0 and s["tests_pass_stop"] == 1.0 and s["no_run_tests"] == 0.0


def t_invalid_and_turns():
    ep = episode()
    ag.run_episodes([ep], Replies(None), max_turns=3, workers=1)
    assert ep["stop"] == "invalid" and not ep["turns"]
    ep = episode()
    ag.run_episodes([ep], Replies(calls(("read_file", {"path": "solution.py"}))), max_turns=3, workers=1)
    assert ep["stop"] == "turns" and len(ep["turns"]) == 3 and ep["tool_calls"] == 3 and not ep["ran_tests"]
    ep = episode()
    ag.run_episodes([ep], Replies(calls(("run_tests", {"path": "x"}))), max_turns=2, workers=1)
    assert ep["tool_errors"] == 2 and ep["turns"][0]["observations"][0].startswith("[run_tests] error:")


def t_grammar_fit():
    assert tool_names(ag.HEADER) == {"read_file", "write_file", "run_python", "run_tests"}
    content = rt.code_of(task(), GOOD) + "# padding\n" * 20
    reply = calls(("write_file", {"path": "solution.py", "content": content}), ("run_tests", {}))
    assert len(json.dumps(content)) > 128
    assert gfeed(START, reply, grammar(ag.HEADER, 2048))[1][0] == "post"
    assert gfeed(START, reply, grammar(ag.HEADER)) is None
    assert gfeed(START, calls(("delete_file", {})), grammar(ag.HEADER)) is None


def t_schema_grammar():
    gram = grammar(ag.HEADER, 2048, schema=True, values=ag.PATHS)
    ok = lambda *cs: (s := gfeed(START, calls(*cs), gram)) is not None and s[1][0] == "post"
    code = rt.code_of(task(), GOOD)
    assert ok(("write_file", {"content": code, "path": "solution.py"}), ("run_tests", {}))
    assert ok(("read_file", {"path": "test_solution.py"}), ("run_python", {"code": "print(add(1, 2))"}))
    assert not ok(("run_tests", {"x": "1"}))  # run_tests takes no arguments
    assert not ok(("write_file", {"path": "solution.py", "text": code}))  # wrong key
    assert not ok(("write_file", {"path": "solution.py"}))  # missing key: the object cannot close
    assert not ok(("write_file", {"path": "solution.py", "content": code, "x": ""}))
    assert not ok(("write_file", {"path": "test_solution.py", "content": ""}))  # path outside PATHS
    assert not ok(("read_file", {"path": "x.py"})) and not ok(("run_python", {"code": 1}))
    assert not ok(("run_python", {"code": ["x"]}))
    for reply in ag.OracleAgent().act([episode()]):
        assert gfeed(START, reply, gram)[1][0] == "post"
    # off by default: the syntax-only grammar still takes the bad calls
    assert gfeed(START, calls(("run_tests", {"x": 1})), grammar(ag.HEADER, 2048))[1][0] == "post"
    assert len(grammar(ag.HEADER)) == 4 and grammar(ag.HEADER)[3] is None


if __name__ == "__main__":
    run({"files_and_errors": t_files_and_errors, "run_python": t_run_python, "run_tests_red_green": t_run_tests_red_green,
         "load_failure": t_load_failure, "prompt_layout_and_history": t_prompt_layout_and_history, "stubs": t_stubs,
         "oracle_one_turn": t_oracle_one_turn, "invalid_and_turns": t_invalid_and_turns,
         "grammar_fit": t_grammar_fit, "schema_grammar": t_schema_grammar})
