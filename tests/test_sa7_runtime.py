"""SA-7 runtime: split parsing, visible-test witness, vote, retry ranking and prompt, budget (stub model)."""
from _common import run

import sa7_runtime as rt
from sa4_data import REPAIR_ASK, REPAIR_HEAD

PROMPT = 'def add(a, b):\n    """Add two numbers. Then return.\n    >>> add(1, 2)\n    3\n    """'
GOOD = "\n    return a + b\n"
WRONG = "\n    return a - b\n"
ERROR = "\n    return a + undefined_name\n"


def task(visible=("assert add(1, 2) == 3",), hidden="assert add(2, 2) == 4\nassert add(0, 5) == 5"):
    return rt.make_task("t/1", "gate", "body", PROMPT, "add", GOOD, list(visible), hidden)


class Stub:
    """Answers prompt -> fixed replies (cycled); records every call."""

    def __init__(self, first, retry=()):
        self.first, self.retry, self.calls = list(first), list(retry), []

    def sample(self, prompts, n):
        self.calls.append((list(prompts), n))
        out = []
        for p in prompts:
            pool = self.retry if p.startswith(REPAIR_HEAD) else self.first
            out.append([(pool[i % len(pool)], True) for i in range(n)])
        return out

    def greedy(self, prompts):
        return [(self.first[0], True) for _ in prompts]


def t_he_examples():
    doc = ('def f(x):\n    """Doc.\n    >>> f(1)\n    2\n    >>> f(2)\n    [1,\n     2]\n    >>> g(\n'
           '    >>> f(3)\n    4\n    """')
    assert rt.he_examples(doc) == ["assert f(1) == 2", "assert f(3) == 4"]


def t_description():
    t = task()
    assert t["description"] == "Add two numbers.", t["description"]
    full = rt.make_task("m/1", "mbpp", "full", '"""\nWrite a function to add. More.\nassert add(1, 2) == 3\n"""',
                        "add", "\ndef add(a, b):\n    return a + b\n", ["assert add(1, 2) == 3"], "")
    assert full["description"] == "Write a function to add."


def t_reference_passes_every_part():
    t = task()
    for part in ("visible", "hidden", "full"):
        assert rt.grade(t, t["reference_code"], part) == (True, "pass"), part


def t_visible_witness():
    t = task()
    assert rt.run_visible(t, rt.code_of(t, GOOD)) == {"kind": "pass"}
    wrong = rt.run_visible(t, rt.code_of(t, WRONG))
    assert wrong["kind"] == "wrong" and wrong["witness"] == {"call": "add(1, 2)", "got": "-1", "expected": "3"}
    err = rt.run_visible(t, rt.code_of(t, ERROR))
    assert err["kind"] == "error" and err["witness"]["error"].startswith("NameError"), err
    assert rt.run_visible(t, rt.code_of(t, "\n    return (\n"))["kind"] == "crash"
    # A non-simple assert runs as written; its AssertionError counts as a wrong answer.
    t2 = task(visible=["assert add(1, 2) in (3,)"])
    assert rt.run_visible(t2, rt.code_of(t2, WRONG))["kind"] == "wrong"


def t_repair_prompt_layout():
    t = task()
    _, attempt = rt.attempt_parts(rt.code_of(t, WRONG), "add")
    p = rt.repair_prompt(t, attempt, {"call": "add(1, 2)", "got": "-1", "expected": "3"})
    assert p.split("\n") == [REPAIR_HEAD, "# Task: Add two numbers.", "def add(a, b):", "    return a - b",
                             "# >>> add(1, 2)", "# returned -1 but expected 3", REPAIR_ASK, "def add(a, b):"]
    p = rt.repair_prompt(t, attempt, {"call": "add(1, 2)", "error": "NameError: name 'x' is not defined"})
    assert "# Traceback (most recent call last):\n# NameError: name 'x' is not defined" in p
    long = rt.repair_prompt(t, attempt, {"call": "add(1, 2)", "got": "x" * 2000, "expected": "3"})
    assert max(len(line) for line in long.split("\n")) == rt.OUTCOME_CAP


def t_vote_and_seed():
    samples = [{"code": "def f():\n    return 1"}, {"code": "def f():\n    return 2"},
               {"code": "def f():  # same AST\n    return 2"}, {"code": "def f():\n    return 1"}]
    assert rt.vote(samples, [0, 1, 2]) == 1  # largest cluster
    assert rt.vote(samples, [0, 1, 2, 3]) == 0  # tie -> shorter code, then first
    t = {"entry_point": "f", "samples": [
        {"code": "def f():\n    return 1", "visible": {"kind": "error"}},
        {"code": "def f():\n    return 2", "visible": {"kind": "timeout"}},
        {"code": "def f():\n    return 3", "visible": {"kind": "wrong"}},
        {"code": "def f():\n    return 4", "visible": {"kind": "wrong"}},
        {"code": "def f():\n    return 4", "visible": {"kind": "wrong"}}]}
    assert rt.retry_seed(t) == 3  # wrong beats error; then the largest cluster
    t["samples"] = [{"code": "def f():\n    return 2", "visible": {"kind": "timeout"}}]
    assert rt.retry_seed(t) is None


def t_solve_filter():
    t = task()
    rt.solve([t], Stub([WRONG, GOOD, GOOD]), k=3, retries=1, k_retry=2, workers=2)
    assert t["source"] == "filter" and t["answer"] == 1 and len(t["samples"]) == 3


def t_solve_retry_and_budget():
    t = task()
    model = Stub([WRONG, ERROR], retry=["\n    return a + b\n"])
    rt.solve([t], model, k=4, retries=2, k_retry=3, workers=2)
    assert t["source"] == "retry" and len(t["samples"]) == 7, (t["source"], len(t["samples"]))
    assert len(model.calls) == 2 and model.calls[1][1] == 3  # the retry stops once a sample passes
    retry = t["samples"][-1]
    assert retry["origin"] == "retry1" and retry["code"].startswith("def add(a, b):\n    return a + b")
    assert "# returned -1 but expected 3" in retry["prompt"]  # seeded from the wrong answer, not the NameError


def t_solve_no_pass_and_vote_only():
    t = task()
    model = Stub([WRONG], retry=[WRONG])
    rt.solve([t], model, k=2, retries=2, k_retry=2, workers=2)
    assert t["source"] == "none" and len(t["samples"]) == 6
    v = task(visible=[])
    model = Stub([WRONG, GOOD, GOOD])
    rt.solve([v], model, k=3, retries=1, k_retry=2, workers=2)
    assert v["source"] == "vote" and v["answer"] == 1 and len(model.calls) == 1  # no retry without visible tests


def t_run_summary():
    class Args:
        k, retries, k_retry, workers = 4, 1, 2, 2

    ts = [task(), task()]
    ts[1]["id"] = "t/2"
    s = rt.run(ts, Stub([WRONG, GOOD, WRONG, WRONG], retry=[GOOD]), Args)
    assert s["runtime_pass"] == 1.0 and s["greedy_pass1"] == 0.0 and s["oracle_passN"] == 1.0
    assert s["sample_pass1"] == 0.25 and s["pass_at_4"] == 1.0 and s["source"] == {"filter": 2}


def t_reference_model_dry_run():
    t = task()
    s = rt.run([t], rt.ReferenceModel([t]), type("A", (), {"k": 1, "retries": 0, "k_retry": 1, "workers": 1}))
    assert s["runtime_pass"] == s["greedy_pass1"] == 1.0


if __name__ == "__main__":
    run({"he_examples": t_he_examples, "description": t_description,
         "reference_passes_every_part": t_reference_passes_every_part, "visible_witness": t_visible_witness,
         "repair_prompt_layout": t_repair_prompt_layout, "vote_and_seed": t_vote_and_seed,
         "solve_filter": t_solve_filter, "solve_retry_and_budget": t_solve_retry_and_budget,
         "solve_no_pass_and_vote_only": t_solve_no_pass_and_vote_only, "run_summary": t_run_summary,
         "reference_model_dry_run": t_reference_model_dry_run})
