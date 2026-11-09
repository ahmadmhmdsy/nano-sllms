import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mex.src import tasks

def test_every_task_has_three_disjoint_splits():
    for name, gen in [("arith", tasks.arith), ("structure", tasks.structure),
                      ("strops", tasks.strops)]:
        d = gen(seed=42, n_val=200, n_test=500)
        assert set(d) == {"train", "val", "test"}
        st = {id(line) for lst in d.values() for line in lst}
        assert len(st) == sum(len(v) for v in d.values())
        assert len(d["val"]) == 200 and len(d["test"]) == 500

def test_arith_train_bounded_by_n_train():
    d = tasks.arith(seed=42, n_val=200, n_test=500, n_train=400)
    assert len(d["train"]) <= 400
    assert len(d["val"]) == 200 and len(d["test"]) == 500

def test_generators_are_deterministic():
    a = tasks.arith(seed=7, n_val=50, n_test=100)
    b = tasks.arith(seed=7, n_val=50, n_test=100)
    assert a == b
    assert tasks.arith(seed=8, n_val=50, n_test=100) != b

def test_arith_lines_are_exact_answerable():
    line = tasks.arith(seed=1, n_val=5, n_test=5)["test"][0]
    prompt, target = line.split("|", 1)   # "123+45=|168\n"
    lhs = prompt[: prompt.index("=")]     # operand-width agnostic
    assert eval(lhs) == int(target)

def test_structure_labels_match_balanced_checker():
    """E-24/E-25 hardening: labels must come straight from _balanced."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    d = tasks.structure(seed=42, n_val=200, n_test=500)
    for lst in d.values():
        for line in lst:
            seq, label = line.rstrip("\n").split("\n")
            assert len(seq) <= 24                      # new maxlen default
            assert ("ok" if tasks._balanced(seq, pairs) else "bad") == label

def test_structure_labels_roughly_balanced_and_subtle():
    """New mix: ~50/50 ok/bad; bad lines half subtle one-pair flips."""
    d = tasks.structure(seed=42, n_val=200, n_test=500)
    labels = [ln.rstrip("\n").split("\n")[1] for lst in d.values() for ln in lst]
    ok_frac = labels.count("ok") / len(labels)
    assert 0.40 <= ok_frac <= 0.60, ok_frac
    # Subtle corruption leaves a line one substitution away from a balanced
    # string (full-flip lines essentially never are) — both kinds must exist.
    pairs = {"(": ")", "[": "]", "{": "}"}

    def one_flip_balances(seq):
        partner = {**pairs, **{v: k for k, v in pairs.items()}}
        return any(tasks._balanced(seq[:i] + partner[c] + seq[i + 1:], pairs)
                   for i, c in enumerate(seq))

    bad = [ln.rstrip("\n").split("\n")[0] for lst in d.values() for ln in lst
           if ln.endswith("bad\n")]
    n_subtle = sum(one_flip_balances(s) for s in bad)
    assert n_subtle >= 0.25 * len(bad)          # subtle branch really fires
    assert n_subtle <= 0.75 * len(bad)          # full-flip branch retained

def test_structure_subtle_bad_is_not_trivially_detectable():
    """Subtle bad lines keep every prefix valid ~half the time: a flip of a
    close bracket cannot be caught before the closing position itself."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    d = tasks.structure(seed=42, n_val=200, n_test=500)
    bad = [ln.rstrip("\n").split("\n")[0] for lst in d.values() for ln in lst
           if ln.endswith("bad\n")]

    def min_prefix_depth(seq):
        depth, m = 0, 0
        for ch in seq:
            depth += 1 if ch in pairs else -1
            m = min(m, depth)
        return m

    # Lines whose violation only appears at the very end (min prefix depth 0
    # but the final depth is negative): no local detector can flag them early.
    late = sum(1 for s in bad if min_prefix_depth(s) == 0 and
               sum(1 for c in s if c not in pairs) - sum(1 for c in s if c in pairs) == -2)
    assert late >= 0.05 * len(bad), late / len(bad)


import unittest  # noqa: E402


class TestTasks(unittest.TestCase):
    """unittest harness over the module-level pytest-style functions."""

    def test_every_task_has_three_disjoint_splits(self):
        test_every_task_has_three_disjoint_splits()

    def test_arith_train_bounded_by_n_train(self):
        test_arith_train_bounded_by_n_train()

    def test_generators_are_deterministic(self):
        test_generators_are_deterministic()

    def test_arith_lines_are_exact_answerable(self):
        test_arith_lines_are_exact_answerable()

    def test_structure_labels_match_balanced_checker(self):
        test_structure_labels_match_balanced_checker()

    def test_structure_labels_roughly_balanced_and_subtle(self):
        test_structure_labels_roughly_balanced_and_subtle()

    def test_structure_subtle_bad_is_not_trivially_detectable(self):
        test_structure_subtle_bad_is_not_trivially_detectable()
