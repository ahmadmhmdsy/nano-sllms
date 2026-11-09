"""Never-seen bench, half A (2026-10-29): convert the web sets in data/bench/raw/ns_* into data/bench/ns/A_<set>.jsonl.

Sets (raw files are untrusted downloads; run this with -I):
  heval_pro   HumanEval Pro (164): the raw HumanEval function + its solution are given, write the self-invoking
              one (kind body; rows whose solution is a whole def are kind full).
              For full rows the raw prompt never names the new function, so a comment gives its signature (deviation).
  mbpp_pro    MBPP Pro (378): raw MBPP problem + its solution are given, write the new function. kind full.
  rwpb        RWPB (140): real-world repo functions, body completion. The reference is renamed `_ns_ref` and kept in
              `support`; tests compare candidate and reference. Rows needing torch/tensorflow/cv2/transformers/
              sklearn/scipy/numpy/pandas are dropped (the sandbox runs the bare base interpreter). kind body.
  cruxeval    CRUXEval-O (800): predict f(input). kind predict, graded by running the completed assert.
  lcb         LiveCodeBench release v5+v6 deltas, easy only: AtCoder/Codeforces stdin (kind stdin) and LeetCode
              functional (kind leetcode; ListNode/TreeNode rows dropped). Hidden cases capped at --max-hidden,
              inputs over --max-input chars dropped. No references ship: only decoding + the stub check run.
Asserts: the first is visible, the rest hidden (>= 2 needed); non-assert test statements become `support`.
Every item must reject a `pass` stub, and every item with a reference must pass it; failures are dropped and counted.

  .venv/Scripts/python.exe -I scripts/bench_ns_build.py
"""
import argparse
import ast
import base64
import io
import json
import pickle
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]  # -I leaves the script directory off sys.path
from bench_ns import NS_DIR, code_of, grade  # noqa: E402
from sa2_build import CHILD_RSS_CAP, MIN_FREE_RAM, parallel, ram_guard  # noqa: E402

RAW = ROOT / "data" / "bench" / "raw"
HEAVY = ("torch", "tensorflow", "cv2", "transformers", "sklearn", "scipy", "numpy", "pandas")
STUB = "    pass\n"


class NoGlobals(pickle.Unpickler):
    """LCB private tests are base64(zlib(pickle(json str))); a str needs no globals, so refuse them all."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} refused")


def lcb_private(s: str) -> list:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load())


def item(set_name: str, idx, kind: str, prompt: str, entry: str, reference: str | None, visible: list,
         hidden: list, support: str = "", category: str = "", **meta) -> dict:
    return {"id": f"A/{set_name}/{idx}", "half": "A", "set": set_name, "category": category or set_name,
            "kind": kind, "prompt": prompt, "entry_point": entry, "reference": reference, "visible": visible,
            "hidden": hidden, "support": support, "meta": meta}


def split_tests(test_code: str) -> tuple[list[str], list[str], str] | None:
    """Asserts -> (visible, hidden); other top-level statements -> setup kept in support."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return None
    asserts, setup = [], []
    for node in tree.body:
        (asserts if isinstance(node, ast.Assert) else setup).append(ast.get_source_segment(test_code, node))
    if len(asserts) < 2:
        return None
    return asserts[:1], asserts[1:], "\n".join(setup)


def def_names(src: str) -> list[str]:
    return re.findall(r"^def (\w+)\(", src, flags=re.M)


def build_heval_pro() -> list[dict]:
    out = []
    for i, r in enumerate(pq.read_table(RAW / "ns_heval_pro" / "humaneval_pro.parquet").to_pylist()):
        tests = split_tests(r["test_code"])
        if not tests:
            continue
        given = def_names(r["new_problem"])
        if given and r["new_solution"][:1] in (" ", "\t"):  # usual row: new_problem ends in the new def, solution is its body
            prompt = r["new_problem"].rstrip() + "\n"
            if not set(def_names(r["raw_problem"])) <= set(given):
                prompt = f"{r['raw_problem'].rstrip()}\n{r['raw_solution'].rstrip()}\n\n\n{prompt}"
            out.append(item("heval_pro", r.get("id", i), "body", prompt, given[-1], r["new_solution"], *tests[:2],
                            support=tests[2]))
            continue
        new = [n for n in def_names(r["new_solution"]) if n not in given]
        if not new:
            continue
        sig = next(line for line in r["new_solution"].split("\n") if line.startswith(f"def {new[-1]}("))
        lines = r["new_problem"].split("\n")
        at = next((j for j, line in enumerate(lines) if not line.startswith("#")), 0)
        hint = f"# The tests call: {sig.rstrip().rstrip(':')}"
        prompt = "\n".join(lines[:at] + [hint] + lines[at:])
        out.append(item("heval_pro", r.get("id", i), "full", prompt, new[-1], r["new_solution"], *tests[:2],
                        support=tests[2]))
    return out


def build_mbpp_pro() -> list[dict]:
    out = []
    for i, r in enumerate(pq.read_table(RAW / "ns_mbpp_pro" / "mbpp_pro.parquet").to_pylist()):
        tests = split_tests(r["test_code"])
        new = def_names(r["new_problem"])
        if not tests or not new:
            continue
        prompt = r["new_problem"]
        if not set(def_names(r["raw_problem"])) <= set(new):
            prompt = f"{r['raw_problem'].rstrip()}\n{r['raw_solution'].rstrip()}\n\n\n{prompt}"
        out.append(item("mbpp_pro", r.get("id", i), "full", prompt, new[-1], r["new_solution"], *tests[:2],
                        support=tests[2]))
    return out


def build_rwpb() -> list[dict]:
    rows = json.loads((RAW / "ns_rwpb" / "rwpb.json").read_text(encoding="utf-8"))
    out = []
    for r in rows if isinstance(rows, list) else rows.values():
        src = r["prompt"] + r["canonical_solution"]
        if any(re.search(rf"\b{m}\b", src) for m in HEAVY):
            continue
        try:
            fns = [n for n in ast.parse(r["prompt"]).body if isinstance(n, ast.FunctionDef)]
        except SyntaxError:
            continue
        if not fns:
            continue
        entry = fns[-1].name
        tests_src = re.sub(rf"\b{entry}\(", "_ns_ref(", r["unprocess_testcases"]).replace("SOLUTION_SIGNATURE(",
                                                                                         f"{entry}(")
        tests = split_tests(tests_src)
        if not tests:
            continue
        ref = re.sub(rf"^def {entry}\(", "def _ns_ref(", src, flags=re.M)
        out.append(item("rwpb", r["task_id"], "body", r["prompt"], entry, r["canonical_solution"], *tests[:2],
                        support=f"{ref}\n{tests[2]}"))
    return out


def build_cruxeval() -> list[dict]:
    out = []
    with (RAW / "ns_cruxeval" / "cruxeval.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(item("cruxeval", r["id"], "predict", f"{r['code']}\nassert f({r['input']}) ==", "f",
                            r["output"], [], [], category="predict"))
    return out


def build_lcb(max_hidden: int, max_input: int) -> list[dict]:
    out, dropped = [], Counter()
    for name in ("test5.jsonl", "test6.jsonl"):
        with (RAW / "ns_lcb" / name).open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["difficulty"] != "easy":
                    continue
                try:
                    public = json.loads(r["public_test_cases"])
                    private = lcb_private(r["private_test_cases"])
                except (ValueError, zlib.error, pickle.UnpicklingError) as e:
                    dropped[f"decode:{type(e).__name__}"] += 1
                    continue
                cases = [{"input": c["input"], "output": c["output"]} for c in public + private
                         if len(c["input"]) <= max_input]
                n_pub = sum(len(c["input"]) <= max_input for c in public)
                if n_pub < 1 or len(cases) < 2:
                    dropped["few-cases"] += 1
                    continue
                visible, hidden = cases[:n_pub], cases[n_pub:n_pub + max_hidden]
                doc = f'"""\n{r["question_content"].strip()}\n"""\n'
                meta = dict(platform=r["platform"], contest_date=r["contest_date"], source_file=name)
                if public[0]["testtype"] == "functional":
                    starter = r["starter_code"]
                    if re.search(r"ListNode|TreeNode", starter):
                        dropped["linked-structure"] += 1
                        continue
                    try:
                        for c in cases:
                            [json.loads(a) for a in c["input"].split("\n") if a.strip()]
                            json.loads(c["output"])
                    except json.JSONDecodeError:
                        dropped["non-json-case"] += 1
                        continue
                    entry = json.loads(r["metadata"])["func_name"]
                    prompt = f"from typing import *\n\n{doc}{starter.rstrip()}\n"
                    out.append(item("lcb", r["question_id"], "leetcode", prompt, entry, None, visible, hidden,
                                    category="leetcode", **meta))
                else:
                    prompt = f"{doc[:-4]}\nRead the input from standard input and print the answer.\n\"\"\"\n"
                    out.append(item("lcb", r["question_id"], "stdin", prompt, "", None, visible, hidden,
                                    category="stdin", **meta))
    print(f"[lcb] dropped {dict(dropped)}", flush=True)
    return out


def checks(it: dict) -> tuple[str, bool]:
    if it["kind"] == "predict":
        ok, _ = grade(it, code_of(it, it["reference"]))
        bad, _ = grade(it, code_of(it, "object()"))
        return ("ok" if ok and not bad else "ref-fail" if not ok else "stub-pass"), ok
    if it["kind"] == "stdin":
        stub = "print()\n"
    elif it["kind"] == "leetcode":
        stub = "        pass\n"
    else:
        stub = STUB
    ok = True if it["reference"] is None else grade(it, code_of(it, it["reference"]))[0]
    bad = grade(it, code_of(it, stub))[0]
    return ("ok" if ok and not bad else "ref-fail" if not ok else "stub-pass"), ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="heval_pro,mbpp_pro,rwpb,cruxeval,lcb")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-hidden", type=int, default=12)
    ap.add_argument("--max-input", type=int, default=20_000)
    args = ap.parse_args()
    ram_guard(MIN_FREE_RAM + args.workers * CHILD_RSS_CAP)
    builders = {"heval_pro": build_heval_pro, "mbpp_pro": build_mbpp_pro, "rwpb": build_rwpb,
                "cruxeval": build_cruxeval, "lcb": lambda: build_lcb(args.max_hidden, args.max_input)}
    NS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.sets.split(","):
        items = builders[name]()
        verdicts = Counter()
        keep = []
        for it, (verdict, _) in parallel(items, checks, args.workers, f"check {name}"):
            verdicts[verdict] += 1
            if verdict == "ok":
                keep.append(it)
        with (NS_DIR / f"A_{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for it in keep:
                f.write(json.dumps(it) + "\n")
        summary[name] = {"converted": len(items), "kept": len(keep), "verdicts": dict(verdicts),
                         "kinds": dict(Counter(it["kind"] for it in keep))}
        print(f"[{name}] {summary[name]}", flush=True)
    (NS_DIR / "A_build.json").write_text(json.dumps({"args": vars(args), "sets": summary}, indent=2) + "\n",
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
