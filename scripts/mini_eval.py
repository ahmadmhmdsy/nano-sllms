"""Execution-based mini-eval (Milestone C): REAL pass@1, not a parse proxy.

Each task is a small Python function prompt; the model completes it, the
completion is appended to the prompt and run per test in a timeout'd
subprocess (fresh temp workdir per test, no network use, per-test credit).
A completion only scores when its code actually executes and passes.

Modes:
  --model canned  harness self-test: scores known-correct solutions; must
                  report pass_at_1 == 1.0 (proves runner/scoring mechanics).
  --model llm     real run: greedy generations from --ckpt (default).

Report lands in <ckpt>/mini_eval_report.json. Raw base models are NOT
instruction tuned, so pass@1 near 0.0 is expected pre-SFT - the harness
measures RELATIVE deltas (e.g., pre-SFT vs post-C12-SFT) plus run-to-run
stability of the greedy pipeline. While any training run is live, always
pass --device cpu (the default 'auto' device picks cuda).

Run: .venv/Scripts/python scripts/mini_eval.py --ckpt runs/smoke/final --device cpu --model canned
     .venv/Scripts/python scripts/mini_eval.py --ckpt runs/smoke/final --device cpu
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_MARKER = "MINIEVAL_OK"

# 16 tasks x 2 tests. Prompts end inside the function body so a plain
# continuation can complete them.
TASKS = [
    {"id": "add", "prompt": "def add(a, b):\n    return",
     "tests": ["assert add(2, 3) == 5", "assert add(-4, 4) == 0"],
     "canned": " a + b"},
    {"id": "is_even", "prompt": "def is_even(n):\n    return",
     "tests": ["assert is_even(4) is True", "assert is_even(7) is False"],
     "canned": " n % 2 == 0"},
    {"id": "max_of_two", "prompt": "def max_of_two(a, b):\n    return",
     "tests": ["assert max_of_two(3, 9) == 9", "assert max_of_two(-2, -7) == -2"],
     "canned": " a if a >= b else b"},
    {"id": "reverse_string", "prompt": "def reverse_string(s):\n    return",
     "tests": ["assert reverse_string('abc') == 'cba'",
               "assert reverse_string('') == ''"],
     "canned": " s[::-1]"},
    {"id": "count_vowels", "prompt": "def count_vowels(s):\n    return",
     "tests": ["assert count_vowels('hello') == 2", "assert count_vowels('XYZ') == 0"],
     "canned": " sum(1 for ch in s.lower() if ch in 'aeiou')"},
    {"id": "sum_list", "prompt": "def sum_list(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return",
     "tests": ["assert sum_list([1, 2, 3]) == 6", "assert sum_list([]) == 0"],
     "canned": " total"},
    {"id": "abs_diff", "prompt": "def abs_diff(a, b):\n    return",
     "tests": ["assert abs_diff(3, 10) == 7", "assert abs_diff(10, 3) == 7"],
     "canned": " abs(a - b)"},
    {"id": "last_char", "prompt": "def last_char(s):\n    return",
     "tests": ["assert last_char('py') == 'y'", "assert last_char('a') == 'a'"],
     "canned": " s[-1]"},
    {"id": "repeat", "prompt": "def repeat(s, n):\n    return",
     "tests": ["assert repeat('ab', 3) == 'ababab'", "assert repeat('x', 0) == ''"],
     "canned": " s * n"},
    {"id": "celsius_to_fahrenheit", "prompt": "def celsius_to_fahrenheit(c):\n    return",
     "tests": ["assert abs(celsius_to_fahrenheit(100) - 212) < 1e-9",
               "assert abs(celsius_to_fahrenheit(0) - 32) < 1e-9"],
     "canned": " c * 9.0 / 5.0 + 32.0"},
    {"id": "string_length", "prompt": "def string_length(s):\n    return",
     "tests": ["assert string_length('code') == 4", "assert string_length('') == 0"],
     "canned": " len(s)"},
    {"id": "min_in_list", "prompt": "def min_in_list(nums):\n    return",
     "tests": ["assert min_in_list([4, 2, 9]) == 2", "assert min_in_list([-3]) == -3"],
     "canned": " min(nums)"},
    {"id": "double_list", "prompt": "def double_list(nums):\n    return",
     "tests": ["assert double_list([1, 2]) == [2, 4]", "assert double_list([]) == []"],
     "canned": " [n * 2 for n in nums]"},
    {"id": "greeting", "prompt": "def greeting(name):\n    return",
     "tests": ["assert greeting('Ada') == 'Hello, Ada'",
               "assert greeting('Bo') == 'Hello, Bo'"],
     "canned": " 'Hello, ' + name"},
    {"id": "is_positive", "prompt": "def is_positive(n):\n    return",
     "tests": ["assert is_positive(5) is True", "assert is_positive(-5) is False"],
     "canned": " n > 0"},
    {"id": "square", "prompt": "def square(n):\n    return",
     "tests": ["assert square(7) == 49", "assert square(-3) == 9"],
     "canned": " n * n"},
]


def run_one_test(code_text: str, timeout: float):
    """Run candidate code in a fresh temp dir; True iff clean exit + marker."""
    with tempfile.TemporaryDirectory(prefix="minieval_") as td:
        path = Path(td) / "candidate.py"
        path.write_text(code_text, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)], cwd=td, capture_output=True,
                text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except OSError as e:
            return False, f"spawn failed: {e}"
        if proc.returncode == 0 and RUN_MARKER in proc.stdout:
            return True, None
        tail = (proc.stderr or proc.stdout or "")[-400:]
        return False, tail


def generate_completion(tok, model, prompt: str, max_new_tokens: int) -> str:
    import torch
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="execution-based mini-eval")
    ap.add_argument("--ckpt", required=True, help="model dir (llm mode)")
    ap.add_argument("--model", choices=["llm", "canned"], default="llm",
                    help="canned = score known-good solutions (self-test)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="per-test subprocess timeout in seconds")
    args = ap.parse_args()

    device = args.device
    if args.model == "canned":
        completions = {t["id"]: t["canned"] for t in TASKS}
    else:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt = Path(args.ckpt)
        if not ckpt.is_dir():
            raise SystemExit(f"model dir not found: {ckpt}")
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForCausalLM.from_pretrained(str(ckpt))
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            device = "cpu"  # never silently touch the GPU
        model.to(device).eval()
        print(f"[mini_eval] mode=llm ckpt={ckpt} device={device} "
              f"max_new_tokens={args.max_new_tokens}", flush=True)

    started = time.time()
    per_task, n_passed, n_tests, n_tests_passed = [], 0, 0, 0
    for i, task in enumerate(TASKS):
        if args.model == "canned":
            completion = completions[task["id"]]
        else:
            completion = generate_completion(tok, model, task["prompt"],
                                             args.max_new_tokens)
        full = task["prompt"] + completion
        tests_passed = 0
        errors = []
        for test in task["tests"]:
            n_tests += 1
            code = full + "\n\n" + test + f"\nprint('{RUN_MARKER}')\n"
            ok, err = run_one_test(code, args.timeout)
            if ok:
                tests_passed += 1
            else:
                errors.append({"test": test, "error": err})
        n_tests_passed += tests_passed
        passed = tests_passed == len(task["tests"])
        n_passed += int(passed)
        per_task.append({"id": task["id"], "passed": passed,
                         "tests_passed": tests_passed,
                         "tests_total": len(task["tests"]),
                         "errors": errors[:1]})
        print(f"[mini_eval] {i + 1:2d}/{len(TASKS)} {task['id']:22s} "
              f"tests {tests_passed}/{len(task['tests'])} "
              f"{'PASS' if passed else 'FAIL'}", flush=True)

    runtime = time.time() - started
    report = {
        "ckpt": args.ckpt if args.model == "llm" else "(canned self-test)",
        "model_mode": args.model,
        "device": device,
        "max_new_tokens": args.max_new_tokens,
        "timeout_s": args.timeout,
        "n_tasks": len(TASKS),
        "n_tests": n_tests,
        "pass_at_1": round(n_passed / len(TASKS), 4),
        "test_credit": round(n_tests_passed / max(n_tests, 1), 4),
        "runtime_s": round(runtime, 1),
        "per_task": per_task,
    }
    print(f"[mini_eval] pass@1 {report['pass_at_1']:.4f} "
          f"({n_passed}/{len(TASKS)} tasks), test_credit "
          f"{report['test_credit']:.4f}, {runtime:.1f}s", flush=True)
    if args.model == "llm":
        out = Path(args.ckpt) / "mini_eval_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"[mini_eval] report written: {out}", flush=True)
    else:
        print("[mini_eval] canned self-test mode: no report file written",
              flush=True)


if __name__ == "__main__":
    main()
