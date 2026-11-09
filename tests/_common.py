"""Path setup + PASS/FAIL runner for the repo's dependency-free test scripts.

No pytest in the venv (repo convention): each tests/test_*.py is run directly
with the venv python, uses plain asserts, prints PASS/FAIL per check and
exits 0/1. Tests add repo root, scripts/ and webui/ to sys.path so they can
import the same modules webui/app.py exposes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "webui")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
        return True
    except AssertionError as e:
        print(f"FAIL {name}: {e}")
        return False
    except Exception as e:  # noqa: BLE001 - report, don't crash the suite
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return False


def run(tests: dict):
    results = [check(n, f) for n, f in tests.items()]
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(tests)} checks passed")
    sys.exit(0 if n_pass == len(tests) else 1)


if __name__ == "__main__":
    # Allow `python tests/_common.py tests/test_<name>.py` as an alias for
    # the canonical direct form `python tests/test_<name>.py`. Without this
    # dispatcher the two-arg form printed NOTHING and exited 0 while running
    # zero checks (final-review minor 4) — a silent false-green footgun.
    import runpy
    if len(sys.argv) < 2:
        print("usage: python tests/_common.py tests/test_<name>.py")
        sys.exit(2)
    runpy.run_path(sys.argv[1], run_name="__main__")
