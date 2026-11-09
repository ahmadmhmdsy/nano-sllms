r"""T0 golden harness for the registry promotion (flow-registry-promotion.md).

Captures flow.server.config_gen.build_config output for EVERY committed
flow/flows/*.flow.json into flow/tests/goldens/<name>.golden.json so the
registry refactor (dict entries -> behavior-carrying NodeDefinitions) can
prove itself shape-preserving: flow/tests/test_goldens.py asserts the
refactored build_config reproduces each golden exactly.

Run BEFORE the refactor (these goldens were generated with the pre-promotion
dict-based registry) and re-run deliberately whenever the mapping contract
changes on purpose:

    & .\.venv\Scripts\python.exe flow\tests\goldens\create_goldens.py

Branching flows (build_config raises ValueError by design) get an
{"error": "<the exact ValueError message>"} golden so even refusal wording
is locked.
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from flow.server import config_gen  # noqa: E402

FLOWS_DIR = REPO / "flow" / "flows"
OUT_DIR = Path(__file__).resolve().parent


def capture(flow_path: Path):
    """build_config result for one flow document, or {"error": message}."""
    with open(flow_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    try:
        return config_gen.build_config(graph)
    except ValueError as exc:
        return {"error": str(exc)}


def main() -> None:
    """Write one golden file per committed flow; print every written name."""
    for flow_path in sorted(FLOWS_DIR.glob("*.flow.json")):
        name = flow_path.name[: -len(".flow.json")]
        golden = capture(flow_path)
        out_path = OUT_DIR / (name + ".golden.json")
        payload = json.dumps(golden, indent=2, sort_keys=True) + "\n"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())  # durable before reporting OK
        print(out_path.name)


if __name__ == "__main__":
    main()
