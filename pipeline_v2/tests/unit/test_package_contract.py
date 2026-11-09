from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def test_help_is_cpu_only_and_lists_commands() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "pipeline_v2" / "run.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for command in ("init", "validate", "plan", "doctor", "preflight", "run", "status", "explain", "diff", "export", "migrate-v1"):
        assert command in result.stdout


def test_v2_source_does_not_import_v1_modules() -> None:
    forbidden = ("import src.", "from src.", "import scripts.", "from scripts.")
    for path in (ROOT / "pipeline_v2" / "slm_train").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
