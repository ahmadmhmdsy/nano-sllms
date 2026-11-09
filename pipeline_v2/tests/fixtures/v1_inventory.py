from hashlib import sha256
import json
from pathlib import Path

INVENTORY_PATHS = (
    "scripts/train.py",
    "scripts/prepare_data.py",
    "scripts/tokenize_data.py",
    "scripts/run_custom.py",
    "scripts/sanity_check.py",
    "scripts/vram_probe.py",
    "scripts/eval.py",
    "scripts/sft.py",
    "scripts/kd.py",
    "scripts/mount.py",
    "src/model.py",
    "src/data.py",
    "configs/smoke.yaml",
    "configs/pilot.yaml",
    "configs/target.yaml",
    "configs/next_pretrain.yaml",
    "tests/_common.py",
    "tests/test_artifacts.py",
    "tests/test_explorer.py",
    "tests/test_simulator.py",
)

ROLE_BY_PATH = {path: "v1-core-boundary" for path in INVENTORY_PATHS}


def inventory(paths: tuple[str, ...], root: Path) -> dict[str, object]:
    rows = []
    for relative in sorted(paths):
        path = root / relative
        payload = path.read_bytes()
        rows.append({
            "path": relative,
            "role": ROLE_BY_PATH[relative],
            "sha256": sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    return {"schema_version": 1, "files": rows}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    target = root / "pipeline_v2/tests/fixtures/v1_inventory.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(inventory(INVENTORY_PATHS, root), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
