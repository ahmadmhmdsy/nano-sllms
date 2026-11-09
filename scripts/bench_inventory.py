"""Inventory the Python-code benchmarks under data/bench/raw and write a manifest.

CPU-only and network-free. Records sha256, size, row count and the columns of
every downloaded file so a future agent can tell provenance from the manifest
alone. These files are EVAL-ONLY: never train on them (contamination).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

RAW = Path("data/bench/raw")
OUT = Path("data/bench/manifest.json")

SOURCES = {
    "HumanEval.jsonl.gz": ("openai/human-eval", "MIT", "https://github.com/openai/human-eval", "164 hand-written problems; the canonical pass@1 benchmark"),
    "mbpp.jsonl": ("google-research/mbpp", "CC-BY-4.0", "https://github.com/google-research/google-research/tree/master/mbpp", "974 crowd-sourced problems (superset of the eval rows)"),
    "sanitized-mbpp.json": ("google-research/mbpp", "CC-BY-4.0", "https://github.com/google-research/google-research/tree/master/mbpp", "427 MBPP problems after removing the auto-generated tests"),
    "humanevalplus.jsonl": ("evalplus/humanevalplus", "Apache-2.0", "https://huggingface.co/datasets/evalplus/humanevalplus", "HumanEval with far more tests; the strict current version"),
    "mbppplus.parquet": ("evalplus/mbppplus", "Apache-2.0", "https://huggingface.co/datasets/evalplus/mbppplus", "MBPP+ - MBPP with extra tests"),
    "bigcodebench_v0.1.4.parquet": ("bigcode/bigcodebench", "Apache-2.0", "https://huggingface.co/datasets/bigcode/bigcodebench", "tasks that require library calls; harder and more realistic"),
    "ds1000.jsonl": ("xlangai/DS-1000", "research use only", "https://huggingface.co/datasets/xlangai/DS-1000", "1000 data-science problems over 7 libraries"),
}

TEXT_COLUMNS = ("canonical_solution", "solution", "code", "prompt", "complete_prompt", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".jsonl":
        with path.open("rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def main() -> int:
    if not RAW.exists():
        print(f"missing {RAW}", file=sys.stderr)
        return 1
    manifest = {"root": str(RAW), "files": {}}
    print(f"{'file':<30} {'bytes':>9} {'rows':>6}  columns")
    for path in sorted(RAW.iterdir()):
        if path.is_dir():
            continue
        entry = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        try:
            rows = load(path)
            entry["rows"] = len(rows)
            entry["columns"] = sorted(rows[0].keys()) if rows else []
            for column in TEXT_COLUMNS:
                if rows and column in rows[0]:
                    entry["chars_" + column] = sum(len(str(row.get(column) or "")) for row in rows)
        except Exception as error:  # one bad file must not hide the others
            entry["error"] = f"{type(error).__name__}: {error}"
        source = SOURCES.get(path.name)
        if source:
            entry["dataset"], entry["license"], entry["url"], entry["note"] = source
        manifest["files"][path.name] = entry
        print(f"{path.name:<30} {entry['bytes']:>9} {str(entry.get('rows', '-')):>6}  {','.join(entry.get('columns', []))[:68]}")
        if "error" in entry:
            print("    ERROR " + entry["error"])
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("wrote " + str(OUT) + " (" + str(len(manifest["files"])) + " files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
