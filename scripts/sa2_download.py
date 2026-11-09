"""Download the SA-2 source datasets (research/2026-10-22_sa_plan_v2.md §2 A+B; user approved 2026-10-22).

Files land in data/hf/sa2/<owner>__<name>/ and a manifest (repo, file, bytes, license) is written to
data/hf/sa2/manifest.json. Re-running skips files that are already complete.

  .venv/Scripts/python.exe scripts/sa2_download.py [--dry-run]
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hf" / "sa2"
MIN_FREE_GB = 5.0  # stop before the disk gets tighter than this

# (repo, files, license). Subsets keep the download near 1 GB: E: is ~96 % full.
SOURCES = [
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", ["data/train-00000-of-00001.parquet"], "odc-by"),
    ("OpenCoder-LLM/opc-sft-stage2", ["educational_instruct/train-00000-of-00001.parquet"], "mit"),
    ("nvidia/OpenCodeInstruct", [f"data/train-{i:05d}-of-00050.parquet" for i in range(3)], "cc-by-4.0"),
    ("evalplus/humanevalplus", ["data/test-00000-of-00001-5973903632b82d40.parquet"], "apache-2.0"),
    ("evalplus/mbppplus", ["data/test-00000-of-00001-d5781c9c51e02795.parquet"], "apache-2.0"),
    ("glaiveai/glaive-function-calling-v2", ["glaive-function-calling-v2.json"], "apache-2.0"),
    ("Salesforce/xlam-function-calling-60k", ["xlam_function_calling_60k.json"], "cc-by-4.0"),
    ("nampdn-ai/tiny-codes", ["part_1_200000.parquet"], "mit"),
]


def read_hf_token():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    token = read_hf_token()
    manifest = []
    for repo, files, lic in SOURCES:
        dest = OUT / repo.replace("/", "__")
        for f in files:
            free_gb = shutil.disk_usage(ROOT).free / 2**30
            if free_gb < MIN_FREE_GB:
                sys.exit(f"ABORT: only {free_gb:.1f} GiB free (< {MIN_FREE_GB})")
            print(f"[get] {repo} :: {f}  (free {free_gb:.1f} GiB)", flush=True)
            if args.dry_run:
                continue
            path = hf_hub_download(repo, f, repo_type="dataset", local_dir=dest, token=token)
            manifest.append({"repo": repo, "file": f, "bytes": pathlib.Path(path).stat().st_size,
                             "license": lic, "path": str(pathlib.Path(path).relative_to(ROOT))})
    if not args.dry_run:
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"[done] {len(manifest)} files, {sum(m['bytes'] for m in manifest) / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
