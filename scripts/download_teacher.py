#!/usr/bin/env python3
r"""Download a teacher model snapshot for C12 Tier 2 (teacher-as-judge).

Default repo: Qwen/Qwen3.5-0.8B (verified 2026-09-06: ungated, ~1.77 GB,
Apache-2.0, 24 text layers hidden 1024, vocab 248,320, 262k ctx; transformers
5.16.1 loads Qwen3_5ForConditionalGeneration natively). The snapshot is
text-only usage for judging; the vision tower ships along but stays unused.

Weights land under data/teacher/ (gitignored - local-only like the ladder
weights; LFS quota). Network on this machine is slow (~230 KB/s): run this in
the background and check for TEACHER_READY in the output.

Usage:
    & .\.venv\Scripts\python.exe scripts\download_teacher.py
    & .\.venv\Scripts\python.exe scripts\download_teacher.py --repo <id> --out data\teacher\<name>
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IGNORE = ["*.bin", "*.pth", "*.onnx", "*.gguf", "*.msgpack", "*.h5"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a Tier 2 teacher snapshot.")
    ap.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default="data/teacher/qwen35_08b")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    target = ROOT / args.out
    print("downloading", args.repo, "->", target, flush=True)
    path = snapshot_download(args.repo, local_dir=str(target), ignore_patterns=IGNORE)
    print("TEACHER_READY", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())