
r"""Download the two Stage 0 candidate Arabic recognizers (inference format).

Both repos ship PaddleOCR inference files (inference.json + inference.pdiparams +
inference.yml), so no paddleocr/paddlex install is needed - paddle.jit.load reads
them directly.

Usage: & .\.venv\Scripts\python.exe arabic_ocr\scripts\dl_stage0_models.py
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = {
    "ppocrv5_mobile_rec": "PaddlePaddle/arabic_PP-OCRv5_mobile_rec",
    "ppocrv6_small_rec": "medyas/arabic_PP-OCRv6_small_rec",
}
ROOT = Path("arabic_ocr/models")


def main() -> int:
    for name, repo in REPOS.items():
        dest = ROOT / name
        print("== %s -> %s" % (repo, dest))
        try:
            got = snapshot_download(repo_id=repo, local_dir=str(dest))
        except Exception as exc:
            print("   FAILED: %s: %s" % (type(exc).__name__, exc))
            continue
        total = 0
        for f in sorted(Path(got).rglob("*")):
            if f.is_file() and ".cache" not in f.parts:
                sz = f.stat().st_size
                total += sz
                print("   %-46s %10d" % (str(f.relative_to(got)), sz))
        print("   total %.2f MB" % (total / 1048576))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
