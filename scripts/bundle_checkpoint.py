"""Bundle a training checkpoint into a zip64 archive for a machine move.

Usage:
  python scripts/bundle_checkpoint.py --checkpoint runs/target/checkpoint-2000 \
      --out "E:/somewhere/checkpoint-2000.zip"

Creates a STORED (uncompressed) zip64 archive - safetensors/optimizer tensors
do not compress meaningfully and STORED keeps packaging fast. Verifies that
every source file made it into the archive (name + size + CRC) and prints a
chunked sha256 for transfer integrity checks.

Keep the archive OUT of the repo (>1 GB breaks the free-LFS quota decision,
see HANDOFF section 7).
"""

import argparse
import hashlib
import zipfile
from pathlib import Path


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="checkpoint directory to bundle")
    ap.add_argument("--out", required=True, help="output .zip path")
    args = ap.parse_args()

    src = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")

    files = sorted(p for p in src.iterdir() if p.is_file())
    total = sum(f.stat().st_size for f in files)
    print(f"[bundle] {len(files)} files, {total:,} B total from {src}")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in files:
            z.write(f, arcname=f"{src.name}/{f.name}")
            print(f"[bundle] + {src.name}/{f.name} ({f.stat().st_size:,} B)")

    with zipfile.ZipFile(out) as z:
        infos = {i.filename: i.file_size for i in z.infolist()}
        missing = [
            f.name
            for f in files
            if f"{src.name}/{f.name}" not in infos
            or infos[f"{src.name}/{f.name}"] != f.stat().st_size
        ]
        bad = z.testzip()
    if missing or bad is not None:
        raise SystemExit(f"[bundle] VERIFY FAILED: missing={missing} bad_crc_entry={bad}")
    print("[bundle] verify OK (all entries present, sizes match, CRC pass)")

    print(f"[bundle] sha256 {sha256_of(out)}")
    print(f"[bundle] wrote {out} ({out.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
