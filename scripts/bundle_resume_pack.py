"""Build the ONE-zip machine-move resume pack: fresh clone of the repo plus
this zip must be a fully working setup (HANDOFF section 3b / H2_RESUME
section 8 pattern, extended to every weights-bearing final).

Usage:
  python scripts/bundle_resume_pack.py \
      --source runs/target/final --source runs/sft_v2_e1/final ... \
      --note checkpoint_backup/RESUME_PACK_<date>.md \
      --out checkpoint_backup/resume_pack_<date>.zip

Conventions (inherited from scripts/bundle_checkpoint.py):
- STORED (uncompressed) zip64 - safetensors do not compress; STORED keeps
  packaging fast and honest on size.
- Arcnames are REPO-ROOT-RELATIVE ("runs/<name>/final/<file>"), so the zip
  extracts AT THE REPO ROOT and every file lands exactly right.
- Verify before declaring success: every entry present, sizes match, CRC
  pass (z.testzip). Then a chunked sha256 of the zip itself.
- Keep the archive OUT of the repo (gitignored checkpoint_backup/; a >1 GB
  tracked artifact would break the free-LFS quota decision, HANDOFF s7).
- .env and data/teacher/ are NEVER bundled (secrets / re-downloadable).
"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
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


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # pragma: no cover - best-effort provenance only
        return f"unavailable ({e})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                    help="repo root (arcnames are relative to it)")
    ap.add_argument("--source", action="append", required=True,
                    help="weights-bearing dir to bundle, repo-relative "
                         "(e.g. runs/target/final); repeatable")
    ap.add_argument("--note", help="optional markdown note embedded at zip root as RESUME.md")
    ap.add_argument("--out", required=True, help="output .zip path")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sources = []
    for s in args.source:
        d = (repo / s).resolve()
        if not d.is_dir():
            raise SystemExit(f"source dir not found: {s}")
        weights = d / "model.safetensors"
        if not weights.is_file():
            raise SystemExit(f"source has no model.safetensors (de-weighted?): {s}")
        files = sorted(p for p in d.iterdir() if p.is_file())
        sources.append((s.replace("\\", "/"), d, files))

    total = sum(f.stat().st_size for _, _, files in sources for f in files)
    print(f"[pack] {len(sources)} sources, {total:,} B total", flush=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for rel, d, files in sources:
            for f in files:
                arc = f"{rel}/{f.name}"
                z.write(f, arcname=arc)
            print(f"[pack] + {rel} ({len(files)} files)", flush=True)
        if args.note:
            note = Path(args.note).resolve()
            z.write(note, arcname="RESUME.md")
            print("[pack] + RESUME.md (root)", flush=True)
        # manifest LAST so it can carry the CRC of everything before it
        entries = [
            {"arcname": i.filename, "size": i.file_size, "crc32": f"{i.CRC:08x}"}
            for i in z.infolist()
        ]
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(repo),
            "format": "zip64 STORED; extract AT THE REPO ROOT (arcnames are repo-relative)",
            "total_bytes": total,
            "entries": entries,
        }
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))

    with zipfile.ZipFile(out) as z:
        infos = {i.filename: i.file_size for i in z.infolist()}
        missing = [f"{rel}/{f.name}" for rel, d, files in sources for f in files
                   if infos.get(f"{rel}/{f.name}") != f.stat().st_size]
        bad = z.testzip()
    if missing or bad is not None:
        raise SystemExit(f"[pack] VERIFY FAILED: missing={missing} bad_crc_entry={bad}")
    print(f"[pack] verify OK ({len(infos)} entries incl manifest, all CRC pass)", flush=True)
    print(f"[pack] sha256 {sha256_of(out)}", flush=True)
    print(f"[pack] wrote {out} ({out.stat().st_size:,} B)", flush=True)


if __name__ == "__main__":
    main()
