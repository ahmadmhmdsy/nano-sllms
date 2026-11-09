"""Download the Tatoeba sentences export for DA-1 (network-only, CPU-safe)."""
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src.data import TATOEBA_URL


def main() -> int:
    raw_dir = pathlib.Path("data/langid/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "sentences.tar.bz2"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[download] already present: {dest} ({dest.stat().st_size} bytes)")
        return 0
    tmp = dest.with_suffix(".part")
    print(f"[download] {TATOEBA_URL} -> {tmp}")
    urllib.request.urlretrieve(TATOEBA_URL, tmp)
    tmp.replace(dest)
    print(f"[download] done: {dest.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
