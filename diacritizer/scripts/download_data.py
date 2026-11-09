"""Download the D-line Phase-1 corpus + benchmark pack (R51).
  fadel    - AliOsm/arabic-text-diacritization (GitHub) Fadel train/val/test split
  wikinews - QCRI advancing-arabic-diacritization WikiNews-2024 (GitHub)
  sadeed_tashkeela - HF Misraj/Sadeed_Tashkeela (Tashkeela-derived, GPL-2 noted)
  sadeed_25 - HF Misraj/SadeedDiac-25 benchmark (expert-reviewed, mixed)

Resumable (HTTP Range for direct files / hf_hub cache for HF files).
CPU + network only: safe beside any active train run (AGENTS.md section 4).
Usage (venv, repo root): .venv/Scripts/python.exe diacritizer/scripts/download_data.py [--source NAME|all]
"""
import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent.parent
RAW = REPO / "data" / "diac" / "raw"

LICENSES = {
    "fadel": "Fadel split (AliOsm/arabic-text-diacritization); research use; verify upstream license on use",
    "wikinews": "QCRI WikiNews-2024 (advancing-arabic-diacritization); CC terms per repo; research",
    "sadeed_tashkeela": "Tashkeela-derived (GPL-2) via Misraj/Sadeed_Tashkeela; training-use OK for research; COMMERCIAL USE = USER DECISION FLAG",
    "sadeed_25": "Misraj/SadeedDiac-25 benchmark; check dataset card before redistribution",
}
FADEL_URLS = {
    "train.txt": "https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/master/dataset/train.txt",
    "val.txt":   "https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/master/dataset/val.txt",
    "test.txt":  "https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/master/dataset/test.txt",
}
WIKINEWS_URLS = {
    # QCRI advancing-arabic-diacritization (verified via git tree 2026-09-11)
    "eval_diac.java": "https://raw.githubusercontent.com/qcri/advancing-arabic-diacritization/main/Evaluation/EvalDiac.java",
    "wikinews2014_multi_ref.diac": "https://raw.githubusercontent.com/qcri/advancing-arabic-diacritization/main/WikiNews_Benchmarks/WikiNews_2014_Multi_Ref.txt.diac",
    "wikinews2024_multi_ref.diac": "https://raw.githubusercontent.com/qcri/advancing-arabic-diacritization/main/WikiNews_Benchmarks/WikiNews_2024_Multi_Ref.txt.diac",
    "wikipedia_diac.jsonl": "https://raw.githubusercontent.com/qcri/advancing-arabic-diacritization/main/Wikipedia_Diacritized_Corpus/Wikipedia_20240420.diac.jsonl",
}
SOURCES = ("fadel", "wikinews", "sadeed_tashkeela", "sadeed_25")


def http_download(url, dest: Path):
    """Simple resumable-within-session download; big-but-single files only."""
    if dest.exists() and dest.stat().st_size > 0:
        print("  [SKIP] %s exists (%d bytes)" % (dest.name, dest.stat().st_size))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "nano_slms-diacritizer/0.1"})
    with urlopen(req, timeout=120) as r:
        data = r.read()
    tmp.write_bytes(data)
    tmp.replace(dest)
    print("  [OK] %s (%d bytes)" % (dest.name, len(data)))


def download_fadel():
    d = RAW / "fadel"
    for name, url in FADEL_URLS.items():
        http_download(url, d / name)


def download_wikinews():
    d = RAW / "wikinews"
    for name, url in WIKINEWS_URLS.items():
        http_download(url, d / name)


def download_hf_maps(dataset_id, files):
    """Download via the HF resolve endpoint (no datasets lib needed)."""
    d = RAW / dataset_id.split("/")[-1].lower()
    for name in files:
        http_download(
            "https://huggingface.co/datasets/%s/resolve/main/%s" % (dataset_id, name),
            d / name,
        )


def download_sadeed_tashkeela():
    download_hf_maps("Misraj/Sadeed_Tashkeela", TASHKEELA_FILES)


def download_sadeed_25():
    d = RAW / "sadeed_25"
    http_download(SADEED_25_URL, d / "sadeed25.parquet")


TASHKEELA_FILES = ["data/train-00000-of-00001.parquet"]
SADEED_FILES = ["default/train/0000.parquet"]

SADEED_25_URL = "https://huggingface.co/datasets/Misraj/SadeedDiac-25/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"


def download_sadeed_25():
    d = RAW / "sadeed_25"
    http_download(SADEED_25_URL, d / "sadeed25.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES + ("all",), default="all")
    args = ap.parse_args()
    print("=== diacritizer download_data ===")
    ok = True
    jobs = {"fadel": download_fadel, "wikinews": download_wikinews,
            "sadeed_tashkeela": download_sadeed_tashkeela, "sadeed_25": download_sadeed_25}
    todo = SOURCES if args.source == "all" else (args.source,)
    for src in todo:
        print("[%s]" % src)
        try:
            jobs[src]()
        except Exception as e:
            ok = False
            print("  [FAIL] %s: %r" % (src, e))
    # provenance sidecar (license field per source, DESIGN section 3)
    meta = {"sources": [{"id": s, "license": LICENSES[s]} for s in SOURCES], "date": "2026-09-11"}
    (RAW / "provenance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("provenance.json written; %s" % ("ALL SOURCES OK" if ok else "SOME SOURCES FAILED (see above)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
