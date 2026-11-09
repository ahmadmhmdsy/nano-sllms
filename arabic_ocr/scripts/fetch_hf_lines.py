"""Fetch real line-level Arabic OCR corpora from the Hugging Face Hub.

Streams each configured dataset and writes a flat, project-local corpus:
    data/hf/<name>/<split>/<idx>.png
    data/hf/<name>/<split>/labels.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from datasets import load_dataset


# name -> (repo, splits, text_field)
DATASETS = {
    "khatt_teklia": ("Teklia/KHATT_v1.0", ["train", "validation", "test"], "text"),
    "muharaf": ("aamijar/muharaf-public", ["train", "validation", "test"], "text"),
    # AMIDDA sub-sets - line-level historical Arabic manuscripts, licence-verified:
    # etalab-2.0 (baybars, iskandar) / apache-2.0 (RASAM-1, tarima). Fetched DIRECTLY
    # rather than via the calfa-ai/AMIDDA aggregate, whose own card is "other" and
    # which also carries our own KHATT/Muharaf test splits (leakage).
    # NOTE: the transcription column here is "transcription", not "text".
    "amidda_baybars": ("calfa-ai/baybars", ["train", "validation", "test"], "transcription"),
    "amidda_iskandar": ("calfa-ai/iskandar", ["train", "validation", "test"], "transcription"),
    "amidda_rasam1": ("calfa-ai/RASAM-1", ["train", "validation", "test"], "transcription"),
    "amidda_tarima": ("calfa-ai/tarima", ["train", "validation", "test"], "transcription"),
}


def fetch(name, repo, splits, text_field, out_root, limit=None):
    for split in splits:
        out = Path(out_root) / name / split
        if (out / "labels.json").exists():
            print("[%s/%s] already present, skipping" % (name, split), flush=True)
            continue
        try:
            ds = load_dataset(repo, split=split, streaming=True)
        except Exception as exc:
            print("[%s/%s] UNAVAILABLE: %s" % (name, split, str(exc)[:200]), flush=True)
            continue
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        labels = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            text = (row.get(text_field) or "").strip()
            img = row.get("image")
            if not text or img is None:
                continue
            try:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                fn = "%06d.png" % i
                img.save(out / fn)
            except Exception:
                continue
            rec = {"image": fn, "text": text, "w": img.width, "h": img.height,
                   "split": split}
            for src_key, dst_key in (("image_name", "src"),
                                     ("manuscript_name", "manuscript"),
                                     ("script", "script")):
                val = row.get(src_key)
                if val:
                    rec[dst_key] = val
            labels.append(rec)
            if i % 1000 == 0 and i:
                print("  [%s/%s] %d rows" % (name, split, i), flush=True)
        (out / "labels.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[%s/%s] DONE %d rows -> %s" % (name, split, len(labels), out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hf")
    ap.add_argument("--only", default=None, help="comma-separated subset of names")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(DATASETS)
    for name in names:
        repo, splits, text_field = DATASETS[name]
        print("=== %s (%s) ===" % (name, repo), flush=True)
        fetch(name, repo, splits, text_field, args.out, args.limit)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
