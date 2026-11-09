
r"""Harvest Arabic text from the superseded OCR data mixes (the "TIER 4" dirs)
before their PNG images are deleted.

Read-only on the sources; writes only under data/ocr_textpool/.
Sources: all_combined all_data combined large extra training full real lines small smoke_lines

Usage:
  & .\.venv\Scripts\python.exe arabic_ocr\scripts\harvest_text_pool.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

SRC_DIRS = ["all_combined", "all_data", "combined", "large", "extra",
            "training", "full", "real", "lines", "small", "smoke_lines"]
EXTS = {".json", ".jsonl", ".txt"}
TEXT_KEYS = ("text", "label", "transcription", "transcript", "content",
             "sentence", "target", "answer", "ocr", "ground_truth", "gt")
IMAGE_KEYS = ("image", "img", "file", "filename", "path", "id", "name")

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t\u00a0]+")


def plain_of(s: str) -> str:
    return WS_RE.sub(" ", TAG_RE.sub(" ", s)).strip()


def read_text_any(p: Path):
    """UTF-8 first; fall back to cp1256 only if UTF-8 decoding fails."""
    raw = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("cp1256", errors="replace"), "cp1256"


def iter_records(obj, depth=0):
    """Yield (text, image_hint) from an arbitrary JSON shape."""
    if isinstance(obj, dict):
        lower = {k.lower(): k for k in obj}
        tk = next((lower[k] for k in TEXT_KEYS if k in lower), None)
        if tk is not None and isinstance(obj[tk], str):
            img = None
            for k in IMAGE_KEYS:
                if k in lower and isinstance(obj[lower[k]], str):
                    img = obj[lower[k]]
                    break
            yield obj[tk], img
            return
        if obj and all(isinstance(v, str) for v in obj.values()):
            for k, v in obj.items():
                yield v, k
            return
        for v in obj.values():
            yield from iter_records(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_records(v, depth + 1)


def load_file(p: Path):
    text, enc = read_text_any(p)
    if p.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield from iter_records(json.loads(line))
            except json.JSONDecodeError:
                pass
        return
    if p.suffix.lower() == ".txt":
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield line, None
        return
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print("  !! parse failed (%s): %s" % (enc, exc))
        return
    yield from iter_records(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="data/ocr_textpool")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-repeat", type=int, default=20,
                    help="texts occurring more often than this are boilerplate")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    raw_dir = out / "raw"

    files = []
    for name in SRC_DIRS:
        base = root / name
        if not base.is_dir():
            print("MISSING  %s" % base)
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                files.append(p)

    print("source files found: %d" % len(files))
    print("")
    print("%-46s %8s %9s %9s" % ("file", "rows", "distinct", "encoding"))

    records = {}
    per_source = Counter()
    table = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        source = rel.split("/")[0]
        _, enc = read_text_any(p)
        rows = 0
        for text, img in load_file(p):
            rows += 1
            t = (text or "").strip()
            if len(t) < args.min_chars:
                continue
            rec = records.get(t)
            if rec is None:
                records[t] = {"text": t, "plain": plain_of(t), "source": source,
                              "file": rel, "image": img, "count": 1}
            else:
                rec["count"] += 1
        per_source[source] += rows
        table.append((rel, rows, len(records), enc))
        print("%-46s %8d %9d %9s" % (rel[:46], rows, len(records), enc))
        if not args.dry_run:
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, raw_dir / rel.replace("/", "__"))

    kept = sorted((r for r in records.values() if r["count"] <= args.max_repeat),
                  key=lambda r: r["text"])
    boiler = sorted((r for r in records.values() if r["count"] > args.max_repeat),
                    key=lambda r: -r["count"])
    nonempty_plain = sum(1 for r in kept if r["plain"])

    print("")
    print("rows extracted      : %d" % sum(per_source.values()))
    print("distinct raw texts  : %d" % len(records))
    print("kept (<=%d repeats) : %d" % (args.max_repeat, len(kept)))
    print("dropped as boilerplate: %d" % len(boiler))
    print("kept with plain text: %d" % nonempty_plain)
    if boiler:
        print("top boilerplate (count): %s" % ", ".join(str(r["count"]) for r in boiler[:5]))
    print("")
    print("rows per source:")
    for k, v in per_source.most_common():
        print("  %-16s %d" % (k, v))

    if args.dry_run:
        print("")
        print("DRY RUN - nothing written")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "texts.jsonl"
    with jsonl.open("w", encoding="utf-8", newline="\n") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "source_dirs": SRC_DIRS,
        "source_files": len(files),
        "rows_extracted": sum(per_source.values()),
        "distinct_raw_texts": len(records),
        "kept": len(kept),
        "dropped_boilerplate": len(boiler),
        "kept_with_plain_text": nonempty_plain,
        "max_repeat": args.max_repeat,
        "min_chars": args.min_chars,
        "rows_per_source": dict(per_source),
        "files": [{"file": t[0], "rows": t[1], "cum_distinct": t[2], "encoding": t[3]}
                  for t in table],
        "top_boilerplate": [{"count": r["count"], "chars": len(r["text"])} for r in boiler[:20]],
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

    # samples for human inspection (UTF-8 file - never judge Arabic from the console)
    with (out / "samples.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("samples from texts.jsonl (kept)\n\n")
        step = max(1, len(kept) // 12)
        for r in kept[::step][:12]:
            fh.write("[%s] count=%d\nRAW  : %s\nPLAIN: %s\n\n"
                     % (r["file"], r["count"], r["text"][:200], r["plain"][:200]))

    readme = out / "README.md"
    lines = []
    lines.append("# data/ocr_textpool - Arabic text harvested from the deleted OCR mixes\n")
    lines.append("Created %s by arabic_ocr/scripts/harvest_text_pool.py.\n" % __import__("datetime").date.today().isoformat())
    lines.append("## Why this exists\n")
    lines.append("The TIER 4 cleanup deleted ~5.2 GB of superseded OCR image mixes")
    lines.append("(all_combined, all_data, combined, large, extra, training, full, real,")
    lines.append("lines, small, smoke_lines). Their PNGs were worthless - trained on")
    lines.append("teacher-forced metrics - but their label files hold real Arabic text,")
    lines.append("which is the bare-text pool Stage 1 of the approved two-stage OCR plan")
    lines.append("needs for synthetic rendering. This directory is that text, kept after")
    lines.append("the images are gone.\n")
    lines.append("## Contents\n")
    lines.append("- raw/ - verbatim copies of every source label file (provenance kept)")
    lines.append("- texts.jsonl - one record per distinct text")
    lines.append("- stats.json - machine-readable counts")
    lines.append("- samples.txt - human-readable samples (UTF-8)")
    lines.append("")
    lines.append("texts.jsonl record: {text, plain, source, file, image, count}")
    lines.append("")
    lines.append("- text  = raw string as found in the source")
    lines.append("- plain = HTML tags stripped and whitespace collapsed")
    lines.append("- count = how many times the identical text occurred in the sources")
    lines.append("")
    lines.append("## Filtering applied\n")
    lines.append("- texts shorter than %d chars dropped" % args.min_chars)
    lines.append("- texts occurring more than %d times dropped as boilerplate" % args.max_repeat)
    lines.append("  (the source files begin with a repeated VLM instruction prompt,")
    lines.append("  'What is the text written in the image? Please answer with the text only')")
    lines.append("- files decoded UTF-8 first, cp1256 only on decode failure\n")
    lines.append("## Numbers\n")
    lines.append("- source files: %d" % len(files))
    lines.append("- rows extracted: %d" % sum(per_source.values()))
    lines.append("- distinct raw texts: %d" % len(records))
    lines.append("- kept: %d (boilerplate dropped: %d)" % (len(kept), len(boiler)))
    lines.append("- kept with non-empty plain text: %d\n" % nonempty_plain)
    lines.append("## Per-file provenance\n")
    lines.append("| file | rows | cum. distinct | encoding |")
    lines.append("|---|---|---|---|")
    for rel, rows, cum, enc in table:
        lines.append("| %s | %d | %d | %s |" % (rel, rows, cum, enc))
    lines.append("")
    lines.append("## Caveats\n")
    lines.append("- QARI-sourced rows (extra/labels_qari_v03.json) carry markup")
    lines.append("  (h1/u/i) and are diacritized; use the plain field for rendering.")
    lines.append("- Some sources are conversational/Egyptian Arabic, not MSA.")
    lines.append("- This is a TEXT pool only: it has no page geometry and no boxes.")
    lines.append("  Stage 1 must shape and render it to obtain glyph and PAW boxes.")
    lines.append("- Do not delete raw/: it is the only surviving copy of these labels.")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print("wrote %s" % jsonl)
    print("wrote %s" % (out / "stats.json"))
    print("wrote %s" % (out / "samples.txt"))
    print("wrote %s" % readme)
    print("raw copies: %d files in %s" % (len(files), raw_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
