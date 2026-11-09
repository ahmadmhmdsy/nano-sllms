"""Pre-training audit for the AMIDDA line corpora (TASKS 155).

Checks, all mandatory before a single AMIDDA row is used for training:

  1. LEAKAGE    - an AMIDDA line whose text matches one of our EVAL references is a leak.
                  Every hit records WHICH eval set it matched and the matched LENGTH, so a
                  split that is wholly derived from one of our benchmarks is recognisable
                  as such (rather than being written off as coincidence).
  2. DEDUPE     - the AMIDDA sub-sets overlap each other; duplicates are counted by
                  normalised text and by (manuscript, image_name).
  3. RESOLUTION - image heights are MEASURED, not assumed: the 27 px regime is our
                  verified failure mode (E-100), so the ingested distribution matters.

Writes data/hf/amidda_audit.json including a @@drop@@ list - the union of every leak - which
finetune_muharaf.py applies as a row-level skip. Exit code 1 when any leak is found: the
corpus is still usable, but ONLY through that drop list.
"""
import glob
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(".")
spec = importlib.util.spec_from_file_location("clc", ROOT / "arabic_ocr" / "scripts" / "char_lm_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def norm(t):
    return " ".join(C.strip_diacritics(t or "").split())


def hard(t):
    return "".join(ch for ch in norm(t) if not ch.isspace())


# path -> tag for the four FULL eval sets the Pareto gates are read from
EVAL_REFS = [
    ("data/hf/muharaf/test", "muharaf_test"),
    ("data/hf/khatt_teklia/test", "khatt_teklia_test"),
    ("data/benchmark/isippt", "isippt_bench"),
    ("data/benchmark/khatt", "khatt_bench"),
]


def load_rows(path):
    p = Path(path) / "labels.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def main():
    ref_of = {}
    all_refs = []
    for path, tag in EVAL_REFS:
        texts = [norm(r.get("text") or "") for r in load_rows(path)]
        texts = [t for t in texts if t]
        all_refs.extend(texts)
        for t in texts:
            ref_of.setdefault(t, set()).add(tag)
        print("eval ref %-30s tag=%-18s n=%d" % (path, tag, len(texts)))
    ref_set = set(ref_of)
    ref_blob = "\n".join(all_refs)
    ref_hard = "".join(hard(t) for t in all_refs)

    files = sorted(glob.glob("data/hf/amidda_*/**/labels.json", recursive=True))
    if not files:
        print("NO AMIDDA labels.json FOUND - run fetch_hf_lines.py first")
        return 1

    leaks = {"exact": [], "blob": [], "hard": []}
    seen_text = {}
    seen_page = {}
    dup_text = 0
    dup_page = 0
    heights = Counter()
    per_set = {}
    total = 0
    for f in files:
        parts = Path(f).parts          # data/hf/<name>/<split>/labels.json
        name, split = parts[2], parts[3]
        rows = json.loads(Path(f).read_text(encoding="utf-8"))
        per_set["%s/%s" % (name, split)] = len(rows)
        for r in rows:
            t = norm(r.get("text") or "")
            if not t:
                continue
            total += 1
            hd = hard(t)
            key = {"name": name, "split": split, "image": r.get("image")}
            if t in ref_set:
                rec = dict(key)
                rec["tags"] = sorted(ref_of[t])
                rec["chars"] = len(t)
                rec["text"] = t[:200]
                leaks["exact"].append(rec)
            elif len(t) >= 12 and t in ref_blob:
                leaks["blob"].append(dict(key, chars=len(t), text=t[:200]))
            elif len(hd) >= 12 and hd in ref_hard:
                leaks["hard"].append(dict(key, chars=len(hd), text=hd[:200]))
            if t in seen_text:
                dup_text += 1
            else:
                seen_text[t] = key
            page = (r.get("manuscript"), r.get("image_name"))
            if page[0] and page[1]:
                if page in seen_page:
                    dup_page += 1
                else:
                    seen_page[page] = key
            heights[int(r.get("h") or 0)] += 1

    print("\nAMIDDA rows=%d unique_text=%d dup_text=%d dup_page=%d" %
          (total, len(seen_text), dup_text, dup_page))
    for kind in ("exact", "blob", "hard"):
        hits = leaks[kind]
        if not hits:
            print("LEAKS %-5s = 0" % kind)
            continue
        print("LEAKS %-5s = %d" % (kind, len(hits)))
        print("  by amidda split: %s" %
              (Counter((h["name"], h["split"]) for h in hits).most_common(),))
        if kind == "exact":
            print("  by eval tag:    %s" %
                  (Counter(t for h in hits for t in h["tags"]).most_common(),))
            ls = sorted(h["chars"] for h in hits)
            print("  matched chars:  min=%d median=%d max=%d  (>=20ch: %d, 10-19ch: %d, <10ch: %d)"
                  % (ls[0], ls[len(ls) // 2], ls[-1], sum(1 for x in ls if x >= 20),
                     sum(1 for x in ls if 10 <= x < 20), sum(1 for x in ls if x < 10)))
            for h in sorted(hits, key=lambda x: -x["chars"])[:6]:
                print("    %4dch %s/%s/%s tags=%s  %s" %
                      (h["chars"], h["name"], h["split"], h["image"], h["tags"], h["text"][:70]))

    hs = sorted(h for h in heights if h)
    if hs:
        tot = sum(heights[h] for h in hs)
        cum, med = 0, None
        for h in hs:
            cum += heights[h]
            if med is None and cum >= tot / 2:
                med = h
        print("HEIGHT min=%d median=%d max=%d" % (hs[0], med, hs[-1]))
        print("  bands: <=32px=%d  33-64=%d  65-128=%d  >128=%d" % (
            sum(v for h, v in heights.items() if 0 < h <= 32),
            sum(v for h, v in heights.items() if 32 < h <= 64),
            sum(v for h, v in heights.items() if 64 < h <= 128),
            sum(v for h, v in heights.items() if h > 128)))

    drop, seen = [], set()
    for kind in ("exact", "blob", "hard"):
        for h in leaks[kind]:
            k = (h["name"], h["split"], h["image"])
            if k not in seen:
                seen.add(k)
                drop.append({"name": h["name"], "split": h["split"], "image": h["image"]})
    out = {
        "rows": total,
        "unique_text": len(seen_text),
        "dup_text": dup_text,
        "dup_page": dup_page,
        "leaks": leaks,
        "drop": drop,
        "per_set": per_set,
        "height_hist": dict(heights),
    }
    Path("data/hf/amidda_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print("\nwrote data/hf/amidda_audit.json  drop_rows=%d" % len(drop))
    if drop:
        print("VERDICT: LEAKAGE FOUND - %d row(s) must be skipped; the trainer applies the "
              "drop list" % len(drop))
        return 1
    print("VERDICT: PASS - no eval-set leakage")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
