"""Build the DA-1 held-out eval set from the official FLORES-200 archive.

Protocol (DESIGN.md section 4): dev + devtest sentences for the 21 target
languages, one per language pair, Arabic-script codes mapped to our tags
(arb/=ar, pes/=fa, urd/=ur, pbt/=pashto 'ps'). Full sentences for 'full'
evaluation; k-word windows are sampled at evaluate time with seeded windows,
so nothing else is needed here. Writes data/langid/eval.tsv.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from langid.src.data import ARABIC_SCRIPT_GROUP, LANGS, TATOEBA_CODE

# FLORES-200 file stem for each of our ISO-639-1 tags
FLORES_STEM = {
    "ar": "arb_Arab", "fa": "pes_Arab", "ur": "urd_Arab", "ps": "pbt_Arab",
    "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "tr": "tur_Latn",
    "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "pl": "pol_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "hi": "hin_Deva",
    "ko": "kor_Hang", "ja": "jpn_Jpan", "zh": "zho_Hans",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/langid/raw/flores200/flores200_dataset")
    ap.add_argument("--out", default="data/langid/eval.tsv")
    ap.add_argument("--manifest", default="data/langid/eval_manifest.json")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    counts = {}
    rows = []
    for tag in LANGS:
        added = 0
        for split in ("dev", "devtest"):
            stem = FLORES_STEM[tag]
            p = root / split / (stem + "." + split)
            if not p.exists():
                raise ValueError("missing FLORES file: " + str(p))
            for line in p.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if text:
                    rows.append((tag, text))
                    added += 1
        counts[tag] = added
    with pathlib.Path(args.out).open("w", encoding="utf-8", newline="\n") as f:
        for tag, text in rows:
            f.write(tag + "\t" + text.replace("\t", " ") + "\n")
    man = {
        "source": "official FLORES-200 archive (dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz), CC-BY-SA-4.0, dev+devtest splits",
        "rows": len(rows),
        "per_lang": counts,
    }
    import json
    pathlib.Path(args.manifest).write_text(
        json.dumps(man, indent=1, ensure_ascii=True), encoding="utf-8")
    print("[evalset] rows:", len(rows))
    for t in counts:
        print("[evalset]", args.out, "tag", t, "matches", counts[t])
    return 0


if __name__ == "__main__":
    sys.exit(main())
