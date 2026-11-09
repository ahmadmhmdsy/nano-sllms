"""Build a Z-Mahmood-style word cache for OUR model from OUR diac training pool.

Sources: data/diac/raw/{sadeed_tashkeela,abdou_tashkeel}/data/train-*.parquet +
sadeed_25.parquet. Rows are (non_vocalized, vocalized) pairs; when the two
tokenize into equal-length word lists, each positional pair votes
bare(normalized) -> diacritic variant. Keep the majority variant when it holds
>= 0.995 of the word's votes and total votes >= 2 (zmahood cache meta rules).
Output: models/e19/our_word_cache.json
"""
import json, re, glob, unicodedata
import pyarrow.parquet as pq

_DIAC = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"
strip_re = re.compile("[" + _DIAC + "]")

def norm_word(w: str) -> str:
    w = strip_re.sub("", unicodedata.normalize("NFKC", w))
    return (w.replace("\u0622", "\u0627")
             .replace("\u0623", "\u0627")
             .replace("\u0625", "\u0627")
             .replace("\u0640", ""))

THRESHOLD = 0.995
MIN_COUNT = 2

files = (sorted(glob.glob("data/diac/raw/sadeed_tashkeela/data/train-*.parquet"))
         + sorted(glob.glob("data/diac/raw/abdou_tashkeel/data/train-*.parquet"))
         + ["data/diac/raw/sadeed_25/sadeed25.parquet"])

counts: dict[str, dict[str, int]] = {}
total_tokens = 0
for fp in files:
    try:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=4096, columns=["non_vocalized", "vocalized"]):
            nvs = batch.column(0).to_pylist()
            vos = batch.column(1).to_pylist()
            for nv, vo in zip(nvs, vos):
                bw = str(nv).split()
                vw = str(vo).split()
                if len(bw) != len(vw):
                    continue
                for b, v in zip(bw, vw):
                    key = norm_word(b)
                    if not key or key.isascii():
                        continue
                    variants = counts.setdefault(key, {})
                    variants[v] = variants.get(v, 0) + 1
                    total_tokens += 1
        print(fp.replace("\\", "/").split("/")[-1], "done; tokens", total_tokens, flush=True)
    except Exception as e:
        print("ERR", fp, str(e)[:150], flush=True)

cache = {}
for key, variants in counts.items():
    if len(variants) == 1:
        tot = sum(variants.values())
        if tot >= MIN_COUNT:
            cache[key] = next(iter(variants))
        continue
    best, bestc = max(variants.items(), key=lambda kv: kv[1])
    tot = sum(variants.values())
    if bestc / tot >= THRESHOLD and tot >= MIN_COUNT:
        cache[key] = best

out = {"meta": {"total_tokens": total_tokens, "unique_keys": len(counts),
                "cached": len(cache), "threshold": THRESHOLD, "min_count": MIN_COUNT},
       "words": cache}
json.dump(out, open("models/e19/our_word_cache.json", "w", encoding="utf-8"), ensure_ascii=False)
print("DONE keys", len(counts), "cached", len(cache), flush=True)
