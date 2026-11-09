"""DA-8 (E-56) title dataset build (dirs handled inline)."""
import json, os, random
import pyarrow.parquet as pq
OUT = "data/langid/title"
random.seed(42)
def clean(s):
    s = s.replace(chr(13), " ").replace(chr(10), " ").replace(chr(9), " ")
    return " ".join(s.split())
def save(rows, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for t, h in rows:
            f.write(clean(t) + chr(9) + clean(h) + chr(10))
def main():
    t = pq.read_table("scratch/asas_summ_train.parquet")
    ar = []
    for text, proc, summ in zip(t.column("text").to_pylist(), t.column("Processed Text").to_pylist(), t.column("summarizer").to_pylist()):
        body = clean(str(proc or text or ""))
        head = clean(str(summ or ""))
        if len(body) > 120 and len(head) > 20:
            ar.append((body[:1200], head))
    random.shuffle(ar)
    n = len(ar)
    c = (int(n * 0.85), int(n * 0.925))
    save(ar[:c[0]], OUT + "/ar/train.tsv")
    save(ar[c[0]:c[1]], OUT + "/ar/val.tsv")
    save(ar[c[1]:], OUT + "/ar/test.tsv")
    print("ar pairs", n)
    en = []
    for line in open("data/langid/raw/huff_top10.json", encoding="utf-8"):
        try: d = json.loads(line)
        except Exception: continue
        if not isinstance(d, dict): continue
        body = clean(str(d.get("short_description") or ""))
        head = clean(str(d.get("headline") or ""))
        if len(body) > 40 and len(head) > 10:
            en.append((body, head))
    random.shuffle(en)
    n = len(en)
    c = (int(n * 0.9), int(n * 0.95))
    save(en[:c[0]], OUT + "/en/train.tsv")
    save(en[c[0]:c[1]], OUT + "/en/val.tsv")
    save(en[c[1]:], OUT + "/en/test.tsv")
    print("en pairs", n)
main()
