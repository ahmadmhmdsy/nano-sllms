"""DA-9 (E-57) Schemer: synthetic Arabic schema-slot corpus builder.

Slots: DATE_G (Gregorian), DATE_H (Hijri), DATE_REL (relative), TIME, NUM_AI
(Arabic-Indic numeral). Filler tokens come from real Arabic text (asas-ai
summaries via data/langid/title/ar/train.tsv column 1). TSV token<TAB>tag like
DA-7 ner/, token-per-two-columns format of read_sents.
"""
import json, os, random, pathlib
BASE = pathlib.Path(__file__).resolve().parents[2]
OUT = BASE / "data" / "langid" / "schemer"
rng = random.Random(42)

AI = str.maketrans("0123456789", "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669")
G_MONTHS = ["\u064a\u0646\u0627\u064a\u0631","\u0641\u0628\u0631\u0627\u064a\u0631","\u0645\u0627\u0631\u0633","\u0623\u0628\u0631\u064a\u0644","\u0645\u0627\u064a\u0648","\u064a\u0648\u0646\u064a\u0648","\u064a\u0648\u0644\u064a\u0648","\u0623\u063a\u0633\u0637\u0633","\u0633\u0628\u062a\u0645\u0628\u0631","\u0623\u0643\u062a\u0648\u0628\u0631","\u0646\u0648\u0641\u0645\u0628\u0631","\u062f\u064a\u0633\u0645\u0628\u0631"]
H_MONTHS = ["\u0645\u062d\u0631\u0645","\u0635\u0641\u0631","\u0631\u0628\u064a\u0639 \u0627\u0644\u0623\u0648\u0644","\u0631\u0628\u064a\u0639 \u0627\u0644\u062b\u0627\u0646\u064a","\u062c\u0645\u0627\u062f\u0649 \u0627\u0644\u0623\u0648\u0644\u0649","\u062c\u0645\u0627\u062f\u0649 \u0627\u0644\u0622\u062e\u0631\u0629","\u0631\u062c\u0628","\u0634\u0639\u0628\u0627\u0646","\u0631\u0645\u0636\u0627\u0646","\u0634\u0648\u0627\u0644","\u0630\u0648 \u0627\u0644\u0642\u0639\u062f\u0629","\u0630\u0648 \u0627\u0644\u062d\u062c\u0629"]
REL = ["\u0623\u0645\u0633","\u063a\u062f\u064b\u0627","\u0627\u0644\u064a\u0648\u0645","\u0642\u0628\u0644 \u064a\u0648\u0645\u064a\u0646","\u0642\u0628\u0644 \u0623\u0633\u0628\u0648\u0639","\u0642\u0628\u0644 \u0623\u0633\u0628\u0648\u0639\u064a\u0646","\u0628\u0639\u062f \u0634\u0647\u0631\u064a\u0646","\u0627\u0644\u0634\u0647\u0631 \u0627\u0644\u0645\u0627\u0636\u064a","\u0627\u0644\u0634\u0647\u0631 \u0627\u0644\u0645\u0642\u0628\u0644","\u0627\u0644\u0639\u0627\u0645 \u0627\u0644\u0645\u0627\u0636\u064a","\u0627\u0644\u0639\u0627\u0645 \u0627\u0644\u0645\u0642\u0628\u0644","\u0645\u0646\u0630 \u0633\u0627\u0639\u062a\u064a\u0646","\u062d\u0627\u0644\u064a\u064b\u0627"]
H_SUFFIX = ["\u0647\u0640","\u0627\u0644\u0647\u062c\u0631\u064a\u0629"]
WORD_T = {1:"\u0627\u0644\u0623\u0648\u0644\u0649",2:"\u0627\u0644\u062b\u0627\u0644\u062b\u0629",3:"\u0627\u0644\u0631\u0627\u0628\u0639\u0629",4:"\u0627\u0644\u062e\u0627\u0645\u0633\u0629",5:"\u0627\u0644\u0633\u0627\u062f\u0633\u0629",6:"\u0627\u0644\u0633\u0627\u0628\u0639\u0629",7:"\u0627\u0644\u062b\u0627\u0645\u0646\u0629",8:"\u0627\u0644\u062a\u0627\u0633\u0639\u0629",9:"\u0627\u0644\u062d\u0627\u062f\u064a\u0629\u0639\u0634\u0631\u0629",10:"\u0627\u0644\u0639\u0627\u0634\u0631\u0629",11:"\u0627\u0644\u062d\u0627\u062f\u064a\u0629 \u0639\u0634\u0631\u0629",12:"\u0627\u0644\u062b\u0627\u0646\u064a\u0629 \u0639\u0634\u0631\u0629"}
AMPM = {"\u0635\u0628\u0627\u062d\u064b\u0627":"AM","\u0645\u0633\u0627\u0621\u064b":"PM"}


_TOK = None

def _tokens():
    global _TOK
    if _TOK is None:
        txt = []
        pth = BASE / "data" / "langid" / "title" / "ar" / "train.tsv"
        for line in open(pth, encoding="utf-8"):
            if not line.strip():
                continue
            txt.append(line.split(chr(9), 1)[0])
        _TOK = [w for t in txt for w in t.split()]
    return _TOK

def fillers(n):
    toks = _tokens()
    out = []
    for _ in range(n * 4):
        i = rng.randint(0, max(len(toks) - 17, 1))
        k = rng.randint(6, 16)
        chunk = toks[i:i + k]
        if len(chunk) < 4:
            continue
        out.append(chunk)
        if len(out) >= n:
            break
    while len(out) < n:
        out.append(toks[0:8])
    return out

def fillers_OLD(n):
    txt = []
    p = BASE / "data" / "langid" / "title" / "ar" / "train.tsv"
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        txt.append(line.split(chr(9), 1)[0])
    toks = [w for t in txt for w in t.split()][:400000]
    out, i = [], 0
    while len(out) < n and i < len(toks):
        k = rng.randint(6, 16)
        chunk = toks[i:i + k]
        i += k
        if len(chunk) < 4:
            continue
        out.append([w.replace("\u0660","d").replace("\u0661","d").replace("\u0662","d") for w in chunk])
    return out

def gen(K):
    sents = []
    for i in range(K):
        f = fillers(3); fill = f[0]; fill2 = f[1]
        n_slots = rng.choice([0,1,1,1,2,2,3])
        parts = []  # (word,label)
        base_labels = {}
        def emit(words, typ):
            for j, w in enumerate(words):
                parts.append((w, ("B-"+typ if j==0 else "I-"+typ)))
        parts += [(w,"O") for w in fill[:rng.randint(3, min(6,len(fill)))]]
        chosen = rng.sample(["DATE_G","DATE_H","DATE_REL","TIME","NUM_AI"], min(n_slots,5))
        for t in chosen:
            if t=="DATE_G":
                d, m, y = rng.randint(1,28), rng.choice(G_MONTHS), rng.randint(1920,2030)
                emit(str(d).translate(AI), "NUM_AI"); emit([m, str(y)], "DATE_G") if False else emit([m], "DATE_G"); emit(str(y), "NUM_AI")
            elif t=="DATE_H":
                d, m, y = rng.randint(1,30), rng.choice(H_MONTHS), rng.randint(1300,1446)
                emit(str(d).translate(AI), "NUM_AI"); emit(m.split(), "DATE_H"); emit(str(y)+rng.choice(H_SUFFIX), "NUM_AI")
            elif t=="DATE_REL":
                emit(rng.choice(REL).split(), "DATE_REL")
            elif t=="TIME":
                h = rng.randint(1,12); mnt = rng.randint(0,59)
                if rng.random()<0.5:
                    emit(str(h).translate(AI)+":"+str(mnt).translate(AI).zfill(2), "TIME")
                else:
                    emit((WORD_T[h]+" "+rng.choice(list(AMPM))).split(), "TIME")
            else:
                v = "".join(str(rng.randint(0,9)) for _ in range(rng.randint(2,5)))
                emit(str(v).translate(AI), "NUM_AI")
            parts += [(w,"O") for w in fill2[:rng.randint(3, 8)]]
        while len(parts) < 8:
            parts.append((rng.choice(fill2),"O"))
        sents.append(parts)
    return sents

LABELS = ["O"] + [p+t for t in ["DATE_G","DATE_H","DATE_REL","TIME","NUM_AI"] for p in ["B-","I-"]]

def write(sents, path):
    with open(path, "w", encoding="utf-8") as f:
        for si, s in enumerate(sents):
            for w, l in s:
                f.write(w + "\t" + l + "\n")
            if si != len(sents)-1:
                f.write("\n")

def main():
    sents = gen(30000)
    rng.shuffle(sents)
    n = len(sents); ntr, nva = int(n*.85), int(n*.075)
    os.makedirs(OUT, exist_ok=True)
    write(sents[:ntr], OUT/"train.tsv")
    write(sents[ntr:ntr+nva], OUT/"val.tsv")
    write(sents[ntr+nva:], OUT/"test.tsv")
    json.dump({"labels": LABELS}, open(OUT/"schemer_vocab.json", "w", encoding="utf-8"))
    print("schemer corpus:", ntr, nva, n-ntr-nva)

if __name__ == "__main__":
    main()
