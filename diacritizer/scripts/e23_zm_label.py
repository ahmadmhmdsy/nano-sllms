"""E-23c part 1 - Z-Mahmood soft-label generation (S3).

Input: data/diac/raw/qcri_diac_clone/qcri_gatesafe.jsonl (modern wiki text,
QCRI machine-vocalized). For each article: take text, split into paragraphs,
strip diacritics (bare input), run ZM-BiLSTM, keep ZM predictions.
Output: data/diac/raw/e23c_zm_labels/zm_labels.jsonl rows
  {"serial": ..., "para": ..., "diac": <zm vocalized>}

--limit N controls article count; a probe with 20 articles first measures
throughput. Chunked flush every 200 paras.
"""
import argparse, json, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DIAC = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0640"

def parags(text, maxlen=280):
    out = []
    for blk in text.replace("\r", "").split("\n"):
        blk = blk.strip()
        if len(blk) < 40:
            continue
        while len(blk) > maxlen:
            cut = blk.rfind(" ", 0, maxlen)
            cut = cut if cut > 200 else maxlen
            out.append(blk[:cut])
            blk = blk[cut:].strip()
            if not blk:
                break
        else:
            out.append(blk)
    return out

main_ap = "--probe" not in sys.argv
a = argparse.ArgumentParser()
a.add_argument("--probe", action="store_true")
a.add_argument("--limit", type=int, default=0)
a.add_argument("--device", default="cpu")
args, _ = a.parse_known_args()

limit = 20 if args.probe else args.limit
src = REPO / "data/diac/raw/qcri_diac_clone/qcri_gatesafe.jsonl"
outdir = REPO / "data/diac/raw/e23c_zm_labels"
outdir.mkdir(parents=True, exist_ok=True)
outp = outdir / ("probe.jsonl" if args.probe else "zm_labels.jsonl")

sys.path.insert(0, str(REPO / "models/e19/zmahood-src/src"))
from diacritize import Diacritizer as ZM
zm = ZM.from_pretrained(no_cache=True)

n_art = n_para = 0
t0 = time.time()
buf = []
with outp.open("w", encoding="utf-8") as f:
    for line in open(src, encoding="utf-8"):
        if limit and n_art >= limit:
            break
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        text = obj.get("text")
        if not text or len(text) < 200:
            continue
        n_art += 1
        for pg in parags(text):
            bare = re.sub("[" + DIAC + "]", "", pg)
            nwords = len(bare.split())
            t1 = time.time()
            diac = zm.diacritize(bare)
            f.write(json.dumps({"serial": obj.get("serial_num"), "para": n_para, "bare": bare, "diac": diac}, ensure_ascii=False) + "\n")
            n_para += 1
            if n_para % 50 == 0:
                rate = n_para / (time.time() - t0)
                print(f"paras {n_para} art {n_art} rate {rate:.1f}/s", flush=True)
print("DONE paras", n_para, "arts", n_art, "secs", round(time.time() - t0, 1), "->", outp.name)
