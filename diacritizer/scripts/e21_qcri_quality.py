"""E-21c — QCRI Wikipedia_20240420.diac.jsonl quality audit (CPU-only).

1. mark-density stats on a 400-article sample vs our human-labeled abdou pool.
2. Validator disagreement: QCRI machine-labels as predictions vs two
   independent diacritizers (hos gold + Z-Mahmood BiLSTM) -> DER proxies.
3. Bare-input reconstruction error: QCRI text must re-tokenize cleanly
   (strip -> re-apply passthrough word identity).
"""
import json, re, random, sys, unicodedata
sys.path.insert(0, "models/e19/zmahood-src/src")

BAND = "[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670]"
strip_re = re.compile(BAND)

rows = open("models/e19/qcri-src/Wikipedia_Diacritized_Corpus/Wikipedia_20240420.diac.jsonl",
            encoding="utf-8").read().splitlines()
random.Random(77).shuffle(rows)

stats = {"lines": 0, "words": 0, "marked": 0, "marks": 0, "low": 0}
sample = []
for line in rows[:400]:
    text = json.loads(line)["text"]
    for para in str(text).split(chr(10)):
        for w in para.split():
            if not any("\u0621" <= c <= "\u064a" for c in w):
                continue
            m = len(re.findall(BAND, w))
            stats["words"] += 1
            stats["marks"] += m
            if m:
                stats["marked"] += 1
            else:
                stats["low"] += 1
        if para.strip() and len(sample) < 3:
            sample.append(para)
print("QCRI sample: marks-per-word-base", round(stats["marks"] / stats["words"], 2),
      "| vowel-carrying words", round(100 * stats["marked"] / stats["words"], 1), "%",
      "| unmarked", stats["low"])
for s in sample:
    print("   ", s[:170])

import pyarrow.parquet as pq
pf = pq.ParquetFile("data/diac/raw/abdou_tashkeel/data/train-00000-of-00008.parquet")
hm = {"words": 0, "marked": 0, "marks": 0}
n = 0
for batch in pf.iter_batches(batch_size=2048, columns=["vocalized"]):
    for text in batch.column(0).to_pylist():
        for w in str(text).split():
            if not any("\u0621" <= c <= "\u064a" for c in w):
                continue
            m = len(re.findall(BAND, w))
            hm["words"] += 1
            hm["marks"] += m
            if m:
                hm["marked"] += 1
    n += 1
    if n >= 6:
        break
print("HUMAN pool:  marks-per-word-base", round(hm["marks"] / hm["words"], 2),
      "| vowel-carrying words", round(100 * hm["marked"] / hm["words"], 1), "%")

# validator disagreement: gold + Z-Mahmood on the same QCRI texts (bare inputs)
import torch, yaml, time
from pathlib import Path
REPO = Path(".").resolve()
sys.path.insert(0, "models/e19/zmahood-src/src")   # ZM package FIRST
from diacritize import Diacritizer                  # before scripts collide
sys.path.insert(1, "diacritizer/src")               # gold modules
sys.path.insert(1, "diacritizer/scripts")           # bench.py
import importlib
importlib.reload(diacritize) if False else None
from model import build_from_config
import bench as B
ck = torch.load("runs/diac/stage2b2500/final/model.pt", map_location="cpu", weights_only=False)
cfg = yaml.safe_load(Path("runs/diac/stage2b2500/final/config.yaml").read_text(encoding="utf-8"))
model = build_from_config(cfg["model"], 171).to("cpu").eval()
model.load_state_dict(ck)
zm = Diacritizer.from_pretrained(no_cache=True)

print("validators loaded; scoring validator-DER of QCRI labels on 150-art sample...")


# reuse eval.py compare as DER engine by writing tmp files? simpler: strip to bare chars then diff per word.
import subprocess
def score(pred_path, ref_path):
    r = subprocess.run([".venv/Scripts/python.exe", "-X", "utf8", "diacritizer/scripts/eval.py", "compare", "--pred", pred_path, "--ref", ref_path], capture_output=True, text=True, encoding="utf-8")
    j = json.loads(r.stdout)["aggregate"]
    return j["DER"] * 100

def clean(t):
    return " ".join(t.replace("\u200f", "").split())

# build sample files
qc_bare, qc_ref = open("models/e19/qcri.sample.bare.txt", "w", encoding="utf-8"), open("models/e19/qcri.sample.ref.txt", "w", encoding="utf-8")
texts = []
for line in rows[:4000]:
    para = clean(json.loads(line)["text"])
    if para and len(para) > 30:
        texts.append(para[:600])
    if len(texts) >= 300:
        break
qc_bare.write(chr(10).join(strip_re.sub("", t) for t in texts) + chr(10)); qc_bare.close()
qc_ref.write(chr(10).join(texts) + chr(10)); qc_ref.close()

outs = [B.predict_bare(model, strip_re.sub("", t), cfg["model"]["ctx"], "cpu") for t in texts]
open("models/e19/qcri.gold_vs.pred.txt", "w", encoding="utf-8").write(chr(10).join(outs) + chr(10))
outs_zm = [zm.diacritize(strip_re.sub("", t)) for t in texts]
print("Z-Mahmood done", len(outs_zm))
open("models/e19/qcri.zm_vs.pred.txt", "w", encoding="utf-8").write(chr(10).join(outs_zm) + chr(10))
print("QCRI-labels vs gold:", score("models/e19/qcri.gold_vs.pred.txt", "models/e19/qcri.sample.ref.txt"))
print("QCRI-labels vs ZM:", score("models/e19/qcri.zm_vs.pred.txt", "models/e19/qcri.sample.ref.txt"))
