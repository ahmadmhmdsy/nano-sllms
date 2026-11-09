"""E-20 probe: gold model predictions + word-cache fallback merge.

For each gate: run the GOLD model over the bare lines (GPU), then for each
word position look up OUR word cache (models/e19/our_word_cache.json) under
the stripped-bare key and SWAP the model word in when the cache has a
majority-vote variant. Writes {gate}.gold.raw.pred.txt and
{gate}.goldcache.pred.txt under models/e19/.
"""
import json, re, sys, unicodedata, time, yaml
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
import sys as _sys
_sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
from model import build_from_config  # noqa: E402
import bench as B  # noqa: E402

_DIAC = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"
strip_re = re.compile("[" + _DIAC + "]")


def norm_word(w: str) -> str:
    w = strip_re.sub("", unicodedata.normalize("NFKC", w))
    return (w.replace("\u0622", "\u0627")
            .replace("\u0623", "\u0627")
            .replace("\u0625", "\u0627")
            .replace("\u0640", ""))


import tokenizer as TK  # noqa: E402
ck = torch.load(REPO / "runs/diac/stage2b2500/final/model.pt", map_location="cuda", weights_only=False)
cfg = yaml.safe_load((REPO / "runs/diac/stage2b2500/final/config.yaml").read_text(encoding="utf-8"))
model = build_from_config(cfg["model"], TK.VOCAB_SIZE).to("cuda").eval()
model.load_state_dict(ck)
ctx = cfg["model"]["ctx"]
cache = json.load(open(REPO / "models/e19/our_word_cache.json", encoding="utf-8"))["words"]
print("cache keys", len(cache), flush=True)

lines_n = {"fadel2500": 2500, "sadeed2500": 1612, "wn2014": 393, "abdou_heldout": 2067}
for gate, n in lines_n.items():
    if not re.fullmatch(r"[a-z0-9_]+", gate):
        print("skip bad name", gate)
        continue
    inp = REPO / "models/e19/inputs" / (gate + ".bare.txt")
    lines = [l for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    t0 = time.time()
    outs = []
    for i, line in enumerate(lines):
        outs.append(B.predict_bare(model, line, ctx, "cuda"))
        if (i + 1) % 250 == 0:
            print(gate, i + 1, "/", len(lines), round((i + 1) / (time.time() - t0), 1), "l/s", flush=True)
    raw = REPO / "models/e19" / (gate + ".gold.raw.pred.txt")
    raw.write_text("\n".join(outs) + "\n", encoding="utf-8")
    merged = []
    hits = 0
    for line, pred in zip(lines, outs):
        p_words = pred.split()
        b_words = line.split()
        if len(p_words) != len(b_words):
            merged.append(pred)
            continue
        out_words = list(p_words)
        for j, bw in enumerate(b_words):
            v = cache.get(norm_word(bw))
            if v is not None:
                out_words[j] = v
                hits += 1
        merged.append(" ".join(out_words))
    out = REPO / "models/e19" / (gate + ".goldcache.pred.txt")
    out.write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(gate, "DONE hits", hits, "lines", len(merged), flush=True)
print("ALL DONE", flush=True)
