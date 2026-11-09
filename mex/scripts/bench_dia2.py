
"""mex/scripts/bench_dia2.py - E-55: bench the mu composite on diacritizer external gates.
Read-only; reuses diacritizer/scripts/bench.py load_pairs/strip_marks and refs;
scores with diacritizer/scripts/eval.py compare (eval_der)."""
import json, os, sys, unicodedata
from pathlib import Path
import numpy as np
import torch
REPO = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "mex"))
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
from safetensors.torch import load_file
from transformers import LlamaForCausalLM
from src.data import PackedDataset
from mex.src.vocab import CharVocab
from mex.scripts.train_mu2_g5_bridge import Bridge
from mex.scripts.train_mu2_g4b_head import MarkHead
import bench as DB

device = "cuda" if torch.cuda.is_available() else "cpu"
voc = CharVocab()
marks = [c for c in voc.vocab if len(c) == 1 and 0x064B <= ord(c) <= 0x0652]
mark_ids = [int(voc.vocab[c]) for c in marks]
NONE, mid, pad = 8, int(voc.vocab["|"]), int(voc.vocab["<pad>"])
seq = int(os.environ.get('DIA2_SEQ', '96'))
mark_ids_t = torch.tensor(mark_ids, device=device)
MARK_SET = set(mark_ids)
REV = {v: k for k, v in voc.vocab.items()}
TRUNK_DIR = REPO / os.environ.get("DIA2_TRUNK", "runs/mex/dia2d_scale/final")
mount_dir = REPO / "runs/mex/dia2_wide"
trunk = LlamaForCausalLM.from_pretrained(str(TRUNK_DIR)).to(device).eval()
trunk.load_state_dict(load_file(str(TRUNK_DIR / "model.safetensors")), strict=True)
mub = torch.load(str(mount_dir / "mu2_bridges.pt"), weights_only=True)
xb = torch.load(str(mount_dir / "x_bridges.pt"), weights_only=True)
towers = []
for i in range(2):
    bm = Bridge(1280, 16).to(device); bm.load_state_dict(mub["bridge" + str(i)])
    bx = Bridge(1280, 16).to(device); bx.load_state_dict(xb["bridge" + str(i)])
    bm._strength = 0.0; bm._hold_kv = None; bx._strength = 0.0; bx._hold_kv = None
    towers.append((bm, bx))
class DualWrapped(torch.nn.Module):
    def __init__(self, base, bm, bx):
        super().__init__(); self.base = base; self.bm = bm; self.bx = bx
    def forward(self, *a, **kw):
        out = self.base(*a, **kw)
        y = out[0] if isinstance(out, tuple) else out
        hm = self.bm._hold_kv
        if hm is not None and self.bm._strength > 0:
            y = self.bm(y, hm.to(y.dtype))
        hx = self.bx._hold_kv
        if hx is not None and self.bx._strength > 0:
            y = self.bx(y, hx.to(y.dtype))
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
for i in range(2):
    trunk.model.layers[i] = DualWrapped(trunk.model.layers[i], towers[i][0], towers[i][1])
head = MarkHead(1280, 9)
head.load_state_dict(load_file(str(REPO / "runs/mex/dia2_init/head.safetensors")))
head = head.to(device).eval()
for m in [trunk, head] + [b for pr in towers for b in pr]:
    m.eval()
    for pp in m.parameters():
        pp.requires_grad = False

@torch.inference_mode()
def arm(x):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        hs = trunk.model(input_ids=x, output_hidden_states=True).hidden_states
    for i, (bm, bx) in enumerate(towers):
        bm._strength = 1.0; bm._hold_kv = hs[i + 1]
        bx._strength = 1.0; bx._hold_kv = hs[i + 1]

@torch.inference_mode()
def piece_labels(piece):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        ids = np.array(voc.encode(piece[:seq]), dtype=np.int64)[None]
        x = torch.as_tensor(ids, dtype=torch.long, device=device)
        arm(x)
        hh = trunk.model(input_ids=x, output_hidden_states=True)[0]
        cls = head(hh).argmax(-1)[0]
        tl = trunk(input_ids=x).logits[0].argmax(-1)
    out = []
    for t in range(x.shape[1]):
        c = int(cls[t]) if False else None
        c = int(cls[t].item())
        nid = int(tl[t].item())
        if c != NONE:
            nid = mark_ids[c if c < 8 else 7]
        out.append(nid)
    return out

def predict_bare_comp(bare, ctx=96):
    pieces = []
    for piece in DB.chunks_of(bare, ctx):
        lab = piece_labels(piece)
        out = []
        ti = 0
        for ch in piece:
            out.append(ch)
            if DB.is_arabic_base(ch) and ti < len(lab):
                nid = lab[ti]
                if nid in MARK_SET:
                    out.append(REV[nid])
            ti += 1
        pieces.append("".join(out))
    return "".join(pieces)

def main():
    src = sys.argv[1]
    out = Path(sys.argv[2])
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    out.parent.mkdir(parents=True, exist_ok=True)
    preds, refs_all = [], []
    n = 0
    for bare, refs in DB.load_pairs(src, seq):
        preds.append(predict_bare_comp(bare))
        refs_all.append("\t".join(refs))
        n += 1
        if lim and n >= lim:
            break
        if n % 200 == 0:
            print(json.dumps({"src": src, "lines": n}), flush=True)
    out.write_text("\n".join(preds) + "\n", encoding="utf-8")
    out.with_suffix(".ref.txt").write_text("\n".join(refs_all) + "\n", encoding="utf-8")
    print(json.dumps({"src": src, "sentences": n, "out": str(out)}))

main()
