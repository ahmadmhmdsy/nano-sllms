"""mex/scripts/bench_e58.py - E-58 external gates with the 15-label head.
Run: python mex/scripts/bench_e58.py <gate> <out_pred.txt>
"""
import json, sys, os
from pathlib import Path
import torch
REPO = Path("E:/python_projects/nano_SLMs")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mex"))
sys.path.insert(0, str(REPO / "diacritizer" / "src"))
sys.path.insert(0, str(REPO / "diacritizer" / "scripts"))
from transformers import LlamaForCausalLM
from safetensors.torch import load_file
from mex.src.vocab import CharVocab
import bench as DB
from labels import marks_for_label

device = "cuda" if torch.cuda.is_available() else "cpu"
voc = CharVocab()
SEQ = int(os.environ.get("DIA2_SEQ", "96"))
TRUNK_DIR = Path(os.environ.get("DIA2_TRUNK", str(REPO / "runs/mex/dia2g_15head")))
trunk = LlamaForCausalLM.from_pretrained(str(TRUNK_DIR)).to(device).eval()
trunk.load_state_dict(load_file(str(TRUNK_DIR / "model.safetensors")), strict=True)
_hd = load_file(str(TRUNK_DIR / "head15.safetensors"))
head = torch.nn.Linear(_hd["weight"].shape[1], _hd["weight"].shape[0]).to(device)
head.load_state_dict(_hd)
for m in (trunk, head):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

@torch.inference_mode()
def piece_labels(piece):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        x = torch.as_tensor([voc.encode(piece[:SEQ])], dtype=torch.long, device=device)
        h = trunk.model(input_ids=x).last_hidden_state
        cls = head(h.float()).argmax(-1)[0]
    return [int(c) for c in cls.tolist()]

def predict_bare_comp(bare, ctx=SEQ):
    out = []
    for piece in DB.chunks_of(bare, ctx):
        cls = piece_labels(piece)
        ti = 0
        for ch in piece:
            out.append(ch)
            o = ord(ch)
            if 0x0621 <= o <= 0x064A or o == 0x0671:
                if ti < len(cls) and cls[ti] >= 1:
                    out.append(marks_for_label(cls[ti]))
                ti += 1
    return "".join(out)

def main():
    src = sys.argv[1]
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    preds, refs_all, n = [], [], 0
    for bare, refs in DB.load_pairs(src, SEQ):
        preds.append(predict_bare_comp(bare))
        refs_all.append("\t".join(refs))
        n += 1
        if n % 200 == 0:
            print(json.dumps({"src": src, "lines": n}), flush=True)
    out.write_text("\n".join(preds) + "\n", encoding="utf-8")
    out.with_suffix(".ref.txt").write_text("\n".join(refs_all) + "\n", encoding="utf-8")
    print(json.dumps({"src": src, "sentences": n, "out": str(out)}))

main()
